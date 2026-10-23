#!/usr/bin/env python3
"""Dump the robot exactly as jumper.posture trains it, for the USD builder.

Run from the Jumper checkout in its venv (it builds the training environment);
bootstrap.sh does this:

    cd third_party/jumper && .venv/bin/python ../../tools/export_model.py

The source is the *compiled* MjModel of the environment `play` builds, not
`assets/jumper/jumper.xml`: mjlab edits the spec before compiling (the collision
set, the actuators, contact parameters), so the raw XML is not what the policy
was trained on. Everything the USD needs comes from the compiled model --
body frames, inertias, joint axes and dynamics, the collision hulls MuJoCo
actually collides, the visual meshes, and each geom's contact masks -- so the
builder never has to guess.

Output: data/crab_model.npz and data/crab_model.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = Path(os.environ.get("CRAB_DATA", HERE.parent / "data"))
PREFIX = "robot/"


def _name(m, kind, i) -> str:
    return mujoco.mj_id2name(m, kind, i) or ""


def _strip(name: str) -> str:
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def _hull(m: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    """The convex hull MuJoCo collides this mesh by, as (vertices, triangles).

    MuJoCo stores it in `mesh_graph`:
    [numvert, numface, vert_edgeadr[numvert], vert_globalid[numvert],
     edge_localid[numvert + 3*numface], face_globalid[3*numface]].
    """
    vadr, vnum = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
    verts = np.asarray(m.mesh_vert[vadr:vadr + vnum], dtype=np.float64)
    gadr = m.mesh_graphadr[mesh_id]
    if gadr < 0:
        raise RuntimeError(f"mesh {mesh_id} has no hull graph; cannot reproduce its collision")
    g = m.mesh_graph
    numvert, numface = int(g[gadr]), int(g[gadr + 1])
    globalid = np.asarray(g[gadr + 2 + numvert: gadr + 2 + 2 * numvert], dtype=np.int64)
    face_start = gadr + 2 + 3 * numvert + 3 * numface
    faces_global = np.asarray(g[face_start: face_start + 3 * numface],
                              dtype=np.int64).reshape(-1, 3)
    remap = {int(gid): i for i, gid in enumerate(globalid)}
    faces = np.vectorize(remap.__getitem__)(faces_global)
    return verts[globalid], faces


#: PhysX cooks a convex hull from at most 255 vertices and simplifies anything
#: larger on its own, by a rule we cannot inspect. The foot tips' hulls carry
#: 667-668, so they are reduced here, measurably, before PhysX sees them.
PHYSX_HULL_VERTEX_LIMIT = 255


def _reduce_hull(verts: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray, float]:
    """A hull of at most `limit` of `verts`, its triangles, and the error [m].

    Farthest-point sampling picks the subset, so the reduced hull lies inside
    the original (its vertices are original hull vertices) and the error is one
    direction only: how far the original surface sticks out past it. Measured on
    the feet (667 verts, 35 mm long) it is 0.51 mm.
    """
    from scipy.spatial import ConvexHull

    if len(verts) <= limit:
        return verts, ConvexHull(verts).simplices, 0.0
    idx = [int(np.argmin(verts[:, 2]))]
    dist = np.linalg.norm(verts - verts[idx[0]], axis=1)
    while len(idx) < limit:
        i = int(np.argmax(dist))
        idx.append(i)
        dist = np.minimum(dist, np.linalg.norm(verts - verts[i], axis=1))
    sub = verts[idx]
    hull = ConvexHull(sub)
    outside = (verts @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)
    used = np.unique(hull.simplices)
    remap = {int(o): n for n, o in enumerate(used)}
    faces = np.vectorize(remap.__getitem__)(hull.simplices)
    return sub[used], faces, float(max(outside.max(), 0.0))


def _mesh(m: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
    vadr, vnum = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
    fadr, fnum = m.mesh_faceadr[mesh_id], m.mesh_facenum[mesh_id]
    verts = np.asarray(m.mesh_vert[vadr:vadr + vnum], dtype=np.float64)
    faces = np.asarray(m.mesh_face[fadr:fadr + fnum], dtype=np.int64)
    return verts, faces


def build_env(task: str):
    resolve = importlib.import_module("mjrl.backend.resolve")
    import tasks
    from mjrl.backend.select import use_backend

    res = resolve.resolve(backend="native", device="cpu", num_envs=1,
                          strip_visual=False, training=False)
    use_backend(res)  # must precede the env's construction
    from mjlab.envs import ManagerBasedRlEnv

    cfg = tasks.load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", default="jumper.posture")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    env = build_env(args.task)
    m = env.sim.mj_model
    robot = env.scene["robot"]

    # Body poses at the reference configuration (all hinges at 0), relative to
    # the base: the USD places every link flat, in the base's frame.
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + "base_link")
    R_base = d.xmat[base_id].reshape(3, 3)
    p_base = d.xpos[base_id]

    body_ids = [b for b in range(m.nbody)
                if _name(m, mujoco.mjtObj.mjOBJ_BODY, b).startswith(PREFIX)]
    bodies = []
    for b in body_ids:
        R = R_base.T @ d.xmat[b].reshape(3, 3)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        parent = int(m.body_parentid[b])
        bodies.append({
            "name": _strip(_name(m, mujoco.mjtObj.mjOBJ_BODY, b)),
            "parent": (_strip(_name(m, mujoco.mjtObj.mjOBJ_BODY, parent))
                       if parent in body_ids else None),
            "pos_in_base": (R_base.T @ (d.xpos[b] - p_base)).tolist(),
            "quat_in_base": q.tolist(),  # w, x, y, z
            "mass": float(m.body_mass[b]),
            "ipos": m.body_ipos[b].tolist(),
            "iquat": m.body_iquat[b].tolist(),  # w, x, y, z
            "inertia": m.body_inertia[b].tolist(),
        })

    joints = []
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        child = int(m.jnt_bodyid[j])
        dof = int(m.jnt_dofadr[j])
        joints.append({
            "name": _strip(_name(m, mujoco.mjtObj.mjOBJ_JOINT, j)),
            "child": _strip(_name(m, mujoco.mjtObj.mjOBJ_BODY, child)),
            "parent": _strip(_name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[child]))),
            "pos": m.jnt_pos[j].tolist(),  # in the child body's frame
            "axis": m.jnt_axis[j].tolist(),  # in the child body's frame
            "range": m.jnt_range[j].tolist(),
            "limited": bool(m.jnt_limited[j]),
            "armature": float(m.dof_armature[dof]),
            "damping": float(m.dof_damping[dof]),
            "frictionloss": float(m.dof_frictionloss[dof]),
        })

    arrays: dict[str, np.ndarray] = {}
    geoms = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in body_ids:
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            raise RuntimeError(
                f"geom {_name(m, mujoco.mjtObj.mjOBJ_GEOM, g)}: only meshes are handled")
        name = _strip(_name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        collides = bool(m.geom_contype[g] or m.geom_conaffinity[g])
        mesh_id = int(m.geom_dataid[g])
        hull_error = 0.0
        if collides:
            verts, faces = _hull(m, mesh_id)
            hull_nvert = len(verts)
            verts, faces, hull_error = _reduce_hull(verts, PHYSX_HULL_VERTEX_LIMIT)
        else:
            verts, faces = _mesh(m, mesh_id)
            hull_nvert = 0
        arrays[f"{name}/verts"] = verts.astype(np.float32)
        arrays[f"{name}/faces"] = faces.astype(np.int32)
        geoms.append({
            "name": name,
            "body": _strip(_name(m, mujoco.mjtObj.mjOBJ_BODY, b)),
            "collision": collides,
            "pos": m.geom_pos[g].tolist(),
            "quat": m.geom_quat[g].tolist(),  # w, x, y, z
            "rgba": m.geom_rgba[g].tolist(),
            "friction": float(m.geom_friction[g][0]),
            "condim": int(m.geom_condim[g]),
            "contype": int(m.geom_contype[g]),
            "conaffinity": int(m.geom_conaffinity[g]),
            "nvert": int(len(verts)),
            "mujoco_hull_nvert": int(hull_nvert),
            "hull_reduction_error_m": hull_error,
        })

    # Forward kinematics at the default stance, base at the identity: the USD
    # check sets the same joint angles in PhysX and compares every link. A
    # flipped axis, a swapped joint or a wrong frame shows up here as metres,
    # where otherwise it would show up as a policy that walks strangely.
    d.qpos[:] = m.qpos0
    for name, q in zip(robot.joint_names, robot.data.default_joint_pos[0].tolist()):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + name)
        d.qpos[m.jnt_qposadr[j]] = q
    mujoco.mj_kinematics(m, d)
    R_base = d.xmat[base_id].reshape(3, 3)
    p_base = d.xpos[base_id].copy()
    fk_check = {
        _strip(_name(m, mujoco.mjtObj.mjOBJ_BODY, b)): (R_base.T @ (d.xpos[b] - p_base)).tolist()
        for b in body_ids
    }

    terrain = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    act = robot.actuators[0]
    meta = {
        "_source": f"tools/export_model.py from the compiled {args.task} play environment",
        "task": args.task,
        "timestep": float(m.opt.timestep),
        "gravity": m.opt.gravity.tolist(),
        "terrain_friction": float(m.geom_friction[terrain][0]),
        "total_mass": float(sum(x["mass"] for x in bodies)),
        "joint_names": list(robot.joint_names),
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
        "root_init": robot.data.default_root_state[0].tolist(),
        "actuator": {
            "type": type(act).__name__,
            "joints": list(act.target_names),
            "force_limit": float(act.force_limit[0, 0]),
            "plateau_torque": float(act.plateau_torque[0, 0]),
            "corner_speed": float(act.corner_speed[0, 0]),
            "decay_speed": float(act.decay_speed[0, 0]),
            "cutoff_speed": float(act.cutoff_speed[0, 0]),
            "continuous_torque": float(act.continuous_torque[0, 0]),
            "thermal_time_constant": float(act.thermal_time_constant[0, 0]),
        },
        "bodies": bodies,
        "joints": joints,
        "geoms": geoms,
        "fk_check_default_pose": fk_check,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "crab_model.npz", **arrays)
    (args.out / "crab_model.json").write_text(json.dumps(meta, indent=1))
    ncol = sum(g["collision"] for g in geoms)
    print(f"[export] {len(bodies)} bodies, {len(joints)} hinges, {len(geoms)} geoms "
          f"({ncol} collision), total mass {meta['total_mass']:.4f} kg -> {args.out}")
    worst = max((g for g in geoms if g["collision"]), key=lambda g: g["hull_reduction_error_m"])
    print(f"[export] hull vertex counts: max {max(g['nvert'] for g in geoms if g['collision'])}; "
          f"worst reduction {worst['hull_reduction_error_m'] * 1e3:.3f} mm ({worst['name']})")


if __name__ == "__main__":
    main()
