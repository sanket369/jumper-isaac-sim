#!/usr/bin/env python3
"""Build data/crab.usd from the exported training model.

Run with Isaac Sim's Python (it needs `pxr`, which only exists inside Kit);
bootstrap.sh does this:

    "$ISAAC_SIM_PATH/python.sh" tools/build_usd.py

Input is data/crab_model.{json,npz} from `tools/export_model.py`. Every
number comes from there; this script only translates MuJoCo's description into
UsdPhysics / PhysX terms. The places where the two engines differ are written
down where they are handled:

* **Frames.** Links are flat siblings, each placed at its pose relative to the
  base at the reference configuration. A revolute joint's frame is built so
  that its X axis is the MuJoCo hinge axis; `tools/isaac_check.py` checks the result
  against MuJoCo's forward kinematics.
* **Collision.** Each link collides with the convex hull MuJoCo collides it with
  (foot hulls pre-reduced to PhysX's 255-vertex limit, error recorded in the
  JSON). MuJoCo's contype/conaffinity masks are turned into PhysX filtered
  pairs body by body, so exactly the pairs that collided in training collide
  here: legs against other legs and against the ground, never a leg against
  itself or the chassis.
* **Friction.** MuJoCo takes the max of two geoms' friction, and `condim=1`
  geoms are frictionless. PhysX picks the combine mode of higher priority
  (average < min < multiply < max), so: ground `average` 1.0, grip geoms
  (feet, palm pads; condim 3) `max` at their coefficient, everything else
  `min` at 0. Every pairing then reproduces MuJoCo's coefficient.
* **Actuation.** No drives. The controller applies the servo torque, passive
  damping and friction loss explicitly each physics step, as mjlab's
  ServoCurveActuator and the MJCF joint class do. Armature is native to PhysX.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CRAB_DATA", HERE.parent / "data"))

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=DATA)
ap.add_argument("--out", type=Path, default=DATA / "crab.usd")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

meta = json.loads((args.data / "crab_model.json").read_text())
arrays = np.load(args.data / "crab_model.npz")

ROOT = "/crab"
CONTACT_OFFSET = 0.002  # m. Detection margin only; the robot stands 0.106 m tall.
REST_OFFSET = 0.0  # MuJoCo's margin is 0 on every geom here.
MAX_JOINT_VELOCITY_DEG = 1.0e6  # PhysX caps joint speed; the servo curve is the real cap.


def quat(wxyz) -> Gf.Quatf:
    w, x, y, z = (float(v) for v in wxyz)
    return Gf.Quatf(w, Gf.Vec3f(x, y, z))


def quatd(wxyz) -> Gf.Quatd:
    w, x, y, z = (float(v) for v in wxyz)
    return Gf.Quatd(w, Gf.Vec3d(x, y, z))


def set_pose(prim, pos, wxyz) -> None:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quatd(wxyz))


def make_mesh(stage, path: str, name: str, pos, wxyz, rgba):
    mesh = UsdGeom.Mesh.Define(stage, path)
    verts = arrays[f"{name}/verts"]
    faces = arrays[f"{name}/faces"]
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, rgba[:3]))]))
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    set_pose(mesh.GetPrim(), pos, wxyz)
    return mesh


def make_material(stage, path: str, friction: float, combine: str):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr(combine)
    px.CreateRestitutionCombineModeAttr("min")
    return mat


def collides(a: dict, b: dict) -> bool:
    """MuJoCo's rule for two geoms (filterparent handled by the caller)."""
    return bool((a["contype"] & b["conaffinity"]) or (b["contype"] & a["conaffinity"]))


stage = Usd.Stage.CreateNew(str(args.out))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, ROOT).GetPrim()
stage.SetDefaultPrim(root)
UsdGeom.Scope.Define(stage, f"{ROOT}/materials")
UsdGeom.Scope.Define(stage, f"{ROOT}/joints")

# Materials, one per distinct contact behaviour found in the export.
materials: dict[tuple[str, float], UsdShade.Material] = {}


def material_for(geom: dict) -> UsdShade.Material:
    if geom["condim"] == 1:
        key = ("frictionless", 0.0)
        combine = "min"
    elif geom["condim"] == 3:
        key = ("grip", geom["friction"])
        combine = "max"
    else:
        raise RuntimeError(f"{geom['name']}: condim {geom['condim']} has no PhysX equivalent here")
    if key not in materials:
        name = f"{key[0]}_{int(round(key[1] * 1000))}"
        materials[key] = make_material(stage, f"{ROOT}/materials/{name}", key[1], combine)
    return materials[key]


bodies = {b["name"]: b for b in meta["bodies"]}
geoms_by_body: dict[str, list[dict]] = {}
for g in meta["geoms"]:
    geoms_by_body.setdefault(g["body"], []).append(g)

link_prims = {}
for b in meta["bodies"]:
    path = f"{ROOT}/{b['name']}"
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    set_pose(prim, b["pos_in_base"], b["quat_in_base"])
    UsdPhysics.RigidBodyAPI.Apply(prim)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateLinearDampingAttr(0.0)  # PhysX defaults to 0.05 angular; MuJoCo has none
    rb.CreateAngularDampingAttr(0.0)
    rb.CreateMaxLinearVelocityAttr(1.0e6)
    rb.CreateMaxAngularVelocityAttr(1.0e7)  # deg/s
    rb.CreateSleepThresholdAttr(0.0)
    rb.CreateStabilizationThresholdAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(b["mass"])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, b["ipos"])))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, b["inertia"])))
    mass.CreatePrincipalAxesAttr(quat(b["iquat"]))
    link_prims[b["name"]] = prim

    for g in geoms_by_body.get(b["name"], []):
        if g["collision"]:
            mesh = make_mesh(stage, f"{path}/collision", g["name"], g["pos"], g["quat"], g["rgba"])
            mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
            p = mesh.GetPrim()
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
            hull = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(p)
            hull.CreateHullVertexLimitAttr(255)
            hull.CreateMinThicknessAttr(0.0001)
            col = PhysxSchema.PhysxCollisionAPI.Apply(p)
            col.CreateContactOffsetAttr(CONTACT_OFFSET)
            col.CreateRestOffsetAttr(REST_OFFSET)
            UsdShade.MaterialBindingAPI.Apply(p).Bind(
                material_for(g), UsdShade.Tokens.weakerThanDescendants, "physics")
        else:
            make_mesh(stage, f"{path}/visual", g["name"], g["pos"], g["quat"], g["rgba"])

# Articulation root on the floating base link.
base = link_prims["base_link"]
UsdPhysics.ArticulationRootAPI.Apply(base)
art = PhysxSchema.PhysxArticulationAPI.Apply(base)
art.CreateEnabledSelfCollisionsAttr(True)
art.CreateSolverPositionIterationCountAttr(8)
art.CreateSolverVelocityIterationCountAttr(1)
art.CreateSleepThresholdAttr(0.0)
art.CreateStabilizationThresholdAttr(0.0)


def pose_in(parent: dict, child: dict) -> tuple[Gf.Vec3d, Gf.Quatd]:
    """The child's frame expressed in the parent's (both given relative to the base)."""
    qp, qc = quatd(parent["quat_in_base"]), quatd(child["quat_in_base"])
    pp, pc = Gf.Vec3d(*parent["pos_in_base"]), Gf.Vec3d(*child["pos_in_base"])
    inv = qp.GetInverse()
    rel_p = Gf.Rotation(inv).TransformDir(pc - pp)
    return rel_p, (inv * qc).GetNormalized()


for j in meta["joints"]:
    parent, child = bodies[j["parent"]], bodies[j["child"]]
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ROOT}/joints/{j['name']}")
    joint.CreateBody0Rel().SetTargets([link_prims[parent["name"]].GetPath()])
    joint.CreateBody1Rel().SetTargets([link_prims[child["name"]].GetPath()])
    axis = Gf.Vec3d(*j["axis"]).GetNormalized()
    q_axis = Gf.Rotation(Gf.Vec3d(1, 0, 0), axis).GetQuat()  # joint X -> hinge axis
    jpos = Gf.Vec3d(*j["pos"])
    rel_p, rel_q = pose_in(parent, child)
    pos0 = rel_p + Gf.Rotation(rel_q).TransformDir(jpos)
    rot0 = (rel_q * q_axis).GetNormalized()
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(pos0))
    joint.CreateLocalRot0Attr(Gf.Quatf(rot0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(jpos))
    joint.CreateLocalRot1Attr(Gf.Quatf(q_axis))
    if j["limited"]:
        joint.CreateLowerLimitAttr(math.degrees(j["range"][0]))
        joint.CreateUpperLimitAttr(math.degrees(j["range"][1]))
    pj = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    pj.CreateArmatureAttr(j["armature"])
    pj.CreateJointFrictionAttr(0.0)  # friction loss is applied explicitly, see the docstring
    pj.CreateMaxJointVelocityAttr(MAX_JOINT_VELOCITY_DEG)

# A MuJoCo body without a joint is welded to its parent (19 of them here: the
# shell, the display, the sensors, the palm pads, the finger and foot tips). In
# PhysX a rigid body with no joint is a separate free body, so each gets a
# fixed joint, keeping its own mass, inertia and collision geometry exactly.
jointed = {j["child"] for j in meta["joints"]}
welded = 0
for b in meta["bodies"]:
    if b["parent"] is None or b["name"] in jointed:
        continue
    parent = bodies[b["parent"]]
    joint = UsdPhysics.FixedJoint.Define(stage, f"{ROOT}/joints/{b['name']}_weld")
    joint.CreateBody0Rel().SetTargets([link_prims[parent["name"]].GetPath()])
    joint.CreateBody1Rel().SetTargets([link_prims[b["name"]].GetPath()])
    rel_p, rel_q = pose_in(parent, b)
    joint.CreateLocalPos0Attr(Gf.Vec3f(rel_p))
    joint.CreateLocalRot0Attr(Gf.Quatf(rel_q))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    welded += 1
roots = [b["name"] for b in meta["bodies"] if b["parent"] is None]
if roots != ["base_link"]:
    raise RuntimeError(f"expected base_link as the only root body, found {roots}")

# Self-collision: filter every link pair MuJoCo would not collide.
names = [b["name"] for b in meta["bodies"]]
coll_geom = {}
for g in meta["geoms"]:
    if g["collision"]:
        if g["body"] in coll_geom:
            raise RuntimeError(f"{g['body']} has two collision geoms; pair filtering assumes one")
        coll_geom[g["body"]] = g
filtered: dict[str, list[str]] = {n: [] for n in names}
kept = 0
for i, a in enumerate(names):
    for b in names[i + 1:]:
        ga, gb = coll_geom.get(a), coll_geom.get(b)
        parent_child = bodies[a]["parent"] == b or bodies[b]["parent"] == a
        if ga is None or gb is None or parent_child or not collides(ga, gb):
            filtered[a].append(b)
        else:
            kept += 1
for a, partners in filtered.items():
    if partners:
        api = UsdPhysics.FilteredPairsAPI.Apply(link_prims[a])
        api.CreateFilteredPairsRel().SetTargets([link_prims[b].GetPath() for b in partners])

stage.GetRootLayer().Save()
print(f"[usd] wrote {args.out}: {len(link_prims)} links, {len(meta['joints'])} revolute "
      f"+ {welded} fixed joints, "
      f"{len(materials)} materials, {kept} self-collision pairs kept "
      f"(of {len(names) * (len(names) - 1) // 2})", flush=True)
# SimulationApp.close() hangs in Kit's shutdown (Isaac Sim 5.1, Ubuntu 24.04,
# observed 2026-10-19); the stage is saved, so leave without it.
os._exit(0)
