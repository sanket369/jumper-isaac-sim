#!/usr/bin/env python3
"""Check the crab in Isaac Sim against MuJoCo, headless.

    "$ISAAC_SIM_PATH/python.sh" tools/isaac_check.py [--friction F] [--payload-kg M] \
        [--motor-strength K] [--out result.json] [--video drive.mp4]

Runs the code the interactive example runs (crab_isaac.robot + crab_core) and:

1. reads back what PhysX built -- link masses, joint limits -- against the
   training model;
2. sets the default stance with gravity off and compares every link's position
   with MuJoCo's forward kinematics (a flipped axis or a swapped joint shows up
   here as millimetres, not as a policy that walks strangely);
3. drives the scripted scenario (crab_core/scenario.py) through the bundle's
   controller and compares speeds, yaw rate, height and tilt with the MuJoCo
   reference from tools/mujoco_reference.py.

With no options it is the sim-to-sim gate: the training conditions, both
engines, within --tolerance (20%). The options move the Isaac world away from
training for robustness sweeps; the reference stays the nominal MuJoCo run, so
the numbers read as "how much of the nominal behaviour survives":

  --friction F        foot-ground friction coefficient (training: 1.2). It sets
                      the feet's and palm pads' grip material and the ground
                      together, because MuJoCo -- and this USD, to match --
                      takes the larger of the two, so lowering the ground alone
                      would change nothing.
  --payload-kg M      mass added to the base link (training: 0; robot 2.54 kg).
  --motor-strength K  the servo torque curve and its limits scaled by K
                      (training: 1.0).

Every run writes a JSON result (--out, default data/isaac_result.json). --video PATH also
records the drive as an MP4 through crab_core/video.py's camera, the one
tools/mujoco_reference.py uses, so the two clips match shot for shot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exts" / "jumper.crab_example"))

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--friction", type=float, default=None,
                    help="foot-ground friction; default: the trained value")
parser.add_argument("--payload-kg", type=float, default=0.0)
parser.add_argument("--motor-strength", type=float, default=1.0)
parser.add_argument("--tolerance", type=float, default=0.2,
                    help="largest relative gap to MuJoCo that passes")
parser.add_argument("--reference", type=Path, default=None,
                    help="MuJoCo reference JSON; default <data>/mujoco_reference.json")
parser.add_argument("--out", type=Path, default=None,
                    help="result JSON; default <data>/isaac_result.json")
parser.add_argument("--label", default="nominal")
parser.add_argument("--video", type=Path, default=None, help="record the drive to this MP4")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import traceback  # noqa: E402


def _fail(kind, value, tb) -> None:
    # Kit swallows the exit status of an uncaught exception; make it a failure.
    traceback.print_exception(kind, value, tb)
    sys.stdout.flush()
    os._exit(2)


sys.excepthook = _fail

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

from jumper_crab.crab_core import paths, scenario, video  # noqa: E402
from jumper_crab.crab_core.app_controller import CrabApp  # noqa: E402
from jumper_crab.crab_isaac.robot import (  # noqa: E402
    ROBOT_PRIM,
    CrabRobot,
    add_ground,
    articulation_report,
    configure_physics,
    quat_to_matrix,
    use_fabric,
)

use_fabric()  # as the example does

PHYSICS_DT = 0.001
FALL_TILT_DEG = 30.0
checks: list[dict] = []


def check(label: str, passed: bool, detail: str) -> None:
    checks.append({"check": label, "pass": bool(passed), "detail": detail})
    print(f"[check] {'PASS' if passed else 'FAIL'} {label}: {detail}", flush=True)


class IsaacRecorder:
    """Off-screen RTX frames of the follow camera, written to an MP4.

    `world.render()` updates Fabric and draws with physics stepping switched off
    (isaacsim.core.api SimulationContext.render), so recording does not change
    the drive it records.
    """

    CAMERA = "/World/VideoCam"

    def __init__(self, world, stage, path: Path) -> None:
        import imageio.v2 as imageio
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom, UsdLux

        cam = UsdGeom.Camera.Define(stage, self.CAMERA)
        cam.CreateFocalLengthAttr(video.focal_length_mm())
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        dome = UsdLux.DomeLight.Define(stage, "/World/VideoDome")
        dome.CreateIntensityAttr(900.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/VideoSun")
        sun.CreateIntensityAttr(2500.0)
        sun.CreateAngleAttr(1.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))

        self._tiled_floor(stage)
        self.world = world
        self.render_product = rep.create.render_product(self.CAMERA, (video.SIZE, video.SIZE))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([self.render_product])
        self.clock = video.FrameClock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=video.FPS, codec="libx264",
                                         quality=8, pixelformat="yuv420p")
        self.path = path
        self.frames = 0
        self.last = None

    @staticmethod
    def _tiled_floor(stage, tile: float = 0.1, half_extent: float = 4.0) -> None:
        """A checkered floor for the camera only: no collision, so the checks are
        unchanged. On a plain floor a follow camera shows a robot that never moves;
        tiles passing under it show the walk, as MuJoCo's checker floor does."""
        from pxr import Gf, UsdGeom, Vt

        n = int(round(2 * half_extent / tile))
        xs = np.linspace(-half_extent, half_extent, n + 1)
        grid = np.stack(np.meshgrid(xs, xs, indexing="ij"), axis=-1).reshape(-1, 2)
        points = np.concatenate([grid, np.full((len(grid), 1), 0.0005)], axis=1)
        idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
        quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]],
                         -1).reshape(-1, 4)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        dark = ((ii + jj) % 2).reshape(-1).astype(bool)
        colors = np.where(dark[:, None], [0.28, 0.30, 0.33], [0.42, 0.44, 0.47])
        mesh = UsdGeom.Mesh.Define(stage, "/World/VideoFloor")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(quads)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(quads.reshape(-1).astype(np.int32)))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        primvar = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
        primvar.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, c)) for c in colors]))
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-half_extent, -half_extent, 0),
                                             Gf.Vec3f(half_extent, half_extent, 0.001)]))

    def aim(self, base_pos) -> None:
        from isaacsim.core.utils.viewports import set_camera_view

        tgt = video.target(base_pos)
        set_camera_view(eye=tgt + video.EYE_OFFSET, target=tgt, camera_prim_path=self.CAMERA)

    def warm_up(self, base_pos, renders: int = 30) -> None:
        """RTX streams textures and settles over its first frames; spend them here."""
        self.aim(base_pos)
        for _ in range(renders):
            self.world.render()

    def maybe_frame(self, t: float, base_pos) -> None:
        if not self.clock.due(t):
            return
        self.aim(base_pos)
        self.world.render()
        rgba = np.asarray(self.annotator.get_data())
        # While the renderer warms up the annotator returns nothing. Skipping those
        # frames shifted every later one early: 27 frames (0.9 s) in a headless
        # container (2026-10-22), so the clip no longer lined up with MuJoCo's.
        # Hold the last frame instead (black before the first), keeping one frame per tick.
        if rgba.size == 0:
            rgb = self.last if self.last is not None else np.zeros(
                (video.SIZE, video.SIZE, 3), np.uint8)
        else:
            rgb = self.last = rgba[:, :, :3]
        self.writer.append_data(rgb)
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        print(f"[isaac] video: {self.frames} frames -> {self.path}", flush=True)


def grip_materials(stage) -> list:
    """The robot's materials combined with `max`: the feet and palm pads."""
    found = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{ROBOT_PRIM}/materials")):
        if prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
            mode = PhysxSchema.PhysxMaterialAPI(prim).GetFrictionCombineModeAttr().Get()
            if mode == "max":
                found.append(prim)
    return found


world = World(physics_dt=PHYSICS_DT, rendering_dt=0.02, stage_units_in_meters=1.0,
              backend="numpy", device="cpu")
configure_physics(world)
add_ground(world, friction=args.friction)
robot = CrabRobot(world)
stage = get_current_stage()
if args.friction is not None:
    grips = grip_materials(stage)
    if not grips:
        raise RuntimeError("no grip material found to set the friction on")
    for prim in grips:
        api = UsdPhysics.MaterialAPI(prim)
        api.GetStaticFrictionAttr().Set(args.friction)
        api.GetDynamicFrictionAttr().Set(args.friction)
world.reset()
robot.initialize()
meta = robot.meta

# 1. What PhysX built.
rep = articulation_report(robot)
want = {b["name"]: b["mass"] for b in meta["bodies"]}
got = dict(zip(rep["body_names"], rep["masses"]))
dm = max(abs(got[n] - want[n]) for n in want)
check("link masses", dm < 1e-5 and set(got) == set(want),
      f"{len(got)} links, total {sum(got.values()):.4f} kg (MuJoCo {meta['total_mass']:.4f}), "
      f"max diff {dm:.2e} kg")
dl = np.abs(rep["dof_limits"] - robot.limits).max()
check("joint limits", dl < 1e-4, f"max diff {dl:.2e} rad over {len(robot.dof_names)} joints")

# 2. Forward kinematics at the default stance, gravity off, base in the air.
ctx = world.get_physics_context()
ctx.set_gravity(0.0)
robot.art.set_world_pose(position=np.array([0.0, 0.0, 1.0]), orientation=np.array([1.0, 0, 0, 0]))
robot.art.set_joint_positions(robot.default_q)
robot.art.set_joint_velocities(np.zeros(len(robot.dof_names)))
world.physics_sim_view.update_articulations_kinematic()
tf = np.asarray(robot.art._articulation_view._physics_view.get_link_transforms()).reshape(-1, 7)
links = dict(zip(rep["body_names"], tf))
base = links["base_link"]
R = quat_to_matrix([base[6], base[3], base[4], base[5]])  # tensors give x, y, z, w
errs = {}
for name, ref in meta["fk_check_default_pose"].items():
    p = R.T @ (links[name][:3] - base[:3])
    errs[name] = float(np.linalg.norm(p - np.asarray(ref)))
worst = max(errs, key=errs.get)
check("forward kinematics", errs[worst] < 1e-4,
      f"worst link {worst} off by {errs[worst] * 1e3:.4f} mm (41 links vs MuJoCo)")
ctx.set_gravity(-9.81)

# The world as the sweep asks for it, after the asset checks.
world.reset()
if args.payload_kg:
    view = robot.art._articulation_view
    masses = np.asarray(view.get_body_masses()).reshape(1, -1).copy()
    masses[0, rep["body_names"].index("base_link")] += args.payload_kg
    view.set_body_masses(masses)
actuator = dict(meta["actuator"])
for key in ("plateau_torque", "force_limit", "continuous_torque"):
    actuator[key] = actuator[key] * args.motor_strength
params = {
    "label": args.label,
    "friction": args.friction if args.friction is not None else "trained (1.2)",
    "payload_kg": args.payload_kg,
    "motor_strength": args.motor_strength,
    "tolerance": args.tolerance,
}
print(f"[isaac] conditions: {params}", flush=True)

# 3. The scripted drive.
robot.place_at_default()
recorder = None
if args.video:
    recorder = IsaacRecorder(world, stage, args.video)
    recorder.warm_up(robot.base_pose()[0])
capp = CrabApp(paths.BUNDLE, robot.dof_names, robot.limits, PHYSICS_DT, actuator,
               log=lambda s: print(s, flush=True))
rec = scenario.Recorder()
script = list(scenario.SCRIPT)
steps = int(round(scenario.DURATION / PHYSICS_DT))
wall = time.perf_counter()
for k in range(steps):
    t = k * PHYSICS_DT
    while script and script[0][0] <= t + 1e-9:
        _, key, down = script.pop(0)
        capp.key(key, down)
    s = robot.state()
    robot.apply(capp.physics_step(s), s.joint_vel)
    world.step(render=False)
    if k % 10 == 0:
        pos, quat = robot.base_pose()
        rec.add(t + PHYSICS_DT, pos, quat)
    if recorder is not None:
        recorder.maybe_frame(t + PHYSICS_DT, robot.base_pose()[0])
wall = time.perf_counter() - wall
if recorder is not None:
    recorder.close()
print(f"[isaac] {scenario.DURATION:.0f} s simulated in {wall:.1f} s wall "
      f"({scenario.DURATION / wall:.2f}x real time, headless)", flush=True)

m = rec.metrics()
scenario.print_metrics("isaac", m)
fell = m["max_tilt_deg"] >= FALL_TILT_DEG or m["final_height"] < 0.05
check("stays upright", not fell,
      f"max tilt {m['max_tilt_deg']:.1f} deg, final height {m['final_height']:.4f} m")

ref_path = args.reference or (paths.DATA / "mujoco_reference.json")
compared = {}
if ref_path.exists():
    ref = json.loads(ref_path.read_text())
    scenario.print_metrics("mujoco", ref)
    for name, key in (("forward", "vx"), ("back", "vx"), ("right", "vy"), ("turn", "yaw_rate")):
        a, b = m[name][key], ref[name][key]
        rel = abs(a - b) / max(abs(b), 1e-6)
        compared[name] = {"isaac": a, "mujoco": b, "gap": round(rel, 4)}
        check(f"{name} {key}", rel < args.tolerance,
              f"isaac {a:+.3f} vs mujoco {b:+.3f} ({rel * 100:.1f}% apart)")
else:
    check("mujoco reference", False, f"{ref_path} not found; run tools/mujoco_reference.py")

ok = all(c["pass"] for c in checks)
out = args.out or (paths.DATA / "isaac_result.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "_source": "tools/isaac_check.py",
    "params": params,
    "pass": ok,
    "checks": checks,
    "compared": compared,
    "metrics": m,
    "wall_s": round(wall, 1),
}, indent=1))
print(f"[check] wrote {out}", flush=True)
print(f"[check] {'ALL PASSED' if ok else 'SOME CHECKS FAILED'}", flush=True)
# SimulationApp.close() hangs in Kit's shutdown here (Isaac Sim 5.1).
os._exit(0 if ok else 1)
