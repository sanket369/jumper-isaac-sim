#!/usr/bin/env python3
"""Record one level of "Can Jumper survive this?" in Isaac Sim, 1920x1080 at 30 fps.

    "$ISAAC_SIM_PATH/python.sh" tools/record_survival.py LEVEL OUT_DIR

LEVEL is one of:
  ice         floor friction 0.05, real ice (trained on 1.2); ICE_FRICTION=F for another
  backpack    +1 kg on the body (+39%), walking, then a jump with it on
  battery     motor power fading live from 100% to 40% while it walks, then recharged

The conditions are jumper_crab/crab_isaac/conditions.py, the same code the interactive
example runs for `run_teleop.sh --ice / --backpack / --battery`. The keys go through the
controller as a person's would. The camera follows the robot's position and, slowly, its
heading, so it keeps the front three-quarter view after a turn.

Writes OUT_DIR/LEVEL.mp4 and OUT_DIR/LEVEL.json: the controller's mode, the base pose
and the motor strength every 10 ms, for captions and on-screen readouts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exts" / "jumper.crab_example"))
if len(sys.argv) != 3:
    sys.exit(__doc__)
LEVEL, OUT = sys.argv[1], Path(sys.argv[2]).resolve()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import traceback  # noqa: E402


def _fail(kind, value, tb) -> None:
    # Kit swallows the exit status of an uncaught exception; make it a failure.
    traceback.print_exception(kind, value, tb)
    sys.stdout.flush()
    os._exit(2)


sys.excepthook = _fail

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from jumper_crab.crab_core import paths, video  # noqa: E402
from jumper_crab.crab_core.app_controller import CrabApp  # noqa: E402
from jumper_crab.crab_isaac.conditions import (  # noqa: E402
    BACKPACK_KG,
    ICE_FRICTION,
    MotorStrength,
    add_backpack,
    add_floor,
    add_ice_floor,
    add_payload,
    set_grip_friction,
)
from jumper_crab.crab_isaac.robot import (  # noqa: E402
    CrabRobot,
    add_ground,
    add_lights,
    configure_physics,
    use_fabric,
)

use_fabric()

PHYSICS_DT = 0.001
W, H, FPS = 1920, 1080, 30
CAMERA = "/World/SurvivalCam"
EYE_OFFSET = np.array([0.50, -0.50, 0.27])  # the three-quarter view, a little closer
FOLLOW_TAU = 0.15  # s; 0.6 s lagged 0.4 m behind a walking robot
HEADING_TAU = 1.0  # s; slow enough that a turn still reads

# (time s, controller key, down), the level's length, and its world.
LEVELS = {
    "ice": dict(script=[(3.0, "key_w", True), (9.0, "key_w", False),
                        (9.5, "key_j", True), (12.5, "key_j", False),
                        (13.0, "key_d", True), (16.5, "key_d", False)],
                duration=17.5, friction=float(os.environ.get("ICE_FRICTION", ICE_FRICTION))),
    "backpack": dict(script=[(3.0, "key_w", True), (9.0, "key_w", False),
                             (9.5, "key_j", True), (12.0, "key_j", False),
                             (13.5, "key_space", True), (13.6, "key_space", False)],
                     duration=19.0, payload_kg=BACKPACK_KG),
    "battery": dict(script=[(3.0, "key_w", True), (34.0, "key_w", False)], duration=35.0,
                    # (t, strength), linear in between: drains, sits at 40%, recharges.
                    strength=[(0.0, 1.0), (5.0, 1.0), (25.0, 0.4), (29.0, 0.4),
                              (29.01, 1.0), (35.0, 1.0)]),
}
if LEVEL not in LEVELS:
    sys.exit(f"LEVEL must be one of {', '.join(LEVELS)}")
cfg = LEVELS[LEVEL]
OUT.mkdir(parents=True, exist_ok=True)

world = World(physics_dt=PHYSICS_DT, rendering_dt=0.02, stage_units_in_meters=1.0,
              backend="numpy", device="cpu")
configure_physics(world)
add_ground(world, friction=cfg.get("friction"))
robot = CrabRobot(world)
stage = get_current_stage()
add_lights(stage)
if cfg.get("friction") is not None:
    set_grip_friction(stage, cfg["friction"])
    add_ice_floor(stage)
    print(f"[survival] floor friction {cfg['friction']:g}", flush=True)
elif LEVEL == "backpack":
    add_floor(stage, [(0.55, 0.40, 0.25), (0.42, 0.30, 0.18)], roughness=0.6)  # warm wood
    add_backpack(stage)
else:
    add_floor(stage, [(0.30, 0.31, 0.33), (0.20, 0.21, 0.23)], roughness=0.7)  # concrete

cam = UsdGeom.Camera.Define(stage, CAMERA)
cam.CreateFocalLengthAttr(video.focal_length_mm())  # 45 deg vertical
cam.CreateVerticalApertureAttr(20.955)
cam.CreateHorizontalApertureAttr(20.955 * W / H)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

world.reset()
robot.initialize()
if cfg.get("payload_kg"):
    before, after = add_payload(robot, cfg["payload_kg"])
    print(f"[survival] mass {before:.3f} kg -> {after:.3f} kg", flush=True)
robot.place_at_default()

render_product = rep.create.render_product(CAMERA, (W, H))
annotator = rep.AnnotatorRegistry.get_annotator("rgb")
annotator.attach([render_product])


def yaw_of(q) -> float:
    w, x, y, z = map(float, q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


look = video.target(robot.base_pose()[0]).astype(float)
yaw0, cam_yaw = yaw_of(robot.base_pose()[1]), 0.0


def aim(pose, dt: float) -> None:
    """Ease after the robot's position, and more slowly after its heading."""
    global look, cam_yaw
    pos, quat = pose
    look += (video.target(pos) - look) * (1.0 - np.exp(-dt / FOLLOW_TAU))
    d = (yaw_of(quat) - yaw0 - cam_yaw + np.pi) % (2 * np.pi) - np.pi  # shortest way round
    cam_yaw += d * (1.0 - np.exp(-dt / HEADING_TAU))
    c, s = np.cos(cam_yaw), np.sin(cam_yaw)
    off = np.array([c * EYE_OFFSET[0] - s * EYE_OFFSET[1], s * EYE_OFFSET[0] + c * EYE_OFFSET[1],
                    EYE_OFFSET[2]])
    set_camera_view(eye=look + off, target=look, camera_prim_path=CAMERA)


aim(robot.base_pose(), 1.0)
for _ in range(40):  # RTX settles over its first frames
    world.render()

capp = CrabApp(paths.BUNDLE, robot.dof_names, robot.limits, PHYSICS_DT, robot.meta["actuator"],
               log=lambda s: None)
motor = MotorStrength(capp.servo)
schedule = cfg.get("strength")
writer = imageio.get_writer(str(OUT / f"{LEVEL}.mp4"), fps=FPS, codec="libx264", quality=9,
                            pixelformat="yuv420p")
clock = video.FrameClock(FPS)
script = list(cfg["script"])
samples, last, frames = [], None, 0
for k in range(int(round(cfg["duration"] / PHYSICS_DT))):
    t = k * PHYSICS_DT
    while script and script[0][0] <= t + 1e-9:
        _, key, down = script.pop(0)
        capp.key(key, down)
    if schedule:
        ts, ks = zip(*schedule)
        motor.set(float(np.interp(t, ts, ks)))
    s = robot.state()
    robot.apply(capp.physics_step(s), s.joint_vel)
    world.step(render=False)
    if k % 10 == 0:
        pos, quat = robot.base_pose()
        samples.append({"t": round(t, 3), "mode": capp.mode, "strength": round(motor.k, 4),
                        "pos": [round(float(v), 4) for v in pos],
                        "quat": [round(float(v), 5) for v in quat]})
    if clock.due(t + PHYSICS_DT):
        aim(robot.base_pose(), 1.0 / FPS)
        world.render()
        rgba = np.asarray(annotator.get_data())
        # Hold the last frame while the renderer has nothing yet: one frame per tick.
        if rgba.size == 0:
            rgb = last if last is not None else np.zeros((H, W, 3), np.uint8)
        else:
            rgb = rgba[:, :, :3]
        last = rgb
        writer.append_data(rgb)
        frames += 1
writer.close()
(OUT / f"{LEVEL}.json").write_text(json.dumps({"level": LEVEL, "script": cfg["script"],
                                               "samples": samples}))
modes = []
for r in samples:
    if not modes or modes[-1][1] != r["mode"]:
        modes.append((r["t"], r["mode"]))
print(f"[survival] {LEVEL}: {frames} frames -> {OUT / (LEVEL + '.mp4')}; modes {modes}",
      flush=True)
os._exit(0)
