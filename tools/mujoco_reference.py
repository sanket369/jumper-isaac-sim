#!/usr/bin/env python3
"""The scripted drive on MuJoCo, through the same controller loop Isaac runs.

Run from the Jumper checkout in its venv; bootstrap.sh does this:

    cd third_party/jumper && .venv/bin/python ../../tools/mujoco_reference.py

The model is the compiled jumper.posture play environment (what the policy was
trained on); the loop is `crab_core.app_controller.CrabApp`, the one the Isaac
example runs. MuJoCo applies joint damping and friction loss itself (they are in
the model), so only the servo torque is written, into the motor actuators.
Writes data/mujoco_reference.json for `tools/isaac_check.py` to compare.

--video PATH records the drive as an MP4 through crab_core/video.py's camera --
the same camera tools/isaac_check.py uses -- so the two clips match shot for
shot. Rendering is off-screen (MUJOCO_GL=egl on a GPU machine, osmesa without).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exts" / "jumper.crab_example"))
sys.path.insert(0, str(HERE))

from export_model import PREFIX, build_env  # noqa: E402
from jumper_crab.crab_core import paths, scenario, video  # noqa: E402
from jumper_crab.crab_core.app_controller import CrabApp, RobotState  # noqa: E402


class MujocoRecorder:
    """Off-screen frames of the follow camera, written to an MP4."""

    def __init__(self, m: mujoco.MjModel, path: Path) -> None:
        import imageio.v2 as imageio

        self.renderer = mujoco.Renderer(m, video.SIZE, video.SIZE)
        if os.environ.get("MUJOCO_GL") == "osmesa":
            # OSMesa's shadow mapping draws flickering dark slabs over the background:
            # 132 of the first 181 frames in a headless CPU container (2026-10-22), at
            # every shadowsize from 2048 to 8192; 0 with shadows off. EGL is unaffected.
            # update_scene keeps scene flags, so setting it once holds for the clip.
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.option = mujoco.MjvOption()
        self.option.geomgroup[1] = 0  # collision hulls: hidden, the meshes show
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance, self.camera.azimuth, self.camera.elevation = video.mujoco_angles()
        self.clock = video.FrameClock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(path), fps=video.FPS, codec="libx264",
                                         quality=8, pixelformat="yuv420p")
        self.path = path
        self.frames = 0

    def maybe_frame(self, t: float, d: mujoco.MjData, base_pos) -> None:
        if not self.clock.due(t):
            return
        self.camera.lookat[:] = video.target(base_pos)
        self.renderer.update_scene(d, camera=self.camera, scene_option=self.option)
        self.writer.append_data(self.renderer.render())
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        print(f"[mujoco] video: {self.frames} frames -> {self.path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", type=Path, default=None, help="record the drive to this MP4")
    args = ap.parse_args()
    meta = json.loads(paths.MODEL_JSON.read_text())
    env = build_env(meta["task"])
    m = env.sim.mj_model
    d = mujoco.MjData(m)
    names = meta["joint_names"]
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + n) for n in names]
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    vadr = np.array([m.jnt_dofadr[j] for j in jid])
    act_of_joint = np.array([
        next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == j) for j in jid])
    limits = np.array([m.jnt_range[j] for j in jid])
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + "floating_base")
    b_q, b_v = m.jnt_qposadr[base], m.jnt_dofadr[base]

    app = CrabApp(paths.BUNDLE, names, limits, m.opt.timestep, meta["actuator"])
    root = meta["root_init"]
    d.qpos[b_q:b_q + 3] = root[0:3]
    d.qpos[b_q + 3:b_q + 7] = root[3:7]
    d.qpos[qadr] = meta["default_joint_pos"]
    mujoco.mj_forward(m, d)

    rec = scenario.Recorder()
    recorder = MujocoRecorder(m, args.video) if args.video else None
    script = list(scenario.SCRIPT)
    dt = m.opt.timestep
    steps = int(round(scenario.DURATION / dt))
    for k in range(steps):
        t = k * dt
        while script and script[0][0] <= t + 1e-9:
            _, key, down = script.pop(0)
            app.key(key, down)
        state = RobotState(
            joint_pos=d.qpos[qadr].copy(),
            joint_vel=d.qvel[vadr].copy(),
            base_quat_wxyz=d.qpos[b_q + 3:b_q + 7].copy(),
            # A free joint's angular velocity is already in the body frame.
            base_ang_vel_body=d.qvel[b_v + 3:b_v + 6].copy(),
        )
        tau = app.physics_step(state)
        d.ctrl[act_of_joint] = tau
        mujoco.mj_step(m, d)
        if k % 10 == 0:
            rec.add(t + dt, d.qpos[b_q:b_q + 3], d.qpos[b_q + 3:b_q + 7])
        if recorder is not None:
            recorder.maybe_frame(t + dt, d, d.qpos[b_q:b_q + 3])

    if recorder is not None:
        recorder.close()
    metrics = rec.metrics()
    scenario.print_metrics("mujoco", metrics)
    out = paths.DATA / "mujoco_reference.json"
    out.write_text(json.dumps({"_source": "tools/mujoco_reference.py", **metrics}, indent=1))
    print(f"[mujoco] wrote {out}")


if __name__ == "__main__":
    main()
