"""Run the interactive example inside the real Isaac Sim app and drive it by its keyboard queue.

    "$ISAAC_SIM_PATH/isaac-sim.sh" --no-window --ext-folder exts \
        --enable jumper.crab_example --exec tools/example_check.py

(scripts/sim2sim_check.sh --app runs exactly this.) `--exec` runs this inside the
full app's own event loop, the one the example browser's LOAD button runs in,
with this repository's extension enabled. It presses LOAD in code, lets Kit step physics
through the example's physics callback, and sends the scripted keys as carb
names through the same queue the keyboard callback fills. Then it compares with
the MuJoCo reference, as tools/isaac_check.py does, and quits with the result.

(A standalone SimulationApp script cannot do this: driving BaseSample's
coroutines from a script's own loop trips asyncio inside Kit, before any crab
code runs.)
"""

import asyncio
import json
import time
import traceback

import omni.kit.app

CARB = {"key_w": "W", "key_s": "S", "key_d": "D", "key_j": "J"}


def say(message: str) -> None:
    print(message, flush=True)


async def main() -> int:
    kit = omni.kit.app.get_app()
    for _ in range(30):
        await kit.next_update_async()
    from jumper_crab.crab_core import paths, scenario
    from jumper_crab.example import CrabExample

    sample = CrabExample()
    await sample.load_world_async()
    say(f"[example] loaded; playing: {sample.get_world().is_playing()}")

    rec = scenario.Recorder()
    script = list(scenario.SCRIPT)
    wall = time.perf_counter()
    frames = 0
    while True:
        capp = sample.app
        t = capp.physics_count * capp.physics_dt
        if t >= scenario.DURATION:
            break
        while script and script[0][0] <= t + 1e-9:
            _, key, down = script.pop(0)
            sample._keys.report(CARB[key], down)
        await kit.next_update_async()
        frames += 1
        if capp.physics_count > 0:
            pos, quat = sample.robot.base_pose()
            rec.add(capp.physics_count * capp.physics_dt, pos, quat)
        if frames > 200000:
            raise RuntimeError("physics is not advancing")
    wall = time.perf_counter() - wall
    say(f"[example] {scenario.DURATION:.0f} s simulated in {wall:.1f} s wall over {frames} frames "
        f"({scenario.DURATION / wall:.2f}x real time, full app, no window)")

    m = rec.metrics()
    scenario.print_metrics("example", m)
    ref = json.loads((paths.DATA / "mujoco_reference.json").read_text())
    ok = m["max_tilt_deg"] < 15.0
    say(f"[example] {'PASS' if ok else 'FAIL'} upright: max tilt {m['max_tilt_deg']:.1f} deg")
    for name, key in (("forward", "vx"), ("back", "vx"), ("right", "vy"), ("turn", "yaw_rate")):
        a, b = m[name][key], ref[name][key]
        rel = abs(a - b) / max(abs(b), 1e-6)
        ok &= rel < 0.2
        say(f"[example] {'PASS' if rel < 0.2 else 'FAIL'} {name} {key}: {a:+.3f} vs mujoco "
            f"{b:+.3f} ({rel * 100:.1f}% apart)")
    say(f"[example] {'ALL PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


async def run() -> None:
    code = 2
    try:
        code = await main()
    except Exception:
        traceback.print_exc()
    finally:
        # post_quit() leaves the app hanging in Kit's shutdown (Isaac Sim 5.1,
        # 2026-10-19), as SimulationApp.close() does; the result is printed.
        import os
        import sys

        sys.stdout.flush()
        os._exit(code)


asyncio.ensure_future(run())
