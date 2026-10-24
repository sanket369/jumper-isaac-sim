"""Load the crab into the open Isaac Sim app, ready to drive: run_teleop.sh's --exec.

    "$ISAAC_SIM_PATH/isaac-sim.sh" --ext-folder exts --enable jumper.crab_example \
        --exec tools/teleop.py

The same as Examples > Policy > Crab > LOAD in the example browser, done once the
app is up. The example subscribes to the keyboard itself; click the viewport so
it has the focus.
"""

import asyncio
import traceback

import omni.kit.app

_sample = None  # held for the life of the app: the example owns the physics callback


async def main() -> None:
    global _sample
    kit = omni.kit.app.get_app()
    for _ in range(60):  # let the window and the extensions finish starting
        await kit.next_update_async()
    try:
        from jumper_crab.example import CrabExample

        _sample = CrabExample()
        await _sample.load_world_async()
        print("[teleop] the crab is loaded. Click the viewport, then W / S, A / D, J / L "
              "to drive; C toggles the follow camera. Keys: docs/CONTROLS.md", flush=True)
        if _sample.conditions.active():
            print(f"[teleop] conditions: {_sample.conditions.describe()}", flush=True)
    except Exception:
        traceback.print_exc()
        print("[teleop] loading failed; the error is above.", flush=True)


asyncio.ensure_future(main())
