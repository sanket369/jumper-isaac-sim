# The Jumper crab robot in Isaac Sim, walking on the robot's own controller.
#
# Structure follows the unitree_g1 example: a BaseSample, a robot wrapper and a
# controller driven from a physics callback. What differs is the controller:
# the bundle's controller.so builds the observation and runs the mode state
# machine (crab_core/app_controller.py), so the 415-value observation is the
# robot's own code rather than a Python re-implementation of it.
#
# No station keeping, no warm-up hold, no pause/play restart: the controller's
# own warm start (2 s) brings the robot up, and the robot does what the policy
# does with the command the keys give it.

from __future__ import annotations

import carb
import omni.appwindow
import omni.timeline
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.interactive.base_sample import BaseSample

from ..crab_core import paths
from ..crab_core.app_controller import CrabApp
from ..crab_core.keys import KeyEdges, carb_to_vocabulary
from ..crab_isaac.robot import (
    CrabRobot,
    add_ground,
    add_lights,
    configure_physics,
    use_fabric_async,
)

PHYSICS_DT = 0.001  # the training timestep; the controller ticks every 5th step (200 Hz)
RENDERING_DT = 0.02
CAMERA_PRIM = "/OmniverseKit_Persp"
FOLLOW_KEY = "C"  # not a key the controller reads


def log(message: str) -> None:
    print(message, flush=True)
    carb.log_info(message)


class CrabExample(BaseSample):
    """Load the crab, start its controller, drive it from the keyboard."""

    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = PHYSICS_DT
        self._world_settings["rendering_dt"] = RENDERING_DT
        self.robot: CrabRobot | None = None
        self.app: CrabApp | None = None
        table, untranslated = carb_to_vocabulary(paths.VOCABULARY)
        if untranslated:
            log(f"[crab] keys with no carb name, unavailable here: {untranslated}")
        self._keys = KeyEdges(table)
        self._sub_keyboard = None
        self._timeline_sub = None
        self._needs_init = True
        self._follow = True
        self._last_base_xy = None
        self._physics_steps = 0

    # ── loading ────────────────────────────────────────────────────────

    async def load_world_async(self):
        await use_fabric_async()  # before the world exists; see crab_isaac/robot.py
        await super().load_world_async()

    def setup_scene(self) -> None:
        world = self.get_world()
        configure_physics(world)
        add_ground(world)
        add_lights(omni.usd.get_context().get_stage())
        self.robot = CrabRobot(world)
        set_camera_view(eye=[0.55, 0.55, 0.35], target=[0.0, 0.0, 0.06],
                        camera_prim_path=CAMERA_PRIM)

    async def setup_post_load(self) -> None:
        self.robot.initialize()
        self.app = CrabApp(paths.BUNDLE, self.robot.dof_names, self.robot.limits,
                           PHYSICS_DT, self.robot.meta["actuator"], log=log)
        self.app.key_source = self._keys
        log(f"[crab] bundle {paths.BUNDLE}")
        log(f"[crab] modes: {', '.join(sorted(self.app.sessions))}; controller ticks at "
            f"{self.app.control_hz:g} Hz, every {self.app.decimation} physics steps")
        for line in self.app.help_lines():
            log(line)
        log(f"[crab] {FOLLOW_KEY}: camera follows the robot on/off. Click the viewport first "
            "so it has the keyboard.")

        appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            appwindow.get_keyboard(), self._on_keyboard)
        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._timeline_sub = stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), self._on_timeline_stop)
        self._needs_init = True
        world = self.get_world()
        if not world.physics_callback_exists("crab_step"):
            world.add_physics_callback("crab_step", callback_fn=self._on_physics_step)
        await world.play_async()

    async def setup_pre_reset(self) -> None:
        world = self.get_world()
        if world.physics_callback_exists("crab_step"):
            world.remove_physics_callback("crab_step")

    async def setup_post_reset(self) -> None:
        self._needs_init = True
        world = self.get_world()
        if not world.physics_callback_exists("crab_step"):
            world.add_physics_callback("crab_step", callback_fn=self._on_physics_step)
        await world.play_async()

    async def setup_post_clear(self) -> None:
        self._drop_subscriptions()
        self.robot = None
        self.app = None

    def world_cleanup(self) -> None:
        self._drop_subscriptions()

    def _drop_subscriptions(self) -> None:
        if self._sub_keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(
                omni.appwindow.get_default_app_window().get_keyboard(), self._sub_keyboard)
        self._sub_keyboard = None
        self._timeline_sub = None

    def _on_timeline_stop(self, event) -> None:
        # Stop returns the stage to its authored state (every joint at zero);
        # the next play starts the robot and the controller again from the stance.
        self._needs_init = True

    # ── the loop ───────────────────────────────────────────────────────

    def _on_physics_step(self, step_size: float) -> None:
        if self.app is None or self.robot is None:
            return
        if abs(step_size - PHYSICS_DT) > 1e-9:
            carb.log_warn(f"[crab] physics step {step_size} s, expected {PHYSICS_DT}; skipped")
            return
        if self._needs_init:
            self.robot.initialize()
            self.robot.place_at_default()
            for name in self._keys.release_all():
                self.app.key(name, False)
            self.app.reset()
            self._last_base_xy = None
            self._needs_init = False
            log("[crab] robot at the stance; controller started")
        state = self.robot.state()
        self.robot.apply(self.app.physics_step(state), state.joint_vel)
        self._physics_steps += 1
        if self._follow and self._physics_steps % 20 == 0:
            self._follow_camera()

    def _follow_camera(self) -> None:
        """Carry the camera along with the robot, keeping the person's view of it."""
        pos, _ = self.robot.base_pose()
        xy = pos[:2].copy()
        if self._last_base_xy is not None:
            delta = xy - self._last_base_xy
            prim = omni.usd.get_context().get_stage().GetPrimAtPath(CAMERA_PRIM)
            attr = prim.GetAttribute("xformOp:translate") if prim.IsValid() else None
            if attr is not None and attr.IsValid() and attr.Get() is not None:
                t = attr.Get()
                attr.Set(type(t)(t[0] + float(delta[0]), t[1] + float(delta[1]), t[2]))
        self._last_base_xy = xy

    # ── the keyboard ───────────────────────────────────────────────────

    def _on_keyboard(self, event, *args, **kwargs) -> bool:
        if event.type not in (carb.input.KeyboardEventType.KEY_PRESS,
                              carb.input.KeyboardEventType.KEY_RELEASE):
            return True  # repeats carry nothing: a held key is held until released
        name = event.input.name if hasattr(event.input, "name") else str(event.input)
        down = event.type == carb.input.KeyboardEventType.KEY_PRESS
        if name == FOLLOW_KEY and down:
            self._follow = not self._follow
            self._last_base_xy = None
            log(f"[crab] camera follow {'on' if self._follow else 'off'}")
            return True
        self._keys.report(name, down)
        return True
