"""The jumper app, driven from any simulator: the robot's own controller in the loop.

This is `mjrl/app_play.py::AppPlayer` with mjlab taken out. The bundle's
`controller.so` -- the same Rust crate the robot's board runs -- builds every
observation (histories at stride 4, the gated gait clock, torque feedback, the
posture command), runs the mode state machine and publishes joint targets and
gains. This module only feeds it the robot's state, runs the ONNX it asks for,
and turns its targets into joint torque with the training actuator law.

Nothing here imports a simulator. A simulator adapter calls `physics_step` once
per physics step, before stepping, with the state in its own joint order, and
applies the torque it gets back. That is what lets the MuJoCo reference
(`tools/mujoco_reference.py`) and Isaac Sim run the identical loop.

Timing, as in `play --app`: the controller is ticked at the app's fastest mode's
rate (200 Hz), each tick stamped with simulation time; the targets it publishes
are held for the physics steps until the next tick, and the PD torque is
recomputed every physics step from the state of that step.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .servo import ServoCurve


def load_extension(bundle: Path):
    """The bundle's controller for this host, imported as `mjrl_fsm`."""
    manifest = json.loads((bundle / "bundle.json").read_text("utf-8"))
    ext = manifest["runtimes"]["mjlab"]["extensions"].get("linux-x86_64")
    if ext is None:
        raise RuntimeError(f"{bundle}: no linux-x86_64 controller in this bundle")
    path = bundle / ext["file"]
    if "mjrl_fsm" in sys.modules:
        return sys.modules["mjrl_fsm"]
    loader = importlib.machinery.ExtensionFileLoader("mjrl_fsm", str(path))
    spec = importlib.util.spec_from_file_location("mjrl_fsm", path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules["mjrl_fsm"] = module
    return module


@dataclass
class RobotState:
    """One physics step's state, in the simulator's joint order."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    base_quat_wxyz: np.ndarray
    base_ang_vel_body: np.ndarray  # rad/s, in the base frame (the IMU's)


class CrabApp:
    """The app: controller, policies, and the servo between them and the joints."""

    def __init__(
        self,
        bundle: Path,
        sim_joint_names: list[str],
        sim_joint_limits: np.ndarray,
        physics_dt: float,
        actuator: dict,
        log=print,
    ) -> None:
        import onnxruntime as ort

        self.bundle = Path(bundle)
        self.log = log
        manifest = json.loads((self.bundle / "bundle.json").read_text("utf-8"))
        self._fsm_mod = load_extension(self.bundle)
        self._contracts = {
            mode: (self.bundle / e["contract"]).read_text("utf-8")
            for mode, e in manifest["modes"].items()
        }
        self._trajectories = {
            mode: (self.bundle / e["trajectory"]).read_text("utf-8")
            for mode, e in manifest["modes"].items() if e.get("trajectory")
        }
        parsed = {mode: json.loads(t) for mode, t in self._contracts.items()}
        self.wire: list[str] = next(iter(parsed.values()))["wire_joint_order"]
        for mode, c in parsed.items():
            if c["wire_joint_order"] != self.wire:
                raise RuntimeError(f"{mode}: its wire joint order differs from the others'")

        # Paired by name: by index is wrong from the fifth joint on and silent.
        missing = [n for n in self.wire if n not in sim_joint_names]
        if missing:
            raise RuntimeError(f"the simulator's robot has no {missing}")
        self.wire_from_sim = np.array([sim_joint_names.index(n) for n in self.wire])
        self.n_sim = len(sim_joint_names)
        limits = np.asarray(sim_joint_limits)[self.wire_from_sim]
        self._limits = limits

        self.control_hz = max(float(c["control"]["control_hz"]) for c in parsed.values())
        steps = 1.0 / (self.control_hz * physics_dt)
        if round(steps) < 1 or abs(steps - round(steps)) > 1e-6:
            raise RuntimeError(f"physics dt {physics_dt} is not a whole fraction of "
                               f"the app's {self.control_hz:g} Hz")
        self.decimation = round(steps)
        self.physics_dt = physics_dt
        self.step_us = round(1e6 / self.control_hz)
        gait = next((t.get("params") or {} for c in parsed.values()
                     for t in c["observation"]["terms"] if t["name"] == "gait_phase"), {})
        self._fsm_config = {
            "joint_names": self.wire,
            "joint_pos_lo": [float(v) for v in limits[:, 0]],
            "joint_pos_hi": [float(v) for v in limits[:, 1]],
            "output_rate_hz": self.control_hz,
            "gait_period": gait.get("period"),
            "gait_gate_threshold": gait.get("command_threshold"),
        }
        self._toml = (self.bundle / "controller.toml").read_text("utf-8")
        self.sessions = {
            mode: ort.InferenceSession(str(self.bundle / e["models"]["onnx"]),
                                       providers=["CPUExecutionProvider"])
            for mode, e in manifest["modes"].items()
        }
        self._inputs = {m: s.get_inputs()[0].name for m, s in self.sessions.items()}
        self.servo = ServoCurve(actuator, len(self.wire), physics_dt)
        #: Something with `changes() -> (downs, ups)`, read once a tick: the
        #: keyboard in the example (`keys.KeyEdges`). Scripted keys use `key()`.
        self.key_source = None
        self.fsm = None
        self.reset()

    # ── lifecycle ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """A fresh controller, cold servos, clock at zero."""
        self.fsm = self._fsm_mod.Fsm(self._toml, self._contracts, self._fsm_config,
                                     trajectories=self._trajectories, operators=True)
        self.servo.reset()
        self.physics_count = 0
        self.tick_count = 0
        self.mode = ""
        self.targets = None
        self.kp = np.zeros(len(self.wire))
        self.kd = np.zeros(len(self.wire))
        self.last_torque = np.zeros(len(self.wire))
        self._pending_keys: list[tuple[str, bool]] = []

    def key(self, name: str, down: bool) -> None:
        """A key edge in the controller's names, delivered at the next tick."""
        self._pending_keys.append((name, down))

    # ── the loop ───────────────────────────────────────────────────────

    def physics_step(self, state: RobotState) -> np.ndarray:
        """Call once per physics step, before stepping. Returns servo torque, sim order."""
        q = np.asarray(state.joint_pos, dtype=np.float64)[self.wire_from_sim]
        qd = np.asarray(state.joint_vel, dtype=np.float64)[self.wire_from_sim]
        if self.physics_count % self.decimation == 0:
            self._tick(q, qd, state)
        self.physics_count += 1
        tau = self.servo.torque(self.kp, self.kd, self.targets, q, qd)
        self.last_torque = tau
        out = np.zeros(self.n_sim)
        out[self.wire_from_sim] = tau
        return out

    def _tick(self, q, qd, state: RobotState) -> None:
        now = self.tick_count * self.step_us
        self.tick_count += 1
        self.fsm.set_state(
            [float(v) for v in q],
            [float(v) for v in qd],
            [float(v) for v in self.last_torque],
            [float(v) for v in state.base_quat_wxyz],
            [float(v) for v in state.base_ang_vel_body],
            now,
        )
        keys, self._pending_keys = self._pending_keys, []
        if self.key_source is not None:
            downs, ups = self.key_source.changes()
            keys += [(n, True) for n in downs] + [(n, False) for n in ups]
        for name, down in keys:
            self.fsm.set_key(name, down, now)
        # Every tick, pad or none: a frame is also what says the person is there.
        self.fsm.set_pad_frame([], {}, connected=False, now_us=now)

        mode = self.fsm.tick(now)
        if mode is not None:
            obs = np.asarray([self.fsm.observation()], dtype=np.float32)
            action = self.sessions[mode].run(None, {self._inputs[mode]: obs})[0][0]
            self.fsm.resume([float(v) for v in action])
        self.targets = np.asarray(self.fsm.positions(), dtype=np.float64)
        self.kp = np.asarray(self.fsm.kp(), dtype=np.float64)
        self.kd = np.asarray(self.fsm.kd(), dtype=np.float64)
        current = self.fsm.mode
        if current != self.mode:
            self.log(f"[crab] mode: {self.mode or '-'} -> {current}  (t={now / 1e6:.2f} s)")
            self.mode = current

    # ── help ───────────────────────────────────────────────────────────

    def help_lines(self) -> list[str]:
        """The keyboard, as the bundle's own manual states it."""
        path = self.bundle / "manual.en.json"
        if not path.exists():
            return []
        manual = json.loads(path.read_text("utf-8"))
        lines = []
        for m in manual.get("modes", []):
            if m.get("default"):
                keys = ", ".join(f"{k['key']}: {k['does']}" for k in m.get("keys", []))
                lines.append(f"[crab] {m['mode']} keys -- {keys}")
        for s in manual.get("switches", []):
            if s.get("device") == "keyboard" and s.get("keys"):
                does = s["does"].split(" (from ")[0]
                lines.append(f"[crab] {' / '.join(s['keys'])}: {does}")
        return lines
