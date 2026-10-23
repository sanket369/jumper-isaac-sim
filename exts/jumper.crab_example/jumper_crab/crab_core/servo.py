"""The joint torque the training simulator applied, step for step.

A numpy transcription of `tasks/jumper/common/actuator.py::ServoCurveActuator`
(mjlab's explicit PD, clipped to the measured torque-speed curve, with the
thermal budget) plus the passive joint terms of the MJCF `motor` class. The
constants are read from crab_model.json, which `tools/export_model.py` took
from the live training actuator, so nothing here is retyped.

Two parts, kept apart because the policy observes only one of them:

* `ServoCurve.torque` -- what mjlab calls `actuator_force`. This is the
  `joint_torque` observation, and is what the controller is handed.
* `passive` -- joint damping and friction loss, which MuJoCo applies inside the
  integrator and which never appear in `actuator_force`.
"""

from __future__ import annotations

import numpy as np

#: Friction loss is a dry-friction constraint in MuJoCo. Applied explicitly it
#: needs smoothing; `tanh(v / eps)` is it. 0.02 rad/s keeps the slope near zero
#: (0.011 / 0.02 = 0.55 N*m*s/rad) inside the explicit-step stability bound at
#: 1 ms with the 0.0015 kg*m^2 armature, and the term itself is 0.6% of the
#: torque plateau.
FRICTIONLOSS_SMOOTHING = 0.02  # rad/s


class ServoCurve:
    """PD torque under the servo's curve, one value per joint."""

    def __init__(self, actuator: dict, n: int, dt: float) -> None:
        self.plateau = float(actuator["plateau_torque"])
        self.corner = float(actuator["corner_speed"])
        self.decay = float(actuator["decay_speed"])
        self.cutoff = float(actuator["cutoff_speed"])
        self.force_limit = float(actuator["force_limit"])
        self.continuous = float(actuator["continuous_torque"])
        self.tau_th = float(actuator["thermal_time_constant"])
        self.dt = float(dt)
        self.thermal = np.zeros(n)

    def reset(self) -> None:
        """Episodes start cold, as in training."""
        self.thermal[:] = 0.0

    def limit(self, vel: np.ndarray) -> np.ndarray:
        w = np.abs(vel)
        lim = self.plateau * np.exp(-np.clip(w - self.corner, 0.0, None) / self.decay)
        return np.where(w >= self.cutoff, 0.0, lim)

    def torque(self, kp, kd, target, pos, vel) -> np.ndarray:
        """kp*(target-pos) - kd*vel, clamped, with heat integrated over one step."""
        tau = kp * (target - pos) + kd * (0.0 - vel)
        force_limit = np.where(self.thermal >= 1.0,
                               min(self.force_limit, self.continuous), self.force_limit)
        lim = np.minimum(self.limit(vel), force_limit)
        tau = np.clip(tau, -lim, lim)
        steady = (tau / self.continuous) ** 2
        self.thermal += (steady - self.thermal) * (self.dt / self.tau_th)
        np.clip(self.thermal, 0.0, None, out=self.thermal)
        return tau


def passive(vel: np.ndarray, damping: np.ndarray, frictionloss: np.ndarray) -> np.ndarray:
    """The MJCF joint class's damping and friction loss, as a torque."""
    return -damping * vel - frictionloss * np.tanh(vel / FRICTIONLOSS_SMOOTHING)
