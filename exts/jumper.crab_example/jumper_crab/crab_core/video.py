"""One camera for both engines, so their clips line up shot for shot.

A follow camera looking at the base from the front-left, three-quarter view.
MuJoCo describes it as lookat/distance/azimuth/elevation, Isaac Sim as an eye
point and a target; both are derived from the same offset here, so the two
renders frame the robot identically. Frames are taken on simulation time at
FPS, so a clip from either engine has the same length and timing as the
scripted drive.
"""

from __future__ import annotations

import math

import numpy as np

FPS = 30
SIZE = 960  # square: two side by side make a 2:1 strip, a grid stays tidy
FOVY_DEG = 45.0  # MuJoCo's default free-camera field of view (vis.global.fovy)
#: Camera position relative to the point it looks at, in metres (world frame).
EYE_OFFSET = np.array([0.56, -0.56, 0.31])
#: Height of the point looked at: the base at its standing height, about.
TARGET_Z = 0.06


def target(base_pos) -> np.ndarray:
    """Follow the base across the floor; ignore its bounce."""
    return np.array([float(base_pos[0]), float(base_pos[1]), TARGET_Z])


def mujoco_angles() -> tuple[float, float, float]:
    """(distance, azimuth deg, elevation deg) for mjvCamera, same view as EYE_OFFSET."""
    d = float(np.linalg.norm(EYE_OFFSET))
    look = -EYE_OFFSET / d  # from the camera towards the target
    azimuth = math.degrees(math.atan2(look[1], look[0]))
    elevation = math.degrees(math.asin(look[2]))
    return d, azimuth, elevation


def focal_length_mm(aperture_mm: float = 20.955) -> float:
    """USD camera focal length giving FOVY_DEG over a square aperture."""
    return aperture_mm / (2.0 * math.tan(math.radians(FOVY_DEG) / 2.0))


class FrameClock:
    """Says when a frame is due, on simulation time."""

    def __init__(self, fps: int = FPS) -> None:
        self.period = 1.0 / fps
        self.next_t = 0.0

    def due(self, t: float) -> bool:
        if t + 1e-9 >= self.next_t:
            self.next_t += self.period
            return True
        return False
