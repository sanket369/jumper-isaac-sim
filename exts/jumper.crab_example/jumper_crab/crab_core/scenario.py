"""One scripted drive, run identically on MuJoCo and on Isaac Sim, and its metrics.

The keys are pressed through the controller exactly as a person's would be, so
the comparison covers the whole path: key -> operator ramp -> command ->
observation -> policy -> targets -> servo -> physics.
"""

from __future__ import annotations

import math

import numpy as np

#: (time [s], controller key name, down). A held key ramps its command to full
#: over 2 s (the task's `full_after_s`), so each hold lasts 5 s and is measured
#: over its last 2.
SCRIPT = [
    (4.0, "key_w", True), (9.0, "key_w", False),
    (11.0, "key_s", True), (16.0, "key_s", False),
    (18.0, "key_d", True), (23.0, "key_d", False),
    (25.0, "key_j", True), (30.0, "key_j", False),
]
DURATION = 32.0

#: (name, t0, t1, what): the windows the metrics average over.
WINDOWS = [
    ("stand", 2.5, 4.0, "standing after the warm start"),
    ("forward", 7.0, 9.0, "W held: walk forward"),
    ("back", 14.0, 16.0, "S held: walk back"),
    ("right", 21.0, 23.0, "D held: step right"),
    ("turn", 28.0, 30.0, "J held: turn left"),
]


class Recorder:
    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def add(self, t: float, pos, quat_wxyz) -> None:
        self.rows.append((t, *map(float, pos), *map(float, quat_wxyz)))

    def metrics(self) -> dict:
        a = np.asarray(self.rows)
        t, p, q = a[:, 0], a[:, 1:4], a[:, 4:8]
        w, x, y, z = q.T
        yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        # Tilt: angle between the base's z axis and the world's.
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
        out = {"max_tilt_deg": float(tilt.max()), "final_height": float(p[-1, 2])}
        for name, t0, t1, _ in WINDOWS:
            i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1) - 1
            dt = t[i1] - t[i0]
            dp = p[i1] - p[i0]
            ym = 0.5 * (yaw[i0] + yaw[i1])
            # Mean velocity in the heading frame (forward x, left y).
            fwd = (math.cos(ym) * dp[0] + math.sin(ym) * dp[1]) / dt
            left = (-math.sin(ym) * dp[0] + math.cos(ym) * dp[1]) / dt
            out[name] = {
                "vx": round(float(fwd), 4),
                "vy": round(float(left), 4),
                "yaw_rate": round(float((yaw[i1] - yaw[i0]) / dt), 4),
                "height": round(float(p[i0:i1, 2].mean()), 4),
                "max_tilt_deg": round(float(tilt[i0:i1].max()), 2),
            }
        return out


def print_metrics(label: str, m: dict) -> None:
    print(f"[{label}] max tilt {m['max_tilt_deg']:.1f} deg, final height {m['final_height']:.4f} m")
    for name, _, _, what in WINDOWS:
        r = m[name]
        print(f"[{label}]   {name:8s} vx {r['vx']:+.3f}  vy {r['vy']:+.3f}  "
              f"yaw {r['yaw_rate']:+.3f} rad/s"
              f"  h {r['height']:.4f}  tilt<= {r['max_tilt_deg']:.1f}   ({what})")
