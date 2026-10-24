"""Harder worlds for the same policy: a slippery floor, a heavier body, a draining battery.

Shared by the interactive example (`run_teleop.sh --ice / --backpack / --battery`),
`tools/isaac_check.py` (`--friction / --payload-kg / --motor-strength`) and
`tools/record_survival.py`, so a condition is the same physics wherever it is used.
Every condition changes the physics for real; the ice floor, the box and the battery
readout only make that visible.

* Friction: the ground's material and the feet's grip material both, because the grip
  pads combine with `max` -- lowering the ground alone changes nothing. (The ground's
  hidden collision plane also needs the material: see `robot.add_ground`.)
* Payload: mass added to the base link, as PhysX sees it (read back in the check).
* Motor strength: the servo curve's three torque limits scaled together -- the plateau,
  the hard limit and the continuous (thermal) torque -- the same law the policy trained
  under, only weaker. The battery lowers it over time while you drive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .robot import ROBOT_PRIM

ICE_FRICTION = 0.05  # real ice is about 0.02-0.1; training is 1.2
BACKPACK_KG = 1.0  # the robot is 2.54 kg, so +39%
DRAIN_S = 20.0  # 100% -> MIN_STRENGTH over this long
MIN_STRENGTH = 0.4  # where the stress test found the policy's margin ends


@dataclass
class Conditions:
    friction: float | None = None  # None: as trained
    payload_kg: float = 0.0
    battery: bool = False
    drain_s: float = DRAIN_S
    min_strength: float = MIN_STRENGTH

    @classmethod
    def from_env(cls) -> Conditions:
        """What run_teleop.sh asked for (it exports CRAB_* before starting Isaac Sim)."""
        env = os.environ
        friction = env.get("CRAB_FRICTION")
        return cls(friction=float(friction) if friction else None,
                   payload_kg=float(env.get("CRAB_PAYLOAD_KG", "0") or 0),
                   battery=env.get("CRAB_BATTERY", "") == "1",
                   drain_s=float(env.get("CRAB_DRAIN_S", DRAIN_S)),
                   min_strength=float(env.get("CRAB_MIN_STRENGTH", MIN_STRENGTH)))

    def active(self) -> bool:
        return self.friction is not None or self.payload_kg > 0 or self.battery

    def describe(self) -> str:
        parts = []
        if self.friction is not None:
            parts.append(f"floor friction {self.friction:g} (trained on 1.2)")
        if self.payload_kg:
            parts.append(f"+{self.payload_kg:g} kg on the body")
        if self.battery:
            parts.append(f"battery draining to {self.min_strength * 100:.0f}% "
                         f"over {self.drain_s:g} s (R recharges)")
        return "; ".join(parts) if parts else "as trained"


# ── friction ──────────────────────────────────────────────────────────────────

def grip_materials(stage) -> list:
    """The robot's materials combined with `max`: the foot tips and palm pads."""
    found = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{ROBOT_PRIM}/materials")):
        if prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
            mode = PhysxSchema.PhysxMaterialAPI(prim).GetFrictionCombineModeAttr().Get()
            if mode == "max":
                found.append(prim)
    return found


def set_grip_friction(stage, friction: float) -> int:
    """The feet's grip; call before the simulation starts, with add_ground(friction=...)."""
    grips = grip_materials(stage)
    if not grips:
        raise RuntimeError("no grip material found to set the friction on")
    for prim in grips:
        api = UsdPhysics.MaterialAPI(prim)
        api.GetStaticFrictionAttr().Set(friction)
        api.GetDynamicFrictionAttr().Set(friction)
    return len(grips)


# ── payload ───────────────────────────────────────────────────────────────────

def add_payload(robot, kg: float) -> tuple[float, float]:
    """Mass added to the base link as PhysX holds it. After robot.initialize(), and again
    after every restart (a stop re-reads the USD). Returns the total mass before, after."""
    view = robot.art._articulation_view
    names = list(view.body_names)
    masses = np.asarray(view.get_body_masses()).reshape(1, -1).copy()
    before = float(masses.sum())
    masses[0, names.index("base_link")] += kg
    view.set_body_masses(masses)
    return before, float(np.asarray(view.get_body_masses()).sum())


# ── motor strength and the battery ────────────────────────────────────────────

class MotorStrength:
    """Scales a ServoCurve's torque limits together, from their trained values."""

    def __init__(self, servo) -> None:
        self.servo = servo
        self.full = (servo.plateau, servo.force_limit, servo.continuous)
        self.k = 1.0

    def set(self, k: float) -> None:
        self.k = float(k)
        self.servo.plateau, self.servo.force_limit, self.servo.continuous = (
            v * self.k for v in self.full)


class Battery:
    """Motor strength falling linearly from 100% to `minimum` over `drain_s` of
    simulated time, then staying there until recharged."""

    def __init__(self, motor: MotorStrength, drain_s: float = DRAIN_S,
                 minimum: float = MIN_STRENGTH) -> None:
        self.motor, self.drain_s, self.minimum = motor, drain_s, minimum
        self.t0 = 0.0

    def recharge(self, now: float) -> None:
        self.t0 = now
        self.motor.set(1.0)

    def update(self, now: float) -> float:
        frac = min(max(now - self.t0, 0.0) / self.drain_s, 1.0)
        self.motor.set(1.0 - (1.0 - self.minimum) * frac)
        return self.motor.k


# ── what you see: the condition, made visible (no physics of its own) ─────────

def _preview_material(stage, path: str, color, roughness: float, specular: float):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(specular, specular, specular))
    sh.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def add_floor(stage, colors, roughness: float, specular: float = 0.04,
              tile: float = 0.25, half: float = 25.0, root: str = "/World/ConditionFloor") -> None:
    """A checkered floor just above the ground, visual only (no collision), in two
    materials so the tiles passing under the robot show it moving. Specular is the
    reflectance at normal incidence: ~0.04 for plastic, wood or concrete (0.5 mirrored
    the dome light and washed the floor out white)."""
    n = int(round(2 * half / tile))
    xs = np.linspace(-half, half, n + 1)
    for parity, color in enumerate(colors):
        quads = [(i, j) for i in range(n) for j in range(n) if (i + j) % 2 == parity]
        pts = np.zeros((4 * len(quads), 3), np.float32)
        for q, (i, j) in enumerate(quads):
            x0, x1, y0, y1 = xs[i], xs[i + 1], xs[j], xs[j + 1]
            pts[4 * q:4 * q + 4] = [(x0, y0, 5e-4), (x1, y0, 5e-4), (x1, y1, 5e-4), (x0, y1, 5e-4)]
        mesh = UsdGeom.Mesh.Define(stage, f"{root}/tiles{parity}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * len(quads)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(list(range(4 * len(quads)))))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-half, -half, 0),
                                             Gf.Vec3f(half, half, 1e-3)]))
        mat = _preview_material(stage, f"{root}/looks/tiles{parity}", color, roughness, specular)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def add_ice_floor(stage) -> None:
    add_floor(stage, [(0.62, 0.82, 1.00), (0.30, 0.55, 0.85)], roughness=0.04, specular=0.08)


def add_backpack(stage) -> None:
    """A cardboard box strapped to the shell: children of the base link, so it rides with
    it; no collision and no mass of its own (add_payload adds the real load)."""
    base = stage.GetPrimAtPath(f"{ROBOT_PRIM}/base_link")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_,
                                                       UsdGeom.Tokens.render])
    rng = cache.ComputeUntransformedBound(base).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    size = (0.10, 0.08, 0.065)
    cx, cy, cz = (lo[0] + hi[0]) / 2 - 0.01, (lo[1] + hi[1]) / 2, hi[2] + size[2] / 2 - 0.004
    looks = {"box": ((0.62, 0.45, 0.26), 0.85, 0.04), "strap": ((0.12, 0.12, 0.13), 0.6, 0.04)}
    parts = [("Backpack", (cx, cy, cz), size, "box")]
    parts += [(f"Backpack_strap{k}", (cx, cy + dy, cz), (size[0] + 0.004, 0.012, size[2] + 0.004),
               "strap") for k, dy in enumerate((-0.022, 0.022))]
    for name, pos, scale, look in parts:
        cube = UsdGeom.Cube.Define(stage, f"{ROBOT_PRIM}/base_link/{name}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        color, rough, spec = looks[look]
        mat = _preview_material(stage, f"/World/ConditionLooks/{look}", color, rough, spec)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
