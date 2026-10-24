"""The crab in an Isaac Sim stage: spawn, state in, torque out.

Shared by the interactive example and `tools/isaac_check.py`, so the check runs
the exact code the example does.
"""

from __future__ import annotations

import json

import numpy as np
from pxr import PhysxSchema

from ..crab_core import paths
from ..crab_core.app_controller import RobotState
from ..crab_core.servo import passive

ROBOT_PRIM = "/World/Crab"
GROUND_FRICTION_DEFAULT = 1.0


def quat_to_matrix(wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def add_ground(world, friction: float | None = None) -> None:
    """A ground plane with MuJoCo's terrain friction, combined so MuJoCo's rule holds.

    The robot's materials carry `max` (grip) and `min` (frictionless); the
    ground's `average` has the lowest priority, so the robot's mode decides
    every pair -- see tools/build_usd.py.
    """
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import GroundPlane

    if friction is None:
        meta = json.loads(paths.MODEL_JSON.read_text())
        friction = float(meta["terrain_friction"])
    material = PhysicsMaterial(
        prim_path="/World/Physics_Materials/ground",
        static_friction=friction, dynamic_friction=friction, restitution=0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
    px.CreateFrictionCombineModeAttr().Set("average")
    px.CreateRestitutionCombineModeAttr().Set("min")
    world.scene.add(GroundPlane(prim_path="/World/GroundPlane", z_position=0.0,
                                physics_material=material, size=50.0))
    # GroundPlane makes two colliders: `geom`, bound to the material above, and an
    # infinite `collisionPlane` bound only to a visual material, which PhysX then gives
    # its default friction (0.5). The feet take the larger of two frictions, so the
    # floor never went below 0.5: at friction 0 the robot still walked at 0.72 m/s
    # (Isaac Sim 5.1, 2026-10-24). Training friction (1.2) was never affected.
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdShade

    stage = get_current_stage()
    shade = UsdShade.Material(material.prim)
    for path in ("/World/GroundPlane/collisionPlane", "/World/GroundPlane/geom"):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
            shade, UsdShade.Tokens.weakerThanDescendants, "physics")


def add_lights(stage) -> None:
    """A dome for soft fill and a sun for shadows, so the robot is lit in any viewport mode.

    Without lights of its own the stage is black in the viewport's "Stage Lights"
    mode (only the "Default" camera light showed the robot). The recorded videos
    use the same two lights.
    """
    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(900.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))


def use_fabric() -> None:
    """Publish physics results to Fabric instead of writing them to USD.

    Writing every link's transform back to USD after each step is what limits
    this robot, not the physics: measured headless (Isaac Sim 5.1, CPU PhysX,
    8 threads, Intel Core Ultra 9 285HX), one 1 ms step costs 2122 us with USD write-back
    and 359 us without -- 0.38x real time against faster than real time. The
    viewport reads Fabric, so nothing visible changes; `isaac-sim.fabric.sh`
    starts Isaac Sim this way, and this does the same for `isaac-sim.sh`.
    """
    import carb
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled("omni.physx.fabric"):
        manager.set_extension_enabled_immediate("omni.physx.fabric", True)
    settings = carb.settings.get_settings()
    settings.set_bool("/physics/updateToUsd", False)
    settings.set_bool("/physics/updateVelocitiesToUsd", False)


async def use_fabric_async(settle_frames: int = 10, max_frames: int = 600) -> None:
    """`use_fabric` from inside a coroutine (the example's LOAD).

    Two things go wrong otherwise, both seen 2026-10-19 on Isaac Sim 5.1:
    enabling the extension *immediately* from a coroutine runs Kit's startup
    work nested inside that task (asyncio "Cannot enter into task"), and
    creating a new stage in the frame the extension starts crashes inside
    omni.graph. So the enable is deferred to Kit's own update, and a few more
    frames pass before LOAD opens its stage.
    """
    import carb
    import omni.kit.app

    kit = omni.kit.app.get_app()
    manager = kit.get_extension_manager()
    if not manager.is_extension_enabled("omni.physx.fabric"):
        manager.set_extension_enabled("omni.physx.fabric", True)
        for _ in range(max_frames):
            await kit.next_update_async()
            if manager.is_extension_enabled("omni.physx.fabric"):
                break
        else:
            raise RuntimeError("omni.physx.fabric did not start")
        for _ in range(settle_frames):
            await kit.next_update_async()
    settings = carb.settings.get_settings()
    settings.set_bool("/physics/updateToUsd", False)
    settings.set_bool("/physics/updateVelocitiesToUsd", False)


def configure_physics(world) -> None:
    """Scene settings: TGS, no stabilisation (it damps a small robot's motion)."""
    ctx = world.get_physics_context()
    ctx.set_solver_type("TGS")
    ctx.enable_stablization(False)
    ctx.enable_gpu_dynamics(False)
    ctx.set_broadphase_type("MBP")


class CrabRobot:
    """The articulation, its joint data, and the conversions the controller needs."""

    def __init__(self, world) -> None:
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        from pxr import Gf, UsdGeom

        self.meta = json.loads(paths.MODEL_JSON.read_text())
        root = self.meta["root_init"]
        self.init_pos = np.array(root[0:3], dtype=np.float64)
        self.init_quat = np.array(root[3:7], dtype=np.float64)
        stage = get_current_stage()
        if not stage.GetPrimAtPath(ROBOT_PRIM).IsValid():
            add_reference_to_stage(str(paths.USD_PATH), ROBOT_PRIM)
        # The links are flat siblings placed relative to this prim, so the whole
        # robot is placed by moving it; moving the root link alone would leave
        # the others behind and PhysX would read that as bent joints.
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_PRIM))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*self.init_pos))

        from isaacsim.core.prims import SingleArticulation

        self.art = SingleArticulation(prim_path=f"{ROBOT_PRIM}/base_link", name="crab")
        world.scene.add(self.art)
        joints = {j["name"]: j for j in self.meta["joints"]}
        self._joints = joints
        self.dof_names: list[str] = []
        self.damping = self.frictionloss = None

    def initialize(self) -> None:
        """After the world has started: read the DOF order and set the stance."""
        self.art.initialize()
        self.dof_names = list(self.art.dof_names)
        missing = set(self.meta["joint_names"]) - set(self.dof_names)
        if missing:
            raise RuntimeError(f"the USD lacks joints {sorted(missing)}")
        self.damping = np.array([self._joints[n]["damping"] for n in self.dof_names])
        self.frictionloss = np.array([self._joints[n]["frictionloss"] for n in self.dof_names])
        self.limits = np.array([self._joints[n]["range"] for n in self.dof_names])
        default = dict(zip(self.meta["joint_names"], self.meta["default_joint_pos"]))
        self.default_q = np.array([default[n] for n in self.dof_names])
        self.art.set_joints_default_state(positions=self.default_q,
                                          velocities=np.zeros(len(self.dof_names)),
                                          efforts=np.zeros(len(self.dof_names)))
        self.art.set_default_state(position=self.init_pos, orientation=self.init_quat)

    def place_at_default(self) -> None:
        self.art.set_world_pose(position=self.init_pos, orientation=self.init_quat)
        self.art.set_linear_velocity(np.zeros(3))
        self.art.set_angular_velocity(np.zeros(3))
        self.art.set_joint_positions(self.default_q)
        self.art.set_joint_velocities(np.zeros(len(self.dof_names)))
        self.art.set_joint_efforts(np.zeros(len(self.dof_names)))

    def state(self) -> RobotState:
        q = np.asarray(self.art.get_joint_positions(), dtype=np.float64)
        qd = np.asarray(self.art.get_joint_velocities(), dtype=np.float64)
        _, quat = self.art.get_world_pose()
        quat = np.asarray(quat, dtype=np.float64)
        w_world = np.asarray(self.art.get_angular_velocity(), dtype=np.float64)
        w_body = quat_to_matrix(quat).T @ w_world  # PhysX reports the world frame
        return RobotState(q, qd, quat, w_body)

    def apply(self, servo_torque: np.ndarray, joint_vel: np.ndarray) -> None:
        """Servo torque plus the passive terms MuJoCo applied inside its integrator."""
        tau = servo_torque + passive(joint_vel, self.damping, self.frictionloss)
        self.art.set_joint_efforts(tau.astype(np.float32))

    def base_pose(self):
        pos, quat = self.art.get_world_pose()
        return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)


def articulation_report(robot: CrabRobot) -> dict:
    """What PhysX actually built, for the check: masses and limits read back."""
    view = robot.art._articulation_view
    names = list(view.body_names)
    masses = np.asarray(view.get_body_masses()).reshape(-1)
    limits = np.asarray(view.get_dof_limits()).reshape(-1, 2)
    return {"body_names": names, "masses": masses, "dof_limits": limits}
