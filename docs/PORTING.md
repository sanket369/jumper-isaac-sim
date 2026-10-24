# How the port works

The policy was trained in MuJoCo (with [mjlab](https://github.com/mujocolab/mjlab), inside
Jumper). Running it in Isaac Sim with no retraining needs the robot, the contacts, the motors
and the controller loop to be what the policy saw in training. This page lists each of them,
what the two engines do differently, and how this repository closes the gap.

## The pipeline

```
Jumper (third_party/jumper)                   this repository
──────────────────────────────                ───────────────────────────────────────────────
jumper.posture task ──► compiled MjModel ──►  tools/export_model.py ──► data/crab_model.{json,npz}
                                                                           │
                                              tools/build_usd.py ◄─────────┘
                                                     │
                                                     ▼
out/bundle_example/jumper                      data/crab.usd ──► Isaac Sim (PhysX)
  controller.so  (observation, modes)                              ▲
  models/*.onnx  (the policies)        ──────► jumper_crab/crab_core/app_controller.py
  manual.en.json (the keys)                    (one physics step at a time, 1 ms)
```

## Faithful to training, by construction

**The model.** The export is the *compiled* MjModel of the `jumper.posture` environment, not
`assets/jumper/jumper.xml`. mjlab edits the spec before compiling (the collision set, the
actuators, contact parameters), so the raw XML is not what the policy was trained on. Every
mass, inertia, frame, joint axis, armature, damping, friction loss and collision hull comes
from the compiled model.

**Welded parts.** MuJoCo bodies without joints are welded to their parent. PhysX has no such
default, so `build_usd.py` adds a fixed joint for each: 18 of them. Without them those parts
fall off the robot.

**Collision.** Each link collides with the convex hull MuJoCo collides it with. The foot hulls
have 667 vertices; PhysX accepts at most 255, so they are reduced by farthest-point sampling,
at most 0.60 mm inward (recorded per geom in `crab_model.json`). MuJoCo's
contype/conaffinity masks become PhysX filtered pairs: 516 of 820 link pairs collide, the
same ones as in training.

**Friction.** Feet and palm pads μ 1.2, everything else frictionless (`condim=1`), ground 1.0.
MuJoCo takes the larger of two geoms' friction; the PhysX combine modes are chosen so every
pair gets that same rule.

**Actuation.** No PhysX joint drives. Each 1 ms physics step applies the training actuator
law: PD under the measured servo torque-speed curve (plateau 1.75 N·m, corner 30.7 rad/s,
cut-off 64 rad/s), with its thermal budget, plus joint damping and friction loss. The policy
observes its own motor torque, so the torque must come from the same curve it was trained
with; a PhysX drive would produce a different one.

**The controller.** The observation (415 values), the mode state machine and the joint
targets come from Jumper's own `controller.so`, the code that runs on the robot. It ticks
every 5th physics step (200 Hz) on simulation time. Rebuilding the observation in Python
would risk silent mistakes such as a wrong joint order; using the robot's controller rules
them out.

**Speed.** Writing every link's transform back to USD after each step costs 2122 µs per 1 ms
step; publishing to Fabric instead costs 359 µs (Isaac Sim 5.1, CPU PhysX, Intel Core Ultra
9 285HX). The example switches to Fabric on LOAD. Headless, the check runs faster than real
time; the full app with rendering runs at about 0.9× real time.

## Not reproduced: soft contact

MuJoCo contacts are soft (bodies overlap slightly and the force follows the overlap); PhysX
contacts are rigid. The robot therefore stands about 4.6 mm higher in Isaac Sim
(0.1072 m vs 0.1026 m). It is left visible on purpose: the policy has to cope with a
different contact model, which is what the comparison is testing.

## Checked, not assumed

`tools/isaac_check.py` reads back what PhysX built before driving anything:

- link masses within 3.5e-9 kg of the training model (2.5430 kg total);
- joint limits within 5e-7 rad;
- every link's position at the stance within 0.0001 mm of MuJoCo's forward kinematics, all
  41 links (a flipped axis or a swapped joint shows up here as millimetres, long before it
  shows up as a strange gait).

Then it drives the scripted keys (`crab_core/scenario.py`: W, S, D, J, 5 s each) through the
controller and compares the speeds with the MuJoCo run of the same script, averaged over the
last 2 s of each key hold. `scripts/sim2sim_check.sh` runs all of it.

## What is checked, and what is not

Only walking (the `locomotion` mode, policy `jumper.posture`) is compared against MuJoCo. The
bundle's other modes (claw, jump, dances and gestures) run through the same controller and
load in the example, but their behaviour in Isaac Sim has not been checked. The claw modes
use a different training task (`jumper.five_foot`) than the one the USD is exported from.

## Harder worlds (`jumper_crab/crab_isaac/conditions.py`)

The same policy in worlds it never trained in, for `run_teleop.sh --ice / --backpack /
--battery`, `tools/isaac_check.py --friction / --payload-kg / --motor-strength` and
`tools/record_survival.py`. One module, so a condition is the same physics everywhere.

- **Friction** goes on the ground's material and on the feet's grip material together. The
  grip pads combine with `max`, so lowering the ground alone would change nothing.
- **Payload** is added to the base link's mass as PhysX holds it (`set_body_masses`), and
  re-applied after every restart, because a stop re-reads the masses from the USD.
- **Motor strength** scales the servo curve's three torque limits together: the plateau, the
  hard limit and the continuous (thermal) torque. The law is the one the policy trained under,
  only weaker; the battery lowers it with simulated time.

**A gotcha worth knowing.** Isaac Sim's `GroundPlane` creates two colliders: a mesh bound to
the physics material you pass, and an infinite `collisionPlane` bound only to a visual
material, which PhysX therefore gives its default friction of 0.5. With the feet taking the
larger of two frictions, the floor could never go below 0.5: at friction 0 the robot still
walked at 0.72 m/s, and every low-friction row read PASS. `robot.add_ground` binds the
material to both colliders. Training friction (1.2) was never affected, nor was anything
compared with MuJoCo. `scripts/sim2sim_check.sh --conditions` keeps it honest: at friction 0
the robot must not get anywhere, and it fails if it does (checked against the unfixed
ground, where it fails as it should).
