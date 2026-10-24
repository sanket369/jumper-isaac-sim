#!/usr/bin/env bash
# Drive the same scripted keys through the same controller in MuJoCo and Isaac Sim,
# and compare: forward / back / sideways speed, turn rate, height, tilt.
#
#   scripts/sim2sim_check.sh              Isaac Sim (headless) against data/mujoco_reference.json
#   scripts/sim2sim_check.sh --mujoco     re-run the MuJoCo reference first
#   scripts/sim2sim_check.sh --app        also run the interactive example in the full app
#   scripts/sim2sim_check.sh --video DIR  record both drives to DIR (mujoco_nominal.mp4,
#                                         isaac_nominal.mp4), same camera, same timing
#   scripts/sim2sim_check.sh --conditions also prove the conditions reach the physics:
#                                         at floor friction 0 the robot must not get anywhere
#
# Passes when every speed is within 20% of MuJoCo's and the robot stays upright.
# Results: data/isaac_result.json.
set -euo pipefail
MUJOCO=0; APP=0; VIDEO=""; CONDITIONS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --mujoco) MUJOCO=1 ;;
    --app) APP=1 ;;
    --conditions) CONDITIONS=1 ;;
    --video) VIDEO="$(realpath -m "${2:?--video needs a directory}")"; MUJOCO=1; shift ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
LOG_TAG=check
# shellcheck source=lib.sh
source "$(dirname "$0")/lib.sh"
require_isaac
require_built

if [ $MUJOCO -eq 1 ] || [ ! -f "$DATA_DIR/mujoco_reference.json" ]; then
  say "MuJoCo reference"
  args=()
  [ -n "$VIDEO" ] && args=(--video "$VIDEO/mujoco_nominal.mp4")
  # Off-screen rendering on the GPU when there is one to record.
  MUJOCO_GL="${MUJOCO_GL:-egl}" CRAB_DATA="$DATA_DIR" \
    jumper_python "$REPO/tools/mujoco_reference.py" "${args[@]}"
fi

say "Isaac Sim, headless"
args=()
[ -n "$VIDEO" ] && args=(--video "$VIDEO/isaac_nominal.mp4")
CRAB_DATA="$DATA_DIR" isaac_python "$REPO/tools/isaac_check.py" "${args[@]}"

if [ $APP -eq 1 ]; then
  say "the interactive example in the full app, no window"
  CRAB_DATA="$DATA_DIR" isaac_app --no-window --exec "$REPO/tools/example_check.py"
fi
if [ $CONDITIONS -eq 1 ]; then
  # A floor friction that never reaches the contacts fails silently: the robot walks
  # as before and every row reads PASS. That happened -- Isaac's ground plane has a
  # second collider, which had kept PhysX's default friction (robot.add_ground) -- so
  # the proof is physical: with no friction at all, walking must get nowhere.
  say "conditions: floor friction 0, the robot must not get anywhere"
  out="$DATA_DIR/isaac_friction0.json"
  CRAB_DATA="$DATA_DIR" isaac_python "$REPO/tools/isaac_check.py" --friction 0 \
    --label friction-0 --out "$out" > "$DATA_DIR/isaac_friction0.log" 2>&1 || true
  isaac_python -c "
import json, sys
r = json.load(open('$out'))['compared']
moved = {k: v['isaac'] for k, v in r.items() if abs(v['isaac']) > 0.1}  # walking: 0.7 m/s, 3.8 rad/s
print('[check] friction 0:', {k: round(v['isaac'], 3) for k, v in r.items()})
sys.exit(1 if moved else 0)
" || die "at friction 0 the robot still moved: friction is not reaching the feet."
  say "conditions reach the physics (friction 0: no motion)"
fi
say "passed"
