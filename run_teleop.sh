#!/usr/bin/env bash
# Drive the crab with the keyboard in Isaac Sim.
#
#   ./run_teleop.sh                 open Isaac Sim with the crab loaded
#   ./run_teleop.sh --browser       only enable the example: open it from
#                                   Window > Examples > Robotics Examples > Policy > Crab > LOAD
#   ./run_teleop.sh -- ARGS         pass ARGS on to isaac-sim.sh
#
# Click the viewport so it gets the keyboard, then W / S walk, A / D step sideways,
# J / L turn, Space jumps, C toggles the follow camera. All keys: docs/CONTROLS.md.
set -euo pipefail
AUTOLOAD=1
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --browser) AUTOLOAD=0 ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
LOG_TAG=teleop
# shellcheck source=scripts/lib.sh
source "$(dirname "$0")/scripts/lib.sh"
require_isaac
require_built

args=()
[ $AUTOLOAD -eq 1 ] && args=(--exec "$REPO/tools/teleop.py")
say "starting Isaac Sim $ISAAC_SIM_VERSION (the first start can take a few minutes)"
CRAB_DATA="$DATA_DIR" isaac_app "${args[@]}" "${EXTRA[@]}"
