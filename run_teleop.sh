#!/usr/bin/env bash
# Drive the crab with the keyboard in Isaac Sim.
#
#   ./run_teleop.sh                 open Isaac Sim with the crab loaded
#   ./run_teleop.sh --browser       only enable the example: open it from
#                                   Window > Examples > Robotics Examples > Policy > Crab > LOAD
#   ./run_teleop.sh -- ARGS         pass ARGS on to isaac-sim.sh
#
# Harder worlds, same policy (they combine; docs/CONTROLS.md):
#   --ice                 floor friction 0.05, real ice (trained on 1.2); --friction F for any value
#   --backpack            +1 kg on the body (+39%), a box on its back; --payload KG for any value
#   --battery             motor power drains 100% -> 40% while you drive; R recharges
#                         --drain-seconds S (default 20), --min-power P (default 0.4)
#
# Click the viewport so it gets the keyboard, then W / S walk, A / D step sideways,
# J / L turn, Space jumps, C toggles the follow camera. All keys: docs/CONTROLS.md.
set -euo pipefail
AUTOLOAD=1
EXTRA=()
FRICTION="" PAYLOAD="" BATTERY="" DRAIN="" MIN_POWER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --browser) AUTOLOAD=0 ;;
    --ice) FRICTION="${FRICTION:-0.05}" ;;
    --friction) FRICTION="${2:?--friction needs a value}"; shift ;;
    --backpack) PAYLOAD="${PAYLOAD:-1.0}" ;;
    --payload) PAYLOAD="${2:?--payload needs kilograms}"; shift ;;
    --battery) BATTERY=1 ;;
    --drain-seconds) DRAIN="${2:?--drain-seconds needs a value}"; BATTERY=1; shift ;;
    --min-power) MIN_POWER="${2:?--min-power needs a value}"; BATTERY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
LOG_TAG=teleop
# shellcheck source=scripts/lib.sh
source "$(dirname "$0")/scripts/lib.sh"
require_isaac
require_built

# The example reads these (jumper_crab/crab_isaac/conditions.py: Conditions.from_env).
export CRAB_FRICTION="$FRICTION" CRAB_PAYLOAD_KG="$PAYLOAD" CRAB_BATTERY="$BATTERY"
[ -n "$DRAIN" ] && export CRAB_DRAIN_S="$DRAIN"
[ -n "$MIN_POWER" ] && export CRAB_MIN_STRENGTH="$MIN_POWER"

args=()
[ $AUTOLOAD -eq 1 ] && args=(--exec "$REPO/tools/teleop.py")
say "starting Isaac Sim $ISAAC_SIM_VERSION (the first start can take a few minutes)"
[ -n "$FRICTION$PAYLOAD$BATTERY" ] && say "conditions: friction=${FRICTION:-trained} \
payload=${PAYLOAD:-0} kg battery=${BATTERY:-off}"
CRAB_DATA="$DATA_DIR" isaac_app "${args[@]}" "${EXTRA[@]}"
