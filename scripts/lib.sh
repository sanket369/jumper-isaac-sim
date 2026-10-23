# Shared by bootstrap.sh, run_teleop.sh and scripts/*.sh. Source it; do not run it.
#
# Defines REPO, loads the pinned versions, and resolves where Isaac Sim is:
#   1. --isaac-sim PATH (the caller sets ISAAC_SIM_PATH before sourcing)
#   2. $ISAAC_SIM_PATH from the environment
#   3. config/local.env, written by bootstrap.sh once it has found one
#   4. the standalone build's default location, ~/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../config/versions.env
source "$REPO/config/versions.env"
LOCAL_ENV="$REPO/config/local.env"
if [ -z "${ISAAC_SIM_PATH:-}" ] && [ -f "$LOCAL_ENV" ]; then
  # shellcheck source=/dev/null
  source "$LOCAL_ENV"
fi
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-$ISAAC_SIM_DEFAULT_PATH}"

JUMPER_DIR="$REPO/third_party/jumper"
JUMPER_VENV="$JUMPER_DIR/.venv"
DATA_DIR="${CRAB_DATA:-$REPO/data}"
EXT_FOLDER="$REPO/exts"
EXT_NAME="jumper.crab_example"

say()  { printf '\033[1;32m[%s]\033[0m %s\n' "${LOG_TAG:-crab}" "$*"; }
warn() { printf '\033[1;33m[%s] %s\033[0m\n' "${LOG_TAG:-crab}" "$*" >&2; }
die()  { printf '\033[1;31m[%s] %s\033[0m\n' "${LOG_TAG:-crab}" "$*" >&2; exit 1; }

isaac_ok() {
  [ -x "$ISAAC_SIM_PATH/python.sh" ] && [ -x "$ISAAC_SIM_PATH/isaac-sim.sh" ] \
    && grep -q "^$ISAAC_SIM_VERSION" "$ISAAC_SIM_PATH/VERSION" 2>/dev/null
}

require_isaac() {
  isaac_ok || die "Isaac Sim $ISAAC_SIM_VERSION not found at $ISAAC_SIM_PATH. Run ./bootstrap.sh \
(--isaac-sim PATH for an existing install, --download-isaac-sim to fetch it)."
}

require_built() {
  [ -f "$DATA_DIR/crab.usd" ] && [ -f "$DATA_DIR/crab_model.json" ] \
    || die "No robot in $DATA_DIR yet. Run ./bootstrap.sh first."
  [ -d "$JUMPER_DIR/out/bundle_example/jumper" ] \
    || die "No Jumper checkout at $JUMPER_DIR. Run ./bootstrap.sh first."
}

# Run a tool in Jumper's venv, from the Jumper checkout (its tasks import from there).
jumper_python() {
  (cd "$JUMPER_DIR" && "$JUMPER_VENV/bin/python" "$@")
}

# Isaac Sim brings its own Python, and an activated environment breaks it: with a
# venv active (VIRTUAL_ENV set, its bin/ first on PATH) the full app starts without
# its bundled packages -- psutil, starlette, and the onnxruntime added to it -- so the
# example cannot load (Isaac Sim 5.1, 2026-10-23). Conda makes python.sh warn. Both
# are cleared for Isaac's commands, whatever the caller's shell has active.
isaac_env() {
  local path="$PATH" dir
  for dir in "${VIRTUAL_ENV:-}" "${CONDA_PREFIX:-}"; do
    [ -n "$dir" ] && path="$(printf '%s' "$path" | tr ':' '\n' | grep -vx "$dir/bin" | paste -sd:)"
  done
  env -u VIRTUAL_ENV -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u CONDA_PYTHON_EXE \
    PATH="$path" "$@"
}

# Run a tool with Isaac Sim's Python.
isaac_python() {
  isaac_env "$ISAAC_SIM_PATH/python.sh" "$@"
}

# The full Isaac Sim app with this repository's extension available and enabled.
isaac_app() {
  isaac_env "$ISAAC_SIM_PATH/isaac-sim.sh" --ext-folder "$EXT_FOLDER" --enable "$EXT_NAME" "$@"
}
