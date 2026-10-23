#!/usr/bin/env bash
# Set up everything the crab needs, build it, and check that it walks.
#
#   ./bootstrap.sh                          use Isaac Sim 5.1 where it already is
#   ./bootstrap.sh --isaac-sim PATH         ... at PATH
#   ./bootstrap.sh --download-isaac-sim     download the standalone build first (~8.8 GB zip)
#   ./bootstrap.sh --rebuild                rebuild the robot even if data/ is current
#   ./bootstrap.sh --skip-check             do not run the headless walking check at the end
#
# Steps, each skipped when it is already done:
#   1. check the machine (never runs sudo: prints what to install and stops)
#   2. clone Jumper at the pinned commit into third_party/jumper
#   3. create Jumper's venv (CPU PyTorch + pinned packages) in third_party/jumper/.venv
#   4. find Isaac Sim 5.1, add onnxruntime to its Python
#   5. export the training model, run the MuJoCo reference, build data/crab.usd
#   6. drive the robot headless in Isaac Sim and compare with MuJoCo
set -euo pipefail

DOWNLOAD=0
DOWNLOAD_DIR="$HOME/isaacsim"
REBUILD=0
CHECK=1
while [ $# -gt 0 ]; do
  case "$1" in
    --isaac-sim) ISAAC_SIM_PATH="$(realpath -m "${2:?--isaac-sim needs a path}")"; shift ;;
    --download-isaac-sim) DOWNLOAD=1 ;;
    --download-dir) DOWNLOAD_DIR="$(realpath -m "${2:?--download-dir needs a path}")"; shift ;;
    --rebuild) REBUILD=1 ;;
    --skip-check) CHECK=0 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

LOG_TAG=bootstrap
# shellcheck source=scripts/lib.sh
source "$(dirname "$0")/scripts/lib.sh"

# ── 1. the machine ──────────────────────────────────────────────────────────
step_host() {
  say "1/6 checking the machine"
  [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ] || die "Isaac Sim's standalone build is Linux x86_64 only."
  local missing=()
  command -v git >/dev/null || missing+=(git)
  command -v curl >/dev/null || missing+=(curl)
  command -v unzip >/dev/null || missing+=(unzip)
  if ! command -v "$JUMPER_PYTHON" >/dev/null; then
    missing+=("$JUMPER_PYTHON" "$JUMPER_PYTHON-venv")
  elif ! "$JUMPER_PYTHON" -c "import ensurepip, venv" 2>/dev/null; then
    missing+=("$JUMPER_PYTHON-venv")
  fi
  if [ ${#missing[@]} -gt 0 ]; then
    warn "missing: ${missing[*]}"
    if [[ " ${missing[*]} " == *" $JUMPER_PYTHON "* ]] && ! grep -q 'VERSION_ID="24' /etc/os-release; then
      warn "Ubuntu 24.04 ships $JUMPER_PYTHON. On 22.04 it comes from the deadsnakes PPA:"
      warn "  sudo add-apt-repository ppa:deadsnakes/ppa"
    fi
    die "install them, then run this again:  sudo apt install -y ${missing[*]}"
  fi
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    say "    GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  else
    die "nvidia-smi does not work: Isaac Sim needs an NVIDIA GPU with its driver loaded."
  fi
}

# ── 2. Jumper ───────────────────────────────────────────────────────────────
step_jumper() {
  say "2/6 Jumper at ${JUMPER_COMMIT:0:7}"
  if [ ! -d "$JUMPER_DIR/.git" ]; then
    mkdir -p "$JUMPER_DIR"
    git -C "$JUMPER_DIR" init -q
    git -C "$JUMPER_DIR" remote add origin "$JUMPER_REPO"
  fi
  if [ "$(git -C "$JUMPER_DIR" rev-parse -q --verify HEAD 2>/dev/null)" != "$JUMPER_COMMIT" ]; then
    [ -z "$(git -C "$JUMPER_DIR" status --porcelain 2>/dev/null)" ] \
      || die "$JUMPER_DIR has local changes; commit or remove them so it can move to $JUMPER_COMMIT."
    git -C "$JUMPER_DIR" fetch -q --depth 1 origin "$JUMPER_COMMIT"
    git -C "$JUMPER_DIR" checkout -q --detach "$JUMPER_COMMIT"
  fi
  [ -f "$JUMPER_DIR/out/bundle_example/jumper/controller.toml" ] \
    || die "the Jumper checkout has no out/bundle_example; is $JUMPER_COMMIT the right commit?"
}

# ── 3. Jumper's venv ────────────────────────────────────────────────────────
step_jumper_venv() {
  local stamp want
  stamp="$JUMPER_VENV/.jumper-isaac-sim"
  want="$(cat "$REPO/config/versions.env" "$REPO/config/jumper-constraints.txt" | sha256sum | cut -c1-16)"
  if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ]; then
    say "3/6 Jumper's venv is current"
    return
  fi
  say "3/6 creating Jumper's venv (CPU PyTorch; a few minutes)"
  rm -rf "$JUMPER_VENV"
  "$JUMPER_PYTHON" -m venv "$JUMPER_VENV"
  local pip=("$JUMPER_VENV/bin/python" -m pip install --quiet --disable-pip-version-check)
  "${pip[@]}" --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
  # The editable install is what makes Jumper's rl/ packages (mjlab, mjrl, rsl_rl) importable.
  "${pip[@]}" --constraint "$REPO/config/jumper-constraints.txt" -e "$JUMPER_DIR"
  jumper_python -c "import mjlab, mujoco, onnxruntime, tasks" \
    || die "Jumper's venv was built but does not import; see the pip output above."
  echo "$want" > "$stamp"
}

# ── 4. Isaac Sim ────────────────────────────────────────────────────────────
download_isaac() {
  local zip="$DOWNLOAD_DIR/$ISAAC_SIM_ZIP"
  local target="$DOWNLOAD_DIR/${ISAAC_SIM_ZIP%.zip}"
  say "    downloading Isaac Sim $ISAAC_SIM_VERSION to $DOWNLOAD_DIR"
  say "    (downloading it means accepting NVIDIA's Isaac Sim licence:"
  say "     https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html)"
  mkdir -p "$DOWNLOAD_DIR"
  local free_gb
  free_gb=$(df -BG --output=avail "$DOWNLOAD_DIR" | tail -1 | tr -dc 0-9)
  [ "$free_gb" -ge 30 ] || die "$DOWNLOAD_DIR has ${free_gb} GB free; the zip and the install need about 30 GB."
  curl -fL -C - --retry 3 -o "$zip" "$ISAAC_SIM_URL"
  say "    verifying the download"
  echo "$ISAAC_SIM_SHA256  $zip" | sha256sum -c --quiet - \
    || die "$zip does not match its checksum; delete it and run again."
  say "    unpacking"
  mkdir -p "$target"
  unzip -q -o "$zip" -d "$target"
  ISAAC_SIM_PATH="$target"
}

step_isaac() {
  say "4/6 Isaac Sim $ISAAC_SIM_VERSION"
  if ! isaac_ok; then
    if [ $DOWNLOAD -eq 1 ]; then
      download_isaac
    else
      die "Isaac Sim $ISAAC_SIM_VERSION is not at $ISAAC_SIM_PATH.
    Point to an install:   ./bootstrap.sh --isaac-sim /path/to/isaac-sim-standalone-5.1.0-linux-x86_64
    or download it:        ./bootstrap.sh --download-isaac-sim   (~8.8 GB, to $DOWNLOAD_DIR)"
    fi
  fi
  isaac_ok || die "$ISAAC_SIM_PATH is not an Isaac Sim $ISAAC_SIM_VERSION install."
  say "    using $ISAAC_SIM_PATH"
  echo "ISAAC_SIM_PATH=\"$ISAAC_SIM_PATH\"" > "$LOCAL_ENV"
  if ! isaac_python -c "import onnxruntime" >/dev/null 2>&1; then
    say "    adding onnxruntime $ONNXRUNTIME_VERSION to Isaac Sim's Python"
    isaac_python -m pip install --quiet --disable-pip-version-check "onnxruntime==$ONNXRUNTIME_VERSION"
  fi
}

# ── 5. the robot ────────────────────────────────────────────────────────────
step_build() {
  local stamp want
  stamp="$DATA_DIR/.build-stamp"
  want="$( (echo "$JUMPER_COMMIT $JUMPER_TASK"; cat "$REPO"/tools/export_model.py \
            "$REPO"/tools/build_usd.py "$REPO"/tools/mujoco_reference.py) | sha256sum | cut -c1-16)"
  if [ $REBUILD -eq 0 ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ] \
     && [ -f "$DATA_DIR/crab.usd" ]; then
    say "5/6 the robot in $DATA_DIR is current"
    return
  fi
  say "5/6 building the robot"
  mkdir -p "$DATA_DIR"
  say "    exporting the compiled training model ($JUMPER_TASK)"
  jumper_python "$REPO/tools/export_model.py" --task "$JUMPER_TASK" --out "$DATA_DIR"
  say "    driving it in MuJoCo for the reference"
  CRAB_DATA="$DATA_DIR" jumper_python "$REPO/tools/mujoco_reference.py"
  say "    building crab.usd (the first Isaac Sim start can take a few minutes)"
  isaac_python "$REPO/tools/build_usd.py" --data "$DATA_DIR" --out "$DATA_DIR/crab.usd"
  echo "$want" > "$stamp"
}

# ── 6. the check ────────────────────────────────────────────────────────────
step_check() {
  if [ $CHECK -eq 0 ]; then
    say "6/6 check skipped (--skip-check)"
    return
  fi
  say "6/6 driving the crab headless in Isaac Sim and comparing with MuJoCo (a few minutes)"
  "$REPO/scripts/sim2sim_check.sh" || die "the check failed; the lines above say which part."
}

step_host
step_jumper
step_jumper_venv
step_isaac
step_build
step_check
say "done. Drive it:  ./run_teleop.sh"
