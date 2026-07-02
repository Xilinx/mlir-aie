#!/usr/bin/env bash
##===- utils/env_install.sh ----------------------------------*- Script -*-===##
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Create a Python virtual environment and install everything needed to build and
# run IRON/MLIR-AIE designs. Source it so the environment stays active in the
# caller:
#
#   source utils/env_install.sh [--dev] [--extras] [--latest] [--version <ver>] [--no-pre-commit] [venv-dir]
#
# venv-dir defaults to "ironenv". Any name may be given.
#
# This can be invoked in two ways: a "runtime" environment for users, or a 
# "from-source" developer environment (--dev flag).
#
# For the "runtime" environment, we download the mlir_aie wheel along with the
# other prerequisites. The wheel is matched to this checkout: if HEAD is on a 
# release tag (e.g. v1.3.4) the wheel for that release is installed; else if 
# HEAD is exactly the commit a latest-wheels build was made from, those rolling
# wheels are installed; otherwise the script errors (no published wheel matches
# this commit).
#
# For the "from-source" build environment: it installs the hash-pinned dev 
# tooling, vendors eudsl, sets up the pre-commit/pre-push hooks, and skips the 
# mlir_aie wheel (you build mlir_aie yourself via 
# utils/build-mlir-aie-from-wheels.sh).
#
# Re-running against an existing venv reuses it and upgrades the installed
# packages in place, so this doubles as an update command.
#
# --extras additionally installs the optional ML and notebook requirements
# (python/requirements_ml.txt and python/requirements_notebook.txt).
#
##===----------------------------------------------------------------------===##

# --- Parse arguments ---------------------------------------------------------
DEV=0
EXTRAS=0
LATEST=0
NO_PRE_COMMIT=0
VERSION=""
VENV_DIR=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dev) DEV=1 ;;
    --extras) EXTRAS=1 ;;
    --latest) LATEST=1 ;;
    --no-pre-commit) NO_PRE_COMMIT=1 ;;
    --version)
      if [ -z "${2:-}" ]; then
        echo "ERROR: --version requires a value, e.g. --version v1.3.4" >&2; return 1
      fi
      VERSION="$2"; shift ;;
    --version=*) VERSION="${1#*=}" ;;
    -*) echo "ERROR: unknown option: $1" >&2; return 1 ;;
    *)
      if [ -z "$VENV_DIR" ]; then
        VENV_DIR="$1"
      else
        echo "ERROR: unexpected argument: $1" >&2; return 1
      fi
      ;;
  esac
  shift
done
VENV_DIR="${VENV_DIR:-ironenv}"

if [ -n "$VERSION" ] && [ "$LATEST" = 1 ]; then
  echo "ERROR: pass only one of --version and --latest." >&2; return 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Resolve and validate the interpreter (python3.12 hard requirement) ------
PY="${PYTHON:-python3.12}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: '$PY' not found. IRON requires python3.12." >&2
  return 1
fi
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
if [ "$PYVER" != "3.12" ]; then
  echo "ERROR: python3.12 is required, but '$PY' reports version '$PYVER'." >&2
  return 1
fi
echo "Using $("$PY" --version) at $(command -v "$PY")"

# --- Refuse to shadow a different, already-active virtual environment ---------
if [ -n "${VIRTUAL_ENV:-}" ]; then
  active_env="$(cd "$VIRTUAL_ENV" && pwd)"
  target_env="$(cd "$(dirname "$VENV_DIR")" 2>/dev/null && pwd)/$(basename "$VENV_DIR")"
  if [ "$active_env" != "$target_env" ]; then
    echo "ERROR: a different virtual environment is already active:" >&2
    echo "         active: $active_env" >&2
    echo "         target: $target_env" >&2
    echo "       Run 'deactivate' first, or target the active environment." >&2
    return 1
  fi
fi

# --- Create/activate the venv and install common requirements ----------------
"$PY" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip
if [ "$EXTRAS" = 1 ]; then
  python3 -m pip install -U -r "$ROOT/python/requirements_ml.txt"
  python3 -m pip install -U -r "$ROOT/python/requirements_notebook.txt"
fi

# --- Peano (llvm-aie) --------------------------------------------------------
# Pinned via utils/peano-requirements.txt (bumped by the update-peano workflow).
python3 -m pip install -U -r "$ROOT/utils/peano-requirements.txt"
export PEANO_INSTALL_DIR="$(pip show llvm-aie | awk '/^Location:/ {print $2 "/llvm-aie"}')"


# --- Developer setup ---------------------------------------------------------
if [ "$DEV" = 1 ]; then
  python3 -m pip install --require-hashes -r "$ROOT/python/requirements_dev.lock"
  # Vendored eudsl is built into from-source mlir_aie builds, so install the
  # rest of the core requirements without fetching eudsl from the index.
  python3 "$ROOT/utils/mlir_aie_wheels/vendor_eudsl.py" \
      --requirements "$ROOT/python/requirements.txt" \
      --install-non-eudsl
  # Contributor hooks defined in .pre-commit-config.yaml (pre-push runs
  # clang-format/black to catch formatting issues before CI).
  if [ "$NO_PRE_COMMIT" = 0 ]; then
    python3 -m pip install pre-commit
    pre-commit install --hook-type pre-commit --hook-type pre-push
  fi
fi

# --- MLIR-AIE (prebuilt wheel, non-dev env only) -----------------------------
if [ "$DEV" = 0 ]; then
  #  wheel_index - where to pull the wheel from (a pip -f location).
  #  wheel_spec  - what to pull: "mlir_aie" for a release tag/channel, or
  #                "mlir_aie==<version+commit>" to pin the wheel built from a
  #                specific commit.
  base_url="https://github.com/Xilinx/mlir-aie/releases/expanded_assets"
  wheel_spec="mlir_aie"
  repo_tag="$(git -C "$ROOT" describe --exact-match --tags HEAD 2>/dev/null)"

  if [ -n "${MLIR_AIE_WHEEL_DIR:-}" ]; then
    wheel_index="$MLIR_AIE_WHEEL_DIR"
  elif [ -n "$VERSION" ]; then
    wheel_index="${base_url}/v${VERSION#v}/"        # accept "1.3.4" or "v1.3.4"
  elif [ "$LATEST" = 1 ]; then
    wheel_index="${base_url}/latest-wheels-4/"      # newest rolling build
  elif [[ "$repo_tag" == v[0-9]* ]]; then
    wheel_index="${base_url}/${repo_tag}/"          # checked-out release tag
  else
    # check if there is a wheel associated with this non-release checkout of the repo
    head_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)"
    if [ -z "$head_sha" ]; then
      echo "ERROR: '$ROOT' is not a git checkout, so the mlir_aie wheel cannot be" >&2
      echo "       matched to a commit. Check out a release tag, build from source" >&2
      echo "       with --dev, or pass --latest to install the latest wheels." >&2
      return 1
    fi
    match="$(python3 "$ROOT/utils/find_mlir_aie_wheel.py" "$head_sha")"
    if [ -z "$match" ]; then
      echo "ERROR: the current commit ($head_sha) has no associated mlir_aie release" >&2
      echo "       and no latest-wheels build matches it, so no published wheel matches" >&2
      echo "       this checkout. Check out a release tag following the instructions in" >&2
      echo "       the README, build from source with --dev, or pass --latest to install" >&2
      echo "       the latest wheels (at your own risk!)." >&2
      return 1
    fi
    wheel_index="${base_url}/${match%% *}/"
    wheel_spec="mlir_aie==${match#* }"
  fi

  echo "Installing ${wheel_spec} from ${wheel_index}"
  python3 -m pip install -U "$wheel_spec" -f "$wheel_index"
  export MLIR_AIE_INSTALL_DIR="$(pip show mlir_aie | awk '/^Location:/ {print $2 "/mlir_aie"}')"

  # Write a .pth so any Python in the venv (including VS Code, which may not
  # inherit PYTHONPATH) can find the installed mlir_aie.
  venv_site_packages="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  echo "${MLIR_AIE_INSTALL_DIR}/python" > "$venv_site_packages/mlir-aie.pth"

  # Register an ipykernel for notebooks. ipykernel is only present with the
  # notebook requirements, so this needs --extras.
  if [ "$EXTRAS" = 1 ]; then
    python3 -m ipykernel install --user --name "$(basename "$VENV_DIR")"
  fi
fi

echo ""
echo "Environment '$VENV_DIR' is ready and active."
if [ "$DEV" = 1 ]; then
  echo "Next: build mlir_aie from source, e.g."
  echo "      ./utils/build-mlir-aie-from-wheels.sh"
  echo "      then configure the shell with: source utils/env_setup.sh"
else
  # mlir_aie and Peano are installed, so configure the shell now.
  source "$ROOT/utils/env_setup.sh"
fi
