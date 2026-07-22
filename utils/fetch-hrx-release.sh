#!/usr/bin/env bash
##===- utils/fetch-hrx-release.sh ---------------------------*- Script -*-===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Download, checksum-verify, and extract the pinned HRX (amdxdna) release so the
# HRX runtime path has a libhrx to dispatch through -- without cloning/building
# HRX from source. Both HRX entry points consume the same libhrx provisioned
# here, and both are selected by a single NPU_RUNTIME=hrx variable: the
# IRON/Python flow reads it at import, and the C++ example `make` flow reads it
# to build the HRX host stack.
#
# Usage:
#   utils/fetch-hrx-release.sh            # fetch + extract (idempotent)
#   eval "$(utils/fetch-hrx-release.sh --print-env)"  # + export HRX_*
#
# The pinned coordinates live in utils/hrx-release.env; any of
# HRX_RELEASE_{REPO,TAG,ASSET,SHA256} may be overridden from the environment.
# Set HRX_RELEASE_DIR to change where the asset is unpacked (default:
# third_party/.hrx-release). Set FORCE=1 to re-download over an existing
# extraction.
#
# Auth: a private release needs a token. The script prefers `gh release
# download` (which honors the runner's gh auth / GH_TOKEN); otherwise it falls
# back to curl and, if set, sends GH_TOKEN/GITHUB_TOKEN as a bearer token.
#
# On success it prints the absolute path to the extracted tree's env.sh (source
# it, or use --print-env to get shell `export` lines) which sets HRX_DIR /
# LD_LIBRARY_PATH / CMAKE_PREFIX_PATH.
#
##===----------------------------------------------------------------------===##

set -euo pipefail

PRINT_ENV=0
[[ "${1:-}" == "--print-env" ]] && PRINT_ENV=1

# Everything the human-facing progress text prints must go to stderr so that
# `eval "$(... --print-env)"` only ever evaluates the export lines on stdout.
log() { echo "[fetch-hrx-release] $*" >&2; }
die() { echo "[fetch-hrx-release] ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hrx-release.env"

# Environment overrides win over the pinned defaults.
REPO="${HRX_RELEASE_REPO:?}"
TAG="${HRX_RELEASE_TAG:?}"
ASSET="${HRX_RELEASE_ASSET:?}"
SHA256="${HRX_RELEASE_SHA256:?}"
# Fetched artifacts live under third_party/ (a build artifact, git-ignored),
# next to the other vendored third-party dependencies.
DEST_DIR="${HRX_RELEASE_DIR:-${REPO_ROOT}/third_party/.hrx-release}"

mkdir -p "${DEST_DIR}"
TARBALL="${DEST_DIR}/${ASSET}"
# The asset unpacks to a single top-level directory named after the asset stem.
# Support both the legacy gzip tarball and the newer zstd tarball.
case "${ASSET}" in
  *.tar.zst) ASSET_STEM="${ASSET%.tar.zst}" ;;
  *.tar.gz)  ASSET_STEM="${ASSET%.tar.gz}" ;;
  *) die "unsupported asset extension: ${ASSET} (expected .tar.zst or .tar.gz)" ;;
esac
EXTRACT_DIR="${DEST_DIR}/${ASSET_STEM}"
ENV_SH="${EXTRACT_DIR}/env.sh"

emit_env() {
  # Either point the caller at env.sh or, with --print-env, emit export lines.
  if [[ "${PRINT_ENV}" == "1" ]]; then
    echo "source \"${ENV_SH}\""
  else
    log "HRX release ready. Source its environment with:"
    log "    source \"${ENV_SH}\""
    echo "${ENV_SH}"
  fi
}

# Idempotent: reuse an existing good extraction unless FORCE=1.
if [[ -f "${ENV_SH}" && "${FORCE:-0}" != "1" ]]; then
  log "Already extracted at ${EXTRACT_DIR} (set FORCE=1 to refetch)."
  emit_env
  exit 0
fi

verify_sha256() {
  local f="$1"
  local actual
  actual="$(sha256sum "${f}" | awk '{print $1}')"
  [[ "${actual}" == "${SHA256}" ]] || \
    die "checksum mismatch for ${f}: got ${actual}, expected ${SHA256}"
  log "Checksum OK (${SHA256})."
}

download() {
  if [[ -f "${TARBALL}" && "${FORCE:-0}" != "1" ]]; then
    log "Reusing cached ${TARBALL}."
    return 0
  fi
  log "Downloading ${ASSET} from ${REPO}@${TAG} ..."
  if command -v gh >/dev/null 2>&1; then
    # gh honors the runner's auth and works for private releases.
    gh release download "${TAG}" --repo "${REPO}" --pattern "${ASSET}" \
      --dir "${DEST_DIR}" --clobber \
      && return 0
    log "gh download failed; falling back to curl."
  fi
  local url="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
  local auth=()
  local token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  [[ -n "${token}" ]] && auth=(-H "Authorization: Bearer ${token}")
  curl -fL "${auth[@]}" -o "${TARBALL}" "${url}" \
    || die "download failed: ${url} (private release? set GH_TOKEN or install gh)"
}

download
verify_sha256 "${TARBALL}"

log "Extracting into ${DEST_DIR} ..."
rm -rf "${EXTRACT_DIR}"
case "${ASSET}" in
  *.tar.zst)
    command -v zstd >/dev/null 2>&1 || command -v unzstd >/dev/null 2>&1 || \
      die "zstd is required to extract ${ASSET} (install zstd)."
    # Prefer tar's native --zstd; fall back to piping through zstd -d.
    if tar --zstd -xf "${TARBALL}" -C "${DEST_DIR}" 2>/dev/null; then
      :
    else
      zstd -dc "${TARBALL}" | tar -xf - -C "${DEST_DIR}"
    fi
    ;;
  *.tar.gz)
    tar -xzf "${TARBALL}" -C "${DEST_DIR}"
    ;;
esac

# Newer releases (>= v2026.07.20) ship a plain relocatable install prefix
# (include/hrx + lib/libhrx.so + lib/cmake/hrx) with no env.sh. Older releases
# shipped a source/build tree with an env.sh. If the extraction did not provide
# an env.sh, synthesize one for the install-prefix layout so consumers (and
# --print-env) have a single, stable entry point in both cases.
if [[ ! -f "${ENV_SH}" ]]; then
  if [[ -f "${EXTRACT_DIR}/lib/libhrx.so" && -f "${EXTRACT_DIR}/include/hrx/hrx_runtime.h" ]]; then
    log "No env.sh in asset; synthesizing one for the install-prefix layout."
    cat > "${ENV_SH}" <<'EOF'
#!/usr/bin/env bash
# Auto-generated by fetch-hrx-release.sh for the install-prefix release layout.
_hrx_pkg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HRX_DIR="$_hrx_pkg_dir"
export LD_LIBRARY_PATH="$_hrx_pkg_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="$_hrx_pkg_dir${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
EOF
  else
    die "extraction did not produce ${ENV_SH} and no install-prefix layout was found in ${EXTRACT_DIR}"
  fi
fi

emit_env
