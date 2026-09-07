#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Functional smoke test for hrx-xclbinutil: version check + a real xclbin section
# round-trip (JSON -> binary -> JSON), which exercises the hrx util JSON/ptree
# engine and the container packaging. Self-contained: no external deps, no NPU.
#
#   run_tests.sh <path-to-hrx-xclbinutil>
set -euo pipefail
BIN="${1:?usage: run_tests.sh <path-to-hrx-xclbinutil>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# Under Git Bash the shell hands out MSYS paths (/d/a/... , /tmp/...), but the
# binary is a native Windows program and only understands D:/a/... . Translate
# any path handed to it. On Linux there is no cygpath and this is the identity.
nat() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
HERE_N="$(nat "$HERE")"
TMP_N="$(nat "$TMP")"

echo "[1/4] --version reports the XRT build version"
"$BIN" --version | grep -q "XRT Build Version: 2.18.0"

echo "[2/4] package a MEM_TOPOLOGY JSON into a .xclbin"
"$BIN" --add-replace-section MEM_TOPOLOGY:JSON:"$HERE_N/data/mem_topology.json" \
       --force --output "$TMP_N/out.xclbin" | grep -q "Successfully wrote"
test -s "$TMP/out.xclbin"

echo "[3/4] --info lists the MEM_TOPOLOGY section"
"$BIN" --info --input "$TMP_N/out.xclbin" | grep -q "MEM_TOPOLOGY"

echo "[4/4] dump MEM_TOPOLOGY back; values round-trip"
"$BIN" --dump-section MEM_TOPOLOGY:JSON:"$TMP_N/dump.json" --input "$TMP_N/out.xclbin"
grep -q "HOST"     "$TMP/dump.json"   # m_tag survived JSON->binary->JSON
grep -q "MEM_DRAM" "$TMP/dump.json"   # m_type survived

echo "PASS"
