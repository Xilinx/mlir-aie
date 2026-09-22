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

echo "[1/5] --version reports the XRT build version"
"$BIN" --version | grep -q "XRT Build Version: 2.18.0"

echo "[2/5] package a MEM_TOPOLOGY JSON into a .xclbin"
"$BIN" --add-replace-section MEM_TOPOLOGY:JSON:"$HERE_N/data/mem_topology.json" \
       --force --output "$TMP_N/out.xclbin" | grep -q "Successfully wrote"
test -s "$TMP/out.xclbin"

echo "[3/5] --info lists the MEM_TOPOLOGY section"
"$BIN" --info --input "$TMP_N/out.xclbin" | grep -q "MEM_TOPOLOGY"

echo "[4/5] dump MEM_TOPOLOGY back; values round-trip"
"$BIN" --dump-section MEM_TOPOLOGY:JSON:"$TMP_N/dump.json" --input "$TMP_N/out.xclbin"
grep -q "HOST"     "$TMP/dump.json"   # m_tag survived JSON->binary->JSON
grep -q "MEM_DRAM" "$TMP/dump.json"   # m_type survived

echo "[5/5] AIE_PARTITION round-trips through its own dump"
# aiecc's --xclbin-input flow dumps the partition of the previous xclbin, appends
# a PDI and re-adds it, so the dump must be something --add-replace-section
# accepts. The section writes each scalar array element with put("", v), which
# names the node itself; splitting "" into an empty key nested every element in
# an array of its own ([["0"]]) and the re-add failed on the empty value.
cat > "$TMP/aie_partition.json" <<JSON
{
  "aie_partition": {
    "name": "QoS",
    "operations_per_cycle": "2048",
    "inference_fingerprint": "23423",
    "pre_post_fingerprint": "12345",
    "partition": { "column_width": 4, "start_columns": [0, 4] },
    "PDIs": [
      {
        "uuid": "acd92aa2-2672-46b4-85df-cfd997367d63",
        "file_name": "$HERE_N/data/sample.pdi",
        "cdo_groups": [
          {
            "name": "DPU",
            "type": "PRIMARY",
            "pdi_id": "0x01",
            "dpu_kernel_ids": ["0x901"],
            "pre_cdo_groups": ["0xC1"]
          }
        ]
      }
    ]
  }
}
JSON
"$BIN" --add-replace-section AIE_PARTITION:JSON:"$TMP_N/aie_partition.json" \
       --force --output "$TMP_N/part.xclbin" | grep -q "Successfully wrote"
"$BIN" --dump-section AIE_PARTITION:JSON:"$TMP_N/part.json" --force --quiet --input "$TMP_N/part.xclbin"
if tr -d ' \n' < "$TMP/part.json" | grep -q '\[\['; then
  echo "FAIL: scalar arrays dumped nested: $(tr -d ' \n' < "$TMP/part.json")"; exit 1
fi
tr -d ' \n' < "$TMP/part.json" | grep -q '"start_columns":\["0","4"\]'
tr -d ' \n' < "$TMP/part.json" | grep -q '"dpu_kernel_ids":\["0x901"\]'
# The dump names the PDI by uuid next to itself; re-add from where it was written.
(cd "$TMP" && "$BIN" --dump-section AIE_PARTITION:JSON:"$TMP_N/part.json" --force --quiet --input "$TMP_N/part.xclbin" \
   && ls *.pdi >/dev/null \
   && "$BIN" --input "$TMP_N/part.xclbin" --add-replace-section AIE_PARTITION:JSON:"$TMP_N/part.json" \
          --force --output "$TMP_N/part2.xclbin" | grep -q "Successfully wrote")

echo "PASS"
