#!/usr/bin/env bash
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
# trace_m8.sh — build + run + parse a HW trace of the m8 2-tile megakernel.
#
# Usage:
#   scripts/trace_m8.sh                 # default 8 events
#   TRACE_EVENTS=INSTR_EVENT_0,INSTR_EVENT_1,LOCK_STALL,STREAM_STALL,MEMORY_STALL,CORE_STALL \
#     scripts/trace_m8.sh               # stall-attribution preset
#   TRACE_SIZE_PER_WORKER=65536 scripts/trace_m8.sh   # bigger trace BO per tile
#
# Output:
#   build/trace_m8.txt   raw trace (one hex word per line)
#   build/trace_m8.json  parsed events
#   stdout              get_trace_summary.py table
set -euo pipefail

cd "$(dirname "$0")/.."
SRCDIR=$(pwd)
BUILD="$SRCDIR/build"
BLOCK=m8
XCLBIN="$BUILD/final_${BLOCK}.xclbin"
INSTS="$BUILD/insts_${BLOCK}.bin"
MLIR="$BUILD/aie_${BLOCK}.mlir"
PHYS_MLIR="$BUILD/aie_${BLOCK}.mlir.prj/input_with_addresses.mlir"

TRACE_SIZE_PER_WORKER="${TRACE_SIZE_PER_WORKER:-32768}"
export TRACE_SIZE_PER_WORKER
# Use ddr_id=-1 (append to last tensor) for per-block; chain uses ddr_id=4.
export TRACE_DDR_ID="${TRACE_DDR_ID:--1}"
# Total trace BO bytes the HOST needs to read back: per-worker × #workers.
# m8 megakernel 2-tile has 2 workers (tile A on (5,3), tile B on (5,4)).
TRACE_TOTAL=$(( TRACE_SIZE_PER_WORKER * 2 ))
# M8_MEGAKERNEL_2TILE=1 is required to dispatch into m8_megakernel_2tile.py.
export M8_MEGAKERNEL_2TILE=1

# TRACE_EVENTS is read by aie2_yolo_per_block.py:60; passing through.
echo "[trace_m8] TRACE_SIZE_PER_WORKER=$TRACE_SIZE_PER_WORKER  TRACE_EVENTS=${TRACE_EVENTS:-<default 8>}  TRACE_DDR_ID=$TRACE_DDR_ID"

# The MLIR depends on env vars (TRACE_*), so always regenerate.
rm -f "$MLIR" "$XCLBIN" "$INSTS"
make -C "$SRCDIR" BLOCK="$BLOCK" "$XCLBIN" "$INSTS"

# Run one warmup + one traced iteration. trace.txt is overwritten per iter,
# so only the final iter's trace is kept. n-iters=1 keeps it.
cd "$SRCDIR"
TRACE_TXT="$BUILD/trace_m8.txt"
TRACE_JSON="$BUILD/trace_m8.json"

# Force load_and_run to write trace.txt into build/ by cd-ing there.
# --trace-sz must be set on the host CLI too — it's what makes create_npu_kernel
# attach a TraceConfig to the NPUKernel; without it the host never reads the
# trace BO back even though the MLIR has trace ops baked in.
# --ddr-id mirrors the build-time TRACE_DDR_ID so host buffer layout matches.
( cd "$BUILD" && python3 "$SRCDIR/scripts/time_block.py" --block "$BLOCK" \
    -x "$XCLBIN" -i "$INSTS" -k MLIR_AIE --n-iters 1 --n-warmup 1 \
    --trace-sz "$TRACE_TOTAL" --ddr-id "$TRACE_DDR_ID" )

# load_and_run writes ./trace.txt; rename for archival.
if [[ -f "$BUILD/trace.txt" ]]; then
    mv "$BUILD/trace.txt" "$TRACE_TXT"
fi
if [[ ! -f "$TRACE_TXT" ]]; then
    echo "[trace_m8] ERROR: no trace.txt produced. Did the build actually include trace ops? Check 'rt.enable_trace' was reached (TRACE_SIZE_PER_WORKER must be set at MLIR-generation time)." >&2
    exit 1
fi

echo "[trace_m8] parsing $TRACE_TXT against $PHYS_MLIR"
python3 "$SRCDIR/../../../python/utils/trace/parse.py" \
    --input "$TRACE_TXT" --mlir "$PHYS_MLIR" --output "$TRACE_JSON"

echo "[trace_m8] summary:"
python3 "$SRCDIR/../../../python/utils/trace/get_trace_summary.py" --input "$TRACE_JSON"

echo "[trace_m8] done. Artifacts: $TRACE_TXT  $TRACE_JSON"
