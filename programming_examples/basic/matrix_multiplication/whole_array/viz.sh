#!/usr/bin/env bash
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
tools=${AIE_TOOLS:-$repo_root/build/bin}
input=${AIE_VIS_INPUT:-$script_dir/build/final_512x512x512_32x32x32_4c.prj/input_with_addresses.mlir}
runtime_sequence=${AIE_VIS_RUNTIME_SEQUENCE:-sequence}
output_dir=${AIE_VIS_OUTPUT_DIR:-$script_dir/viz}
routes=$output_dir/routes-with-vias.mlir

mkdir -p "$output_dir"
rm -rf "$output_dir/flows"

# aie-find-flows remains authoritative for route reconstruction. The visualizer
# consumes its materialized via information and adds DMA, runtime, and call data.
"$tools/aie-opt" --verify-each=false \
  "$input" \
  --aie-find-flows=emit-vias=true \
  -o "$routes"

emit_view() {
  local name=$1
  shift

  "$tools/aie-visualize" \
    --emit-dot \
    --runtime-sequence="$runtime_sequence" \
    --skip-verify \
    "$@" \
    -o "$output_dir/$name.dot" \
    "$routes"

  neato -Gsplines=line -Tsvg \
    -o"$output_dir/$name.svg" "$output_dir/$name.dot"
  cp "$tools/aie-visualize.html" "$output_dir/$name.html"
}

# High abstraction: contract every directed 1-to-1 buffer and port path. Keep
# actual fan-in and fan-out nodes, runtime arguments, calls, and endpoint tiles.
emit_view topology \
  --tile-width=3.2 \
  --topology-only

# Medium abstraction: show endpoint ports and tile-local data movement. Hide
# intermediate switchbox via nodes and arbitration annotations.
emit_view endpoints \
  --tile-width=3.2 \
  --hide-vias

# Low abstraction: show every reconstructed route via, DMA connection, call,
# runtime argument, packet label, and arbitration annotation.
emit_view physical \
  --tile-width=3.2 \
  --tile-height=2.8 \
  --tile-gap-x=1.4 \
  --tile-gap-y=1.4 \
  --local-spacing=0.48

# Per-flow output is useful when the low-abstraction graph is too dense.
"$tools/aie-visualize" \
  --emit-dot-per-flow="$output_dir/flows" \
  --runtime-sequence="$runtime_sequence" \
  --skip-verify \
  "$routes"

for graph in "$output_dir"/flows/flow-*.dot; do
  neato -Gsplines=line -Tsvg -o"$graph.svg" "$graph"
done

printf 'Serve the viewers with:\n'
printf '  python3 -m http.server 8765 --directory %q\n' "$output_dir"
printf 'Then open:\n'
printf '  http://127.0.0.1:8765/%s.html\n' \
  topology \
  endpoints \
  physical
