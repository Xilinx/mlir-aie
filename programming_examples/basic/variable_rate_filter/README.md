<!--
   Copyright (C) 2026 Advanced Micro Devices, Inc.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# variable_rate_filter (G-T3.2-007 worked example)

End-to-end IRON design demonstrating the new
[`VariableRateFifo`](../../../python/iron/variable_rate.py)
primitive. Closes the *single-producer / conditional-forward* half
of `G-T6.2-001` + `G-T7.4-200`.

## What this example shows

A producer worker reads each input window from a fixed-rate
upstream `ObjectFifo`, runs an external C++ predicate kernel
(`filterFirstByteEven`), and conditionally forwards the window to a
downstream `VariableRateFifo`. The downstream consumer (the host
runtime drain) sees only the forwarded windows.

Topology:

```
  shim DMA (host)
        |
        v
  in_of (ObjectFifo)
        |
        v
  Tile A (filter_kernel)
        |
        v
  out_of (VariableRateFifo)   <-- aie.variable_rate = true
        |
        v
  shim DMA (host)
```

The filter kernel uses
[`VariableRateFifoHandle.discard(n)`](../../../python/iron/variable_rate.py)
on skip iterations -- the auditable counterpart to "just don't
call acquire/release in the skip branch". Discard emits no MLIR;
the static-rate invariant is intentionally relaxed via the
`aie.variable_rate = true` discardable attribute pinned by
`VariableRateFifo.resolve()` and consumed by the
[`AIEObjectFifoStatefulTransformPass`](../../../lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp)
in two places:

1. The LCM-based loop-unroll skips variable-rate fifos.
2. The split-fifo path propagates the marker to consumer-side
   fifos (mirrors the SparseFifo G-T3.2-006 propagation).

## Build

```sh
source /home/matteius/xdna-bringup/ironenv/bin/activate
source /opt/xilinx/xrt/setup.sh
export MLIR_AIE_DIR=$(pwd)/../../..  # adjust to the worktree root
export PEANO_INSTALL_DIR=/home/matteius/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie
export PATH="$MLIR_AIE_DIR/install/bin:$PEANO_INSTALL_DIR/bin:$PATH"
cd $MLIR_AIE_DIR/build && ninja -j8 install
cd $MLIR_AIE_DIR/programming_examples/basic/variable_rate_filter
make NPU2=1
```

This rebuilds against the worktree's `python/iron/variable_rate.py`
+ the patched
`lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp`,
emits the lowered MLIR + xclbin under `build/`.

## Inspect the lowered marker

```sh
grep variable_rate build/aie_4096.mlir.prj/input_with_addresses.mlir
```

Expected output (two lines):

- `aie.objectfifo @out_of (...) {aie.variable_rate = true} ...`
  -- the producer-side ObjectFifoCreateOp the IRON
  `VariableRateFifo.resolve()` pinned the attribute on.
- `aie.objectfifo @out_of_cons (...) {aie.variable_rate = true} ...`
  -- the consumer-side ObjectFifoCreateOp produced by the
  split-fifo path; the G-T3.2-007 propagation slot picked up
  the marker.

Cross-reference with
[VARIABLE_RATE_DESIGN.md](../../../python/iron/VARIABLE_RATE_DESIGN.md)
for the full design rationale.

## Sibling primitive

For the `N:1 multi-producer fan-in` half of G-T6.2-001 / G-T7.4-200
(many independent producers fanning into one consumer at
runtime-decided rates), use [`PacketFifo`](../../../python/iron/packet.py)
instead. The two are sibling primitives -- choose based on the
topology:

| Topology | Use |
|---|---|
| 1 producer, conditional forward | `VariableRateFifo` (this example) |
| N producers fanning to 1 consumer | `PacketFifo` |
