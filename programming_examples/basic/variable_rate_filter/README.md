<!--
   Copyright (C) 2026 Advanced Micro Devices, Inc.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# variable_rate_filter

End-to-end IRON design demonstrating the new
[`VariableRateFifo`](../../../python/iron/variable_rate.py)
primitive — the single-producer / conditional-forward dataflow
path that vanilla `ObjectFifo` cannot express.

## What this example shows

A producer worker reads input windows from a fixed-rate upstream
`ObjectFifo`. On every other window it forwards the window to a
downstream `VariableRateFifo` via a C++ window-copy kernel
(`filterFirstByteEven` — currently a pure copy; the file name is
historical). On the alternate window it calls `discard(1)` on the
producer handle, telling the variable-rate fifo to skip that slot
without forwarding. The downstream consumer (the host runtime
drain) sees only the forwarded windows.

The skip decision is made at the IRON Python layer using a
deterministic alternating pattern. A richer "predicate decided in
the C++ kernel per window" variant would require a first-class
`scf.if` lowering on the conditional acquire/release that IRON
Python does not currently expose; the alternating-Python pattern
is sufficient to exercise `discard(1)` and the
`aie.variable_rate = true` marker end-to-end.

Topology:

```
  shim DMA (host)
        |
        v
  in_of (ObjectFifo)
        |
        v
  Tile A (alternating skip + window-copy kernel)
        |
        v
  out_of (VariableRateFifo)   <-- aie.variable_rate = true
        |
        v
  shim DMA (host)
```

[`VariableRateFifoHandle.discard(n)`](../../../python/iron/variable_rate.py)
is invoked on skip iterations -- the auditable counterpart to
"just don't call acquire/release in the skip branch". Discard
emits no MLIR; the static-rate invariant is intentionally relaxed
via the `aie.variable_rate = true` discardable attribute pinned by
`VariableRateFifo.resolve()` and consumed by the
[`AIEObjectFifoStatefulTransformPass`](../../../lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp)
in two places:

1. The LCM-based loop-unroll skips variable-rate fifos.
2. The split-fifo path propagates the marker to consumer-side
   fifos so diagnostic dumps and the runtime-counter machinery
   on both halves see the marker.

## Build

```sh
source <path/to/ironenv>/bin/activate
source /opt/xilinx/xrt/setup.sh
export MLIR_AIE_DIR=$(pwd)/../../..  # adjust to the worktree root
export PEANO_INSTALL_DIR=<path/to/llvm-aie>  # the installed Peano tree
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
  the marker.

Cross-reference with
[VARIABLE_RATE_DESIGN.md](../../../python/iron/VARIABLE_RATE_DESIGN.md)
for the full design rationale.

## Sibling primitive

(many independent producers fanning into one consumer at
runtime-decided rates), use [`PacketFifo`](../../../python/iron/packet.py)
instead. The two are sibling primitives -- choose based on the
topology:

| Topology | Use |
|---|---|
| 1 producer, conditional forward | `VariableRateFifo` (this example) |
| N producers fanning to 1 consumer | `PacketFifo` |
