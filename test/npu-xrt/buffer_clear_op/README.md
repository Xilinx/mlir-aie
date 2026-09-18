<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Buffer clear

Zeros a core tile's resident accumulator from the runtime sequence with the
`aiex.buffer_clear` op, and shows the design keeps working across it without
reloading and without re-running the compute core.

The core on tile `(0, 2)` runs once and fills an 8-word accumulator `acc`
(pinned at local address `0`) with `[1, 2, ..., 8]`. Batch 1 reads it back
unchanged. `aiex.buffer_clear(%tile_0_2, 0, 8)` then zeros it directly from the
sequence; I re-trigger the readback DMA with `aiex.set_lock` rather than
running the core again, so the only thing that could have zeroed the
accumulator between batch 1 and batch 2 is the op itself. Batch 2 reads back
`[0, 0, ..., 0]`.

## Why no core re-run

`aiex.core_reset` (see `../local_reset/core_reset_op`) clears the program
counter but not data memory: reset-and-rerun proves the core is back in a known
state, not that the op under test cleared memory. `aiex.buffer_clear` is the
opposite case, a data-memory op with no PC involvement, so isolating it means
keeping the core out of the picture after its one run and driving the second
readback purely from the runtime sequence.

## Behaviour

- **Correct protocol (this test):** `aiex.buffer_clear` zeros `acc` in place
  -> batch2 is all zero while batch1 is `[1..8]`.
- **No clear at all:** batch2 would equal batch1 (`[1..8]`), since nothing else
  touches `acc` after the core's single run. Reproduce by removing
  `aiex.buffer_clear` from `aie.mlir`.

## Reference

`aiex.buffer_clear` is defined in `include/aie/Dialect/AIEX/IR/AIEX.td` and
lowered by `lib/Dialect/AIEX/Transforms/AIELowerBufferClear.cpp` (pass
`--aie-lower-buffer-clear`, run in the default `aiecc` pipeline, right after
`--aie-lower-core-reset`). It lowers to a single `aiex.npu.blockwrite` of
`length` zero words, the same mechanism the public aie-rt driver
(<https://github.com/Xilinx/aie-rt>, vendored at `third_party/aie-rt/`) uses to
write tile data memory in bulk: `XAie_DataMemBlockWrite` in
`driver/src/memory/xaie_mem.c` builds the payload in a caller buffer and issues
a block write; there is no dedicated memset entry point at the driver layer
either. The verifier's tile-local memory bounds come from
`AIETargetModel::getLocalMemorySize` / `getMemTileSize`, matching aie-rt's
`Aie2PTileMemMod.Size` / `Aie2PMemTileMemMod.Size` in
`driver/src/global/xaie2pgbl_reginit.c`.

Unlike `aiex.core_reset` / `aiex.dma_channel_reset`, this op does not lower to
a fixed-size register pulse: the emitted `npu.blockwrite` carries `length`
literal zero words, so its cost scales with the region size (see the op's
description in `AIEX.td` for why a DMA-of-zeros lowering is not a better fit
here, and where it would be).

## A note on this test's buffer address

`acc` is pinned at local address `0` with the `address` attribute on
`aie.buffer` so `aiex.buffer_clear` (which addresses tile-local memory by
offset, not by symbol) has a known target. I could not build this design in
the environment I wrote it in, so I have not confirmed on a real allocator run
that address `0` does not collide with the core's own stack or other
compiler-placed state for this particular design; verify this on the first
real build alongside the rest of the suite.
