<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Resident objectFIFO re-arm (`aiex.dma_channel_reset_for`)

On-board test for `aiex.dma_channel_reset_for`
([#3393](https://github.com/Xilinx/mlir-aie/pull/3393)), which merged with lit
tests only. It exercises the case the op exists for: an objectFIFO the core
acquires once and **holds across dispatches**, whose lock and channel-queue state
therefore has to be re-established at the top of every runtime sequence, with no
PDI reload in between.

`aiex.dma_channel_reset` ([`../local_reset/dma_channel_reset_op`](../local_reset/dma_channel_reset_op/README.md))
is the reset-only sibling: it flushes one named channel and leaves the re-push and
lock arm to the caller. `dma_channel_reset_for` names the *fifo* and emits the
whole trio -- reset, `START_QUEUE` re-push, `aiex.set_lock` -- for every non-shim
endpoint of it.

## Design

`aie2.py`, on `npu2`:

| Fifo | Path | Depth | Per dispatch |
|------|------|-------|--------------|
| `weights` | memtile `(1,1)` -> core `(1,2)` | 1 | **resident** -- `initValues` `[1..256]`, acquired once and held across all 16 inner iterations |
| `inputs` | shim `(0,0)` -> core `(1,2)` | 2 | streamed, 16 tiles of 16 |
| `outputs` | core `(1,2)` -> shim `(0,0)` | 2 | streamed, 16 tiles of 16 |

The core adds `weights[i]` to `inputs[i]`, so a correct dispatch returns
`output[i] == input[i] + i + 1` exactly. The host varies the input every
dispatch, which is what makes a fifo that stopped delivering visible: the collect
comes back holding the *previous* dispatch's result.

## Behaviour

- **With the op (this test):** 1000/1000 dispatches exact on one hardware
  context.
- **Without it:** the resident fifo is never re-armed. Dispatches alternate
  between correct and returning the previous dispatch's output verbatim (256/256
  elements wrong); measured 12-14 of 32 exact. The run can also degrade further
  and leave the mailbox channel unable to accept commands, so the failure is not
  always a clean wrong-data one. It is not part of the automated run for that
  reason -- reproduce it by deleting the `dma_channel_reset_for` line from
  `aie2.py`.

## Memtile column

The memtile is placed in array column 1, and that placement is load-bearing.
With the same design and the memtile in **column 0**, the re-arm leaves the first
word of the resident `initValues` buffer reading `0x00CD0CD0` instead of `1`, on
every dispatch after the first -- 1 of 32 dispatches exact, one element wrong
each time. Columns 1 through 7 are all exact.

Measured on `npu2` (Strix), memtile column swept 0-7 with the shim held at column
0 and the core at `(0,2)`, 32 dispatches per column, both sweep orderings. The
declared partition width does not change it (`npu2_1col`, `npu2_2col`,
`npu2_4col` and `npu2` all fail identically with the memtile in column 0), which
is consistent with `amdxdna` allocating every hardware context on this part at
array column 0 across the full array width (`aie2_alloc_resource`, `AIE2_TEMPORAL_ONLY`).
A memtile buffer filled from the shim through a link instead of `initValues` is
exact in column 0, so the residual is specific to a resident compile-time-initialized
buffer there. The constant itself is not in any build artifact and is unidentified.

## Reference

`aiex.dma_channel_reset_for` is defined in
`include/aie/Dialect/AIEX/IR/AIEX.td`, bound to the fifo by
`AIEObjectFifoStatefulTransform` (via `aie.objectfifo_rearm_binding`, whose head
BD id and repeat count `--aie-assign-bd-ids` folds in), and lowered by
`lib/Dialect/AIEX/Transforms/AIELowerDmaChannelReset.cpp`.
