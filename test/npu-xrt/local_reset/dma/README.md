<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# DMA channel reset

MM2S channel 0 on tile `(0, 2)` has two buffer descriptors, both gated on the same
`cons` lock (init 0) so neither runs until the runtime sequence arms it:

- **BD 0 ("bad")** sends `[900..907]`.
- **BD 1 ("good")** sends `[100..107]`.

Each dispatch enqueues the bad BD, **resets the channel to flush the queue**,
enqueues the good BD, then arms the lock -- so only the good BD runs and the host
collects `[100..107]`. The reset is load-bearing: the bad BD is enqueued *ahead*
of the good one, so only the flush keeps it from sending first.

The reset pulses the `Reset` bit of `DMA_MM2S_0_Ctrl` (`0x1DE10`) with masked
writes, exactly as aie-rt's `XAie_DmaChannelReset`; BDs are enqueued through
`DMA_MM2S_0_Start_Queue` (`0x1DE14`), and the lock is armed with `aiex.set_lock`
(`XAie_LockSetValue`, `LOCK0_VALUE` `0x1F000`). Every offset is identical on npu1
and npu2. See [`../README.md`](../README.md) for the shared design and references.

## Behaviour

- **Correct protocol (this test):** enqueue bad -> reset -> enqueue good -> arm
  lock -> the host collects `[100..107]` on every dispatch.
- **No reset (falsifies the test):** the bad BD sends first -> the host collects
  `[900..907]` -- a wrong-data failure, not a hang.
- **Reset without re-push:** the flushed channel has nothing queued -> the collect
  never completes (a hang).

`run.lit` runs only the correct protocol; reproduce the negatives by editing the
reset sequence in `aie.mlir`.

## Reference

Defined in the vendored aie-rt driver (<https://github.com/Xilinx/aie-rt>,
`third_party/aie-rt/`): `XAie_DmaChannelReset` and `XAie_DmaChannelPushBdToQueue`
(`driver/src/dma/xaie_dma.c`), `XAie_LockSetValue` (`driver/src/locks/xaie_locks.c`),
and the register defines in `driver/src/global/xaie2pgbl_params.h`. See
[`../README.md`](../README.md#references) for the full table.
