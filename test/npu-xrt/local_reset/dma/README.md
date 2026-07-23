<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# DMA channel reset

Interrupting and restarting a **run-forever** DMA channel by resetting it. Tile
`(0, 2)` streams a resident buffer to the host on an MM2S channel whose single BD
loops back to itself (`next_bd`), lock-gated so that between sends the channel is
**stalled on the lock acquire**.

A DMA channel reset is only valid while the channel is stalled (on a lock acquire,
or S2MM stream starvation) -- not mid-transfer. Its effect is to **flush the
channel's task queue** and return it to idle. So the reset by itself *stops* a
run-forever channel; to keep using it you reset it, **re-push a BD** to the queue
(`aiex.npu.push_queue`), and re-arm the lock.

The reset pulses the `Reset` bit (bit 1) of `DMA_MM2S_0_Ctrl` (tile-local
`0x1DE10` for MM2S ch0) with masked writes -- `assert (mask 0x2) -> deassert` --
exactly as aie-rt's `XAie_DmaChannelReset` does; masking to the reset bit preserves
the channel's other CTRL fields, so no disable/enable is needed. It then re-pushes
BD 0 via `DMA_MM2S_0_Start_Queue` (`0x1DE14`) and re-arms the lock. The host
dispatches the same kernel repeatedly; the buffer is resident, so every dispatch
must return the same bytes (`100 + i`).

The lock re-arm writes `LOCK0_VALUE` (`0x1F000`) with a full-word `write32`, as the
driver's `XAie_LockSetValue` does. Every register this test uses has the same
offset on npu1 (AIE-ML) and npu2 (AIE2P), so it runs on both. See
[`../README.md`](../README.md) for the shared design, the architecture note, and
how to run.

## Behaviour

- **Correct protocol (this test):** reset while stalled + re-push BD 0 + re-arm
  the lock -> the channel stays correct across every dispatch.
- **Reset without re-push:** the reset flushes the task queue, so with nothing
  re-pushed the channel stays idle and the next collect never completes.
- **No reset:** a run-forever channel that is never interrupted needs no reset and
  streams correctly.

The failure modes hang by design; reproduce them by hand by editing the reset
sequence in `aie.mlir`.

## Reference

Registers and the channel-reset / push-queue procedure are defined in the public
aie-rt driver (<https://github.com/Xilinx/aie-rt>, vendored at
`third_party/aie-rt/`): `XAIE2PGBL_MEMORY_MODULE_DMA_MM2S_0_CTRL`,
`..._DMA_MM2S_0_START_QUEUE`, and `..._LOCK0_VALUE` in
`driver/src/global/xaie2pgbl_params.h`; `XAie_DmaChannelReset` and
`XAie_DmaChannelPushBdToQueue` in `driver/src/dma/xaie_dma.c` (`XAie_DmaChannelReset`
is a `MaskWrite32` of the channel `Reset` bit -- this test issues that same masked
pulse directly); `XAie_LockSetValue` in `driver/src/locks/xaie_locks.c` (a
full-word `Write32`, matching the lock re-arm here). See
[`../README.md`](../README.md#references) for the full table.
