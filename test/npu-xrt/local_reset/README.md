<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Local reset

On-board tests that exercise **local reset**: returning a single tile-local
hardware block to a clean state from the runtime sequence, without rebuilding the
design or reloading the array. Each family resets one resettable block and shows
that the design keeps working across it. The `core` and `dma` families issue the
reset as **raw register writes**; the two `*_op` families issue it through the
merged `aiex.core_reset` / `aiex.dma_channel_reset` runtime-sequence ops, which
drive the same registers (see [Op-based variants](#op-based-variants)).

| Family | Block reset | Key register(s) | Reset mechanism |
|--------|-------------|-----------------|-----------------|
| [`core`](core/README.md) | AI Engine core | `Core_Control` (`0x32000`) | masked reset -> unreset -> enable (one field per write) |
| [`dma`](dma/README.md) | MM2S DMA channel | `DMA_MM2S_0_Ctrl` (`0x1DE10`) | masked reset pulse flushes a queued BD, then push good BD + arm lock |

Both families reset with the driver's write *type*: masked `maskwrite32` (aie-rt's
`XAie_CoreReset`/`Unreset`/`Enable` and `XAie_DmaChannelReset` are `MaskWrite32` of
one bit-field). The `dma` lock re-arm uses the `aiex.set_lock` op, which lowers to
the full-word write of the lock value register that aie-rt's `XAie_LockSetValue`
performs.

## Op-based variants

The two `*_op` families are the on-board counterparts of `core` and `dma` that
drive the merged reset ops
([#3375](https://github.com/Xilinx/mlir-aie/pull/3375) /
[#3370](https://github.com/Xilinx/mlir-aie/pull/3370)) rather than issuing the
`maskwrite32` reset pulse directly, as the raw families do.

| Family | Op | Lowers to | Raw-register sibling |
|--------|----|-----------|----------------------|
| [`core_reset_op`](core_reset_op/README.md) | `aiex.core_reset` | mask-preserving reset pulse on `Core_Control` (`0x32000`, bit 1) | [`core`](core/README.md) |
| [`dma_channel_reset_op`](dma_channel_reset_op/README.md) | `aiex.dma_channel_reset` | mask-preserving reset pulse on `DMA_MM2S_0_Ctrl` (`0x1DE10`, bit 1) | [`dma`](dma/README.md) |

Both ops lower (in the default `aiecc` pipeline) to a two-write
`aiex.npu.maskwrite32` pulse on the **same register and reset bit** the raw
sibling writes, so the op-based and raw families exercise the same protocol. The
ops are **reset-only**: they mask to the reset bit (preserving the surrounding
fields) and do not re-enable, re-push, or re-arm. So:

- `dma_channel_reset_op` is a drop-in for `dma`'s masked reset pulse; the BD
  enqueues + lock arm remain around it, and it passes unchanged.
- `core_reset_op` supplies the `reset -> unreset` pulse
  (`XAie_CoreReset`/`XAie_CoreUnreset`); because this core has run to `aie.end` (no
  longer enabled) it composes the op with a **masked** re-enable mirroring
  `XAie_CoreEnable`, so the whole test is the driver's
  `XAie_CoreReset -> XAie_CoreUnreset -> XAie_CoreEnable` sequence, every write
  masked to one field. The op *alone* does not re-enable, so on its own it leaves
  such a core halted; this pins the op's documented scope (it assumes a
  still-enabled resident core). See its README.

## Shared design

All families run on a single column, core tile `(0,2)` -> shim `(0,0)`:

- **Resident workload.** The data stays in tile memory across dispatches, so a
  correct reset means every dispatch returns identical bytes (or, for `core`, a
  deterministic `+1` from a counter kept in data memory).
- **Reset while settled.** The reset acts on a settled block -- a core halted at
  `aie.end`, or a DMA channel stalled on a lock acquire with a BD queued behind the
  lock. The writes take effect on a settled block; they are not used to preempt a
  running one.
- **Driven from the runtime sequence.** Each test issues the reset from
  `aie.runtime_sequence` with `aiex.npu.maskwrite32` (core/dma reset), plus
  `aiex.set_lock` (`dma` lock re-arm) and `aiex.npu.push_queue` (`dma` start
  queue), then dispatches and checks the host-visible result.

## What the automated run covers

`run.lit` runs **only the correct protocol** -- the reset sequence that keeps the
design working. The `dma` families are falsifiable: removing the reset leaves a
"bad" buffer descriptor queued ahead of the good one, so the host collects the
wrong bytes -- a clean failure rather than a hang. The remaining negatives (reset
without re-push/re-arm) **hang by design**: a collect that never completes. All are
described in each family's README and reproduced by hand by editing the reset
sequence in `aie.mlir`; they are not part of the automated run.

## Architecture and register offsets

The tests write raw tile-local offsets. All families target both NPU generations
-- **npu1** (Phoenix, AIE-ML) and **npu2** (Strix, AIE2P) -- because every offset
and bit field they touch is the same on both:

| Register | npu1 (AIE-ML) | npu2 (AIE2P) |
|----------|---------------|--------------|
| `Core_Control` | `0x32000` | `0x32000` |
| `DMA_MM2S_0_Ctrl` / `_Start_Queue` | `0x1DE10` / `0x1DE14` | `0x1DE10` / `0x1DE14` |
| `LOCK0_VALUE` | `0x1F000` | `0x1F000` |

In aie-rt the raw offsets are the `XAIEMLGBL_*` defines (`xaiemlgbl_params.h`, 
used for npu1 via `XAIE_DEV_GEN_AIE2IPU`) and the matching `XAIE2PGBL_*` defines
(`xaie2pgbl_params.h`, npu2); see [References](#references). (The older AIE1 /
Versal parts in `xaiegbl_params.h` use a different layout, but they are not NPU
targets and are not exercised here.)

## Running

```
lit -sv core/run.lit                              # one family
lit -sv build/test/npu-xrt/local_reset            # all families, from the build dir
```

Requires a Ryzen AI device (`REQUIRES: ryzen_ai`). The board generation is
auto-detected; every family runs on both npu1 (Phoenix) and npu2 (Strix).

## References

Every offset, bit field, and reset procedure asserted here is defined in the
vendored AI Engine driver **aie-rt** (public: <https://github.com/Xilinx/aie-rt>),
under `third_party/aie-rt/` in this repo. The table lists the AIE2P (npu2) names
from `driver/src/global/xaie2pgbl_params.h`; the npu1 (AIE-ML) equivalents are the
identically-suffixed `XAIEMLGBL_*` defines in `xaiemlgbl_params.h` at the same
offsets. The tests issue each protocol directly, with the **same register-write
type the driver routine uses** -- `aiex.npu.maskwrite32` where the driver is a
`MaskWrite32` (core/dma reset), plus `aiex.npu.push_queue` for the DMA start
queue. The lock re-arm uses the `aiex.set_lock` op, which lowers to the `Write32`
of the lock value register that `XAie_LockSetValue` performs.

| Protocol (test) | Register define -- `xaie2pgbl_params.h` / `xaiemlgbl_params.h` | Driver routine |
|-----------------|----------------------------------------|----------------|
| Core reset -> unreset -> enable | `..._CORE_MODULE_CORE_CONTROL` (`0x32000`), `_RESET_LSB`=1, `_ENABLE_LSB`=0 | `XAie_CoreReset` / `XAie_CoreUnreset` / `XAie_CoreEnable` -- `driver/src/core/xaie_core.c` |
| DMA channel reset | `..._MEMORY_MODULE_DMA_MM2S_0_CTRL` (`0x1DE10`), `_RESET_LSB`=1, `_ENABLE_LSB`=0 | `XAie_DmaChannelReset` -- `driver/src/dma/xaie_dma.c` |
| Re-push BD to start queue | `..._MEMORY_MODULE_DMA_MM2S_0_START_QUEUE` (`0x1DE14`) | `XAie_DmaChannelPushBdToQueue` -- `driver/src/dma/xaie_dma.c` |
| Lock re-arm (set value) | `..._MEMORY_MODULE_LOCK0_VALUE` (`0x1F000`) | `XAie_LockSetValue` -- `driver/src/locks/xaie_locks.c` |

Each test issues these as the driver's own write type: `XAie_CoreReset`/`Unreset`/
`Enable` and `XAie_DmaChannelReset` are `MaskWrite32` of a single bit-field, so the
`core` and `dma` tests use `maskwrite32` (reset/unreset mask `0x2`, enable mask
`0x1`); the `dma` lock re-arm uses the `aiex.set_lock` op, which lowers to the
full-word `write32` that `XAie_LockSetValue` performs.
