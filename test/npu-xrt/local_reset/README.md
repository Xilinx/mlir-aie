<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Local reset

On-board tests that exercise **local reset**: returning a single tile-local
hardware block to a clean state with raw register writes issued from the runtime
sequence, without rebuilding the design or reloading the array. Each family
resets one resettable block and shows that the design keeps working across it.

| Family | Block reset | Key register(s) | Reset mechanism |
|--------|-------------|-----------------|-----------------|
| [`core`](core/README.md) | AI Engine core | `Core_Control` (`0x32000`) | assert reset -> release -> enable |
| [`dma`](dma/README.md) | MM2S DMA channel | `DMA_MM2S_0_Ctrl` (`0x1DE10`) | disable -> reset -> deassert -> enable, re-push BD, re-arm lock |
| [`switch`](switch/README.md) | Stream-switch connection | `Stream_Switch_Slave_DMA_0_Config` (`0x3F104`) | re-enable slave port (torn down each dispatch end), re-arm lock |

## Shared design

All three run on a single column, core tile `(0,2)` -> shim `(0,0)`:

- **Resident workload.** The data stays in tile memory across dispatches, so a
  correct reset means every dispatch returns identical bytes (or, for `core`, a
  deterministic `+1` from a counter kept in data memory).
- **Reset while quiescent.** The reset is issued only when the target is settled --
  a core halted at `aie.end`, or a run-forever channel stalled on a lock acquire.
  Raw `write32`s take effect on a settled block; they are not used to preempt a
  running one.
- **Driven from the runtime sequence.** Each test issues the reset from
  `aie.runtime_sequence` with `aiex.npu.write32` (plus `aiex.npu.push_queue` for
  the DMA start queue), then dispatches and checks the host-visible result.

## What the automated run covers

`run.lit` runs **only the correct protocol** -- the reset sequence that keeps the
design working. The failure modes (no reset, reset without re-arm, disable
without re-enable) **hang by design**: a collect that never completes. They are
described in each family's README and reproduced by hand by editing the reset
sequence in `aie.mlir`; they are not part of the automated run.

## Architecture and register offsets

The tests write raw tile-local offsets. All three families target both NPU
generations -- **npu1** (Phoenix, AIE-ML) and **npu2** (Strix, AIE2P) -- because
every offset and bit field they touch is the same on both:

| Register | npu1 (AIE-ML) | npu2 (AIE2P) |
|----------|---------------|--------------|
| `Core_Control` | `0x32000` | `0x32000` |
| `DMA_MM2S_0_Ctrl` / `_Start_Queue` | `0x1DE10` / `0x1DE14` | `0x1DE10` / `0x1DE14` |
| `Stream_Switch_Slave_DMA_0_Config` (bit 31 = `Slave_Enable`) | `0x3F104` | `0x3F104` |
| `LOCK0_VALUE` | `0x1F000` | `0x1F000` |

In aie-rt the raw offsets are the `XAIEMLGBL_*` defines (`xaiemlgbl_params.h`, 
used for npu1 via `XAIE_DEV_GEN_AIE2IPU`) and the matching `XAIE2PGBL_*` defines
(`xaie2pgbl_params.h`, npu2); see [References](#references). (The older AIE1 /
Versal parts in `xaiegbl_params.h` use a different layout, but they are not NPU
targets and are not exercised here.)

## Running

```
lit -sv core/run.lit                              # one family
lit -sv build/test/npu-xrt/local_reset            # all three, from the build dir
```

Requires a Ryzen AI device (`REQUIRES: ryzen_ai`). The board generation is
auto-detected; every family runs on both npu1 (Phoenix) and npu2 (Strix).

## References

Every offset, bit field, and reset procedure asserted here is defined in the
vendored AI Engine driver **aie-rt** (public: <https://github.com/Xilinx/aie-rt>),
under `third_party/aie-rt/` in this repo. The table lists the AIE2P (npu2) names
from `driver/src/global/xaie2pgbl_params.h`; the npu1 (AIE-ML) equivalents are the
identically-suffixed `XAIEMLGBL_*` defines in `xaiemlgbl_params.h` at the same
offsets. The driver routines below implement -- as masked register writes -- the
same protocols these tests issue directly as `aiex.npu.write32` /
`aiex.npu.push_queue`.

| Protocol (test) | Register define -- `xaie2pgbl_params.h` / `xaiemlgbl_params.h` | Driver routine |
|-----------------|----------------------------------------|----------------|
| Core reset -> unreset -> enable | `..._CORE_MODULE_CORE_CONTROL` (`0x32000`), `_RESET_LSB`=1, `_ENABLE_LSB`=0 | `XAie_CoreReset` / `XAie_CoreUnreset` / `XAie_CoreEnable` -- `driver/src/core/xaie_core.c` |
| DMA channel reset | `..._MEMORY_MODULE_DMA_MM2S_0_CTRL` (`0x1DE10`), `_RESET_LSB`=1, `_ENABLE_LSB`=0 | `XAie_DmaChannelReset` -- `driver/src/dma/xaie_dma.c` |
| Re-push BD to start queue | `..._MEMORY_MODULE_DMA_MM2S_0_START_QUEUE` (`0x1DE14`) | `XAie_DmaChannelPushBdToQueue` -- `driver/src/dma/xaie_dma.c` |
| Stream-switch slave port enable | `..._CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_DMA_0` (`0x3F104`), `_SLAVE_ENABLE_LSB`=31 | `XAie_StrmConnCctEnable` / `XAie_StrmConnCctDisable` -- `driver/src/stream_switch/xaie_ss.c` |
| Lock re-arm (set value) | `..._MEMORY_MODULE_LOCK0_VALUE` (`0x1F000`) | `XAie_LockSetValue` -- `driver/src/locks/xaie_locks.c` |

`XAie_DmaChannelReset` masks in only the channel `Reset` bit, matching the
`disable -> assert reset -> deassert -> enable` sequence the `dma` test performs
explicitly; `XAie_CoreReset`/`Unreset`/`Enable` are the per-bit equivalents of
the `core` test's three writes.
