<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Core reset

Restarting a finite core kernel from a clean program counter by driving its
`Core_Control` register (tile-local offset `0x32000`) with masked `maskwrite32`s,
one per field, exactly as the aie-rt driver does. Bit 0 is `Enable`, bit 1 is
`Reset`; the offset and bit layout are identical on AIE-ML (npu1) and AIE2P (npu2),
so this family runs on both. See [`../README.md`](../README.md) for the shared
design and how to run.

The core on tile `(0, 2)` runs once, fills the output buffer with a data-memory
counter, increments the counter, then halts at `aie.end`. Data memory survives a
reset, but the program counter does not -- so a restart re-runs the kernel and
emits `counter + 1`. The host collects two batches (before/after the restart) and
checks `batch2 == batch1 + 1`.

The reset issues `reset -> unreset -> enable` after batch 1, each a masked write of
a single `Core_Control` field (mirroring `XAie_CoreReset` / `XAie_CoreUnreset` /
`XAie_CoreEnable`). These writes only take effect on a **settled** core (the kernel
has reached `aie.end`); they are issued between two completed transfers, not to
preempt a running core.

## Behaviour

- **Correct protocol (this test):** `reset -> unreset -> enable` re-runs the core ->
  `batch2 == batch1 + 1`.
- **No reset / reset without enable:** the core stays `Done`/halted, so the second
  batch never arrives and the collect never completes. `Enable` (bit 0) is what
  re-runs a settled core; the reset bit clears state but does not restart on its
  own. This mode hangs by design -- reproduce it by hand by editing the reset
  sequence in `aie.mlir`.

## Reference

`Core_Control` and its `Reset`/`Enable` bits, and the reset/unreset/enable
procedure, are defined in the public aie-rt driver
(<https://github.com/Xilinx/aie-rt>, vendored at `third_party/aie-rt/`):
`XAIE2PGBL_CORE_MODULE_CORE_CONTROL` in `driver/src/global/xaie2pgbl_params.h`,
and `XAie_CoreReset` / `XAie_CoreUnreset` / `XAie_CoreEnable` in
`driver/src/core/xaie_core.c` (each a `MaskWrite32` of a single `Core_Control`
field, as this test issues them). See [`../README.md`](../README.md#references) for
the full table.
