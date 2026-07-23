<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Core reset (op-based)

The op-based analog of [`../core`](../core/README.md): identical design, but the
core is reset with the merged `aiex.core_reset` runtime-sequence op
([#3375](https://github.com/Xilinx/mlir-aie/pull/3375)) rather than issuing the
register writes directly as `../core` does. See [`../README.md`](../README.md) for
the shared design and how to run.

The core on tile `(0, 2)` runs once, fills the output buffer with a data-memory
counter, increments it, then halts at `aie.end`. After batch 1,
`aiex.core_reset(%tile_0_2)` clears the program counter (data memory survives) and
a masked re-enable restarts the kernel, so the host sees `batch2 == batch1 + 1`
-- the same oracle as `../core`, whose `test.cpp` this test compiles directly
(`%S/../core/test.cpp`) rather than duplicating.

## Same protocol as `../core`

`aiex.core_reset` lowers to a **mask-preserving reset pulse** on the same register
`../core` writes -- `CORE_CONTROL` (tile-local `0x32000`), reset bit 1 -- emitted
as two `aiex.npu.maskwrite32`s (assert bit 1 with mask `0x2`, then clear it). This
is the `reset -> unreset` half of `../core`'s raw sequence, and it mirrors aie-rt's
`XAie_CoreReset` / `XAie_CoreUnreset`.

The op is **reset-only** (`XAie_CoreReset` + `XAie_CoreUnreset`): masking to bit 1
preserves the `ENABLE` field but the op does not set it, because by design it
assumes the core is still enabled (re-arming a resident kernel across dispatches).
This core has run to `aie.end` and is no longer enabled, so `aiex.core_reset` alone
does not restart it. The test therefore composes the op with a **masked** re-enable
of `CORE_CONTROL` bit 0 (`aiex.npu.maskwrite32`, mask `0x1`), mirroring aie-rt's
`XAie_CoreEnable` (a `MaskWrite32` of the enable field). Op + this write is the full
driver `XAie_CoreReset -> XAie_CoreUnreset -> XAie_CoreEnable` sequence, every write
masked to a single field so no other `CORE_CONTROL` bit is clobbered.

## Behaviour

- **Correct protocol (this test):** `aiex.core_reset` + re-enable re-runs the core
  -> `batch2 == batch1 + 1`.
- **`aiex.core_reset` alone (no re-enable):** the core stays halted (`ENABLE`
  already clear after `aie.end`), the second batch never arrives, and the kernel
  does not complete. Reproduce by removing the re-enable `maskwrite32` from
  `aie.mlir`.
- **No reset at all:** likewise hangs; the reset pulse is what clears the PC so the
  re-enabled core re-runs from the top.

## Reference

`aiex.core_reset` is defined in `include/aie/Dialect/AIEX/IR/AIEX.td` and lowered
by `lib/Dialect/AIEX/Transforms/AIELowerCoreReset.cpp` (pass
`--aie-lower-core-reset`, run in the default `aiecc` pipeline). It mirrors the
public aie-rt driver (<https://github.com/Xilinx/aie-rt>, vendored at
`third_party/aie-rt/`): `XAIE2PGBL_CORE_MODULE_CORE_CONTROL` in
`driver/src/global/xaie2pgbl_params.h`, and `XAie_CoreReset` / `XAie_CoreUnreset`
in `driver/src/core/xaie_core.c`. The test's masked re-enable mirrors
`XAie_CoreEnable` (same file; `MaskWrite32` of `CORE_CONTROL` `ENABLE`, bit 0). See
[`../README.md`](../README.md#references) for the full table.
