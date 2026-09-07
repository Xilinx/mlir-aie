<!---//===- README.md ---------------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Out-of-order S2MM shim receiver (runtime bd_id)

The [`dma_s2mm_ooo`](../dma_s2mm_ooo/README.md) merge with the receiver moved to a
**shim** tile and its receive BDs' `bd_id`s drawn from the dynamic free-list pool
at **runtime**. Senders, header-id placement, and the rotation proof are the same;
this note states only what differs.

## What differs from the base

- **Shim receiver, straight to host DDR.** The out-of-order S2MM channel is on the
  shim tile, so each packet lands directly in the host output buffer (a slice of
  the runtime argument, address-patched on the dynamic path) rather than an on-chip
  buffer drained by an MM2S egress.
- **Runtime pool `bd_id`.** Each receive BD's `bd_id` is a runtime value from
  `aiex.dma_bd_pool_pop`, which forces the shim-NOC dynamic BD path (it cannot fold
  into a static `insts.bin`).
- **Single-BD per task.** The dynamic-pool lowering emits one BD per
  `dma_configure_task` -- a runtime-bd_id `next_bd` chain is not yet supported -- so
  each receive BD is its own single-BD out-of-order task on the one channel.
- **Companion-token completion.** With no on-chip buffer to drain, the base's MM2S
  egress is gone. An out-of-order channel cannot issue a completion token, so a
  **companion in-order** shim S2MM BD acquires the counting lock once all packets
  land, consumes a sentinel packet, and issues the token the sequence `dma_await`s.

## Test

`aie.mlir` / `test.cpp` -- two senders merge into one out-of-order channel; the
runtime bd_ids map to slots in reverse of configuration order, so a correct result
is only possible if placement follows the header id, not arrival or configuration
order.

## Compiling

The runtime bd_id cannot fold into a static `insts.bin`, so the design uses the
dynamic-pool recipe (see `dynamic_pingpong_passthrough`): `aiecc` builds the xclbin
(device structure) and `aie-translate --aie-npu-to-cpp` generates a C++ TXN builder
that draws the bd_ids from the pool at host runtime. The host `#include`s that
builder and runs it via XRT.

## TODO: fold into `dma_s2mm_ooo.py`

Hand-rolled MLIR + C++ rather than an IRON `.py` because the DSL cannot yet express
a shim receiver: `Buffer` only emits tile-SRAM `aie.buffer` (not the host
`aie.external_buffer` a shim S2MM drains to), and `iron.jit` bakes a static
instruction stream (not the runtime-built one a pool bd_id needs). Once IRON gains
host/external receive buffers and a runtime-instruction path, this collapses into a
`--recv-tile shim` axis on `dma_s2mm_ooo.py`, unifying the out-of-order e2e.
