//===- kernel.cc - core-side AIE-ML DMA compression toggles ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
//
// Peano-side companion to test/npu-xrt/tile_mapped_read/kernel.cpp (PR #2348,
// chess-only `chess_storage(TM:...)`). Issue #2346 asks for first-class
// support for "core reads/writes its own DMA registers at runtime"; this
// kernel exercises the write side using peano's aiev2intrin.h primitives.
//
// On Phoenix (AIE-ML / aie2), enabling DMA compression on a given direction
// requires TWO register writes per direction:
//   1) per-BD Enable_Compression bit in MEMORY_MODULE_DMA_BD?_1 (bit 31)
//   2) per-channel (De)compression_Enable bit in
//      MEMORY_MODULE_DMA_{S2MM,MM2S}_0_CTRL (bit 4)
//
// `write_tm` from <aiev2intrin.h> emits an `st.tm` instruction that targets
// the tile-mapped processor-bus address space (default
// TMAddrSpaceStart=0x80000).
//
// `__builtin_aiev2_sched_barrier()` separates back-to-back `st.tm` operations
// so the peano backend does not pack them into one VLIW bundle. The hazard
// without the barrier was documented by @joeldushouyu on issue #2346
// (2025-05-31): merged stores can cause the second store to issue before the
// first completes and stall the kernel. Verifiable with `llvm-objdump -d
// kernel.o`: with barriers each `st.tm` lands in its own bundle; without
// barriers `mova` and `st.tm` get bundled together.

#include <aiev2intrin.h>
#include <cstdint>

static constexpr uint32_t BD1_BASE = 0x1D004;
static constexpr uint32_t BD_STRIDE = 0x20;
static constexpr uint32_t S2MM_0_CTRL = 0x1DE00;
static constexpr uint32_t MM2S_0_CTRL = 0x1DE10;
static constexpr uint32_t COMPRESS_BIT = 0x80000000; // BD?_1 bit 31
static constexpr uint32_t CHAN_BIT = 0x10;           // *_CTRL bit 4

extern "C" {

void enable_mm2s_compression() {
  write_tm(COMPRESS_BIT, BD1_BASE + 2 * BD_STRIDE); // BD2_1
  __builtin_aiev2_sched_barrier();
  write_tm(COMPRESS_BIT, BD1_BASE + 3 * BD_STRIDE); // BD3_1
  __builtin_aiev2_sched_barrier();
  write_tm(CHAN_BIT, MM2S_0_CTRL);
  __builtin_aiev2_sched_barrier();
}

void enable_s2mm_decompression() {
  write_tm(COMPRESS_BIT, BD1_BASE + 0 * BD_STRIDE); // BD0_1
  __builtin_aiev2_sched_barrier();
  write_tm(COMPRESS_BIT, BD1_BASE + 1 * BD_STRIDE); // BD1_1
  __builtin_aiev2_sched_barrier();
  write_tm(CHAN_BIT, S2MM_0_CTRL);
  __builtin_aiev2_sched_barrier();
}

// Negative control: same MM2S-compression sequence without barriers.
// Reproduces the issue #2346 hazard - the peano backend packs the stores
// into a single bundle (verified via llvm-objdump). On silicon this may
// stall the kernel.
void enable_mm2s_compression_no_barrier() {
  write_tm(COMPRESS_BIT, BD1_BASE + 2 * BD_STRIDE);
  write_tm(COMPRESS_BIT, BD1_BASE + 3 * BD_STRIDE);
  write_tm(CHAN_BIT, MM2S_0_CTRL);
}

} // extern "C"
