//===- kernel.cc - core-side DMA compression toggles + regdump -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
//
// Peano-side companion to test/npu-xrt/tile_mapped_read/ (PR #2348). Closes
// the write/read sides of issue #2346 via `write_tm` / `read_tm`.
//
// SCHED_BARRIER() prevents peano from packing back-to-back st.tm/lda.tm into
// one VLIW bundle — the hazard documented on issue #2346 (2025-05-31).

#if defined(__AIE2P__)
#include <aie2pintrin.h>
#define SCHED_BARRIER() __builtin_aie2p_sched_barrier()
#else
#include <aiev2intrin.h>
#define SCHED_BARRIER() __builtin_aiev2_sched_barrier()
#endif
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
  SCHED_BARRIER();
  write_tm(COMPRESS_BIT, BD1_BASE + 3 * BD_STRIDE); // BD3_1
  SCHED_BARRIER();
  write_tm(CHAN_BIT, MM2S_0_CTRL);
  SCHED_BARRIER();
}

void enable_s2mm_decompression() {
  write_tm(COMPRESS_BIT, BD1_BASE + 0 * BD_STRIDE); // BD0_1
  SCHED_BARRIER();
  write_tm(COMPRESS_BIT, BD1_BASE + 1 * BD_STRIDE); // BD1_1
  SCHED_BARRIER();
  write_tm(CHAN_BIT, S2MM_0_CTRL);
  SCHED_BARRIER();
}

// write_tm + read_tm self-test for the `regdump` config.
//   buf[0..5]  initial BD0/1/2/3_1, S2MM_CTRL, MM2S_CTRL  (informational)
//   buf[6..9]  BD0/1/2/3_1 after write_tm(COMPRESS_BIT)   (each must ==
//   COMPRESS_BIT)
// CTRLs are not clobbered (write_tm is a full-register write and CTRL holds
// CONTROLLER_ID etc. that the IRON-configured channel relies on).
void dump_compress_regs(uint32_t *buf) {
  for (int i = 10; i < 1024; i++)
    buf[i] = 0;

  buf[0] = read_tm(BD1_BASE + 0 * BD_STRIDE);
  SCHED_BARRIER();
  buf[1] = read_tm(BD1_BASE + 1 * BD_STRIDE);
  SCHED_BARRIER();
  buf[2] = read_tm(BD1_BASE + 2 * BD_STRIDE);
  SCHED_BARRIER();
  buf[3] = read_tm(BD1_BASE + 3 * BD_STRIDE);
  SCHED_BARRIER();
  buf[4] = read_tm(S2MM_0_CTRL);
  SCHED_BARRIER();
  buf[5] = read_tm(MM2S_0_CTRL);
  SCHED_BARRIER();

  for (int i = 0; i < 4; i++) {
    write_tm(COMPRESS_BIT, BD1_BASE + i * BD_STRIDE);
    SCHED_BARRIER();
    buf[6 + i] = read_tm(BD1_BASE + i * BD_STRIDE);
    SCHED_BARRIER();
  }
}

// Negative control: same as enable_mm2s_compression without barriers.
// Reproduces the issue #2346 VLIW-bundle hazard (visible via llvm-objdump).
// Not wired into any runtime config.
void enable_mm2s_compression_no_barrier() {
  write_tm(COMPRESS_BIT, BD1_BASE + 2 * BD_STRIDE);
  write_tm(COMPRESS_BIT, BD1_BASE + 3 * BD_STRIDE);
  write_tm(CHAN_BIT, MM2S_0_CTRL);
}

} // extern "C"
