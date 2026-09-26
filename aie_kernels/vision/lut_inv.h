/* Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

#ifndef X_LUT_INV
#define X_LUT_INV

#include "../aie_arch.h"
#include "aie_bank_placement.h"

constexpr uint16 num_entries_lut_inv_16b = 256;

// aie::lut<4> needs each bank-width run of entries stored twice in a row: 128
// bits (8 entries) on AIE2, 256 bits (16) on AIE2P. Laid out for the wrong one,
// half the lanes read the neighbouring run.
constexpr unsigned lut_inv_16b_run = AIE_LUT_16B_RUN;

// Where entry d sits in either table.
constexpr unsigned lut_inv_16b_index(unsigned d) {
  return (d / lut_inv_16b_run) * (2 * lut_inv_16b_run) + d % lut_inv_16b_run;
}

// Q7.9 data format (inv * 85): entry d is (85 * 512) / d, and entry 0 repeats
// entry 1.
constexpr std::array<uint16, 2 * num_entries_lut_inv_16b> make_lut_inv_16b() {
  std::array<uint16, 2 * num_entries_lut_inv_16b> t{};
  for (unsigned d = 0; d < num_entries_lut_inv_16b; d++) {
    uint16 inv = (85 * 512) / (d ? d : 1);
    t[lut_inv_16b_index(d)] = inv;
    t[lut_inv_16b_index(d) + lut_inv_16b_run] = inv;
  }
  return t;
}

AIE_BANK_A alignas(aie::vector_decl_align)
    const std::array<uint16, 2 * num_entries_lut_inv_16b> lut_inv_16b_ab =
        make_lut_inv_16b();
AIE_BANK_B alignas(aie::vector_decl_align)
    const std::array<uint16, 2 * num_entries_lut_inv_16b> lut_inv_16b_cd =
        make_lut_inv_16b();

#endif
