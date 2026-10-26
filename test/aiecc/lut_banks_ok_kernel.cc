//===- lut_banks_ok_kernel.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for lut_banks_ok.mlir. The two tables are pinned to separate
// banks through the sections the linker script routes, which is what the
// gather needs.

#include <aie_api/aie.hpp>
#include <stdint.h>

__attribute__((section(".aie.bank1"), aligned(32))) int16 tbl_ab[512];
__attribute__((section(".aie.bank2"), aligned(32))) int16 tbl_cd[512];

extern "C" void classify(uint8_t *out) {
  using lut_t = aie::lut<4, bfloat16, bfloat16>;
  lut_t l(256, (bfloat16 *)tbl_ab, (bfloat16 *)tbl_cd);
  aie::parallel_lookup<uint16, lut_t, aie::lut_oor_policy::truncate> lk(l, 0);
  aie::vector<int16, 16> idx = aie::load_v<16>((int16 *)out);
  *(v16bfloat16 *)out = lk.fetch(idx.cast_to<uint16>());
}
