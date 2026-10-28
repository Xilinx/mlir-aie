//===- lut_banks_stack_kernel.cc --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for lut_banks_stack.mlir. The two tables are function locals, so
// they live on the stack: one contiguous run, and therefore one bank.

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void classify(uint8_t *out) {
  int16 tbl_ab[512];
  int16 tbl_cd[512];
  for (int i = 0; i < 512; ++i) {
    tbl_ab[i] = i;
    tbl_cd[i] = i;
  }
  using lut_t = aie::lut<4, bfloat16, bfloat16>;
  lut_t l(256, (bfloat16 *)tbl_ab, (bfloat16 *)tbl_cd);
  aie::parallel_lookup<uint16, lut_t, aie::lut_oor_policy::truncate> lk(l, 0);
  aie::vector<int16, 16> idx = aie::load_v<16>((int16 *)out);
  *(v16bfloat16 *)out = lk.fetch(idx.cast_to<uint16>());
}
