//===- lut_banks_same_bank_kernel.cc ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Support file for lut_banks_same_bank.mlir and lut_banks_no_ir.mlir. Two
// file-scope tables carrying no bank annotation at all, which is how the
// in-tree LUTs are written: the linker packs them adjacently, into one bank.

#include <aie_api/aie.hpp>
#include <stdint.h>

alignas(32) int16 tbl_ab[512];
alignas(32) int16 tbl_cd[512];

extern "C" void classify(uint8_t *out) {
  using lut_t = aie::lut<4, bfloat16, bfloat16>;
  lut_t l(256, (bfloat16 *)tbl_ab, (bfloat16 *)tbl_cd);
  aie::parallel_lookup<uint16, lut_t, aie::lut_oor_policy::truncate> lk(l, 0);
  aie::vector<int16, 16> idx = aie::load_v<16>((int16 *)out);
  *(v16bfloat16 *)out = lk.fetch(idx.cast_to<uint16>());
}
