//===- softmax.cc --------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===-------------------------------------------------- --------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

using namespace aie;

void softmax_simple_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         const int32_t vector_size) {
  event0();

  int num_elems = vector_size;
  float accum_exp_val;
  auto it_exp_in = aie::cbegin_vector<16>((bfloat16 *)input_vector);
  auto it_exp_out = aie::begin_vector<16>((bfloat16 *)output_vector);
  auto it_scale = aie::cbegin_restrict_vector<16>((bfloat16 *)output_vector);
  auto it_soft_out = aie::begin_restrict_vector<16>((bfloat16 *)output_vector);

  bfloat16 col_sum_inv;
  aie::vector<bfloat16, 16> in_elems, va;
  aie::accum<accfloat, 16> out_vals;
  int col_iters = num_elems >> 4;
  accum_exp_val = 0;

  /////////////////////
  //// Compute exp ////
  /////////////////////
  aie::vector<bfloat16, 16> exp_val;
  aie::vector<float, 16> input_fp32;

  const int elem_iters = num_elems / 16;
  aie::vector<bfloat16, 16> input_bf16;
  aie::accum<accfloat, 16> exp_val_accum;
  exp_val_accum = aie::zeros<accfloat, 16>();
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_exp_in++;
    exp_val = to_v16bfloat16(getExpBf16(input_bf16));
    exp_val_accum = add(exp_val_accum, exp_val);
    *it_exp_out++ = exp_val;
  }
  aie::vector<float, 16> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);
  /////////////////////

  col_sum_inv = (bfloat16)aie::inv(accum_exp_val);
  for (int c = 0; c < col_iters; c++) {
    in_elems = *it_scale++;
    out_vals = aie::mul(in_elems, col_sum_inv);
    *it_soft_out++ = out_vals.to_vector<bfloat16>();
  }

  event1();

  return;
}

extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                  const int32_t input_size) {
  softmax_simple_bf16(input, output, input_size);
}

} // extern "C"
