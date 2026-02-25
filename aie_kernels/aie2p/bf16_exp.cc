//===- bf16_exp.cc ---------------------------*- C++-----*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===-----------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#define VEC_LEN 16
#define log2e 1.44269504089

using namespace aie;

template <const int N>
void exp_bf16_func(bfloat16 *restrict in, bfloat16 *restrict out) {

  auto it_exp_in = aie::cbegin_vector<VEC_LEN>((bfloat16 *)in);
  auto it_exp_out = aie::begin_vector<VEC_LEN>((bfloat16 *)out);

  const int elem_iters = N / VEC_LEN;

  // Calculate the e^(x) function as 2^(log2e * x)
  aie::vector<bfloat16, VEC_LEN> input_bf16;
  aie::accum<accfloat, VEC_LEN> exp_in;
  aie::vector<bfloat16, VEC_LEN> exp_val;
  aie::vector<bfloat16, VEC_LEN> log2e_vec =
      aie::broadcast<bfloat16, VEC_LEN>(log2e);

  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_exp_in++;
    exp_in = aie::mul(input_bf16, log2e_vec);
    exp_val = aie::exp2<bfloat16>(exp_in.to_vector<float>());
    *it_exp_out++ = exp_val;
  }
}

extern "C" {

void exp_bf16_1024(bfloat16 *a_in, bfloat16 *c_out) {
  exp_bf16_func<1024>(a_in, c_out);
}

} // extern "C"
