//===- softmax_aie2.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <limits>
#include <lut_based_ops.h>
#include <stdint.h>

using namespace aie;

template <int MinIters>
void softmax_simple_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         const int32_t vector_size) {
  event0();

  int num_elems = vector_size;
  float accum_exp_val;
  auto it_max_in = aie::cbegin_restrict_vector<16>((bfloat16 *)input_vector);
  auto it_exp_in = aie::cbegin_restrict_vector<16>((bfloat16 *)input_vector);
  auto it_exp_out = aie::begin_restrict_vector<16>((bfloat16 *)output_vector);
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

  // Subtract the tile maximum before exponentiating, so the largest exponent
  // argument is exactly 0 and every other one is negative: exp <= 1, the sum
  // lies in [1, vector_size], and its reciprocal stays finite and nonzero.
  // Without this, large inputs saturate exp and the whole tile normalises to
  // zero.
  aie::vector<bfloat16, 16> max_accum_vec =
      aie::broadcast<bfloat16, 16>(std::numeric_limits<bfloat16>::lowest());
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(MinIters)
  for (int i = 0; i < elem_iters; i++) {
    max_accum_vec = aie::max(max_accum_vec, *it_max_in++);
  }
  aie::vector<bfloat16, 16> max_val_vec =
      aie::broadcast<bfloat16, 16>(aie::reduce_max(max_accum_vec));

  // Rotated by one, as in exp_bf16_func: see bf16_exp_aie2.h.
  aie::vector<bfloat16, 16> prev =
      to_v16bfloat16(getExpBf16(aie::sub(*it_exp_in++, max_val_vec)));
  exp_val_accum = add(exp_val_accum, prev);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(MinIters)
  for (int i = 1; i < elem_iters; i++) {
    exp_val = to_v16bfloat16(getExpBf16(aie::sub(*it_exp_in++, max_val_vec)));
    exp_val_accum = add(exp_val_accum, exp_val);
    *it_exp_out++ = prev;
    prev = exp_val;
  }
  *it_exp_out++ = prev;
  aie::vector<float, 16> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);
  /////////////////////

  col_sum_inv = (bfloat16)aie::inv(accum_exp_val);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(MinIters)
  for (int c = 0; c < col_iters; c++) {
    in_elems = *it_scale++;
    out_vals = aie::mul(in_elems, col_sum_inv);
    *it_soft_out++ = out_vals.to_vector<bfloat16>();
  }

  event1();

  return;
}

// The three passes of softmax_simple_bf16 over a row split into blocks, one
// call per block and pass. Between calls state[0, 16) holds the per-lane exp
// sums and state[16] the row maximum, so every lane sums its elements in the
// same order and the row rounds exactly as one call would.
template <int MinIters>
void softmax_max_block(bfloat16 *restrict input_vector, float *restrict state,
                       const int32_t first, const int32_t vector_size) {
  auto it_in = aie::cbegin_restrict_vector<16>(input_vector);
  const int elem_iters = vector_size / 16;

  aie::vector<bfloat16, 16> max_accum_vec = aie::broadcast<bfloat16, 16>(
      first ? std::numeric_limits<bfloat16>::lowest() : (bfloat16)state[16]);
  if (first)
    aie::store_v(state, aie::zeros<float, 16>());

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(MinIters)
  for (int i = 0; i < elem_iters; i++) {
    max_accum_vec = aie::max(max_accum_vec, *it_in++);
  }
  state[16] = (float)aie::reduce_max(max_accum_vec);
}

template <int MinIters>
void softmax_sum_block(bfloat16 *restrict input_vector, float *restrict state,
                       const int32_t vector_size) {
  auto it_in = aie::cbegin_restrict_vector<16>(input_vector);
  const int elem_iters = vector_size / 16;

  aie::vector<bfloat16, 16> max_val_vec =
      aie::broadcast<bfloat16, 16>((bfloat16)state[16]);
  aie::accum<accfloat, 16> exp_val_accum;
  exp_val_accum.from_vector(aie::load_v<16>(state));

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(MinIters)
  for (int i = 0; i < elem_iters; i++) {
    aie::vector<bfloat16, 16> exp_val =
        to_v16bfloat16(getExpBf16(aie::sub(*it_in++, max_val_vec)));
    exp_val_accum = add(exp_val_accum, exp_val);
  }
  aie::store_v(state, exp_val_accum.to_vector<float>());
}

template <int MinIters>
void softmax_scale_block(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         float *restrict state, const int32_t vector_size) {
  auto it_in = aie::cbegin_restrict_vector<16>(input_vector);
  auto it_out = aie::begin_restrict_vector<16>(output_vector);
  const int elem_iters = vector_size / 16;

  aie::vector<bfloat16, 16> max_val_vec =
      aie::broadcast<bfloat16, 16>((bfloat16)state[16]);
  aie::vector<float, 16> reduce = aie::load_v<16>(state);
  bfloat16 col_sum_inv = (bfloat16)aie::inv(aie::reduce_add(reduce));

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(MinIters)
  for (int i = 0; i < elem_iters; i++) {
    aie::vector<bfloat16, 16> exp_val =
        to_v16bfloat16(getExpBf16(aie::sub(*it_in++, max_val_vec)));
    *it_out++ = aie::mul(exp_val, col_sum_inv).to_vector<bfloat16>();
  }
}

extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                  const int32_t input_size) {
  // The pipelined loops need 8 trips; the exp loop runs one fewer than the
  // others.
  if (input_size >= 16 * 9)
    softmax_simple_bf16<8>(input, output, input_size);
  else
    softmax_simple_bf16<1>(input, output, input_size);
}

void softmax_max_bf16(bfloat16 *restrict input, float *restrict state,
                      const int32_t first, const int32_t input_size) {
  event0();
  if (input_size >= 16 * 8)
    softmax_max_block<8>(input, state, first, input_size);
  else
    softmax_max_block<1>(input, state, first, input_size);
  event1();
}

void softmax_sum_bf16(bfloat16 *restrict input, float *restrict state,
                      const int32_t input_size) {
  event0();
  if (input_size >= 16 * 8)
    softmax_sum_block<8>(input, state, input_size);
  else
    softmax_sum_block<1>(input, state, input_size);
  event1();
}

void softmax_scale_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                        float *restrict state, const int32_t input_size) {
  event0();
  if (input_size >= 16 * 8)
    softmax_scale_block<8>(input, output, state, input_size);
  else
    softmax_scale_block<1>(input, output, state, input_size);
  event1();
}

// Fill [unmasked_size, total_size) with -inf, so a following softmax zeros
// those positions (causal / padding mask).
void mask_bf16(bfloat16 *inout, const int32_t unmasked_size,
               const int32_t total_size) {
  for (int32_t i = unmasked_size; i < total_size; i++) {
    inout[i] = (bfloat16)(-INFINITY);
  }
}

} // extern "C"
