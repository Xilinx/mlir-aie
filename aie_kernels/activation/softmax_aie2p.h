//===- softmax_aie2p.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <limits>
#include <stdint.h>

#include "../common/exp2_bf16.h"

#define SM_VEC_LEN 32   // 32
#define log2e 1.4453125 // 1.44269504089

using namespace aie;

// x * log2(e) in f32. Under EXP2_BF16_ACCURATE log2(e) is three bf16 limbs,
// to 2^-26, summed smallest product first; otherwise it is bf16(log2(e)),
// 1.8e-3 low. A row's maximum is scaled through this too, so its own exponent
// argument is exactly 0.
static inline aie::accum<accfloat, SM_VEC_LEN>
scale_log2e(aie::vector<bfloat16, SM_VEC_LEN> x) {
#ifdef EXP2_BF16_ACCURATE
  auto limb = [](int16_t bits) {
    return aie::broadcast<int16_t, SM_VEC_LEN>(bits).cast_to<bfloat16>();
  };
  aie::accum<accfloat, SM_VEC_LEN> t = aie::mul(x, limb(0x36ec));
  t = aie::mac(t, x, limb(0xbb2c));
  return aie::mac(t, x, limb(0x3fb9));
#else
  return aie::mul(x, aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)log2e));
#endif
}

void softmax_simple_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         const int32_t vector_size) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);

  // VJUNG: We do 3 passes on the vector:
  // 1. Find the max value scaled by log2e in the vector
  // 2. Calculate the exponentials of the scaled values minus the maximum
  // 3. Calculate the softmax by dividing each exponential by the sum of all
  // exponentials Note: The multiplication by log2e is very sensitive, casting
  // it to bf16 before exponentiation leads to wrong output.

  auto it_log_in =
      aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_log_out =
      aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_in =
      aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_out =
      aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
  auto it_scale =
      aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
  auto it_soft_out =
      aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);

  aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, max_val_vec;
  aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, exp_in_accum;

  float accum_exp_val = 0;
  bfloat16 col_sum_inv;
  const int elem_iters = (uint32_t)vector_size / SM_VEC_LEN;

  exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();

  // First pass - Optimized: element-wise max + single final reduce_max
  // Use vector max accumulation, then reduce once at the end.
  //
  // Take the max of the raw input, not of the scaled value. Scaling by log2e
  // is monotonic, so the largest element is the same either way, and a bf16
  // max over bf16 inputs is exact -- rounding the scaled value to bf16 first
  // was not. When that rounding landed below the true maximum, the largest
  // element's exponent argument came out positive instead of zero and exp2
  // ran away with it.
  aie::vector<bfloat16, SM_VEC_LEN> max_accum_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>(
          std::numeric_limits<bfloat16>::lowest());
  AIE_LOOP_UNROLL(2)
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    max_accum_vec = aie::max(max_accum_vec, *it_log_in++);
  }
  max_val_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>(aie::reduce_max(max_accum_vec));
  // Scale the maximum through the same multiply its element takes below, so
  // the subtraction cancels exactly there. That is what the whole
  // formulation rests on: the largest exponent argument is exactly 0 and
  // every other one is negative, so exp2 <= 1, the sum is >= 1, and the
  // reciprocal of the sum cannot overflow.
  aie::vector<float, SM_VEC_LEN> max_scaled =
      scale_log2e(max_val_vec).to_vector<float>();

  // Second pass
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {

    input_bf16 = *it_exp_in++;

    exp_in_accum = aie::sub(scale_log2e(input_bf16), max_scaled);
    exp_val = exp2_bf16(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);

    *it_exp_out++ = exp_val;
  }

  // Final reduction after loop
  aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);
  col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int c = 0; c < elem_iters; c++) {
    in_elems = *it_scale++;
    out_vals = aie::mul(in_elems, col_sum_inv);
    *it_soft_out++ = out_vals.to_vector<bfloat16>();
  }

  ::aie::set_rounding(saved_rounding);

  return;
}

// Online (flash-attention) partial softmax over one key-block row.  Unlike
// softmax_simple_bf16 this does NOT normalize: it applies the running-max
// rescale used by streaming attention and stashes the block's new max and
// exp-sum into scale_buffer so the caller (see ../linalg/mha.cc) can combine
// blocks.  scale_buffer layout, indexed by row: [0*num_rows + r] = prev max
// m_{i-1}; [1*num_rows + r] = new max m_i (written here); [3*num_rows + r] =
// this block's exp-sum l_i (written here).  `scale` is the log2-domain factor
// (1/sqrt(d) folded with log2e) broadcast in place of the plain log2e.
void partial_softmax_alias_bf16(bfloat16 *restrict input_vector,
                                bfloat16 *restrict output_vector,
                                float *restrict scale_buffer,
                                const int32_t vector_size,
                                const int32_t row_idx, const int32_t num_rows,
                                const bfloat16 scale) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);

  auto it_log_in =
      aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_in =
      aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_out =
      aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);

  aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec,
      max_val_vec;
  aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, scaled_accum,
      exp_in_accum;

  float accum_exp_val = 0;
  const int elem_iters = (uint32_t)vector_size / SM_VEC_LEN;

  exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();

  log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)scale);

  // First pass - running max over the block, reduced once at the end: a
  // scalar float compare is a libcall.
  aie::vector<bfloat16, SM_VEC_LEN> max_accum_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>(
          std::numeric_limits<bfloat16>::lowest());
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    max_accum_vec = aie::max(
        max_accum_vec, aie::mul(*it_log_in++, log2e_vec).to_vector<bfloat16>());
  }
  bfloat16 max_val = aie::reduce_max(max_accum_vec);

  // Compute m_{i}: max of this block and the carried-in running max.
  max_val = aie::max(max_val, (bfloat16)scale_buffer[row_idx]);
  scale_buffer[num_rows + row_idx] = (float)max_val;

  max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(max_val);

  // Second pass - unnormalized exponentials, accumulating the block sum.
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_exp_in++;
    scaled_accum = aie::mul(input_bf16, log2e_vec);
    exp_in_accum = aie::sub(scaled_accum, max_val_vec);
    exp_val = exp2_bf16(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);
    *it_exp_out++ = exp_val;
  }

  aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);

  scale_buffer[3 * num_rows + row_idx] = accum_exp_val;

  ::aie::set_rounding(saved_rounding);

  return;
}

// The three passes of softmax_simple_bf16 over a row split into blocks, one
// call per block and pass. Between calls state[0, SM_VEC_LEN) holds the
// per-lane exp sums and state[SM_VEC_LEN] the row maximum, so every lane sums
// its elements in the same order and the row rounds exactly as one call
// would.
void softmax_max_block(bfloat16 *restrict input_vector, float *restrict state,
                       const int32_t first, const int32_t vector_size) {
  auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>(input_vector);
  const int elem_iters = (uint32_t)vector_size / SM_VEC_LEN;

  aie::vector<bfloat16, SM_VEC_LEN> max_accum_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>(
          first ? std::numeric_limits<bfloat16>::lowest()
                : (bfloat16)state[SM_VEC_LEN]);
  if (first)
    aie::store_v(state, aie::zeros<float, SM_VEC_LEN>());

  AIE_LOOP_UNROLL(2)
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    max_accum_vec = aie::max(max_accum_vec, *it_in++);
  }
  state[SM_VEC_LEN] = (float)aie::reduce_max(max_accum_vec);
}

void softmax_sum_block(bfloat16 *restrict input_vector, float *restrict state,
                       const int32_t vector_size) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);

  auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>(input_vector);
  const int elem_iters = (uint32_t)vector_size / SM_VEC_LEN;

  aie::vector<float, SM_VEC_LEN> max_scaled =
      scale_log2e(
          aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)state[SM_VEC_LEN]))
          .to_vector<float>();

  aie::accum<accfloat, SM_VEC_LEN> exp_val_accum;
  exp_val_accum.from_vector(aie::load_v<SM_VEC_LEN>(state));

  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    aie::accum<accfloat, SM_VEC_LEN> exp_in_accum =
        aie::sub(scale_log2e(*it_in++), max_scaled);
    aie::vector<bfloat16, SM_VEC_LEN> exp_val =
        exp2_bf16(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);
  }
  aie::store_v(state, exp_val_accum.to_vector<float>());

  ::aie::set_rounding(saved_rounding);
}

void softmax_scale_block(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         float *restrict state, const int32_t vector_size) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);

  auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>(input_vector);
  auto it_out = aie::begin_restrict_vector<SM_VEC_LEN>(output_vector);
  const int elem_iters = (uint32_t)vector_size / SM_VEC_LEN;

  aie::vector<float, SM_VEC_LEN> max_scaled =
      scale_log2e(
          aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)state[SM_VEC_LEN]))
          .to_vector<float>();

  aie::vector<float, SM_VEC_LEN> reduce = aie::load_v<SM_VEC_LEN>(state);
  float accum_exp_val = aie::reduce_add(reduce);
  bfloat16 col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    aie::accum<accfloat, SM_VEC_LEN> exp_in_accum =
        aie::sub(scale_log2e(*it_in++), max_scaled);
    aie::vector<bfloat16, SM_VEC_LEN> exp_val =
        exp2_bf16(exp_in_accum.to_vector<float>());
    *it_out++ = aie::mul(exp_val, col_sum_inv).to_vector<bfloat16>();
  }

  ::aie::set_rounding(saved_rounding);
}

extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                  const int32_t input_size) {
  event0();
  softmax_simple_bf16(input, output, input_size);
  event1();
}

void partial_softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                          float *restrict scale_buffer,
                          const int32_t input_size, const int32_t row_idx,
                          const int32_t num_rows, const bfloat16 scale) {
  // Not in partial_softmax_alias_bf16, so mha.cc can time a whole block.
  event0();
  partial_softmax_alias_bf16(input, output, scale_buffer, input_size, row_idx,
                             num_rows, scale);
  event1();
}

void softmax_max_bf16(bfloat16 *restrict input, float *restrict state,
                      const int32_t first, const int32_t input_size) {
  event0();
  softmax_max_block(input, state, first, input_size);
  event1();
}

void softmax_sum_bf16(bfloat16 *restrict input, float *restrict state,
                      const int32_t input_size) {
  event0();
  softmax_sum_block(input, state, input_size);
  event1();
}

void softmax_scale_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                        float *restrict state, const int32_t input_size) {
  event0();
  softmax_scale_block(input, output, state, input_size);
  event1();
}

// Fill [unmasked_size, total_size) with -inf so a subsequent softmax drops the
// masked tail (causal / padding mask).
void mask_bf16(bfloat16 *inout, const int32_t unmasked_size,
               const int32_t total_size) {
  for (int32_t i = unmasked_size; i < total_size; i++) {
    inout[i] = std::numeric_limits<bfloat16>::lowest();
  }
}

// softmax_bf16 over `rows` rows of `row_len` each.
void softmax_rows_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                       const int32_t rows, const int32_t row_len) {
  event0();
  for (int32_t r = 0; r < rows; r++) {
    softmax_simple_bf16(input + r * row_len, output + r * row_len, row_len);
  }
  event1();
}

// softmax_rows_bf16 under a causal mask, the first row being query row_offset.
void softmax_rows_causal_bf16(bfloat16 *restrict input,
                              bfloat16 *restrict output, const int32_t rows,
                              const int32_t row_len, const int32_t row_offset) {
  event0();
  for (int32_t r = 0; r < rows; r++) {
    mask_bf16(input + r * row_len, row_offset + r + 1, row_len);
    softmax_simple_bf16(input + r * row_len, output + r * row_len, row_len);
  }
  event1();
}

} // extern "C"
