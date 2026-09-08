//===- softmax.cc --------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#define SM_VEC_LEN 32   // 32
#define log2e 1.4453125 // 1.44269504089

using namespace aie;

void softmax_simple_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         const int32_t vector_size) {
  event0();

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

  aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, log2e_vec,
      max_val_vec;
  aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, scaled_accum,
      exp_in_accum;

  float max_val = 0;
  float accum_exp_val = 0;
  float running_max = 0;
  bfloat16 col_sum_inv;
  const int elem_iters = vector_size / SM_VEC_LEN;

  exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();

  log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)log2e);

  // First pass - Optimized: element-wise max + single final reduce_max
  // Use vector max accumulation, then reduce once at the end
  aie::vector<bfloat16, SM_VEC_LEN> max_accum_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)-32768.0f);
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_log_in++;
    scaled_accum = aie::mul(input_bf16, log2e_vec);
    max_accum_vec = aie::max(max_accum_vec, scaled_accum.to_vector<bfloat16>());
  }
  max_val = aie::reduce_max(max_accum_vec);
  max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(max_val);

  // Second pass
  for (int i = 0; i < elem_iters; i++) {

    input_bf16 = *it_exp_in++;

    scaled_accum = aie::mul(input_bf16, log2e_vec);
    exp_in_accum = aie::sub(scaled_accum, max_val_vec);
    exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);

    *it_exp_out++ = exp_val;
  }

  // Final reduction after loop
  aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);
  col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

  for (int c = 0; c < elem_iters; c++) {
    in_elems = *it_scale++;
    out_vals = aie::mul(in_elems, col_sum_inv);
    *it_soft_out++ = out_vals.to_vector<bfloat16>();
  }

  event1();

  return;
}

// Online (flash-attention) partial softmax over one key-block row.  Unlike
// softmax_simple_bf16 this does NOT normalize: it applies the running-max
// rescale used by streaming attention and stashes the block's new max and
// exp-sum into scale_buffer so the caller (see aie2p/mha.cc) can combine
// blocks.  scale_buffer layout, indexed by row: [0*num_rows + r] = prev max
// m_{i-1}; [1*num_rows + r] = new max m_i (written here); [3*num_rows + r] =
// this block's exp-sum l_i (written here).  `scale` is the log2-domain factor
// (1/sqrt(d) folded with log2e) broadcast in place of the plain log2e.
void partial_softmax_alias_bf16(bfloat16 *restrict input_vector,
                                bfloat16 *restrict output_vector,
                                bfloat16 *restrict scale_buffer,
                                const int32_t vector_size,
                                const int32_t row_idx, const int32_t num_rows,
                                const bfloat16 scale) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);

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

  float max_val = 0;
  float accum_exp_val = 0;
  float running_max = 0;
  const int elem_iters = vector_size / SM_VEC_LEN;

  exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();

  log2e_vec = aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)scale);

  // First pass - running max over the block.
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_log_in++;
    scaled_accum = aie::mul(input_bf16, log2e_vec);
    running_max = aie::reduce_max(scaled_accum.to_vector<bfloat16>());
    if (running_max > max_val) {
      max_val = running_max;
    }
  }

  // Compute m_{i}: max of this block and the carried-in running max.
  if (max_val > scale_buffer[row_idx]) {
    scale_buffer[num_rows + row_idx] = max_val;
  } else {
    scale_buffer[num_rows + row_idx] = scale_buffer[row_idx];
    max_val = scale_buffer[row_idx];
  }

  max_val_vec = aie::broadcast<bfloat16, SM_VEC_LEN>(max_val);

  // Second pass - unnormalized exponentials, accumulating the block sum.
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_exp_in++;
    scaled_accum = aie::mul(input_bf16, log2e_vec);
    exp_in_accum = aie::sub(scaled_accum, max_val_vec);
    exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);
    *it_exp_out++ = exp_val;
  }

  aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);

  scale_buffer[3 * num_rows + row_idx] = accum_exp_val;

  event1();

  return;
}

extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                  const int32_t input_size) {
  softmax_simple_bf16(input, output, input_size);
}

void partial_softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                          bfloat16 *restrict scale_buffer,
                          const int32_t input_size, const int32_t row_idx,
                          const int32_t num_rows, const bfloat16 scale) {
  partial_softmax_alias_bf16(input, output, scale_buffer, input_size, row_idx,
                             num_rows, scale);
}

// Fill [unmasked_size, total_size) with -inf so a subsequent softmax drops the
// masked tail (causal / padding mask).
void mask_bf16(bfloat16 *inout, const int32_t unmasked_size,
               const int32_t total_size) {
  for (int32_t i = unmasked_size; i < total_size; i++) {
    inout[i] = (bfloat16)(-INFINITY);
  }
}

} // extern "C"
