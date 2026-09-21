//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <aie_api/aie.hpp>

// Required command-line definitions:
// - DIM_m
// - DIM_n
// compile e.g. as
// xchesscc_wrapper aie2 -I/tools/Xilinx/Vitis/2023.2/aietools/include
// -DDIM_m=96 -DDIM_n=32 -c ../kernel.cc -o kernel.o

template <typename T_in, typename T_out, int m, int n, int t>
void row_wise_bias_add(const T_in *__restrict in, const T_in *__restrict bias,
                       T_out *__restrict out) {
  const T_in *__restrict bias_ptr = bias;
  const T_in *__restrict in_base_ptr = in;
  const T_in *__restrict in_ptr = in_base_ptr;
  T_out *__restrict out_base_ptr = out;
  T_out *__restrict out_ptr = out_base_ptr;
  constexpr int n_div_t = n / t;

  for (int j = 0; j < n_div_t; j++) {
    aie::vector<T_in, t> bias = aie::load_v<t>(bias_ptr);
    for (int i = 0; i < m; i += 1) {
      aie::vector<T_in, t> in = aie::load_v<t>(in_ptr);
      aie::store_v(out_ptr, aie::add(in, bias));
      in_ptr += n;
      out_ptr += n;
    }
    bias_ptr += t;
    in_base_ptr += t;
    in_ptr = in_base_ptr;
    out_base_ptr += t;
    out_ptr = out_base_ptr;
  }
}

extern "C" {

void row_wise_bias_add_f32_f32(const float *__restrict in,
                               const float *__restrict bias,
                               float *__restrict out) {
  constexpr int t = 32;
  static_assert(DIM_n % t == 0);
  row_wise_bias_add<float, float, DIM_m, DIM_n, t>(in, bias, out);
}
}

// Row-wise affine transform with a narrowing output cast: out = in*gamma +
// beta, gamma/beta per-column (broadcast over rows), narrowed to T_out.
// gamma and beta are packed into one [2*n] buffer per column-block (gamma
// then beta) rather than passed as separate inputs -- an AIE2 tile has only
// two input DMA channels, and `in` already takes one.
template <typename T_in, typename T_out, int m, int n, int t>
void row_wise_affine_cast(const T_in *__restrict in, const T_in *__restrict gb,
                          T_out *__restrict out) {
  const T_in *__restrict gamma_ptr = gb;
  const T_in *__restrict beta_ptr = gb + n;
  const T_in *__restrict in_base_ptr = in;
  const T_in *__restrict in_ptr = in_base_ptr;
  T_out *__restrict out_base_ptr = out;
  T_out *__restrict out_ptr = out_base_ptr;
  constexpr int n_div_t = n / t;

  // Round-to-nearest-even, matching a host f32->bf16 pack: the AIE default
  // truncates toward zero, which would drift from the host by up to 1 ULP.
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (int j = 0; j < n_div_t; j++) {
    aie::vector<T_in, t> gamma = aie::load_v<t>(gamma_ptr);
    aie::vector<T_in, t> beta = aie::load_v<t>(beta_ptr);
    for (int i = 0; i < m; i += 1) {
      aie::vector<T_in, t> in = aie::load_v<t>(in_ptr);
      aie::vector<T_in, t> scaled = aie::mul(in, gamma);
      aie::vector<T_in, t> affine = aie::add(scaled, beta);
      // aie::vector has no direct to_vector<T_out>(); narrow through an
      // accfloat accumulator, which does.
      aie::accum<accfloat, t> acc;
      acc.from_vector(affine);
      aie::store_v(out_ptr, acc.template to_vector<T_out>());
      in_ptr += n;
      out_ptr += n;
    }
    gamma_ptr += t;
    beta_ptr += t;
    in_base_ptr += t;
    in_ptr = in_base_ptr;
    out_base_ptr += t;
    out_ptr = out_base_ptr;
  }
  aie::set_rounding(saved_rounding);
}

extern "C" {

void row_wise_affine_cast_f32_bf16(const float *__restrict in,
                                   const float *__restrict gb,
                                   bfloat16 *__restrict out) {
  constexpr int t = 32;
  static_assert(DIM_n % t == 0);
  row_wise_affine_cast<float, bfloat16, DIM_m, DIM_n, t>(in, gb, out);
}
}