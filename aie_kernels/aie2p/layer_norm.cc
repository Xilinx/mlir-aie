//===- layernorm.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T, int N>
void layer_norm(const T *restrict input, T *restrict output, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f;
  const float beta = 0.0f;

  ::aie::vector<T, N> gamma_v = ::aie::broadcast<T, N>(gamma);
  ::aie::vector<T, N> beta_v = ::aie::broadcast<T, N>(beta);

  int vector_chunks = cols / N;

  // Reduce the row sum in an f32 accumulator, not a bf16 vector: a bf16 running
  // sum drops low-order bits as the reduction length grows (embedding_dim is
  // typically thousands), so the mean -- and every quantity derived from it --
  // is already lossy before the variance is computed. The sum of squares is
  // already reduced in f32.
  ::aie::accum<accfloat, N> sum_acc = ::aie::zeros<accfloat, N>();
  ::aie::vector<float, N> sum_sq_acc = ::aie::zeros<float, N>();
  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    sum_acc = ::aie::add(sum_acc, reg_a);
    ::aie::vector<float, N> sq_acc = ::aie::mul(reg_a, reg_a);
    sum_sq_acc = ::aie::add(sum_sq_acc, sq_acc);
  }

  float mean =
      ::aie::reduce_add(sum_acc.template to_vector<float>()) / float(cols);
  float variance = ::aie::reduce_add(sum_sq_acc) / float(cols) - mean * mean;
  float inv_std = aie::invsqrt(variance + epsilon);

  ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>((T)mean);
  ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>((T)inv_std);

  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    ::aie::vector<T, N> diff_v = ::aie::sub(reg_a, mean_v);
    ::aie::vector<T, N> norm_v = ::aie::mul(diff_v, inv_std_v);
    ::aie::vector<T, N> scaled_v = ::aie::mul(norm_v, gamma_v);
    ::aie::vector<T, N> out_v = ::aie::add(scaled_v, beta_v);
    ::aie::store_v(output + i * N, out_v);
  }
  event1();
}

// f32 per-row LayerNorm, optionally with a per-column affine and a narrowing
// output cast. The bf16 layer_norm above centers with a single
// E[x^2] - mean^2 reduction, which the bf16 input contract makes safe: a bf16
// value near a large mean has an ulp wider than the std, so that regime is
// unrepresentable. On f32 input the mean can be large relative to the std and
// E[x^2] - mean^2 catastrophically cancels, so this one takes the two-pass
// centered variance instead: center first, then square.
template <typename TIn, typename TOut, int N, bool kAffine>
static inline void layer_norm_f32_impl(const TIn *restrict input,
                                       TOut *restrict output,
                                       const TIn *restrict gamma,
                                       const TIn *restrict beta, int32_t cols) {
  static_assert(kAffine || std::is_same_v<TOut, TIn>,
                "the non-affine instantiation writes TIn straight through, so "
                "TOut must equal TIn");
  event0();
  constexpr float epsilon = 1e-5f;
  int chunks = cols / N;

  // Pass 1: mean = sum(x) / cols.
  ::aie::vector<TIn, N> sum_v = ::aie::zeros<TIn, N>();
  for (int i = 0; i < chunks; i++) {
    sum_v = ::aie::add(sum_v, ::aie::load_v<N>(input + i * N));
  }
  float mean = ::aie::reduce_add(sum_v) / float(cols);
  ::aie::vector<TIn, N> mean_v = ::aie::broadcast<TIn, N>((TIn)mean);

  // Pass 2: variance = sum((x - mean)^2) / cols (centered two-pass).
  ::aie::vector<TIn, N> var_v = ::aie::zeros<TIn, N>();
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<TIn, N> diff_v =
        ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<TIn, N> sq = ::aie::mul(diff_v, diff_v);
    var_v = ::aie::add(var_v, sq);
  }
  float variance = ::aie::reduce_add(var_v) / float(cols);
  float inv_std = aie::invsqrt(variance + epsilon);
  ::aie::vector<TIn, N> inv_std_v = ::aie::broadcast<TIn, N>((TIn)inv_std);

  // The two instantiations diverge only in where gamma/beta come from and
  // whether the write narrows.
  if constexpr (kAffine) {
    // conv_even makes the narrowing write agree bit-for-bit with a host
    // f32 -> bf16 pack. The mode is one sticky register shared by every
    // kernel on this core, so it is handed back before returning.
    ::aie::rounding_mode saved_rounding =
        ::aie::swap_rounding(::aie::rounding_mode::conv_even);
    for (int i = 0; i < chunks; i++) {
      ::aie::vector<TIn, N> diff_v =
          ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
      ::aie::vector<TIn, N> norm_v = ::aie::mul(diff_v, inv_std_v);
      ::aie::vector<TIn, N> gamma_v = ::aie::load_v<N>(gamma + i * N);
      ::aie::vector<TIn, N> beta_v = ::aie::load_v<N>(beta + i * N);
      ::aie::vector<TIn, N> scaled_v = ::aie::mul(norm_v, gamma_v);
      ::aie::vector<TIn, N> out_v = ::aie::add(scaled_v, beta_v);
      ::aie::accum<accfloat, N> a;
      a.from_vector(out_v);
      ::aie::store_v(output + i * N, a.template to_vector<TOut>());
    }
    ::aie::set_rounding(saved_rounding);
  } else {
    // gamma = 1, beta = 0, TOut == TIn
    ::aie::vector<TIn, N> gamma_v = ::aie::broadcast<TIn, N>((TIn)1.0f);
    ::aie::vector<TIn, N> beta_v = ::aie::broadcast<TIn, N>((TIn)0.0f);
    for (int i = 0; i < chunks; i++) {
      ::aie::vector<TIn, N> diff_v =
          ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
      ::aie::vector<TIn, N> norm_v = ::aie::mul(diff_v, inv_std_v);
      ::aie::vector<TIn, N> scaled_v = ::aie::mul(norm_v, gamma_v);
      ::aie::vector<TIn, N> out_v = ::aie::add(scaled_v, beta_v);
      ::aie::store_v(output + i * N, out_v);
    }
  }

  event1();
}

extern "C" {
void layer_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  // N=32 bf16 = 512 bits = one AIE2P vector register.  conv_even rounding
  // matches the reference math more closely than the default floor mode for
  // the normalize pass.
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  layer_norm<bfloat16, 32>(input, output, cols);
}

void layer_norm_f32(float *input, float *output, int32_t cols) {
  layer_norm_f32_impl<float, float, 16, false>(input, output, nullptr, nullptr,
                                               cols);
}

// LayerNorm + per-column affine + f32 -> bfloat16 cast in one dispatch. `gb`
// packs gamma then beta into one `[2 * cols]` buffer so that the kernel takes
// two DMA inputs, the AIE2p compute-tile limit; see `norm_affine` in
// programming_examples/ml/norm/norm.py for the matching packing.
void layer_norm_affine_cast(float *input, float *gb, bfloat16 *output,
                            int32_t cols) {
  layer_norm_f32_impl<float, bfloat16, 16, true>(input, output, gb, gb + cols,
                                                 cols);
}
}
