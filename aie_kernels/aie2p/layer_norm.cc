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

// f32 per-row LayerNorm, optionally with a real per-column affine and an
// output cast, sharing one core: the bf16 layer_norm above centers with a
// single E[x^2] - mean^2 reduction, which is fine because the bf16 input
// contract keeps the mean/std ratio small (a bf16 value near a large mean has
// an ulp wider than the std, so that regime is unrepresentable). On f32 input
// the mean can be large relative to the std and E[x^2] - mean^2 then
// catastrophically cancels, so this variant reduces over the feature axis
// (cols) per row with a numerically stable two-pass centered variance: center
// first, then square.
//
// kAffine selects between the two extern "C" entry points below:
//   false: gamma = 1, beta = 0 (broadcast identity constants), TOut == TIn,
//          no rounding-mode change: this is `layer_norm_f32`'s original
//          body, unchanged.
//   true:  gamma/beta are real per-column values loaded from the caller, and
//          the result is narrowed to TOut (bfloat16). The narrowing write
//          needs round-to-nearest-even (`conv_even`), so this path saves the
//          core's rounding mode before the write and restores it after.
//          The AIE2p core's rounding mode is a single sticky register,
//          shared by every kernel that runs on that core, so a kernel that
//          cares about its own narrowing behavior has to set the mode
//          itself and hand it back, the same save/restore mm.cc's
//          AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 block uses. conv_even
//          also matches a typical host f32 -> bf16 pack (e.g. AVX512-BF16's
//          vcvtne2ps2bf16), so an on-chip and a host cast of the same value
//          agree bit-for-bit.
// `if constexpr` on kAffine picks one of two whole loop bodies at compile
// time (no runtime branch in either): the false instantiation has no
// gamma/beta loads and no rounding-mode calls at all, so `layer_norm_f32`'s
// codegen is unaffected by this refactor.
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
    // Real per-column affine + narrowing write. Only this instantiation
    // touches the rounding mode, and it hands the caller's mode back before
    // returning: the AIE2p core's rounding mode is a single sticky register
    // shared by every kernel that runs on that core, so a kernel that cares
    // about its own narrowing behavior has to set the mode itself and hand
    // it back, the same save/restore mm.cc's
    // AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 block uses. conv_even also
    // matches a typical host f32 -> bf16 pack (e.g. AVX512-BF16's
    // vcvtne2ps2bf16), so an on-chip and a host cast of the same value agree
    // bit-for-bit.
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
  layer_norm<bfloat16, 16>(input, output, cols);
}

// Unchanged ABI: gamma = 1, beta = 0, f32 in and out. See the kAffine
// comment above layer_norm_f32_impl.
void layer_norm_f32(float *input, float *output, int32_t cols) {
  layer_norm_f32_impl<float, float, 16, false>(input, output, nullptr, nullptr,
                                               cols);
}

// Fused row-wise LayerNorm + real per-column affine + f32 -> bfloat16
// narrowing cast, one dispatch. `gb` packs gamma then beta into one
// `[2 * cols]` buffer on a single DMA input channel, alongside the `[cols]`
// `input`, two inputs total, the AIE2p compute-tile input-DMA limit (see
// programming_examples/ml/norm/norm.py's `norm_affine` design, which mirrors
// this exact packing on the IRON side).
void layer_norm_affine_cast(float *input, float *gb, bfloat16 *output,
                            int32_t cols) {
  layer_norm_f32_impl<float, bfloat16, 16, true>(input, output, gb, gb + cols,
                                                 cols);
}
}
