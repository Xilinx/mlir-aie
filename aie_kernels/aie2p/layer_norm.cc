//===- layernorm.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

// AIE2P has no scalar float multiply, divide or C-style int-to-float convert;
// those lower to the soft-float helpers __mulsf3, __divsf3 and __floatsisf.
// aie::to_float is a single fx2flt and aie::inv is native, so a reciprocal
// multiply through one vector lane keeps the row statistics off the libcalls.
static inline float scalar_mul(float a, float b) {
  return ::aie::mul(::aie::broadcast<float, 16>(a), b).to_vector<float>()[0];
}

// a - b * c. Subtracting one scalar_mul from another directly makes Peano
// form a <16 x float> G_FSUB it cannot legalize, so the subtraction has to
// stay inside the accumulator.
static inline float scalar_mul_sub(float a, float b, float c) {
  ::aie::accum<accfloat, 16> acc;
  acc.from_vector(::aie::broadcast<float, 16>(a));
  return ::aie::msc(acc, ::aie::broadcast<float, 16>(b), c)
      .to_vector<float>()[0];
}

template <typename T, int N>
void layer_norm(const T *restrict input, T *restrict output, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;

  // cols is non-negative, so the unsigned divide lowers to a shift.
  const int vector_chunks = (uint32_t)cols / N;

  // Reduce the row sum in an f32 accumulator, not a bf16 vector: a bf16 running
  // sum drops low-order bits as the reduction length grows (embedding_dim is
  // typically thousands), so the mean -- and every quantity derived from it --
  // is already lossy before the variance is computed. The sum of squares is
  // already reduced in f32.
  ::aie::accum<accfloat, N> sum_acc = ::aie::zeros<accfloat, N>();
  ::aie::accum<accfloat, N> sum_sq_acc = ::aie::zeros<accfloat, N>();
  if (vector_chunks > 0) {
    const T *restrict p = input;
    AIE_LOOP_UNROLL(2)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> reg_a = ::aie::load_v<N>(p);
      sum_acc = ::aie::add(sum_acc, reg_a);
      sum_sq_acc = ::aie::mac_square(sum_sq_acc, reg_a);
      p += N;
    }
  }

  const float inv_cols = ::aie::inv(::aie::to_float<float>(cols));
  float mean = scalar_mul(
      ::aie::reduce_add(sum_acc.template to_vector<float>()), inv_cols);
  float variance = scalar_mul_sub(
      scalar_mul(::aie::reduce_add(sum_sq_acc.template to_vector<float>()),
                 inv_cols),
      mean, mean);
  float inv_std = aie::invsqrt(variance + epsilon);

  ::aie::vector<T, N> mean_v = ::aie::broadcast<T, N>((T)mean);
  ::aie::vector<T, N> inv_std_v = ::aie::broadcast<T, N>((T)inv_std);

  // gamma = 1 and beta = 0 here, so the affine pair is not applied at all.
  if (vector_chunks > 0) {
    const T *restrict pi = input;
    T *restrict po = output;
    AIE_LOOP_UNROLL(2)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < vector_chunks; i++) {
      ::aie::vector<T, N> diff_v = ::aie::sub(::aie::load_v<N>(pi), mean_v);
      ::aie::store_v(po, ::aie::mul(diff_v, inv_std_v).template to_vector<T>());
      pi += N;
      po += N;
    }
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
  // cols is non-negative, so the unsigned divide lowers to a shift.
  const int chunks = (uint32_t)cols / N;
  const float inv_cols = ::aie::inv(::aie::to_float<float>(cols));

  // Pass 1: mean = sum(x) / cols.
  ::aie::accum<accfloat, N> sum_acc = ::aie::zeros<accfloat, N>();
  if (chunks > 0) {
    const TIn *restrict p = input;
    AIE_LOOP_UNROLL(2)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < chunks; i++) {
      sum_acc = ::aie::add(sum_acc, ::aie::load_v<N>(p));
      p += N;
    }
  }
  float mean = scalar_mul(
      ::aie::reduce_add(sum_acc.template to_vector<float>()), inv_cols);
  ::aie::vector<TIn, N> mean_v = ::aie::broadcast<TIn, N>((TIn)mean);

  // Pass 2: variance = sum((x - mean)^2) / cols (centered two-pass).
  ::aie::accum<accfloat, N> var_acc = ::aie::zeros<accfloat, N>();
  if (chunks > 0) {
    const TIn *restrict p = input;
    AIE_LOOP_UNROLL(2)
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int i = 0; i < chunks; i++) {
      var_acc =
          ::aie::mac_square(var_acc, ::aie::sub(::aie::load_v<N>(p), mean_v));
      p += N;
    }
  }
  float variance = scalar_mul(
      ::aie::reduce_add(var_acc.template to_vector<float>()), inv_cols);
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
    if (chunks > 0) {
      const TIn *restrict pi = input;
      const TIn *restrict pg = gamma;
      const TIn *restrict pb = beta;
      TOut *restrict po = output;
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(1)
      for (int i = 0; i < chunks; i++) {
        ::aie::vector<TIn, N> diff_v = ::aie::sub(::aie::load_v<N>(pi), mean_v);
        ::aie::vector<TIn, N> norm_v =
            ::aie::mul(diff_v, inv_std_v).template to_vector<TIn>();
        // Kept as a separate multiply and add rather than a mac: an FMA would
        // skip the rounding of norm * gamma that the host reference performs.
        ::aie::vector<TIn, N> scaled_v =
            ::aie::mul(norm_v, ::aie::load_v<N>(pg)).template to_vector<TIn>();
        ::aie::vector<TIn, N> out_v =
            ::aie::add(scaled_v, ::aie::load_v<N>(pb));
        ::aie::accum<accfloat, N> a;
        a.from_vector(out_v);
        ::aie::store_v(po, a.template to_vector<TOut>());
        pi += N;
        pg += N;
        pb += N;
        po += N;
      }
    }
    ::aie::set_rounding(saved_rounding);
  } else {
    // gamma = 1 and beta = 0 here, so the affine pair is not applied at all.
    if (chunks > 0) {
      const TIn *restrict pi = input;
      TOut *restrict po = output;
      AIE_LOOP_UNROLL(2)
      AIE_PREPARE_FOR_PIPELINING
      AIE_LOOP_MIN_ITERATION_COUNT(1)
      for (int i = 0; i < chunks; i++) {
        ::aie::vector<TIn, N> diff_v = ::aie::sub(::aie::load_v<N>(pi), mean_v);
        ::aie::store_v(po,
                       ::aie::mul(diff_v, inv_std_v).template to_vector<TIn>());
        pi += N;
        po += N;
      }
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
