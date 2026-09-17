//===- activations.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef __ACTIVATIONS_H__
#define __ACTIVATIONS_H__
#include <aie_api/aie.hpp>

// Branch-free vector-width activations over aie::vector<float, N> for
// mm_fused's epilogue, applied to the f32 accumulator already in registers --
// unlike the whole-buffer {gelu,silu}.cc entry points. sigmoid is exact by
// identity (tanh(x/2)+1)/2; gelu uses x*sigmoid(1.702x), a DIFFERENT curve from
// gelu.cc's tanh approximation, so results are not bit-identical to it.
//
// tanh is the one architecture-specific step: AIE2P has native f32 aie::tanh;
// AIE2 falls back to the 16-lane getTanhBf16 LUT in aie_runtime_lib/AIE2, so
// the AIE2 path is not bit-identical and carries the LUT error (test.py budgets
// accuracy per arch).

#if __AIE_ARCH__ >= 21
#define ACTIVATIONS_NATIVE_TANH 1
#else
#define ACTIVATIONS_NATIVE_TANH 0
// Supplies getTanhBf16. Resolved from the runtime-lib include directory the
// build adds for the target arch (aie_runtime_lib/AIE2), not from this file's
// own directory.
#include "lut_based_ops.h"
#endif

// Stay in f32 until the single bf16 conversion at the end. Rounding to bf16
// before the activation rounds twice and lets the activation slope amplify the
// first rounding (measured 1.35x error on silu, 1.17x gelu, 1.08x sigmoid). The
// only unavoidable early narrowing is AIE2's bf16-only tanh LUT.

// tanh of an f32 vector, on whichever path this architecture has.
template <int vec_size>
__attribute__((always_inline)) aie::vector<float, vec_size>
tanh_vec(aie::vector<float, vec_size> x) {
  // bf16 out on both paths, widened back to f32. Asking AIE2P for the f32
  // tanh instead defeats aiecc's stack measurement -- it reports a spurious
  // "__start -> _main_init -> core -> _main_init" recursion -- and tanh's
  // output is in [-1, 1], where bf16 costs at most 2^-9 absolute anyway. The
  // arithmetic AROUND it is where the f32 actually pays.
  aie::accum<accfloat, vec_size> widened;
#if ACTIVATIONS_NATIVE_TANH
  widened.from_vector(aie::tanh<bfloat16>(x));
#else
  static_assert(vec_size == 16,
                "AIE2's LUT tanh is fixed at 16 lanes, which is mm_fused's "
                "epilogue width; widening V needs an explicit split here");
  aie::accum<accfloat, vec_size> narrowed;
  narrowed.from_vector(x);
  aie::vector<bfloat16, vec_size> tanh_bf16 =
      getTanhBf16(narrowed.template to_vector<bfloat16>());
  widened.from_vector(tanh_bf16);
#endif
  return widened.template to_vector<float>();
}

// sigmoid(x) = (tanh(x/2) + 1) / 2
template <int vec_size>
__attribute__((always_inline)) aie::vector<float, vec_size>
sigmoid_vec(aie::vector<float, vec_size> x) {
  const aie::vector<float, vec_size> v_half =
      aie::broadcast<float, vec_size>(0.5f);
  const aie::vector<float, vec_size> v_one =
      aie::broadcast<float, vec_size>(1.0f);
  aie::vector<float, vec_size> t =
      tanh_vec<vec_size>(aie::mul(x, v_half).template to_vector<float>());
  return aie::mul(aie::add(t, v_one), v_half).template to_vector<float>();
}

// silu(x) = x * sigmoid(x)
template <int vec_size>
__attribute__((always_inline)) aie::vector<float, vec_size>
silu_vec(aie::vector<float, vec_size> x) {
  return aie::mul(x, sigmoid_vec<vec_size>(x)).template to_vector<float>();
}

// gelu(x) ~= x * sigmoid(1.702x)
template <int vec_size>
__attribute__((always_inline)) aie::vector<float, vec_size>
gelu_vec(aie::vector<float, vec_size> x) {
  const aie::vector<float, vec_size> v_scale =
      aie::broadcast<float, vec_size>(1.702f);
  aie::vector<float, vec_size> scaled =
      aie::mul(x, v_scale).template to_vector<float>();
  return aie::mul(x, sigmoid_vec<vec_size>(scaled)).template to_vector<float>();
}

#endif // __ACTIVATIONS_H__
