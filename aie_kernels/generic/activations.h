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
// tanh is the one architecture-specific step, and on AIE2P it is a choice.
//
//   ACTIVATIONS_TANH_LUT=0 (AIE2P default)  aie::tanh, one vtanh instruction.
//   ACTIVATIONS_TANH_LUT=1                  getTanhBf16, the interpolated LUT.
//
// They are not equally accurate: vtanh returns its argument unchanged for
// |x| <= 0.5, so tanh(0.5) comes back 19 bf16 ulps out, where the LUT stays
// within 5.1e-3 absolute everywhere. Whether that is worth the extra loads is
// the caller's call, not a fixed per-architecture one. AIE2 has no tanh
// instruction, so the LUT is its only path whatever this is set to.
#ifndef ACTIVATIONS_TANH_LUT
#define ACTIVATIONS_TANH_LUT 0
#endif

#if __AIE_ARCH__ >= 21 && !ACTIVATIONS_TANH_LUT
#define ACTIVATIONS_NATIVE_TANH 1
#else
#define ACTIVATIONS_NATIVE_TANH 0
// Supplies getTanhBf16. Resolved from the runtime-lib include directory the
// build adds for the target arch (aie_runtime_lib/AIE2[P]), not from this
// file's own directory. Its tables need lut_based_ops.cpp linked in, which is
// what aie2/lut_kernel.cc exists to do.
#include "lut_based_ops.h"
#endif

// tanh of 16 bf16 lanes, on whichever path this architecture has. This is the
// whole of what separates the standalone bf16 activation kernels
// (tanh/sigmoid/silu/swiglu/gelu.cc) between the two architectures, so they
// share this rather than each carrying its own #if. 16 lanes because that is
// what AIE2's LUT is fixed at; a 32-wide kernel splits and concatenates.
//
// The accumulator overload is the primitive: a caller that has just multiplied
// holds one, and where it narrows to bf16 is exactly what differs. AIE2P feeds
// aie::tanh the f32; AIE2 must narrow first because its LUT is bf16-in. Taking
// bf16 here instead would force that narrowing on AIE2P too.
__attribute__((always_inline)) inline aie::vector<bfloat16, 16>
tanh_bf16_v16(aie::accum<accfloat, 16> x) {
#if ACTIVATIONS_NATIVE_TANH
  return aie::tanh<bfloat16>(x.to_vector<float>());
#else
  return getTanhBf16(x.to_vector<bfloat16>());
#endif
}

// For a caller whose input is already bf16 and has no accumulator to hand.
// Carries its own #if rather than widening into the overload above: on AIE2
// the LUT takes bf16 directly, and the bf16 -> accum -> bf16 round trip does
// not fold away, costing three instructions per call in tanh.cc.
__attribute__((always_inline)) inline aie::vector<bfloat16, 16>
tanh_bf16_v16(aie::vector<bfloat16, 16> x) {
#if ACTIVATIONS_NATIVE_TANH
  aie::accum<accfloat, 16> acc;
  acc.from_vector(x, 0);
  return aie::tanh<bfloat16>(acc.to_vector<float>());
#else
  return getTanhBf16(x);
#endif
}

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
