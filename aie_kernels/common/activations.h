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
// what lut_kernel.cc exists to do.
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
//
// There is no f32 vector multiplier: aie::mul over aie::vector<float, N> is a
// three-term bf16 emulation, 224 bytes of code at 16 lanes against 4 for one
// bf16 mac. Every multiply below therefore has a bf16 operand -- a constant
// exact in bf16, or tanh's result, which already is one.

// tanh of an f32 vector, bf16 out. AIE2P could return f32, but asking for it
// defeats aiecc's stack measurement -- it reports a spurious "__start ->
// _main_init -> core -> _main_init" recursion -- and tanh's output is in
// [-1, 1], where bf16 costs at most 2^-9 absolute anyway. AIE2's LUT has
// nothing else to offer. The narrow result goes straight into a bf16 mac
// below, so neither caller pays to widen it.
template <int vec_size>
__attribute__((always_inline)) aie::vector<bfloat16, vec_size>
tanh_bf16_vec(aie::vector<float, vec_size> x) {
#if ACTIVATIONS_NATIVE_TANH
  return aie::tanh<bfloat16>(x);
#else
  static_assert(vec_size == 16,
                "AIE2's LUT tanh is fixed at 16 lanes, which is mm_fused's "
                "epilogue width; widening V needs an explicit split here");
  aie::accum<accfloat, vec_size> narrowed;
  narrowed.from_vector(x);
  return getTanhBf16(narrowed.template to_vector<bfloat16>());
#endif
}

// acc + x * b, for an f32 x and a bf16 b. An f32 splits into three bf16 terms
// that sum back to it exactly -- three 8-bit mantissas cover f32's 24 -- so
// macing each term against b is the same arithmetic, not an approximation of
// it, and costs three products where aie::mul would spend nine splitting an
// operand that is already narrow.
template <int vec_size>
__attribute__((always_inline)) aie::accum<accfloat, vec_size>
mac_f32_bf16(aie::accum<accfloat, vec_size> acc, aie::vector<float, vec_size> x,
             aie::vector<bfloat16, vec_size> b) {
  const aie::vector<bfloat16, vec_size> one =
      aie::broadcast<bfloat16, vec_size>((bfloat16)1.0f);
  aie::accum<accfloat, vec_size> rem(x);
  const aie::vector<bfloat16, vec_size> x0 = rem.template to_vector<bfloat16>();
  rem = aie::msc(rem, x0, one);
  const aie::vector<bfloat16, vec_size> x1 = rem.template to_vector<bfloat16>();
  rem = aie::msc(rem, x1, one);
  const aie::vector<bfloat16, vec_size> x2 = rem.template to_vector<bfloat16>();
  acc = aie::mac(acc, x0, b);
  acc = aie::mac(acc, x1, b);
  return aie::mac(acc, x2, b);
}

// x * c. The constant decomposes into exact bf16 terms the same way x does, so
// a power of two needs one pass and a general constant needs three. Even at
// three this beats aie::mul, which forms the same nine products but sums them
// through f32 adds and re-splits x for each one.
template <int vec_size, float c>
__attribute__((always_inline)) aie::accum<accfloat, vec_size>
scale_vec(aie::vector<float, vec_size> x) {
  constexpr float c0 = (float)(bfloat16)c;
  constexpr float c1 = (float)(bfloat16)(c - c0);
  constexpr float c2 = c - c0 - c1;
  static_assert(c0 + c1 + c2 == c,
                "three bf16 terms must reproduce the constant exactly");
  aie::accum<accfloat, vec_size> acc;
  acc.from_vector(aie::zeros<float, vec_size>());
  acc = mac_f32_bf16<vec_size>(
      acc, x, aie::broadcast<bfloat16, vec_size>((bfloat16)c0));
  if constexpr (c1 != 0.0f)
    acc = mac_f32_bf16<vec_size>(
        acc, x, aie::broadcast<bfloat16, vec_size>((bfloat16)c1));
  if constexpr (c2 != 0.0f)
    acc = mac_f32_bf16<vec_size>(
        acc, x, aie::broadcast<bfloat16, vec_size>((bfloat16)c2));
  return acc;
}

// sigmoid(x) = (tanh(x/2) + 1) / 2, written as 0.5 * tanh(x/2) + 0.5 so that
// the halving and the offset are one bf16 mac rather than an f32 add and an
// f32 multiply. 0.5 is exact in bf16 and tanh's result is bf16, so the product
// is exact in the accumulator and the sum rounds where (t + 1) * 0.5 rounded.
template <int vec_size>
__attribute__((always_inline)) aie::vector<float, vec_size>
sigmoid_vec(aie::vector<float, vec_size> x) {
  const aie::accum<accfloat, vec_size> half(
      aie::broadcast<float, vec_size>(0.5f));
  const aie::vector<float, vec_size> half_x =
      scale_vec<vec_size, 0.5f>(x).template to_vector<float>();
  return aie::mac(half, tanh_bf16_vec<vec_size>(half_x),
                  aie::broadcast<bfloat16, vec_size>((bfloat16)0.5f))
      .template to_vector<float>();
}

// x * sigmoid(2u), for a caller holding half the sigmoid argument. That is
// x/2 * tanh(u) + x/2, and halving a bf16 is exact, so sigmoid's /2 rides on
// tanh's result and both terms are three-product macs against a bf16 operand.
// silu passes u = x/2 and shares the x/2 it already computed.
template <int vec_size>
__attribute__((always_inline)) aie::vector<float, vec_size>
x_times_sigmoid_vec(aie::vector<float, vec_size> x,
                    aie::vector<float, vec_size> u) {
  const aie::vector<bfloat16, vec_size> v_half =
      aie::broadcast<bfloat16, vec_size>((bfloat16)0.5f);
  const aie::vector<bfloat16, vec_size> half_tanh =
      aie::mul(tanh_bf16_vec<vec_size>(u), v_half)
          .template to_vector<bfloat16>();
  return mac_f32_bf16<vec_size>(scale_vec<vec_size, 0.5f>(x), x, half_tanh)
      .template to_vector<float>();
}

// silu(x) = x * sigmoid(x)
template <int vec_size>
__attribute__((always_inline)) aie::vector<float, vec_size>
silu_vec(aie::vector<float, vec_size> x) {
  return x_times_sigmoid_vec<vec_size>(
      x, scale_vec<vec_size, 0.5f>(x).template to_vector<float>());
}

// gelu(x) ~= x * sigmoid(1.702x)
template <int vec_size>
__attribute__((always_inline)) aie::vector<float, vec_size>
gelu_vec(aie::vector<float, vec_size> x) {
  // 1.702 / 2: the halving sigmoid owes its argument rides along in the scaling
  // x needs anyway. Halving a float is exact, so every partial product shifts
  // by one exponent and the result is the one the separate multiply gave.
  return x_times_sigmoid_vec<vec_size>(
      x, scale_vec<vec_size, 0.851f>(x).template to_vector<float>());
}

#endif // __ACTIVATIONS_H__
