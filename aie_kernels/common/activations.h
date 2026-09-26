//===- activations.h --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef __ACTIVATIONS_H__
#define __ACTIVATIONS_H__
#include "../aie_arch.h"
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

#if AIE_HAS_NATIVE_TANH && !ACTIVATIONS_TANH_LUT
#define ACTIVATIONS_NATIVE_TANH 1
#else
#define ACTIVATIONS_NATIVE_TANH 0
// Supplies getTanhBf16. Resolved from the runtime-lib include directory the
// build adds for the target arch (aie_runtime_lib/AIE2[P]), not from this
// file's own directory. Its tables need lut_based_ops.cpp linked in, which is
// what lut_kernel.cc exists to do.
#include "lut_based_ops.h"

#if AIE_TUNED_AIE2P
// AIE2P's getTanhBf16 written out, the accumulator kept for a caller to narrow
// where it likes: 32 segments of 0.25 over [-4, 4), each offset + slope * x.
// getTanhBf16 builds an aie::linear_approx on every call, and its scratchpad
// member escapes, so each call stored the whole object to the stack and read
// the input back through it. x is clamped to the table's range first; the end
// segments are the constants -1 and 1, so no finite result changes and +-inf
// no longer makes 0 * inf. The tables' centring offset goes on the index, not
// the pointers: a table pointer offset from the array loses the reads' memory
// operands, and every load and store around them is then kept in order.
//
// Any table laid out as tanh_lut_ab/cd reads the same way; shift sets the
// segment width, 2^(4 - shift), and so the range, [-2^(8 - shift), 2^(8 -
// shift)). sigmoid.cc's table is the other one.
template <int shift>
__attribute__((always_inline)) inline aie::accum<accfloat, 16>
lut_segments_acc(const float *ab, const float *cd,
                 aie::vector<bfloat16, 16> x) {
  constexpr int bias_bytes = 16 << 4;
  constexpr float range = 1 << (8 - shift);
  const aie::vector<bfloat16, 16> xc = aie::max(
      aie::min(x, bfloat16(range - 1.0f / (1 << shift))), bfloat16(-range));
  const aie::vector<int32, 16> index =
      aie::add(aie::vector<int32, 16>(bfloat16_to_int(xc, shift)), bias_bytes);
  v32bfloat16 coeff0, coeff1;
  load_lut_2x_float(ab, cd, index, coeff0, coeff1);
  aie::accum<accfloat, 32> offset;
  offset.insert(1, aie::accum<accfloat, 16>(
                       (v16accfloat)::shuffle(coeff0, coeff1, T32_16x2_hi)));
  aie::vector<bfloat16, 32> xx = aie::zeros<bfloat16, 32>();
  xx.insert(1, xc);
  aie::accum<accfloat, 32> result =
      mac_elem_32(::shuffle(coeff0, coeff1, T16_16x4_lo), xx, offset);
  return result.extract<16>(1);
}

__attribute__((always_inline)) inline aie::accum<accfloat, 16>
tanh_lut_acc(aie::vector<bfloat16, 16> x) {
  return lut_segments_acc<6>(tanh_lut_ab, tanh_lut_cd, x);
}
#endif

__attribute__((always_inline)) inline aie::vector<bfloat16, 16>
tanh_lut_bf16(aie::vector<bfloat16, 16> x) {
#if AIE_TUNED_AIE2P
  return tanh_lut_acc(x).to_vector<bfloat16>();
#else
  return getTanhBf16(x);
#endif
}

#if AIE_TUNED_AIE2P
// tanh from in to out, which may be the same buffer, 32 lanes a trip, n a
// multiple of 32. A loop holding only the table reads pipelines best, so
// callers put their arithmetic in separate passes before and after this one
// (sigmoid's multiply inside it: II 30, against 1 + 22 + 1 split). The pre-RA
// pipeliner keeps each store ahead of the next trip's table reads (recurrence
// 18) and settles on II 27; asked for 16, it gives up, and the post-RA
// pipeliner reaches 22-23.
__attribute__((always_inline)) inline void tanh_lut_map(const bfloat16 *in,
                                                        bfloat16 *out, int n) {
  auto it_in = aie::begin_vector<32>(in);
  auto it_out = aie::begin_vector<32>(out);
#pragma clang loop pipeline_initiation_interval(16)
  for (int i = 0; i < n; i += 32) {
    const aie::vector<bfloat16, 32> x = *it_in++;
    *it_out++ = aie::concat(tanh_lut_bf16(x.extract<16>(0)),
                            tanh_lut_bf16(x.extract<16>(1)));
  }
}
#endif
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
  return tanh_lut_bf16(x.to_vector<bfloat16>());
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
  return tanh_lut_bf16(x);
#endif
}

// Stay in f32 until the single bf16 conversion at the end. Rounding to bf16
// before the activation rounds twice and lets the activation slope amplify the
// first rounding (measured 1.35x error on silu, 1.17x gelu, 1.08x sigmoid). The
// only unavoidable early narrowing is AIE2's bf16-only tanh LUT.
//
// There is no f32 vector multiplier: aie::mul over aie::vector<float, N> is a
// bf16 emulation. Every multiply below has a bf16 operand -- a constant exact
// in bf16, or tanh's result.

// tanh of an f32 vector, bf16 out: tanh is in [-1, 1], where bf16 costs at most
// 2^-9. An f32 result on AIE2P makes aiecc's stack measurement report a
// spurious recursion.
template <int vec_size>
__attribute__((always_inline)) aie::vector<bfloat16, vec_size>
tanh_bf16_vec(aie::vector<float, vec_size> x) {
#if ACTIVATIONS_NATIVE_TANH
  return aie::tanh<bfloat16>(x);
#else
  static_assert(vec_size % 16 == 0, "AIE2's LUT tanh is 16 lanes wide");
  aie::accum<accfloat, vec_size> narrowed;
  narrowed.from_vector(x);
  const aie::vector<bfloat16, vec_size> n =
      narrowed.template to_vector<bfloat16>();
  aie::vector<bfloat16, vec_size> out;
  for (unsigned i = 0; i < vec_size / 16; i++)
    out.insert(i, tanh_lut_bf16(n.template extract<16>(i)));
  return out;
#endif
}

// acc + x * b, for an f32 x and a bf16 b. x splits exactly into three bf16
// terms, so three macs against b give the exact product.
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

// x * c. c splits into exact bf16 terms the same way x does: one for a power
// of two, three in general.
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

// sigmoid(x) = 0.5 * tanh(x/2) + 0.5, one bf16 mac. 0.5 and tanh's result are
// bf16, so the product is exact in the accumulator.
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

// x * sigmoid(2u) = x/2 * tanh(u) + x/2, for a caller holding half the sigmoid
// argument. Halving is exact, so both terms are macs against a bf16 operand.
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
  // 1.702 / 2 folds sigmoid's halving into x's scaling; halving is exact.
  return x_times_sigmoid_vec<vec_size>(
      x, scale_vec<vec_size, 0.851f>(x).template to_vector<float>());
}

#endif // __ACTIVATIONS_H__
