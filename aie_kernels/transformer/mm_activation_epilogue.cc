//===- mm_activation_epilogue.cc ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/activations.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

static inline void mm_identity_row(uint32_t n, const float *__restrict acc,
                                   float *__restrict out) {
  event0();
  // Through begin_restrict_vector iterators, this copy gets no zero-overhead
  // loop on aie2p.
  auto body = [&]() __attribute__((always_inline)) {
    aie::store_v(out, aie::load_v<16>(acc));
    acc += 16;
    out += 16;
  };
  VERSIONED_LOOP(4, n / 16, body);
  event1();
}

// aie2p has no f32 multiplier: `aie::mul` on two float vectors expands to a
// three-way bf16 split of *both* operands. Where one operand is already exact
// in bf16, splitting the other in two keeps ~16 mantissa bits and stays on the
// native bf16 multiplier. hi stops at bf16's largest finite value: an x past
// it would round to inf and leave x - hi -inf.
struct bf16_split {
  aie::vector<bfloat16, 16> hi;
  aie::vector<bfloat16, 16> lo;
};

static inline bf16_split split_f32(const aie::vector<float, 16> &x) {
  const bfloat16 bf16_max = 3.38953139e38f;
  aie::accum<accfloat, 16> a;
  a.from_vector(x);
  bf16_split s;
  s.hi = aie::max(aie::min(a.to_vector<bfloat16>(), bf16_max),
                  bfloat16(-bf16_max));
  aie::accum<accfloat, 16> h;
  h.from_vector(s.hi);
  aie::accum<accfloat, 16> r;
  r.from_vector(aie::sub(x, h.to_vector<float>()));
  s.lo = r.to_vector<bfloat16>();
  return s;
}

static inline aie::vector<float, 16>
mul_split(const bf16_split &x, const aie::vector<bfloat16, 16> &y) {
  return aie::mac(aie::mul(x.hi, y), x.lo, y).to_vector<float>();
}

// SiLU: out = x * sigmoid(x), sigmoid built from tanh (see activations.h).
// Both multipliers of x are exact in bf16, so a two-term split of x carries
// the whole f32 input into each product. An all-f32 chain overruns the
// per-tile cycle budget and hangs.
static inline void mm_silu_hiprec_row(uint32_t n, const float *__restrict acc,
                                      float *__restrict out) {
  event0();
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> halfb = aie::broadcast<bfloat16, 16>(0.5f);
  auto it_in = aie::begin_restrict_vector<16>(acc);
  auto it_out = aie::begin_restrict_vector<16>(out);
  auto body = [&]() __attribute__((always_inline)) {
    aie::vector<float, 16> x = *it_in++;
    bf16_split xs = split_f32(x);
    aie::vector<float, 16> half_x = mul_split(xs, halfb);
    aie::vector<bfloat16, 16> tanh_half_x = tanh_bf16_vec<16>(half_x);
    aie::vector<bfloat16, 16> tanh_p1 = aie::add(tanh_half_x, one);
    aie::vector<bfloat16, 16> sig = aie::mul(tanh_p1, halfb);
    *it_out++ = mul_split(xs, sig);
  };
  // Unrolled by 2, this loop gives wrong results on npu2.
  VERSIONED_LOOP(8, n / 16, body, AIE_LOOP_UNROLL(4));
  event1();
}

// GELU (tanh approximation, matches torch's gelu(approximate="tanh")):
//   gelu(x) = 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
static inline void mm_gelu_row(uint32_t n, const float *__restrict acc,
                               float *__restrict out) {
  event0();
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  // sqrt(2/pi) and sqrt(2/pi)*0.044715, for x*(c0 + c0c1*x^2): each
  // `aie::mul` result converts back to bf16 before the next one consumes it,
  // so fewer multiplies shorten the serial chain.
  const aie::vector<bfloat16, 16> c0 =
      aie::broadcast<bfloat16, 16>(0.7978845608f);
  const aie::vector<bfloat16, 16> c0c1 =
      aie::broadcast<bfloat16, 16>(0.7978845608f * 0.044715f);
  aie::accum<accfloat, 16> c0acc;
  c0acc.from_vector(c0);
  auto it_in = aie::begin_restrict_vector<16>(acc);
  auto it_out = aie::begin_restrict_vector<16>(out);
  auto body = [&]() __attribute__((always_inline)) {
    aie::accum<accfloat, 16> a;
    a.from_vector(*it_in++);
    aie::vector<bfloat16, 16> x = a.to_vector<bfloat16>();
    // 1 + tanh is 0 from x = -8 down, so 0.5x is clamped there rather than
    // making -inf * 0.
    aie::vector<bfloat16, 16> half_x =
        aie::mul(half, aie::max(x, bfloat16(-8.0f)));
    aie::vector<bfloat16, 16> x2 = aie::mul(x, x);
    aie::vector<bfloat16, 16> poly = aie::mac(c0acc, c0c1, x2);
    auto inner = aie::mul(x, poly);
    aie::vector<bfloat16, 16> t = tanh_bf16_v16(inner);
    aie::vector<bfloat16, 16> t_p1 = aie::add(t, one);
    *it_out++ = aie::mul(half_x, t_p1).to_vector<float>();
  };
  // Unrolled by 4, this loop gives wrong results on npu2.
  VERSIONED_LOOP(4, n / 16, body, AIE_LOOP_UNROLL(2));
  event1();
}

#if AIE_TUNED_AIE2
// aie2 reads tanh from getTanhBf16's table, and each store is ordered before
// the next vector's table reads. A loop that loads, computes and stores one
// vector per iteration does not pipeline. Storing each result one iteration
// late puts the next vector's table reads ahead of the store, and with an II
// hint that loop pipelines.
template <int II, typename F>
static inline void mm_lut_rows(uint32_t n, const float *__restrict acc,
                               float *__restrict out, F f) {
  event0();
  auto it_in = aie::begin_restrict_vector<16>(acc);
  auto it_out = aie::begin_restrict_vector<16>(out);
  aie::vector<float, 16> prev = f(*it_in++);
  auto body = [&]() __attribute__((always_inline)) {
    aie::vector<float, 16> cur = f(*it_in++);
    *it_out++ = prev;
    prev = cur;
  };
  // Not VERSIONED_LOOP: its fallback assumes a trip, which a one-vector row
  // does not have.
  const int count = (int)(n / 16) - 1;
  if (count >= 4) {
    AIE_PREPARE_FOR_PIPELINING
    AIE_TRY_INITIATION_INTERVAL(II)
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int i = 0; i < count; i++)
      body();
  } else {
    AIE_LOOP_NO_UNROLL
    for (int i = 0; i < count; i++)
      body();
  }
  *it_out = prev;
  event1();
}

// split_f32 for an x that may lie past bf16's largest finite value, where a
// rounded hi would be inf and x - hi -inf. hi is x's top 16 bits instead,
// finite for a finite x, and x - hi is exact in f32.
static inline bf16_split split_f32_trunc(const aie::vector<float, 16> &x) {
  aie::vector<int32_t, 16> bits = x.cast_to<int32_t>();
  aie::vector<int32_t, 16> hi_bits =
      aie::bit_and(bits, aie::broadcast<int32_t, 16>((int32_t)0xffff0000));
  bf16_split s;
  s.hi = aie::filter_odd(bits.cast_to<int16_t>(), 1).cast_to<bfloat16>();
  aie::accum<accfloat, 16> r;
  r.from_vector(aie::sub(x, hi_bits.cast_to<float>()));
  s.lo = r.to_vector<bfloat16>();
  return s;
}

// SiLU as in mm_silu_hiprec_row, except that tanh reads bf16(x)/2, which is
// exact in bf16, instead of rounding hi/2 + lo/2 again. The two differ only
// where that rounding ties, and dropping it shortens the chain. Past bf16's
// range bf16(x) is inf, which the table takes, but the product's split must
// stay finite, so it truncates.
static inline aie::vector<float, 16> mm_silu_lut(aie::vector<float, 16> x) {
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  aie::accum<accfloat, 16> half_acc;
  half_acc.from_vector(half);
  bf16_split xs = split_f32_trunc(x);
  aie::accum<accfloat, 16> a;
  a.from_vector(x);
  aie::vector<bfloat16, 16> t =
      tanh_bf16_v16(aie::mul(a.to_vector<bfloat16>(), half));
  // bf16(0.5 + 0.5*t) is bf16(t + 1) * 0.5.
  aie::vector<bfloat16, 16> sig =
      aie::mac(half_acc, t, half).to_vector<bfloat16>();
  return mul_split(xs, sig);
}

// GELU as in mm_gelu_row. t + 1 is a mac on an accumulator holding 1: a bf16
// aie::add goes through f32 and back on aie2.
static inline aie::vector<float, 16> mm_gelu_lut(aie::vector<float, 16> xf) {
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> c0 =
      aie::broadcast<bfloat16, 16>(0.7978845608f);
  const aie::vector<bfloat16, 16> c0c1 =
      aie::broadcast<bfloat16, 16>(0.7978845608f * 0.044715f);
  aie::accum<accfloat, 16> c0acc;
  c0acc.from_vector(c0);
  aie::accum<accfloat, 16> one_acc;
  one_acc.from_vector(one);
  aie::accum<accfloat, 16> a;
  a.from_vector(xf);
  aie::vector<bfloat16, 16> x = a.to_vector<bfloat16>();
  // t + 1 is 0 from x = -8 down, so -inf is clamped there rather than
  // making -inf * 0.
  aie::vector<bfloat16, 16> xl = aie::max(x, bfloat16(-8.0f));
  aie::vector<bfloat16, 16> half_x = aie::mul(half, xl);
  aie::vector<bfloat16, 16> x2 = aie::mul(xl, xl);
  aie::vector<bfloat16, 16> poly = aie::mac(c0acc, c0c1, x2);
  aie::vector<bfloat16, 16> t = tanh_bf16_v16(aie::mul(xl, poly));
  aie::vector<bfloat16, 16> t_p1 =
      aie::mac(one_acc, t, one).to_vector<bfloat16>();
  return aie::mul(half_x, t_p1).to_vector<float>();
}

#endif

// ReLU, the epilogue a conv2d-as-GEMM patch-embed stem applies after its
// bias-augmented matmul, as an integer select on the bit pattern: lanes where
// x - 1 is below `neg` become +0. With neg = -inf's pattern those are the
// negative floats other than -0 and -NaN; with INT32_MIN there are none, which
// aie2 uses for identity. aie2 has no f32 max and aie2p's is emulated.
static inline void mm_floor_row(uint32_t n, const float *__restrict acc,
                                float *__restrict out, int32_t neg) {
  event0();
  const aie::vector<int32_t, 16> zero = aie::zeros<int32_t, 16>();
  const aie::vector<int32_t, 16> one = aie::broadcast<int32_t, 16>(1);
  auto it_in = aie::begin_restrict_vector<16>((const int32_t *)acc);
  auto it_out = aie::begin_restrict_vector<16>((int32_t *)out);
  auto body = [&]() __attribute__((always_inline)) {
    aie::vector<int32_t, 16> x = *it_in++;
    *it_out++ = aie::select(x, zero, aie::lt(aie::sub(x, one), neg));
  };
  VERSIONED_LOOP(4, n / 16, body);
  event1();
}
extern "C" {

// mode: 0 = identity, 1 = SiLU, 2 = GELU, 3 = ReLU. `n` a positive multiple
// of 16.
void mm_activation_epilogue_row(const float *__restrict c_in,
                                float *__restrict c_out, int32_t n,
                                int32_t mode) {
#if AIE_TUNED_AIE2
  if (mode == 1) {
    mm_lut_rows<37>((uint32_t)n, c_in, c_out, mm_silu_lut);
  } else if (mode == 2) {
    mm_lut_rows<58>((uint32_t)n, c_in, c_out, mm_gelu_lut);
  } else {
    mm_floor_row((uint32_t)n, c_in, c_out,
                 mode == 3 ? (int32_t)0xff800000 : INT32_MIN);
  }
#else
  if (mode == 1) {
    mm_silu_hiprec_row((uint32_t)n, c_in, c_out);
  } else if (mode == 2) {
    mm_gelu_row((uint32_t)n, c_in, c_out);
  } else if (mode == 3) {
    mm_floor_row((uint32_t)n, c_in, c_out, (int32_t)0xff800000);
  } else {
    mm_identity_row((uint32_t)n, c_in, c_out);
  }
#endif
}

} // extern "C"
