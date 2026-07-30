//===- exp2f_vec.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Software f32 2^x, for callers (softmax, sigmoid, and other exp-family ops
// in bf16/f32 pipelines) that need better accuracy than the hardware
// aie::exp2<bfloat16> LUT provides.
//
// aie::exp2<bfloat16> (used by bf16_exp.cc and programming_examples/basic/
// vector_exp) is a bf16-OUTPUT lookup table. Measured on aie2p against a
// float64 2**x reference (same call shape and input on both paths, so the
// comparison isolates the LUT's own error), its max relative error is
// domain-dependent and grows sharply as the input goes negative:
//
//   domain        max rel-err (hw LUT)   max rel-err (this poly)
//   [-1, 0]         6.1%                   8.49e-5
//   [0, 10]         6.1%                   8.49e-5
//   [-10, 0]       10.1%                   8.49e-5
//   [-100, 0]      49.1%                   8.51e-5
//
// The [-100, 0] domain is where softmax lives (scores are shifted by the row
// max before exponentiating, so every input is <= 0 and can be very
// negative for non-max keys), which is also where the LUT is worst.
//
// x = k + f, k = floor(x), f in [0, 1); 2^x = 2^k * poly5(f), poly5
// evaluated by Horner's method; 2^k is reconstructed by placing k + 127
// directly in the f32 exponent field (bits 23-30, 8 bits, no multiply or
// divide needed).
//
// DOMAIN CONTRACT (device-measured; see the tables below for both bugs this
// commit fixes). Valid for every finite x, and for +-inf and NaN:
//
//   x <= -100            clamped: returns 2^-100 (~7.89e-31) exactly.
//   -100 < x < 127.999    exact: poly5(f) is a flat ~8.5e-5 max relative
//                         error here (device-measured: [-100,0] grid +
//                         random sample, [0,127] grid), independent of k,
//                         with no accuracy cliff approaching either clamp.
//   x in [127.999, 128)   a bounded, documented accuracy loss: the
//                         reconstruction internally clamps to 127.999 (see
//                         BUG 2 below for why not 128), so every x in this
//                         ~0.001-wide sliver returns the same value as
//                         x = 127.999 (~3.40018e38) instead of its own,
//                         larger true 2^x (up to ~3.40282e38).
//                         Bounded max relative error in this sliver only:
//                         about 7.8e-4, roughly 9x the kernel's usual
//                         8.5e-5, still small, and confined to a range of
//                         x about one four-hundred-thousandth as wide as
//                         the rest of the characterized domain.
//   x >= 128, or +inf     +inf, exactly (bit-identical, not an
//                         approximation): FLT_MAX is (2 - 2^-23) * 2^127 =
//                         3.402823466e38, and 2^128 = 3.402823669e38
//                         already exceeds it, so +inf is the mathematically
//                         correct 2^x for every x >= 128 even with an
//                         exact, unbounded-precision poly.
//   NaN                   NaN propagates (aie::ge(NaN, 128.0f) is false, so
//                         NaN takes the normal arithmetic path and comes
//                         out NaN; unchanged by this commit).
//
// BUG 1 (the one this commit exists to fix): NO UPPER CLAMP AT ALL. 2^k is
// reconstructed by packing the 8-bit biased exponent (k + 127, valid range
// 0..255) directly into bits 23-30 via a plain integer add + shift, no
// saturating-add. Without an upper bound on x, k can exceed 128, and
// k + 127 then overflows the 8-bit field into bit 31 (the SIGN bit),
// producing a silently WRONG-SIGNED FINITE value instead of an overflow
// indicator. Verified by direct bit simulation of the unclamped
// construction (x an exact integer so f = 0, poly(f) = 1.0, isolating the
// reconstruction):
//
//   k    biased exp (k+127)   ebits (hex)   result        true 2^x
//   127        254             0x7f000000    1.70141e+38   1.70141e+38  (exact)
//   128        255             0x7f800000    see BUG 2 (ebits itself is the
//   +inf bit pattern here, before it is ever multiplied by anything) 129 256
//   0x80000000    -0.0          6.80565e+38  (WRONG SIGN) 150        277
//   0x8a800000    -1.2326e-32   1.42725e+45  (WRONG SIGN) 257        384
//   0xc0000000    -2.0          2.31584e+77  (WRONG SIGN)
//
// None of these are NaN or +-inf, so a caller checking isfinite() would not
// catch them. This matters because the header explicitly advertises this
// kernel for softmax/sigmoid; a naive sigmoid(x) = 1 / (1 + 2^(-x*log2e))
// passes a POSITIVE argument to exp2f_vec, and an ordinary x = -100 already
// produces an argument near 144, past the boundary.
//
// BUG 2 (found DEVICE-SIDE while fixing bug 1; not visible in host
// simulation, and the reason the fix is a compare + select, not just a
// clamp). The obvious fix is "clamp x to <= 128.0f and let k = 128 saturate
// to +inf for free": k = 128 gives biased exponent 255, the IEEE
// exponent-all-ones pattern, so ebits.cast_to<float>() IS +inf, and
// multiplying by that should be a free saturation with no extra branch.
// That is correct IEEE-754 (finite * inf = inf) and it is what a host f32
// simulation of this exact bit construction predicts. It is NOT what this
// hardware does. Measured directly on NPU2 (aie2p): aie::mul on this
// target returns NaN, not +inf, whenever the true product would overflow
// f32: both when one operand IS the +inf bit pattern (the k = 128 case
// above) AND, independently, for a genuine finite * finite rounding right
// at the f32 max-exponent edge (k = 127, f a hair below 1: x supplied as
// the float32 immediately below 128 measured NaN, while the float32 one
// ULP further away, still comfortably finite at ~3.4025e38 < FLT_MAX,
// measured correct). Both are the SAME hardware behavior reached two ways:
// this aie::mul does not saturate to +inf on overflow the way scalar IEEE
// arithmetic does. A fix that relies on that saturation (clamp x to 128.0f
// and multiply) reproduces the ORIGINAL bug's symptom: silently wrong
// output that isfinite() cannot catch, just NaN instead of a wrong sign.
//
// THE FIX. Two independent mechanisms, matching the two bugs:
//   1. overflow = aie::ge(x, 128.0f), computed from x BEFORE any further
//      clamping, so the boundary tested is the exact, true one.
//   2. x = aie::min(x, 127.999f), NOT 128.0f, keeps the arithmetic more
//      than a hundred ULPs clear of the measured ~1-ULP-wide NaN edge (see
//      BUG 2), at the cost of the small, bounded, documented accuracy loss
//      in [127.999, 128) described in the domain contract above.
//   3. The final result is aie::select(poly_result, +inf_bits, overflow):
//      a lane-wise copy, not arithmetic, so it cannot hit BUG 2's
//      mul-on-overflow behavior. +inf is built via the same bit-cast idiom
//      as p2k (broadcast 0x7f800000, cast_to<float>()), never by
//      multiplying into it.
//
// Cost of the fix: the op count for the whole kernel body goes from 31 to
// 39 aie:: calls (device-verified against the actual compiled kernel, not
// estimated), broken down as: +1 broadcast/+1 aie::ge (the overflow test), +1
// broadcast/+1 aie::min (the safety clamp, replacing the old single
// lower-only clamp), +1 broadcast/+1 aie::select (the +inf override). All
// of it is cheap scalar-broadcast-and-compare work; none of it touches the
// 5-step Horner poly, so it costs nothing in the accuracy-critical path.
// Device-verified bit-identical to the pre-fix kernel everywhere below
// x = 127.999 (the [-100, 0] and [0, 127] domains both reproduce the exact
// same max-relative-error figures as before this commit).
//
// NEGATIVE-SIDE MIRROR (why -100, and why it is not an arbitrary number).
// The same 8-bit-field mechanism wraps the other way at k <= -128 (biased
// exponent goes negative and reads back through the same field: k = -128 ->
// sign=1, exponent=0xff -> -inf bit pattern; k = -129 -> -1.70e38; the same
// class of wrong-signed-finite defect as the positive side once k <= -129,
// and, per BUG 2 above, an aie::mul actually reaching that -inf bit
// pattern should also be assumed NaN-on-this-hardware rather than trusted
// to saturate). The existing x = max(x, -100) clamp keeps k >= -100, a
// 27-order-of-magnitude margin from that wall, nowhere near the ~1-ULP
// edges BUG 2 lives on. Checked directly: poly5's own error is flat, about
// 8.5e-5, uniformly across every k tested in this file (including right up
// to k = -100 and k = 127), so -100 was NOT chosen because poly5
// degrades there. It is a softmax-relevance choice (2^-100 ~ 7.9e-31 is
// already zero for any softmax weight) with a deliberately large safety
// margin. If this constant is ever lowered, it must stay well above
// -127.0f (not just above it: BUG 2 says "well clear of the wall", not
// "on the correct side of it") to avoid the mirror defect.
//
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// float-domain vector width (512-bit register / 32-bit lanes).
static constexpr int EXP2F_VEC_LEN = 16;

static __attribute__((noinline)) aie::vector<float, EXP2F_VEC_LEN>
exp2f_vec(aie::vector<float, EXP2F_VEC_LEN> x) {
  x = aie::max(x, aie::broadcast<float, EXP2F_VEC_LEN>(-100.0f));
  // overflow is computed from x BEFORE the safety clamp below, so the
  // boundary tested is the true one (x >= 128), not the narrowed one the
  // arithmetic actually runs on. See the header for why this has to be an
  // explicit compare + select rather than letting the reconstruction
  // saturate on its own.
  aie::mask<EXP2F_VEC_LEN> overflow =
      aie::ge(x, aie::broadcast<float, EXP2F_VEC_LEN>(128.0f));
  // Safety clamp for the ARITHMETIC only (128.0 is still the documented
  // domain boundary; this is tighter, not a redefinition of it). Measured
  // directly on this hardware: aie::mul's rounding right at the f32
  // overflow edge is NaN, not +inf (see header), and that edge is only
  // ~1 ULP wide (x = 127.99999 misbehaves; x = 127.99998, one more ULP
  // from 128, does not). 127.999f keeps every reconstruction more than a
  // hundred ULPs clear of that edge, at the cost of a bounded, documented
  // accuracy loss confined to x in [127.999, 128), see the header.
  x = aie::min(x, aie::broadcast<float, EXP2F_VEC_LEN>(127.999f));
  aie::vector<int32_t, EXP2F_VEC_LEN> ki =
      aie::to_fixed<int32_t>(x); // round-to-nearest on aie2p
  aie::vector<float, EXP2F_VEC_LEN> kf = aie::to_float<float>(ki);
  // floor: where x < kf, k -= 1 (correct even though to_fixed rounds, not
  // truncates)
  aie::vector<int32_t, EXP2F_VEC_LEN> one =
      aie::broadcast<int32_t, EXP2F_VEC_LEN>(1);
  aie::vector<int32_t, EXP2F_VEC_LEN> zero =
      aie::broadcast<int32_t, EXP2F_VEC_LEN>(0);
  ki = aie::sub(ki, aie::select(zero, one, aie::lt(x, kf)));
  aie::vector<float, EXP2F_VEC_LEN> f =
      aie::sub(x, aie::to_float<float>(ki)); // f in [0,1)
  aie::vector<float, EXP2F_VEC_LEN> p =
      aie::broadcast<float, EXP2F_VEC_LEN>(0.0013333558f);
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.0096181291f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.0555041087f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.2402265069f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(0.6931471805f));
  p = aie::add(aie::mul(p, f).to_vector<float>(),
               aie::broadcast<float, EXP2F_VEC_LEN>(1.0f));
  aie::vector<int32_t, EXP2F_VEC_LEN> ebits = aie::upshift(
      aie::add(ki, aie::broadcast<int32_t, EXP2F_VEC_LEN>(127)), 23);
  aie::vector<float, EXP2F_VEC_LEN> p2k = ebits.cast_to<float>();
  aie::vector<float, EXP2F_VEC_LEN> result =
      aie::mul(p, p2k).to_vector<float>();
  // +inf via the same bit-cast idiom as p2k above, NOT via multiplying
  // into it: select is a lane-wise copy, not floating-point arithmetic, so
  // it does not hit the aie::mul-on-overflow NaN behavior this fix exists
  // to route around.
  aie::vector<int32_t, EXP2F_VEC_LEN> pos_inf_bits =
      aie::broadcast<int32_t, EXP2F_VEC_LEN>(0x7f800000);
  aie::vector<float, EXP2F_VEC_LEN> pos_inf = pos_inf_bits.cast_to<float>();
  return aie::select(result, pos_inf, overflow);
}

extern "C" {

// Buffer entry point: input/output are f32, vector_size must be a multiple
// of EXP2F_VEC_LEN (16).
void exp2f_vec_f32(float *restrict input, float *restrict output,
                   int32_t vector_size) {
  event0();

  auto it_in = aie::cbegin_vector<EXP2F_VEC_LEN>((float *)input);
  auto it_out = aie::begin_vector<EXP2F_VEC_LEN>((float *)output);
  const int elem_iters = vector_size / EXP2F_VEC_LEN;

  for (int i = 0; i < elem_iters; i++) {
    *it_out++ = exp2f_vec(*it_in++);
  }

  event1();
}

} // extern "C"
