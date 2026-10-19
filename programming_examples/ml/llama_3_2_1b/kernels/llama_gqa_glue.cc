//===- llama_gqa_glue.cc ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Glue kernels for the multi-head layer: q_requant (per-head self-calibrated
// int8 Q), qkv_combine (per-KV-head chunks for the on-chip KV append),
// sv_merge_selfcal (de-interleave the attention output) and af_concat
// (per-Q-head dequant + global self-calibrated requant), mirroring
// numpy_layer_mh's steps.
//===---------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#ifndef LLAMA_GQA_HEAD_DIM
#define LLAMA_GQA_HEAD_DIM 64
#endif
#ifndef LLAMA_GQA_N_HEADS_Q
#define LLAMA_GQA_N_HEADS_Q 32
#endif
#ifndef LLAMA_GQA_N_HEADS_KV
#define LLAMA_GQA_N_HEADS_KV 8
#endif

static constexpr int kHD = LLAMA_GQA_HEAD_DIM;
static constexpr int kNHeadsQ = LLAMA_GQA_N_HEADS_Q;
static constexpr int kNHeadsKV = LLAMA_GQA_N_HEADS_KV;
static constexpr int kREP = kNHeadsQ / kNHeadsKV;
static constexpr int kQD = kNHeadsQ * kHD;             // 2048
static constexpr int kBodyChunk = kREP * kHD;          // 256
static constexpr int kTailChunk = kREP * 8;            // 32
static constexpr int kChunk = kBodyChunk + kTailChunk; // 288
static constexpr int kHalfKV = kNHeadsKV / 2;          // 4

// See llama_rescale_add.cc for the lane width, vec_round_to_i8,
// vec_mul_rne and the integer-bit absmax.
static constexpr int kVec = 16;
static_assert(kHD % kVec == 0, "LLAMA_GQA_HEAD_DIM must be a multiple of 16");

static inline ::aie::vector<int8, kVec>
vec_round_to_i8(::aie::vector<float, kVec> v) {
  ::aie::vector<int32, kVec> half =
      ::aie::select(::aie::broadcast<int32, kVec>(0x3f000000),
                    ::aie::broadcast<int32, kVec>(int32(0xbf000000)),
                    ::aie::lt(v.template cast_to<int32>(), int32(0)));
  ::aie::vector<float, kVec> t = ::aie::add(v, half.template cast_to<float>());
  ::aie::vector<int32, kVec> n = ::aie::to_fixed<int32>(t, 0);
  ::aie::vector<int32, kVec> n_bits =
      ::aie::to_float<float>(n, 0).template cast_to<int32>();
  auto away =
      ::aie::lt(::aie::bit_and(int32(0x7fffffff), t.template cast_to<int32>()),
                ::aie::bit_and(int32(0x7fffffff), n_bits));
  ::aie::vector<int32, kVec> toward_zero = ::aie::select(
      ::aie::add(n, int32(1)), ::aie::sub(n, int32(1)), ::aie::gt(n, int32(0)));
  ::aie::accum<acc32, kVec> acc;
  acc.from_vector(::aie::select(n, toward_zero, away));
  return acc.template to_vector<int8>(0);
}

static inline ::aie::vector<float, kVec>
vec_mul_rne(::aie::vector<float, kVec> a, ::aie::vector<float, kVec> b) {
  ::aie::vector<int32, kVec> ai = a.template cast_to<int32>();
  ::aie::vector<int32, kVec> bi = b.template cast_to<int32>();
  ::aie::vector<int32, kVec> ea =
      ::aie::bit_and(int32(0xff), ::aie::downshift(ai, 23));
  ::aie::vector<int32, kVec> eb =
      ::aie::bit_and(int32(0xff), ::aie::downshift(bi, 23));
  ::aie::accum<acc64, kVec> p = ::aie::mul(
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), ai)),
      ::aie::bit_or(int32(0x800000), ::aie::bit_and(int32(0x7fffff), bi)));
  // p is in [2^46, 2^48). A rounded p >> 23 reaching 2^24 means p >= 2^47,
  // or p rounds up to 2^47 where both shifts give the same float.
  ::aie::vector<int32, kVec> lo = p.template to_vector<int32>(23);
  ::aie::vector<int32, kVec> hi = p.template to_vector<int32>(24);
  auto wide = ::aie::ge(lo, int32(1 << 24));
  ::aie::vector<int32, kVec> e = ::aie::add(
      ::aie::add(ea, eb), ::aie::select(int32(-127), int32(-126), wide));
  // The significand's implicit bit carries into the exponent field.
  ::aie::vector<int32, kVec> mag = ::aie::add(
      ::aie::upshift(::aie::sub(e, int32(1)), 23), ::aie::select(lo, hi, wide));
  mag = ::aie::select(mag, ::aie::broadcast<int32, kVec>(0x7f800000),
                      ::aie::ge(e, int32(255)));
  mag = ::aie::select(mag, ::aie::zeros<int32, kVec>(),
                      ::aie::lt(::aie::min(::aie::min(ea, eb), e), int32(1)));
  return ::aie::bit_or(
             ::aie::bit_and(int32(0x80000000), ::aie::bit_xor(ai, bi)), mag)
      .template cast_to<float>();
}

// Software fp32 reciprocal (see llama_flowkv.cc); matches the other
// kernels' sw_recip so the numpy reference and device share the function.
static inline float sw_recip(float a) {
  int32_t bits;
  memcpy(&bits, &a, 4);
  bits = (int32_t)0x7EF477D5 - bits;
  float x;
  memcpy(&x, &bits, 4);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  return x;
}

// sw_recip on kVec lanes: the same seed and Newton steps, with the scalar
// form's soft-float products as vec_mul_rne and its `-` as the same vsub.f.
static inline ::aie::vector<float, kVec>
vec_sw_recip(::aie::vector<float, kVec> a) {
  ::aie::vector<float, kVec> two = ::aie::broadcast<float, kVec>(2.0f);
  ::aie::vector<float, kVec> x =
      ::aie::sub(::aie::broadcast<int32, kVec>(0x7EF477D5),
                 a.template cast_to<int32>())
          .template cast_to<float>();
  for (int i = 0; i < 4; i++)
    x = vec_mul_rne(x, ::aie::sub(two, vec_mul_rne(a, x)));
  return x;
}

static inline float bits_to_float(int32_t bits) {
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// q_requant (Phase 2a self-cal): the q_proj gemm now emits fp32[kQD]; this
// stage computes each Q head's q_out_scale ON-CHIP (absmax over the head's
// 64-wide fp32 slice) and requants to int8, writing the per-head scale tail
// rope/flowkv expect: out[kQD + h*8 + 0..4] = q_scale_h, [+4..8] = pad
// (the old sv_inv_out slot, now self-calibrated by flowkv -- kept for the 8 B
// stride those kernels hardcode). out is int8[kQD + kNHeadsQ*8].
extern "C" void llama_q_requant(float *restrict q_fp, int8_t *restrict out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  static_assert(kNHeadsQ % kVec == 0,
                "LLAMA_GQA_N_HEADS_Q must be a multiple of 16");
  // q_scale holds each head's absmax until the scales are computed in place.
  alignas(64) float q_scale[kNHeadsQ];
  alignas(64) float q_inv[kNHeadsQ];
  for (int h = 0; h < kNHeadsQ; h++) {
    const float *src = q_fp + h * kHD;
    ::aie::vector<int32, kVec> mag = ::aie::zeros<int32, kVec>();
    for (int c = 0; c < kHD / kVec; c++)
      mag = ::aie::max(
          mag,
          ::aie::bit_and(
              int32(0x7fffffff),
              ::aie::load_v<kVec>(src + c * kVec).template cast_to<int32>()));
    q_scale[h] = bits_to_float(::aie::reduce_max(mag));
  }
  // Every head's scale and reciprocal at once. max(absmax, 1e-12f) is an
  // integer max of the non-negative bits.
  for (int g = 0; g < kNHeadsQ / kVec; g++) {
    ::aie::vector<float, kVec> absmax =
        ::aie::max(
            ::aie::load_v<kVec>(q_scale + g * kVec).template cast_to<int32>(),
            int32(0x2b8cbccc))
            .template cast_to<float>();
    ::aie::vector<float, kVec> scale =
        vec_mul_rne(absmax, ::aie::broadcast<float, kVec>(1.0f / 127.0f));
    ::aie::store_v(q_scale + g * kVec, scale);
    ::aie::store_v(q_inv + g * kVec, vec_sw_recip(scale));
  }
  int8_t *tail = out + kQD;
  for (int h = 0; h < kNHeadsQ; h++) {
    const float *src = q_fp + h * kHD;
    int8_t *dst = out + h * kHD;
    ::aie::vector<float, kVec> q_inv_v =
        ::aie::broadcast<float, kVec>(q_inv[h]);
    for (int c = 0; c < kHD / kVec; c++)
      ::aie::store_v(dst + c * kVec,
                     vec_round_to_i8(vec_mul_rne(
                         ::aie::load_v<kVec>(src + c * kVec), q_inv_v)));
    memcpy(tail + h * 8 + 0, &q_scale[h], 4);
    int32_t zero = 0;
    memcpy(tail + h * 8 + 4, &zero, 4);
  }
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}

// Self-calibrating sv_merge: the selfcal flowkv writes each KV group's chunk
// as [REP*kHD body | REP*4 sv_out_scale] (272 B). The memtile join packs 4
// such chunks per half (in_lo/in_hi). This kernel DE-INTERLEAVES them into
//   out[0 .. kQD]                 = all 32 Q-head sv bodies, contiguous
//   out[kQD .. kQD + kNHeadsQ*4]  = all 32 per-Q-head sv_out_scales (fp32)
// so the downstream af_concat reads both bodies and sv_out_scales from this
// single buffer (no host af_scales fifo). Q-head order is preserved:
// KV group g holds Q heads [g*REP .. g*REP+REP).
static constexpr int kSvBody = kREP * kHD; // 256 (one KV group's body)
static constexpr int kSvTail = kREP * 4;   // 16  (one KV group's scales)
static constexpr int kSvChunk = kSvBody + kSvTail; // 272
static constexpr int kSvScalesOff = kQD;           // scales region start in out

extern "C" void llama_sv_merge_selfcal(int8_t *restrict in_lo,
                                       int8_t *restrict in_hi,
                                       int8_t *restrict out) {
  event0();
  int8_t *scales_out = out + kSvScalesOff;
  for (int g = 0; g < kNHeadsKV; g++) {
    int8_t *src = (g < kHalfKV) ? in_lo : in_hi;
    int local = (g < kHalfKV) ? g : (g - kHalfKV);
    int8_t *chunk = src + local * kSvChunk;
    memcpy(out + g * kSvBody, chunk, kSvBody); // body -> contiguous
    memcpy(scales_out + g * kSvTail, chunk + kSvBody, kSvTail); // scales
  }
  event1();
}

// qkv_combine (on-chip KV append): builds per-head COMBINED chunks
// [q_chunk 288 B | k_fp 256 B | v_fp 256 B | cs 256 B] (1056 B/head) from
// rope's q output (288 B/head: 256 B body + 32 B scale tail) and a
// pre-joined kvcs buffer (kfp 2048 | vfp 2048 | cs 256 shared B, 8 heads
// of 64 fp32 each). Split into lo (heads 0..3) / hi (4..7) to keep memtile
// fanout <= 4.
static constexpr int kKfpHead = kHD * 4;     // 256 (64 fp32 for one head)
static constexpr int kCsBytes = 2 * kHD * 2; // 256 (cos+sin bf16)
static constexpr int kCombChunk = kChunk + 2 * kKfpHead + kCsBytes; // 1056

extern "C" void llama_qkv_combine(int8_t *restrict qr, int8_t *restrict kvcs,
                                  int8_t *restrict out_lo,
                                  int8_t *restrict out_hi) {
  event0();
  constexpr int kKfpRegion = kNHeadsKV * kKfpHead; // 8*256 = 2048
  int8_t *qr_body = qr;
  int8_t *qr_tail = qr + kQD;
  int8_t *kfp = kvcs;
  int8_t *vfp = kvcs + kKfpRegion;
  int8_t *cs = kvcs + 2 * kKfpRegion;
  for (int g = 0; g < kNHeadsKV; g++) {
    int8_t *base = (g < kHalfKV) ? out_lo : out_hi;
    int local = (g < kHalfKV) ? g : (g - kHalfKV);
    int8_t *dst = base + local * kCombChunk;
    // q_chunk: body (256) + tail (32)
    memcpy(dst, qr_body + g * kBodyChunk, kBodyChunk);
    memcpy(dst + kBodyChunk, qr_tail + g * kTailChunk, kTailChunk);
    // k_fp, v_fp, cs
    memcpy(dst + kChunk, kfp + g * kKfpHead, kKfpHead);
    memcpy(dst + kChunk + kKfpHead, vfp + g * kKfpHead, kKfpHead);
    memcpy(dst + kChunk + 2 * kKfpHead, cs, kCsBytes);
  }
  event1();
}

// af_concat, fully self-calibrating: sv_out_scales come from the merged sv
// buffer tail (written by sv_merge_selfcal), and the global o_act_scale is
// computed on-chip (absmax of af_fp = sv[h]*sv_out_scale[h] over Q_DIM) and
// written to af_out[kOutDim..kOutDim+4] (af_out is int8[kOutDim+8]) for the
// downstream o_proj. Input `sv` = [kQD bodies | kNHeadsQ*4 sv_out_scales].
extern "C" void llama_af_concat_selfcal2(int8_t *restrict sv,
                                         int8_t *restrict af_out) {
  event0();
  const ::aie::saturation_mode prev_sat =
      ::aie::swap_saturation(::aie::saturation_mode::saturate);
  const ::aie::rounding_mode prev_rnd =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  constexpr int kOutDim = kNHeadsQ * kHD;
  const float *sv_out_scales = reinterpret_cast<const float *>(sv + kOutDim);

  // Pass A: absmax of af_fp = sv[h]*sv_out_scale[h].
  ::aie::vector<int32, kVec> mag = ::aie::zeros<int32, kVec>();
  for (int h = 0; h < kNHeadsQ; h++) {
    ::aie::vector<float, kVec> s_v =
        ::aie::broadcast<float, kVec>(sv_out_scales[h]);
    int8_t *src = sv + h * kHD;
    for (int c = 0; c < kHD / kVec; c++)
      mag = ::aie::max(
          mag,
          ::aie::bit_and(int32(0x7fffffff),
                         vec_mul_rne(::aie::to_float<float>(
                                         ::aie::load_v<kVec>(src + c * kVec)),
                                     s_v)
                             .template cast_to<int32>()));
  }
  float absmax = bits_to_float(::aie::reduce_max(mag));
  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float o_act_scale = absmax * (1.0f / 127.0f);
  float o_inv_act_scale = sw_recip(o_act_scale);

  // Pass B: requant with combined = sv_out_scale[h] * o_inv_act_scale.
  alignas(64) float combined[kNHeadsQ];
  for (int g = 0; g < kNHeadsQ / kVec; g++)
    ::aie::store_v(combined + g * kVec,
                   vec_mul_rne(::aie::load_v<kVec>(sv_out_scales + g * kVec),
                               ::aie::broadcast<float, kVec>(o_inv_act_scale)));
  for (int h = 0; h < kNHeadsQ; h++) {
    ::aie::vector<float, kVec> combined_v =
        ::aie::broadcast<float, kVec>(combined[h]);
    int8_t *src = sv + h * kHD;
    int8_t *dst = af_out + h * kHD;
    for (int c = 0; c < kHD / kVec; c++)
      ::aie::store_v(
          dst + c * kVec,
          vec_round_to_i8(vec_mul_rne(
              ::aie::to_float<float>(::aie::load_v<kVec>(src + c * kVec)),
              combined_v)));
  }

  memcpy(af_out + kOutDim, &o_act_scale, 4);
  int32_t zero = 0;
  memcpy(af_out + kOutDim + 4, &zero, 4);
  ::aie::set_rounding(prev_rnd);
  ::aie::set_saturation(prev_sat);
  event1();
}
