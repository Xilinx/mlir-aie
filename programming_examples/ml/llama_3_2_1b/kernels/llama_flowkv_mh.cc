//===- llama_flowkv_mh.cc -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Multi-head GQA flowkv -- REP Q heads sharing 1 KV head, fused qk+sv per
// head in one kernel call. Arithmetic mirrors llama_flowkv.cc's qk+sv
// (sw_recip/round_to_i8 copied verbatim), but takes the softmax's exp from
// exp_nonpos, as llama_attn_block does, where llama_flowkv.cc indexes an exp
// LUT; numpy_attention_perslot_flowkv mirrors it. q_chunk carries REP*kHD
// body + a REP*8 B per-head [q_scale, sv_inv_out_scale] tail; k_one/v_one each
// carry a 4 B scale + kT*kHD B cache. One worker's cache (k+v, T=128 HD=64)
// fits the 64 KB AIE2P L1 budget comfortably.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#ifndef LLAMA_FLOWKV_HEAD_DIM
#define LLAMA_FLOWKV_HEAD_DIM 64
#endif
#ifndef LLAMA_FLOWKV_T
#define LLAMA_FLOWKV_T 128
#endif
#ifndef LLAMA_FLOWKV_REP
#define LLAMA_FLOWKV_REP 4
#endif

static constexpr int kHD = LLAMA_FLOWKV_HEAD_DIM;
static constexpr int kT = LLAMA_FLOWKV_T;
static constexpr int kREP = LLAMA_FLOWKV_REP;

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int8_t round_to_i8(float v) {
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > I8_MAX)
    r = I8_MAX;
  if (r < I8_MIN)
    r = I8_MIN;
  return (int8_t)r;
}

// llama_attn_block.cc's vec_exp_nonpos on one lane (e^x for x <= 0): with
// t = x * log2(e), k = trunc(t) and f = t - k, e^x = 2^k * 2^-g for g = -f in
// Q23 (from 1 - f), 2^-g a degree-6 Q29 polynomial whose Horner products
// round half to even. 0 below t = -125.
static inline float exp_nonpos(float x) {
  static constexpr int32_t kC[] = {-684167,   5142538,    -29791239,
                                   128969442, -372130494, 536870912};
  float t = x * 1.442695022f;
  if (t < -125.0f)
    return 0.0f;
  int32_t k = (int32_t)t;
  float f = t - (float)k;
  float u = 1.0f - f;
  int32_t g;
  memcpy(&g, &u, 4);
  g -= 0x3f800000;
  int32_t p = 58465;
  for (int i = 0; i < 6; i++) {
    int64_t m = (int64_t)p * g;
    int64_t q = m >> 23;
    int64_t r = m & ((1 << 23) - 1);
    if (r > (1 << 22) || (r == (1 << 22) && (q & 1)))
      q += 1;
    p = (int32_t)q + kC[i];
  }
  float e = (float)p * (1.0f / 536870912.0f);
  int32_t bits;
  memcpy(&bits, &e, 4);
  bits += k << 23;
  memcpy(&e, &bits, 4);
  return e;
}

// Software fp32 reciprocal (see llama_flowkv.cc for why Peano's `/` needs
// this workaround).
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

// Forward decl of the inner kernel with explicit T_used (causal-mask
// support). The combined wrapper reads T_used + per-slot scale arrays from
// kv_one's prefix and forwards. k_slot/v_slot are per-cache-slot fp32 scale
// arrays (one entry per cached position) -- each cached K/V position is
// dequantized by ITS OWN scale (fixes the per-head-scalar bug where the whole
// cache was dequantized with the latest token's scale).
extern "C" void llama_flowkv_mh(int8_t *restrict q_chunk,
                                float *restrict k_slot, int8_t *restrict k_body,
                                float *restrict v_slot, int8_t *restrict v_body,
                                int8_t *restrict out_chunk, int T_used);

// Per-slot kv_one layout (per KV head):
//   [0..4]:                          T_used (int32)
//   [4..8]:                          pad (so the scale region starts at 8)
//   [8 .. 8+kT*4]:                   k_slot_scales (kT fp32)
//   [8+kT*4 .. 8+kT*4+kT*kHD]:       k body (kT * kHD int8)
//   [.. + kT*4]:                     v_slot_scales (kT fp32)
//   [.. + kT*kHD]:                   v body (kT * kHD int8)
// Total per head = 8 + 2*(kT*4 + kT*kHD).
constexpr int kTUsedPrefix = 8;
extern "C" void llama_flowkv_mh_kvc(int8_t *restrict q_chunk,
                                    int8_t *restrict kv_one,
                                    int8_t *restrict out_chunk) {
  int T_used;
  memcpy(&T_used, kv_one, 4);
  if (T_used <= 0)
    T_used = 1;
  if (T_used > kT)
    T_used = kT;
  constexpr int kSlotBytes = kT * 4;
  constexpr int kBodyBytes = kT * kHD;
  constexpr int kKHalf = kSlotBytes + kBodyBytes;
  int8_t *k_region = kv_one + kTUsedPrefix;
  int8_t *v_region = k_region + kKHalf;
  llama_flowkv_mh(q_chunk, reinterpret_cast<float *>(k_region),
                  k_region + kSlotBytes, reinterpret_cast<float *>(v_region),
                  v_region + kSlotBytes, out_chunk, T_used);
}

extern "C" void llama_flowkv_mh(int8_t *restrict q_chunk,
                                float *restrict k_slot, int8_t *restrict k_body,
                                float *restrict v_slot, int8_t *restrict v_body,
                                int8_t *restrict out_chunk, int T_used) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  static_assert(kHD == 64, "qk_scale hardcoded for head_dim=64");
  constexpr float kInvSqrtHD = 0.125f;

  // Per-head scale tail layout: tail[h*8..h*8+4] = q_scale_h,
  //                             tail[h*8+4..h*8+8] = sv_inv_out_scale_h.
  int8_t *tail = q_chunk + kREP * kHD;

  // Declared once, reused (overwritten) across all REP iterations. ebits
  // holds exp_v as int32 bits, not floats: the same stack-corruption
  // workaround llama_flowkv.cc uses -- see its comment there.
  float scores[kT];
  int32_t ebits[kT];

  for (int h = 0; h < kREP; h++) {
    float q_scale, sv_inv_out_scale;
    memcpy(&q_scale, tail + h * 8 + 0, 4);
    memcpy(&sv_inv_out_scale, tail + h * 8 + 4, 4);

    int8_t *restrict q_h = q_chunk + h * kHD;
    int8_t *restrict out_h = out_chunk + h * kHD;

    // 1) Scores: int32 dot (exact -- see llama_flowkv.cc's vectorized dot
    // for why reordering this reduction can't change the result) -> per-slot
    // combined float multiply. Each cached position uses its OWN k_slot[i]
    // scale (the per-slot KV fix). Causal mask: only the first T_used
    // positions. Scale order (q_scale*k_slot[i]) * kInvSqrtHD mirrors the
    // numpy reference (and the per-head predecessor's q_scale*k_scale*
    // kInvSqrtHD).
    // k_body may not be 64-byte aligned (kv_one's prefix/slot layout), so
    // its load is unaligned; q_h is always a multiple of kHD from q_chunk.
    float max_s = -1e30f;
    ::aie::vector<int8, kHD> qv = ::aie::load_v<kHD>(q_h);
    for (int i = 0; i < T_used; i++) {
      ::aie::vector<int8, kHD> kv =
          ::aie::load_unaligned_v<kHD>(k_body + i * kHD);
      ::aie::accum<acc32, kHD> dot_acc;
      dot_acc.from_vector(::aie::zeros<int32, kHD>());
      dot_acc = ::aie::mac(dot_acc, qv, kv);
      int32_t dot = ::aie::reduce_add(dot_acc.to_vector<int32>());
      float s = (float)dot * ((q_scale * k_slot[i]) * kInvSqrtHD);
      scores[i] = s;
      if (s > max_s)
        max_s = s;
    }

    // 2) Shift, exp in place, sum.
    float sum = 0.0f;
    for (int i = 0; i < T_used; i++) {
      const float e = exp_nonpos(scores[i] - max_s);
      ebits[i] = __builtin_bit_cast(int32_t, e);
      sum += e;
    }
    const float inv_sum = sw_recip(sum);

    // 3) sv: probs @ V then requant, kHD lanes at a time. i (the KV
    // position) stays the serial outer loop, same discipline as
    // llama_flowkv.cc's vectorized sv. p*v_slot[i] is the same combined
    // per-position factor the scalar code recomputed redundantly inside
    // the j loop; computing it once per i is value-identical, just not
    // wasteful.
    ::aie::accum<accfloat, kHD> acc;
    acc.from_vector(::aie::zeros<float, kHD>());
    for (int i = 0; i < T_used; i++) {
      float p = __builtin_bit_cast(float, ebits[i]) * inv_sum;
      float pv_i = p * v_slot[i];
      ::aie::vector<int8, kHD> vv =
          ::aie::load_unaligned_v<kHD>(v_body + i * kHD);
      ::aie::vector<float, kHD> v_f = ::aie::to_float<float>(vv);
      ::aie::vector<float, kHD> pv_v = ::aie::broadcast<float, kHD>(pv_i);
      acc = ::aie::mac(acc, v_f, pv_v);
    }
    ::aie::vector<float, kHD> acc_f = acc.to_vector<float>();
    ::aie::vector<float, kHD> sv_inv_out_scale_v =
        ::aie::broadcast<float, kHD>(sv_inv_out_scale);
    ::aie::vector<float, kHD> out_f =
        ::aie::mul(acc_f, sv_inv_out_scale_v).template to_vector<float>();
    for (int j = 0; j < kHD; j++)
      out_h[j] = round_to_i8(out_f[j]);
  }

  event1();
}

// Self-calibrating inner: computes each Q head's sv_out_scale ON-CHIP (absmax
// of the head's fp32 sv) instead of reading sv_inv_out_scale from the q_chunk
// tail. Writes the per-head sv_out_scale to the out_chunk tail at
// out_chunk[kREP*kHD + h*4] (out_chunk is REP*kHD + REP*4 bytes) so the
// downstream sv_merge/af_concat read sv_out_scales from the merged sv tail
// instead of the host scales buffer. q_scale per head still from the q_chunk
// tail (q_proj's per-head scale; that's a Phase-2 tiled scale).
// Two-pass over the sv accumulation (Bug 2: no fp32 stack array across loops;
// recompute acc in pass B). The doubled sv cost is acceptable (correctness).
extern "C" void
llama_flowkv_mh_selfcal(int8_t *restrict q_chunk, float *restrict k_slot,
                        int8_t *restrict k_body, float *restrict v_slot,
                        int8_t *restrict v_body, int8_t *restrict out_chunk,
                        int T_used) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  static_assert(kHD == 64, "qk_scale hardcoded for head_dim=64");
  constexpr float kInvSqrtHD = 0.125f;

  int8_t *tail = q_chunk + kREP * kHD;       // per-head [q_scale, sv_inv_out]
  int8_t *out_tail = out_chunk + kREP * kHD; // per-head sv_out_scale (REP*4)

  float scores[kT];
  int32_t ebits[kT];

  for (int h = 0; h < kREP; h++) {
    float q_scale;
    memcpy(&q_scale, tail + h * 8 + 0, 4); // sv_inv_out (tail+4) now ignored

    int8_t *restrict q_h = q_chunk + h * kHD;
    int8_t *restrict out_h = out_chunk + h * kHD;

    // 1) scores (identical to llama_flowkv_mh).
    float max_s = -1e30f;
    ::aie::vector<int8, kHD> qv = ::aie::load_v<kHD>(q_h);
    for (int i = 0; i < T_used; i++) {
      ::aie::vector<int8, kHD> kv =
          ::aie::load_unaligned_v<kHD>(k_body + i * kHD);
      ::aie::accum<acc32, kHD> dot_acc;
      dot_acc.from_vector(::aie::zeros<int32, kHD>());
      dot_acc = ::aie::mac(dot_acc, qv, kv);
      int32_t dot = ::aie::reduce_add(dot_acc.to_vector<int32>());
      float s = (float)dot * ((q_scale * k_slot[i]) * kInvSqrtHD);
      scores[i] = s;
      if (s > max_s)
        max_s = s;
    }

    // 2) shift, exp in place, sum.
    float sum = 0.0f;
    for (int i = 0; i < T_used; i++) {
      const float e = exp_nonpos(scores[i] - max_s);
      ebits[i] = __builtin_bit_cast(int32_t, e);
      sum += e;
    }
    const float inv_sum = sw_recip(sum);

    // 3a) Pass A: compute sv acc[j] on the fly, kHD lanes at a time, track
    // absmax (see llama_rmsnorm_int8_dyn for why no fp32 array is kept
    // across loops here).
    float absmax = 0.0f;
    {
      ::aie::accum<accfloat, kHD> acc;
      acc.from_vector(::aie::zeros<float, kHD>());
      for (int i = 0; i < T_used; i++) {
        float p = __builtin_bit_cast(float, ebits[i]) * inv_sum;
        float pv_i = p * v_slot[i];
        ::aie::vector<int8, kHD> vv =
            ::aie::load_unaligned_v<kHD>(v_body + i * kHD);
        ::aie::vector<float, kHD> v_f = ::aie::to_float<float>(vv);
        ::aie::vector<float, kHD> pv_v = ::aie::broadcast<float, kHD>(pv_i);
        acc = ::aie::mac(acc, v_f, pv_v);
      }
      ::aie::vector<float, kHD> acc_f = acc.to_vector<float>();
      absmax = ::aie::reduce_max(::aie::abs(acc_f));
    }
    if (absmax < 1e-12f)
      absmax = 1e-12f;
    float sv_out_scale = absmax * (1.0f / 127.0f);
    float sv_inv_out_scale = sw_recip(sv_out_scale);

    // 3b) Pass B: recompute acc[j], requant with the self-computed scale.
    {
      ::aie::accum<accfloat, kHD> acc;
      acc.from_vector(::aie::zeros<float, kHD>());
      for (int i = 0; i < T_used; i++) {
        float p = __builtin_bit_cast(float, ebits[i]) * inv_sum;
        float pv_i = p * v_slot[i];
        ::aie::vector<int8, kHD> vv =
            ::aie::load_unaligned_v<kHD>(v_body + i * kHD);
        ::aie::vector<float, kHD> v_f = ::aie::to_float<float>(vv);
        ::aie::vector<float, kHD> pv_v = ::aie::broadcast<float, kHD>(pv_i);
        acc = ::aie::mac(acc, v_f, pv_v);
      }
      ::aie::vector<float, kHD> acc_f = acc.to_vector<float>();
      ::aie::vector<float, kHD> sv_inv_out_scale_v =
          ::aie::broadcast<float, kHD>(sv_inv_out_scale);
      ::aie::vector<float, kHD> out_f =
          ::aie::mul(acc_f, sv_inv_out_scale_v).template to_vector<float>();
      for (int j = 0; j < kHD; j++)
        out_h[j] = round_to_i8(out_f[j]);
    }
    memcpy(out_tail + h * 4, &sv_out_scale, 4);
  }

  event1();
}

// Self-calibrating kvc wrapper (mirrors llama_flowkv_mh_kvc but calls the
// selfcal inner). out_chunk is REP*kHD + REP*4.
extern "C" void llama_flowkv_mh_kvc_selfcal(int8_t *restrict q_chunk,
                                            int8_t *restrict kv_one,
                                            int8_t *restrict out_chunk) {
  int T_used;
  memcpy(&T_used, kv_one, 4);
  if (T_used <= 0)
    T_used = 1;
  if (T_used > kT)
    T_used = kT;
  constexpr int kSlotBytes = kT * 4;
  constexpr int kBodyBytes = kT * kHD;
  constexpr int kKHalf = kSlotBytes + kBodyBytes;
  int8_t *k_region = kv_one + kTUsedPrefix;
  int8_t *v_region = k_region + kKHalf;
  llama_flowkv_mh_selfcal(q_chunk, reinterpret_cast<float *>(k_region),
                          k_region + kSlotBytes,
                          reinterpret_cast<float *>(v_region),
                          v_region + kSlotBytes, out_chunk, T_used);
}
