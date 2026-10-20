//===- llama_gemm_int8_srs_tiled_layer.cc -----------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Full-single-layer tiled decode GEMV: one .o exposing every K-shape a
// single Llama 3.2 1B layer (D=2048, single-head, HD=8192) needs, with
// per-channel weight quant + per-token dynamic activation scales:
// `gemm_tile_perchan_impl` is the closure-baked-scale path (gate's
// output scale is locked to the silu LUT's gate_scale);
// `gemm_tile_perchan_v2_impl` reads dynamic scales from a 64 B slot
// prefix (smallest 64-aligned size for `aie::load_v<64>`, enforced by
// static asserts); `_v2_up_impl` also mirrors an 8 B downstream-scales
// payload into out_full[kOutDim..kOutDim+8].
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#include "llama_vec_fp.h"

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

template <int kK, int kNTile>
static inline void gemm_tile_impl(int8_t *restrict act, int8_t *restrict w_tile,
                                  int8_t *restrict out_tile,
                                  int32_t right_shift) {
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = w_tile;
  const int32_t *bias = reinterpret_cast<const int32_t *>(w_tile + kNTile * kK);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }

    int32_t sum = aie::reduce_add(acc.template to_vector<int32>()) + bias[n];
    int32_t s = banker_srs(sum, right_shift);
    if (s > I8_MAX)
      s = I8_MAX;
    if (s < I8_MIN)
      s = I8_MIN;
    out_tile[n] = (int8_t)s;
  }
}

// Slot: [N_TILE*K i8 weights | N_TILE i32 bias | N_TILE fp32 w_scales].
template <int kK, int kNTile>
static inline void
gemm_tile_perchan_impl(int8_t *restrict act, int8_t *restrict w_tile,
                       int8_t *restrict out_tile, float act_scale,
                       float inv_out_scale) {
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = w_tile;
  const int32_t *bias = reinterpret_cast<const int32_t *>(w_tile + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(w_tile + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];

    float fp = (float)sum_i32 * act_scale * w_scales[n];
    float scaled = fp * inv_out_scale;

    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (r > I8_MAX)
      r = I8_MAX;
    if (r < I8_MIN)
      r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

// Dynamic-scale variant: scales come from slot prefix.
template <int kK, int kNTile, int kPrefix = 64>
static inline void gemm_tile_perchan_v2_impl(int8_t *restrict act,
                                             int8_t *restrict w_tile,
                                             int8_t *restrict out_tile) {
  static_assert(kPrefix >= 8, "prefix must hold at least 8 B of scales");
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 to keep aie::load_v<64> "
                "of the weights aligned");
  float act_scale, inv_out_scale;
  memcpy(&act_scale, w_tile, 4);
  memcpy(&inv_out_scale, w_tile + 4, 4);

  int8_t *body = w_tile + kPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];

    float fp = (float)sum_i32 * act_scale * w_scales[n];
    float scaled = fp * inv_out_scale;

    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (r > I8_MAX)
      r = I8_MAX;
    if (r < I8_MIN)
      r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

// fp32-out variant: same MAC + bias + per-channel fp scale, but writes the
// TRUE fp32 output (no inv_out_scale / round / clamp). Used by down when the
// downstream rescale-add consumes the fp32 projection directly and computes
// the dynamic residual scale itself. out_tile is float* (already offset by the
// caller to tile_idx * kNTile).
template <int kK, int kNTile, int kPrefix = 64>
static inline void gemm_tile_perchan_v2_fp32out_impl(int8_t *restrict act,
                                                     int8_t *restrict w_tile,
                                                     float *restrict out_tile) {
  static_assert(kPrefix >= 8, "prefix must hold at least 8 B of scales");
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 to keep aie::load_v<64> "
                "of the weights aligned");
  float act_scale;
  memcpy(&act_scale, w_tile, 4);

  int8_t *body = w_tile + kPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];

    out_tile[n] = (float)sum_i32 * act_scale * w_scales[n];
  }
}

// fp32-out, activation-tail act_scale: act is int8[kK+8] with the per-token
// scale in act[kK..kK+4] (e.g. the self-calibrated silu sf tail). Used by the
// down gemm when silu self-calibrates its output scale.
template <int kK, int kNTile, int kPrefix = 64>
static inline void gemm_tile_perchan_v2_fp32out_acttail_impl(
    int8_t *restrict act, int8_t *restrict w_tile, float *restrict out_tile) {
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 for aligned weight loads");
  float act_scale = *reinterpret_cast<const float *>(act + kK);

  int8_t *body = w_tile + kPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0, "K must be a multiple of 64");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  int32_t sums[kNTile];
  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());
    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    sums[n] = aie::reduce_add(acc.template to_vector<int32>()) + bias[n];
  }

  if (vec_epilogue_exact<80, 174, kNTile>(sums, act_scale, w_scales)) {
    ::aie::vector<float, 16> fp =
        vec_scaled_sums<kNTile>(sums, act_scale, w_scales);
    for (int n = 0; n < kNTile; n++)
      out_tile[n] = fp.get(n);
  } else {
    for (int n = 0; n < kNTile; n++)
      out_tile[n] = (float)sums[n] * act_scale * w_scales[n];
  }
}

// Dynamic-scale + downstream-scale mirror variant: slot prefix bytes
// [8..16] are 8 B of "downstream" scales that the kernel mirrors into
// out_full[kOutDim..kOutDim+8] each iter so the next consumer can pick
// them up from the output buffer's tail.
template <int kK, int kNTile, int kOutDim, int kPrefix = 64>
static inline void
gemm_tile_perchan_v2_up_impl(int8_t *restrict act, int8_t *restrict w_tile,
                             int8_t *restrict out_tile, int8_t *out_full_base) {
  static_assert(kPrefix >= 16,
                "v2_up prefix must hold 8 B own + 8 B downstream");
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 to keep aie::load_v<64> "
                "of the weights aligned");
  float act_scale, inv_out_scale;
  memcpy(&act_scale, w_tile, 4);
  memcpy(&inv_out_scale, w_tile + 4, 4);
  // downstream scales sit at slot[8..16]; copy them through to out tail.
  memcpy(out_full_base + kOutDim, w_tile + 8, 8);

  int8_t *body = w_tile + kPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];

    float fp = (float)sum_i32 * act_scale * w_scales[n];
    float scaled = fp * inv_out_scale;

    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (r > I8_MAX)
      r = I8_MAX;
    if (r < I8_MIN)
      r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

// Activation-tail variants: act_scale is read from the ACTIVATION buffer
// tail (act[kK..kK+4]) instead of the weight-slot prefix. The dyn rmsnorm
// upstream writes its per-token scale there; this is the on-device path
// that replaces the host-baked static rmsnorm act_scale for q/gate/up.
// The activation buffer must be int8[kK + 8] (4 B scale + 4 B pad). The
// MAC loop only reads act[0..kK], so the tail is untouched by the dot.

// Gate-style: per-channel w scale, act_scale from act tail, inv_out a kernel
// arg (locked to the silu LUT's 1/gate_scale).
template <int kK, int kNTile>
static inline void
gemm_tile_perchan_acttail_impl(int8_t *restrict act, int8_t *restrict w_tile,
                               int8_t *restrict out_tile, float inv_out_scale) {
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  float act_scale = *reinterpret_cast<const float *>(act + kK);

  const int8_t *weights = w_tile;
  const int32_t *bias = reinterpret_cast<const int32_t *>(w_tile + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(w_tile + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  int32_t sums[kNTile];
  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    sums[n] = aie::reduce_add(acc.template to_vector<int32>()) + bias[n];
  }

  // Three products deep, so the scale exponents are held tighter than the
  // two-product epilogues' [80, 174]. At |scaled| >= 2^31 the scalar int cast
  // below is undefined, so those calls keep the scalar code too.
  if (vec_epilogue_exact<96, 158, kNTile>(sums, act_scale, w_scales,
                                          inv_out_scale)) {
    ::aie::vector<float, 16> scaled =
        vec_mul_rne(vec_scaled_sums<kNTile>(sums, act_scale, w_scales),
                    ::aie::broadcast<float, 16>(inv_out_scale));
    if (::aie::reduce_max(::aie::bit_and(int32(0x7fffffff),
                                         scaled.template cast_to<int32>())) <
        0x4f000000) {
      ::aie::vector<int8, 16> r = vec_round_to_i8(scaled);
      for (int n = 0; n < kNTile; n++)
        out_tile[n] = r.get(n);
      return;
    }
  }
  for (int n = 0; n < kNTile; n++) {
    float fp = (float)sums[n] * act_scale * w_scales[n];
    float scaled = fp * inv_out_scale;

    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (r > I8_MAX)
      r = I8_MAX;
    if (r < I8_MIN)
      r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

// up-style: act_scale from act tail; inv_out_scale + downstream-mirror bytes
// still come from the weight slot prefix (they are genuinely dynamic and
// host-baked). Mirrors prefix[8..16] into out_full[kOutDim..kOutDim+8].
template <int kK, int kNTile, int kOutDim, int kPrefix = 64>
static inline void gemm_tile_perchan_v2_up_acttail_impl(
    int8_t *restrict act, int8_t *restrict w_tile, int8_t *restrict out_tile,
    int8_t *out_full_base) {
  static_assert(kPrefix >= 16,
                "v2_up prefix must hold 8 B own + 8 B downstream");
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 to keep aie::load_v<64> "
                "of the weights aligned");
  float act_scale, inv_out_scale;
  memcpy(&act_scale, act + kK, 4);
  memcpy(&inv_out_scale, w_tile + 4, 4);
  // downstream scales sit at slot[8..16]; copy them through to out tail.
  memcpy(out_full_base + kOutDim, w_tile + 8, 8);

  int8_t *body = w_tile + kPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];

    float fp = (float)sum_i32 * act_scale * w_scales[n];
    float scaled = fp * inv_out_scale;

    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (r > I8_MAX)
      r = I8_MAX;
    if (r < I8_MIN)
      r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

// A signed key ordered as IEEE orders non-NaN floats, with -0 tying +0.
static inline int32_t logit_key(float f) {
  int32_t b = __builtin_bit_cast(int32_t, f);
  if (b == INT32_MIN)
    b = 0;
  return b ^ ((b >> 31) & 0x7fffffff);
}

// Fused lm_head GEMM + top-k insert (the one-stream sampler+gather
// front). Computes this slot's kNTile logits and, instead of writing
// to a DDR logits buffer, inserts {logit, global index, embed_sc, embed
// row} into a resident top-k set -- the embed row IS the GEMM weight
// row and embed_sc IS w_scale (tied embedding), so one table pass needs
// no second gather stream. llama_topk_finalize
// (kernels/llama_topk_sample.cc) samples the set afterward; set_*
// layout and kKset must match it exactly. Min-eviction unsorted set:
// same scheme as llama_topk_insert (see llama_topk_sample.cc).
template <int kKset>
static inline void
lmhead_topk_insert_impl(int8_t *restrict act, int8_t *restrict w_tile,
                        float *restrict set_logit, int32_t *restrict set_gidx,
                        float *restrict set_scale, int8_t *restrict set_row,
                        int32_t *restrict set_len, int32_t tile_idx) {
  constexpr int kK = 2048, kNTile = 4, kPrefix = 64, kVec = 64;
  constexpr int kGroups = kK / kVec;

  // Reset the resident set at the start of each table pass. The set buffers
  // persist across dispatches (and across iterations of the persistent loop),
  // so tile_idx==0 must clear the occupancy or stale entries leak in.
  if (tile_idx == 0)
    *set_len = 0;

  float act_scale = *reinterpret_cast<const float *>(act + kK);

  int8_t *body = w_tile + kPrefix;
  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  int32_t sums[kNTile];
  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());
    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    sums[n] = aie::reduce_add(acc.template to_vector<int32>()) + bias[n];
  }

  float logits[kNTile];
  if (vec_epilogue_exact<80, 174, kNTile>(sums, act_scale, w_scales)) {
    ::aie::vector<float, 16> logit_v =
        vec_scaled_sums<kNTile>(sums, act_scale, w_scales);
    for (int n = 0; n < kNTile; n++)
      logits[n] = logit_v.get(n);
  } else {
    for (int n = 0; n < kNTile; n++)
      logits[n] = (float)sums[n] * act_scale * w_scales[n];
  }

  // Same set order as llama_topk_insert (llama_topk_sample.cc).
  int len = *set_len;
  if (len == kKset) {
    // Almost every call inserts nothing: compare the call's best key with the
    // set's lowest key (one vector pass) before the scan that breaks ties.
    ::aie::vector<int32, kKset> b =
        ::aie::load_unaligned_v<kKset>(set_logit).template cast_to<int32>();
    b = ::aie::select(b, ::aie::zeros<int32, kKset>(),
                      ::aie::eq(b, int32(INT32_MIN)));
    int32_t set_min = ::aie::reduce_min(::aie::bit_xor(
        b, ::aie::bit_and(int32(0x7fffffff), ::aie::downshift(b, 31))));
    int32_t best = logit_key(logits[0]);
    for (int n = 1; n < kNTile; n++) {
      int32_t k = logit_key(logits[n]);
      best = k > best ? k : best;
    }
    if (best < set_min)
      return;
  }
  bool rescan = true;
  int mi = 0;
  int32_t min_key = 0;
  int32_t min_gidx = 0;
  for (int n = 0; n < kNTile; n++) {
    const int32_t gidx = tile_idx * kNTile + n;
    int slot;
    if (len < kKset) {
      slot = len++;
    } else {
      if (rescan) {
        mi = 0;
        min_key = logit_key(set_logit[0]);
        min_gidx = set_gidx[0];
        for (int i = 1; i < kKset; i++) {
          int32_t k = logit_key(set_logit[i]);
          if (k < min_key || (k == min_key && set_gidx[i] > min_gidx)) {
            min_key = k;
            min_gidx = set_gidx[i];
            mi = i;
          }
        }
        rescan = false;
      }
      int32_t key = logit_key(logits[n]);
      if (key < min_key || (key == min_key && gidx > min_gidx))
        continue;
      slot = mi;
      rescan = true;
    }
    const int8_t *__restrict w_row = weights + n * kK;
    set_logit[slot] = logits[n];
    set_gidx[slot] = gidx;
    set_scale[slot] = w_scales[n];
    for (int d = 0; d < kK; d++)
      set_row[slot * kK + d] = w_row[d];
  }
  *set_len = len;
}

#ifndef LLAMA_LMHEAD_KSET
#define LLAMA_LMHEAD_KSET 16
#endif

extern "C" {

// ---------- Legacy SRS-shift entries (kept for 6c.3b.3 design) ----------

void llama_gemm_tiled_K2048_N4_qproj(int8_t *restrict act,
                                     int8_t *restrict w_tile,
                                     int8_t *restrict out_full,
                                     int32_t tile_idx, int32_t right_shift) {
  event0();
  gemm_tile_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

void llama_gemm_tiled_K2048_N4_hproj(int8_t *restrict act,
                                     int8_t *restrict w_tile,
                                     int8_t *restrict out_full,
                                     int32_t tile_idx, int32_t right_shift) {
  event0();
  gemm_tile_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

void llama_gemm_tiled_K64_N4(int8_t *restrict act, int8_t *restrict w_tile,
                             int8_t *restrict out_full, int32_t tile_idx,
                             int32_t right_shift) {
  event0();
  gemm_tile_impl<64, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

void llama_gemm_tiled_K8192_N4(int8_t *restrict act, int8_t *restrict w_tile,
                               int8_t *restrict out_full, int32_t tile_idx,
                               int32_t right_shift) {
  event0();
  gemm_tile_impl<8192, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

// ---------- Phase 6c.5b.4 perchan + dynamic-scale entries ----------

// Gate (K=2048, output HD=8192). Closure-baked scales: its act_scale is
// the residual ACT_SCALE; its inv_out_scale is locked to 1/SILU_GATE_SCALE
// (silu LUT is baked at SILU_GATE_SCALE, so gate must produce int8 at
// that scale for the LUT lookup to be valid).
void llama_gemm_tiled_layer_K2048_N4_perchan_gate(
    int8_t *restrict act, int8_t *restrict w_tile, int8_t *restrict out_full,
    int32_t tile_idx, float act_scale, float inv_out_scale) {
  event0();
  gemm_tile_perchan_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4,
                                  act_scale, inv_out_scale);
  event1();
}

// Gate acttail (K=2048, output HD=8192). act_scale from act tail; inv_out
// stays the silu-LUT lock constant passed as a kernel arg.
void llama_gemm_tiled_layer_K2048_N4_perchan_gate_acttail(
    int8_t *restrict act, int8_t *restrict w_tile, int8_t *restrict out_full,
    int32_t tile_idx, float inv_out_scale) {
  event0();
  gemm_tile_perchan_acttail_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4,
                                          inv_out_scale);
  event1();
}

// up_proj acttail (K=2048, output HD=8192): act_scale from the act tail
// instead of a slot prefix; otherwise identical to up_proj below.
void llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_u_acttail(
    int8_t *restrict act, int8_t *restrict w_tile, int8_t *restrict out_full,
    int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_up_acttail_impl<2048, 4, 8192, 64>(
      act, w_tile, out_full + tile_idx * 4, out_full);
  event1();
}

// q_proj (K=2048, output QD=64). v2_up: mirrors 8 B downstream scales
// (q_out_scale + spare) into out[QD..QD+8] for rope/qk.
void llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_q(int8_t *restrict act,
                                                     int8_t *restrict w_tile,
                                                     int8_t *restrict out_full,
                                                     int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_up_impl<2048, 4, 64, 64>(
      act, w_tile, out_full + tile_idx * 4, out_full);
  event1();
}

// up_proj (K=2048, output HD=8192). v2_up: mirrors 8 B silu scales
// (up_scale + silu_inv_out_scale) into out[HD..HD+8] for silu.
void llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_u(int8_t *restrict act,
                                                     int8_t *restrict w_tile,
                                                     int8_t *restrict out_full,
                                                     int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_up_impl<2048, 4, 8192, 64>(
      act, w_tile, out_full + tile_idx * 4, out_full);
  event1();
}

// o_proj (K=64, output D=2048). Standard v2 (64 B prefix, no tail).
void llama_gemm_tiled_layer_K64_N4_perchan_v2_o(int8_t *restrict act,
                                                int8_t *restrict w_tile,
                                                int8_t *restrict out_full,
                                                int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_impl<64, 4, 64>(act, w_tile, out_full + tile_idx * 4);
  event1();
}

// down_proj (K=8192, output D=2048). Standard v2.
void llama_gemm_tiled_layer_K8192_N4_perchan_v2_d(int8_t *restrict act,
                                                  int8_t *restrict w_tile,
                                                  int8_t *restrict out_full,
                                                  int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_impl<8192, 4, 64>(act, w_tile, out_full + tile_idx * 4);
  event1();
}

// down_proj fp32 output (K=8192, output D=2048). Writes the true fp32 down
// output for the rescale-add to consume.
void llama_gemm_tiled_layer_K8192_N4_perchan_v2_d_fp32out(
    int8_t *restrict act, int8_t *restrict w_tile, float *restrict out_full,
    int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_fp32out_impl<8192, 4, 64>(act, w_tile,
                                                 out_full + tile_idx * 4);
  event1();
}

// down_proj fp32 out, act_scale from the sf activation tail (= the
// self-calibrated silu_out_scale). Used when silu self-calibrates its output.
void llama_gemm_tiled_layer_K8192_N4_perchan_v2_d_fp32out_acttail(
    int8_t *restrict act, int8_t *restrict w_tile, float *restrict out_full,
    int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_fp32out_acttail_impl<8192, 4, 64>(
      act, w_tile, out_full + tile_idx * 4);
  event1();
}

// up_proj fp32 out (Phase 2b self-cal): K=D=2048, output HD=8192. act_scale
// from the h2 activation tail; up_out_scale computed downstream by up_requant.
void llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_fp32out_acttail(
    int8_t *restrict act, int8_t *restrict w_tile, float *restrict out_full,
    int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_fp32out_acttail_impl<2048, 4, 64>(
      act, w_tile, out_full + tile_idx * 4);
  event1();
}

// lm_head fused GEMM + running argmax (greedy), K=D=2048, N_TILE=4 logits
// per call: instead of materializing fp32[V], folds each call's 4 logits
// into a running max in L1 (state = [best_idx i32, best_val_bits f32]),
// self-seeding at tile_idx==0, first-occurrence tie-break to match
// np.argmax.
//
// lm_head fp32-out GEMM (full sampler path): writes N_TILE=4 fp32 logits
// per call to out+tile_idx*4 instead; same impl as up_fp32out_acttail.
void llama_gemm_tiled_layer_K2048_N4_lmhead_fp32out(int8_t *restrict act,
                                                    int8_t *restrict w_tile,
                                                    float *restrict out_chunk,
                                                    int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_fp32out_acttail_impl<2048, 4, 64>(
      act, w_tile, out_chunk + tile_idx * 4);
  event1();
}

void llama_gemm_tiled_layer_K2048_N4_lmhead_argmax(int8_t *restrict act,
                                                   int8_t *restrict w_tile,
                                                   int8_t *restrict state,
                                                   int32_t tile_idx) {
  event0();
  constexpr int kK = 2048, kNTile = 4, kPrefix = 64, kVec = 64;
  constexpr int kGroups = kK / kVec;

  float act_scale;
  memcpy(&act_scale, act + kK, 4);

  int8_t *body = w_tile + kPrefix;
  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  int32_t best_idx;
  float best_val;
  if (tile_idx == 0) {
    best_idx = -1; // sentinel: first comparison initializes the max
    best_val = 0.0f;
  } else {
    memcpy(&best_idx, state, 4);
    memcpy(&best_val, state + 4, 4);
  }

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());
    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];
    float logit = (float)sum_i32 * act_scale * w_scales[n];
    int32_t gidx = tile_idx * kNTile + n;
    if (best_idx < 0 || logit > best_val) {
      best_val = logit;
      best_idx = gidx;
    }
  }

  memcpy(state, &best_idx, 4);
  memcpy(state + 4, &best_val, 4);
  event1();
}

void llama_lmhead_topk_insert(int8_t *restrict act, int8_t *restrict w_tile,
                              float *restrict set_logit,
                              int32_t *restrict set_gidx,
                              float *restrict set_scale,
                              int8_t *restrict set_row,
                              int32_t *restrict set_len, int32_t tile_idx) {
  event0();
  lmhead_topk_insert_impl<LLAMA_LMHEAD_KSET>(
      act, w_tile, set_logit, set_gidx, set_scale, set_row, set_len, tile_idx);
  event1();
}

} // extern "C"
