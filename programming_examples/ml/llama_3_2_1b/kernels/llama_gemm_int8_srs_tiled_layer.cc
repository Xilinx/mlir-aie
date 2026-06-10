//===- llama_gemm_int8_srs_tiled_layer.cc -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Full-single-layer integration variant of the tiled decode GEMV: one
// .o exposing every K-shape a single Llama 3.2 1B decoder layer at
// D=2048, single-head, HD=8192 needs.
//
// Phase 6c.5b.4: per-channel weight quant + per-token dynamic activation
// scales. Mirrors the patterns from llama_gemm_int8_srs_tiled_ffn.cc
// and llama_gemm_int8_srs_tiled_attn.cc:
//   - `gemm_tile_perchan_impl` for closure-baked-scale path (gate's
//     output scale is locked to the silu LUT's gate_scale, so gate keeps
//     this path).
//   - `gemm_tile_perchan_v2_impl` for dynamic scales read from a 64 B
//     slot prefix.
//   - `gemm_tile_perchan_v2_up_impl` for the same plus mirroring an
//     8 B "downstream scales" payload into out_full[kOutDim..kOutDim+8].
//
// 64 B prefix is the smallest multiple of 64 (Bug 7 alignment for
// `aie::load_v<64>` on the weight body). Static asserts enforce it.
//
// Extern entries (Phase 6c.5b.4):
//   K=2048 perchan          (gate; closure scales)
//   K=2048 perchan_v2_up_q  (q_proj; kOutDim=QD=64)
//   K=2048 perchan_v2_up_u  (up_proj; kOutDim=HD=8192)
//   K=64   perchan_v2_o     (o_proj)
//   K=8192 perchan_v2_d     (down)
// Legacy SRS-shift entries kept for the 6c.3b.3 design.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

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

// Per-channel weight scale, closure-baked activation scales.
// Slot layout: [N_TILE*K i8 weights | N_TILE i32 bias | N_TILE fp32 w_scales]
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

  float act_scale;
  memcpy(&act_scale, act + kK, 4);

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

// up_proj acttail (K=2048, output HD=8192). act_scale from act tail; still
// mirrors 8 B silu scales (up_scale + silu_inv_out_scale) into out[HD..HD+8].
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

} // extern "C"
