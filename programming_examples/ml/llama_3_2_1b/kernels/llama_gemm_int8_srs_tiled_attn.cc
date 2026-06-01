//===- llama_gemm_int8_srs_tiled_attn.cc ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Attention-half integration variant of the tiled decode GEMV: one .o
// exposing two K-shapes for the attention path:
//   - K=2048: q_proj (h1[D=2048] -> qf[N=64] for single-head)
//   - K=64:   o_proj (attn[HD=64] -> op[N=D=2048])
//
// Same template logic as `llama_gemm_int8_srs_tiled_ffn.cc` and the
// standalone `llama_gemm_int8_srs_tiled.cc`. They're separate .o files
// (and only one is linked into any given xclbin) because aiecc would
// hit duplicate-symbol errors otherwise. When 6c.3b.3 merges FFN + attn
// into one single-layer xclbin, we'll consolidate into a single kernel
// file with all four (K, N_TILE) entries.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

static constexpr int32_t I8_MAX =  127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

template <int kK, int kNTile>
static inline void gemm_tile_impl(int8_t *restrict act,
                                  int8_t *restrict w_tile,
                                  int8_t *restrict out_tile,
                                  int32_t right_shift) {
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t  *weights = w_tile;
  const int32_t *bias =
      reinterpret_cast<const int32_t *>(w_tile + kNTile * kK);

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

    int32_t sum =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];
    int32_t s = banker_srs(sum, right_shift);
    if (s > I8_MAX) s = I8_MAX;
    if (s < I8_MIN) s = I8_MIN;
    out_tile[n] = (int8_t)s;
  }
}

// ---------------------------------------------------------------------------
// Per-channel + dynamic-scale variants (Phase 6c.5b.3).
// Mirror llama_gemm_int8_srs_tiled_ffn.cc's perchan/v2/v2_up templates.
// Slot layouts:
//   _perchan_v2 (o_proj): [kPrefix B prefix | N_TILE*K weights | N_TILE i32 bias
//                          | N_TILE fp32 w_scales]. First 8 B of prefix are
//                          (act_scale fp32, inv_out_scale fp32); rest is pad.
//   _perchan_v2_up (q_proj): same as v2 but with 8 B of "downstream scales"
//                          right after own scales (16 B total used; rest pad).
//                          Kernel mirrors those 8 B into out_full[QD..QD+8]
//                          each iter so rope can pick them up from qf tail.
// kPrefix must be a multiple of 64 to keep aie::load_v<64> of weights aligned
// (Bug 7).
// ---------------------------------------------------------------------------
template <int kK, int kNTile, int kPrefix = 64>
static inline void gemm_tile_perchan_v2_impl(int8_t *restrict act,
                                             int8_t *restrict w_tile,
                                             int8_t *restrict out_tile) {
  static_assert(kPrefix >= 8, "prefix must hold at least 8 B of scales");
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 to keep aie::load_v<64> "
                "of the weights aligned");
  float act_scale, inv_out_scale;
  memcpy(&act_scale,     w_tile,     4);
  memcpy(&inv_out_scale, w_tile + 4, 4);

  int8_t *body = w_tile + kPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t  *weights = body;
  const int32_t *bias =
      reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float   *w_scales =
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
    if (r > I8_MAX) r = I8_MAX;
    if (r < I8_MIN) r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

template <int kK, int kNTile, int kOutDim, int kPrefix = 64>
static inline void gemm_tile_perchan_v2_up_impl(int8_t *restrict act,
                                                int8_t *restrict w_tile,
                                                int8_t *restrict out_tile,
                                                int8_t *out_full_base) {
  // out_full_base aliases out_tile (out_tile = out_full_base + tile_idx*kNTile),
  // so it's deliberately NOT marked restrict. The tail write at
  // out_full_base[kOutDim..] and per-tile writes at out_tile[0..kNTile] do
  // not overlap, but the aliasing makes restrict UB.
  static_assert(kPrefix >= 16, "v2_up prefix must hold 8 B own + 8 B downstream");
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 to keep aie::load_v<64> "
                "of the weights aligned");
  float act_scale, inv_out_scale;
  memcpy(&act_scale,     w_tile,     4);
  memcpy(&inv_out_scale, w_tile + 4, 4);
  // downstream scales sit at slot[8..16]; copy them through to out tail.
  memcpy(out_full_base + kOutDim, w_tile + 8, 8);

  int8_t *body = w_tile + kPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t  *weights = body;
  const int32_t *bias =
      reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float   *w_scales =
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
    if (r > I8_MAX) r = I8_MAX;
    if (r < I8_MIN) r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

extern "C" {

// q_proj: K = D = 2048, N_TILE = 4.
// Worker passes the FULL output buffer + a tile_idx scalar; kernel
// pointer-offsets to the right slice.
void llama_gemm_tiled_K2048_N4(int8_t *restrict act,
                               int8_t *restrict w_tile,
                               int8_t *restrict out_full,
                               int32_t tile_idx,
                               int32_t right_shift) {
  event0();
  gemm_tile_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

// o_proj: K = HEAD_DIM = 64, N_TILE = 4. K=64 fits in one MAC group so
// the inner loop trivially unrolls.
void llama_gemm_tiled_K64_N4(int8_t *restrict act,
                             int8_t *restrict w_tile,
                             int8_t *restrict out_full,
                             int32_t tile_idx,
                             int32_t right_shift) {
  event0();
  gemm_tile_impl<64, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

// Phase 6c.5b.3 perchan + dynamic-scale entries:
// q_proj: writes downstream scales (8 B from slot[8..16]) into
// out_full[QD..QD+8] so rope/qk can read them from qf's tail. QD=64.
void llama_gemm_tiled_attn_K2048_N4_perchan_v2_up(int8_t *restrict act,
                                                  int8_t *restrict w_tile,
                                                  int8_t *restrict out_full,
                                                  int32_t tile_idx) {
  event0();
  // QD = N_HEADS*HEAD_DIM = 1*64 = 64 for single-head attn-half.
  gemm_tile_perchan_v2_up_impl<2048, 4, 64, 64>(
      act, w_tile, out_full + tile_idx * 4, out_full);
  event1();
}

// o_proj: standard perchan_v2 (no tail-write; output -> add residual).
void llama_gemm_tiled_attn_K64_N4_perchan_v2(int8_t *restrict act,
                                             int8_t *restrict w_tile,
                                             int8_t *restrict out_full,
                                             int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_impl<64, 4, 64>(act, w_tile, out_full + tile_idx * 4);
  event1();
}

} // extern "C"
