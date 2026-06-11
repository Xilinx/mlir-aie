//===- llama_gemm_int8_srs_tiled_ffn.cc -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// FFN-half integration variant of the tiled decode GEMV: one .o that
// exposes BOTH K-shapes the FFN needs (gate/up at K=2048, down at
// K=8192) so a single xclbin can link both. The standalone
// `llama_gemm_int8_srs_tiled.cc` builds with a single `-DLLAMA_GEMM_*`
// pair; if we tried to link two such builds into one xclbin the symbols
// would collide.
//
// Arithmetic is byte-identical to the standalone kernel (and to the
// legacy non-tiled `llama_gemm_int8_srs.cc`), so a single numpy
// reference is bit-exact for all of them.
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

// ---------------------------------------------------------------------------
// Per-channel weight scale variant (Phase 6c.5b.1).
// Slot layout: [N_TILE*K i8 weights | N_TILE i32 bias | N_TILE fp32 w_scales]
// Kernel: int32 accumulate -> + bias (int32) -> per-channel fp32 dequant
//         (act_scale * w_scales[n]) -> re-quantize via host-pre-divided
//         inv_out_scale -> int8 with round-half-away-from-zero + clamp.
// inv_out_scale must be pre-computed on the HOST in IEEE fp32 (numpy or
// equivalent) and passed in -- avoids Peano Bug 1 (HW reciprocal on
// `1.0f/x` is not IEEE-correct).
// ---------------------------------------------------------------------------
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

    // Per-channel dequant (fp32 left-to-right: int->float, * act_scale,
    // * w_scale[n], * inv_out_scale). Numpy ref must use identical order
    // to be bit-equal.
    float fp = (float)sum_i32 * act_scale * w_scales[n];
    float scaled = fp * inv_out_scale;

    // Round-half-away-from-zero + clamp (matches numpy_gemm_perchan ref).
    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (r > I8_MAX)
      r = I8_MAX;
    if (r < I8_MIN)
      r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

// ---------------------------------------------------------------------------
// _v2 variant (Phase 6c.5b.2): scales delivered via slot prefix (first 8 B
// of the weight slot) instead of as kernel scalar args. Slot layout:
//   [8 B: act_scale fp32, inv_out_scale fp32 | N_TILE*K i8 weights
//    | N_TILE i32 bias | N_TILE fp32 w_scales]
// Same arithmetic as gemm_tile_perchan_impl. Kernel takes only 4 args
// (act, w_tile, out, tile_idx) so the same xclbin handles every dispatch
// without rebuild.
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

// _v2_up variant: same as _v2 but slot prefix is 16 B
//   [8 B: gemm act_scale + inv_out_scale | 8 B: silu up_scale + silu
//   inv_out_scale]
// Additionally copies the trailing 8 B (silu's scales) from slot prefix
// into out_full[kHD..kHD+8] every tile iter. Idempotent — the LAST iter
// is the one silu actually reads, but writing every iter avoids needing
// per-iter conditional logic. out_full must be kHD + 8 bytes.
template <int kK, int kNTile, int kHD, int kPrefix = 64>
static inline void
gemm_tile_perchan_v2_up_impl(int8_t *restrict act, int8_t *restrict w_tile,
                             int8_t *restrict out_tile, int8_t *out_full_base) {
  // out_full_base aliases out_tile (out_tile = out_full_base + tile_idx*4),
  // so it's deliberately NOT marked restrict. The tail write at
  // out_full_base[kHD..] and per-tile writes at out_tile[0..kNTile] do
  // not overlap, but the aliasing makes restrict UB.
  static_assert(kPrefix >= 16, "v2_up prefix must hold 8 B own + 8 B silu");
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 to keep aie::load_v<64> "
                "of the weights aligned");
  float act_scale, inv_out_scale;
  memcpy(&act_scale, w_tile, 4);
  memcpy(&inv_out_scale, w_tile + 4, 4);
  // silu scales sit at slot[8..16]; copy them through to out tail.
  memcpy(out_full_base + kHD, w_tile + 8, 8);

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

// gate_proj / up_proj: K = D = 2048, N_TILE = 4.
// out_full points at the worker's full-N output buffer; tile_idx selects
// which N_TILE slice to write. (Worker passes the range_ loop index.)
// Done this way because IRON's Kernel signature can't accept a strided
// memref slice of an acquired buffer; we'd need a contiguous memref of
// the slice's shape, which the slice doesn't produce. Pointer-arith in
// the kernel sidesteps that.
void llama_gemm_tiled_K2048_N4(int8_t *restrict act, int8_t *restrict w_tile,
                               int8_t *restrict out_full, int32_t tile_idx,
                               int32_t right_shift) {
  event0();
  gemm_tile_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

// down_proj: K = HD = 8192, N_TILE = 4.
void llama_gemm_tiled_K8192_N4(int8_t *restrict act, int8_t *restrict w_tile,
                               int8_t *restrict out_full, int32_t tile_idx,
                               int32_t right_shift) {
  event0();
  gemm_tile_impl<8192, 4>(act, w_tile, out_full + tile_idx * 4, right_shift);
  event1();
}

// Per-channel variants (Phase 6c.5b.1): gate/up at K=2048, down at K=8192.
void llama_gemm_tiled_K2048_N4_perchan(int8_t *restrict act,
                                       int8_t *restrict w_tile,
                                       int8_t *restrict out_full,
                                       int32_t tile_idx, float act_scale,
                                       float inv_out_scale) {
  event0();
  gemm_tile_perchan_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4,
                                  act_scale, inv_out_scale);
  event1();
}

void llama_gemm_tiled_K8192_N4_perchan(int8_t *restrict act,
                                       int8_t *restrict w_tile,
                                       int8_t *restrict out_full,
                                       int32_t tile_idx, float act_scale,
                                       float inv_out_scale) {
  event0();
  gemm_tile_perchan_impl<8192, 4>(act, w_tile, out_full + tile_idx * 4,
                                  act_scale, inv_out_scale);
  event1();
}

// _v2 variants (Phase 6c.5b.2): scales from slot prefix.
void llama_gemm_tiled_K2048_N4_perchan_v2(int8_t *restrict act,
                                          int8_t *restrict w_tile,
                                          int8_t *restrict out_full,
                                          int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_impl<2048, 4>(act, w_tile, out_full + tile_idx * 4);
  event1();
}

// Down's slot prefix is padded to 32 B (see WD_PREFIX in aie2_ffn_half.py)
// so the per-slot TAP factors into AIE2P BD dims; scales still live in the
// first 8 B, next 24 B are ignored padding.
void llama_gemm_tiled_K8192_N4_perchan_v2(int8_t *restrict act,
                                          int8_t *restrict w_tile,
                                          int8_t *restrict out_full,
                                          int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_impl<8192, 4, 64>(act, w_tile, out_full + tile_idx * 4);
  event1();
}

// _v2_up: gate/up variant whose slot prefix also carries silu's two scales,
// which the kernel mirrors into out_full[HD..HD+8] for silu to pick up.
void llama_gemm_tiled_K2048_N4_perchan_v2_up(int8_t *restrict act,
                                             int8_t *restrict w_tile,
                                             int8_t *restrict out_full,
                                             int32_t tile_idx) {
  event0();
  // kPrefix=64 — must be a multiple of 64 to keep vector loads aligned.
  // First 8 B = (act_scale, inv_out_scale); next 8 B = silu (up_scale,
  // inv_out_scale); remaining 48 B = pad.
  gemm_tile_perchan_v2_up_impl<2048, 4, 8192, 64>(
      act, w_tile, out_full + tile_idx * 4, out_full);
  event1();
}

} // extern "C"
