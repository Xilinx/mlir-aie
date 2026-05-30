//===- yolo_m8_back_cv3_cv2_fused_vec.cc ---------------------*- C++ -*-===//
//
// Fused cv3 + cv2 kernel for the m8 megakernel design. Combines:
//   - cv3 (1x1, 2-input concat: inner1 + split_b -> 128 ch)
//   - cv2 (1x1, 3-input concat: top + bot_to_cv2 + cv3_out -> 256 ch, chunked)
//
// Called once per (row, cv2_chunk_idx). On chunk_idx == 0 we recompute cv3
// for this row and stash it in a static scratch array (persists across the
// N_CV2_CHUNKS calls within the same row). On chunk_idx > 0 we reuse the
// stashed cv3_out and just do this chunk of cv2.
//
// Replaces two separate kernel calls (k_cv3 + k_cv2) per cv2-chunk-iter with
// one fused call. Used by m8_megakernel.py to cut Worker-side orchestration
// program-memory footprint.
//
// Numerics bit-exact with yolo_c3k2_heavy_cv3_concat2_vec.cc +
// yolo_c3k2_small_cv2_concat3_streamed_vec.cc when called in sequence.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

static inline int wts_tile_off_1x1(int oc_tile, int ic_tile, int ic_tiles) {
  return (oc_tile * ic_tiles + ic_tile) << 6;
}

// Build a 64-wide bias accumulator (8 int32 bias values replicated to
// 8 pixels x 8 channels) for direct mmul<8,8,8> init.
static __attribute__((always_inline)) inline aie::accum<acc32, 64>
make_bias_acc(const int32_t *bias_8) {
  aie::vector<int32, 8> b8 = aie::load_v<8>(bias_8);
  aie::vector<int32, 16> b16 = aie::concat(b8, b8);
  aie::vector<int32, 32> b32 = aie::concat(b16, b16);
  aie::vector<int32, 64> b64 = aie::concat(b32, b32);
  aie::accum<acc32, 64> a;
  a.from_vector(b64);
  return a;
}

// cv3 inner: 1x1, 128 ic (split into 64+64 from inner1+split_b), 128 oc.
// Writes the full output row into `cv3_out` (input_width * c bytes).
static inline void cv3_compute_row(int8_t *inner1, int8_t *split_b, int8_t *wts,
                                   int32_t *bias, int8_t *silu_lut,
                                   int8_t *cv3_out, int input_width, int two_cp,
                                   int output_channels, int right_shift) {
  // mmul<8,8,8>: 8 pixels per acc (vs 4 with <4,8,8>); same HW.
  using MMUL8x8x8 = aie::mmul<8, 8, 8, int8, int8>;
  constexpr int MMUL_M = 8;

  // Hardcoded for m8 cv3 call site (cp=64, oc=128, W=16). Matches cv2's
  // constexpr trip count fix: lets peano lower divides to immediates.
  (void)two_cp;
  (void)output_channels;
  (void)input_width;
  constexpr int cp = 64;
  constexpr int ic_tiles = 16;        // two_cp / 8
  constexpr int ic_tiles_per_src = 8; // cp / 8
  constexpr int oc_tiles = 16;        // output_channels / 8
  constexpr int x_tiles = 2;          // input_width / MMUL_M

  // split_b is now in mmul-packed (ic_t, x_block, p*8+chan) layout from
  // m_0_split. inner1 still (W,cp) raster (pair_cv2_skip output unchanged
  // in this stage). Inner IC loop split in two so each branch is
  // straight-line — no per-iteration if inside the mac loop.
  constexpr int kPackedIcStride = x_tiles * 64;

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    auto bias_acc = make_bias_acc(&bias[oc_t * 8]);
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL8x8x8 acc;
      acc = bias_acc;
      const int x_base = x_tile * MMUL_M;

      // inner1 (raster, ic_t = 0..ic_tiles_per_src-1)
      for (int local_ic_t = 0; local_ic_t < ic_tiles_per_src; ++local_ic_t) {
        alignas(64) int8_t a_buf[64];
        for (int p = 0; p < MMUL_M; ++p) {
          int col = x_base + p;
          int8_t *psrc = inner1 + col * cp + local_ic_t * 8;
          for (int b = 0; b < 8; ++b)
            a_buf[p * 8 + b] = psrc[b];
        }
        aie::vector<int8, 64> in_a = aie::load_v<64>(a_buf);
        int wts_off = wts_tile_off_1x1(oc_t, local_ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }
      // split_b (packed, ic_t = ic_tiles_per_src..2*ic_tiles_per_src-1)
      for (int local_ic_t = 0; local_ic_t < ic_tiles_per_src; ++local_ic_t) {
        aie::vector<int8, 64> in_a = aie::load_v<64>(
            split_b + local_ic_t * kPackedIcStride + x_tile * 64);
        int ic_t = ic_tiles_per_src + local_ic_t;
        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int8, 64> srs_v = acc.template to_vector<int8>(right_shift);
      // mmul-packed scratch write: cv2_compute_chunk reads in_m0 as packed
      // via the same vec_load path as in_top / in_bot.
      alignas(64) int8_t silu_buf[64];
      for (int i = 0; i < 64; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      aie::vector<int8, 64> silu_v = aie::load_v<64>(silu_buf);
      aie::store_v(cv3_out + oc_t * (x_tiles * 64) + x_tile * 64, silu_v);
    }
  }
}

// cv2 chunk: 1x1, 384 ic (= 3 * 128 from top + bot_to_cv2 + cv3_out), 256 oc
// total, chunked into n_chunks OC slices. Writes chunk_oc oc-channels into
// `output` at offset chunk_idx * chunk_oc.
static inline void cv2_compute_chunk(int8_t *in_top, int8_t *in_bot,
                                     int8_t *in_m0, int8_t *wts_chunk,
                                     int32_t *bias_full, int8_t *silu_lut,
                                     int8_t *output, int input_width, int c,
                                     int output_channels, int n_chunks,
                                     int chunk_idx, int right_shift) {
  using MMUL8x8x8 = aie::mmul<8, 8, 8, int8, int8>;
  constexpr int MMUL_M = 8;

  // Hardcoded for m8 cv2 call site (c=128, out_c=256, W=16, N_CHUNKS=8).
  // Eliminates __udivsi3 from the ict/ic_tiles_per_src setup divide.
  (void)output_channels;
  (void)input_width;
  (void)n_chunks;
  constexpr int chunk_oc = 32;         // out_c / n_chunks = 256 / 8
  constexpr int three_c = 384;         // 3 * c
  constexpr int ic_tiles = 48;         // three_c / 8
  constexpr int ic_tiles_per_src = 16; // c / 8
  constexpr int chunk_oc_tiles = 4;    // chunk_oc / 8
  constexpr int x_tiles = 2;           // input_width / MMUL_M
  const int oc_offset = chunk_idx * chunk_oc;

  // in_top (tiles 0..15) + in_bot (tiles 16..31): mmul-packed
  // (ic_t, x_block, p*8+chan) layout from front's cv1. in_m0 = cv3 scratch
  // (tiles 32..47): (W,c) raster. Inner IC loop is split in two so each
  // branch is straight-line — no per-iteration if inside the mac loop.
  constexpr int kPackedIcStride = x_tiles * 64;
  constexpr int kPackedIcTiles = 2 * ic_tiles_per_src; // top + bot

  for (int chunk_oc_t = 0; chunk_oc_t < chunk_oc_tiles; ++chunk_oc_t) {
    const int oc_full_base = oc_offset + chunk_oc_t * 8;
    auto bias_acc = make_bias_acc(&bias_full[oc_full_base]);

    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL8x8x8 acc;
      acc = bias_acc;
      const int x_base = x_tile * MMUL_M;

      // in_top (packed, ic_t = 0..ic_tiles_per_src-1)
      for (int local_ic_t = 0; local_ic_t < ic_tiles_per_src; ++local_ic_t) {
        aie::vector<int8, 64> in_a = aie::load_v<64>(
            in_top + local_ic_t * kPackedIcStride + x_tile * 64);
        int wts_off = wts_tile_off_1x1(chunk_oc_t, local_ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
        acc.mac(in_a, in_b);
      }
      // in_bot (packed, ic_t = ic_tiles_per_src..2*ic_tiles_per_src-1)
      for (int local_ic_t = 0; local_ic_t < ic_tiles_per_src; ++local_ic_t) {
        aie::vector<int8, 64> in_a = aie::load_v<64>(
            in_bot + local_ic_t * kPackedIcStride + x_tile * 64);
        int ic_t = ic_tiles_per_src + local_ic_t;
        int wts_off = wts_tile_off_1x1(chunk_oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
        acc.mac(in_a, in_b);
      }
      // in_m0 = cv3 scratch (packed, ic_t = 2*ic_tiles_per_src..)
      for (int local_ic_t = 0; local_ic_t < ic_tiles_per_src; ++local_ic_t) {
        aie::vector<int8, 64> in_a = aie::load_v<64>(
            in_m0 + local_ic_t * kPackedIcStride + x_tile * 64);
        int ic_t = kPackedIcTiles + local_ic_t;
        int wts_off = wts_tile_off_1x1(chunk_oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int8, 64> srs_v = acc.template to_vector<int8>(right_shift);
      for (int p = 0; p < MMUL_M; ++p) {
        int x_out = x_base + p;
        for (int j = 0; j < 8; ++j) {
          int oc_full = oc_full_base + j;
          output[x_out * output_channels + oc_full] =
              silu_lut[int(srs_v[p * 8 + j]) + 128];
        }
      }
    }
  }
}

extern "C" {

// `scratch` is a tile-allocated 2 KB Buffer (16 W * 128 c) passed in by the
// IRON design. Persists across the N_CV2_CHUNKS calls within one row (and
// across megakernel sub-ops — it's safe to share with m8_front since the
// two layers use it at different times within an iter).
void KERNEL_NAME(yolo_m8_back_cv3_cv2_fused_i8_i8)(
    // cv3 inputs / weights
    int8_t *inner1, int8_t *split_b, int8_t *wts_cv3, int32_t *bias_cv3,
    int8_t *silu_lut_cv3,
    // cv2 inputs / weights (chunked)
    int8_t *in_top, int8_t *in_bot, int8_t *cv2_wts_chunk, int32_t *bias_cv2,
    int8_t *silu_lut_cv2, int8_t *output,
    int8_t *scratch, // cv3 output buffer (16 * 128 = 2 KB)
    // dims
    const int32_t input_width,
    const int32_t cp,    // cv3 each-source channels (= 64)
    const int32_t c,     // cv1 half-output channels = cv3 output (= 128)
    const int32_t out_c, // cv2 output channels (= 256)
    const int32_t n_cv2_chunks, const int32_t cv2_chunk_idx,
    const int32_t rs_cv3, const int32_t rs_cv2) {
#ifdef NOOP_KERNEL
  return; // Ablation: skip compute, preserve DMA/lock pattern.
#endif
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  // Only recompute cv3 on chunk 0 of each row; subsequent chunks reuse.
  if (cv2_chunk_idx == 0) {
    cv3_compute_row(inner1, split_b, wts_cv3, bias_cv3, silu_lut_cv3, scratch,
                    input_width, 2 * cp, c, rs_cv3);
  }

  cv2_compute_chunk(in_top, in_bot, scratch, cv2_wts_chunk, bias_cv2,
                    silu_lut_cv2, output, input_width, c, out_c, n_cv2_chunks,
                    cv2_chunk_idx, rs_cv2);

  event1();
}

} // extern "C"
