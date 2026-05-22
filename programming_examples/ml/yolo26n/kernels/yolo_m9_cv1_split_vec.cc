//===- yolo_m9_cv1_split_vec.cc -----------------------------------*- C++ -*-===//
//
// Vectorized chunked-OC variant of m9's cv1 1x1 INT8 conv. Drop-in
// .o-level replacement for yolo_m9_cv1_split.cc (same symbol + signature).
//
// Per call processes chunk_oc = twoc/n_chunks output channels (a contiguous
// slice of the OC dim starting at chunk_idx*chunk_oc). All chunk_oc OCs for
// a given call land in EITHER out_top (chunk_idx < n_chunks/2) OR out_bot
// (chunk_idx >= n_chunks/2). Inner reduction: aie::mmul<4, 8, 8, int8, int8>;
// bias + SRS + SiLU LUT in the scalar tail.
//
// Symbol is m9-specific (no KERNEL_SUFFIX mangling).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// Chunk-local OIYXI8O8 weight base offset for (chunk_oc_tile, ic_tile).
// chunk_oc_tile = (chunk-local oc_full) / 8, in [0..chunk_oc/8).
static inline int wts_chunk_tile_off(int chunk_oc_tile, int ic_tile, int ic_tiles) {
  return ((chunk_oc_tile * ic_tiles) + ic_tile) << 6;
}

// Reference scalar weight index (tail path).
static inline int wts_chunk_idx_1x1(int chunk_oc_full, int ic_full, int in_c) {
  int oc_t = chunk_oc_full >> 3;
  int oc_i = chunk_oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void yolo_m9_cv1_split_silu_bias_i8_i8(
    int8_t *in_row,
    int8_t *wts_chunk,
    int32_t *bias_full,
    int8_t *silu_lut,
    int8_t *out_top,
    int8_t *out_bot,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t twoc,
    const int32_t n_chunks,
    const int32_t chunk_idx,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  const int32_t chunk_oc = twoc / n_chunks;
  const int32_t c = twoc >> 1;
  const int32_t chunks_per_half = n_chunks >> 1;
  const bool is_top = (chunk_idx < chunks_per_half);
  const int32_t dst_oc_offset = is_top
      ? chunk_idx * chunk_oc
      : (chunk_idx - chunks_per_half) * chunk_oc;
  const int32_t bias_offset = chunk_idx * chunk_oc;
  int8_t *dst = is_top ? out_top : out_bot;

  const int ic_tiles = input_channels / 8;
  const int chunk_oc_tiles = chunk_oc / 8;
  const int x_tiles = input_width / 4;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  for (int oc_t = 0; oc_t < chunk_oc_tiles; ++oc_t) {
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();

      const int x_out_base = x_tile * 4;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_out_base + p;
          int8_t *src = in_row + col * input_channels + ic_t * 8;
          for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_chunk_tile_off(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias_full[bias_offset + oc_t * 8 + j];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          dst[x_out * c + dst_oc_offset + oc_t * 8 + j] = silu_lut[sr + 128];
        }
      }
    }

    // Tail scalar fallback for x_tiles * 4 < input_width.
    for (int x = x_tiles * 4; x < input_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int chunk_oc_full = oc_t * 8 + j;
        int32_t sum = bias_full[bias_offset + chunk_oc_full];
        for (int ic = 0; ic < input_channels; ++ic) {
          sum += in_row[x * input_channels + ic] *
                 wts_chunk[wts_chunk_idx_1x1(chunk_oc_full, ic, input_channels)];
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX) sr = I8_MAX;
        if (sr < I8_MIN) sr = I8_MIN;
        dst[x * c + dst_oc_offset + chunk_oc_full] = silu_lut[sr + 128];
      }
    }
  }

  event1();
}

} // extern "C"
