//===- yolo_c3k2_small_cv2_concat3_vec.cc -------------------------*- C++ -*-===//
//
// Vectorized 1x1 INT8 conv on three concatenated input rows + SiLU LUT.
// Drop-in .o-level replacement for yolo_c3k2_small_cv2_concat3.cc.
//
// ic indices [0, c) come from in_top, [c, 2c) from in_bot, [2c, 3c) from
// in_m0. Weights are packed OIYXI8O8 over the full three_c input axis.
// Per ic_tile (8 ic_inner), all bytes come from a single source buffer
// since c (16 for m2, 32 for m4) is divisible by 8.
//
// Inner reduction: aie::mmul<4, 8, 8, int8, int8>. 4 contiguous output
// pixels x 8 oc_inner per call. Scalar epilogue (bias + SRS + clamp + LUT).
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

// OIYXI8O8 weight base for one (oc_tile, ic_tile) of a 1x1 conv:
//   (((oc_tile * ic_tiles) + ic_tile)) * 64 bytes.
static inline int wts_tile_off_1x1(int oc_tile, int ic_tile, int ic_tiles) {
  return (oc_tile * ic_tiles + ic_tile) << 6;
}

// Scalar fallback weight index (tail path).
static inline int wts_idx_oiyxi8o8_1x1(int oc_full, int ic_full, int in_c) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_cv2_concat3_silu_bias_i8_i8)(
    int8_t *in_top,
    int8_t *in_bot,
    int8_t *in_m0,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t three_c,
    const int32_t output_channels,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  const int32_t c = three_c / 3;
  const int ic_tiles = three_c / 8;        // total ic tiles across all 3 sources
  const int ic_tiles_per_src = c / 8;      // ic tiles per individual source
  const int oc_tiles = output_channels / 8;
  const int x_tiles = input_width / 4;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  // Pointer-to-source for each ic_tile (precomputed; same per pixel).
  // Supports up to 48 ic_tiles (three_c up to 384) — covers all yolo26n uses
  // including c3k2_heavy (m6: three_c=192 = 24 ic_tiles).
  int8_t *src_for_ic_tile[48];
  int local_ic_t_for[48];
  for (int ict = 0; ict < ic_tiles; ++ict) {
    int src_idx = ict / ic_tiles_per_src;  // 0, 1, or 2
    src_for_ic_tile[ict] = (src_idx == 0) ? in_top : (src_idx == 1) ? in_bot : in_m0;
    local_ic_t_for[ict] = ict - src_idx * ic_tiles_per_src;
  }

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();

      const int x_out_base = x_tile * 4;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        int8_t *src = src_for_ic_tile[ic_t];
        int local_ic_t = local_ic_t_for[ic_t];

        // Build A: 4 contiguous output pixels x 8 ic_inner from this source's
        // local ic_tile. Pixel p (p in 0..3) at col x_out_base + p in source HWC.
        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_out_base + p;
          int8_t *psrc = src + col * c + local_ic_t * 8;
          for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = psrc[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias[oc_t * 8 + j];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          output[x_out * output_channels + oc_t * 8 + j] = silu_lut[sr + 128];
        }
      }
    }

    // Tail scalar fallback for output_width not a multiple of 4.
    for (int x = x_tiles * 4; x < input_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic = 0; ic < three_c; ++ic) {
          int8_t a;
          if (ic < c) a = in_top[x * c + ic];
          else if (ic < 2 * c) a = in_bot[x * c + (ic - c)];
          else a = in_m0[x * c + (ic - 2 * c)];
          sum += a * wts[wts_idx_oiyxi8o8_1x1(oc_full, ic, three_c)];
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX) sr = I8_MAX;
        if (sr < I8_MIN) sr = I8_MIN;
        output[x * output_channels + oc_full] = silu_lut[sr + 128];
      }
    }
  }

  event1();
}

} // extern "C"
