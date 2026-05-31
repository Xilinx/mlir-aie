//===- yolo_c3k2_small_m0_cv2_skip_vec.cc --------------------------*- C++
//-*-===//
//
// Vectorized 3x3 stride-1 INT8 conv + SiLU LUT + int8-saturating skip-add.
// Drop-in .o-level replacement for yolo_c3k2_small_m0_cv2_skip.cc.
//
// Identical to yolo_c3k2_small_m0_cv1_vec.cc (3x3 stride-1, mmul<4,8,8>),
// plus an int8 skip-add step in the epilogue.
//
// Per-block deep-opt: if YOLO_C3K2_M0CV2_SKIP_IN_W etc. are defined at
// compile time, shape constants fold into shifts/immediates. Otherwise
// fall back to the runtime-arg path. Same pattern as m0_cv1_vec.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

#ifdef YOLO_C3K2_M0CV2_SKIP_IN_W
#define IN_W YOLO_C3K2_M0CV2_SKIP_IN_W
#define IN_C YOLO_C3K2_M0CV2_SKIP_IN_C
#define OUT_C YOLO_C3K2_M0CV2_SKIP_OUT_C
#define KW 3
#define KH 3
#define SHAPES_ARE_CONST 1
#else
#define IN_W input_width
#define IN_C input_channels
#define OUT_C output_channels
#define KW kernel_width
#define KH kernel_height
#define SHAPES_ARE_CONST 0
#endif

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

static inline int wts_tile_off(int oc_tile, int ic_tile, int ky, int kx,
                               int ic_tiles, int kH, int kW) {
  return (((oc_tile * ic_tiles + ic_tile) * kH + ky) * kW + kx) << 6;
}

static inline int wts_idx_oiyxi8o8(int oc_full, int ic_full, int ky, int kx,
                                   int in_c, int kH, int kW) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) + ic_i * 8 +
         oc_i;
}

#if SHAPES_ARE_CONST
// 4-pixel mmul A load with kx slide from mmul-packed line buffer. Mirror
// of m0_cv1's helper.
template <int IN_C_LOCAL, int IN_W_LOCAL>
static __attribute__((always_inline)) inline aie::vector<int8, 32>
load_a_mmul_kx_4p(int8_t *line_ptr, int ic_t, int x_tile, int kx) {
  constexpr int kXTiles4 = IN_W_LOCAL / 4;
  constexpr int kIcStride = kXTiles4 * 32;
  int8_t *base = line_ptr + ic_t * kIcStride;
  if (kx == 1) {
    return aie::load_v<32>(base + x_tile * 32);
  }
  int blk_lo = (kx == 0) ? x_tile - 1 : x_tile;
  int blk_hi = blk_lo + 1;
  aie::vector<int8, 32> lo = (blk_lo >= 0 && blk_lo < kXTiles4)
                                 ? aie::load_v<32>(base + blk_lo * 32)
                                 : aie::zeros<int8, 32>();
  aie::vector<int8, 32> hi = (blk_hi >= 0 && blk_hi < kXTiles4)
                                 ? aie::load_v<32>(base + blk_hi * 32)
                                 : aie::zeros<int8, 32>();
  aie::vector<int8, 64> combined = aie::concat(lo, hi);
  const unsigned shift = (kx == 0) ? 24u : 8u;
  return aie::shuffle_down(combined, shift).template extract<32>(0);
}
#endif

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_m0_cv2_skip_silu_bias_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts, int32_t *bias,
    int8_t *silu_lut, int8_t *skip_row, int8_t *output,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t kernel_width,
    const int32_t kernel_height, const int32_t border,
    const int32_t right_shift, const int32_t /*skip_scale*/) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

#if SHAPES_ARE_CONST
  (void)input_width;
  (void)input_channels;
  (void)output_channels;
  (void)kernel_width;
  (void)kernel_height;
#endif

  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);
  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  const int ic_tiles = IN_C / 8;
  const int oc_tiles = OUT_C / 8;
  const int output_width = IN_W; // stride-1
  const int x_tiles = output_width / 4;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches the scalar banker_srs (round-to-nearest, ties to even)
  // used by both the SHAPES_ARE_CONST vec path below and the runtime scalar
  // tail; runtime path's banker_srs is unaffected by this setting.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  int8_t *line[3] = {line0, line1, line2};

#if SHAPES_ARE_CONST
#define AIE_HINT_OC AIE_LOOP_RANGE(OUT_C / 8, OUT_C / 8)
#define AIE_HINT_X AIE_LOOP_RANGE(IN_W / 4, IN_W / 4)
#define AIE_HINT_IC AIE_LOOP_RANGE(IN_C / 8, IN_C / 8)
#else
#define AIE_HINT_OC
#define AIE_HINT_X
#define AIE_HINT_IC
#endif

  AIE_HINT_OC
  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    // Bias seed: 8 int32 biases -> 32-wide acc<acc32>, reused across all
    // x_tile iters of this oc_t. Lets to_vector<int8>(rs) below emit
    // bias-added + SRS'd i8 in one vec op (no post-mac vec add).
    aie::accum<acc32, 32> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias[oc_t * 8]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      bias_acc.from_vector(b32);
    }

    constexpr int kXTiles4_in = IN_W / 4;
    constexpr int kXTiles4_out = IN_W / 4; // output_width == IN_W (stride-1)
    constexpr int kInOcStride = kXTiles4_in * 32;
    constexpr int kOutOcStride = kXTiles4_out * 32;

    AIE_HINT_X
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;

      // mmul-packed input read (producer: m0_cv1's packed output).
      AIE_HINT_IC
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        AIE_LOOP_RANGE(1, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];

          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < KW; ++kx) {
            aie::vector<int8, 32> in_a =
                load_a_mmul_kx_4p<IN_C, IN_W>(line_ptr, ic_t, x_tile, kx);
            int wts_off = wts_tile_off(oc_t, ic_t, ky, kx, ic_tiles, KH, KW);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }

      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);

      // Scalar LUT into silu_buf; skip_v is now one vec_load<32> from the
      // packed skip producer (cv1_split's bot, m6 c3k2_small never uses
      // this kernel). Use int16 vec add (range [-256, 254] fits cleanly).
      alignas(32) int8_t silu_buf[32];
      for (int i = 0; i < 32; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      aie::vector<int8, 32> silu_v8 = aie::load_v<32>(silu_buf);
      aie::vector<int8, 32> skip_v8 = aie::load_v<32>(
          skip_row + oc_t * kOutOcStride + x_tile * 32);
      aie::vector<int16, 32> silu_v16 = aie::unpack(silu_v8);
      aie::vector<int16, 32> skip_v16 = aie::unpack(skip_v8);
      aie::vector<int16, 32> sum16 = aie::add(silu_v16, skip_v16);
      aie::accum<acc32, 32> sum16_acc;
      sum16_acc.from_vector(sum16);
      aie::vector<int8, 32> out_v = sum16_acc.template to_vector<int8>(0);

      aie::store_v(output + oc_t * kOutOcStride + x_tile * 32, out_v);
    }

    // Tail scalar fallback.
    for (int x = x_tiles * 4; x < output_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic_full = 0; ic_full < IN_C; ++ic_full) {
          for (int kx = 0; kx < KW; ++kx) {
            int col = x - 1 + kx;
            if (col < 0 || col >= IN_W)
              continue;
            int in_indx = col * IN_C + ic_full;
            int w0 =
                wts[wts_idx_oiyxi8o8(oc_full, ic_full, 0, kx, IN_C, KH, KW)];
            int w1 =
                wts[wts_idx_oiyxi8o8(oc_full, ic_full, 1, kx, IN_C, KH, KW)];
            int w2 =
                wts[wts_idx_oiyxi8o8(oc_full, ic_full, 2, kx, IN_C, KH, KW)];
            if (!skip_top)
              sum += line0[in_indx] * w0;
            sum += line1[in_indx] * w1;
            if (!skip_bot)
              sum += line2[in_indx] * w2;
          }
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX)
          sr = I8_MAX;
        if (sr < I8_MIN)
          sr = I8_MIN;
        int32_t silu = silu_lut[sr + 128];
        int32_t added = silu + (int32_t)skip_row[x * OUT_C + oc_full];
        if (added > I8_MAX)
          added = I8_MAX;
        if (added < I8_MIN)
          added = I8_MIN;
        output[x * OUT_C + oc_full] = (int8_t)added;
      }
    }
  }

  event1();
}

} // extern "C"
