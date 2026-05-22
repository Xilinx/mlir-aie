//===- yolo_c3k2_small_m0_cv1_vec.cc -------------------------------*- C++ -*-===//
//
// Vectorized 3x3 stride-1 INT8 conv with OIYXI8O8 weight layout. Drop-in
// .o-level replacement for yolo_c3k2_small_m0_cv1.cc on AIE2P.
//
// Same math as Phase 1's stride-2 vec kernel but stride-1: 4 contiguous
// output pixels per mmul<4,8,8> call, input pixel cols = x_out + kx - 1
// (vs 2*x_out + kx - 1 for stride-2).
//
// Per-block deep-opt: if YOLO_C3K2_M0CV1_IN_W etc. are defined at compile
// time (passed via -D from the Makefile for blocks that have shape-stable
// kernels), shape constants fold into shifts/immediates and inner loop
// trip counts become compile-time. Blocks without these defines fall
// back to the runtime-arg path (slower but works for any shape).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// Per-block symbol mangling.
#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

// Compile-time shape macros. If -DYOLO_C3K2_M0CV1_IN_W=… etc. are passed
// at build time, IN_W/IN_C/OUT_C/KW/KH become integer literals and peano
// folds the addressing math + unrolls the inner kx/ic loops. Otherwise
// they fall back to the runtime args (legacy path, used by blocks that
// don't pre-declare shapes).
#ifdef YOLO_C3K2_M0CV1_IN_W
#define IN_W YOLO_C3K2_M0CV1_IN_W
#define IN_C YOLO_C3K2_M0CV1_IN_C
#define OUT_C YOLO_C3K2_M0CV1_OUT_C
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

// Scalar fallback weight index (tail path).
static inline int wts_idx_oiyxi8o8(int oc_full, int ic_full, int ky, int kx,
                                   int in_c, int kH, int kW) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) +
         ic_i * 8 + oc_i;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_m0_cv1_conv2dk3_silu_bias_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t /*padding*/) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

#if SHAPES_ARE_CONST
  // Runtime args must match compile-time shape macros.
  // Cheap assert; can be compiled out with -DNDEBUG (already on by default).
  (void)input_width; (void)input_channels; (void)output_channels;
  (void)kernel_width; (void)kernel_height;
#endif

  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);
  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  const int ic_tiles = IN_C / 8;
  const int oc_tiles = OUT_C / 8;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  int8_t *line[3] = {line0, line1, line2};

  // Output width == input width for stride-1.
  const int output_width = IN_W;
  const int x_tiles = output_width / 4;

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();

      const int x_out_base = x_tile * 4;
      // Leftmost input col for output 0 of this tile, kx=0: x_out_base - 1.
      const int x_in_base = x_out_base - 1;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];

          for (int kx = 0; kx < KW; ++kx) {
            // 4 contiguous output pixels: input cols = x_in_base + p + kx
            // (p in 0..3). Border (col < 0 or >= IN_W) -> zero-fill.
            alignas(32) int8_t a_buf[32];
            bool any_valid = false;
            for (int p = 0; p < 4; ++p) {
              int col = x_in_base + p + kx;
              if (col < 0 || col >= IN_W) {
                for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = 0;
              } else {
                int8_t *src = line_ptr + col * IN_C + ic_t * 8;
                for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
                any_valid = true;
              }
            }
            if (!any_valid) continue;
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

            int wts_off = wts_tile_off(oc_t, ic_t, ky, kx, ic_tiles, KH, KW);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias[oc_t * 8 + j];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          output[x_out * OUT_C + oc_t * 8 + j] = silu_lut[sr + 128];
        }
      }
    }

    // Tail outputs if output_width not a multiple of 4: scalar fallback.
    for (int x = x_tiles * 4; x < output_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic_full = 0; ic_full < IN_C; ++ic_full) {
          for (int kx = 0; kx < KW; ++kx) {
            int col = x - 1 + kx;
            if (col < 0 || col >= IN_W) continue;
            int in_indx = col * IN_C + ic_full;
            int w0 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 0, kx, IN_C, KH, KW)];
            int w1 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 1, kx, IN_C, KH, KW)];
            int w2 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 2, kx, IN_C, KH, KW)];
            if (!skip_top) sum += line0[in_indx] * w0;
            sum += line1[in_indx] * w1;
            if (!skip_bot) sum += line2[in_indx] * w2;
          }
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX) sr = I8_MAX;
        if (sr < I8_MIN) sr = I8_MIN;
        output[x * OUT_C + oc_full] = silu_lut[sr + 128];
      }
    }
  }

  event1();
}

} // extern "C"
