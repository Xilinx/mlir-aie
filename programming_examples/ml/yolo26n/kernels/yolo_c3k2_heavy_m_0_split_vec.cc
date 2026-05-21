//===- yolo_c3k2_heavy_m_0_split_vec.cc -----------------------*- C++ -*-===//
//
// Vectorized 1x1 INT8 conv with two parallel-branch outputs (same input,
// two independent weight sets / biases / LUTs / right-shifts). Drop-in
// .o-level replacement for yolo_c3k2_heavy_m_0_split.cc.
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

static inline int wts_idx_oiyxi8o8_1x1(int oc_full, int ic_full, int in_c) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

// One 1x1 branch: input -> SiLU LUT -> output.
static void branch_1x1(
    int8_t *in_row, int8_t *wts, int32_t *bias, int8_t *silu_lut,
    int8_t *out, int input_width, int input_channels, int output_channels,
    int right_shift) {
  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;
  const int ic_tiles = input_channels / 8;
  const int oc_tiles = output_channels / 8;
  const int x_tiles = input_width / 4;

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();
      const int x_base = x_tile * 4;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_base + p;
          int8_t *src = in_row + col * input_channels + ic_t * 8;
          for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_base + p;
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias[oc_t * 8 + j];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          out[x_out * output_channels + oc_t * 8 + j] = silu_lut[sr + 128];
        }
      }
    }

    // Tail scalar fallback.
    for (int x = x_tiles * 4; x < input_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic = 0; ic < input_channels; ++ic) {
          sum += in_row[x * input_channels + ic] *
                 wts[wts_idx_oiyxi8o8_1x1(oc_full, ic, input_channels)];
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX) sr = I8_MAX;
        if (sr < I8_MIN) sr = I8_MIN;
        out[x * output_channels + oc_full] = silu_lut[sr + 128];
      }
    }
  }
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_heavy_m_0_split_silu_bias_i8_i8)(
    int8_t *in_row,
    int8_t *wts_a, int32_t *bias_a, int8_t *silu_lut_a,
    int8_t *wts_b, int32_t *bias_b, int8_t *silu_lut_b,
    int8_t *out_a, int8_t *out_b,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels_a,
    const int32_t output_channels_b,
    const int32_t right_shift_a,
    const int32_t right_shift_b) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  branch_1x1(in_row, wts_a, bias_a, silu_lut_a, out_a,
             input_width, input_channels, output_channels_a, right_shift_a);
  branch_1x1(in_row, wts_b, bias_b, silu_lut_b, out_b,
             input_width, input_channels, output_channels_b, right_shift_b);

  event1();
}

} // extern "C"
