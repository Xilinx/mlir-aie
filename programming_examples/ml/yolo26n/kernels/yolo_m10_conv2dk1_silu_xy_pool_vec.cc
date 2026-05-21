//===- yolo_m10_conv2dk1_silu_xy_pool_vec.cc ----------------------*- C++ -*-===//
//
// Vectorized fused 1x1 conv 256→1280 + HardSiLU + spatial xy-pool (GAP).
// Drop-in .o-level replacement for yolo_m10_conv2dk1_silu_xy_pool.cc.
//
// Inner reduction: aie::mmul<4, 8, 8, int8, int8>. 4 spatial positions x
// 8 oc per call. After SRS+LUT, sum across the 4 positions into the row
// total, then accumulate into the persistent `accum` buffer.
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

static inline int wts_tile_off_1x1(int oc_tile, int ic_tile, int ic_tiles) {
  return (oc_tile * ic_tiles + ic_tile) << 6;
}

static inline int wts_chunk_idx_1x1(int chunk_oc_full, int ic_full, int in_c) {
  int oc_t = chunk_oc_full >> 3;
  int oc_i = chunk_oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void yolo_m10_conv2dk1_silu_xy_pool_i8_i8(
    int8_t *in_row,
    int8_t *wts_chunk,
    int32_t *bias_full,
    int8_t *silu_lut,
    int32_t *accum,
    int8_t *elem_out,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t expand_c,
    const int32_t in_h,
    const int32_t right_shift,
    const int32_t yi,
    const int32_t n_splits,
    const int32_t wi) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  const int32_t chunk_oc = expand_c / n_splits;
  const int32_t oc_offset = wi * chunk_oc;

  if (yi == 0 && wi == 0) {
    for (int i = 0; i < expand_c; i++) accum[i] = 0;
  }

  const int ic_tiles = input_channels / 8;
  const int chunk_oc_tiles = chunk_oc / 8;
  const int x_tiles = input_width / 4;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  for (int chunk_oc_t = 0; chunk_oc_t < chunk_oc_tiles; ++chunk_oc_t) {
    int32_t row_sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};

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

        int wts_off = wts_tile_off_1x1(chunk_oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      // SRS + clamp + LUT for 4 spatial positions x 8 oc; accumulate into
      // per-oc row_sums.
      for (int p = 0; p < 4; ++p) {
        for (int j = 0; j < 8; ++j) {
          int chunk_oc_full = chunk_oc_t * 8 + j;
          int32_t s = acc_vec[p * 8 + j] + bias_full[oc_offset + chunk_oc_full];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          row_sums[j] += (int32_t)silu_lut[sr + 128];
        }
      }
    }

    // Tail spatial fallback (in_w not multiple of 4).
    for (int x = x_tiles * 4; x < input_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int chunk_oc_full = chunk_oc_t * 8 + j;
        int32_t sum = bias_full[oc_offset + chunk_oc_full];
        for (int ic = 0; ic < input_channels; ++ic) {
          sum += in_row[x * input_channels + ic] *
                 wts_chunk[wts_chunk_idx_1x1(chunk_oc_full, ic, input_channels)];
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX) sr = I8_MAX;
        if (sr < I8_MIN) sr = I8_MIN;
        row_sums[j] += (int32_t)silu_lut[sr + 128];
      }
    }

    // Commit per-oc row sums into the persistent accumulator.
    for (int j = 0; j < 8; ++j) {
      int chunk_oc_full = chunk_oc_t * 8 + j;
      accum[oc_offset + chunk_oc_full] += row_sums[j];
    }
  }

  // Last call: finalize (same as scalar).
  if (yi == in_h - 1 && wi == n_splits - 1) {
    for (int i = 0; i < expand_c; i++) {
      int32_t pool_q = banker_srs(accum[i], 8);
      if (pool_q > I8_MAX) pool_q = I8_MAX;
      if (pool_q < I8_MIN) pool_q = I8_MIN;
      int32_t f = pool_q << 3;
      if (f > I8_MAX) f = I8_MAX;
      if (f < I8_MIN) f = I8_MIN;
      elem_out[i] = (int8_t)f;
    }
  }

  event1();
}

} // extern "C"
