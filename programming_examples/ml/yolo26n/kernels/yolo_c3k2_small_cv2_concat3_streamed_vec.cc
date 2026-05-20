//===- yolo_c3k2_small_cv2_concat3_streamed_vec.cc -----------------*- C++ -*-===//
//
// Vectorized chunked-OC 1x1 INT8 conv on three concatenated input rows
// + SiLU LUT. Drop-in for yolo_c3k2_small_cv2_concat3_streamed.cc.
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

static inline int wts_chunk_idx_1x1(int chunk_oc_full, int ic_full, int in_c) {
  int oc_t = chunk_oc_full >> 3;
  int oc_i = chunk_oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_cv2_concat3_streamed_silu_bias_i8_i8)(
    int8_t *in_top,
    int8_t *in_bot,
    int8_t *in_m0,
    int8_t *wts_chunk,
    int32_t *bias_full,
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t c,
    const int32_t output_channels,
    const int32_t n_chunks,
    const int32_t chunk_idx,
    const int32_t right_shift) {
  event0();

  const int32_t chunk_oc = output_channels / n_chunks;
  const int32_t three_c = 3 * c;
  const int32_t oc_offset = chunk_idx * chunk_oc;

  const int ic_tiles = three_c / 8;
  const int ic_tiles_per_src = c / 8;
  const int chunk_oc_tiles = chunk_oc / 8;
  const int x_tiles = input_width / 4;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  int8_t *src_for_ic_tile[48];
  int local_ic_t_for[48];
  for (int ict = 0; ict < ic_tiles; ++ict) {
    int src_idx = ict / ic_tiles_per_src;
    src_for_ic_tile[ict] = (src_idx == 0) ? in_top : (src_idx == 1) ? in_bot : in_m0;
    local_ic_t_for[ict] = ict - src_idx * ic_tiles_per_src;
  }

  for (int chunk_oc_t = 0; chunk_oc_t < chunk_oc_tiles; ++chunk_oc_t) {
    const int oc_full_base = oc_offset + chunk_oc_t * 8;

    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();
      const int x_base = x_tile * 4;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        int8_t *src = src_for_ic_tile[ic_t];
        int local_ic_t = local_ic_t_for[ic_t];

        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_base + p;
          int8_t *psrc = src + col * c + local_ic_t * 8;
          for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = psrc[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off_1x1(chunk_oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_base + p;
        for (int j = 0; j < 8; ++j) {
          int oc_full = oc_full_base + j;
          int32_t s = acc_vec[p * 8 + j] + bias_full[oc_full];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          output[x_out * output_channels + oc_full] = silu_lut[sr + 128];
        }
      }
    }

    // Tail scalar fallback.
    for (int x = x_tiles * 4; x < input_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int chunk_oc_local = chunk_oc_t * 8 + j;
        int oc_full = oc_offset + chunk_oc_local;
        int32_t sum = bias_full[oc_full];
        for (int ic = 0; ic < three_c; ++ic) {
          int8_t a = (ic < c) ? in_top[x * c + ic]
                  : (ic < 2 * c) ? in_bot[x * c + (ic - c)]
                                 : in_m0[x * c + (ic - 2 * c)];
          sum += a * wts_chunk[wts_chunk_idx_1x1(chunk_oc_local, ic, three_c)];
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
