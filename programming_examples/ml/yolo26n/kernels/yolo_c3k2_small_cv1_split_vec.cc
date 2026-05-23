//===- yolo_c3k2_small_cv1_split_vec.cc ---------------------------*- C++ -*-===//
//
// Vectorized 1x1 INT8 conv with channel-wise output split. Drop-in
// .o-level replacement for yolo_c3k2_small_cv1_split.cc.
//
// Output is split into two halves: oc [0, c) -> out_top, oc [c, 2c) -> out_bot.
// Each half has c channels. Inner reduction: aie::mmul<4, 8, 8, int8, int8>.
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

#ifdef YOLO_C3K2_CV1_IN_W
#define IN_W  YOLO_C3K2_CV1_IN_W
#define IN_C  YOLO_C3K2_CV1_IN_C
#define OUT_C YOLO_C3K2_CV1_OUT_C
#define SHAPES_ARE_CONST 1
#else
#define IN_W  input_width
#define IN_C  input_channels
#define OUT_C output_channels
#define SHAPES_ARE_CONST 0
#endif

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

// Vectorized epilogue: bias add as one int32 vector op, SRS+saturate
// via aie::accum::to_vector<int8>(rs), scalar LUT lookup + strided store.
// Uses kernel-level rounding mode -- caller must set conv_even.
static __attribute__((always_inline)) inline void
write_x_tile_result_vec(aie::mmul<4, 8, 8, int8, int8> &acc,
                        int32_t *bias, int8_t *silu_lut, int8_t *dst,
                        int oc_t, int local_oc_t, int c, int x_out_base, int32_t rs) {
  aie::vector<int32, 8>  bias_v8  = aie::load_v<8>(&bias[oc_t * 8]);
  aie::vector<int32, 16> bias_v16 = aie::concat(bias_v8, bias_v8);
  aie::vector<int32, 32> bias_v32 = aie::concat(bias_v16, bias_v16);
  aie::vector<int32, 32> sum_v = acc.template to_vector<int32>();
  sum_v = aie::add(sum_v, bias_v32);
  aie::accum<acc32, 32> sum_acc;
  sum_acc.from_vector(sum_v);
  aie::vector<int8, 32> srs_v = sum_acc.template to_vector<int8>(rs);

  for (int p = 0; p < 4; ++p) {
    int x_out = x_out_base + p;
    for (int j = 0; j < 8; ++j) {
      dst[x_out * c + local_oc_t * 8 + j] = silu_lut[int(srs_v[p * 8 + j]) + 128];
    }
  }
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_cv1_split_silu_bias_i8_i8)(
    int8_t *in_row,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *out_top,
    int8_t *out_bot,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

#if SHAPES_ARE_CONST
  (void)input_width; (void)input_channels; (void)output_channels;
#endif

  const int32_t c = OUT_C >> 1;
  const int ic_tiles = IN_C / 8;
  const int oc_tiles = OUT_C / 8;
  const int x_tiles = IN_W / 4;
  // Channel midpoint: oc_tile index where out_bot begins (assumes c % 8 == 0).
  const int top_oc_tiles = c / 8;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches the scalar banker_srs used by the runtime tail.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

#if SHAPES_ARE_CONST
#define AIE_HINT_OC AIE_LOOP_RANGE(OUT_C / 8, OUT_C / 8)
#define AIE_HINT_X  AIE_LOOP_RANGE(IN_W  / 4, IN_W  / 4)
#define AIE_HINT_IC AIE_LOOP_RANGE(IN_C  / 8, IN_C  / 8)
#else
#define AIE_HINT_OC
#define AIE_HINT_X
#define AIE_HINT_IC
#endif

  AIE_HINT_OC
  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    // Pick destination buffer + local oc offset within that half.
    int8_t *dst = (oc_t < top_oc_tiles) ? out_top : out_bot;
    int local_oc_t = (oc_t < top_oc_tiles) ? oc_t : (oc_t - top_oc_tiles);

    AIE_HINT_X
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();

      const int x_out_base = x_tile * 4;

      AIE_HINT_IC
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_out_base + p;
          int8_t *src = in_row + col * IN_C + ic_t * 8;
          for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      write_x_tile_result_vec(acc, bias, silu_lut, dst, oc_t, local_oc_t, c, x_out_base, right_shift);
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
        int8_t *dst_tail = (oc_full < c) ? out_top : out_bot;
        int dst_oc = (oc_full < c) ? oc_full : (oc_full - c);
        dst_tail[x * c + dst_oc] = silu_lut[sr + 128];
      }
    }
  }

  event1();
}

} // extern "C"
