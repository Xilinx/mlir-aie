//===- yolo_c3k2_heavy_cv3_concat2_vec.cc ---------------------*- C++ -*-===//
//
// Vectorized 1x1 INT8 conv on TWO concatenated input rows + SiLU LUT.
// Drop-in .o-level replacement for yolo_c3k2_heavy_cv3_concat2.cc.
//
// Same pattern as yolo_c3k2_small_cv2_concat3_vec.cc but with 2 inputs
// (in_a, in_b) instead of 3. Per ic_tile, all 8 ic_inner come from one
// source since cp is divisible by 8.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

// Compile-time shape macros — required.
#ifndef YOLO_M6_CV3_IN_W
#error "YOLO_M6_CV3_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M6_CV3_TWO_CP
#error "YOLO_M6_CV3_TWO_CP must be defined at compile time"
#endif
#ifndef YOLO_M6_CV3_OUT_C
#error "YOLO_M6_CV3_OUT_C must be defined at compile time"
#endif
#define IN_W YOLO_M6_CV3_IN_W
#define TWO_CP YOLO_M6_CV3_TWO_CP
#define OUT_C YOLO_M6_CV3_OUT_C

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int wts_tile_off_1x1(int oc_tile, int ic_tile, int ic_tiles) {
  return (oc_tile * ic_tiles + ic_tile) << 6;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_heavy_cv3_concat2_silu_bias_i8_i8)(
    int8_t *in_a, int8_t *in_b, int8_t *wts, int32_t *bias, int8_t *silu_lut,
    int8_t *output, const int32_t input_width, const int32_t two_cp,
    const int32_t output_channels, const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  (void)input_width;
  (void)two_cp;
  (void)output_channels;
  constexpr int32_t cp = TWO_CP / 2;
  constexpr int ic_tiles = TWO_CP / 8;
  constexpr int ic_tiles_per_src = cp / 8;
  constexpr int oc_tiles = OUT_C / 8;
  constexpr int x_tiles = IN_W / 4;
#define AIE_HINT_OC AIE_LOOP_RANGE(oc_tiles, oc_tiles)
#define AIE_HINT_X AIE_LOOP_RANGE(x_tiles, x_tiles)
#define AIE_HINT_IC AIE_LOOP_RANGE(ic_tiles, ic_tiles)

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs; enables vec to_vector<int8>(rs).
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  // in_a (m_0_split's split_b) and in_b (pair_cv2_skip's mmul-packed output)
  // are both mmul-packed (oc_t, x_block, p*8+chan) with 8-pixel 64-byte
  // blocks. cv3 uses mmul<4,8,8>, so each (x_tile, ic_t) loads 32 bytes
  // (4 pixels x 8 chans) — exactly one half of an 8-pixel block.
  constexpr int kXTiles8 = IN_W / 8;
  const int kPackedIcStride = kXTiles8 * 64;

  AIE_HINT_OC
  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    aie::accum<acc32, 32> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias[oc_t * 8]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      bias_acc.from_vector(b32);
    }

    AIE_HINT_X
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;
      const int x_base = x_tile * 4;

      // in_a (src_idx=0, ic_t = 0..ic_tiles_per_src-1)
      for (int local_ic_t = 0; local_ic_t < ic_tiles_per_src; ++local_ic_t) {
        aie::vector<int8, 32> in_a_vec = aie::load_v<32>(
            in_a + local_ic_t * kPackedIcStride + (x_tile >> 1) * 64 +
            (x_tile & 1) * 32);
        int wts_off = wts_tile_off_1x1(oc_t, local_ic_t, ic_tiles);
        aie::vector<int8, 64> in_b_vec = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a_vec, in_b_vec);
      }
      // in_b (src_idx=1, ic_t = ic_tiles_per_src..2*ic_tiles_per_src-1)
      for (int local_ic_t = 0; local_ic_t < ic_tiles_per_src; ++local_ic_t) {
        aie::vector<int8, 32> in_a_vec = aie::load_v<32>(
            in_b + local_ic_t * kPackedIcStride + (x_tile >> 1) * 64 +
            (x_tile & 1) * 32);
        int ic_t = ic_tiles_per_src + local_ic_t;
        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b_vec = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a_vec, in_b_vec);
      }

      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      // mmul-packed output: consumer (c3k2_small cv2_concat3, m6 only)
      // reads via vec_load<32> at 4-pixel-block packed offset.
      alignas(32) int8_t silu_buf[32];
      for (int i = 0; i < 32; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      aie::vector<int8, 32> silu_v = aie::load_v<32>(silu_buf);
      aie::store_v(output + oc_t * (x_tiles * 32) + x_tile * 32, silu_v);
    }

  }

  event1();
}

} // extern "C"
