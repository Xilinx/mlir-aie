//===- yolo_c3k2_heavy_m_0_split_vec.cc -----------------------*- C++ -*-===//
//
// Vectorized 1x1 INT8 conv with two parallel-branch outputs (same input,
// two independent weight sets / biases / LUTs / right-shifts). Drop-in
// .o-level replacement for yolo_c3k2_heavy_m_0_split.cc.
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

// m_0_split is a 1x1 conv with two parallel branches sharing the same shape.
// Branch a and b take the same input row and use independent weight/bias/lut/
// shift sets but identical (in_w, in_c, out_c).
// Compile-time shape macros — required.
#ifndef YOLO_M6_SPLIT_IN_W
#error "YOLO_M6_SPLIT_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M6_SPLIT_IN_C
#error "YOLO_M6_SPLIT_IN_C must be defined at compile time"
#endif
#ifndef YOLO_M6_SPLIT_OUT_C
#error "YOLO_M6_SPLIT_OUT_C must be defined at compile time"
#endif
#define SPLIT_IN_W YOLO_M6_SPLIT_IN_W
#define SPLIT_IN_C YOLO_M6_SPLIT_IN_C
#define SPLIT_OUT_C YOLO_M6_SPLIT_OUT_C

static inline int wts_tile_off_1x1(int oc_tile, int ic_tile, int ic_tiles) {
  return (oc_tile * ic_tiles + ic_tile) << 6;
}

// One 1x1 branch: input -> SiLU LUT -> output. Bias-seeded mmul so
// to_vector<int8>(rs) directly emits bias+SRS+saturate (requires the
// caller to have set conv_even rounding to match scalar banker_srs).
static void branch_1x1(int8_t *in_row, int8_t *wts, int32_t *bias,
                       int8_t *silu_lut, int8_t *out, int input_width,
                       int input_channels, int output_channels,
                       int right_shift) {
  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;
  (void)input_width;
  (void)input_channels;
  (void)output_channels;
  constexpr int ic_tiles = SPLIT_IN_C / 8;
  constexpr int oc_tiles = SPLIT_OUT_C / 8;
  constexpr int x_tiles = SPLIT_IN_W / 4;
  constexpr int kInC = SPLIT_IN_C;
  constexpr int kOutC = SPLIT_OUT_C;
#define BR_HINT_OC AIE_LOOP_RANGE(oc_tiles, oc_tiles)
#define BR_HINT_X AIE_LOOP_RANGE(x_tiles, x_tiles)
#define BR_HINT_IC AIE_LOOP_RANGE(ic_tiles, ic_tiles)

  BR_HINT_OC
  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    aie::accum<acc32, 32> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias[oc_t * 8]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      bias_acc.from_vector(b32);
    }

    BR_HINT_X
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;
      const int x_base = x_tile * 4;

      // mmul-packed input read: in_row producer (c3k2_small cv1_split,
      // m6 only) now writes (ic_t, x_block, p*8+chan) 4-pixel block
      // layout. Single 32-byte vec_load per (ic_t, x_tile).
      constexpr int kXTiles4_in = SPLIT_IN_W / 4;
      const int kInPackedIcStride = kXTiles4_in * 32;
      BR_HINT_IC
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        aie::vector<int8, 32> in_a = aie::load_v<32>(
            in_row + ic_t * kInPackedIcStride + x_tile * 32);
        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      // mmul-packed output: pair x_tile iters into 8-pixel blocks. Consumer
      // (inner_pair_cv1's input for split_a, cv3_concat2's in_a/in_b for
      // split_b) reads via vec_load (32-byte half-blocks for the 4-pixel
      // mmul<4,8,8> consumer in cv3_concat2; full 64-byte blocks for the
      // 8-pixel mmul<8,8,8> consumer in pair_cv1).
      constexpr int kXTiles8 = SPLIT_IN_W / 8; // 32/8 = 4 for m6
      alignas(32) int8_t silu_buf[32];
      for (int i = 0; i < 32; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      aie::vector<int8, 32> silu_v = aie::load_v<32>(silu_buf);
      aie::store_v(out + oc_t * (kXTiles8 * 64) + (x_tile >> 1) * 64 +
                       (x_tile & 1) * 32,
                   silu_v);
    }

  }
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_heavy_m_0_split_silu_bias_i8_i8)(
    int8_t *in_row, int8_t *wts_a, int32_t *bias_a, int8_t *silu_lut_a,
    int8_t *wts_b, int32_t *bias_b, int8_t *silu_lut_b, int8_t *out_a,
    int8_t *out_b, const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels_a, const int32_t output_channels_b,
    const int32_t right_shift_a, const int32_t right_shift_b) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs; enables vec to_vector<int8>(rs)
  // in branch_1x1 below.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  branch_1x1(in_row, wts_a, bias_a, silu_lut_a, out_a, input_width,
             input_channels, output_channels_a, right_shift_a);
  branch_1x1(in_row, wts_b, bias_b, silu_lut_b, out_b, input_width,
             input_channels, output_channels_b, right_shift_b);

  event1();
}

} // extern "C"
