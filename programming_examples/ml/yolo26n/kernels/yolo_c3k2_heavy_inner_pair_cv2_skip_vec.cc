//===- yolo_c3k2_heavy_inner_pair_cv2_skip_vec.cc -----------------*- C++
//-*-===//
//
// Vectorized 3x3 stride-1 INT8 conv + SiLU LUT + CROSS-SCALE skip-add.
// Drop-in .o-level replacement for yolo_c3k2_heavy_inner_pair_cv2_skip.cc.
//
// Same inner mmul<4, 8, 8> pattern as yolo_c3k2_small_m0_cv2_skip_vec; the
// skip-add epilogue uses a higher-precision cross-scale formula:
//   out_i8 = banker_srs(y_i8 * y_mult + cv2silu_i8 * cv2_mult, rsh_add)
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"
#include "yolo_kernel_common.h"

#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

// Compile-time shape macros — required.
#ifndef YOLO_M6_PAIR_IN_W
#error "YOLO_M6_PAIR_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M6_PAIR_IN_C
#error "YOLO_M6_PAIR_IN_C must be defined at compile time"
#endif
#ifndef YOLO_M6_PAIR_OUT_C
#error "YOLO_M6_PAIR_OUT_C must be defined at compile time"
#endif
#define IN_W YOLO_M6_PAIR_IN_W
#define IN_C YOLO_M6_PAIR_IN_C
#define OUT_C YOLO_M6_PAIR_OUT_C
#define KW 3
#define KH 3

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int wts_tile_off(int oc_tile, int ic_tile, int ky, int kx,
                               int ic_tiles, int kH, int kW) {
  return (((oc_tile * ic_tiles + ic_tile) * kH + ky) * kW + kx) << 6;
}

// load_a_mmul_kx now lives in yolo_kernel_common.h

extern "C" {

void KERNEL_NAME(yolo_c3k2_heavy_inner_pair_cv2_skip_silu_bias_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts, int32_t *bias,
    int8_t *silu_lut, int8_t *skip_row, int8_t *output,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t kernel_width,
    const int32_t kernel_height, const int32_t border,
    const int32_t right_shift, const int32_t skip_y_mult,
    const int32_t skip_cv2_mult, const int32_t skip_rsh_add) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  (void)input_width;
  (void)input_channels;
  (void)output_channels;
  (void)kernel_width;
  (void)kernel_height;

  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);
  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  constexpr int ic_tiles = IN_C / 8;
  constexpr int oc_tiles = OUT_C / 8;
  constexpr int output_width = IN_W;
  constexpr int x_tiles = output_width / 4;
#define AIE_HINT_OC AIE_LOOP_RANGE(oc_tiles, oc_tiles)
#define AIE_HINT_X AIE_LOOP_RANGE(x_tiles, x_tiles)
#define AIE_HINT_IC AIE_LOOP_RANGE(ic_tiles, ic_tiles)
#define AIE_HINT_KX AIE_LOOP_UNROLL_FULL

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs; enables vec to_vector<int8>(rs).
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  // mmul<8,8,8> + hoisted pixel vec gather, mirroring inner_pair_cv1's
  // SHAPES_ARE_CONST upgrade. Skip-add widens to acc32<64> too (which
  // happens to dodge the known peano UNPACKLoad ISel bug on acc32<16,32>).
  using MMUL8x8x8 = aie::mmul<8, 8, 8, int8, int8>;
  constexpr int MMUL_M = 8;
  constexpr int MMUL_MN = 64;
  constexpr int kXTiles8 = IN_W / 8;

  int8_t *line[3] = {line0, line1, line2};

  AIE_HINT_OC
  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    aie::accum<acc32, MMUL_MN> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias[oc_t * 8]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      aie::vector<int32, 64> b64 = aie::concat(b32, b32);
      bias_acc.from_vector(b64);
    }

    AIE_LOOP_RANGE(kXTiles8, kXTiles8)
    for (int x_tile = 0; x_tile < kXTiles8; ++x_tile) {
      MMUL8x8x8 acc;
      acc = bias_acc;

      const int x_out_base = x_tile * MMUL_M;
      const int x_in_base = x_out_base - 1;

      // mmul-packed input read (producer: pair_cv1 mmul-packed output).
      constexpr int kIcStride = kXTiles8 * 64;

      AIE_LOOP_RANGE(1, 3)
      for (int ky = ky_start; ky < ky_end; ++ky) {
        int8_t *line_ptr = line[ky];

        AIE_LOOP_UNROLL_FULL
        for (int kx = 0; kx < KW; ++kx) {
          AIE_HINT_IC
          for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
            aie::vector<int8, 64> in_a =
                load_a_mmul_kx(line_ptr, ic_t, x_tile, kx, kIcStride, kXTiles8);
            int wts_off = wts_tile_off(oc_t, ic_t, ky, kx, ic_tiles, KH, KW);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }

      aie::vector<int8, MMUL_MN> srs_v = acc.template to_vector<int8>(right_shift);
      alignas(64) int8_t silu_buf[64];
      for (int i = 0; i < 64; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      aie::vector<int8, 64> silu_v = aie::load_v<64>(silu_buf);
      // skip_row producer (m_0_split's split_a / pair_cv2's output) now also
      // mmul-packed; one vec_load instead of 64 scalar gathers.
      aie::vector<int8, 64> skip_v = aie::load_v<64>(
          skip_row + oc_t * (kXTiles8 * 64) + x_tile * 64);

      aie::accum<acc32, 64> add_acc = aie::mul(skip_v, (int16)skip_y_mult);
      add_acc = aie::mac(add_acc, silu_v, (int16)skip_cv2_mult);
      aie::vector<int8, 64> out_v =
          add_acc.template to_vector<int8>(skip_rsh_add);

      // mmul-packed output: consumer (next pair_cv1 input, or cv3_concat2
      // in_b) reads via vec_load.
      aie::store_v(output + oc_t * (kXTiles8 * 64) + x_tile * 64, out_v);
    }

  }

  event1();
}

} // extern "C"
