//===- yolo_m9_qkv_vec.cc -----------------------------------------*- C++
//-*-===//
//
// Vectorized 1x1 INT8 conv 128 -> 256 (no activation; bias-init only).
// Drop-in .o-level replacement for yolo_m9_qkv.cc (same symbol + ABI).
//
// Same pattern as yolo_m9_cv1_split_vec.cc, stripped of the chunked-OC
// + SiLU + top/bot split machinery: per call processes the full OC dim
// (output_channels = 256) into a single (input_width, output_channels)
// row. Weights are the complete OIYXI8O8 tensor (non-streamed; 32KB fits
// L1 alongside in_row + out_row).
//
// Inner reduction: aie::mmul<4, 8, 8, int8, int8>; bias-seeded acc so
// to_vector<int8>(rs) emits bias+SRS+saturate as one vec op. No SiLU
// here — strided i8 store directly from the vec.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifdef YOLO_M9_QKV_IN_W
#define IN_W YOLO_M9_QKV_IN_W
#define IN_C YOLO_M9_QKV_IN_C
#define OUT_C YOLO_M9_QKV_OUT_C
#define SHAPES_ARE_CONST 1
#else
#define IN_W input_width
#define IN_C input_channels
#define OUT_C output_channels
#define SHAPES_ARE_CONST 0
#endif

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// OIYXI8O8 weight base offset for (oc_tile, ic_tile).
static inline int wts_tile_off(int oc_tile, int ic_tile, int ic_tiles) {
  return ((oc_tile * ic_tiles) + ic_tile) << 6;
}

// Reference scalar weight index (tail path).
static inline int wts_idx_oiyxi8o8_1x1(int oc_full, int ic_full, int in_c) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void yolo_m9_qkv_i8_i8(int8_t *in_row, int8_t *wts, int32_t *bias,
                       int8_t *out_row, const int32_t input_width,
                       const int32_t input_channels,
                       const int32_t output_channels,
                       const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

#if SHAPES_ARE_CONST
  (void)input_width;
  (void)input_channels;
  (void)output_channels;
  constexpr int ic_tiles = IN_C / 8;
  constexpr int oc_tiles = OUT_C / 8;
  constexpr int x_tiles = IN_W / 4;
#define AIE_HINT_OC AIE_LOOP_RANGE(oc_tiles, oc_tiles)
#define AIE_HINT_X AIE_LOOP_RANGE(x_tiles, x_tiles)
#define AIE_HINT_IC AIE_LOOP_RANGE(ic_tiles, ic_tiles)
#else
  const int ic_tiles = IN_C / 8;
  const int oc_tiles = OUT_C / 8;
  const int x_tiles = IN_W / 4;
#define AIE_HINT_OC
#define AIE_HINT_X
#define AIE_HINT_IC
#endif

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  AIE_HINT_OC
  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    // Bias seed: reused across all x_tile iters of this oc_t.
    aie::accum<acc32, 32> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias[oc_t * 8]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      bias_acc.from_vector(b32);
    }

    AIE_PREPARE_FOR_PIPELINING
    AIE_HINT_X
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;

      const int x_out_base = x_tile * 4;

      AIE_HINT_IC
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        // uint64 word copies instead of byte-by-byte scalar A-pack.
        alignas(32) int8_t a_buf[32];
        int8_t *s0 = in_row + (x_out_base + 0) * IN_C + ic_t * 8;
        *(reinterpret_cast<uint64_t *>(&a_buf[0])) =
            *reinterpret_cast<const uint64_t *>(s0);
        *(reinterpret_cast<uint64_t *>(&a_buf[8])) =
            *reinterpret_cast<const uint64_t *>(s0 + IN_C);
        *(reinterpret_cast<uint64_t *>(&a_buf[16])) =
            *reinterpret_cast<const uint64_t *>(s0 + 2 * IN_C);
        *(reinterpret_cast<uint64_t *>(&a_buf[24])) =
            *reinterpret_cast<const uint64_t *>(s0 + 3 * IN_C);
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      // uint64 word stores (8 bytes per pixel = 1 op) instead of 8 byte stores.
      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      alignas(32) int8_t out_buf[32];
      aie::store_v(out_buf, srs_v);
      AIE_LOOP_UNROLL_FULL
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        int8_t *dst = out_row + x_out * OUT_C + oc_t * 8;
        *(reinterpret_cast<uint64_t *>(dst)) =
            *reinterpret_cast<const uint64_t *>(&out_buf[p * 8]);
      }
    }

    // Tail scalar fallback for input_width % 4 != 0 (unused for in_w=16;
    // kept for safety with the cv1 pattern).
    for (int x = x_tiles * 4; x < input_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic = 0; ic < input_channels; ++ic) {
          sum += in_row[x * input_channels + ic] *
                 wts[wts_idx_oiyxi8o8_1x1(oc_full, ic, input_channels)];
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX)
          sr = I8_MAX;
        if (sr < I8_MIN)
          sr = I8_MIN;
        out_row[x * output_channels + oc_full] = (int8_t)sr;
      }
    }
  }

  event1();
}

} // extern "C"
