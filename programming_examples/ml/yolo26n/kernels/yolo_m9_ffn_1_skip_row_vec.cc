//===- yolo_m9_ffn_1_skip_row_vec.cc -----------------------------*- C++ -*-===//
//
// Deep-opt vectorized 1x1 i8 conv 256 -> 128 + plain (same-scale)
// skip-add with attn_block_out. Drop-in .o-level replacement.
//
// Same toolbox as yolo_m9_cv1_split_vec.cc: pre-pack input into a
// YCXC8 scratch once per call, compile-time shape #defines + AIE_LOOP_RANGE
// for tight inner loops, multi-acc fold across x_tiles to share the
// B-load and break the inner ic_t reduction dep chain.
//
// Skip-add is at the same scale (both ffn.1 out and attn_block_out at
// 2^-4 per ONNX) so the epilogue is just plain int add + clip.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef YOLO_M9_FFN1_IN_W
#error "YOLO_M9_FFN1_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M9_FFN1_IN_C
#error "YOLO_M9_FFN1_IN_C must be defined at compile time"
#endif
#ifndef YOLO_M9_FFN1_OUT_C
#error "YOLO_M9_FFN1_OUT_C must be defined at compile time"
#endif

static constexpr int kInW    = YOLO_M9_FFN1_IN_W;
static constexpr int kInC    = YOLO_M9_FFN1_IN_C;
static constexpr int kOutC   = YOLO_M9_FFN1_OUT_C;
static constexpr int kIcTiles  = kInC / 8;
static constexpr int kOcTiles  = kOutC / 8;
static constexpr int kXTiles   = kInW / 4;

static_assert(kInC % 8 == 0,  "FFN1 IN_C must be multiple of 8");
static_assert(kOutC % 8 == 0, "FFN1 OUT_C must be multiple of 8");
static_assert(kInW % 4 == 0,  "FFN1 IN_W must be multiple of 4");

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void yolo_m9_ffn_1_skip_row_i8_i8(
    int8_t *mid_row,
    int8_t *wts,
    int32_t *bias,
    int8_t *skip_row,
    int8_t *out_row,
    const int32_t /*input_width*/,
    const int32_t /*input_channels*/,
    const int32_t /*output_channels*/,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  alignas(32) int8_t scratch[kIcTiles * kInW * 8];
  AIE_LOOP_RANGE(kInW, kInW)
  for (int x = 0; x < kInW; ++x) {
    AIE_LOOP_RANGE(kIcTiles, kIcTiles)
    for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
      const int8_t *__restrict src = mid_row + x * kInC + ic_t * 8;
      int8_t *__restrict d = scratch + ic_t * kInW * 8 + x * 8;
      for (int b = 0; b < 8; ++b) d[b] = src[b];
    }
  }

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  // NUM_ACC=1 single-acc body (vs cv1's kXTiles fold). The ffn worker
  // shares its tile with ffn.0, so the multi-acc full unroll for both
  // kernels overflows the 16 KB per-tile .text cap (ffn.1 multi-acc
  // alone = 14.6 KB). Pre-pack + shape #defines + AIE_LOOP_RANGE still
  // apply.
  AIE_LOOP_RANGE(kOcTiles, kOcTiles)
  for (int oc_t = 0; oc_t < kOcTiles; ++oc_t) {
    AIE_LOOP_RANGE(kXTiles, kXTiles)
    for (int x_tile = 0; x_tile < kXTiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();

      const int8_t *__restrict a_ptr = scratch + x_tile * 4 * 8;
      const int8_t *__restrict b_ptr = wts + ((oc_t * kIcTiles) << 6);

      AIE_LOOP_RANGE(kIcTiles, kIcTiles)
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_ptr);
        a_ptr += kInW * 8;
        aie::vector<int8, 64> in_b = aie::load_v<64>(b_ptr);
        b_ptr += 64;
        acc.mac(in_a, in_b);
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      const int32_t *__restrict bias_p = bias + oc_t * 8;
      const int x_out_base = x_tile * 4;
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        int8_t *__restrict row_dst = out_row + x_out * kOutC + oc_t * 8;
        const int8_t *__restrict skip_p = skip_row + x_out * kOutC + oc_t * 8;
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias_p[j];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          int32_t add = sr + (int32_t)skip_p[j];
          if (add > I8_MAX) add = I8_MAX;
          if (add < I8_MIN) add = I8_MIN;
          row_dst[j] = (int8_t)add;
        }
      }
    }
  }

  event1();
}

} // extern "C"
