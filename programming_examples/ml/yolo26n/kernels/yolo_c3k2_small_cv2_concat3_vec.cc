//===- yolo_c3k2_small_cv2_concat3_vec.cc -------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Vectorized 1x1 INT8 conv on three concatenated input rows + SiLU LUT.
// Drop-in .o-level replacement for yolo_c3k2_small_cv2_concat3.cc.
//
// ic indices [0, c) come from in_top, [c, 2c) from in_bot, [2c, 3c) from
// in_m0. Weights are packed OIYXI8O8 over the full three_c input axis.
// Per ic_tile (8 ic_inner), all bytes come from a single source buffer
// since c (16 for m2, 32 for m4) is divisible by 8.
//
// Inner reduction: aie::mmul<4, 8, 8, int8, int8>. 4 contiguous output
// pixels x 8 oc_inner per call.
//
// Per-block deep-opt: if YOLO_C3K2_CV2_IN_W etc. are defined at compile
// time, the SHAPES_ARE_CONST path runs a 2X x_pair fold (2 parallel accs
// per x_pair) with loop_range hints, plus a vectorized bias+SRS epilogue
// (LUT lookup stays scalar). Blocks without these defines fall back to
// the single-acc runtime-arg path. 4X fold was tried and corrupted
// output -- short 6-iter ic_t loop + 4 live accs hit register pressure.
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
#ifndef YOLO_C3K2_CV2_IN_W
#error "YOLO_C3K2_CV2_IN_W must be defined at compile time"
#endif
#ifndef YOLO_C3K2_CV2_THREE_C
#error "YOLO_C3K2_CV2_THREE_C must be defined at compile time"
#endif
#ifndef YOLO_C3K2_CV2_OUT_C
#error "YOLO_C3K2_CV2_OUT_C must be defined at compile time"
#endif
#define IN_W YOLO_C3K2_CV2_IN_W
#define THREE_C YOLO_C3K2_CV2_THREE_C
#define OUT_C YOLO_C3K2_CV2_OUT_C

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

// OIYXI8O8 weight base for one (oc_tile, ic_tile) of a 1x1 conv:
//   (((oc_tile * ic_tiles) + ic_tile)) * 64 bytes.
static inline int wts_tile_off_1x1(int oc_tile, int ic_tile, int ic_tiles) {
  return (oc_tile * ic_tiles + ic_tile) << 6;
}

// 8 int32 biases -> 32-wide acc<acc32> seed.
static __attribute__((always_inline)) inline aie::accum<acc32, 32>
make_bias_acc(const int32_t *bias_8) {
  aie::vector<int32, 8> b8 = aie::load_v<8>(bias_8);
  aie::vector<int32, 16> b16 = aie::concat(b8, b8);
  aie::vector<int32, 32> b32 = aie::concat(b16, b16);
  aie::accum<acc32, 32> a;
  a.from_vector(b32);
  return a;
}

// Vectorized epilogue: acc is bias-seeded so to_vector<int8>(rs) emits
// bias-added + SRS'd + saturated i8 in one vec op. Scalar LUT lookup
// (no SIMD int8 LUT gather on AIE2P). conv_even matches banker_srs.
static __attribute__((always_inline)) inline void
write_x_tile_result_vec(aie::mmul<4, 8, 8, int8, int8> &acc, int8_t *silu_lut,
                        int8_t *output, int oc_t, int out_c, int x_out_base,
                        int32_t rs) {
  aie::vector<int8, 32> result = acc.template to_vector<int8>(rs);
  for (int p = 0; p < 4; ++p) {
    int x_out = x_out_base + p;
    for (int j = 0; j < 8; ++j) {
      output[x_out * out_c + oc_t * 8 + j] =
          silu_lut[int(result[p * 8 + j]) + 128];
    }
  }
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_cv2_concat3_silu_bias_i8_i8)(
    int8_t *in_top, int8_t *in_bot, int8_t *in_m0, int8_t *wts, int32_t *bias,
    int8_t *silu_lut, int8_t *output, const int32_t input_width,
    const int32_t three_c, const int32_t output_channels,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  (void)input_width;
  (void)three_c;
  (void)output_channels;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  constexpr int kC = THREE_C / 3;
  constexpr int kIcTiles = THREE_C / 8;
  constexpr int kIcTilesPerSrc = kC / 8;
  constexpr int kOcTiles = OUT_C / 8;
  constexpr int kXTiles = IN_W / 4;
  constexpr int kXPairs = kXTiles / 2;
  constexpr int kXPairTailStart = kXPairs * 2;
  // For m2 (IN_W=128, THREE_C=48,  OUT_C=64):  kXTiles=32, kXPairs=16,
  // kIcTiles=6,  kIcTilesPerSrc=2, kOcTiles=8. For m4 (IN_W=64,  THREE_C=96,
  // OUT_C=128): kXTiles=16, kXPairs=8,  kIcTiles=12, kIcTilesPerSrc=4,
  // kOcTiles=16. Both have kXTiles % 2 == 0 so the tail loop never runs.

  // in_top, in_bot, in_m0 are now in mmul-packed (oc_t, x_block, p*8+chan)
  // layout from cv1_split (top, bot) and m0_cv2_skip (m0). Inner IC loop
  // split per source -- straight-line vec_load + mac loops, no per-iter
  // branch on src_idx.
  constexpr int kPackedIcStride = kXTiles * 32;

  constexpr int kOcPairs = kOcTiles / 2;
  static_assert(kOcTiles % 2 == 0,
                "cv2_concat3 2X*2OC fold requires kOcTiles even");

  for (int oc_pair = 0; oc_pair < kOcPairs; ++oc_pair) {
    const int oc_t_a = oc_pair * 2;
    const int oc_t_b = oc_t_a + 1;
    auto bias_acc_a = make_bias_acc(&bias[oc_t_a * 8]);
    auto bias_acc_b = make_bias_acc(&bias[oc_t_b * 8]);

    // 2X*2OC fold: 4 accs (acc_x{a,b} x oc{0,1}). One A gather per X feeds
    // both OC weights. After mac, emit per-pixel vec_store<16> combining
    // both OC tiles -- vs the original 2X*1OC fold which had to scalar
    // scatter 8-byte chunks per pixel.
    AIE_LOOP_RANGE(kXPairs, kXPairs)
    for (int xp = 0; xp < kXPairs; ++xp) {
      const int x_tile_base = 2 * xp;
      const int x_out_base = x_tile_base * 4;

      MMUL4x8x8 acc_a0, acc_a1, acc_b0, acc_b1;
      acc_a0 = bias_acc_a;
      acc_a1 = bias_acc_b;
      acc_b0 = bias_acc_a;
      acc_b1 = bias_acc_b;

      // in_top
      AIE_LOOP_RANGE(kIcTilesPerSrc, kIcTilesPerSrc)
      for (int local_ic_t = 0; local_ic_t < kIcTilesPerSrc; ++local_ic_t) {
        aie::vector<int8, 32> in_a_a = aie::load_v<32>(
            in_top + local_ic_t * kPackedIcStride + x_tile_base * 32);
        aie::vector<int8, 32> in_a_b = aie::load_v<32>(
            in_top + local_ic_t * kPackedIcStride + (x_tile_base + 1) * 32);
        int wts_off_a = wts_tile_off_1x1(oc_t_a, local_ic_t, kIcTiles);
        int wts_off_b = wts_tile_off_1x1(oc_t_b, local_ic_t, kIcTiles);
        aie::vector<int8, 64> in_b_0 = aie::load_v<64>(&wts[wts_off_a]);
        aie::vector<int8, 64> in_b_1 = aie::load_v<64>(&wts[wts_off_b]);
        acc_a0.mac(in_a_a, in_b_0);
        acc_a1.mac(in_a_a, in_b_1);
        acc_b0.mac(in_a_b, in_b_0);
        acc_b1.mac(in_a_b, in_b_1);
      }
      // in_bot
      AIE_LOOP_RANGE(kIcTilesPerSrc, kIcTilesPerSrc)
      for (int local_ic_t = 0; local_ic_t < kIcTilesPerSrc; ++local_ic_t) {
        aie::vector<int8, 32> in_a_a = aie::load_v<32>(
            in_bot + local_ic_t * kPackedIcStride + x_tile_base * 32);
        aie::vector<int8, 32> in_a_b = aie::load_v<32>(
            in_bot + local_ic_t * kPackedIcStride + (x_tile_base + 1) * 32);
        int ic_t = kIcTilesPerSrc + local_ic_t;
        int wts_off_a = wts_tile_off_1x1(oc_t_a, ic_t, kIcTiles);
        int wts_off_b = wts_tile_off_1x1(oc_t_b, ic_t, kIcTiles);
        aie::vector<int8, 64> in_b_0 = aie::load_v<64>(&wts[wts_off_a]);
        aie::vector<int8, 64> in_b_1 = aie::load_v<64>(&wts[wts_off_b]);
        acc_a0.mac(in_a_a, in_b_0);
        acc_a1.mac(in_a_a, in_b_1);
        acc_b0.mac(in_a_b, in_b_0);
        acc_b1.mac(in_a_b, in_b_1);
      }
      // in_m0
      AIE_LOOP_RANGE(kIcTilesPerSrc, kIcTilesPerSrc)
      for (int local_ic_t = 0; local_ic_t < kIcTilesPerSrc; ++local_ic_t) {
        aie::vector<int8, 32> in_a_a = aie::load_v<32>(
            in_m0 + local_ic_t * kPackedIcStride + x_tile_base * 32);
        aie::vector<int8, 32> in_a_b = aie::load_v<32>(
            in_m0 + local_ic_t * kPackedIcStride + (x_tile_base + 1) * 32);
        int ic_t = 2 * kIcTilesPerSrc + local_ic_t;
        int wts_off_a = wts_tile_off_1x1(oc_t_a, ic_t, kIcTiles);
        int wts_off_b = wts_tile_off_1x1(oc_t_b, ic_t, kIcTiles);
        aie::vector<int8, 64> in_b_0 = aie::load_v<64>(&wts[wts_off_a]);
        aie::vector<int8, 64> in_b_1 = aie::load_v<64>(&wts[wts_off_b]);
        acc_a0.mac(in_a_a, in_b_0);
        acc_a1.mac(in_a_a, in_b_1);
        acc_b0.mac(in_a_b, in_b_0);
        acc_b1.mac(in_a_b, in_b_1);
      }

      // Per-pixel vec_store<16> emit (per-pixel scratch to keep register
      // pressure low -- combining into one big silu_buf disturbed peano
      // scheduling, see commit 21c15253 for the chunked-conv version).
      {
        aie::vector<int8, 32> sa0 =
            acc_a0.template to_vector<int8>(right_shift);
        aie::vector<int8, 32> sa1 =
            acc_a1.template to_vector<int8>(right_shift);
        for (int p = 0; p < 4; ++p) {
          alignas(16) int8_t pix_buf[16];
          for (int j = 0; j < 8; ++j) {
            pix_buf[j] = silu_lut[int(sa0[p * 8 + j]) + 128];
            pix_buf[8 + j] = silu_lut[int(sa1[p * 8 + j]) + 128];
          }
          aie::vector<int8, 16> chunk = aie::load_v<16>(pix_buf);
          aie::store_v(output + (x_out_base + p) * OUT_C + oc_t_a * 8, chunk);
        }
      }
      {
        aie::vector<int8, 32> sb0 =
            acc_b0.template to_vector<int8>(right_shift);
        aie::vector<int8, 32> sb1 =
            acc_b1.template to_vector<int8>(right_shift);
        for (int p = 0; p < 4; ++p) {
          alignas(16) int8_t pix_buf[16];
          for (int j = 0; j < 8; ++j) {
            pix_buf[j] = silu_lut[int(sb0[p * 8 + j]) + 128];
            pix_buf[8 + j] = silu_lut[int(sb1[p * 8 + j]) + 128];
          }
          aie::vector<int8, 16> chunk = aie::load_v<16>(pix_buf);
          aie::store_v(output + (x_out_base + 4 + p) * OUT_C + oc_t_a * 8,
                       chunk);
        }
      }
    }

    // --- x_tile tail ---------------------------------------------------
    // Note: kXPairs * 2 == kXTiles for m2/m4/m6 shapes (IN_W divisible by
    // 8), so the prior x_pair tail loop never executed. Removed along with
    // the gather4 src_for_ic_tile arrays that fed it.
    static_assert(kXPairs * 2 == kXTiles,
                  "cv2_concat3 SHAPES_ARE_CONST path assumes IN_W % 8 == 0");
  }

  event1();
}

} // extern "C"
