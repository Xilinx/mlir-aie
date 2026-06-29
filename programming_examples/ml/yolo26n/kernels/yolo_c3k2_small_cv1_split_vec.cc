//===- yolo_c3k2_small_cv1_split_vec.cc ---------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
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

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

// Compile-time shape macros — required (master-class cleanup removed the
// runtime-arg fallback path).
#ifndef YOLO_C3K2_CV1_IN_W
#error "YOLO_C3K2_CV1_IN_W must be defined at compile time"
#endif
#ifndef YOLO_C3K2_CV1_IN_C
#error "YOLO_C3K2_CV1_IN_C must be defined at compile time"
#endif
#ifndef YOLO_C3K2_CV1_OUT_C
#error "YOLO_C3K2_CV1_OUT_C must be defined at compile time"
#endif
#define IN_W YOLO_C3K2_CV1_IN_W
#define IN_C YOLO_C3K2_CV1_IN_C
#define OUT_C YOLO_C3K2_CV1_OUT_C

static inline int wts_tile_off_1x1(int oc_tile, int ic_tile, int ic_tiles) {
  return (oc_tile * ic_tiles + ic_tile) << 6;
}

// 8 int32 biases -> 32-wide acc<acc32> (4 pix × 8 ch) seed.
static __attribute__((always_inline)) inline aie::accum<acc32, 32>
make_bias_acc(const int32_t *bias_8) {
  aie::vector<int32, 8> b8 = aie::load_v<8>(bias_8);
  aie::vector<int32, 16> b16 = aie::concat(b8, b8);
  aie::vector<int32, 32> b32 = aie::concat(b16, b16);
  aie::accum<acc32, 32> a;
  a.from_vector(b32);
  return a;
}

// Vectorized epilogue: acc is bias-seeded so to_vector<int8>(rs) emits the
// bias-added + SRS'd + saturated i8 in one vec op. mmul-packed output: one
// 32-byte vec_store at (local_oc_t, x_block_of_4, p*8+chan). Consumer
// (c3k2_small m0_cv1, cv2_concat3, or m6's heavy m_0_split / cv3_concat2)
// reads via vec_load<32> instead of 32 scalar lda.s8.
static __attribute__((always_inline)) inline void
write_x_tile_result_vec(aie::mmul<4, 8, 8, int8, int8> &acc, int8_t *silu_lut,
                        int8_t *dst, int local_oc_t, int x_tile, int kXTiles4,
                        int32_t rs) {
  aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(rs);
  alignas(32) int8_t silu_buf[32];
  for (int i = 0; i < 32; ++i)
    silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
  aie::vector<int8, 32> silu_v = aie::load_v<32>(silu_buf);
  aie::store_v(dst + local_oc_t * (kXTiles4 * 32) + x_tile * 32, silu_v);
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_cv1_split_silu_bias_i8_i8)(
    int8_t *in_row, int8_t *wts, int32_t *bias, int8_t *silu_lut,
    int8_t *out_top, int8_t *out_bot, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  (void)input_width;
  (void)input_channels;
  (void)output_channels;

  constexpr int c = OUT_C >> 1;
  constexpr int ic_tiles = IN_C / 8;
  constexpr int oc_tiles = OUT_C / 8;
  constexpr int x_tiles = IN_W / 4;
  constexpr int kXTiles4 = x_tiles;
  constexpr int top_oc_tiles = c / 8;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches banker_srs (round-half-to-even).
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

#define AIE_HINT_OC AIE_LOOP_RANGE(OUT_C / 8, OUT_C / 8)
#define AIE_HINT_X AIE_LOOP_RANGE(IN_W / 4, IN_W / 4)
#define AIE_HINT_IC AIE_LOOP_RANGE(IN_C / 8, IN_C / 8)

  AIE_HINT_OC
  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    // Pick destination buffer + local oc offset within that half.
    int8_t *dst = (oc_t < top_oc_tiles) ? out_top : out_bot;
    int local_oc_t = (oc_t < top_oc_tiles) ? oc_t : (oc_t - top_oc_tiles);

    auto bias_acc = make_bias_acc(&bias[oc_t * 8]);

    AIE_HINT_X
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;

      const int x_out_base = x_tile * 4;

      AIE_HINT_IC
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        // 4× uint64 word copies replace byte-by-byte gather (peano
        // doesn't reliably lower the scalar loop — see
        // feedback_explicit_uint64_over_byte_loop).
        alignas(32) int8_t a_buf[32];
        int8_t *s0 = in_row + (x_out_base + 0) * IN_C + ic_t * 8;
        *reinterpret_cast<uint64_t *>(&a_buf[0]) =
            *reinterpret_cast<const uint64_t *>(s0);
        *reinterpret_cast<uint64_t *>(&a_buf[8]) =
            *reinterpret_cast<const uint64_t *>(s0 + IN_C);
        *reinterpret_cast<uint64_t *>(&a_buf[16]) =
            *reinterpret_cast<const uint64_t *>(s0 + 2 * IN_C);
        *reinterpret_cast<uint64_t *>(&a_buf[24]) =
            *reinterpret_cast<const uint64_t *>(s0 + 3 * IN_C);
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      write_x_tile_result_vec(acc, silu_lut, dst, local_oc_t, x_tile, kXTiles4,
                              right_shift);
    }
  }

  event1();
}

} // extern "C"
