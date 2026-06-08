//===- yolo_c3k2_heavy_inner_pair_cv1_streamed_vec.cc ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Vectorized chunked-OC 3x3 stride-1 conv + SiLU LUT. Drop-in .o-level
// replacement for yolo_c3k2_heavy_inner_pair_cv1_streamed.cc.
//
// Same inner mmul<4,8,8> pattern as the non-streamed inner_pair_cv1 vec,
// adapted for the chunked-OC API: kernel takes n_chunks + chunk_idx,
// weight buffer holds only this chunk's OC slice, output writes to the
// full-row offset at (oc_offset + chunk_local_oc).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"
#include "yolo_kernel_common.h"

// Per-block deep-opt: if YOLO_M8_PAIR_IN_W etc. are defined at compile
// time, shape constants fold into shifts + immediates, peano gets exact
// loop trip counts via AIE_LOOP_RANGE hints, and bias is folded into the
// mmul accumulator init (no separate vec epilogue needed).
// Compile-time shape macros — required.
#ifndef YOLO_M8_PAIR_IN_W
#error "YOLO_M8_PAIR_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M8_PAIR_IN_C
#error "YOLO_M8_PAIR_IN_C must be defined at compile time"
#endif
#ifndef YOLO_M8_PAIR_OUT_C
#error "YOLO_M8_PAIR_OUT_C must be defined at compile time"
#endif
#ifndef YOLO_M8_PAIR_N_CHUNKS
#error "YOLO_M8_PAIR_N_CHUNKS must be defined at compile time"
#endif
#define IN_W YOLO_M8_PAIR_IN_W
#define IN_C YOLO_M8_PAIR_IN_C
#define OUT_C YOLO_M8_PAIR_OUT_C
#define N_CHUNKS YOLO_M8_PAIR_N_CHUNKS
#define KW 3
#define KH 3

#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int wts_chunk_tile_off(int chunk_oc_tile, int ic_tile, int ky,
                                     int kx, int ic_tiles, int kH, int kW) {
  return (((chunk_oc_tile * ic_tiles + ic_tile) * kH + ky) * kW + kx) << 6;
}

// load_a_mmul_kx now lives in yolo_kernel_common.h

extern "C" {

void KERNEL_NAME(
    yolo_c3k2_heavy_inner_pair_cv1_streamed_conv2dk3_silu_bias_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts_chunk,
    int32_t *bias_full, int8_t *silu_lut, int8_t *output,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t kernel_width,
    const int32_t kernel_height, const int32_t border,
    const int32_t right_shift, const int32_t n_chunks,
    const int32_t chunk_idx) {
#ifdef NOOP_KERNEL
  return; // Ablation: skip compute, preserve DMA/lock pattern.
#endif
  event0();

  (void)input_width;
  (void)input_channels;
  (void)output_channels;
  (void)kernel_width;
  (void)kernel_height;
  (void)n_chunks;

  const int32_t chunk_oc = OUT_C / N_CHUNKS;
  const int32_t oc_offset = chunk_idx * chunk_oc;
  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);
  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  const int ic_tiles = IN_C / 8;
  const int chunk_oc_tiles = chunk_oc / 8;
  const int output_width = IN_W;
  const int x_tiles = output_width / 4;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches the scalar banker_srs used by the runtime tail.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  // SHAPES_ARE_CONST path uses mmul<8,8,8>: 8 pixels per acc (vs 4 in the
  // runtime fallback's <4,8,8>). Same underlying HW instruction
  // (mac_8x8_8x8_conf); the 8-wide variant exposes more outputs per call,
  // halving acc invocations + epilogue calls. Dense int8 weights only
  // (sparse_vector variants like <4,16,8> are not applicable here).
  using MMUL8x8x8 = aie::mmul<8, 8, 8, int8, int8>;
  using MMUL_T = MMUL8x8x8;
  constexpr int MMUL_M = 8;
  constexpr int MMUL_MN = 64; // M*N outputs per acc

  int8_t *line[3] = {line0, line1, line2};

  constexpr int kXTiles8 = IN_W / 8; // pair_cv1 (IN_W=16): 2
#define AIE_HINT_OC AIE_LOOP_RANGE(chunk_oc_tiles, chunk_oc_tiles)
#define AIE_HINT_X AIE_LOOP_RANGE(kXTiles8, kXTiles8)
#define AIE_HINT_IC AIE_LOOP_RANGE(IN_C / 8, IN_C / 8)

  AIE_HINT_OC
  for (int chunk_oc_t = 0; chunk_oc_t < chunk_oc_tiles; ++chunk_oc_t) {
    const int oc_full_base = oc_offset + chunk_oc_t * 8;

    // Hoisted bias_acc init: bias is constant across x_tile iters for a
    // given chunk_oc_t. Init the mmul with bias instead of zero so the
    // post-mac to_vector<int8>(rs) directly produces the bias-added
    // SRS+saturated result. For MMUL_M=8, replicate bias 8 times (one per
    // pixel) instead of 4.
    aie::accum<acc32, MMUL_MN> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias_full[oc_full_base]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      aie::vector<int32, 64> b64 = aie::concat(b32, b32);
      bias_acc.from_vector(b64);
    }

    // M=8 path: x_tile loop runs IN_W/8 iters (vs IN_W/4 for M=4).
    AIE_HINT_X
    for (int x_tile = 0; x_tile < kXTiles8; ++x_tile) {
      MMUL_T acc;
      acc = bias_acc;

      const int x_out_base = x_tile * MMUL_M;
      const int x_in_base = x_out_base - 1;

      // mmul-layout input read (producers: m_0_split split_a for pair0_cv1
      // or pair_cv2_skip output for pair1_cv1). kx=1 = 1 vlda; kx=0/2 =
      // 2 vlda + shuffle_down. No scalar pack.
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
            int wts_off =
                wts_chunk_tile_off(chunk_oc_t, ic_t, ky, kx, ic_tiles, KH, KW);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }

      aie::vector<int8, 64> srs_v = acc.template to_vector<int8>(right_shift);
      // mmul-layout output: write 64 bytes (8 pixels x 8 chans) as ONE vec
      // store at offset (chunk_oc_t_full, x_tile). Consumer (cv2) reads with
      // a vec load instead of 64 scalar lda.s8 + vpush gather.
      alignas(64) int8_t silu_buf[64];
      AIE_LOOP_UNROLL_FULL
      for (int i = 0; i < 64; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      aie::vector<int8, 64> silu_v = aie::load_v<64>(silu_buf);
      const int chunk_oc_t_full = oc_full_base >> 3;
      aie::store_v(output + chunk_oc_t_full * (kXTiles8 * 64) + x_tile * 64,
                   silu_v);
    }
  }

  event1();
}

} // extern "C"
