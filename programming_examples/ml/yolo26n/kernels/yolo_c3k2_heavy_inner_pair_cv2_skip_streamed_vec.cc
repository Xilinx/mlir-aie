//===- yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_vec.cc ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Vectorized chunked-OC 3x3 stride-1 + cross-scale skip-add. Drop-in
// for yolo_c3k2_heavy_inner_pair_cv2_skip_streamed.cc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"
#include "yolo_kernel_common.h"

// Per-block deep-opt: see pair_cv1_streamed_vec.cc for the pattern. Shape
// constants fold + AIE_LOOP_RANGE hints + bias-init into mmul acc; the
// per-output weighted skip-add (y*mult_y + cv2silu*mult_cv2 + SRS + clamp)
// is vec via aie::mul + aie::mac into an acc<acc32, 64> + to_vector<int8>
// (rs) (only the SiLU LUT gather stays scalar).
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

void KERNEL_NAME(yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_silu_bias_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts_chunk,
    int32_t *bias_full, int8_t *silu_lut, int8_t *skip_row, int8_t *output,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t kernel_width,
    const int32_t kernel_height, const int32_t border,
    const int32_t right_shift, const int32_t n_chunks, const int32_t chunk_idx,
    const int32_t skip_y_mult, const int32_t skip_cv2_mult,
    const int32_t skip_rsh_add) {
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

  // See pair_cv1 _streamed_vec.cc: cast to unsigned so /power-of-2 lowers
  // to shifts instead of calling __divsi3.
  const int32_t chunk_oc = (uint32_t)OUT_C / (uint32_t)N_CHUNKS;
  const int32_t oc_offset = chunk_idx * chunk_oc;
  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);
  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  const int ic_tiles = (uint32_t)IN_C / 8u;
  const int chunk_oc_tiles = (uint32_t)chunk_oc / 8u;
  const int output_width = IN_W;
  const int x_tiles = (uint32_t)output_width / 4u;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  // mmul<8,8,8>: 8 pixels per acc (vs 4); same HW instruction; halves
  // acc invocations + epilogue calls. See pair_cv1_streamed_vec.cc.
  using MMUL8x8x8 = aie::mmul<8, 8, 8, int8, int8>;
  using MMUL_T = MMUL8x8x8;
  constexpr int MMUL_M = 8;
  constexpr int MMUL_MN = 64;

  int8_t *line[3] = {line0, line1, line2};

  constexpr int kXTiles8 = IN_W / 8;
#define AIE_HINT_OC AIE_LOOP_RANGE(chunk_oc_tiles, chunk_oc_tiles)
#define AIE_HINT_X AIE_LOOP_RANGE(kXTiles8, kXTiles8)
#define AIE_HINT_IC AIE_LOOP_RANGE(IN_C / 8, IN_C / 8)

  AIE_HINT_OC
  for (int chunk_oc_t = 0; chunk_oc_t < chunk_oc_tiles; ++chunk_oc_t) {
    const int oc_full_base = oc_offset + chunk_oc_t * 8;

    aie::accum<acc32, MMUL_MN> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias_full[oc_full_base]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      aie::vector<int32, 64> b64 = aie::concat(b32, b32);
      bias_acc.from_vector(b64);
    }

    AIE_HINT_X
    for (int x_tile = 0; x_tile < kXTiles8; ++x_tile) {
      MMUL_T acc;
      acc = bias_acc;

      const int x_out_base = x_tile * MMUL_M;
      const int x_in_base = x_out_base - 1;

      // mmul-layout input read: producer cv1 wrote line buffers as
      // (ic_t, x_block, p*8+chan). For kx=1 the A vec is a single vlda;
      // kx=0/2 load 2 adjacent blocks + shuffle_down. No scalar pack.
      constexpr int kPackedRowStride = (IN_C / 8) * kXTiles8 * 64;
      (void)kPackedRowStride;
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

      // Vec skip-add: gather silu(LUT) + skip into 64-wide vecs, then
      // accumulate (skip*y_mult + cv2silu*cv2_mult) in acc32 and SRS via
      // to_vector<int8>(rs). Saturation + conv_even rounding are set above,
      // so this matches scalar banker_srs + I8_MIN/MAX clamp.
      alignas(64) int8_t silu_buf[64];
      AIE_LOOP_UNROLL_FULL
      for (int i = 0; i < 64; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      aie::vector<int8, 64> silu_v = aie::load_v<64>(silu_buf);
      // skip_row is the mid row of split_a (pair0) or pair0's output (pair1)
      // — both producers now write mmul-packed format, and the orchestration
      // copy preserves bytes. Read skip_v as one vec_load instead of the
      // prior 64 scalar gathers.
      const int chunk_oc_t_full_skip = oc_full_base >> 3;
      aie::vector<int8, 64> skip_v = aie::load_v<64>(
          skip_row + chunk_oc_t_full_skip * (kXTiles8 * 64) + x_tile * 64);

      aie::accum<acc32, 64> add_acc = aie::mul(skip_v, (int16)skip_y_mult);
      add_acc = aie::mac(add_acc, silu_v, (int16)skip_cv2_mult);
      aie::vector<int8, 64> added_v =
          add_acc.template to_vector<int8>(skip_rsh_add);

      // mmul-layout output write: added_v is already (p*8+chan) inside the
      // 64-byte mmul output. Consumers (pair1_cv1 reading inner_0_xt or
      // back cv3 reading inner_1_out) load via the same mmul vec_load path
      // used by pair_cv2's own input.
      const int chunk_oc_t_full = oc_full_base >> 3;
      aie::store_v(output + chunk_oc_t_full * (kXTiles8 * 64) + x_tile * 64,
                   added_v);
    }
  }

  event1();
}

} // extern "C"
