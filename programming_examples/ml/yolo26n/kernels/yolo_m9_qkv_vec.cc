//===- yolo_m9_qkv_vec.cc ----------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Vectorized 1x1 INT8 conv 128 -> 256 (no activation; bias-init only).
// Drop-in .o-level replacement for yolo_m9_qkv.cc (same symbol + ABI).
//
// Master-class pattern (ported directly from yolo_m9_cv1_split_vec.cc):
//  - Compile-time shape #defines fold all addressing into shifts/immediates
//    and dead-strip every fallback path.
//  - Input row is pre-packed into a 2 KB YCXC8 scratch ONCE per call, so the
//    inner mmul loop does a single aligned aie::load_v<32> per (x_tile,
//    ic_tile) — no scalar a_buf gather inside the hot loop.
//  - Multi-acc fold across all kXTiles=4 x_tiles per oc_t iter: ONE B-load
//    serves 4 independent accs, breaking the inner ic_t reduction dep
//    chain so peano can pipeline tighter.
//  - Bias-seeded mmul: to_vector<int8>(rs) directly emits bias+SRS+saturate.
//  - uint64 word copies for both scratch pack and output emit (peano does
//    not reliably auto-lower the scalar byte loop — see
//    feedback_explicit_uint64_over_byte_loop).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef YOLO_M9_QKV_IN_W
#error "YOLO_M9_QKV_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M9_QKV_IN_C
#error "YOLO_M9_QKV_IN_C must be defined at compile time"
#endif
#ifndef YOLO_M9_QKV_OUT_C
#error "YOLO_M9_QKV_OUT_C must be defined at compile time"
#endif

static constexpr int kInW = YOLO_M9_QKV_IN_W;
static constexpr int kInC = YOLO_M9_QKV_IN_C;
static constexpr int kOutC = YOLO_M9_QKV_OUT_C;
static constexpr int kIcTiles = kInC / 8;
static constexpr int kOcTiles = kOutC / 8;
static constexpr int kXTiles = kInW / 4;

static_assert(kInC % 8 == 0, "QKV IN_C must be multiple of 8");
static_assert(kOutC % 8 == 0, "QKV OUT_C must be multiple of 8");
static_assert(kInW % 4 == 0, "QKV IN_W must be multiple of 4");

extern "C" {

void yolo_m9_qkv_i8_i8(int8_t *in_row, int8_t *wts, int32_t *bias,
                       int8_t *out_row, const int32_t /*input_width*/,
                       const int32_t /*input_channels*/,
                       const int32_t /*output_channels*/,
                       const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  // Pre-pack input row from natural (x, ic) to YCXC8 (ic_t, x, ic_i).
  // 2 KB scratch (qkv: 16 ic_tiles × 16 x × 8 ic_i). Explicit uint64
  // word copies (peano doesn't reliably lower the scalar byte loop).
  alignas(32) int8_t scratch[kIcTiles * kInW * 8];
  AIE_LOOP_RANGE(kInW, kInW)
  for (int x = 0; x < kInW; ++x) {
    AIE_LOOP_RANGE(kIcTiles, kIcTiles)
    for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
      const int8_t *__restrict src = in_row + x * kInC + ic_t * 8;
      int8_t *__restrict d = scratch + ic_t * kInW * 8 + x * 8;
      *reinterpret_cast<uint64_t *>(d) =
          *reinterpret_cast<const uint64_t *>(src);
    }
  }

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  // Multi-acc fold across x_tiles: process all kXTiles=4 x_tiles per oc_t
  // iter so one B-load serves 4 independent accs.
  AIE_LOOP_RANGE(kOcTiles, kOcTiles)
  for (int oc_t = 0; oc_t < kOcTiles; ++oc_t) {
    aie::accum<acc32, 32> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias[oc_t * 8]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      bias_acc.from_vector(b32);
    }
    MMUL4x8x8 acc[kXTiles];
    AIE_LOOP_UNROLL_FULL
    for (int xt = 0; xt < kXTiles; ++xt)
      acc[xt] = bias_acc;

    const int8_t *__restrict b_ptr = wts + ((oc_t * kIcTiles) << 6);

    AIE_LOOP_RANGE(kIcTiles, kIcTiles)
    for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
      aie::vector<int8, 64> in_b = aie::load_v<64>(b_ptr);
      b_ptr += 64;
      const int8_t *__restrict a_base = scratch + ic_t * kInW * 8;
      AIE_LOOP_UNROLL_FULL
      for (int xt = 0; xt < kXTiles; ++xt) {
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_base + xt * 32);
        acc[xt].mac(in_a, in_b);
      }
    }

    // Bias-baked vec SRS+saturate per x_tile, then 4 uint64 strided stores
    // per pixel into the (W, oc) raster output.
    AIE_LOOP_UNROLL_FULL
    for (int xt = 0; xt < kXTiles; ++xt) {
      aie::vector<int8, 32> srs_v =
          acc[xt].template to_vector<int8>(right_shift);
      alignas(8) int8_t out_buf[32];
      aie::store_v(out_buf, srs_v);
      const int x_out_base = xt * 4;
      AIE_LOOP_UNROLL_FULL
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        int8_t *__restrict dst = out_row + x_out * kOutC + oc_t * 8;
        *reinterpret_cast<uint64_t *>(dst) =
            *reinterpret_cast<const uint64_t *>(&out_buf[p * 8]);
      }
    }
  }

  event1();
}

} // extern "C"
