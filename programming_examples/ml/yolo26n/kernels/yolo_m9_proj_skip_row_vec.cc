//===- yolo_m9_proj_skip_row_vec.cc ------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Vectorized fused attn/proj 1x1 + cross-scale skip-add (b) for the PSA
// pipe. Drop-in .o-level replacement for yolo_m9_proj_skip_row.cc.
//
// Math:
//   proj_q = clip_i8(banker_srs(acc + bias, right_shift))
//   add_q  = proj_q + (b_row[x, oc] << skip_shift)
//   out_i8 = clip_i8(banker_srs(add_q, skip_shift))
//
// Master-class pattern (ported from yolo_m9_cv1_split_vec.cc):
//  - Compile-time shape #defines fold all addressing.
//  - YCXC8 prepack scratch: input row packed once per call, inner mmul
//    loop does aligned vec_load<32> instead of per-iter scalar a_buf gather.
//  - Multi-acc fold across kXTiles=4 x_tiles per oc_t: ONE B-load serves
//    4 independent accs, breaking the inner ic_t reduction dep chain.
//  - Bias-seeded mmul: to_vector<int8>(rs) emits bias+SRS+saturate.
//
// Skip-add stays scalar: vec mul+mac chain crashes peano (see
// reference_peano_to_vector_int8_crash.md — the narrow→aie::mul SSA chain
// trips a LOAD+UNPACK fusion bug even with store/reload+asm barriers).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef YOLO_M9_PROJ_IN_W
#error "YOLO_M9_PROJ_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M9_PROJ_IN_C
#error "YOLO_M9_PROJ_IN_C must be defined at compile time"
#endif
#ifndef YOLO_M9_PROJ_OUT_C
#error "YOLO_M9_PROJ_OUT_C must be defined at compile time"
#endif

static constexpr int kInW = YOLO_M9_PROJ_IN_W;
static constexpr int kInC = YOLO_M9_PROJ_IN_C;
static constexpr int kOutC = YOLO_M9_PROJ_OUT_C;
static constexpr int kIcTiles = kInC / 8;
static constexpr int kOcTiles = kOutC / 8;
static constexpr int kXTiles = kInW / 4;

static_assert(kInC % 8 == 0, "PROJ IN_C must be multiple of 8");
static_assert(kOutC % 8 == 0, "PROJ OUT_C must be multiple of 8");
static_assert(kInW % 4 == 0, "PROJ IN_W must be multiple of 4");

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void yolo_m9_proj_skip_row_i8_i8(
    int8_t *in_row, int8_t *b_cache, int8_t *wts, int32_t *bias,
    int8_t *out_row, const int32_t yi, const int32_t /*input_width*/,
    const int32_t /*input_channels*/, const int32_t /*output_channels*/,
    const int32_t right_shift, const int32_t skip_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  int8_t *b_row = b_cache + yi * kInW * kOutC;

  // YCXC8 prepack of input row, one uint64 word per (x, ic_t) entry.
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

    // Per x_tile epilogue: narrow → scalar skip-add tail → uint64 strided
    // store per pixel. Scalar skip-add unavoidable (peano vec narrow bug,
    // see file header).
    AIE_LOOP_UNROLL_FULL
    for (int xt = 0; xt < kXTiles; ++xt) {
      aie::vector<int8, 32> proj_v =
          acc[xt].template to_vector<int8>(right_shift);
      const int x_out_base = xt * 4;

      // Load b for the 4 pixels in this x_tile (4 uint64s, stride kOutC).
      alignas(32) int8_t b_buf[32];
      int8_t *b0 = b_row + (x_out_base + 0) * kOutC + oc_t * 8;
      *reinterpret_cast<uint64_t *>(&b_buf[0]) =
          *reinterpret_cast<const uint64_t *>(b0);
      *reinterpret_cast<uint64_t *>(&b_buf[8]) =
          *reinterpret_cast<const uint64_t *>(b0 + kOutC);
      *reinterpret_cast<uint64_t *>(&b_buf[16]) =
          *reinterpret_cast<const uint64_t *>(b0 + 2 * kOutC);
      *reinterpret_cast<uint64_t *>(&b_buf[24]) =
          *reinterpret_cast<const uint64_t *>(b0 + 3 * kOutC);
      aie::vector<int8, 32> b_v = aie::load_v<32>(b_buf);

      AIE_LOOP_UNROLL_FULL
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        alignas(8) int8_t pix_buf[8];
        AIE_LOOP_UNROLL_FULL
        for (int j = 0; j < 8; ++j) {
          int32_t pq = (int32_t)proj_v[p * 8 + j];
          int32_t bb = (int32_t)b_v[p * 8 + j];
          int32_t add_q = pq + (bb << skip_shift);
          int32_t add_i8 = banker_srs(add_q, skip_shift);
          if (add_i8 > I8_MAX)
            add_i8 = I8_MAX;
          if (add_i8 < I8_MIN)
            add_i8 = I8_MIN;
          pix_buf[j] = (int8_t)add_i8;
        }
        *reinterpret_cast<uint64_t *>(&out_row[x_out * kOutC + oc_t * 8]) =
            *reinterpret_cast<const uint64_t *>(pix_buf);
      }
    }
  }

  event1();
}

} // extern "C"
