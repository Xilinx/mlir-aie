//===- yolo_c3k2_heavy_inner_pair_cv1_vec.cc -------------------------------*-
// C++ -*-===//
//
// Vectorized 3x3 stride-1 INT8 conv with OIYXI8O8 weight layout. Drop-in
// .o-level replacement for yolo_c3k2_small_m0_cv1.cc on AIE2P.
//
// Same math as Phase 1's stride-2 vec kernel but stride-1: 4 contiguous
// output pixels per mmul<4,8,8> call, input pixel cols = x_out + kx - 1
// (vs 2*x_out + kx - 1 for stride-2).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

// Per-block symbol mangling.
#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

#ifdef YOLO_M6_PAIR_IN_W
#define IN_W YOLO_M6_PAIR_IN_W
#define IN_C YOLO_M6_PAIR_IN_C
#define OUT_C YOLO_M6_PAIR_OUT_C
#define KW 3
#define KH 3
#define SHAPES_ARE_CONST 1
#else
#define IN_W input_width
#define IN_C input_channels
#define OUT_C output_channels
#define KW kernel_width
#define KH kernel_height
#define SHAPES_ARE_CONST 0
#endif

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

static inline int wts_tile_off(int oc_tile, int ic_tile, int ky, int kx,
                               int ic_tiles, int kH, int kW) {
  return (((oc_tile * ic_tiles + ic_tile) * kH + ky) * kW + kx) << 6;
}

// Scalar fallback weight index (tail path).
static inline int wts_idx_oiyxi8o8(int oc_full, int ic_full, int ky, int kx,
                                   int in_c, int kH, int kW) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) + ic_i * 8 +
         oc_i;
}

#if SHAPES_ARE_CONST
// 3x3 conv mmul A load with kx slide (kx=1 fast path; kx=0/2 use 2 vlda +
// shuffle_down). Consumer-side mirror of the producer's mmul-packed output
// (m_0_split's split_a or pair_cv2_skip's output). Mirror of m8 streamed
// helper.
static __attribute__((always_inline)) inline aie::vector<int8, 64>
load_a_mmul_kx(int8_t *line_ptr, int ic_t, int x_tile, int kx, int stride,
               int kXTiles8) {
  int8_t *base = line_ptr + ic_t * stride;
  if (kx == 1) {
    return aie::load_v<64>(base + x_tile * 64);
  }
  int blk_lo = (kx == 0) ? x_tile - 1 : x_tile;
  int blk_hi = blk_lo + 1;
  aie::vector<int8, 64> lo = (blk_lo >= 0 && blk_lo < kXTiles8)
                                 ? aie::load_v<64>(base + blk_lo * 64)
                                 : aie::zeros<int8, 64>();
  aie::vector<int8, 64> hi = (blk_hi >= 0 && blk_hi < kXTiles8)
                                 ? aie::load_v<64>(base + blk_hi * 64)
                                 : aie::zeros<int8, 64>();
  aie::vector<int8, 128> combined = aie::concat(lo, hi);
  const unsigned shift = (kx == 0) ? 56u : 8u;
  return aie::shuffle_down(combined, shift).template extract<64>(0);
}
#endif

extern "C" {

// Bank-aware example: line[0/1/2] qualified as bank B, wts as bank A,
// paired with IRON-side pinning in aie2_yolo_per_block.py (split_a /
// inner_0_out consumer_mem_banks=[1], wts Buffer mem_bank=0). Intent:
// let peano parallel-issue acts+wts loads in the inner mac loop.
//
// VERIFIED EFFECT ON THIS KERNEL: zero — peano emits a byte-identical
// .o with and without the qualifiers (md5 match). For statically-known
// disjoint buffers, peano's aliasing analysis already issues the loads
// in parallel without help. The qualifier matters only when peano's
// analysis CAN'T prove non-aliasing (aliased pointers, scratchpads,
// dynamically-resolved addresses). Kept as a worked example of the
// producer_mem_bank / consumer_mem_banks IRON API for those cases.
void KERNEL_NAME(yolo_c3k2_heavy_inner_pair_cv1_conv2dk3_silu_bias_i8_i8)(
    int8_t __aie_dm_resource_b *line0, int8_t __aie_dm_resource_b *line1,
    int8_t __aie_dm_resource_b *line2, int8_t __aie_dm_resource_a *wts,
    int32_t *bias, int8_t *silu_lut, int8_t *output,
    const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int32_t kernel_width, const int32_t kernel_height,
    const int32_t border, const int32_t right_shift,
    const int32_t /*padding*/) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

#if SHAPES_ARE_CONST
  (void)input_width;
  (void)input_channels;
  (void)output_channels;
  (void)kernel_width;
  (void)kernel_height;
#endif

  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);
  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

#if SHAPES_ARE_CONST
  constexpr int ic_tiles = IN_C / 8;
  constexpr int oc_tiles = OUT_C / 8;
  constexpr int output_width = IN_W;
  constexpr int x_tiles = output_width / 4;
#define AIE_HINT_OC AIE_LOOP_RANGE(oc_tiles, oc_tiles)
#define AIE_HINT_X AIE_LOOP_RANGE(x_tiles, x_tiles)
#define AIE_HINT_IC AIE_LOOP_RANGE(ic_tiles, ic_tiles)
#define AIE_HINT_KX AIE_LOOP_RANGE(3, 3)
#else
  const int ic_tiles = IN_C / 8;
  const int oc_tiles = OUT_C / 8;
  const int output_width = IN_W;
  const int x_tiles = output_width / 4;
#define AIE_HINT_OC
#define AIE_HINT_X
#define AIE_HINT_IC
#define AIE_HINT_KX
#endif

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

#if SHAPES_ARE_CONST
  // SHAPES_ARE_CONST path: mmul<8,8,8> (8 pixels per acc, halves epilogue
  // calls vs <4,8,8>) + hoisted pixel vec gather (load 8 pixels' IN_C
  // channels ONCE per (ky,kx); ic_t inner extracts the 8-byte slice from
  // registers instead of re-reading DM). Mirrors the m8 streamed pair_cv1
  // upgrade. Requires IN_W divisible by 8 — m6 IN_W=32 satisfies this.
  using MMUL8x8x8 = aie::mmul<8, 8, 8, int8, int8>;
  constexpr int MMUL_M = 8;
  constexpr int MMUL_MN = 64;
  constexpr int kXTiles8 = IN_W / 8;
#else
  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;
#endif

  int8_t *line[3] = {line0, line1, line2};

  AIE_HINT_OC
  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
#if SHAPES_ARE_CONST
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

      // mmul-packed input read (producer is m_0_split's split_a or
      // pair_cv2_skip's mmul-packed output). kx=1 single vlda; kx=0/2 two
      // vlda + shuffle_down. No scalar pack.
      constexpr int kIcStride = kXTiles8 * 64;

      AIE_LOOP_RANGE(1, 3)
      for (int ky = ky_start; ky < ky_end; ++ky) {
        int8_t *line_ptr = line[ky];

        AIE_LOOP_RANGE(3, 3)
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
      // mmul-packed output: vec_store consumed by pair_cv2_skip vec_load.
      alignas(64) int8_t silu_buf[64];
      for (int i = 0; i < 64; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      aie::vector<int8, 64> silu_v = aie::load_v<64>(silu_buf);
      aie::store_v(output + oc_t * (kXTiles8 * 64) + x_tile * 64, silu_v);
    }
#else
    aie::accum<acc32, 32> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias[oc_t * 8]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      bias_acc.from_vector(b32);
    }

    AIE_HINT_X
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;

      const int x_out_base = x_tile * 4;
      const int x_in_base = x_out_base - 1;

      AIE_HINT_IC
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        AIE_LOOP_RANGE(1, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];

          AIE_HINT_KX
          for (int kx = 0; kx < KW; ++kx) {
            alignas(32) int8_t a_buf[32];
            bool any_valid = false;
            for (int p = 0; p < 4; ++p) {
              int col = x_in_base + p + kx;
              if (col < 0 || col >= IN_W) {
                for (int b = 0; b < 8; ++b)
                  a_buf[p * 8 + b] = 0;
              } else {
                int8_t *src = line_ptr + col * IN_C + ic_t * 8;
                for (int b = 0; b < 8; ++b)
                  a_buf[p * 8 + b] = src[b];
                any_valid = true;
              }
            }
            if (!any_valid)
              continue;
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

            int wts_off = wts_tile_off(oc_t, ic_t, ky, kx, ic_tiles, KH, KW);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }

      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        for (int j = 0; j < 8; ++j) {
          output[x_out * OUT_C + oc_t * 8 + j] =
              silu_lut[int(srs_v[p * 8 + j]) + 128];
        }
      }
    }
#endif

    // Tail outputs if output_width not a multiple of 4: scalar fallback.
    for (int x = x_tiles * 4; x < output_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic_full = 0; ic_full < input_channels; ++ic_full) {
          for (int kx = 0; kx < kernel_width; ++kx) {
            int col = x - 1 + kx;
            if (col < 0 || col >= input_width)
              continue;
            int in_indx = col * input_channels + ic_full;
            int w0 =
                wts[wts_idx_oiyxi8o8(oc_full, ic_full, 0, kx, input_channels,
                                     kernel_height, kernel_width)];
            int w1 =
                wts[wts_idx_oiyxi8o8(oc_full, ic_full, 1, kx, input_channels,
                                     kernel_height, kernel_width)];
            int w2 =
                wts[wts_idx_oiyxi8o8(oc_full, ic_full, 2, kx, input_channels,
                                     kernel_height, kernel_width)];
            if (!skip_top)
              sum += line0[in_indx] * w0;
            sum += line1[in_indx] * w1;
            if (!skip_bot)
              sum += line2[in_indx] * w2;
          }
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX)
          sr = I8_MAX;
        if (sr < I8_MIN)
          sr = I8_MIN;
        output[x * output_channels + oc_full] = silu_lut[sr + 128];
      }
    }
  }

  event1();
}

} // extern "C"
