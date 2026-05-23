//===- yolo_c3k2_small_cv2_concat3_vec.cc -------------------------*- C++ -*-===//
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
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

#ifdef YOLO_C3K2_CV2_IN_W
#define IN_W YOLO_C3K2_CV2_IN_W
#define THREE_C YOLO_C3K2_CV2_THREE_C
#define OUT_C YOLO_C3K2_CV2_OUT_C
#define SHAPES_ARE_CONST 1
#else
#define IN_W input_width
#define THREE_C three_c
#define OUT_C output_channels
#define SHAPES_ARE_CONST 0
#endif

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// OIYXI8O8 weight base for one (oc_tile, ic_tile) of a 1x1 conv:
//   (((oc_tile * ic_tiles) + ic_tile)) * 64 bytes.
static inline int wts_tile_off_1x1(int oc_tile, int ic_tile, int ic_tiles) {
  return (oc_tile * ic_tiles + ic_tile) << 6;
}

// Scalar fallback weight index (tail path).
static inline int wts_idx_oiyxi8o8_1x1(int oc_full, int ic_full, int in_c) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

// Per-pixel 8-byte gather (always 4 pixels). C_LOCAL = per-source channel
// stride (c = THREE_C/3, or runtime c).
template <int C_LOCAL>
static __attribute__((always_inline)) inline void
gather4(int8_t *src, int x_base, int local_ic_t, int8_t *a_buf) {
  for (int p = 0; p < 4; ++p) {
    int8_t *psrc = src + (x_base + p) * C_LOCAL + local_ic_t * 8;
    for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = psrc[b];
  }
}

// Scalar epilogue for one mmul<4,8,8> accumulator's worth of output.
static __attribute__((always_inline)) inline void
write_x_tile_result(const aie::vector<int32, 32> &acc_vec,
                    int32_t *bias, int8_t *silu_lut, int8_t *output,
                    int oc_t, int out_c, int x_out_base, int32_t right_shift) {
  for (int p = 0; p < 4; ++p) {
    int x_out = x_out_base + p;
    for (int j = 0; j < 8; ++j) {
      int32_t s = acc_vec[p * 8 + j] + bias[oc_t * 8 + j];
      int32_t sr = banker_srs(s, right_shift);
      if (sr > I8_MAX) sr = I8_MAX;
      if (sr < I8_MIN) sr = I8_MIN;
      output[x_out * out_c + oc_t * 8 + j] = silu_lut[sr + 128];
    }
  }
}

// Vectorized epilogue: bias add as 32-wide int32 vector op, SRS+saturate
// via the accumulator's to_vector<int8>(rs). LUT lookup stays scalar
// (no SIMD gather for int8 LUT on AIE2P). conv_even rounding mode matches
// the scalar banker_srs used by the runtime-fallback tail.
static __attribute__((always_inline)) inline void
write_x_tile_result_vec(aie::mmul<4, 8, 8, int8, int8> &acc,
                        int32_t *bias, int8_t *silu_lut, int8_t *output,
                        int oc_t, int out_c, int x_out_base, int32_t rs) {
  aie::vector<int32, 8>  bias_v8  = aie::load_v<8>(&bias[oc_t * 8]);
  aie::vector<int32, 16> bias_v16 = aie::concat(bias_v8, bias_v8);
  aie::vector<int32, 32> bias_v32 = aie::concat(bias_v16, bias_v16);

  aie::vector<int32, 32> sum_v = acc.template to_vector<int32>();
  sum_v = aie::add(sum_v, bias_v32);

  aie::accum<acc32, 32> sum_acc;
  sum_acc.from_vector(sum_v);
  aie::vector<int8, 32> result = sum_acc.template to_vector<int8>(rs);

  for (int p = 0; p < 4; ++p) {
    int x_out = x_out_base + p;
    for (int j = 0; j < 8; ++j) {
      output[x_out * out_c + oc_t * 8 + j] = silu_lut[int(result[p * 8 + j]) + 128];
    }
  }
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_cv2_concat3_silu_bias_i8_i8)(
    int8_t *in_top,
    int8_t *in_bot,
    int8_t *in_m0,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t three_c,
    const int32_t output_channels,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

#if SHAPES_ARE_CONST
  (void)input_width; (void)three_c; (void)output_channels;
#endif

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

#if SHAPES_ARE_CONST
  constexpr int kC                = THREE_C / 3;
  constexpr int kIcTiles          = THREE_C / 8;
  constexpr int kIcTilesPerSrc    = kC / 8;
  constexpr int kOcTiles          = OUT_C / 8;
  constexpr int kXTiles           = IN_W / 4;
  constexpr int kXPairs           = kXTiles / 2;
  constexpr int kXPairTailStart   = kXPairs * 2;
  // For m2 (IN_W=128, THREE_C=48, OUT_C=64): kXTiles=32, kXPairs=16 (no tail),
  // kIcTiles=6, kIcTilesPerSrc=2, kOcTiles=8.

  // Source pointer per ic_tile: first kIcTilesPerSrc from in_top, next from
  // in_bot, last from in_m0. local_ic_t = ic_t % kIcTilesPerSrc.
  int8_t *src_for_ic_tile[kIcTiles];
  int local_ic_t_for[kIcTiles];
  AIE_LOOP_UNROLL_FULL
  for (int ict = 0; ict < kIcTiles; ++ict) {
    int src_idx = ict / kIcTilesPerSrc;
    src_for_ic_tile[ict] = (src_idx == 0) ? in_top : (src_idx == 1) ? in_bot : in_m0;
    local_ic_t_for[ict] = ict - src_idx * kIcTilesPerSrc;
  }

  for (int oc_t = 0; oc_t < kOcTiles; ++oc_t) {
    // --- 2X x_pair fold: 2 accs per x_pair, all macs back-to-back so
    //     peano can software-pipeline the inner ic_t reduction. ----------
    AIE_LOOP_RANGE(kXPairs, kXPairs)
    for (int xp = 0; xp < kXPairs; ++xp) {
      const int x_tile_base = 2 * xp;
      const int x_out_base  = x_tile_base * 4;
      const int x_in_base0  = x_out_base;
      const int x_in_base1  = x_in_base0 + 4;

      MMUL4x8x8 acc0, acc1;
      acc0 = aie::zeros<acc32, 32>();
      acc1 = aie::zeros<acc32, 32>();

      alignas(32) int8_t a_buf0[32], a_buf1[32];

      AIE_LOOP_RANGE(kIcTiles, kIcTiles)
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        int8_t *src    = src_for_ic_tile[ic_t];
        int local_ic_t = local_ic_t_for[ic_t];
        gather4<kC>(src, x_in_base0, local_ic_t, a_buf0);
        gather4<kC>(src, x_in_base1, local_ic_t, a_buf1);
        aie::vector<int8, 32> in_a0 = aie::load_v<32>(a_buf0);
        aie::vector<int8, 32> in_a1 = aie::load_v<32>(a_buf1);
        int wts_off = wts_tile_off_1x1(oc_t, ic_t, kIcTiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc0.mac(in_a0, in_b);
        acc1.mac(in_a1, in_b);
      }

      write_x_tile_result_vec(acc0, bias, silu_lut, output, oc_t, OUT_C, x_out_base + 0, right_shift);
      write_x_tile_result_vec(acc1, bias, silu_lut, output, oc_t, OUT_C, x_out_base + 4, right_shift);
    }

    // --- x_tile tail ---------------------------------------------------
    for (int x_tile = kXPairTailStart; x_tile < kXTiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();
      const int x_out_base = x_tile * 4;
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        int8_t *src    = src_for_ic_tile[ic_t];
        int local_ic_t = local_ic_t_for[ic_t];
        alignas(32) int8_t a_buf[32];
        gather4<kC>(src, x_out_base, local_ic_t, a_buf);
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
        int wts_off = wts_tile_off_1x1(oc_t, ic_t, kIcTiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }
      write_x_tile_result_vec(acc, bias, silu_lut, output, oc_t, OUT_C, x_out_base, right_shift);
    }
  }
#else
  // Runtime-fallback path (legacy single-acc body).
  const int32_t c = three_c / 3;
  const int ic_tiles = three_c / 8;
  const int ic_tiles_per_src = c / 8;
  const int oc_tiles = output_channels / 8;
  const int x_tiles = input_width / 4;

  int8_t *src_for_ic_tile[48];
  int local_ic_t_for[48];
  for (int ict = 0; ict < ic_tiles; ++ict) {
    int src_idx = ict / ic_tiles_per_src;
    src_for_ic_tile[ict] = (src_idx == 0) ? in_top : (src_idx == 1) ? in_bot : in_m0;
    local_ic_t_for[ict] = ict - src_idx * ic_tiles_per_src;
  }

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();
      const int x_out_base = x_tile * 4;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        int8_t *src = src_for_ic_tile[ic_t];
        int local_ic_t = local_ic_t_for[ic_t];
        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_out_base + p;
          int8_t *psrc = src + col * c + local_ic_t * 8;
          for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = psrc[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      write_x_tile_result(acc_vec, bias, silu_lut, output, oc_t, output_channels, x_out_base, right_shift);
    }

    // Tail scalar fallback for output_width not a multiple of 4.
    for (int x = x_tiles * 4; x < input_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic = 0; ic < three_c; ++ic) {
          int8_t a;
          if (ic < c) a = in_top[x * c + ic];
          else if (ic < 2 * c) a = in_bot[x * c + (ic - c)];
          else a = in_m0[x * c + (ic - 2 * c)];
          sum += a * wts[wts_idx_oiyxi8o8_1x1(oc_full, ic, three_c)];
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX) sr = I8_MAX;
        if (sr < I8_MIN) sr = I8_MIN;
        output[x * output_channels + oc_full] = silu_lut[sr + 128];
      }
    }
  }
#endif

  event1();
}

} // extern "C"
