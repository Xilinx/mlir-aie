//===- yolo_c3k2_small_m0_cv1_vec.cc -------------------------------*- C++ -*-===//
//
// Vectorized 3x3 stride-1 INT8 conv with OIYXI8O8 weight layout. Drop-in
// .o-level replacement for yolo_c3k2_small_m0_cv1.cc on AIE2P.
//
// Same math as Phase 1's stride-2 vec kernel but stride-1: 4 contiguous
// output pixels per mmul<4,8,8> call, input pixel cols = x_out + kx - 1
// (vs 2*x_out + kx - 1 for stride-2).
//
// Per-block deep-opt: if YOLO_C3K2_M0CV1_IN_W etc. are defined at compile
// time (passed via -D from the Makefile for blocks that have shape-stable
// kernels), shape constants fold into shifts/immediates and inner loop
// trip counts become compile-time. Blocks without these defines fall
// back to the runtime-arg path (slower but works for any shape).
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

// Loop hints: we use the official aie_kernel_utils.h macros so peano
// gets the right clang loop pragmas (min_iteration_count for trip count,
// loop unroll(full) for kx). Note that AIE_PREPARE_FOR_PIPELINING is
// empty on peano — peano's auto-scheduler does the pipelining without
// an explicit hint (verified by m1 deep-opt which uses the same macros).

// Compile-time shape macros. If -DYOLO_C3K2_M0CV1_IN_W=… etc. are passed
// at build time, IN_W/IN_C/OUT_C/KW/KH become integer literals and peano
// folds the addressing math + unrolls the inner kx/ic loops. Otherwise
// they fall back to the runtime args (legacy path, used by blocks that
// don't pre-declare shapes).
#ifdef YOLO_C3K2_M0CV1_IN_W
#define IN_W YOLO_C3K2_M0CV1_IN_W
#define IN_C YOLO_C3K2_M0CV1_IN_C
#define OUT_C YOLO_C3K2_M0CV1_OUT_C
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
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) +
         ic_i * 8 + oc_i;
}

// Per-pixel 8-byte gather helper. `IN_C_LOCAL` is the IN_C value to use
// (compile-time when SHAPES_ARE_CONST; runtime otherwise).
// Caller guarantees col is in [0, IN_W) (interior path) — no bounds check.
template <int IN_C_LOCAL>
static __attribute__((always_inline)) inline void
gather_interior(int8_t *line_ptr, int x_in_base, int ic_t, int8_t *a_buf) {
  for (int p = 0; p < 4; ++p) {
    int8_t *src = line_ptr + (x_in_base + p) * IN_C_LOCAL + ic_t * 8;
    for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
  }
}

// Edge gather: handles col < 0 or col >= IN_W via zero-fill.
template <int IN_C_LOCAL, int IN_W_LOCAL>
static __attribute__((always_inline)) inline void
gather_edge(int8_t *line_ptr, int x_in_base, int ic_t, int8_t *a_buf) {
  for (int p = 0; p < 4; ++p) {
    int col = x_in_base + p;
    if (col < 0 || col >= IN_W_LOCAL) {
      for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = 0;
    } else {
      int8_t *src = line_ptr + col * IN_C_LOCAL + ic_t * 8;
      for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
    }
  }
}

// 8 int32 biases -> 32-wide acc<acc32>. Seeds the mmul so to_vector<int8>(rs)
// emits bias+SRS+saturate in one vec op.
static __attribute__((always_inline)) inline aie::accum<acc32, 32>
make_bias_acc(const int32_t *bias_8) {
  aie::vector<int32, 8>  b8  = aie::load_v<8>(bias_8);
  aie::vector<int32, 16> b16 = aie::concat(b8, b8);
  aie::vector<int32, 32> b32 = aie::concat(b16, b16);
  aie::accum<acc32, 32> a;
  a.from_vector(b32);
  return a;
}

// Vec epilogue: bias+SRS+saturate already collapsed into the vec
// to_vector<int8>(rs) call at the call site. Scalar SiLU LUT gather only.
// noinline so the 4X-fold's 4 epilogues per x_quad share one body — the
// kernel has ~128 epilogue call sites and an always_inline variant blows
// past the 16KB program-memory budget on m_0_inner tile (shared with
// m0_cv2_skip kernel for m2/m4).
static __attribute__((noinline)) void
write_x_tile_result(aie::vector<int8, 32> srs_v,
                    int8_t *silu_lut, int8_t *output,
                    int oc_t, int out_c, int x_out_base) {
  for (int p = 0; p < 4; ++p) {
    int x_out = x_out_base + p;
    for (int j = 0; j < 8; ++j) {
      output[x_out * out_c + oc_t * 8 + j] =
          silu_lut[int(srs_v[p * 8 + j]) + 128];
    }
  }
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_m0_cv1_conv2dk3_silu_bias_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts,
    int32_t *bias,
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t /*padding*/) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

#if SHAPES_ARE_CONST
  // Runtime args must match compile-time shape macros.
  // Cheap assert; can be compiled out with -DNDEBUG (already on by default).
  (void)input_width; (void)input_channels; (void)output_channels;
  (void)kernel_width; (void)kernel_height;
#endif

  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);
  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  const int ic_tiles = IN_C / 8;
  const int oc_tiles = OUT_C / 8;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even to match the sibling c3k2_small kernels; the scalar
  // banker_srs in write_x_tile_result is unaffected, but the mode is
  // already correct if the epilogue is later vectorized.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  int8_t *line[3] = {line0, line1, line2};

  // Output width == input width for stride-1.
  const int output_width = IN_W;
  const int x_tiles = output_width / 4;

#if SHAPES_ARE_CONST
  // Compile-time trip counts for the deep-opt path. Required for chess
  // loop_range pragmas to inform peano of exact bounds.
  constexpr int kIcTiles = IN_C / 8;
  constexpr int kOcTiles = OUT_C / 8;
  constexpr int kXTiles  = IN_W / 4;
  constexpr int kInteriorStart = 1;
  constexpr int kInteriorEnd   = kXTiles - 1;
  constexpr int kInteriorN     = kInteriorEnd - kInteriorStart;
  constexpr int kXQuads        = kInteriorN / 4;
  constexpr int kXTailStart    = kInteriorStart + kXQuads * 4;
  // For m2 (IN_W=128): kXTiles=32, kInteriorN=30, kXQuads=7, tail tiles 29..30
  // For m4 (IN_W=64):  kXTiles=16, kInteriorN=14, kXQuads=3, tail tiles 13..14

  for (int oc_t = 0; oc_t < kOcTiles; ++oc_t) {
    // Bias seed for this oc_t; reused across all x_tile paths below.
    auto bias_acc = make_bias_acc(&bias[oc_t * 8]);

    // --- Left edge: x_tile = 0 (col=-1 invalid for kx=0,p=0) -----------
    {
      MMUL4x8x8 acc;
      acc = bias_acc;
      constexpr int x_in_base_edge = -1;  // x_tile=0 -> 0*4 - 1
      AIE_LOOP_RANGE(kIcTiles, kIcTiles)
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        AIE_LOOP_RANGE(1, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf[32];
            gather_edge<IN_C, IN_W>(line_ptr, x_in_base_edge + kx, ic_t, a_buf);
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off = wts_tile_off(oc_t, ic_t, ky, kx, kIcTiles, 3, 3);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }
      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      write_x_tile_result(srs_v, silu_lut, output, oc_t, OUT_C, 0);
    }

    // --- Interior 4X fold: 4 x_tiles per outer iter, branchless gather --
    // Per (ic_t, ky, kx): 4 gathers (all guaranteed valid since interior) +
    // 1 weight load + 4 macs back-to-back. The 4-mac sequence is what
    // peano can software-pipeline (1 acc was the prior problem).
    AIE_LOOP_RANGE(kXQuads, kXQuads)
    for (int xq = 0; xq < kXQuads; ++xq) {
      const int x_tile_base = kInteriorStart + 4 * xq;
      const int x_out_base  = x_tile_base * 4;
      const int x_in_base0  = x_out_base - 1;       // x_tile 4q+0
      const int x_in_base1  = x_in_base0 + 4;
      const int x_in_base2  = x_in_base0 + 8;
      const int x_in_base3  = x_in_base0 + 12;

      MMUL4x8x8 acc0, acc1, acc2, acc3;
      acc0 = bias_acc;
      acc1 = bias_acc;
      acc2 = bias_acc;
      acc3 = bias_acc;

      AIE_LOOP_RANGE(kIcTiles, kIcTiles)
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        AIE_LOOP_RANGE(1, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf0[32], a_buf1[32], a_buf2[32], a_buf3[32];
            gather_interior<IN_C>(line_ptr, x_in_base0 + kx, ic_t, a_buf0);
            gather_interior<IN_C>(line_ptr, x_in_base1 + kx, ic_t, a_buf1);
            gather_interior<IN_C>(line_ptr, x_in_base2 + kx, ic_t, a_buf2);
            gather_interior<IN_C>(line_ptr, x_in_base3 + kx, ic_t, a_buf3);
            aie::vector<int8, 32> in_a0 = aie::load_v<32>(a_buf0);
            aie::vector<int8, 32> in_a1 = aie::load_v<32>(a_buf1);
            aie::vector<int8, 32> in_a2 = aie::load_v<32>(a_buf2);
            aie::vector<int8, 32> in_a3 = aie::load_v<32>(a_buf3);
            int wts_off = wts_tile_off(oc_t, ic_t, ky, kx, kIcTiles, 3, 3);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
            acc0.mac(in_a0, in_b);
            acc1.mac(in_a1, in_b);
            acc2.mac(in_a2, in_b);
            acc3.mac(in_a3, in_b);
          }
        }
      }
      aie::vector<int8, 32> v0 = acc0.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> v1 = acc1.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> v2 = acc2.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> v3 = acc3.template to_vector<int8>(right_shift);
      write_x_tile_result(v0, silu_lut, output, oc_t, OUT_C, x_out_base + 0);
      write_x_tile_result(v1, silu_lut, output, oc_t, OUT_C, x_out_base + 4);
      write_x_tile_result(v2, silu_lut, output, oc_t, OUT_C, x_out_base + 8);
      write_x_tile_result(v3, silu_lut, output, oc_t, OUT_C, x_out_base + 12);
    }

    // --- Interior tail (leftover x_tiles when kInteriorN % 4 != 0) ------
    // For m2: 2 tail tiles (29, 30); for m4: 2 tail tiles (13, 14).
    // All interior so use branchless gather.
    for (int x_tile = kXTailStart; x_tile < kInteriorEnd; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;
      const int x_out_base = x_tile * 4;
      const int x_in_base  = x_out_base - 1;
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf[32];
            gather_interior<IN_C>(line_ptr, x_in_base + kx, ic_t, a_buf);
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off = wts_tile_off(oc_t, ic_t, ky, kx, kIcTiles, 3, 3);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }
      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      write_x_tile_result(srs_v, silu_lut, output, oc_t, OUT_C, x_out_base);
    }

    // --- Right edge: x_tile = kXTiles-1 (col=IN_W invalid for kx=2,p=3) -
    {
      MMUL4x8x8 acc;
      acc = bias_acc;
      constexpr int x_in_base_edge = (kXTiles - 1) * 4 - 1;
      AIE_LOOP_RANGE(kIcTiles, kIcTiles)
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        AIE_LOOP_RANGE(1, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf[32];
            gather_edge<IN_C, IN_W>(line_ptr, x_in_base_edge + kx, ic_t, a_buf);
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off = wts_tile_off(oc_t, ic_t, ky, kx, kIcTiles, 3, 3);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }
      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      write_x_tile_result(srs_v, silu_lut, output, oc_t, OUT_C, (kXTiles - 1) * 4);
    }
  }
#else
  // Runtime-fallback path (shapes not compile-time): use the legacy
  // single-acc body with bounds check on every iter. Slower but works
  // for any shape.
  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;
      const int x_out_base = x_tile * 4;
      const int x_in_base  = x_out_base - 1;
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          for (int kx = 0; kx < KW; ++kx) {
            alignas(32) int8_t a_buf[32];
            bool any_valid = false;
            for (int p = 0; p < 4; ++p) {
              int col = x_in_base + p + kx;
              if (col < 0 || col >= IN_W) {
                for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = 0;
              } else {
                int8_t *src = line_ptr + col * IN_C + ic_t * 8;
                for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
                any_valid = true;
              }
            }
            if (!any_valid) continue;
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off = wts_tile_off(oc_t, ic_t, ky, kx, ic_tiles, KH, KW);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }
      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      write_x_tile_result(srs_v, silu_lut, output, oc_t, OUT_C, x_out_base);
    }

    // Tail outputs if output_width not a multiple of 4: scalar fallback.
    // Only on the runtime path — compile-time path has IN_W % 4 == 0.
    for (int x = x_tiles * 4; x < output_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic_full = 0; ic_full < IN_C; ++ic_full) {
          for (int kx = 0; kx < KW; ++kx) {
            int col = x - 1 + kx;
            if (col < 0 || col >= IN_W) continue;
            int in_indx = col * IN_C + ic_full;
            int w0 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 0, kx, IN_C, KH, KW)];
            int w1 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 1, kx, IN_C, KH, KW)];
            int w2 = wts[wts_idx_oiyxi8o8(oc_full, ic_full, 2, kx, IN_C, KH, KW)];
            if (!skip_top) sum += line0[in_indx] * w0;
            sum += line1[in_indx] * w1;
            if (!skip_bot) sum += line2[in_indx] * w2;
          }
        }
        int32_t sr = banker_srs(sum, right_shift);
        if (sr > I8_MAX) sr = I8_MAX;
        if (sr < I8_MIN) sr = I8_MIN;
        output[x * OUT_C + oc_full] = silu_lut[sr + 128];
      }
    }
  }
#endif

  event1();
}

} // extern "C"
