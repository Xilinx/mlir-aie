//===- yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_vec.cc -----------------*- C++
//-*-===//
//
// Deep-opt vectorized 3x3 stride-2 INT8 conv with OIYXI8O8 weight layout.
// Drop-in for the m1 conv_stride block; bit-exact with the prior naive vec.
//
// Pattern is the same toolbox as the chunked variant (m3/m5/m7) — compile-
// time shape #defines, 3-way x_tile split (left edge / interior /
// right edge with edges keeping the bounds check and interior branch-free),
// 2X×2OC accumulator fold inside the interior. The differences from the
// chunked version:
//   - No chunked-OC framing: per call processes the full out_c (m1 = 32)
//     so the worker outer loop iterates oc_tiles/2 = 2 oc_pairs.
//   - No oc_offset arg (no chunking).
//   - Smaller weight buffer (m1 weights are ~5 KB, fit on tile alongside
//     the activation; no streaming).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"
#include "yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.h"

#ifndef YOLO_M1_IN_W
#error "YOLO_M1_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M1_IN_C
#error "YOLO_M1_IN_C must be defined at compile time"
#endif
#ifndef YOLO_M1_OUT_C
#error "YOLO_M1_OUT_C must be defined at compile time"
#endif

static constexpr int kInW = YOLO_M1_IN_W;
static constexpr int kInC = YOLO_M1_IN_C;
static constexpr int kOutC = YOLO_M1_OUT_C;
static constexpr int kKW = 3;
static constexpr int kKH = 3;

static constexpr int kOutW = kInW / 2;
static constexpr int kIcTiles = kInC / 8;
static constexpr int kOcTiles = kOutC / 8;
static constexpr int kXTiles = kOutW / 4;
static constexpr int kOcPairs = kOcTiles / 2;
// Per-(ic,ky,kx) stride to step from oc_pair's first OC bank to the
// second (oc_t+1). One oc_tile takes ic_tiles*kH*kW*64 bytes.
static constexpr int kOcBankStride = (kIcTiles * kKH * kKW) << 6;

static_assert(kInW % 2 == 0, "M1 IN_W must be even (stride 2)");
static_assert(kInC % 8 == 0, "M1 IN_C must be multiple of 8");
static_assert(kOutC % 16 == 0, "M1 OUT_C must be multiple of 16 (OCx2 fold)");
static_assert(kOutW % 4 == 0, "M1 OUT_W must be multiple of 4");
static_assert(kXTiles >= 2, "M1 needs x_tiles >= 2 for the edge split");

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// OIYXI8O8 weight base offset for one (oc_tile, ic_tile, ky, kx).
// The 64 bytes at this offset are [ic_inner=8][oc_inner=8] row-major,
// matching aie::mmul.B layout.
static inline int wts_tile_off(int oc_tile, int ic_tile, int ky, int kx) {
  return (((oc_tile * kIcTiles + ic_tile) * kKH + ky) * kKW + kx) << 6;
}

static inline int wts_idx_oiyxi8o8(int oc_full, int ic_full, int ky, int kx) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * kIcTiles + ic_t) * kKH + ky) * kKW + kx) << 6) + ic_i * 8 +
         oc_i;
}

using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

// 8 int32 biases → 32-wide acc<acc32> (4 pix × 8 ch). Seeds the mmul so
// to_vector<int8>(rs) emits bias+SRS+saturate in one vec op.
static __attribute__((always_inline)) inline aie::accum<acc32, 32>
make_bias_acc(const int32_t *__restrict bias_8) {
  aie::vector<int32, 8> b8 = aie::load_v<8>(bias_8);
  aie::vector<int32, 16> b16 = aie::concat(b8, b8);
  aie::vector<int32, 32> b32 = aie::concat(b16, b16);
  aie::accum<acc32, 32> a;
  a.from_vector(b32);
  return a;
}

// Per-OC×4-pix epilog: SiLU LUT (SRS+clamp+bias collapsed into the vec
// to_vector<int8>(rs) at the call site).
static __attribute__((always_inline)) inline void write_x_tile_result(
    aie::vector<int8, 32> srs_v, const int8_t *__restrict silu_lut,
    int8_t *__restrict output, int oc_full_base, int x_out_base) {
  for (int p = 0; p < 4; ++p) {
    int x_out = x_out_base + p;
    for (int j = 0; j < 8; ++j) {
      int oc_full = oc_full_base + j;
      output[x_out * kOutC + oc_full] = silu_lut[int(srs_v[p * 8 + j]) + 128];
    }
  }
}

// Edge gather: 4 pixels × 8 ic, with per-pixel bounds check + zero-fill.
static __attribute__((always_inline)) inline bool
gather_edge(int8_t *__restrict line_ptr, int x_in_base, int kx, int ic_t,
            int8_t *__restrict a_buf) {
  bool any_valid = false;
  for (int p = 0; p < 4; ++p) {
    int col = x_in_base + 2 * p + kx;
    if (col < 0 || col >= kInW) {
      *(reinterpret_cast<uint64_t *>(&a_buf[p * 8])) = 0;
    } else {
      int8_t *src = line_ptr + col * kInC + ic_t * 8;
      *(reinterpret_cast<uint64_t *>(&a_buf[p * 8])) =
          *reinterpret_cast<const uint64_t *>(src);
      any_valid = true;
    }
  }
  return any_valid;
}

// Interior gather: 4 pixels × 8 ic, no bounds check; src pointers
// computed by stride-2 column step + kx + ic_t * 8.
static __attribute__((always_inline)) inline void
gather_interior(int8_t *__restrict line_ptr, int x_in_base, int kx, int ic_t,
                int8_t *__restrict a_buf) {
  int8_t *s0 = line_ptr + (x_in_base + kx) * kInC + ic_t * 8;
  int8_t *s1 = s0 + 2 * kInC;
  int8_t *s2 = s0 + 4 * kInC;
  int8_t *s3 = s0 + 6 * kInC;
  *(reinterpret_cast<uint64_t *>(&a_buf[0])) =
      *reinterpret_cast<const uint64_t *>(s0);
  *(reinterpret_cast<uint64_t *>(&a_buf[8])) =
      *reinterpret_cast<const uint64_t *>(s1);
  *(reinterpret_cast<uint64_t *>(&a_buf[16])) =
      *reinterpret_cast<const uint64_t *>(s2);
  *(reinterpret_cast<uint64_t *>(&a_buf[24])) =
      *reinterpret_cast<const uint64_t *>(s3);
}

static void yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_vec(
    int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts, int32_t *bias,
    int8_t *silu_lut, int8_t *output, const int32_t /*input_width*/,
    const int32_t /*input_channels*/, const int32_t /*output_channels*/,
    const int32_t /*kernel_width*/, const int32_t /*kernel_height*/,
    const int32_t border, const int32_t right_shift,
    const int32_t /*padding*/) {
  event0();

  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);
  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs (round-half-to-even); enables
  // vec to_vector<int8>(rs) in place of the scalar SRS+clamp tail.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  int8_t *line[3] = {line0, line1, line2};

  // Outer loop over OC pairs. m1: oc_pairs=2. Each pair processes
  // 2 oc_tiles (16 OCs total per pair) in lockstep via 2X×2OC fold
  // inside the interior body, or single-X×2OC on the edges.
  AIE_LOOP_RANGE(kOcPairs, kOcPairs)
  for (int oc_pair = 0; oc_pair < kOcPairs; ++oc_pair) {
    const int oc_t_a = oc_pair * 2;
    const int oc_t_b = oc_t_a + 1;
    const int oc_full_base_a = oc_t_a * 8;
    const int oc_full_base_b = oc_t_b * 8;

    // Bias seeds for this oc_pair, reused across all x_tile iters.
    auto bias_acc_a = make_bias_acc(&bias[oc_full_base_a]);
    auto bias_acc_b = make_bias_acc(&bias[oc_full_base_b]);

    // ===== Left edge: x_tile=0 — kx=0 makes col=-1 invalid for p=0. =====
    {
      MMUL4x8x8 acc_a, acc_b;
      acc_a = bias_acc_a;
      acc_b = bias_acc_b;
      const int x_in_base = -1;
      AIE_LOOP_RANGE(kIcTiles, kIcTiles)
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        AIE_LOOP_RANGE(1, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf[32];
            bool any_valid = gather_edge(line_ptr, x_in_base, kx, ic_t, a_buf);
            if (!any_valid)
              continue;
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off = wts_tile_off(oc_t_a, ic_t, ky, kx);
            acc_a.mac(in_a, aie::load_v<64>(&wts[wts_off]));
            acc_b.mac(in_a, aie::load_v<64>(&wts[wts_off + kOcBankStride]));
          }
        }
      }
      aie::vector<int8, 32> srs_a = acc_a.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> srs_b = acc_b.template to_vector<int8>(right_shift);
      write_x_tile_result(srs_a, silu_lut, output, oc_full_base_a, 0);
      write_x_tile_result(srs_b, silu_lut, output, oc_full_base_b, 0);
    }

    // ===== Interior: x_tile in [1, kXTiles-1]. 2X×2OC fold. =====
    // Pair up adjacent x_tiles; one A-gather per (ic, ky, kx) feeds
    // 4 mmul.macs (2 X positions × 2 OC banks). x_in_base for the
    // second x_tile is +8 cols from the first (stride-2 over 4 outs).
    constexpr int kInteriorStart = 1;
    constexpr int kInteriorEnd = kXTiles - 1;
    constexpr int kInteriorN = kInteriorEnd - kInteriorStart;
    constexpr int kXPairs = kInteriorN >> 1;
    constexpr bool kHasXTail = (kInteriorN & 1) != 0;

    AIE_LOOP_RANGE(kXPairs, kXPairs)
    for (int x_pair = 0; x_pair < kXPairs; ++x_pair) {
      const int x_tile_a = kInteriorStart + 2 * x_pair;
      const int x_out_base_a = x_tile_a * 4;
      const int x_out_base_b = x_out_base_a + 4;
      const int x_in_base_a = 2 * x_out_base_a - 1;
      const int x_in_base_b = x_in_base_a + 8;

      MMUL4x8x8 acc_a0, acc_a1, acc_b0, acc_b1;
      acc_a0 = bias_acc_a; // x_tile_a + oc_a
      acc_a1 = bias_acc_b; // x_tile_a + oc_b
      acc_b0 = bias_acc_a; // x_tile_b + oc_a
      acc_b1 = bias_acc_b; // x_tile_b + oc_b

      AIE_LOOP_RANGE(kIcTiles, kIcTiles)
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        AIE_LOOP_RANGE(1, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf_a[32];
            alignas(32) int8_t a_buf_b[32];
            gather_interior(line_ptr, x_in_base_a, kx, ic_t, a_buf_a);
            gather_interior(line_ptr, x_in_base_b, kx, ic_t, a_buf_b);
            aie::vector<int8, 32> in_a_a = aie::load_v<32>(a_buf_a);
            aie::vector<int8, 32> in_a_b = aie::load_v<32>(a_buf_b);
            int wts_off = wts_tile_off(oc_t_a, ic_t, ky, kx);
            aie::vector<int8, 64> in_b_0 = aie::load_v<64>(&wts[wts_off]);
            aie::vector<int8, 64> in_b_1 =
                aie::load_v<64>(&wts[wts_off + kOcBankStride]);
            acc_a0.mac(in_a_a, in_b_0);
            acc_a1.mac(in_a_a, in_b_1);
            acc_b0.mac(in_a_b, in_b_0);
            acc_b1.mac(in_a_b, in_b_1);
          }
        }
      }

      aie::vector<int8, 32> vec_a0 =
          acc_a0.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> vec_a1 =
          acc_a1.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> vec_b0 =
          acc_b0.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> vec_b1 =
          acc_b1.template to_vector<int8>(right_shift);
      write_x_tile_result(vec_a0, silu_lut, output, oc_full_base_a,
                          x_out_base_a);
      write_x_tile_result(vec_a1, silu_lut, output, oc_full_base_b,
                          x_out_base_a);
      write_x_tile_result(vec_b0, silu_lut, output, oc_full_base_a,
                          x_out_base_b);
      write_x_tile_result(vec_b1, silu_lut, output, oc_full_base_b,
                          x_out_base_b);
    }

    // Odd-count tail: one straggler interior x_tile (kInteriorN was odd).
    // For m1's kXTiles=32, kInteriorN=30 → even → kHasXTail=false →
    // dead-stripped.
    if constexpr (kHasXTail) {
      const int x_tile = kInteriorStart + 2 * kXPairs;
      const int x_out_base = x_tile * 4;
      const int x_in_base = 2 * x_out_base - 1;
      MMUL4x8x8 acc_a, acc_b;
      acc_a = bias_acc_a;
      acc_b = bias_acc_b;
      AIE_LOOP_RANGE(kIcTiles, kIcTiles)
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        AIE_LOOP_RANGE(1, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf[32];
            gather_interior(line_ptr, x_in_base, kx, ic_t, a_buf);
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off = wts_tile_off(oc_t_a, ic_t, ky, kx);
            acc_a.mac(in_a, aie::load_v<64>(&wts[wts_off]));
            acc_b.mac(in_a, aie::load_v<64>(&wts[wts_off + kOcBankStride]));
          }
        }
      }
      aie::vector<int8, 32> va = acc_a.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> vb = acc_b.template to_vector<int8>(right_shift);
      write_x_tile_result(va, silu_lut, output, oc_full_base_a, x_out_base);
      write_x_tile_result(vb, silu_lut, output, oc_full_base_b, x_out_base);
    }

    // ===== Right edge: x_tile=kXTiles-1 — kx=2 makes col=kInW invalid for p=3.
    // =====
    {
      const int x_tile = kXTiles - 1;
      const int x_out_base = x_tile * 4;
      const int x_in_base = 2 * x_out_base - 1;
      MMUL4x8x8 acc_a, acc_b;
      acc_a = bias_acc_a;
      acc_b = bias_acc_b;
      AIE_LOOP_RANGE(kIcTiles, kIcTiles)
      for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
        AIE_LOOP_RANGE(1, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf[32];
            bool any_valid = gather_edge(line_ptr, x_in_base, kx, ic_t, a_buf);
            if (!any_valid)
              continue;
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off = wts_tile_off(oc_t_a, ic_t, ky, kx);
            acc_a.mac(in_a, aie::load_v<64>(&wts[wts_off]));
            acc_b.mac(in_a, aie::load_v<64>(&wts[wts_off + kOcBankStride]));
          }
        }
      }
      aie::vector<int8, 32> va = acc_a.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> vb = acc_b.template to_vector<int8>(right_shift);
      write_x_tile_result(va, silu_lut, output, oc_full_base_a, x_out_base);
      write_x_tile_result(vb, silu_lut, output, oc_full_base_b, x_out_base);
    }
  }

  event1();
}

extern "C" {

void yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_i8_i8(
    int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts, int32_t *bias,
    int8_t *silu_lut, int8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int32_t kernel_width, const int32_t kernel_height,
    const int32_t border, const int32_t right_shift, const int32_t padding) {
#ifdef NOOP_KERNEL
  (void)line0;
  (void)line1;
  (void)line2;
  (void)wts;
  (void)bias;
  (void)silu_lut;
  (void)output;
  (void)input_width;
  (void)input_channels;
  (void)output_channels;
  (void)kernel_width;
  (void)kernel_height;
  (void)border;
  (void)right_shift;
  (void)padding;
  return;
#else
  yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_vec(
      line0, line1, line2, wts, bias, silu_lut, output, input_width,
      input_channels, output_channels, kernel_width, kernel_height, border,
      right_shift, padding);
#endif
}

} // extern "C"
