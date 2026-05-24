//===- yolo_m0_conv2dk3_silu_bias_vec.cc ---------------------------*- C++ -*-===//
//
// Deep-opt vectorized stem kernel for m0: 3x3 stride-2 INT8 conv, raw OIYX
// weight layout (in_c=3 padded to 8 — not OIYXI8O8 since in_c isn't
// 8-aligned). Drop-in .o-level replacement.
//
// Inner reduction: aie::mmul<4, 8, 8, int8, int8>. The B-operand is
// pre-packed once per (oc_tile) into a 9×64-byte buffer covering all
// 3×3 (ky, kx) tiles. K=8 with only 3 real ic + 5 zero-padded ic
// (host-managed) means 62.5% of the K throughput is wasted on zeros;
// fixing that needs a smaller-K mmul + host weight re-pack (Phase 2).
//
// Deep-opt levers in this pass:
//   - Compile-time shape #defines: in_w=512, in_c=8, out_c=16, K=3.
//     Folds all addressing into shifts/immediates, dead-strips the
//     scalar tail (out_w=256 always divides 4).
//   - 2-way x_tile split: x_tile=0 is the only left-edge case
//     (x_in_col -1 for kx=0); x_tile in [1..63] is interior, branch-free.
//   - OC×2 fold: both oc tiles in lockstep per (x_tile, ky, kx) — the
//     A-gather is shared across both, B-loads alternate. Halves the
//     gather work.
//   - AIE_LOOP_RANGE hints with exact trip counts.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"
#include "yolo_m0_conv2dk3_silu_bias.h"

#ifndef YOLO_M0_IN_W
#error "YOLO_M0_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M0_IN_C
#error "YOLO_M0_IN_C must be defined at compile time"
#endif
#ifndef YOLO_M0_OUT_C
#error "YOLO_M0_OUT_C must be defined at compile time"
#endif

static constexpr int kInW   = YOLO_M0_IN_W;
static constexpr int kInC   = YOLO_M0_IN_C;
static constexpr int kOutC  = YOLO_M0_OUT_C;
static constexpr int kKW    = 3;
static constexpr int kKH    = 3;

static constexpr int kOutW    = kInW / 2;
static constexpr int kOcTiles = kOutC / 8;
static constexpr int kXTiles  = kOutW / 4;

static_assert(kInW % 2 == 0,  "M0 IN_W must be even (stride 2)");
static_assert(kInC == 8,      "M0 IN_C must be 8 (3 real + 5 zero-pad)");
static_assert(kOutC % 8 == 0, "M0 OUT_C must be multiple of 8");
static_assert(kOutW % 4 == 0, "M0 OUT_W must be multiple of 4");
static_assert(kOcTiles >= 2,  "M0 deep-opt assumes oc_tiles >= 2 (OCx2 fold)");

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

static inline int wts_idx_oiyx(int oc_full, int ic_full, int ky, int kx) {
  // OIYX raw: byte[oc][ic][ky][kx]. in_c=8 padded.
  return ((oc_full * kInC + ic_full) * kKH + ky) * kKW + kx;
}

using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

// Pack the 9 (ky, kx) weight tiles for one oc_tile into a contiguous
// 9 × 64-byte buffer in row-major [ic_inner=8][oc_inner=8] order so a
// single aie::load_v<64> per (ky, kx) yields the mmul.B vector.
static __attribute__((always_inline)) inline void pack_wts(
    const int8_t *__restrict wts, int oc_tile_base,
    int8_t *__restrict wbuf) {
  // No UNROLL_FULL — peano blows up compile time on 9×8×8=576 unrolled
  // iters and the pack runs once per (oc_tile) per call, dwarfed by
  // the inner mmul work anyway.
  for (int ky = 0; ky < kKH; ++ky) {
    for (int kx = 0; kx < kKW; ++kx) {
      int wt_off = (ky * kKW + kx) * 64;
      for (int ii = 0; ii < 8; ++ii) {
        for (int oo = 0; oo < 8; ++oo) {
          wbuf[wt_off + ii * 8 + oo] =
              wts[wts_idx_oiyx(oc_tile_base + oo, ii, ky, kx)];
        }
      }
    }
  }
}

// Gather 4 contiguous x_outs × 8 ic_inner from one input line into a
// 32-byte aligned a_buf. Caller is responsible for `col` being in range
// (interior path) — the edge path zeros out-of-range bytes.
static __attribute__((always_inline)) inline void gather_interior(
    const int8_t *__restrict line_ptr, int x_in_base, int kx,
    int8_t *__restrict a_buf) {
  AIE_LOOP_UNROLL_FULL
  for (int p = 0; p < 4; ++p) {
    const int col = x_in_base + 2 * p + kx;
    const int8_t *__restrict src = line_ptr + col * kInC;
    // 8-byte block copy; peano lowers to a single 64-bit load/store pair.
    for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
  }
}

static __attribute__((always_inline)) inline void gather_edge(
    const int8_t *__restrict line_ptr, int x_in_base, int kx,
    int8_t *__restrict a_buf,
    bool &any_valid) {
  any_valid = false;
  AIE_LOOP_UNROLL_FULL
  for (int p = 0; p < 4; ++p) {
    const int col = x_in_base + 2 * p + kx;
    if (col < 0 || col >= kInW) {
      for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = 0;
    } else {
      const int8_t *__restrict src = line_ptr + col * kInC;
      for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
      any_valid = true;
    }
  }
}

// Per-OC×2 bias accumulator helper: 8 int32 biases → 32-wide acc<acc32>
// (4 pix × 8 ch). Used to seed the mmul so to_vector<int8>(rs) does
// bias+SRS+saturate in one vec op.
static __attribute__((always_inline)) inline aie::accum<acc32, 32>
make_bias_acc(const int32_t *__restrict bias_8) {
  aie::vector<int32, 8>  b8  = aie::load_v<8>(bias_8);
  aie::vector<int32, 16> b16 = aie::concat(b8, b8);
  aie::vector<int32, 32> b32 = aie::concat(b16, b16);
  aie::accum<acc32, 32> a;
  a.from_vector(b32);
  return a;
}

// Per-OC×2 epilog: SiLU LUT for 8 lanes (SRS+clamp+bias already collapsed
// into the vec to_vector<int8>(rs) call at the call site).
static __attribute__((always_inline)) inline void emit_8_lanes(
    aie::vector<int8, 32> srs_v, const int8_t *__restrict silu_lut,
    int8_t *__restrict output, int oc_full_base, int x_out_base) {
  AIE_LOOP_UNROLL_FULL
  for (int p = 0; p < 4; ++p) {
    int x_out = x_out_base + p;
    int8_t *__restrict row_dst = output + x_out * kOutC + oc_full_base;
    AIE_LOOP_UNROLL_FULL
    for (int j = 0; j < 8; ++j) {
      row_dst[j] = silu_lut[int(srs_v[p * 8 + j]) + 128];
    }
  }
}

static void
yolo_m0_conv2dk3_i8_stride2_silu_bias_vec(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts, int32_t *bias, int8_t *silu_lut, int8_t *output,
    const int32_t /*input_width*/,
    const int32_t /*input_channels*/,
    const int32_t /*output_channels*/,
    const int32_t /*kernel_width*/,
    const int32_t /*kernel_height*/,
    const int32_t border,
    const int32_t right_shift,
    const int32_t /*padding*/) {
  event0();

  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);
  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs (round-half-to-even). Required
  // for to_vector<int8>(rs) to produce the same bit pattern as the
  // scalar reference path.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  const int8_t *__restrict line[3] = {line0, line1, line2};

  // Per (oc_tile_pair) — both oc_tiles in lockstep (OC×2 fold). For
  // m0 oc_tiles=2 so the outer loop is 1 iter; structuring it this way
  // keeps the body amenable to oc_tiles=4 (out_c=32) callers in the future.
  AIE_LOOP_RANGE(kOcTiles / 2, kOcTiles / 2)
  for (int oc_pair = 0; oc_pair < kOcTiles / 2; ++oc_pair) {
    alignas(32) int8_t wbuf_a[9 * 64];
    alignas(32) int8_t wbuf_b[9 * 64];
    pack_wts(wts, (oc_pair * 2 + 0) * 8, wbuf_a);
    pack_wts(wts, (oc_pair * 2 + 1) * 8, wbuf_b);
    const int oc_full_base_a = (oc_pair * 2 + 0) * 8;
    const int oc_full_base_b = (oc_pair * 2 + 1) * 8;

    // Bias seeds for the two oc-tiles in this pair; reused across all
    // x_tile iters of this oc_pair.
    auto bias_acc_a = make_bias_acc(&bias[oc_full_base_a]);
    auto bias_acc_b = make_bias_acc(&bias[oc_full_base_b]);

    // ----- Left edge: x_tile=0, x_in cols ∈ {-1, 1, 3, 5} + kx. kx=0
    // gives col=-1 → zeroed; kx in {1,2} gives all valid. Only this
    // tile needs the bounds check.
    {
      MMUL4x8x8 acc_a, acc_b;
      acc_a = bias_acc_a;
      acc_b = bias_acc_b;

      const int x_out_base = 0;
      const int x_in_base = -1;

      AIE_LOOP_RANGE(1, 3)
      for (int ky = ky_start; ky < ky_end; ++ky) {
        const int8_t *__restrict line_ptr = line[ky];
        AIE_LOOP_RANGE(3, 3)
        for (int kx = 0; kx < kKW; ++kx) {
          alignas(32) int8_t a_buf[32];
          bool any_valid;
          gather_edge(line_ptr, x_in_base, kx, a_buf, any_valid);
          if (!any_valid) continue;
          aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
          int wt_off = (ky * kKW + kx) * 64;
          acc_a.mac(in_a, aie::load_v<64>(&wbuf_a[wt_off]));
          acc_b.mac(in_a, aie::load_v<64>(&wbuf_b[wt_off]));
        }
      }

      aie::vector<int8, 32> srs_a = acc_a.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> srs_b = acc_b.template to_vector<int8>(right_shift);
      emit_8_lanes(srs_a, silu_lut, output, oc_full_base_a, x_out_base);
      emit_8_lanes(srs_b, silu_lut, output, oc_full_base_b, x_out_base);
    }

    // ----- Interior: x_tile ∈ [1, kXTiles). Branch-free gather. ----
    AIE_LOOP_RANGE(kXTiles - 1, kXTiles - 1)
    for (int x_tile = 1; x_tile < kXTiles; ++x_tile) {
      MMUL4x8x8 acc_a, acc_b;
      acc_a = bias_acc_a;
      acc_b = bias_acc_b;

      const int x_out_base = x_tile * 4;
      const int x_in_base = 2 * x_out_base - 1;

      AIE_LOOP_RANGE(1, 3)
      for (int ky = ky_start; ky < ky_end; ++ky) {
        const int8_t *__restrict line_ptr = line[ky];
        AIE_LOOP_RANGE(3, 3)
        for (int kx = 0; kx < kKW; ++kx) {
          alignas(32) int8_t a_buf[32];
          gather_interior(line_ptr, x_in_base, kx, a_buf);
          aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
          int wt_off = (ky * kKW + kx) * 64;
          acc_a.mac(in_a, aie::load_v<64>(&wbuf_a[wt_off]));
          acc_b.mac(in_a, aie::load_v<64>(&wbuf_b[wt_off]));
        }
      }

      aie::vector<int8, 32> srs_a = acc_a.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> srs_b = acc_b.template to_vector<int8>(right_shift);
      emit_8_lanes(srs_a, silu_lut, output, oc_full_base_a, x_out_base);
      emit_8_lanes(srs_b, silu_lut, output, oc_full_base_b, x_out_base);
    }
  }

  event1();
}

extern "C" {

void yolo_m0_conv2dk3_stride2_silu_bias_i8_i8(
    int8_t *line0, int8_t *line1, int8_t *line2,
    int8_t *wts, int32_t *bias, int8_t *silu_lut, int8_t *output,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t output_channels,
    const int32_t kernel_width,
    const int32_t kernel_height,
    const int32_t border,
    const int32_t right_shift,
    const int32_t padding) {
#ifdef NOOP_KERNEL
  (void)line0; (void)line1; (void)line2; (void)wts; (void)bias; (void)silu_lut;
  (void)output; (void)input_width; (void)input_channels; (void)output_channels;
  (void)kernel_width; (void)kernel_height; (void)border; (void)right_shift; (void)padding;
  return;
#else
  yolo_m0_conv2dk3_i8_stride2_silu_bias_vec(
      line0, line1, line2, wts, bias, silu_lut, output,
      input_width, input_channels, output_channels,
      kernel_width, kernel_height, border, right_shift, padding);
#endif
}

} // extern "C"
