//===- yolo_m0_conv2dk3_silu_bias_vec.cc ---------------------------*- C++
//-*-===//
//
// Deep-opt vectorized stem kernel for m0: 3x3 stride-2 INT8 conv. Weights
// arrive host-pre-packed as [oc_tile][ky][kx][ic_inner=8][oc_inner=8] (the
// mmul.B layout), so the kernel does aie::load_v<64> at the right offset
// per (ky,kx) — no per-row pack on the hot path.
//
// Inner reduction: aie::mmul<4, 8, 8, int8, int8>. K=8 with only 3 real ic
// + 5 zero-padded ic (host-managed) means 62.5% of the K throughput is
// wasted on zeros; fixing that needs a smaller-K mmul + different pack
// (Phase 2).
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

static constexpr int kInW = YOLO_M0_IN_W;
static constexpr int kInC = YOLO_M0_IN_C;
static constexpr int kOutC = YOLO_M0_OUT_C;
static constexpr int kKW = 3;
static constexpr int kKH = 3;

static constexpr int kOutW = kInW / 2;
static constexpr int kOcTiles = kOutC / 8;
static constexpr int kXTiles = kOutW / 4;

static_assert(kInW % 2 == 0, "M0 IN_W must be even (stride 2)");
static_assert(kInC == 8, "M0 IN_C must be 8 (3 real + 5 zero-pad)");
static_assert(kOutC % 8 == 0, "M0 OUT_C must be multiple of 8");
static_assert(kOutW % 4 == 0, "M0 OUT_W must be multiple of 4");
static_assert(kOcTiles >= 2, "M0 deep-opt assumes oc_tiles >= 2 (OCx2 fold)");

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

// Weights are pre-packed host-side as [oc_tile][ky][kx][ic_inner=8][oc_inner=8],
// so the per-(ky,kx) mmul.B vector is at byte offset
// (oc_tile * 9 + ky * 3 + kx) * 64 in the input buffer.
static constexpr int kWtsBytesPerOcTile = 9 * 64; // 576

// DMA-deinterleaved layout: each row is [even_half (kInW/2 pixels),
// odd_half (kInW/2 pixels)] via memtile dims_to_stream in _build_m0.
// Stride-2 conv: kx=0/2 -> odd half; kx=1 -> even half. All 4 mmul
// pixels are 32 contiguous aligned bytes -- single aie::load_v<32>.
static constexpr int kInHalfBytes = (kInW / 2) * kInC; // 2048 for in_w=512

// Interior x_tile in [1, kXTiles-1] -- no bounds issue. AIE2P vec_load<32>
// requires 32-byte (= 4-pixel) alignment. kx=1 and kx=2 land on aligned
// pixel-quad boundaries; kx=0 lands at pixel index 4q-1 (odd_idx0 mod 4 = 3)
// so the natural load would be unaligned. For kx=0, do two aligned 32-byte
// loads of adjacent quads and shuffle_down by 24 bytes to extract the
// desired 32 bytes (= last 8 of quad q-1 + first 24 of quad q).
static __attribute__((always_inline)) inline aie::vector<int8, 32>
load_a_deinterleaved_interior(const int8_t *__restrict line_ptr, int x_tile,
                              int kx) {
  if (kx == 1) {
    return aie::load_v<32>(line_ptr + x_tile * 32);
  }
  if (kx == 2) {
    return aie::load_v<32>(line_ptr + kInHalfBytes + x_tile * 32);
  }
  // kx == 0: unaligned offset. Two aligned 32-byte loads + shuffle_down 24.
  const int aligned_base = kInHalfBytes + (x_tile - 1) * 32;
  aie::vector<int8, 32> lo = aie::load_v<32>(line_ptr + aligned_base);
  aie::vector<int8, 32> hi = aie::load_v<32>(line_ptr + aligned_base + 32);
  aie::vector<int8, 64> combined = aie::concat(lo, hi);
  return aie::shuffle_down(combined, 24).template extract<32>(0);
}

// Left edge x_tile=0: kx=0 needs odd[-1, 0, 1, 2] -- zero-prefix.
static __attribute__((always_inline)) inline aie::vector<int8, 32>
load_a_deinterleaved_left(const int8_t *__restrict line_ptr, int kx) {
  if (kx == 1) {
    return aie::load_v<32>(line_ptr);
  }
  if (kx == 2) {
    return aie::load_v<32>(line_ptr + kInHalfBytes);
  }
  // kx == 0: load odd[0..3] then shuffle to [0, odd[0..2]].
  aie::vector<int8, 32> v = aie::load_v<32>(line_ptr + kInHalfBytes);
  aie::vector<int8, 32> z = aie::zeros<int8, 32>();
  aie::vector<int8, 64> combined = aie::concat(z, v);
  return aie::shuffle_down(combined, 24).template extract<32>(0);
}

// Per-OC×2 bias accumulator helper: 8 int32 biases → 32-wide acc<acc32>
// (4 pix × 8 ch). Used to seed the mmul so to_vector<int8>(rs) does
// bias+SRS+saturate in one vec op.
static __attribute__((always_inline)) inline aie::accum<acc32, 32>
make_bias_acc(const int32_t *__restrict bias_8) {
  aie::vector<int32, 8> b8 = aie::load_v<8>(bias_8);
  aie::vector<int32, 16> b16 = aie::concat(b8, b8);
  aie::vector<int32, 32> b32 = aie::concat(b16, b16);
  aie::accum<acc32, 32> a;
  a.from_vector(b32);
  return a;
}

// Per-OC×2 epilog: interleave acc_a (pixels 0..3, chans 0..7) and acc_b
// (pixels 0..3, chans 8..15) via aie::interleave_zip with chunk=8 so the
// resulting vec<int8, 64> matches the output layout (4 pixels x 16 chans
// contiguous = exactly one 64-byte aligned x_tile block). Scalar SiLU LUT
// then ONE aie::store_v<64> instead of 64 scalar stores.
static __attribute__((always_inline)) inline void
emit_16_lanes(aie::vector<int8, 32> srs_a, aie::vector<int8, 32> srs_b,
              const int8_t *__restrict silu_lut,
              int8_t *__restrict output, int x_out_base) {
  auto [lo, hi] = aie::interleave_zip(srs_a, srs_b, 8u);
  aie::vector<int8, 64> combined = aie::concat(lo, hi);
  alignas(64) int8_t silu_buf[64];
  // Explicit 4-way batch: 4 independent loads issued before any store, so
  // peano allocates distinct dest registers and pipelines load/store across
  // the 4 lanes (vs the dependent-chain pattern of the single-element loop).
  AIE_LOOP_UNROLL_FULL
  for (int i = 0; i < 64; i += 4) {
    int idx0 = int(combined[i + 0]) + 128;
    int idx1 = int(combined[i + 1]) + 128;
    int idx2 = int(combined[i + 2]) + 128;
    int idx3 = int(combined[i + 3]) + 128;
    int8_t v0 = silu_lut[idx0];
    int8_t v1 = silu_lut[idx1];
    int8_t v2 = silu_lut[idx2];
    int8_t v3 = silu_lut[idx3];
    silu_buf[i + 0] = v0;
    silu_buf[i + 1] = v1;
    silu_buf[i + 2] = v2;
    silu_buf[i + 3] = v3;
  }
  aie::vector<int8, 64> silu_v = aie::load_v<64>(silu_buf);
  aie::store_v(output + x_out_base * kOutC, silu_v);
}

static void yolo_m0_conv2dk3_i8_stride2_silu_bias_vec(
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
    const int8_t *__restrict wts_a = wts + (oc_pair * 2 + 0) * kWtsBytesPerOcTile;
    const int8_t *__restrict wts_b = wts + (oc_pair * 2 + 1) * kWtsBytesPerOcTile;
    const int oc_full_base_a = (oc_pair * 2 + 0) * 8;
    const int oc_full_base_b = (oc_pair * 2 + 1) * 8;

    // Bias seeds for the two oc-tiles in this pair; reused across all
    // x_tile iters of this oc_pair.
    auto bias_acc_a = make_bias_acc(&bias[oc_full_base_a]);
    auto bias_acc_b = make_bias_acc(&bias[oc_full_base_b]);

    // ----- Left edge: x_tile=0 (kx=0 needs zero-prefix for odd[-1]).
    {
      MMUL4x8x8 acc_a, acc_b;
      acc_a = bias_acc_a;
      acc_b = bias_acc_b;

      AIE_LOOP_RANGE(1, 3)
      for (int ky = ky_start; ky < ky_end; ++ky) {
        const int8_t *__restrict line_ptr = line[ky];
        AIE_LOOP_UNROLL_FULL
        for (int kx = 0; kx < kKW; ++kx) {
          aie::vector<int8, 32> in_a =
              load_a_deinterleaved_left(line_ptr, kx);
          int wt_off = (ky * kKW + kx) * 64;
          acc_a.mac(in_a, aie::load_v<64>(&wts_a[wt_off]));
          acc_b.mac(in_a, aie::load_v<64>(&wts_b[wt_off]));
        }
      }

      aie::vector<int8, 32> srs_a = acc_a.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> srs_b = acc_b.template to_vector<int8>(right_shift);
      emit_16_lanes(srs_a, srs_b, silu_lut, output, 0);
    }

    // ----- Interior: x_tile ∈ [1, kXTiles). Pure vec_load per (q, kx).
    AIE_LOOP_RANGE(kXTiles - 1, kXTiles - 1)
    for (int x_tile = 1; x_tile < kXTiles; ++x_tile) {
      MMUL4x8x8 acc_a, acc_b;
      acc_a = bias_acc_a;
      acc_b = bias_acc_b;

      AIE_LOOP_RANGE(1, 3)
      for (int ky = ky_start; ky < ky_end; ++ky) {
        const int8_t *__restrict line_ptr = line[ky];
        AIE_LOOP_UNROLL_FULL
        for (int kx = 0; kx < kKW; ++kx) {
          aie::vector<int8, 32> in_a =
              load_a_deinterleaved_interior(line_ptr, x_tile, kx);
          int wt_off = (ky * kKW + kx) * 64;
          acc_a.mac(in_a, aie::load_v<64>(&wts_a[wt_off]));
          acc_b.mac(in_a, aie::load_v<64>(&wts_b[wt_off]));
        }
      }

      aie::vector<int8, 32> srs_a = acc_a.template to_vector<int8>(right_shift);
      aie::vector<int8, 32> srs_b = acc_b.template to_vector<int8>(right_shift);
      emit_16_lanes(srs_a, srs_b, silu_lut, output, x_tile * 4);
    }
  }

  event1();
}

extern "C" {

void yolo_m0_conv2dk3_stride2_silu_bias_i8_i8(
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
  yolo_m0_conv2dk3_i8_stride2_silu_bias_vec(
      line0, line1, line2, wts, bias, silu_lut, output, input_width,
      input_channels, output_channels, kernel_width, kernel_height, border,
      right_shift, padding);
#endif
}

} // extern "C"
