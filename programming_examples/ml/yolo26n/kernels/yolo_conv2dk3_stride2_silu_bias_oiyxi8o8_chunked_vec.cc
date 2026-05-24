//===- yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_vec.cc -----*- C++
//-*-===//
//
// Vectorized chunked variant of the OIYXI8O8 stride-2 conv. Same math as
// the non-chunked vec kernel (yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_vec.cc),
// but the weight buffer holds only one chunk of output channels and the
// kernel writes to the global output row at oc_offset + chunk_oc.
//
// One .cc file produces three .o (m3, m5, m7) via -DKERNEL_SUFFIX=_mN.
// Drop-in .o-level replacement for the scalar chunked .o.
//
// Inner reduction uses aie::mmul<4, 8, 8, int8, int8>; bias folded into
// the scalar SRS+LUT epilogue.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

// Per-block symbol mangling. Compile per-block with -DKERNEL_SUFFIX=_mN.
#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

// Compile-time shape specialization. Caller-side Makefile MUST pass:
//   -DYOLO_IN_W=<input spatial width>
//   -DYOLO_IN_C=<input channel count, multiple of 8>
//   -DYOLO_OUT_C=<output channel count, multiple of 8>
//   -DYOLO_INTERIOR_MODE=<1 for OC×2 single-X, 2 for 2X×2OC>
//   -DYOLO_OC_PER_CHUNK=<oc_count per call, default 16>  (optional)
// This lets peano (a) constant-fold all addressing arithmetic into shifts
// instead of runtime muls, (b) dead-strip the chunk_oc_tiles!=2 fallback and
// the INTERIOR_MODE-not-selected interior body, (c) emit shape-specialized
// pipelined inner loops. The runtime args (input_width, input_channels,
// output_channels, oc_count) are kept in the C signature for ABI stability
// but the kernel body only reads the constexprs below.
#ifndef YOLO_IN_W
#error "YOLO_IN_W must be defined at compile time"
#endif
#ifndef YOLO_IN_C
#error "YOLO_IN_C must be defined at compile time"
#endif
#ifndef YOLO_OUT_C
#error "YOLO_OUT_C must be defined at compile time"
#endif
#ifndef YOLO_INTERIOR_MODE
#error "YOLO_INTERIOR_MODE must be defined (1=OC×2 single-X, 2=2X×2OC)"
#endif
#ifndef YOLO_OC_PER_CHUNK
#define YOLO_OC_PER_CHUNK 16
#endif

static constexpr int kInputWidth = YOLO_IN_W;
static constexpr int kInputChannels = YOLO_IN_C;
static constexpr int kOutputChannels = YOLO_OUT_C;
static constexpr int kKernelW = 3;
static constexpr int kKernelH = 3;
static constexpr int kOcPerChunk = YOLO_OC_PER_CHUNK;
static constexpr int kInteriorMode = YOLO_INTERIOR_MODE;

static constexpr int kIcTiles = kInputChannels / 8;
static constexpr int kOutputWidth = kInputWidth / 2;
static constexpr int kXTiles = kOutputWidth / 4;
static constexpr int kChunkOcTiles = kOcPerChunk / 8;
static constexpr bool kOcPair = (kChunkOcTiles == 2);

static_assert(kInputChannels % 8 == 0, "YOLO_IN_C must be a multiple of 8");
static_assert(kOcPerChunk % 8 == 0,
              "YOLO_OC_PER_CHUNK must be a multiple of 8");
static_assert(
    kInputWidth >= 8 && (kInputWidth % 8 == 0),
    "YOLO_IN_W must be a multiple of 8 (output_width/4 = x_tiles ≥ 1)");
static_assert(kInteriorMode == 1 || kInteriorMode == 2,
              "YOLO_INTERIOR_MODE must be 1 (OC×2 single-X) or 2 (2X×2OC)");
// Both interior modes rely on the OC×2 fold (two weight banks per call) for
// their epilogue + addressing. The current per-block builds (m3/m5/m7) all
// pass oc_per_chunk=16 → chunk_oc_tiles=2. Bump YOLO_OC_PER_CHUNK and add a
// non-OC-paired variant if a future caller ever needs a different cadence.
static_assert(
    kOcPair,
    "kOcPair (chunk_oc_tiles == 2) is required; pass YOLO_OC_PER_CHUNK=16");

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

// Chunk-local OIYXI8O8 weight base offset for (chunk_oc_tile, ic_tile, ky, kx).
// chunk_oc_tile = (chunk-local oc_full) / 8, in [0..oc_count/8).
static inline int wts_chunk_tile_off(int chunk_oc_tile, int ic_tile, int ky,
                                     int kx, int ic_tiles, int kH, int kW) {
  return (((chunk_oc_tile * ic_tiles + ic_tile) * kH + ky) * kW + kx) << 6;
}

// Reference scalar weight index (tail path).
static inline int wts_chunk_idx_oiyxi8o8(int chunk_oc_full, int ic_full, int ky,
                                         int kx, int in_c, int kH, int kW) {
  int oc_t = chunk_oc_full >> 3;
  int oc_i = chunk_oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) + ic_i * 8 +
         oc_i;
}

static void yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_chunked_vec(
    int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts_chunk,
    int32_t *bias, int8_t *silu_lut, int8_t *output,
    const int32_t /*input_width*/, // shape-defined; arg kept for ABI
    const int32_t /*input_channels*/, const int32_t /*output_channels*/,
    const int32_t /*kernel_width*/, const int32_t /*kernel_height*/,
    const int32_t border, const int32_t right_shift, const int32_t oc_offset,
    const int32_t /*oc_count*/) {
  event0();

  // Shape constants live at file scope (kInputWidth, kInputChannels, ...).
  // Only `border`, `right_shift`, `oc_offset` remain runtime — they vary
  // per-call within a sample.
  constexpr int32_t input_width = kInputWidth;
  constexpr int input_channels = kInputChannels;
  constexpr int output_channels = kOutputChannels;
  constexpr int kernel_width = kKernelW;
  constexpr int kernel_height = kKernelH;
  constexpr int ic_tiles = kIcTiles;
  constexpr int chunk_oc_tiles = kChunkOcTiles;
  constexpr int output_width = kOutputWidth;
  constexpr int x_tiles = kXTiles;

  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);

  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs (round-half-to-even); enables
  // vec to_vector<int8>(rs) for SRS+saturate without LSB drift.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  // 8 int32 biases -> 32-wide acc<acc32> (4 pix × 8 ch). Seeds the mmul
  // so to_vector<int8>(rs) emits bias+SRS+saturate in one vec op.
  auto make_bias_acc = [&](const int32_t *bias_8) {
    aie::vector<int32, 8> b8 = aie::load_v<8>(bias_8);
    aie::vector<int32, 16> b16 = aie::concat(b8, b8);
    aie::vector<int32, 32> b32 = aie::concat(b16, b16);
    aie::accum<acc32, 32> a;
    a.from_vector(b32);
    return a;
  };

  int8_t *line[3] = {line0, line1, line2};

  // Interior x_tiles (1..x_tiles-2) need no col bounds check:
  //   kx=0 col = 2*x_out_base-1 ≥ 1 when x_tile ≥ 1; kx=2 col ≤ input_width-1
  //   when x_tile ≤ x_tiles-2. So we split into 3 loop nests — left edge,
  //   interior (straight-line vector ops, no branches), right edge — to keep
  //   the hot interior body free of the per-p bounds check that would block
  //   peano's loop pipelining.
  //
  // Interior body shape selected at compile time via YOLO_INTERIOR_MODE:
  //   2 = 2X×2OC (example.h-style, 4 accs)  — best for x_tiles ≥ 8
  //   1 = OC×2 single-X (2 accs)            — best for x_tiles == 4 (m7)
  // When kOcPair is false (oc_per_chunk != 16, not used today) we fall
  // back to a per-OC single-acc interior — that path is dead-stripped in
  // the m3/m5/m7 builds.

  // Vec write helper: acc is bias-seeded so to_vector<int8>(rs) is the
  // bias-added, SRS'd, saturated i8 result. Only the SiLU LUT gather stays
  // scalar (32 lookups per call, no vec int8-LUT primitive on AIE2P).
  auto write_x_tile_result = [&](MMUL4x8x8 &acc, int x_out_base,
                                 int oc_full_base) {
    aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
    for (int p = 0; p < 4; ++p) {
      int x_out = x_out_base + p;
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_full_base + j;
        output[x_out * output_channels + oc_full] =
            silu_lut[int(srs_v[p * 8 + j]) + 128];
      }
    }
  };

  // -------------------------------------------------------------------------
  // Edges + scalar tail: loop per OC bank. Small share of total compute,
  // not worth fusing.
  // -------------------------------------------------------------------------
  AIE_LOOP_RANGE(1, 2)
  for (int chunk_oc_t = 0; chunk_oc_t < chunk_oc_tiles; ++chunk_oc_t) {
    const int oc_full_base = oc_offset + chunk_oc_t * 8;
    auto edge_bias_acc = make_bias_acc(&bias[oc_full_base]);

    // Left edge (x_tile = 0) — kx=0 makes col=-1 invalid for p=0.
    {
      MMUL4x8x8 acc;
      acc = edge_bias_acc;
      const int x_in_base = -1;
      AIE_LOOP_RANGE(8, 16)
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        AIE_LOOP_RANGE(2, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf[32];
            bool any_valid = false;
            for (int p = 0; p < 4; ++p) {
              int col = x_in_base + 2 * p + kx;
              if (col < 0 || col >= input_width) {
                *(reinterpret_cast<uint64_t *>(&a_buf[p * 8])) = 0;
              } else {
                int8_t *src = line_ptr + col * input_channels + ic_t * 8;
                *(reinterpret_cast<uint64_t *>(&a_buf[p * 8])) =
                    *reinterpret_cast<const uint64_t *>(src);
                any_valid = true;
              }
            }
            if (!any_valid)
              continue;
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off = wts_chunk_tile_off(chunk_oc_t, ic_t, ky, kx, ic_tiles,
                                             kernel_height, kernel_width);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }
      write_x_tile_result(acc, 0, oc_full_base);
    }

    // Right edge (x_tile = x_tiles - 1) — kx=2 makes col=input_width invalid.
    if (x_tiles >= 2) {
      const int x_tile = x_tiles - 1;
      MMUL4x8x8 acc;
      acc = edge_bias_acc;
      const int x_out_base = x_tile * 4;
      const int x_in_base = 2 * x_out_base - 1;
      AIE_LOOP_RANGE(8, 16)
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        AIE_LOOP_RANGE(2, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf[32];
            bool any_valid = false;
            for (int p = 0; p < 4; ++p) {
              int col = x_in_base + 2 * p + kx;
              if (col < 0 || col >= input_width) {
                *(reinterpret_cast<uint64_t *>(&a_buf[p * 8])) = 0;
              } else {
                int8_t *src = line_ptr + col * input_channels + ic_t * 8;
                *(reinterpret_cast<uint64_t *>(&a_buf[p * 8])) =
                    *reinterpret_cast<const uint64_t *>(src);
                any_valid = true;
              }
            }
            if (!any_valid)
              continue;
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off = wts_chunk_tile_off(chunk_oc_t, ic_t, ky, kx, ic_tiles,
                                             kernel_height, kernel_width);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }
      write_x_tile_result(acc, x_out_base, oc_full_base);
    }

    // Scalar tail (output_width not a multiple of 4). Cheap, kept per-OC.
    for (int x = x_tiles * 4; x < output_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int chunk_oc_full = chunk_oc_t * 8 + j;
        int oc_full = oc_offset + chunk_oc_full;
        int32_t sum = bias[oc_full];
        for (int ic_full = 0; ic_full < input_channels; ++ic_full) {
          for (int ki = 0; ki < kernel_width; ++ki) {
            int col = 2 * x - 1 + ki;
            if (col < 0 || col >= input_width)
              continue;
            int in_indx = col * input_channels + ic_full;
            int w0 = wts_chunk[wts_chunk_idx_oiyxi8o8(
                chunk_oc_full, ic_full, 0, ki, input_channels, kernel_height,
                kernel_width)];
            int w1 = wts_chunk[wts_chunk_idx_oiyxi8o8(
                chunk_oc_full, ic_full, 1, ki, input_channels, kernel_height,
                kernel_width)];
            int w2 = wts_chunk[wts_chunk_idx_oiyxi8o8(
                chunk_oc_full, ic_full, 2, ki, input_channels, kernel_height,
                kernel_width)];
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

  // -------------------------------------------------------------------------
  // Interior: x_tile in [1, x_tiles-2]. Hot path. The chunk_oc_t loop (always
  // trips 2 in the m3/m5/m7 calling convention, oc_per_chunk=16) is fused
  // into the inner mmul body so two weight banks share one A-operand gather.
  // Two YOLO_INTERIOR_MODE variants, compile-time selected per block:
  //   MODE=2  2X×2OC, 4 accs — for x_tiles ≥ 8 (m3, m5). example.h pattern:
  //           gather two adjacent x_tiles' A operands, load both OC weight
  //           banks, do 4 macs (2 X × 2 OC) per (ic,ky,kx). Best amortization.
  //   MODE=1  OC×2 single-X, 2 accs — for x_tiles = 4 (m7). No X-pairing;
  //           pair-setup overhead would overwhelm the saving on so few
  //           interior tiles.
  // The non-selected variant is dead-stripped by `if constexpr`.
  // -------------------------------------------------------------------------
  constexpr int oc_full_base_0_offset = 0;
  constexpr int oc_full_base_1_offset = 8;
  const int oc_full_base_0 = oc_offset + oc_full_base_0_offset;
  const int oc_full_base_1 = oc_offset + oc_full_base_1_offset;
  // Bias seeds for the two OC banks; reused across all interior x_tiles.
  auto bias_acc_oc0 = make_bias_acc(&bias[oc_full_base_0]);
  auto bias_acc_oc1 = make_bias_acc(&bias[oc_full_base_1]);
  // Per-(ic,ky,kx) offset between the two weight banks: stepping chunk_oc_t
  // by 1 adds (ic_tiles * kH * kW * 64) bytes to wts_chunk_tile_off. With
  // ic_tiles, kH, kW all compile-time constants this folds to a literal.
  constexpr int wts_oc_bank_stride = (kIcTiles * kKernelH * kKernelW) << 6;

  if constexpr (kInteriorMode == 2) {
    // ----- 2X×2OC interior (m3, m5) -----
    constexpr int n_interior = kXTiles - 2; // x_tile in [1, x_tiles-2]
    constexpr int n_x_pairs = n_interior >> 1;
    constexpr bool has_x_tail = (n_interior & 1) != 0;

    // 2X×2OC pair loop. m3 → 7 pairs, m5 → 3 pairs.
    AIE_LOOP_RANGE(1, 7)
    for (int x_pair = 0; x_pair < n_x_pairs; ++x_pair) {
      const int x_tile_a = 1 + 2 * x_pair;
      const int x_out_base_a = x_tile_a * 4;
      const int x_out_base_b = x_out_base_a + 4;
      const int x_in_base_a = 2 * x_out_base_a - 1; // = 1 when x_pair=0
      // x_tile_b's input base is +8 cols (stride-2 over 4 output cols).

      MMUL4x8x8 acc_a0, acc_a1, acc_b0, acc_b1;
      acc_a0 = bias_acc_oc0; // x_a + oc0
      acc_a1 = bias_acc_oc1; // x_a + oc1
      acc_b0 = bias_acc_oc0; // x_b + oc0
      acc_b1 = bias_acc_oc1; // x_b + oc1

      AIE_LOOP_RANGE(8, 16)
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        AIE_LOOP_RANGE(2, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];

          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            // Gather x_tile_a inputs: 4 cols × 8 ic-bytes, 64-bit word copies.
            alignas(32) int8_t a_buf_a[32];
            int8_t *sa0 =
                line_ptr + (x_in_base_a + kx) * input_channels + ic_t * 8;
            int8_t *sa1 = sa0 + 2 * input_channels;
            int8_t *sa2 = sa0 + 4 * input_channels;
            int8_t *sa3 = sa0 + 6 * input_channels;
            *(reinterpret_cast<uint64_t *>(&a_buf_a[0])) =
                *reinterpret_cast<const uint64_t *>(sa0);
            *(reinterpret_cast<uint64_t *>(&a_buf_a[8])) =
                *reinterpret_cast<const uint64_t *>(sa1);
            *(reinterpret_cast<uint64_t *>(&a_buf_a[16])) =
                *reinterpret_cast<const uint64_t *>(sa2);
            *(reinterpret_cast<uint64_t *>(&a_buf_a[24])) =
                *reinterpret_cast<const uint64_t *>(sa3);

            // Gather x_tile_b inputs: shifted +8 input cols from x_tile_a.
            alignas(32) int8_t a_buf_b[32];
            int8_t *sb0 = sa0 + 8 * input_channels;
            int8_t *sb1 = sb0 + 2 * input_channels;
            int8_t *sb2 = sb0 + 4 * input_channels;
            int8_t *sb3 = sb0 + 6 * input_channels;
            *(reinterpret_cast<uint64_t *>(&a_buf_b[0])) =
                *reinterpret_cast<const uint64_t *>(sb0);
            *(reinterpret_cast<uint64_t *>(&a_buf_b[8])) =
                *reinterpret_cast<const uint64_t *>(sb1);
            *(reinterpret_cast<uint64_t *>(&a_buf_b[16])) =
                *reinterpret_cast<const uint64_t *>(sb2);
            *(reinterpret_cast<uint64_t *>(&a_buf_b[24])) =
                *reinterpret_cast<const uint64_t *>(sb3);

            aie::vector<int8, 32> in_a_a = aie::load_v<32>(a_buf_a);
            aie::vector<int8, 32> in_a_b = aie::load_v<32>(a_buf_b);

            // Two weight banks, each shared across both X positions.
            int wts_off_0 = wts_chunk_tile_off(0, ic_t, ky, kx, ic_tiles,
                                               kernel_height, kernel_width);
            aie::vector<int8, 64> in_b_0 =
                aie::load_v<64>(&wts_chunk[wts_off_0]);
            aie::vector<int8, 64> in_b_1 =
                aie::load_v<64>(&wts_chunk[wts_off_0 + wts_oc_bank_stride]);

            // 4 macs: 2 X × 2 OC. One gather per X, one weight per OC.
            acc_a0.mac(in_a_a, in_b_0);
            acc_a1.mac(in_a_a, in_b_1);
            acc_b0.mac(in_a_b, in_b_0);
            acc_b1.mac(in_a_b, in_b_1);
          }
        }
      }
      write_x_tile_result(acc_a0, x_out_base_a, oc_full_base_0);
      write_x_tile_result(acc_a1, x_out_base_a, oc_full_base_1);
      write_x_tile_result(acc_b0, x_out_base_b, oc_full_base_0);
      write_x_tile_result(acc_b1, x_out_base_b, oc_full_base_1);
    }

    // Odd-x_tile tail (dead-stripped when n_interior is even, which it is
    // for all current chunked-conv shapes). Single-X OC×2 form, 2 accs.
    if constexpr (has_x_tail) {
      constexpr int x_tile = 1 + 2 * n_x_pairs;
      constexpr int x_out_base = x_tile * 4;
      constexpr int x_in_base = 2 * x_out_base - 1;

      MMUL4x8x8 acc_0;
      MMUL4x8x8 acc_1;
      acc_0 = bias_acc_oc0;
      acc_1 = bias_acc_oc1;

      AIE_LOOP_RANGE(8, 16)
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        AIE_LOOP_RANGE(2, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf[32];
            int8_t *src0 =
                line_ptr + (x_in_base + kx) * input_channels + ic_t * 8;
            int8_t *src1 = src0 + 2 * input_channels;
            int8_t *src2 = src0 + 4 * input_channels;
            int8_t *src3 = src0 + 6 * input_channels;
            *(reinterpret_cast<uint64_t *>(&a_buf[0])) =
                *reinterpret_cast<const uint64_t *>(src0);
            *(reinterpret_cast<uint64_t *>(&a_buf[8])) =
                *reinterpret_cast<const uint64_t *>(src1);
            *(reinterpret_cast<uint64_t *>(&a_buf[16])) =
                *reinterpret_cast<const uint64_t *>(src2);
            *(reinterpret_cast<uint64_t *>(&a_buf[24])) =
                *reinterpret_cast<const uint64_t *>(src3);
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off_0 = wts_chunk_tile_off(0, ic_t, ky, kx, ic_tiles,
                                               kernel_height, kernel_width);
            aie::vector<int8, 64> in_b_0 =
                aie::load_v<64>(&wts_chunk[wts_off_0]);
            aie::vector<int8, 64> in_b_1 =
                aie::load_v<64>(&wts_chunk[wts_off_0 + wts_oc_bank_stride]);
            acc_0.mac(in_a, in_b_0);
            acc_1.mac(in_a, in_b_1);
          }
        }
      }
      write_x_tile_result(acc_0, x_out_base, oc_full_base_0);
      write_x_tile_result(acc_1, x_out_base, oc_full_base_1);
    }
  } else {
    // ----- OC×2 single-X interior (m7) -----
    AIE_LOOP_RANGE(2, 14)
    for (int x_tile = 1; x_tile < x_tiles - 1; ++x_tile) {
      MMUL4x8x8 acc_0;
      MMUL4x8x8 acc_1;
      acc_0 = bias_acc_oc0;
      acc_1 = bias_acc_oc1;
      const int x_out_base = x_tile * 4;
      const int x_in_base = 2 * x_out_base - 1;
      AIE_LOOP_RANGE(8, 16)
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        AIE_LOOP_RANGE(2, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];
          AIE_LOOP_UNROLL_FULL
          for (int kx = 0; kx < 3; ++kx) {
            alignas(32) int8_t a_buf[32];
            int8_t *src0 =
                line_ptr + (x_in_base + kx) * input_channels + ic_t * 8;
            int8_t *src1 = src0 + 2 * input_channels;
            int8_t *src2 = src0 + 4 * input_channels;
            int8_t *src3 = src0 + 6 * input_channels;
            *(reinterpret_cast<uint64_t *>(&a_buf[0])) =
                *reinterpret_cast<const uint64_t *>(src0);
            *(reinterpret_cast<uint64_t *>(&a_buf[8])) =
                *reinterpret_cast<const uint64_t *>(src1);
            *(reinterpret_cast<uint64_t *>(&a_buf[16])) =
                *reinterpret_cast<const uint64_t *>(src2);
            *(reinterpret_cast<uint64_t *>(&a_buf[24])) =
                *reinterpret_cast<const uint64_t *>(src3);
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
            int wts_off_0 = wts_chunk_tile_off(0, ic_t, ky, kx, ic_tiles,
                                               kernel_height, kernel_width);
            aie::vector<int8, 64> in_b_0 =
                aie::load_v<64>(&wts_chunk[wts_off_0]);
            aie::vector<int8, 64> in_b_1 =
                aie::load_v<64>(&wts_chunk[wts_off_0 + wts_oc_bank_stride]);
            acc_0.mac(in_a, in_b_0);
            acc_1.mac(in_a, in_b_1);
          }
        }
      }
      write_x_tile_result(acc_0, x_out_base, oc_full_base_0);
      write_x_tile_result(acc_1, x_out_base, oc_full_base_1);
    }
  }

  event1();
}

extern "C" {

void KERNEL_NAME(yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts_chunk,
    int32_t *bias, int8_t *silu_lut, int8_t *output, const int32_t input_width,
    const int32_t input_channels, const int32_t output_channels,
    const int32_t kernel_width, const int32_t kernel_height,
    const int32_t border, const int32_t right_shift, const int32_t oc_offset,
    const int32_t oc_count) {
#ifdef NOOP_KERNEL
  // Ablation: skip compute; chain still runs the same DMA/lock pattern.
  (void)line0;
  (void)line1;
  (void)line2;
  (void)wts_chunk;
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
  (void)oc_offset;
  (void)oc_count;
  return;
#else
  yolo_conv2dk3_i8_stride2_silu_bias_oiyxi8o8_chunked_vec(
      line0, line1, line2, wts_chunk, bias, silu_lut, output, input_width,
      input_channels, output_channels, kernel_width, kernel_height, border,
      right_shift, oc_offset, oc_count);
#endif
}

} // extern "C"
