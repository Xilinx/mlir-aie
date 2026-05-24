//===- yolo_m9_cv1_split_vec.cc -----------------------------------*- C++
//-*-===//
//
// Vectorized chunked-OC variant of m9's cv1 1x1 INT8 conv. Drop-in
// .o-level replacement (same symbol + ABI as the scalar .cc).
//
// Per call processes chunk_oc = twoc/n_chunks output channels (a contiguous
// slice of the OC dim starting at chunk_idx*chunk_oc). All chunk_oc OCs for
// a given call land in EITHER out_top (chunk_idx < n_chunks/2) OR out_bot
// (chunk_idx >= n_chunks/2). Inner reduction: aie::mmul<4, 8, 8, int8, int8>.
//
// Implementation notes (deep-opt pass):
//  - Compile-time shape #defines (YOLO_M9_CV1_*) let peano fold all
//    addressing arithmetic into shifts/immediates and dead-strip the
//    scalar tail (input_width is a known multiple of 4).
//  - Input is pre-packed into a 4 KB YCXC8 scratch buffer ONCE per call so
//    the inner mmul loop does a single aligned aie::load_v<32> per
//    (x_tile, ic_tile) instead of a 32-byte gather inside the hot path.
//  - AIE_LOOP_RANGE hints give peano exact trip counts so it can pipeline
//    the inner ic_t reduction.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

// Compile-time shape specialization. Caller-side Makefile MUST pass:
//   -DYOLO_M9_CV1_IN_W=<input spatial width>
//   -DYOLO_M9_CV1_IN_C=<input channels, multiple of 8>
//   -DYOLO_M9_CV1_TWOC=<output channels (pre-split), multiple of 16>
//   -DYOLO_M9_CV1_N_CHUNKS=<weight chunks per row, must divide TWOC>
// (right_shift stays a runtime arg — it varies per ONNX block instance and
// folding it doesn't unlock any extra pass.)
#ifndef YOLO_M9_CV1_IN_W
#error "YOLO_M9_CV1_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M9_CV1_IN_C
#error "YOLO_M9_CV1_IN_C must be defined at compile time"
#endif
#ifndef YOLO_M9_CV1_TWOC
#error "YOLO_M9_CV1_TWOC must be defined at compile time"
#endif
#ifndef YOLO_M9_CV1_N_CHUNKS
#error "YOLO_M9_CV1_N_CHUNKS must be defined at compile time"
#endif

static constexpr int kInW = YOLO_M9_CV1_IN_W;
static constexpr int kInC = YOLO_M9_CV1_IN_C;
static constexpr int kTwoC = YOLO_M9_CV1_TWOC;
static constexpr int kNChunks = YOLO_M9_CV1_N_CHUNKS;

static constexpr int kChunkOc = kTwoC / kNChunks;
static constexpr int kHalfC = kTwoC / 2;
static constexpr int kChunksPerHalf = kNChunks / 2;
static constexpr int kIcTiles = kInC / 8;
static constexpr int kChunkOcTiles = kChunkOc / 8;
static constexpr int kXTiles = kInW / 4;

static_assert(kInC % 8 == 0, "YOLO_M9_CV1_IN_C must be a multiple of 8");
static_assert(kChunkOc % 8 == 0, "TWOC/N_CHUNKS must be a multiple of 8");
static_assert(kInW % 4 == 0, "YOLO_M9_CV1_IN_W must be a multiple of 4");
static_assert(kTwoC % 2 == 0, "YOLO_M9_CV1_TWOC must be even");
static_assert(kNChunks % 2 == 0, "YOLO_M9_CV1_N_CHUNKS must be even");

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void yolo_m9_cv1_split_silu_bias_i8_i8(
    int8_t *in_row, int8_t *wts_chunk, int32_t *bias_full, int8_t *silu_lut,
    int8_t *out_top, int8_t *out_bot, const int32_t /*input_width*/,
    const int32_t /*input_channels*/, const int32_t /*twoc*/,
    const int32_t /*n_chunks*/, const int32_t chunk_idx,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  const bool is_top = (chunk_idx < kChunksPerHalf);
  const int32_t dst_oc_offset =
      is_top ? chunk_idx * kChunkOc : (chunk_idx - kChunksPerHalf) * kChunkOc;
  const int32_t bias_offset = chunk_idx * kChunkOc;
  int8_t *__restrict dst = is_top ? out_top : out_bot;

  // Pre-pack input row from natural (x, ic) to YCXC8 (ic_t, x, ic_i).
  // 4 KB scratch (cv1: 32 ic_tiles × 16 x × 8 ic_i). Strided 8-byte
  // copies; peano lowers these to wide loads/stores. Done once per call
  // so the inner mmul loop sees aligned, contiguous loads.
  //
  // Default AIE2P per-worker stack is 1 KB which a 4 KB array would
  // overflow silently; cv1's Worker bumps stack_size=8192 in m9_stage.py.
  alignas(32) int8_t scratch[kIcTiles * kInW * 8];
  AIE_LOOP_RANGE(kInW, kInW)
  for (int x = 0; x < kInW; ++x) {
    AIE_LOOP_RANGE(kIcTiles, kIcTiles)
    for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
      const int8_t *__restrict src = in_row + x * kInC + ic_t * 8;
      int8_t *__restrict d = scratch + ic_t * kInW * 8 + x * 8;
      // 8-byte block copy (peano lowers to a single 64-bit load/store pair).
      for (int b = 0; b < 8; ++b)
        d[b] = src[b];
    }
  }

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  // Multi-acc fold across x_tiles: process all kXTiles=4 x_tiles in
  // lockstep per oc_t iter so a single B-load is reused across 4 mmul.mac
  // calls. Also breaks the inner ic_t dependency chain (4 independent
  // accumulators), letting peano pipeline the inner loop tighter.
  AIE_LOOP_RANGE(kChunkOcTiles, kChunkOcTiles)
  for (int oc_t = 0; oc_t < kChunkOcTiles; ++oc_t) {
    MMUL4x8x8 acc[kXTiles];
    AIE_LOOP_UNROLL_FULL
    for (int xt = 0; xt < kXTiles; ++xt)
      acc[xt] = aie::zeros<acc32, 32>();

    const int8_t *__restrict b_ptr = wts_chunk + ((oc_t * kIcTiles) << 6);

    AIE_LOOP_RANGE(kIcTiles, kIcTiles)
    for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
      // ONE B-load shared across all 4 x_tile accumulators.
      aie::vector<int8, 64> in_b = aie::load_v<64>(b_ptr);
      b_ptr += 64;
      // A vectors land contiguously in scratch for ic_t's row of 16 x's
      // (16 x × 8 ic_i = 128 B = 4 × 32 B), so the 4 a_ptrs are
      // a_base, a_base+32, a_base+64, a_base+96 within the same row.
      const int8_t *__restrict a_base = scratch + ic_t * kInW * 8;
      AIE_LOOP_UNROLL_FULL
      for (int xt = 0; xt < kXTiles; ++xt) {
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_base + xt * 32);
        acc[xt].mac(in_a, in_b);
      }
    }

    // Scalar bias + SRS + SiLU LUT tail, per x_tile, per (m=p, n=j) lane.
    const int32_t *__restrict bias_p = bias_full + bias_offset + oc_t * 8;
    AIE_LOOP_UNROLL_FULL
    for (int xt = 0; xt < kXTiles; ++xt) {
      aie::vector<int32, 32> acc_vec = acc[xt].template to_vector<int32>();
      const int x_out_base = xt * 4;
      AIE_LOOP_UNROLL_FULL
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        int8_t *__restrict row_dst =
            dst + x_out * kHalfC + dst_oc_offset + oc_t * 8;
        AIE_LOOP_UNROLL_FULL
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias_p[j];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX)
            sr = I8_MAX;
          if (sr < I8_MIN)
            sr = I8_MIN;
          row_dst[j] = silu_lut[sr + 128];
        }
      }
    }
  }

  event1();
}

} // extern "C"
