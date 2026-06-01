//===- yolo_m9_ffn_0_silu_row_vec.cc -----------------------------*- C++
//-*-===//
//
// Deep-opt vectorized streamed (chunked-OC) 1x1 i8 conv 128 -> 256 + bias
// + SiLU LUT, for the PSA ffn's first conv. Drop-in .o-level replacement
// for yolo_m9_ffn_0_silu_row.cc (same symbol + ABI).
//
// Mirrors the yolo_m9_cv1_split_vec.cc deep-opt pattern minus the top/bot
// output split: per call writes its kChunkOc OC slice into a single
// `mid_out` buffer at offset `chunk_idx * kChunkOc`. Inner reduction:
// aie::mmul<4, 8, 8, int8, int8>.
//
// Toolbox applied:
//  - Compile-time shape #defines (YOLO_M9_FFN0_*) let peano fold all
//    addressing arithmetic into shifts/immediates and dead-strip the
//    scalar tail (input_width is a known multiple of 4).
//  - Input is pre-packed into a 2 KB YCXC8 scratch ONCE per call so the
//    inner mmul loop does a single aligned aie::load_v<32> per
//    (x_tile, ic_tile) instead of a 32-byte gather inside the hot path.
//  - Multi-acc fold across kXTiles=4 x_tiles per oc_t: one B-load
//    serves 4 independent accumulators, breaking the inner ic_t
//    dependency chain so peano can pipeline it tighter.
//  - AIE_LOOP_RANGE hints give peano exact trip counts.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

// Compile-time shape specialization. Caller-side Makefile MUST pass:
//   -DYOLO_M9_FFN0_IN_W=<input spatial width>
//   -DYOLO_M9_FFN0_IN_C=<input channels, multiple of 8>
//   -DYOLO_M9_FFN0_OUT_C=<output channels, multiple of 16>
//   -DYOLO_M9_FFN0_N_CHUNKS=<weight chunks per row, must divide OUT_C/8>
// (right_shift stays a runtime arg.)
#ifndef YOLO_M9_FFN0_IN_W
#error "YOLO_M9_FFN0_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M9_FFN0_IN_C
#error "YOLO_M9_FFN0_IN_C must be defined at compile time"
#endif
#ifndef YOLO_M9_FFN0_OUT_C
#error "YOLO_M9_FFN0_OUT_C must be defined at compile time"
#endif
#ifndef YOLO_M9_FFN0_N_CHUNKS
#error "YOLO_M9_FFN0_N_CHUNKS must be defined at compile time"
#endif

static constexpr int kInW = YOLO_M9_FFN0_IN_W;
static constexpr int kInC = YOLO_M9_FFN0_IN_C;
static constexpr int kOutC = YOLO_M9_FFN0_OUT_C;
static constexpr int kNChunks = YOLO_M9_FFN0_N_CHUNKS;

static constexpr int kChunkOc = kOutC / kNChunks;
static constexpr int kIcTiles = kInC / 8;
static constexpr int kChunkOcTiles = kChunkOc / 8;
static constexpr int kXTiles = kInW / 4;

static_assert(kInC % 8 == 0, "YOLO_M9_FFN0_IN_C must be a multiple of 8");
static_assert(kChunkOc % 8 == 0, "OUT_C/N_CHUNKS must be a multiple of 8");
static_assert(kInW % 4 == 0, "YOLO_M9_FFN0_IN_W must be a multiple of 4");

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void yolo_m9_ffn_0_silu_row_i8_i8(
    int8_t *in_row, int8_t *wts_chunk, int32_t *bias_full, int8_t *silu_lut,
    int8_t *mid_out, const int32_t /*input_width*/,
    const int32_t /*input_channels*/, const int32_t /*out_c*/,
    const int32_t /*n_chunks*/, const int32_t chunk_idx,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  const int32_t dst_oc_offset = chunk_idx * kChunkOc;
  const int32_t bias_offset = chunk_idx * kChunkOc;

  // Pre-pack input row from natural (x, ic) to YCXC8 (ic_t, x, ic_i).
  // 2 KB scratch (ffn.0: 16 ic_tiles × 16 x × 8 ic_i). Strided 8-byte
  // copies; peano lowers these to wide loads/stores. Done once per call
  // so the inner mmul loop sees aligned, contiguous loads.
  //
  // Default AIE2P per-worker stack is 1 KB which a 2 KB array would
  // overflow silently; ffn.0's Worker bumps stack_size=8192 in m9_stage.py.
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
  // conv_even matches scalar banker_srs; enables vec to_vector<int8>(rs).
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  // Multi-acc fold across x_tiles: process all kXTiles=4 x_tiles in
  // lockstep per oc_t iter so a single B-load is reused across 4 mmul.mac
  // calls. Also breaks the inner ic_t dependency chain (4 independent
  // accumulators), letting peano pipeline the inner loop tighter.
  AIE_LOOP_RANGE(kChunkOcTiles, kChunkOcTiles)
  for (int oc_t = 0; oc_t < kChunkOcTiles; ++oc_t) {
    aie::accum<acc32, 32> bias_acc;
    {
      aie::vector<int32, 8> b8 =
          aie::load_v<8>(&bias_full[bias_offset + oc_t * 8]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      bias_acc.from_vector(b32);
    }
    MMUL4x8x8 acc[kXTiles];
    AIE_LOOP_UNROLL_FULL
    for (int xt = 0; xt < kXTiles; ++xt)
      acc[xt] = bias_acc;

    const int8_t *__restrict b_ptr = wts_chunk + ((oc_t * kIcTiles) << 6);

    AIE_LOOP_RANGE(kIcTiles, kIcTiles)
    for (int ic_t = 0; ic_t < kIcTiles; ++ic_t) {
      // ONE B-load shared across all 4 x_tile accumulators.
      aie::vector<int8, 64> in_b = aie::load_v<64>(b_ptr);
      b_ptr += 64;
      const int8_t *__restrict a_base = scratch + ic_t * kInW * 8;
      AIE_LOOP_UNROLL_FULL
      for (int xt = 0; xt < kXTiles; ++xt) {
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_base + xt * 32);
        acc[xt].mac(in_a, in_b);
      }
    }

    // Vec SRS+saturate per x_tile (bias baked in), scalar SiLU LUT gather
    // into a contiguous 8B/pixel buffer, then one uint64 strided store per
    // pixel (vs 8 byte stores). Same pattern as m8 B::cv2 emit (+63% fps).
    AIE_LOOP_UNROLL_FULL
    for (int xt = 0; xt < kXTiles; ++xt) {
      aie::vector<int8, 32> srs_v =
          acc[xt].template to_vector<int8>(right_shift);
      alignas(8) int8_t silu_buf[32];
      AIE_LOOP_UNROLL_FULL
      for (int i = 0; i < 32; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      const int x_out_base = xt * 4;
      AIE_LOOP_UNROLL_FULL
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        int8_t *__restrict row_dst =
            mid_out + x_out * kOutC + dst_oc_offset + oc_t * 8;
        *reinterpret_cast<uint64_t *>(row_dst) =
            *reinterpret_cast<const uint64_t *>(&silu_buf[p * 8]);
      }
    }
  }

  event1();
}

} // extern "C"
