//===- yolo_m9_cv2_concat2_streamed_vec.cc -----------------------*- C++
//-*-===//
//
// Vectorized streamed (chunked-OC) cv2 1x1 conv for the final m9 mixing
// layer. Drop-in .o-level replacement.
//
// Logical input is the concat of two halves (a = top_cache[yi], b = ffn_row),
// each (in_w, c_per_half). cv2's tile already hosts a 32 KB top_cache,
// 8 KB streamed weight slot, plus I/O fifos — no L1 budget for a 4 KB
// pre-pack scratch on top. So the kernel uses the inline-gather pattern
// (small per-(x_tile, ic_t) 32-byte a_buf assembled from either top_row
// or ffn_row depending on whether ic_t < kCHalfTiles).
//
// Inner reduction: aie::mmul<4, 8, 8, int8, int8>; single-acc fold (no
// multi-acc unroll — keeps .o size moderate). Bias + SRS + SiLU LUT in
// scalar tail.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef YOLO_M9_CV2_IN_W
#error "YOLO_M9_CV2_IN_W must be defined at compile time"
#endif
#ifndef YOLO_M9_CV2_C_HALF
#error "YOLO_M9_CV2_C_HALF must be defined at compile time"
#endif
#ifndef YOLO_M9_CV2_OUT_C
#error "YOLO_M9_CV2_OUT_C must be defined at compile time"
#endif
#ifndef YOLO_M9_CV2_N_CHUNKS
#error "YOLO_M9_CV2_N_CHUNKS must be defined at compile time"
#endif

static constexpr int kInW = YOLO_M9_CV2_IN_W;
static constexpr int kCHalf = YOLO_M9_CV2_C_HALF;
static constexpr int kOutC = YOLO_M9_CV2_OUT_C;
static constexpr int kNChunks = YOLO_M9_CV2_N_CHUNKS;

static constexpr int kInC = 2 * kCHalf;
static constexpr int kChunkOc = kOutC / kNChunks;
static constexpr int kIcTiles = kInC / 8;
static constexpr int kChunkOcTiles = kChunkOc / 8;
static constexpr int kXTiles = kInW / 4;
static constexpr int kCHalfTiles = kCHalf / 8;

static_assert(kInW % 4 == 0, "CV2 IN_W must be multiple of 4");
static_assert(kCHalf % 8 == 0, "CV2 C_HALF must be multiple of 8");
static_assert(kOutC % 8 == 0, "CV2 OUT_C must be multiple of 8");

extern "C" {

void yolo_m9_cv2_concat2_streamed_silu_bias_i8_i8(
    int8_t *top_cache, int8_t *ffn_row, int8_t *wts_chunk, int32_t *bias_full,
    int8_t *silu_lut, int8_t *out_row, const int32_t yi,
    const int32_t /*input_width*/, const int32_t /*c_per_half*/,
    const int32_t /*out_c*/, const int32_t /*n_chunks*/,
    const int32_t chunk_idx, const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  const int32_t dst_oc_offset = chunk_idx * kChunkOc;
  const int32_t bias_offset = chunk_idx * kChunkOc;
  const int8_t *__restrict top_row = top_cache + yi * kInW * kCHalf;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs; enables vec to_vector<int8>(rs).
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

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

    AIE_LOOP_RANGE(kXTiles, kXTiles)
    for (int x_tile = 0; x_tile < kXTiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;

      const int x_out_base = x_tile * 4;
      const int8_t *__restrict b_ptr = wts_chunk + ((oc_t * kIcTiles) << 6);

      // Split ic loop into 2 straight-line per-source halves (per "no branches
      // in inner mac loop" feedback) — peano can VLIW-bundle vlda+vldb+vmac
      // when the inner body is a single source. uint64 word copies for the
      // 32-byte A-pack (instead of byte-by-byte) cuts pack cost too.
      AIE_LOOP_RANGE(kCHalfTiles, kCHalfTiles)
      for (int ic_t = 0; ic_t < kCHalfTiles; ++ic_t) {
        alignas(32) int8_t a_buf[32];
        const int8_t *__restrict sp = top_row + x_out_base * kCHalf + ic_t * 8;
        *(reinterpret_cast<uint64_t *>(&a_buf[0])) =
            *reinterpret_cast<const uint64_t *>(sp);
        *(reinterpret_cast<uint64_t *>(&a_buf[8])) =
            *reinterpret_cast<const uint64_t *>(sp + kCHalf);
        *(reinterpret_cast<uint64_t *>(&a_buf[16])) =
            *reinterpret_cast<const uint64_t *>(sp + 2 * kCHalf);
        *(reinterpret_cast<uint64_t *>(&a_buf[24])) =
            *reinterpret_cast<const uint64_t *>(sp + 3 * kCHalf);
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
        aie::vector<int8, 64> in_b = aie::load_v<64>(b_ptr);
        b_ptr += 64;
        acc.mac(in_a, in_b);
      }
      AIE_LOOP_RANGE(kCHalfTiles, kCHalfTiles)
      for (int ic_t = 0; ic_t < kCHalfTiles; ++ic_t) {
        alignas(32) int8_t a_buf[32];
        const int8_t *__restrict sp = ffn_row + x_out_base * kCHalf + ic_t * 8;
        *(reinterpret_cast<uint64_t *>(&a_buf[0])) =
            *reinterpret_cast<const uint64_t *>(sp);
        *(reinterpret_cast<uint64_t *>(&a_buf[8])) =
            *reinterpret_cast<const uint64_t *>(sp + kCHalf);
        *(reinterpret_cast<uint64_t *>(&a_buf[16])) =
            *reinterpret_cast<const uint64_t *>(sp + 2 * kCHalf);
        *(reinterpret_cast<uint64_t *>(&a_buf[24])) =
            *reinterpret_cast<const uint64_t *>(sp + 3 * kCHalf);
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
        aie::vector<int8, 64> in_b = aie::load_v<64>(b_ptr);
        b_ptr += 64;
        acc.mac(in_a, in_b);
      }

      // Vec SiLU emit: build 32-byte silu_buf via scalar gather (LUT is
      // scalar-bound), then 4 vec_store<8> stuck across the strided output.
      // Output stride is kOutC (not contiguous), so 4 8-byte stores.
      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      alignas(32) int8_t silu_buf[32];
      AIE_LOOP_UNROLL_FULL
      for (int i = 0; i < 32; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      AIE_LOOP_UNROLL_FULL
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        int8_t *__restrict row_dst =
            out_row + x_out * kOutC + dst_oc_offset + oc_t * 8;
        *(reinterpret_cast<uint64_t *>(row_dst)) =
            *reinterpret_cast<const uint64_t *>(&silu_buf[p * 8]);
      }
    }
  }

  event1();
}

} // extern "C"
