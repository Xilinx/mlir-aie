//===- yolo_m8_front_cv1_split_fused_vec.cc -----------------*- C++ -*-===//
//
// Fused cv1 + m_0_split kernel for the m8 megakernel design. Combines:
//   - cv1 (1x1 split, 256 ic -> 128 top + 128 bot, chunked over oc)
//   - m_0_split (1x1x2 on bot: 128 ic -> 64 split_a + 64 split_b)
//
// Called once per (row, cv1_chunk_idx). Within a row:
//   - chunks 0..N/2-1 write to out_top (first 128 oc)
//   - chunks N/2..N-1 write to (s_bot scratch AND out_bot_to_cv2) (second 128
//   oc)
//   - on the LAST chunk (chunk_idx == n_chunks - 1), bot is fully assembled
//     in s_bot; run m_0_split using s_bot -> split_a + split_b
//
// s_bot persists across calls within a row (it's just the accumulating
// bot output). cv1's chunked OC writes touch disjoint slices, so no
// inter-chunk clearing needed; m_0_split simply reads the full row
// once the last chunk has landed.
//
// Numerics bit-exact with yolo_c3k2_small_cv1_split_streamed_vec.cc +
// yolo_c3k2_heavy_m_0_split_vec.cc when called in sequence.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

static inline int wts_tile_off_1x1(int oc_tile, int ic_tile, int ic_tiles) {
  return (oc_tile * ic_tiles + ic_tile) << 6;
}

// Build a 32-wide bias accumulator (8 int32 bias values replicated to
// 4 pixels x 8 channels) for direct mmul init. Avoids the int32-vec
// round-trip through a helper.
static __attribute__((always_inline)) inline aie::accum<acc32, 32>
make_bias_acc(const int32_t *bias_8) {
  aie::vector<int32, 8> b8 = aie::load_v<8>(bias_8);
  aie::vector<int32, 16> b16 = aie::concat(b8, b8);
  aie::vector<int32, 32> b32 = aie::concat(b16, b16);
  aie::accum<acc32, 32> a;
  a.from_vector(b32);
  return a;
}

// cv1 chunked compute: this chunk's chunk_oc channels of the 1x1 conv.
// First-half chunks (chunk_idx < n_chunks/2) write to out_top.
// Second-half chunks write to BOTH s_bot (scratch for m_0_split, to be
// consumed within this same kernel call when chunk_idx == last) AND
// out_bot_to_cv2 (cv2 skip path).
static inline void cv1_chunk_compute(int8_t *in_row, int8_t *wts_chunk,
                                     int32_t *bias_full, int8_t *silu_lut,
                                     int8_t *out_top, int8_t *s_bot,
                                     int8_t *out_bot_to_cv2, int input_width,
                                     int input_channels, int twoc, int n_chunks,
                                     int chunk_idx, int right_shift) {
  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  // Hardcoded for m8 cv1 call site (in_w=16, in_c=256, twoc=256, N=8).
  // The original `twoc / n_chunks` was signed-runtime division → __divsi3
  // call. Constexpr lowers everything to shifts/immediates.
  (void)input_width;
  (void)input_channels;
  (void)twoc;
  (void)n_chunks;
  constexpr int chunk_oc = 32;       // twoc / n_chunks = 256/8
  constexpr int c = 128;             // twoc / 2
  constexpr int chunks_per_half = 4; // n_chunks / 2
  const bool is_top = (chunk_idx < chunks_per_half);
  const int dst_oc_offset =
      is_top ? chunk_idx * chunk_oc : (chunk_idx - chunks_per_half) * chunk_oc;
  const int bias_offset = chunk_idx * chunk_oc;
  constexpr int ic_tiles = 32;      // input_channels / 8 = 256/8
  constexpr int chunk_oc_tiles = 4; // chunk_oc / 8
  constexpr int x_tiles = 4;        // input_width / 4 = 16/4

  for (int chunk_oc_t = 0; chunk_oc_t < chunk_oc_tiles; ++chunk_oc_t) {
    const int dst_oc_full_base = dst_oc_offset + chunk_oc_t * 8;
    const int bias_full_base = bias_offset + chunk_oc_t * 8;
    auto bias_acc = make_bias_acc(&bias_full[bias_full_base]);

    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;
      const int x_base = x_tile * 4;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_base + p;
          int8_t *src = in_row + col * input_channels + ic_t * 8;
          for (int b = 0; b < 8; ++b)
            a_buf[p * 8 + b] = src[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off_1x1(chunk_oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      // mmul-layout output for out_top / out_bot_to_cv2 (consumed by back
      // cv2 via vec_load); s_bot kept in (W,c) raster because m_0_split's
      // current reader scalar-packs from it. mmul<4,8,8> output is 32 bytes
      // covering 4 pixels x 8 chans; pair pairs of x_tile iters into one
      // 8-pixel block in the back-side layout.
      alignas(32) int8_t silu_buf[32];
      for (int i = 0; i < 32; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      aie::vector<int8, 32> silu_v = aie::load_v<32>(silu_buf);
      constexpr int kXTiles8 = 2; // 16W / 8 pixels per back-side block
      const int packed_oc_t = dst_oc_full_base >> 3;
      const int packed_off = packed_oc_t * (kXTiles8 * 64) + (x_tile >> 1) * 64 +
                             (x_tile & 1) * 32;
      if (is_top) {
        aie::store_v(out_top + packed_off, silu_v);
      } else {
        aie::store_v(out_bot_to_cv2 + packed_off, silu_v);
        for (int p = 0; p < 4; ++p) {
          int x_out = x_base + p;
          for (int j = 0; j < 8; ++j)
            s_bot[x_out * c + dst_oc_full_base + j] = silu_buf[p * 8 + j];
        }
      }
    }
  }
}

// m_0_split branch: input bot (128 ch) -> SiLU LUT -> output (64 ch).
// Called twice (once per branch: a -> split_a, b -> split_b). When
// packed_output is true, the output write uses mmul-packed
// (ic_t, x_block, p*8+chan) layout for consumer-side vec_load (back cv3);
// otherwise (W,c) raster (pair_cv1 still scalar-packs from split_a).
static inline void m0_split_branch(int8_t *in_bot, int8_t *wts, int32_t *bias,
                                   int8_t *silu_lut, int8_t *out,
                                   int input_width, int input_channels,
                                   int output_channels, int right_shift,
                                   bool packed_output) {
  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;
  // Hardcoded for m8 m_0_split call site (in_w=16, in_c=128, out_c=64).
  (void)input_width;
  (void)input_channels;
  (void)output_channels;
  constexpr int ic_tiles = 16; // 128 / 8
  constexpr int oc_tiles = 8;  // 64 / 8
  constexpr int x_tiles = 4;   // 16 / 4

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    auto bias_acc = make_bias_acc(&bias[oc_t * 8]);
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;
      const int x_base = x_tile * 4;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_base + p;
          int8_t *src = in_bot + col * input_channels + ic_t * 8;
          for (int b = 0; b < 8; ++b)
            a_buf[p * 8 + b] = src[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int8, 32> srs_v = acc.template to_vector<int8>(right_shift);
      alignas(32) int8_t silu_buf[32];
      for (int i = 0; i < 32; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      if (packed_output) {
        constexpr int kXTiles8 = 2; // 16W / 8 pixels per back-side block
        aie::vector<int8, 32> silu_v = aie::load_v<32>(silu_buf);
        aie::store_v(out + oc_t * (kXTiles8 * 64) + (x_tile >> 1) * 64 +
                         (x_tile & 1) * 32,
                     silu_v);
      } else {
        for (int p = 0; p < 4; ++p) {
          int x_out = x_base + p;
          for (int j = 0; j < 8; ++j)
            out[x_out * output_channels + oc_t * 8 + j] = silu_buf[p * 8 + j];
        }
      }
    }
  }
}

extern "C" {

// `scratch` is a tile-allocated 2 KB Buffer (16 W * 128 c) passed in by the
// IRON design. Accumulates bot half across the N_CV1_CHUNKS calls within a
// row; on the last chunk, m_0_split runs reading from it. Safe to share
// with m8_back's scratch buffer (used at different times per iter).
void KERNEL_NAME(yolo_m8_front_cv1_split_fused_i8_i8)(
    // cv1 inputs / weights
    int8_t *in_row, int8_t *cv1_wts_chunk, int32_t *bias_cv1,
    int8_t *silu_lut_cv1, int8_t *out_top, int8_t *out_bot_to_cv2,
    // m_0_split static weights (used only on last chunk)
    int8_t *wts_m0c1, int32_t *bias_m0c1, int8_t *silu_lut_m0c1,
    int8_t *wts_m0c2, int32_t *bias_m0c2, int8_t *silu_lut_m0c2,
    int8_t *out_split_a, int8_t *out_split_b,
    int8_t *scratch, // accumulating bot buffer (16 * 128 = 2 KB)
    // dims
    const int32_t input_width,
    const int32_t input_channels, // cv1 ic = 256
    const int32_t twoc,           // cv1 oc = 256
    const int32_t cp,             // m_0_split oc per branch = 64
    const int32_t n_cv1_chunks, const int32_t cv1_chunk_idx,
    const int32_t rs_cv1, const int32_t rs_m0c1, const int32_t rs_m0c2) {
#ifdef NOOP_KERNEL
  return; // Ablation: skip compute, preserve DMA/lock pattern.
#endif
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  const int32_t c = twoc >> 1; // 128

  // Always: do this cv1 chunk. Writes either out_top or (scratch +
  // out_bot_to_cv2).
  cv1_chunk_compute(in_row, cv1_wts_chunk, bias_cv1, silu_lut_cv1, out_top,
                    scratch, out_bot_to_cv2, input_width, input_channels, twoc,
                    n_cv1_chunks, cv1_chunk_idx, rs_cv1);

  // On last chunk: bot is now fully assembled in scratch. Run m_0_split.
  if (cv1_chunk_idx == n_cv1_chunks - 1) {
    m0_split_branch(scratch, wts_m0c1, bias_m0c1, silu_lut_m0c1, out_split_a,
                    input_width, c, cp, rs_m0c1, /*packed_output=*/false);
    m0_split_branch(scratch, wts_m0c2, bias_m0c2, silu_lut_m0c2, out_split_b,
                    input_width, c, cp, rs_m0c2, /*packed_output=*/true);
  }

  event1();
}

} // extern "C"
