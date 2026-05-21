//===- yolo_m8_front_cv1_split_fused_vec.cc -----------------*- C++ -*-===//
//
// Fused cv1 + m_0_split kernel for the m8 megakernel design. Combines:
//   - cv1 (1x1 split, 256 ic -> 128 top + 128 bot, chunked over oc)
//   - m_0_split (1x1x2 on bot: 128 ic -> 64 split_a + 64 split_b)
//
// Called once per (row, cv1_chunk_idx). Within a row:
//   - chunks 0..N/2-1 write to out_top (first 128 oc)
//   - chunks N/2..N-1 write to (s_bot scratch AND out_bot_to_cv2) (second 128 oc)
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

// cv1 chunked compute: this chunk's chunk_oc channels of the 1x1 conv.
// First-half chunks (chunk_idx < n_chunks/2) write to out_top.
// Second-half chunks write to BOTH s_bot (scratch for m_0_split, to be
// consumed within this same kernel call when chunk_idx == last) AND
// out_bot_to_cv2 (cv2 skip path).
static inline void cv1_chunk_compute(
    int8_t *in_row, int8_t *wts_chunk, int32_t *bias_full, int8_t *silu_lut,
    int8_t *out_top, int8_t *s_bot, int8_t *out_bot_to_cv2,
    int input_width, int input_channels, int twoc,
    int n_chunks, int chunk_idx, int right_shift) {
  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  const int chunk_oc = twoc / n_chunks;
  const int c = twoc >> 1;
  const int chunks_per_half = n_chunks >> 1;
  const bool is_top = (chunk_idx < chunks_per_half);
  const int dst_oc_offset =
      is_top ? chunk_idx * chunk_oc
             : (chunk_idx - chunks_per_half) * chunk_oc;
  const int bias_offset = chunk_idx * chunk_oc;

  const int ic_tiles = input_channels / 8;
  const int chunk_oc_tiles = chunk_oc / 8;
  const int x_tiles = input_width / 4;

  for (int chunk_oc_t = 0; chunk_oc_t < chunk_oc_tiles; ++chunk_oc_t) {
    const int dst_oc_full_base = dst_oc_offset + chunk_oc_t * 8;
    const int bias_full_base = bias_offset + chunk_oc_t * 8;

    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();
      const int x_base = x_tile * 4;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_base + p;
          int8_t *src = in_row + col * input_channels + ic_t * 8;
          for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off_1x1(chunk_oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_base + p;
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias_full[bias_full_base + j];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          int8_t silu = silu_lut[sr + 128];
          const int idx = x_out * c + (dst_oc_full_base + j);
          if (is_top) {
            out_top[idx] = silu;
          } else {
            s_bot[idx] = silu;
            out_bot_to_cv2[idx] = silu;
          }
        }
      }
    }
  }
}

// m_0_split branch: input bot (128 ch) -> SiLU LUT -> output (64 ch).
// Called twice (once per branch: a -> split_a, b -> split_b).
static inline void m0_split_branch(
    int8_t *in_bot, int8_t *wts, int32_t *bias, int8_t *silu_lut,
    int8_t *out, int input_width, int input_channels, int output_channels,
    int right_shift) {
  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;
  const int ic_tiles = input_channels / 8;
  const int oc_tiles = output_channels / 8;
  const int x_tiles = input_width / 4;

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();
      const int x_base = x_tile * 4;

      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_base + p;
          int8_t *src = in_bot + col * input_channels + ic_t * 8;
          for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = src[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_base + p;
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias[oc_t * 8 + j];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          out[x_out * output_channels + oc_t * 8 + j] = silu_lut[sr + 128];
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
    int8_t *in_row,
    int8_t *cv1_wts_chunk, int32_t *bias_cv1, int8_t *silu_lut_cv1,
    int8_t *out_top, int8_t *out_bot_to_cv2,
    // m_0_split static weights (used only on last chunk)
    int8_t *wts_m0c1, int32_t *bias_m0c1, int8_t *silu_lut_m0c1,
    int8_t *wts_m0c2, int32_t *bias_m0c2, int8_t *silu_lut_m0c2,
    int8_t *out_split_a, int8_t *out_split_b,
    int8_t *scratch,           // accumulating bot buffer (16 * 128 = 2 KB)
    // dims
    const int32_t input_width,
    const int32_t input_channels,  // cv1 ic = 256
    const int32_t twoc,            // cv1 oc = 256
    const int32_t cp,              // m_0_split oc per branch = 64
    const int32_t n_cv1_chunks,
    const int32_t cv1_chunk_idx,
    const int32_t rs_cv1,
    const int32_t rs_m0c1,
    const int32_t rs_m0c2) {
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  const int32_t c = twoc >> 1;  // 128

  // Always: do this cv1 chunk. Writes either out_top or (scratch + out_bot_to_cv2).
  cv1_chunk_compute(in_row, cv1_wts_chunk, bias_cv1, silu_lut_cv1,
                    out_top, scratch, out_bot_to_cv2,
                    input_width, input_channels, twoc,
                    n_cv1_chunks, cv1_chunk_idx, rs_cv1);

  // On last chunk: bot is now fully assembled in scratch. Run m_0_split.
  if (cv1_chunk_idx == n_cv1_chunks - 1) {
    m0_split_branch(scratch, wts_m0c1, bias_m0c1, silu_lut_m0c1, out_split_a,
                    input_width, c, cp, rs_m0c1);
    m0_split_branch(scratch, wts_m0c2, bias_m0c2, silu_lut_m0c2, out_split_b,
                    input_width, c, cp, rs_m0c2);
  }

  event1();
}

}  // extern "C"
