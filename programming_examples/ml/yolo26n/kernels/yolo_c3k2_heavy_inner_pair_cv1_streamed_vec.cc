//===- yolo_c3k2_heavy_inner_pair_cv1_streamed_vec.cc ---------*- C++ -*-===//
//
// Vectorized chunked-OC 3x3 stride-1 conv + SiLU LUT. Drop-in .o-level
// replacement for yolo_c3k2_heavy_inner_pair_cv1_streamed.cc.
//
// Same inner mmul<4,8,8> pattern as the non-streamed inner_pair_cv1 vec,
// adapted for the chunked-OC API: kernel takes n_chunks + chunk_idx,
// weight buffer holds only this chunk's OC slice, output writes to the
// full-row offset at (oc_offset + chunk_local_oc).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

// Per-block deep-opt: if YOLO_M8_PAIR_IN_W etc. are defined at compile
// time, shape constants fold into shifts + immediates, peano gets exact
// loop trip counts via AIE_LOOP_RANGE hints, and bias is folded into the
// mmul accumulator init (no separate vec epilogue needed).
#ifdef YOLO_M8_PAIR_IN_W
#define IN_W YOLO_M8_PAIR_IN_W
#define IN_C YOLO_M8_PAIR_IN_C
#define OUT_C YOLO_M8_PAIR_OUT_C
#define N_CHUNKS YOLO_M8_PAIR_N_CHUNKS
#define KW 3
#define KH 3
#define SHAPES_ARE_CONST 1
#else
#define IN_W input_width
#define IN_C input_channels
#define OUT_C output_channels
#define N_CHUNKS n_chunks
#define KW kernel_width
#define KH kernel_height
#define SHAPES_ARE_CONST 0
#endif

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

static inline int wts_chunk_tile_off(int chunk_oc_tile, int ic_tile, int ky,
                                     int kx, int ic_tiles, int kH, int kW) {
  return (((chunk_oc_tile * ic_tiles + ic_tile) * kH + ky) * kW + kx) << 6;
}

static inline int wts_chunk_idx(int chunk_oc, int ic_full, int ky, int kx,
                                int in_c, int kH, int kW) {
  int oc_t = chunk_oc >> 3;
  int oc_i = chunk_oc & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((((oc_t * (in_c >> 3) + ic_t) * kH + ky) * kW + kx) << 6) + ic_i * 8 +
         oc_i;
}

extern "C" {

void KERNEL_NAME(
    yolo_c3k2_heavy_inner_pair_cv1_streamed_conv2dk3_silu_bias_i8_i8)(
    int8_t *line0, int8_t *line1, int8_t *line2, int8_t *wts_chunk,
    int32_t *bias_full, int8_t *silu_lut, int8_t *output,
    const int32_t input_width, const int32_t input_channels,
    const int32_t output_channels, const int32_t kernel_width,
    const int32_t kernel_height, const int32_t border,
    const int32_t right_shift, const int32_t n_chunks,
    const int32_t chunk_idx) {
#ifdef NOOP_KERNEL
  return; // Ablation: skip compute, preserve DMA/lock pattern.
#endif
  event0();

#if SHAPES_ARE_CONST
  (void)input_width;
  (void)input_channels;
  (void)output_channels;
  (void)kernel_width;
  (void)kernel_height;
  (void)n_chunks;
#endif

  const int32_t chunk_oc = OUT_C / N_CHUNKS;
  const int32_t oc_offset = chunk_idx * chunk_oc;
  const bool skip_top = (border == 0);
  const bool skip_bot = (border == 2);
  const int ky_start = skip_top ? 1 : 0;
  const int ky_end = skip_bot ? 2 : 3;

  const int ic_tiles = IN_C / 8;
  const int chunk_oc_tiles = chunk_oc / 8;
  const int output_width = IN_W;
  const int x_tiles = output_width / 4;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches the scalar banker_srs used by the runtime tail.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

#if SHAPES_ARE_CONST
  // SHAPES_ARE_CONST path uses mmul<8,8,8>: 8 pixels per acc (vs 4 in the
  // runtime fallback's <4,8,8>). Same underlying HW instruction
  // (mac_8x8_8x8_conf); the 8-wide variant exposes more outputs per call,
  // halving acc invocations + epilogue calls. Dense int8 weights only
  // (sparse_vector variants like <4,16,8> are not applicable here).
  using MMUL8x8x8 = aie::mmul<8, 8, 8, int8, int8>;
  using MMUL_T = MMUL8x8x8;
  constexpr int MMUL_M = 8;
  constexpr int MMUL_MN = 64; // M*N outputs per acc
#else
  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;
  using MMUL_T = MMUL4x8x8;
  constexpr int MMUL_M = 4;
  constexpr int MMUL_MN = 32;
#endif

  int8_t *line[3] = {line0, line1, line2};

#if SHAPES_ARE_CONST
  constexpr int kXTiles8 = IN_W / 8; // pair_cv1 (IN_W=16): 2
#define AIE_HINT_OC AIE_LOOP_RANGE(chunk_oc_tiles, chunk_oc_tiles)
#define AIE_HINT_X AIE_LOOP_RANGE(kXTiles8, kXTiles8)
#define AIE_HINT_IC AIE_LOOP_RANGE(IN_C / 8, IN_C / 8)
#else
#define AIE_HINT_OC
#define AIE_HINT_X
#define AIE_HINT_IC
#endif

  AIE_HINT_OC
  for (int chunk_oc_t = 0; chunk_oc_t < chunk_oc_tiles; ++chunk_oc_t) {
    const int oc_full_base = oc_offset + chunk_oc_t * 8;

    // Hoisted bias_acc init: bias is constant across x_tile iters for a
    // given chunk_oc_t. Init the mmul with bias instead of zero so the
    // post-mac to_vector<int8>(rs) directly produces the bias-added
    // SRS+saturated result. For MMUL_M=8, replicate bias 8 times (one per
    // pixel) instead of 4.
    aie::accum<acc32, MMUL_MN> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias_full[oc_full_base]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
#if SHAPES_ARE_CONST
      aie::vector<int32, 64> b64 = aie::concat(b32, b32);
      bias_acc.from_vector(b64);
#else
      bias_acc.from_vector(b32);
#endif
    }

#if SHAPES_ARE_CONST
    // M=8 path: x_tile loop runs IN_W/8 iters (vs IN_W/4 for M=4).
    AIE_HINT_X
    for (int x_tile = 0; x_tile < kXTiles8; ++x_tile) {
      MMUL_T acc;
      acc = bias_acc;

      const int x_out_base = x_tile * MMUL_M;
      const int x_in_base = x_out_base - 1;

      // Hoisted pixel loads: per (ky, kx) load 8 pixels' FULL 64 channels
      // ONCE (vs current per-(ic_t, ky, kx) scalar lda.u8 chain that re-
      // loads 64 bytes from DM per mac group). Per (ic_t, ky, kx) we then
      // extract the 8-byte slice for THIS ic_t from each pixel register.
      // Reduces 64 scalar mem loads/mac-group to 1 vec mem load + scalar
      // register extracts per mac group. The bounds check is now O(MMUL_M)
      // per (ky, kx) instead of per (ic_t, ky, kx).
      AIE_LOOP_RANGE(1, 3)
      for (int ky = ky_start; ky < ky_end; ++ky) {
        int8_t *line_ptr = line[ky];

        AIE_LOOP_RANGE(3, 3)
        for (int kx = 0; kx < KW; ++kx) {
          // Load 8 pixels × 64 channels = 512 bytes total into vec regs.
          // Pixels at out-of-bounds cols use zero.
          aie::vector<int8, 64> pix[MMUL_M];
          bool col_valid[MMUL_M];
          bool any_valid = false;
          for (int p = 0; p < MMUL_M; ++p) {
            int col = x_in_base + p + kx;
            col_valid[p] = (col >= 0 && col < IN_W);
            if (col_valid[p]) {
              pix[p] = aie::load_v<64>(line_ptr + col * IN_C);
              any_valid = true;
            } else {
              pix[p] = aie::zeros<int8, 64>();
            }
          }
          if (!any_valid)
            continue;

          AIE_HINT_IC
          for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
            // Build mmul input: byte[p*8 + b] = pix[p][ic_t*8 + b]
            alignas(64) int8_t a_buf[64];
            for (int p = 0; p < MMUL_M; ++p) {
              for (int b = 0; b < 8; ++b)
                a_buf[p * 8 + b] = pix[p][ic_t * 8 + b];
            }
            aie::vector<int8, 64> in_a = aie::load_v<64>(a_buf);

            int wts_off =
                wts_chunk_tile_off(chunk_oc_t, ic_t, ky, kx, ic_tiles, KH, KW);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }

      aie::vector<int8, 64> srs_v = acc.template to_vector<int8>(right_shift);
      // mmul-layout output: write 64 bytes (8 pixels x 8 chans) as ONE vec
      // store at offset (chunk_oc_t_full, x_tile). Consumer (cv2) reads with
      // a vec load instead of 64 scalar lda.s8 + vpush gather.
      alignas(64) int8_t silu_buf[64];
      for (int i = 0; i < 64; ++i)
        silu_buf[i] = silu_lut[int(srs_v[i]) + 128];
      aie::vector<int8, 64> silu_v = aie::load_v<64>(silu_buf);
      const int chunk_oc_t_full = oc_full_base / 8;
      aie::store_v(output + chunk_oc_t_full * (kXTiles8 * 64) + x_tile * 64,
                   silu_v);
    }
#else
    AIE_HINT_X
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL_T acc;
      acc = bias_acc;

      const int x_out_base = x_tile * MMUL_M;
      const int x_in_base = x_out_base - 1;

      AIE_HINT_IC
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        AIE_LOOP_RANGE(1, 3)
        for (int ky = ky_start; ky < ky_end; ++ky) {
          int8_t *line_ptr = line[ky];

          AIE_LOOP_RANGE(3, 3)
          for (int kx = 0; kx < KW; ++kx) {
            alignas(32) int8_t a_buf[32];
            bool any_valid = false;
            for (int p = 0; p < MMUL_M; ++p) {
              int col = x_in_base + p + kx;
              if (col < 0 || col >= IN_W) {
                for (int b = 0; b < 8; ++b)
                  a_buf[p * 8 + b] = 0;
              } else {
                int8_t *src = line_ptr + col * IN_C + ic_t * 8;
                for (int b = 0; b < 8; ++b)
                  a_buf[p * 8 + b] = src[b];
                any_valid = true;
              }
            }
            if (!any_valid)
              continue;
            aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

            int wts_off =
                wts_chunk_tile_off(chunk_oc_t, ic_t, ky, kx, ic_tiles, KH, KW);
            aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_chunk[wts_off]);
            acc.mac(in_a, in_b);
          }
        }
      }

      aie::vector<int8, MMUL_MN> srs_v =
          acc.template to_vector<int8>(right_shift);
      for (int p = 0; p < MMUL_M; ++p) {
        int x_out = x_out_base + p;
        for (int j = 0; j < 8; ++j) {
          int oc_full = oc_full_base + j;
          output[x_out * OUT_C + oc_full] =
              silu_lut[int(srs_v[p * 8 + j]) + 128];
        }
      }
    }
#endif

    // Tail scalar fallback.
    for (int x = x_tiles * 4; x < output_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int chunk_oc_local = chunk_oc_t * 8 + j;
        int oc_full = oc_offset + chunk_oc_local;
        int32_t sum = bias_full[oc_full];
        for (int ic_full = 0; ic_full < input_channels; ++ic_full) {
          for (int kx = 0; kx < kernel_width; ++kx) {
            int col = x - 1 + kx;
            if (col < 0 || col >= input_width)
              continue;
            int in_indx = col * input_channels + ic_full;
            int w0 = wts_chunk[wts_chunk_idx(chunk_oc_local, ic_full, 0, kx,
                                             input_channels, kernel_height,
                                             kernel_width)];
            int w1 = wts_chunk[wts_chunk_idx(chunk_oc_local, ic_full, 1, kx,
                                             input_channels, kernel_height,
                                             kernel_width)];
            int w2 = wts_chunk[wts_chunk_idx(chunk_oc_local, ic_full, 2, kx,
                                             input_channels, kernel_height,
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

  event1();
}

} // extern "C"
