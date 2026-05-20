//===- yolo_c3k2_heavy_m_0_split_plus_cv3_vec.cc -------------*- C++ -*-===//
//
// Fused m_0_split + cv3 kernel for m8 stage 5. Runs both operations on
// a single compute tile, eliminating:
//   - The (5,3) -> (5,4) split_b fifo (depth=16 = 16 KB freed on (5,3))
//   - Tile (5,4) entirely (reclaimable for other use)
//   - Per-pixel scalar gathers on cv3's split_b input (replaced by
//     contiguous `aie::load_v<32>` over ic-tile-major scratch)
//
// Vectorization / alignment improvements from fusion:
//   - Phase A (split) writes split_b to internal scratch in IC-TILE-MAJOR
//     layout: scratch[ic_tile_local][col][ic_inner]. Consecutive bytes
//     span pixels of one ic_tile slab — exactly the layout mmul<4,8,8>'s
//     A operand wants for a 4-pixel x 8-ic gather.
//   - Phase B (cv3) reads split_b via one contiguous 32-byte load per
//     (oc_tile, x_tile, ic_tile_local) iteration — NO per-pixel scalar
//     copies. Inner1 input stays HWC (its producer pair1_cv2 still
//     emits HWC) so inner1 retains the scalar gather pattern; the
//     split_b half of the concat is the optimized half.
//
// Numerics bit-exact with the unfused split + cv3 kernels.
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

// 1x1 branch that writes output in HWC layout (used for split_a -> pair0,
// which expects HWC).
static void branch_1x1_hwc_out(
    int8_t *in_row, int8_t *wts, int32_t *bias, int8_t *silu_lut,
    int8_t *out_hwc, int input_width, int input_channels, int output_channels,
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
          int8_t *src = in_row + col * input_channels + ic_t * 8;
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
          out_hwc[x_out * output_channels + oc_t * 8 + j] = silu_lut[sr + 128];
        }
      }
    }
  }
}

// 1x1 branch that writes output in IC-TILE-MAJOR layout to scratch:
//   out_ic_tile_major[ic_tile][col][ic_inner] flat
//                    = byte at (ic_tile * input_width * 8) + (col * 8) + ic_inner.
// Used for split_b -> cv3 (intra-tile fusion path).
static void branch_1x1_ictile_out(
    int8_t *in_row, int8_t *wts, int32_t *bias, int8_t *silu_lut,
    int8_t *out_ic_tile_major,
    int input_width, int input_channels, int output_channels,
    int right_shift) {
  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;
  const int ic_tiles = input_channels / 8;
  const int oc_tiles = output_channels / 8;
  const int x_tiles = input_width / 4;
  const int W8 = input_width * 8;

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    // Output ic_tile_local == oc_t for the output (we're writing the
    // produced channels, which are the "input channels" of the next conv).
    int8_t *out_ic_slab = out_ic_tile_major + oc_t * W8;

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
        int wts_off = wts_tile_off_1x1(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }
      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      // Write 4 contiguous pixels of this oc_tile slab in ic-tile-major
      // layout: out_ic_slab[(x_base + p) * 8 + j] = silu(...).
      for (int p = 0; p < 4; ++p) {
        int col = x_base + p;
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias[oc_t * 8 + j];
          int32_t sr = banker_srs(s, right_shift);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          out_ic_slab[col * 8 + j] = silu_lut[sr + 128];
        }
      }
    }
  }
}

extern "C" {

// Phase A: m_0_split for one row. Outputs split_a in HWC (-> pair0 fifo);
// stores split_b in ic-tile-major within `split_b_scratch` at row_idx's slab.
// split_b_scratch layout per row: (cp/8) * input_width * 8 bytes contiguous.
// Total scratch: (cp/8) * input_width * 8 * in_h bytes.
void KERNEL_NAME(yolo_m8_split_plus_cv3_phaseA_i8_i8)(
    int8_t *in_row,
    int8_t *wts_a, int32_t *bias_a, int8_t *silu_lut_a,
    int8_t *wts_b, int32_t *bias_b, int8_t *silu_lut_b,
    int8_t *split_a_out,
    int8_t *split_b_scratch,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t cp,
    const int32_t right_shift_a,
    const int32_t right_shift_b,
    const int32_t row_idx) {
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  // split_a: HWC for pair0_cv1 consumption (unchanged).
  branch_1x1_hwc_out(in_row, wts_a, bias_a, silu_lut_a, split_a_out,
                     input_width, input_channels, cp, right_shift_a);

  // split_b: ic-tile-major into per-row scratch slab.
  const int row_bytes = (cp / 8) * input_width * 8;  // = input_width * cp
  int8_t *split_b_row = split_b_scratch + row_idx * row_bytes;
  branch_1x1_ictile_out(in_row, wts_b, bias_b, silu_lut_b, split_b_row,
                        input_width, input_channels, cp, right_shift_b);
  event1();
}

// Phase B: cv3 for one row. inner1 source is HWC (per-pixel gather kept);
// split_b source is ic-tile-major (contiguous load_v<32>).
void KERNEL_NAME(yolo_m8_split_plus_cv3_phaseB_i8_i8)(
    int8_t *inner1_row,
    int8_t *split_b_scratch,
    int8_t *wts_cv3, int32_t *bias_cv3, int8_t *silu_lut_cv3,
    int8_t *cv3_out_row,
    const int32_t input_width,
    const int32_t cp,
    const int32_t out_c_cv3,
    const int32_t right_shift_cv3,
    const int32_t row_idx) {
  event0();
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;
  const int two_cp = 2 * cp;
  const int ic_tiles_total = two_cp / 8;       // OIYXI8O8 weight ic_tile count
  const int ic_tiles_per_src = cp / 8;
  const int oc_tiles = out_c_cv3 / 8;
  const int x_tiles = input_width / 4;
  const int W8 = input_width * 8;

  const int row_bytes = ic_tiles_per_src * W8;
  int8_t *split_b_row = split_b_scratch + row_idx * row_bytes;

  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = aie::zeros<acc32, 32>();
      const int x_base = x_tile * 4;

      // First half of ic_tiles (inner1, HWC source — scalar gather).
      for (int ic_t_local = 0; ic_t_local < ic_tiles_per_src; ++ic_t_local) {
        alignas(32) int8_t a_buf[32];
        for (int p = 0; p < 4; ++p) {
          int col = x_base + p;
          int8_t *psrc = inner1_row + col * cp + ic_t_local * 8;
          for (int b = 0; b < 8; ++b) a_buf[p * 8 + b] = psrc[b];
        }
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);
        int ic_t_global = ic_t_local;
        int wts_off = wts_tile_off_1x1(oc_t, ic_t_global, ic_tiles_total);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_cv3[wts_off]);
        acc.mac(in_a, in_b);
      }

      // Second half of ic_tiles (split_b, ic-tile-major scratch —
      // contiguous 32-byte vector load, NO gather).
      for (int ic_t_local = 0; ic_t_local < ic_tiles_per_src; ++ic_t_local) {
        int8_t *src = split_b_row + ic_t_local * W8 + x_base * 8;
        aie::vector<int8, 32> in_a = aie::load_v<32>(src);
        int ic_t_global = ic_tiles_per_src + ic_t_local;
        int wts_off = wts_tile_off_1x1(oc_t, ic_t_global, ic_tiles_total);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts_cv3[wts_off]);
        acc.mac(in_a, in_b);
      }

      aie::vector<int32, 32> acc_vec = acc.template to_vector<int32>();
      for (int p = 0; p < 4; ++p) {
        int x_out = x_base + p;
        for (int j = 0; j < 8; ++j) {
          int32_t s = acc_vec[p * 8 + j] + bias_cv3[oc_t * 8 + j];
          int32_t sr = banker_srs(s, right_shift_cv3);
          if (sr > I8_MAX) sr = I8_MAX;
          if (sr < I8_MIN) sr = I8_MIN;
          cv3_out_row[x_out * out_c_cv3 + oc_t * 8 + j] = silu_lut_cv3[sr + 128];
        }
      }
    }
  }

  event1();
}

} // extern "C"
