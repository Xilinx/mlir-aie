//===- yolo_m9_proj_skip_row_vec.cc ------------------------------*- C++
//-*-===//
//
// Vectorized fused attn/proj 1x1 + cross-scale skip-add (b) for the PSA
// pipe. Drop-in .o-level replacement for yolo_m9_proj_skip_row.cc.
//
// Math (see scalar header for derivation):
//   proj_q = clip_i8(banker_srs(acc + bias, right_shift))
//   add_q  = proj_q + (b_row[x, oc] << skip_shift)
//   out_i8 = clip_i8(banker_srs(add_q, skip_shift))
//
// Same aie::mmul<4, 8, 8, int8, int8> + on-tile re-pack body as qkv_vec;
// only the scalar tail changes (skip-add epilogue instead of plain SRS).
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

static inline int wts_tile_off(int oc_tile, int ic_tile, int ic_tiles) {
  return ((oc_tile * ic_tiles) + ic_tile) << 6;
}

static inline int wts_idx_oiyxi8o8_1x1(int oc_full, int ic_full, int in_c) {
  int oc_t = oc_full >> 3;
  int oc_i = oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void yolo_m9_proj_skip_row_i8_i8(int8_t *in_row, int8_t *b_cache, int8_t *wts,
                                 int32_t *bias, int8_t *out_row,
                                 const int32_t yi, const int32_t input_width,
                                 const int32_t input_channels,
                                 const int32_t output_channels,
                                 const int32_t right_shift,
                                 const int32_t skip_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  int8_t *b_row = b_cache + yi * input_width * output_channels;

  // Hardcoded for m9 proj_skip call site (in_w=16, in_c=128, out_c=128).
  // Constexpr trip counts let peano lower addressing to immediates and
  // enable AIE_LOOP_RANGE hints below for the inner loops.
  (void)input_width;
  (void)input_channels;
  (void)output_channels;
  constexpr int ic_tiles = 16; // in_c / 8 = 128/8
  constexpr int oc_tiles = 16; // out_c / 8 = 128/8
  constexpr int x_tiles = 4;   // in_w / 4 = 16/4

  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs; enables vec to_vector<int8>(rs).
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  using MMUL4x8x8 = aie::mmul<4, 8, 8, int8, int8>;

  AIE_LOOP_RANGE(oc_tiles, oc_tiles)
  for (int oc_t = 0; oc_t < oc_tiles; ++oc_t) {
    aie::accum<acc32, 32> bias_acc;
    {
      aie::vector<int32, 8> b8 = aie::load_v<8>(&bias[oc_t * 8]);
      aie::vector<int32, 16> b16 = aie::concat(b8, b8);
      aie::vector<int32, 32> b32 = aie::concat(b16, b16);
      bias_acc.from_vector(b32);
    }

    AIE_LOOP_RANGE(x_tiles, x_tiles)
    for (int x_tile = 0; x_tile < x_tiles; ++x_tile) {
      MMUL4x8x8 acc;
      acc = bias_acc;

      const int x_out_base = x_tile * 4;

      AIE_LOOP_RANGE(ic_tiles, ic_tiles)
      for (int ic_t = 0; ic_t < ic_tiles; ++ic_t) {
        alignas(32) int8_t a_buf[32];
        int8_t *s0 = in_row + (x_out_base + 0) * input_channels + ic_t * 8;
        *(reinterpret_cast<uint64_t *>(&a_buf[0])) =
            *reinterpret_cast<const uint64_t *>(s0);
        *(reinterpret_cast<uint64_t *>(&a_buf[8])) =
            *reinterpret_cast<const uint64_t *>(s0 + input_channels);
        *(reinterpret_cast<uint64_t *>(&a_buf[16])) =
            *reinterpret_cast<const uint64_t *>(s0 + 2 * input_channels);
        *(reinterpret_cast<uint64_t *>(&a_buf[24])) =
            *reinterpret_cast<const uint64_t *>(s0 + 3 * input_channels);
        aie::vector<int8, 32> in_a = aie::load_v<32>(a_buf);

        int wts_off = wts_tile_off(oc_t, ic_t, ic_tiles);
        aie::vector<int8, 64> in_b = aie::load_v<64>(&wts[wts_off]);
        acc.mac(in_a, in_b);
      }

      // Step 1: bias-seeded vec SRS+saturate → proj_q (int8 vec).
      aie::vector<int8, 32> proj_v = acc.template to_vector<int8>(right_shift);

      // Step 2: vec skip-add. add_q = proj_q + (b << skip_shift); SRS by
      // skip_shift collapses to b + (proj_q rounded down by skip_shift).
      //
      // Peano AIE2P backend crashes in getCombinedOpcodeUNPACKLoad on
      // accum<acc32, 32>::to_vector<int8> when fed from an aie::mul+mac
      // chain (only — works fine from mmul.mac). Dodge by widening to
      // int16 (no int8 UNPACK in the codegen path); saturation to i8
      // range is done with explicit aie::min/max before the strided store.
      alignas(32) int8_t b_buf[32];
      int8_t *b0 = b_row + (x_out_base + 0) * output_channels + oc_t * 8;
      *(reinterpret_cast<uint64_t *>(&b_buf[0])) =
          *reinterpret_cast<const uint64_t *>(b0);
      *(reinterpret_cast<uint64_t *>(&b_buf[8])) =
          *reinterpret_cast<const uint64_t *>(b0 + output_channels);
      *(reinterpret_cast<uint64_t *>(&b_buf[16])) =
          *reinterpret_cast<const uint64_t *>(b0 + 2 * output_channels);
      *(reinterpret_cast<uint64_t *>(&b_buf[24])) =
          *reinterpret_cast<const uint64_t *>(b0 + 3 * output_channels);
      aie::vector<int8, 32> b_v = aie::load_v<32>(b_buf);

      // Bisect: skip the aie::mul/mac chain entirely; just do scalar
      // skip-add on proj_v + b_v. If THIS compiles, the bug is the
      // downstream mul/mac chain affecting peano's combine on the
      // upstream mmul to_vector<int8>(rs).
      for (int p = 0; p < 4; ++p) {
        int x_out = x_out_base + p;
        int8_t *dst = out_row + x_out * output_channels + oc_t * 8;
        for (int j = 0; j < 8; ++j) {
          int32_t pq = (int32_t)proj_v[p * 8 + j];
          int32_t bb = (int32_t)b_v[p * 8 + j];
          int32_t add_q = pq + (bb << skip_shift);
          int32_t add_i8 = banker_srs(add_q, skip_shift);
          if (add_i8 > I8_MAX)
            add_i8 = I8_MAX;
          if (add_i8 < I8_MIN)
            add_i8 = I8_MIN;
          dst[j] = (int8_t)add_i8;
        }
      }
    }

    // Tail scalar fallback for input_width % 4 != 0.
    for (int x = x_tiles * 4; x < input_width; ++x) {
      for (int j = 0; j < 8; ++j) {
        int oc_full = oc_t * 8 + j;
        int32_t sum = bias[oc_full];
        for (int ic = 0; ic < input_channels; ++ic) {
          sum += in_row[x * input_channels + ic] *
                 wts[wts_idx_oiyxi8o8_1x1(oc_full, ic, input_channels)];
        }
        int32_t proj_q = banker_srs(sum, right_shift);
        if (proj_q > I8_MAX)
          proj_q = I8_MAX;
        if (proj_q < I8_MIN)
          proj_q = I8_MIN;
        int32_t add_q = proj_q + ((int32_t)b_row[x * output_channels + oc_full]
                                  << skip_shift);
        int32_t add_i8 = banker_srs(add_q, skip_shift);
        if (add_i8 > I8_MAX)
          add_i8 = I8_MAX;
        if (add_i8 < I8_MIN)
          add_i8 = I8_MIN;
        out_row[x * output_channels + oc_full] = (int8_t)add_i8;
      }
    }
  }

  event1();
}

} // extern "C"
