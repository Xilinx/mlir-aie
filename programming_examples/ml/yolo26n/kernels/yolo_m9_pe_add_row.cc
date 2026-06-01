//===- yolo_m9_pe_add_row.cc ---------------------------------*- C++ -*-===//
//
// Stage 7 fused kernel: compute one (c=128, x=in_w=16) row of pe (dw3x3
// stride-1 with zero-padding) PLUS the cross-channel addition with sv,
// emitting a chunk of (chunk_cols=in_w, c) i8 values.
//
// V is held as two per-head buffers (head_dim=64, N=256) on the sv_tile.
// V_chw[c, y, x] reconstructed as:
//   c <  head_dim : v_h0[c,         y*in_w + x]
//   c >= head_dim : v_h1[c-head_dim, y*in_w + x]
//
// pe weights are (c, 1, 3, 3) OIYX_raw (group=128). pe_wts[c*9 + ky*3 + kx]
// is the (ky, kx) tap for output channel c.
//
// sv contributions for row y come from:
//   sv_h0_acc : (N=256, head_dim) — accumulator filled in Phase 2a
//               sv_h0_acc[y*in_w + x, c] for c in [0, head_dim)
//   sv_h1_row : (in_w, head_dim) — per-y scratch filled before this call
//               sv_h1_row[x, c] for c in [0, head_dim) → maps to c'=64..127
//               globally
//
// Output (chunk_cols=in_w, c) = clip_i8( sv + pe ) with both at the
// same QL scale 2^-4 (verified from ONNX: MatMul_1 / pe / Add all at
// scale 2^-4), so the add is plain integer sum + clip — no cross-scale
// shift needed.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void yolo_m9_pe_add_row_i8_i8(
    int8_t *v_h0,      // (head_dim, N)
    int8_t *v_h1,      // (head_dim, N)
    int8_t *sv_h0_acc, // (N, head_dim)
    int8_t *sv_h1_row, // (in_w, head_dim)  (per-y scratch)
    int8_t *pe_wts,    // (c, 1, 3, 3) OIYX_raw flat (= c*9)
    int32_t *pe_bias,  // (c,)
    int8_t *chunk_out, // (in_w, c) one full output row
    const int32_t y_idx,
    const int32_t in_w,     // 16
    const int32_t in_h,     // 16
    const int32_t head_dim, // 64
    const int32_t c_total,  // 128 (= 2 * head_dim)
    const int32_t N,        // in_w * in_h = 256
    const int32_t rs_pe) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  // Hardcoded for m9 PE call site: in_w=in_h=16, head_dim=64, c_total=128,
  // N=256, rs_pe=runtime. Constexpr enables loop hints + addressing immediates.
  (void)in_w;
  (void)in_h;
  (void)head_dim;
  (void)c_total;
  (void)N;
  constexpr int kInW = 16;
  constexpr int kInH = 16;
  constexpr int kHeadDim = 64;
  constexpr int kCTotal = 128;
  constexpr int kN = 256;

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  // Per c: 3x3 dw conv. Vec along x (16-wide). Per ky we gather a
  // padded row (zero-fill at the two edges) into a 18-byte scratch so
  // the 3 kx loads are `aie::load_v<16>(&padded[kx])` (covers cols
  // [kx-1 .. kx+14] mapped to [-1..14], [0..15], [1..16] with zero pad).
  // 9 vec macs per c (or fewer with y-edge pad) replace 9*16 scalar macs.
  AIE_LOOP_RANGE(kCTotal, kCTotal)
  for (int c = 0; c < kCTotal; c++) {
    const int32_t binit = pe_bias[c];
    int8_t *v_chans = (c < kHeadDim) ? v_h0 : v_h1;
    const int32_t v_c = (c < kHeadDim) ? c : (c - kHeadDim);
    int8_t *v_row = &v_chans[v_c * kN];
    int8_t *pe_w_c = &pe_wts[c * 9];

    aie::accum<acc32, 16> x_acc;
    x_acc.from_vector(aie::broadcast<int32, 16>(binit));

    AIE_LOOP_RANGE(3, 3)
    for (int ky = 0; ky < 3; ky++) {
      int32_t yy = y_idx + ky - 1;
      if (yy < 0 || yy >= kInH)
        continue; // y-edge zero pad → skip whole ky band

      // Pad row: [0] and [17] are zero; [1..16] hold v_row[yy*kInW+0..15].
      // 16-byte vec_load + offset vec_store replaces 16-iter scalar copy.
      // padded is alignas(32) so padded[0] is 32-aligned; storing to
      // padded+1 (unaligned) — peano supports unaligned aie::store_v at
      // this width (verified by sibling aie::load_v<16>(&padded[kx])
      // unaligned loads at kx=1,2 in the inner kx loop below).
      alignas(32) int8_t padded[32] = {0};
      const int8_t *row_p = v_row + yy * kInW;
      aie::store_v(&padded[1], aie::load_v<16>(row_p));

      AIE_LOOP_UNROLL_FULL
      for (int kx = 0; kx < 3; kx++) {
        aie::vector<int8, 16> v_v = aie::load_v<16>(&padded[kx]);
        x_acc = aie::mac(x_acc, v_v, (int8)pe_w_c[ky * 3 + kx]);
      }
    }

    // Scalar SRS+clamp+sv-add+clamp+store tail per x (vec to_vector<int8>(rs)
    // crashes peano on 16-wide acc — known bug in getCombinedOpcodeUNPACKLoad).
    aie::vector<int32, 16> x_v = x_acc.template to_vector<int32>();
    const bool from_h0 = (c < kHeadDim);
    const int8_t *sv_h0_p = sv_h0_acc + (y_idx * kInW) * kHeadDim + c;
    const int8_t *sv_h1_p = sv_h1_row + (c - kHeadDim);
    AIE_LOOP_UNROLL_FULL
    for (int x = 0; x < kInW; x++) {
      int32_t pe_q = banker_srs(x_v[x], rs_pe);
      if (pe_q > I8_MAX)
        pe_q = I8_MAX;
      if (pe_q < I8_MIN)
        pe_q = I8_MIN;
      int8_t sv_val = from_h0 ? sv_h0_p[x * kHeadDim] : sv_h1_p[x * kHeadDim];
      int32_t s = (int32_t)sv_val + (int32_t)pe_q;
      if (s > I8_MAX)
        s = I8_MAX;
      if (s < I8_MIN)
        s = I8_MIN;
      chunk_out[x * kCTotal + c] = (int8_t)s;
    }
  }

  event1();
}

} // extern "C"
