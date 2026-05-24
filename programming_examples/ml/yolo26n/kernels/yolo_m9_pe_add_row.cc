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
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

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

  // For each output channel c in [0, c_total), compute pe[c, x] for
  // x in [0, in_w) using dw3x3 with zero-padding on the (y_idx, x)
  // grid. Combine with the per-channel sv contribution and emit.
  for (int c = 0; c < c_total; c++) {
    const int32_t binit = pe_bias[c];
    int8_t *v_chans = (c < head_dim) ? v_h0 : v_h1;
    const int32_t v_c = (c < head_dim) ? c : (c - head_dim);
    int8_t *v_row = &v_chans[v_c * N];

    int8_t *pe_w_c = &pe_wts[c * 9];

    for (int x = 0; x < in_w; x++) {
      int32_t acc = binit;
      for (int ky = 0; ky < 3; ky++) {
        int32_t yy = y_idx + ky - 1;
        if (yy < 0 || yy >= in_h)
          continue; // zero pad top/bottom
        for (int kx = 0; kx < 3; kx++) {
          int32_t xx = x + kx - 1;
          if (xx < 0 || xx >= in_w)
            continue; // zero pad left/right
          int8_t v_val = v_row[yy * in_w + xx];
          int8_t w_val = pe_w_c[ky * 3 + kx];
          acc += (int32_t)v_val * (int32_t)w_val;
        }
      }
      int32_t pe_q = banker_srs(acc, rs_pe);
      if (pe_q > I8_MAX)
        pe_q = I8_MAX;
      if (pe_q < I8_MIN)
        pe_q = I8_MIN;

      // Pull sv contribution for this (c, x) at y=y_idx.
      int8_t sv_val;
      if (c < head_dim) {
        sv_val = sv_h0_acc[(y_idx * in_w + x) * head_dim + c];
      } else {
        sv_val = sv_h1_row[x * head_dim + (c - head_dim)];
      }

      int32_t s = (int32_t)sv_val + (int32_t)pe_q;
      if (s > I8_MAX)
        s = I8_MAX;
      if (s < I8_MIN)
        s = I8_MIN;
      chunk_out[x * c_total + c] = (int8_t)s;
    }
  }

  event1();
}

} // extern "C"
