//===- yolo_m9_sv_row.cc -------------------------------------*- C++ -*-===//
//
// Scalar i8 sv-matmul column kernel for the PSA attention. For a single
// (head, query_idx=n) output column, computes:
//     out[c, n] = SRS_i8( sum_{m=0..N-1} V[c, m] * attn[n, m], rs_sv )
// for c in [0, head_dim). Equivalent to V (head_dim, N) @ attn[n].T (N,)
// scalar dot products.
//
// attn is the softmaxed scores row for one (head, n), already i8 at
// scale 2^softmax_out_log2 (=-7 for m9). V is i8 at scale 2^qkv_out_log2.
// The matmul-output QL has its own right_shift; for m9 that's
// rs_sv = 6 (from manifest /model.9/m/m.0/attn/MatMul_1.right_shift).
//
// Writes one (head_dim,) column into chunk_out at column `n_in_chunk`
// (since chunks may bundle multiple n values to amortize shim DMA).
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

void yolo_m9_sv_row_i8_i8(
    int8_t *v_frame,            // (head_dim, N) per-head V
    int8_t *attn_chunk,         // (chunk_rows, N) inbound softmaxed-scores chunk
    int8_t *chunk_out,          // (chunk_cols, head_dim) destination chunk
    const int32_t chunk_row,    // which row of attn_chunk to read
    const int32_t n_in_chunk,   // which column of chunk_out to write
    const int32_t head_dim,
    const int32_t N,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  int8_t *attn_row = attn_chunk + chunk_row * N;
  int8_t *out_col = chunk_out + n_in_chunk * head_dim;
  for (int c = 0; c < head_dim; c++) {
    int32_t sum = 0;
    for (int m = 0; m < N; m++) {
      sum += (int32_t)v_frame[c * N + m] * (int32_t)attn_row[m];
    }
    int32_t s = banker_srs(sum, right_shift);
    if (s > I8_MAX) s = I8_MAX;
    if (s < I8_MIN) s = I8_MIN;
    out_col[c] = (int8_t)s;
  }

  event1();
}

} // extern "C"
