//===- yolo_m9_sv_row_acc.cc ---------------------------------*- C++ -*-===//
//
// sv_row variant that writes into a (N, head_dim) accumulator buffer
// instead of a (chunk_cols, head_dim) chunk. Same math as sv_row but
// the output indexing treats the destination as a per-sample accumulator
// indexed by absolute query_idx (0..N-1) rather than chunk-relative
// chunk_row.
//
// Used in stage 7's Phase 2a where sv outputs for head 0 are accumulated
// across all 16 chunks before being read in head-1+pe-add phase.
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

void yolo_m9_sv_row_acc_i8_i8(
    int8_t *v_frame,            // (head_dim, N) per-head V
    int8_t *attn_chunk,         // (chunk_rows, N) softmaxed scores chunk
    int8_t *acc_out,            // (N, head_dim) per-sample accumulator
    const int32_t chunk_row,    // which row of attn_chunk to read
    const int32_t n_global,     // absolute query_idx in [0, N) for acc_out
    const int32_t head_dim,
    const int32_t N,
    const int32_t right_shift) {
  event0();

  int8_t *attn_row = attn_chunk + chunk_row * N;
  int8_t *out_col = acc_out + n_global * head_dim;
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
