//===- yolo_m9_qk_row.cc -------------------------------------*- C++ -*-===//
//
// Scalar i8 row-wise matmul kernel for one (head, query_row) of the PSA
// attention. Given a per-head Q+K buffer laid out as
//   qk_frame[s, n], s in [0, 2*kd), n in [0, N)
// with Q in rows [0, kd), K in rows [kd, 2*kd), computes one row of the
// scores matrix:
//   scores_row[j] = SRS_i8( sum_{k=0..kd-1} Q[k, query_idx] * K[k, j], rs )
//
// for j in [0, N). Output is i8; downstream softmax consumes it. Scalar
// implementation; aie::mmul vectorization (8x8x8 i8 via mm.cc) is a
// follow-up after the scalar version is bit-exact on HW.
//
// Symbol is m9-specific; m9 is the sole consumer.
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

void yolo_m9_qk_row_i8_i8(
    int8_t *qk_frame,         // (2*kd, N) per-head Q || K (rows 0..kd-1 = Q, kd..2*kd-1 = K)
    int8_t *chunk_out,        // (chunk_rows, N) destination chunk
    const int32_t chunk_row,  // which row of chunk_out to write
    const int32_t kd,         // 32
    const int32_t N,          // 256
    const int32_t query_idx,  // 0..N-1: row of scores we are computing
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  int8_t *scores_row = chunk_out + chunk_row * N;
  for (int j = 0; j < N; j++) {
    int32_t sum = 0;
    for (int k = 0; k < kd; k++) {
      const int8_t q = qk_frame[k * N + query_idx];
      const int8_t kv = qk_frame[(kd + k) * N + j];
      sum += (int32_t)q * (int32_t)kv;
    }
    int32_t s = banker_srs(sum, right_shift);
    s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
    scores_row[j] = (int8_t)s;
  }

  event1();
}

} // extern "C"
