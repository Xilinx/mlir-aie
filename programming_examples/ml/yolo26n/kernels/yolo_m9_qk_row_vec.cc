//===- yolo_m9_qk_row_vec.cc ----------------------------------*- C++ -*-===//
//
// Vectorized i8 row-wise matmul kernel for one (head, query_row) of the
// PSA attention. Drop-in .o-level replacement for yolo_m9_qk_row.cc.
//
// qk_frame is (2*kd, N) row-major: rows [0..kd) hold Q, rows [kd..2*kd)
// hold K. For the given query_idx, computes one row of N scores:
//   scores_row[j] = SRS_i8( sum_{k=0..kd-1} Q[k, query_idx] * K[k, j], rs )
//
// Mmul-style tiling is awkward here because the M=1 query batch is
// strided across qk_frame's rows. So this is a vector-broadcast reduction
// instead: tile j in groups of 16, keep an aie::accum<acc32, 16> across
// the kd inner loop, broadcast each Q[k] to a 16-wide vector and vmac
// against the K[k, j_base..j_base+15] strip.
//
// Per row: 16 j-groups × 32 k = 512 vmacs instead of scalar's 256*32 =
// 8192 scalar MACs. Bias not used (qk has no bias). aie::mmul deep-opt
// pass is a follow-up that would also need a chunk-batched API change
// + qk_pack producing an mmul-friendly layout.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef YOLO_M9_QK_KD
#error "YOLO_M9_QK_KD must be defined at compile time"
#endif
#ifndef YOLO_M9_QK_N
#error "YOLO_M9_QK_N must be defined at compile time"
#endif

static constexpr int kKd      = YOLO_M9_QK_KD;
static constexpr int kN       = YOLO_M9_QK_N;
static constexpr int kJVec    = 16;
static constexpr int kJGroups = kN / kJVec;

static_assert(kKd > 0,            "QK_KD must be > 0");
static_assert(kN % kJVec == 0,    "QK_N must be a multiple of 16");

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void yolo_m9_qk_row_i8_i8(
    int8_t *qk_frame,
    int8_t *chunk_out,
    const int32_t chunk_row,
    const int32_t /*kd*/,
    const int32_t /*N*/,
    const int32_t query_idx,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  int8_t *__restrict scores_row = chunk_out + chunk_row * kN;
  const int8_t *__restrict q_col = qk_frame + query_idx;        // stride kN per k
  const int8_t *__restrict k_base = qk_frame + kKd * kN;        // K rows start here

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  AIE_LOOP_RANGE(kJGroups, kJGroups)
  for (int g = 0; g < kJGroups; ++g) {
    aie::accum<acc32, kJVec> acc;
    acc.from_vector(aie::zeros<int32, kJVec>());

    const int8_t *__restrict k_strip = k_base + g * kJVec;       // (kd, kJVec) strip

    AIE_LOOP_RANGE(kKd, kKd)
    for (int k = 0; k < kKd; ++k) {
      const int8_t q_scalar = q_col[k * kN];
      aie::vector<int8, kJVec> q_v = aie::broadcast<int8, kJVec>(q_scalar);
      aie::vector<int8, kJVec> k_v = aie::load_v<kJVec>(k_strip + k * kN);
      acc = aie::mac(acc, q_v, k_v);
    }

    aie::vector<int32, kJVec> sum_v = acc.template to_vector<int32>();
    int8_t *__restrict out_p = scores_row + g * kJVec;
    AIE_LOOP_UNROLL_FULL
    for (int i = 0; i < kJVec; ++i) {
      int32_t sr = banker_srs(sum_v[i], right_shift);
      if (sr > I8_MAX) sr = I8_MAX;
      if (sr < I8_MIN) sr = I8_MIN;
      out_p[i] = (int8_t)sr;
    }
  }

  event1();
}

} // extern "C"
