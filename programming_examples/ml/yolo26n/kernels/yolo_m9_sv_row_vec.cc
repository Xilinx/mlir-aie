//===- yolo_m9_sv_row_vec.cc ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// Vectorized sv-matmul column kernel for the PSA attention. Drop-in
// .o-level replacement for yolo_m9_sv_row.cc — exports BOTH
// `yolo_m9_sv_row_i8_i8` and `yolo_m9_sv_row_acc_i8_i8` (same body, one
// difference: the dst pointer + per-call output offset). The Makefile
// builds this source once as yolo_m9_sv_row.o; m9_stage.py references
// that same .o for both Kernel decls.
//
// For a single (head, n) output column, computes:
//   out[c, n] = SRS_i8( sum_{m=0..N-1} V[c, m] * attn[n, m], rs_sv )
// for c in [0, head_dim).
//
// V is (head_dim, N) row-major and attn_row is (N,) — both rows are
// contiguous in m, so this maps cleanly to vector-mac across 16-lane
// strips of m. A 4-way c-fold reuses each attn-strip load across 4 V
// rows (4 acc accumulators); per-c reduce + SRS + store closes out the
// scalar tail.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../../../../aie_kernels/aie_kernel_utils.h"

#ifndef YOLO_M9_SV_HEAD_DIM
#error "YOLO_M9_SV_HEAD_DIM must be defined at compile time"
#endif
#ifndef YOLO_M9_SV_N
#error "YOLO_M9_SV_N must be defined at compile time"
#endif

static constexpr int kHeadDim = YOLO_M9_SV_HEAD_DIM;
static constexpr int kN = YOLO_M9_SV_N;
static constexpr int kMVec = 16;
static constexpr int kCBlock = 4;

static_assert(kHeadDim % kCBlock == 0, "SV HEAD_DIM must be multiple of 4");
static_assert(kN % kMVec == 0, "SV N must be multiple of 16");

static constexpr int kCBlocks = kHeadDim / kCBlock;
static constexpr int kMGroups = kN / kMVec;

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

static __attribute__((always_inline)) inline void
sv_row_body(int8_t *v_frame, const int8_t *__restrict attn_row,
            int8_t *__restrict out_col, int32_t right_shift) {
  ::aie::set_saturation(aie::saturation_mode::saturate);
  // conv_even matches scalar banker_srs (round-half-to-even) for any
  // future vec to_vector<int8>(rs) use. Scalar banker_srs() in this
  // file is unaffected by the rounding mode setting.
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  AIE_LOOP_RANGE(kCBlocks, kCBlocks)
  for (int cb = 0; cb < kCBlocks; ++cb) {
    aie::accum<acc32, kMVec> acc0, acc1, acc2, acc3;
    acc0.from_vector(aie::zeros<int32, kMVec>());
    acc1.from_vector(aie::zeros<int32, kMVec>());
    acc2.from_vector(aie::zeros<int32, kMVec>());
    acc3.from_vector(aie::zeros<int32, kMVec>());

    const int c_base = cb * kCBlock;
    const int8_t *__restrict v0 = v_frame + (c_base + 0) * kN;
    const int8_t *__restrict v1 = v_frame + (c_base + 1) * kN;
    const int8_t *__restrict v2 = v_frame + (c_base + 2) * kN;
    const int8_t *__restrict v3 = v_frame + (c_base + 3) * kN;

    AIE_LOOP_RANGE(kMGroups, kMGroups)
    for (int g = 0; g < kMGroups; ++g) {
      const int m_base = g * kMVec;
      aie::vector<int8, kMVec> a_v = aie::load_v<kMVec>(attn_row + m_base);
      aie::vector<int8, kMVec> v0_v = aie::load_v<kMVec>(v0 + m_base);
      aie::vector<int8, kMVec> v1_v = aie::load_v<kMVec>(v1 + m_base);
      aie::vector<int8, kMVec> v2_v = aie::load_v<kMVec>(v2 + m_base);
      aie::vector<int8, kMVec> v3_v = aie::load_v<kMVec>(v3 + m_base);
      acc0 = aie::mac(acc0, v0_v, a_v);
      acc1 = aie::mac(acc1, v1_v, a_v);
      acc2 = aie::mac(acc2, v2_v, a_v);
      acc3 = aie::mac(acc3, v3_v, a_v);
    }

    int32_t s0 = aie::reduce_add(acc0.template to_vector<int32>());
    int32_t s1 = aie::reduce_add(acc1.template to_vector<int32>());
    int32_t s2 = aie::reduce_add(acc2.template to_vector<int32>());
    int32_t s3 = aie::reduce_add(acc3.template to_vector<int32>());
    int32_t r0 = banker_srs(s0, right_shift);
    int32_t r1 = banker_srs(s1, right_shift);
    int32_t r2 = banker_srs(s2, right_shift);
    int32_t r3 = banker_srs(s3, right_shift);
    if (r0 > I8_MAX)
      r0 = I8_MAX;
    if (r0 < I8_MIN)
      r0 = I8_MIN;
    if (r1 > I8_MAX)
      r1 = I8_MAX;
    if (r1 < I8_MIN)
      r1 = I8_MIN;
    if (r2 > I8_MAX)
      r2 = I8_MAX;
    if (r2 < I8_MIN)
      r2 = I8_MIN;
    if (r3 > I8_MAX)
      r3 = I8_MAX;
    if (r3 < I8_MIN)
      r3 = I8_MIN;
    out_col[c_base + 0] = (int8_t)r0;
    out_col[c_base + 1] = (int8_t)r1;
    out_col[c_base + 2] = (int8_t)r2;
    out_col[c_base + 3] = (int8_t)r3;
  }
}

extern "C" {

// Chunk-relative variant: writes into chunk_out[n_in_chunk * head_dim, :].
void yolo_m9_sv_row_i8_i8(int8_t *v_frame, int8_t *attn_chunk,
                          int8_t *chunk_out, const int32_t chunk_row,
                          const int32_t n_in_chunk, const int32_t /*head_dim*/,
                          const int32_t /*N*/, const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();
  sv_row_body(v_frame, attn_chunk + chunk_row * kN,
              chunk_out + n_in_chunk * kHeadDim, right_shift);
  event1();
}

// Absolute variant: writes into acc_out[n_global * head_dim, :]. Same
// math as sv_row, only the output offset differs.
void yolo_m9_sv_row_acc_i8_i8(int8_t *v_frame, int8_t *attn_chunk,
                              int8_t *acc_out, const int32_t chunk_row,
                              const int32_t n_global,
                              const int32_t /*head_dim*/, const int32_t /*N*/,
                              const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();
  sv_row_body(v_frame, attn_chunk + chunk_row * kN,
              acc_out + n_global * kHeadDim, right_shift);
  event1();
}

} // extern "C"
