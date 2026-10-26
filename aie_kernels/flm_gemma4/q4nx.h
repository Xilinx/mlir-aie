//===- q4nx.h ---------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The q4nx weight block and its matrix-vector accumulation, shared by the
// decode projections and the LM head. Independent of the model geometry.
#ifndef AIE_KERNELS_FLM_GEMMA4_Q4NX_H
#define AIE_KERNELS_FLM_GEMMA4_Q4NX_H

#include "utils.h"

#include <stdint.h>

// One M by N q4nx block: 4-bit codes with a scale and a minimum per group of 32
// input columns. The scales and the minima are indexed [n / 32, m]; the codes
// sit in strips of 16 rows, each strip column-major.
template <int M, int N>
struct q4nx_block {
  bf16 scales[M * N / 32];
  bf16 mins[M * N / 32];
  uint4 qs[M * N];
};

/// c[0:M] += dequantize(A) * B[0:N].
///
/// `b_group_sums` turns each minimum into a single MAC instead of 32 adds.
/// The codes times B sum in an accumulator that is narrowed to bf16 before
/// the scale multiplies it.
template <int M, int N>
void q4nx_accumulate(const q4nx_block<M, N> *A, const bf16 *B, float *c,
                     const bf16 *b_group_sums) {
  constexpr int pr = 16;
  const uint4 *qs_ptr = A->qs;

  AIE_LOOP_RANGE(M / pr, M / pr)
  for (int row = 0; row < M; row += pr) {
    aie::accum<accfloat, pr> c_accum;
    c_accum.from_vector(aie::load_v<pr>(c + row));
    const bf16 *it_B = B;

    uint32_t scale_min_offset = row;
    const bf16 *it_sums = b_group_sums;

    AIE_LOOP_RANGE(N / 32, N / 32)
    for (int sub_chunk = 0; sub_chunk < N / 32; sub_chunk++) {

      aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);
      it_B += 32;

      aie::vector<bf16, pr> a_scales =
          aie::load_v<pr>(A->scales + scale_min_offset);
      aie::vector<bf16, pr> a_mins =
          aie::load_v<pr>(A->mins + scale_min_offset);

      scale_min_offset += M;
      bf16 sum_b = *it_sums;
      it_sums++;

      aie::accum<accfloat, pr> temp_acc;

      // 32 columns = 4 uint4 sub-vectors x 8 lanes each. Both loops are fully
      // unrolled on purpose: the flat 32-long MAC chain is what the scheduler
      // pipelines. A rolled MAC loop serializes the accumulator and drops
      // about 4x the vmac count.
      aie::vector<bf16, pr * 8> a_cc_bf16[4];
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
        aie::vector<uint4, pr * 8> a_cc = aie::load_v<pr * 8>(qs_ptr);
        qs_ptr += pr * 4;
        aie::accum<accfloat, pr * 8> acc;
        acc.from_vector(aie::to_float(a_cc, 0));
        a_cc_bf16[sv] = acc.template to_vector<bf16>();
      }

      temp_acc = aie::mul(a_cc_bf16[0].extract<pr>(0), b_col.get(0));
#pragma clang loop unroll(full)
      for (int sv = 0; sv < 4; sv++) {
#pragma clang loop unroll(full)
        for (int cc = 0; cc < 8; cc++) {
          if (sv == 0 && cc == 0)
            continue;
          temp_acc = aie::mac(temp_acc, a_cc_bf16[sv].extract<pr>(cc),
                              b_col.get(sv * 8 + cc));
        }
      }

      c_accum =
          aie::mac(c_accum, temp_acc.template to_vector<bf16>(), a_scales);
      c_accum = aie::mac(c_accum, a_mins, sum_b);
    }
    aie::store_v(c + row, c_accum.template to_vector<float>());
  }
}

#endif // AIE_KERNELS_FLM_GEMMA4_Q4NX_H
