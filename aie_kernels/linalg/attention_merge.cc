//===- attention_merge.cc ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attention over several windows of keys, merged a window per call: window j
// gives a row its normalized output O_j and its online softmax's state, the
// running max m_j (an unscaled score) and the sum l_j. Over the windows so
// far, with M the rows' largest m_j,
//
//   w_j = l_j 2^((m_j - M) s),   O = sum_j w_j O_j / sum_j w_j,
//
// s the exponents' scale (mha.cc's). The running max, sum and unnormalized
// sum of w_j O_j carry in the core's memory from one call to the next and
// restart at window 0, so one core merges a tile's windows in order. A row no
// window kept (every l_j 0) is 0. Every float32 product is taken from bf16
// limbs (f32_split.h) and 2^x from the accurate exp2f, so the merge is
// float32's before O's one bf16 rounding.
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include "../common/exp2f.h"
#include "../common/f32_split.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef MERGE_ROWS
#define MERGE_ROWS 32
#endif
#ifndef MERGE_DIM
#define MERGE_DIM 64
#endif

static_assert(MERGE_ROWS % f32_lanes == 0, "rows are whole 32-lane vectors");
static_assert(MERGE_DIM % f32_lanes == 0, "a row is whole 32-lane vectors");
static_assert(EXP2F_VEC_LEN == f32_lanes, "exp2f_v takes one vector of rows");

using vf = aie::vector<float, f32_lanes>;
using vi = aie::vector<int32_t, f32_lanes>;

alignas(aie::vector_decl_align) static float merged[MERGE_ROWS * MERGE_DIM];
alignas(aie::vector_decl_align) static float running_max[MERGE_ROWS];
alignas(aie::vector_decl_align) static float running_sum[MERGE_ROWS];

// 1 / x for positive normal float32 x: its exponent taken out, inv_1_2 of the
// rest, and the exponent taken back off the result's bits.
static inline vf inv_f32(vf x) {
  const vi bits = x.cast_to<int32_t>();
  const vi exponent =
      aie::bit_and(bits, aie::broadcast<int32_t, f32_lanes>(0x7f800000));
  const vi mantissa = aie::bit_or(
      aie::bit_and(bits, aie::broadcast<int32_t, f32_lanes>(0x007fffff)),
      aie::broadcast<int32_t, f32_lanes>(0x3f800000));
  const vf r = inv_1_2(f32_acc(mantissa.cast_to<float>())).to_vector<float>();
  return aie::sub(r.cast_to<int32_t>(),
                  aie::sub(exponent,
                           aie::broadcast<int32_t, f32_lanes>(0x3f800000)))
      .cast_to<float>();
}

// 2^((x - top) s): x - top is exact, its limbs times the bf16 s exact products.
static inline vf weight(vf x, vf top, bf16_vec s) {
  const Limbs<f32_lanes> dl = bf16_limbs(f32_acc(aie::sub(x, top)));
  f32_acc e = aie::mul(dl.hi, s);
  e = aie::mac(e, dl.mid, s);
  e = aie::mac(e, dl.lo, s);
  return exp2f_v(e.to_vector<float>());
}

// Row i's scalar as three limbs. Split here rather than stored as limbs:
// AIE2's Peano asserts on bf16 limbs read back from memory by row
// (next/issue-draft-peano-accfloat-limbs.md), and the split is exact.
static inline Limbs<f32_lanes> row_limbs(const float *v, int i) {
  return bf16_limbs(f32_acc(aie::broadcast<float, f32_lanes>(v[i])));
}

extern "C" {

// o: MERGE_ROWS x MERGE_DIM bf16, window `window`'s output; state: its
// MERGE_ROWS maxima m then MERGE_ROWS sums l, float32; out: MERGE_ROWS x
// MERGE_DIM bf16, the merge of windows 0 to `window`.
void attention_merge_bf16(const bfloat16 *__restrict o,
                          const float *__restrict state,
                          bfloat16 *__restrict out, int32_t window,
                          bfloat16 scale) {
  event0();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const bf16_vec s = aie::broadcast<bfloat16, f32_lanes>(scale);
  const vf zeros = aie::zeros<float, f32_lanes>();
  alignas(aie::vector_decl_align) float rescale[f32_lanes];
  alignas(aie::vector_decl_align) float w[f32_lanes];
  alignas(aie::vector_decl_align) float inv[f32_lanes];
  for (int g = 0; g < MERGE_ROWS; g += f32_lanes) {
    const vf m = aie::load_v<f32_lanes>(state + g);
    const vf l = aie::load_v<f32_lanes>(state + MERGE_ROWS + g);
    const vf before =
        window == 0 ? m : aie::load_v<f32_lanes>(running_max + g);
    const vf top = aie::max(before, m);
    // The first window starts from nothing: no running state to rescale.
    const vf a = window == 0 ? zeros : weight(before, top, s);
    const vf held =
        window == 0 ? zeros : aie::load_v<f32_lanes>(running_sum + g);
    const f32_acc wj = mac_limbs(f32_acc(zeros), bf16_limbs(f32_acc(l)),
                                 bf16_limbs(f32_acc(weight(m, top, s))));
    const vf total = mac_limbs(wj, bf16_limbs(f32_acc(held)),
                               bf16_limbs(f32_acc(a)))
                         .to_vector<float>();
    aie::store_v(running_max + g, top);
    aie::store_v(running_sum + g, total);
    aie::store_v(rescale, a);
    aie::store_v(w, wj.to_vector<float>());
    // A row no window kept sums to +0 and stays 0.
    aie::store_v(inv, aie::select(inv_f32(total), zeros,
                                  aie::eq(total.cast_to<int32_t>(),
                                          aie::zeros<int32_t, f32_lanes>())));
    for (int i = 0; i < f32_lanes; i++) {
      const int row = g + i;
      const Limbs<f32_lanes> ra = row_limbs(rescale, i);
      const Limbs<f32_lanes> rw = row_limbs(w, i);
      const Limbs<f32_lanes> ri = row_limbs(inv, i);
      for (int c = 0; c < MERGE_DIM; c += f32_lanes) {
        float *acc = merged + row * MERGE_DIM + c;
        const bf16_vec v = aie::load_v<f32_lanes>(o + row * MERGE_DIM + c);
        f32_acc next =
            window == 0
                ? f32_acc(zeros)
                : mac_limbs(f32_acc(zeros),
                            bf16_limbs(f32_acc(aie::load_v<f32_lanes>(acc))),
                            ra);
        next = aie::mac(next, v, rw.hi);
        next = aie::mac(next, v, rw.mid);
        next = aie::mac(next, v, rw.lo);
        aie::store_v(acc, next.to_vector<float>());
        aie::store_v(out + row * MERGE_DIM + c,
                     mac_limbs(f32_acc(zeros), bf16_limbs(next), ri)
                         .to_vector<bfloat16>());
      }
    }
  }
  aie::set_rounding(saved);
  event1();
}

} // extern "C"
