//===- mm_fused_mmul.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef __MM_FUSED_MMUL_H__
#define __MM_FUSED_MMUL_H__
#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

// The 2x2 blocked mmul at the heart of mm_fused.h: the j loop keeps four
// accumulators live so each pair of A/B loads feeds four macs (load
// amortization, not unrolling, so j and z step by two). A is one contiguous
// buffer across z slices; B is column-major over blocks, block (i, j) at
// (j * colA + i). Two definitions below, one per B storage format: bfp16ebs8
// (AIE2P only -- AIE2's aie_api has an empty placeholder that won't compile)
// and bf16 (AIE2), since neither compiles on both arches.
#ifdef MM_FUSED_BFP16_B
using mm_fused_b_elem_t = bfp16ebs8;
#else
using mm_fused_b_elem_t = bfloat16;
#endif

// Write one 2x2 block of accumulators back to C. Templated on the accumulator
// type because the two forms hold it differently -- aie::mmul in one,
// aie::accum in the other -- while presenting the same to_vector.
template <unsigned sizeC, typename Acc>
__aie_inline void
mm_fused_store_2x2(float *__restrict pC1, float *__restrict pC2, const Acc &C00,
                   const Acc &C01, const Acc &C10, const Acc &C11) {
  aie::store_v(pC1, C00.template to_vector<float>());
  aie::store_v(pC1 + sizeC, C01.template to_vector<float>());
  aie::store_v(pC2, C10.template to_vector<float>());
  aie::store_v(pC2 + sizeC, C11.template to_vector<float>());
}

#ifdef MM_FUSED_BFP16_B

// B arrives already quantized to bfp16ebs8. The bf16 form below converts B
// inside every mac; since B is static weights, pack_B hoists that (already-
// happened) rounding to the host once, and makes B 9 bytes per 8 elements
// instead of 16 -- the point, as the operator is data-movement bound.
//
// B is streamed, not pointer-indexed: a block_vector cannot be aie::load_v'd,
// and bfp16ebs8 pointer arithmetic counts bytes not blocks. TODO: index
// directly once llvm-aie#1232 (sizeof(bfp16ebs8) == 1, not 9) is fixed.
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s,
          unsigned t>
__aie_inline void mm_fused_mmul_2x2(const bfloat16 *__restrict pA,
                                    const bfp16ebs8 *__restrict pB,
                                    float *__restrict pC) {
  constexpr unsigned sizeA = r * s;
  constexpr unsigned sizeB = s * t;
  constexpr unsigned sizeC = r * t;
  // Unrolling z lets one row pair's last C stores overlap the next pair's
  // first C loads. Leave j rolled: unrolling it costs short i loops more than
  // it saves long ones.
  AIE_LOOP_MAX_ITERATION_COUNT(rowA / 2)
  AIE_LOOP_UNROLL(2)
  for (unsigned z = 0; z < rowA; z += 2) {
    float *__restrict pC1 = pC + (z * colB) * sizeC;
    float *__restrict pC2 = pC + ((z + 1) * colB) * sizeC;
    const bfloat16 *__restrict pA_cur = pA + (z >> 1) * (2 * r * colA * s);

    AIE_LOOP_MAX_ITERATION_COUNT(colB / 2)
    for (unsigned j = 0; j < colB; j += 2) {
      const bfloat16 *__restrict pA1 = pA_cur;
      const bfloat16 *__restrict pA2 = pA_cur + colA * sizeA;

      aie::block_vector_input_buffer_stream<bfp16ebs8, sizeB> pB1(pB);
      aie::block_vector_input_buffer_stream<bfp16ebs8, sizeB> pB2(pB);
      pB1.seek(j * colA);
      pB2.seek((j + 1) * colA);

      aie::accum<accfloat, sizeC> C00(aie::load_v<sizeC>(pC1));
      aie::accum<accfloat, sizeC> C01(aie::load_v<sizeC>(pC1 + sizeC));
      aie::accum<accfloat, sizeC> C10(aie::load_v<sizeC>(pC2));
      aie::accum<accfloat, sizeC> C11(aie::load_v<sizeC>(pC2 + sizeC));

      aie::vector<bfloat16, sizeA> A0;
      aie::vector<bfloat16, sizeA> A1;
      aie::accum<accfloat, sizeA> accA0;
      aie::accum<accfloat, sizeA> accA1;

      // Keep this loop rolled and its body minimal. Extra live state across it
      // costs more than it saves, because Peano's register allocator runs
      // before the post-RA pipeliner and manufactures false loop-carried
      // anti-dependences. TODO: re-measure a hand-unroll once llvm-aie#1066 is
      // fixed; open as of 2026-09-10.
      AIE_LOOP_MAX_ITERATION_COUNT(colA)
      for (unsigned i = 0; i < colA; i++) {
        // One conversion per A operand per i, reused by both j accumulators,
        // rather than one inside each of the four macs. Same values.
        //
        // The two operands are widened by different routes, as mlir-aie's
        // mm_bfp_mixed.cc also does: widening both by assignment makes Peano's
        // AIE2P backend abort with "Use not jointly dominated by defs".
        // mul_elem_64 by one is the same arithmetic and does codegen.
        A0 = aie::load_v<sizeA>(pA1);
        pA1 += sizeA;
        A1 = aie::load_v<sizeA>(pA2);
        pA2 += sizeA;
        accA0 = A0;
        accA1 = mul_elem_64(A1, concat(broadcast_one_to_v32bfloat16(),
                                       broadcast_one_to_v32bfloat16()));

        aie::block_vector<bfp16ebs8, sizeB> B0 = pB1.pop();
        aie::block_vector<bfp16ebs8, sizeB> B1 = pB2.pop();

        C00 = mac_8x8_8x8T(accA0.template to_vector<bfp16ebs8>(), B0, C00);
        C01 = mac_8x8_8x8T(accA0.template to_vector<bfp16ebs8>(), B1, C01);
        C10 = mac_8x8_8x8T(accA1.template to_vector<bfp16ebs8>(), B0, C10);
        C11 = mac_8x8_8x8T(accA1.template to_vector<bfp16ebs8>(), B1, C11);
      }
      mm_fused_store_2x2<sizeC>(pC1, pC2, C00, C01, C10, C11);
      pC1 += 2 * sizeC;
      pC2 += 2 * sizeC;
    }
  }
}

#else

// B arrives as plain bf16, in row-major s x t blocks. Used on AIE2, which has
// no bfp16 hardware and composes this shape from four native 4x8x4 macs.
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s,
          unsigned t>
__aie_inline void mm_fused_mmul_2x2(const bfloat16 *__restrict pA,
                                    const bfloat16 *__restrict pB,
                                    float *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;
  static_assert(r * s == MMUL::size_A);
#if AIE_TUNED_AIE2
  // A pipelined loop of two or four trips is almost all prologue and epilogue.
  // Unrolling i and j instead leaves one straight block per z pair, in which
  // the next block's C loads overlap this one's macs.
  constexpr bool unroll_ij = colA <= 4 && colB <= 4;
#endif
  // Rolled, the z loop does not pipeline and each trip pays the j loop's
  // entry and exit; two trips per body let those overlap.
  AIE_LOOP_MAX_ITERATION_COUNT(rowA / 2)
  AIE_LOOP_UNROLL(2)
  for (unsigned z = 0; z < rowA; z += 2) {
    float *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
    float *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
    const bfloat16 *__restrict pA_cur = pA + (z >> 1) * (2 * r * colA * s);

    aie::vector<bfloat16, MMUL::size_A> A0;
    aie::vector<bfloat16, MMUL::size_A> A1;
    aie::vector<bfloat16, MMUL::size_B> B0;
    aie::vector<bfloat16, MMUL::size_B> B1;

    AIE_LOOP_MAX_ITERATION_COUNT(colB / 2)
#if AIE_TUNED_AIE2
    AIE_LOOP_UNROLL(unroll_ij ? colB / 2 : 1)
#endif
    for (unsigned j = 0; j < colB; j += 2) {
      const bfloat16 *__restrict pA1 = pA_cur;
      const bfloat16 *__restrict pA2 = pA_cur + colA * MMUL::size_A;
      const bfloat16 *__restrict pB1 = pB + (j * colA) * MMUL::size_B;
      const bfloat16 *__restrict pB2 = pB + ((j + 1) * colA) * MMUL::size_B;

      MMUL C00(aie::load_v<MMUL::size_C>(pC1));
      MMUL C01(aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C));
      MMUL C10(aie::load_v<MMUL::size_C>(pC2));
      MMUL C11(aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C));

      // Rolled, for the same reason as the bfp16 form above (llvm-aie#1066).
      AIE_LOOP_MAX_ITERATION_COUNT(colA)
#if AIE_TUNED_AIE2
      AIE_LOOP_UNROLL(unroll_ij ? colA : 1)
#endif
      for (unsigned i = 0; i < colA; i++) {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        A1 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;
        B0 = aie::load_v<MMUL::size_B>(pB1);
        pB1 += MMUL::size_B;
        B1 = aie::load_v<MMUL::size_B>(pB2);
        pB2 += MMUL::size_B;

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }
      mm_fused_store_2x2<MMUL::size_C>(pC1, pC2, C00, C01, C10, C11);
      pC1 += 2 * MMUL::size_C;
      pC2 += 2 * MMUL::size_C;
    }
  }
}

#endif // MM_FUSED_BFP16_B

#endif // __MM_FUSED_MMUL_H__
