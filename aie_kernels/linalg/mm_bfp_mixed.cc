//===- mm.cc ----------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// This kernel is a variation of the conventional matrix multiplications in the
// repo that uses different datatypes for the A and B and performs a conversion
// for the A matrix. This kernel should be followed along with the equivalent on
// in bfp16 only on mm.cc
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s,
          unsigned t>
void matmul_vectorized_2x2_bfp16_bf16(const bfloat16 *__restrict pA,
                                      const bfp16ebs8 *__restrict pB,
                                      bfloat16 *__restrict pC) {
  const unsigned sizeA = r * s;
  const unsigned sizeB = s * t;
  const unsigned sizeC = r * t;

  // The 2x2 output tiles are walked by one loop, the column wrap a select:
  // only the innermost loop is software-pipelined.
  const bfloat16 *__restrict pArow = pA;
  bfloat16 *__restrict pC1 = pC;
  unsigned j = 0;

  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned tile = 0; tile < rowA * colB / 4; ++tile) {
    bfloat16 *__restrict pC2 = pC1 + colB * sizeC;
    const bfloat16 *__restrict pA1 = pArow;
    const bfloat16 *__restrict pA2 = pArow + colA * sizeA;

    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB1bfp16(pB);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB2bfp16(pB);
    // For non transposed matrix
    // pB1bfp16.seek(j);
    // pB2bfp16.seek(j + 1);
    pB1bfp16.seek(j * colA);
    pB2bfp16.seek((j + 1) * colA);

    aie::vector<bfloat16, sizeA> A0;
    aie::vector<bfloat16, sizeA> A1;
    aie::block_vector<bfp16ebs8, sizeB> B0;
    aie::block_vector<bfp16ebs8, sizeB> B1;

    aie::accum<accfloat, sizeC> accC00(aie::load_v<sizeC>(pC1));
    aie::accum<accfloat, sizeC> accC01(aie::load_v<sizeC>(pC1 + sizeC));
    aie::accum<accfloat, sizeC> accC10(aie::load_v<sizeC>(pC2));
    aie::accum<accfloat, sizeC> accC11(aie::load_v<sizeC>(pC2 + sizeC));

    aie::accum<accfloat, 64> accA0;
    aie::accum<accfloat, 64> accA1;

    for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        A0 = aie::load_v<sizeA>(pA1);
        pA1 += sizeA;
        A1 = aie::load_v<sizeA>(pA2);
        pA2 += sizeA;

        // Convert A0 into bfp16
        accA0 = A0;
        // Convert A1 into bfp16 through a different path (see bfp
        // conversion example)
        accA1 = mul_elem_64(A1, concat(broadcast_one_to_v32bfloat16(),
                                       broadcast_one_to_v32bfloat16()));

        // For non transposed matrix
        // B0 = pB1bfp16.pop_seek(colB - 1);
        // B1 = pB2bfp16.pop_seek(colB - 1);
        B0 = pB1bfp16.pop();
        B1 = pB2bfp16.pop();

        accC00 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B0, accC00);
        accC01 = mac_8x8_8x8T(accA0.to_vector<bfp16ebs8>(), B1, accC01);
        accC10 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B0, accC10);
        accC11 = mac_8x8_8x8T(accA1.to_vector<bfp16ebs8>(), B1, accC11);
      }

    aie::store_v(pC1, accC00.template to_vector<bfloat16>());
    aie::store_v(pC1 + sizeC, accC01.template to_vector<bfloat16>());
    aie::store_v(pC2, accC10.template to_vector<bfloat16>());
    aie::store_v(pC2 + sizeC, accC11.template to_vector<bfloat16>());

    j += 2;
    const bool wrap = j == colB;
    j = wrap ? 0 : j;
    pC1 += wrap ? (colB + 2) * sizeC : 2 * sizeC;
    pArow += wrap ? 2 * colA * sizeA : 0;
  }
}

extern "C" {

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

void matmul_vectorized_different_datatypes(bfloat16 *__restrict pA,
                                           bfp16ebs8 *__restrict pB,
                                           bfloat16 *__restrict pC) {
  event0();

  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  constexpr int m = DIM_M;
  constexpr int k = DIM_K;
  constexpr int n = DIM_N;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  // A is converted to bfp16 on the core, and that conversion follows the
  // core's rounding mode rather than being fixed here: conv_even keeps the K
  // reduction unbiased, where floor costs a 64x64x64 tile of large inputs
  // 2791 mismatching outputs against 10. The caller owns the mode --
  // kernels.mm_bfp names conv_even as its contract's setup -- so setting it
  // here too would be the design and the kernel fighting over it.
  matmul_vectorized_2x2_bfp16_bf16<m / r, k / s, n / t, r, s, t>(pA, pB, pC);
  event1();
}
}
