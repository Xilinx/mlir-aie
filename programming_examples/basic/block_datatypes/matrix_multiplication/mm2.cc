//===- mm.cc ----------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

template <typename T, int M, int N>
void zero_vectorized(T *__restrict c) {
  constexpr int r = 512 / (sizeof(T) * 8);
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}

// This kernel is a variation of the conventional matrix multiplications in the repo that uses
// different datatypes for the A and B and performs a conversion for the A matrix.
// This kernel should be followed along with the equivalent on in bfp16 only on mm.cc
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
void matmul_vectorized_2x2_bfp16_bf16(const bfloat16 *__restrict pA, const bfp16ebs8 *__restrict pB,
                                      bfloat16 *__restrict pC) {
  const unsigned sizeA = r * s;
  const unsigned sizeB = s * t;
  const unsigned sizeC = r * t;

  for (unsigned z = 0; z < rowA; z += 2) chess_loop_range(2,) {
    bfloat16 *__restrict pC1 = pC + (z * colB + 0) * sizeC;
    bfloat16 *__restrict pC2 = pC + ((z + 1) * colB + 0) * sizeC;

    for (unsigned j = 0; j < colB; j += 2) chess_loop_range(2,) {
      const bfloat16 *__restrict pA1 = pA + (z * colA + 0) * sizeA;
      const bfloat16 *__restrict pA2 = pA + ((z + 1) * colA + 0) * sizeA;

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

      for (unsigned i = 0; i < colA; ++i) chess_prepare_for_pipelining chess_loop_range(4,) {
        A0 = aie::load_v<sizeA>(pA1);
        pA1 += sizeA;
        A1 = aie::load_v<sizeA>(pA2);
        pA2 += sizeA;

        // Convert A0 into bfp16
        accA0 = A0;
        // Convert A1 into bfp16 through a different path (see bfp conversion example)
        accA1 =
            mul_elem_64(A1, concat(broadcast_one_to_v32bfloat16(), broadcast_one_to_v32bfloat16()));

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
      pC1 += sizeC;
      aie::store_v(pC1, accC01.template to_vector<bfloat16>());
      pC1 += sizeC;
      aie::store_v(pC2, accC10.template to_vector<bfloat16>());
      pC2 += sizeC;
      aie::store_v(pC2, accC11.template to_vector<bfloat16>());
      pC2 += sizeC;
    }
  }
}

extern "C" {

void matmul_vectorized_different_datatypes(bfloat16 *__restrict pA, bfp16ebs8 *__restrict pB,
                                           bfloat16 *__restrict pC) {

  matmul_vectorized_2x2_bfp16_bf16<8, 8, 8, 8, 8, 8>(pA, pB, pC);
}


void zero_kernel_bf16(bfloat16 *__restrict cOut) { zero_vectorized<bfloat16, 64, 64>(cOut); }
}
