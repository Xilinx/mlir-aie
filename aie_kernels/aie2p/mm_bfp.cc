//===- mm.cc ----------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

template <int M, int N>
void zero_vectorized_v64bfp16ebs8(bfp16ebs8 *__restrict cOut) {
  int const vectorSize = 64;

  const aie::accum<accfloat, vectorSize> acc =
      aie::zeros<accfloat, vectorSize>();

  aie::block_vector_output_buffer_stream<bfp16ebs8, vectorSize> outStreamC(
      cOut);

  for (int i = 0; i < M * N / 64; i++) {
    outStreamC << acc.to_vector<bfp16ebs8>();
  }
}

// There is a CPU version of this function in the helper.h file
void scalarShuffleMatrixForBfp16ebs8(size_t tileWidth, size_t tileHeight,
                                     uint8_t *inBfpMatrix,
                                     uint8_t *outBfpMatrix,
                                     bool unshuffle = false) {

  tileWidth = tileWidth * 1.125;

  size_t subtileWidth = 8 * 1.125;
  size_t subtileHeight = 8;

  size_t tileCountingIndex = 0;
  for (size_t subtileStartY = 0; subtileStartY < tileHeight;
       subtileStartY += subtileHeight) {
    for (size_t subtileStartX = 0; subtileStartX < tileWidth;
         subtileStartX += subtileWidth) {

      for (size_t i = 0; i < subtileHeight; ++i) {
        for (size_t j = 0; j < subtileWidth; ++j) {
          size_t inputGlobalX = subtileStartX + j;
          size_t inputGlobalY = subtileStartY + i;
          size_t inputIndex = inputGlobalY * tileWidth + inputGlobalX;

          size_t outputGlobalX = tileCountingIndex % tileWidth;
          size_t outputGlobalY = tileCountingIndex / tileWidth;
          size_t outputIndex = outputGlobalY * tileWidth + outputGlobalX;

          if (!unshuffle) {
            outBfpMatrix[outputIndex] = inBfpMatrix[inputIndex];
          } else {
            outBfpMatrix[inputIndex] = inBfpMatrix[outputIndex];
          }
          tileCountingIndex++;
        }
      }
    }
  }
}

// This kernel mirrors the one found in
// https://xilinx.github.io/aie_api/group__group__mmul.html Go through them in
// parallel to understand how the bfp datatype modifies accesses to memory Note
// that this kernel assumes that the B matrix is already transposed, which is
// not the case for the example in the link. The equivalent transformations for
// a non transposed B matrix are commented out below. Also note that assuming
// the 8x8 tile are already transposed (the ones done during the shuffle), the
// higher level tiling transposition should be free using data layout
// transformations.
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s,
          unsigned t>
void matmul_vectorized_2x2_bfp16(const bfp16ebs8 *__restrict pA,
                                 const bfp16ebs8 *__restrict pB,
                                 bfp16ebs8 *__restrict pC) {
  const unsigned sizeA = r * s;
  const unsigned sizeB = s * t;
  const unsigned sizeC = r * t;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned z = 0; z < rowA; z += 2) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pC1In(pC);
    pC1In.seek(z * colB);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pC2In(pC);
    pC2In.seek((z + 1) * colB);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pC1Out(pC);
    pC1Out.seek(z * colB);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pC2Out(pC);
    pC2Out.seek((z + 1) * colB);

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA1bfp16(pA);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA2bfp16(pA);
        pA1bfp16.seek(z * colA);
        pA2bfp16.seek((z + 1) * colA);

        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB1bfp16(pB);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB2bfp16(pB);
        // For non transposed matrix
        // pB1bfp16.seek(j);
        // pB2bfp16.seek(j + 1);
        pB1bfp16.seek(j * colA);
        pB2bfp16.seek((j + 1) * colA);

        aie::block_vector<bfp16ebs8, sizeA> A0;
        aie::block_vector<bfp16ebs8, sizeA> A1;
        aie::block_vector<bfp16ebs8, sizeB> B0;
        aie::block_vector<bfp16ebs8, sizeB> B1;

        // Note that unlike the example mentioned above, we need
        // to use a mac to take into account results from previous kernel
        // calls but this is completely unrelated to the block datatype.
        aie::accum<accfloat, sizeC> accC00(pC1In.pop());
        aie::accum<accfloat, sizeC> accC01(pC1In.pop());
        aie::accum<accfloat, sizeC> accC10(pC2In.pop());
        aie::accum<accfloat, sizeC> accC11(pC2In.pop());

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A0 = pA1bfp16.pop();
            A1 = pA2bfp16.pop();

            // For non transposed matrix
            // B0 = pB1bfp16.pop_seek(colB - 1);
            // B1 = pB2bfp16.pop_seek(colB - 1);
            B0 = pB1bfp16.pop();
            B1 = pB2bfp16.pop();

            accC00 = mac_8x8_8x8T(A0, B0, accC00);
            accC01 = mac_8x8_8x8T(A0, B1, accC01);
            accC10 = mac_8x8_8x8T(A1, B0, accC10);
            accC11 = mac_8x8_8x8T(A1, B1, accC11);
          }

        pC1Out.push(accC00.template to_vector<bfp16ebs8>());
        pC1Out.push(accC01.template to_vector<bfp16ebs8>());
        pC2Out.push(accC10.template to_vector<bfp16ebs8>());
        pC2Out.push(accC11.template to_vector<bfp16ebs8>());
      }
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

void matmul_vectorized_bfp16(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB,
                             bfp16ebs8 *__restrict pC) {

  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  constexpr int m = DIM_M;
  constexpr int k = DIM_K;
  constexpr int n = DIM_N;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  matmul_vectorized_2x2_bfp16<m / r, k / s, n / t, r, s, t>(pA, pB, pC);
}

void zero_kernel(bfp16ebs8 *__restrict cOut) {
  zero_vectorized_v64bfp16ebs8<DIM_M, DIM_N>(cOut);
}

void scalar_shuffle(uint8_t *pA, uint8_t *pC, size_t tileWidth,
                    size_t tileHeight, bool unshuffle = false) {
  scalarShuffleMatrixForBfp16ebs8(tileWidth, tileHeight, pA, pC, unshuffle);
}
}
