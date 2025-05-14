//===- kernel.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

// This is a workaround for a missing definition in peano.
// Will have to be removed when the issue is fixed.
#ifdef PEANO_UNDEF_WORKAROUND
inline __attribute__((always_inline)) v64bfp16ebs8 undef_v64bfp16ebs8() { return v64bfp16ebs8(); };
#endif

extern "C" {

// Note that bfp vector types are referenced differently in IRON and in the AIE API.
// The v64bfp16ebs8 type is a vector of 8 bfp16ebs8 elements, which corresponds
// to 64 floating points in total.
// Meanwhile, IRON interprets 1 bfp16ebs8 as 8 floating points.
void bfp16_matrix_multiplication(bfp16ebs8 *__restrict inA,
                                 bfp16ebs8 *__restrict inB,
                                 bfp16ebs8 *__restrict out) {
  // We cannot load the bfp16ebs8 vectors directly doing this:
  // aie::vector<bfloat16, 64> matrixA = aie::load_v<64>(inA);
  // The reason is that the bfp16ebs8 type is not necessarily aligned and the
  // memory interface must be adapted through the use of buffer streams for both
  // input and output.
  aie::block_vector<bfp16ebs8, 64> matrixA;
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> inStreamA(inA);
  inStreamA >> matrixA;

  aie::block_vector<bfp16ebs8, 64> matrixB;
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> inStreamB(inB);
  inStreamB >> matrixB;

  aie::accum<accfloat, 64> res = mul_8x8_8x8T(matrixA, matrixB);

  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> outStream(out);
  outStream << res.to_vector<bfp16ebs8>();
}

void bf16_to_bfp_conversion(bfloat16 *__restrict inA, bfloat16 *__restrict inB,
                            bfp16ebs8 *__restrict outA,
                            bfp16ebs8 *__restrict outB) {
  // Here there are two conversion paths for bf16 to bfp16ebs8. This split is
  // meant to optimize resource usage in the inner loop, since we would
  // otherwise be using the same slot in the core.

  // The conversion for inA to bfp16 is carried out from an accumulator
  aie::vector<bfloat16, 64> vecA = aie::load_v<64>(inA);
  aie::accum<accfloat, 64> accA(vecA);

  // The conversion for inB to bfp16 is carried out through an element-wise
  // multiplication with ones
  aie::vector<bfloat16, 64> vecB = aie::load_v<64>(inB);
  // Porperly aligning dot products between blocks for a matrix multiplication
  // implies transposing before the conversion!
  aie::vector<bfloat16, 64> vecBT =
      aie::detail::transpose<bfloat16, 64>::run(vecB, 8, 8);
  aie::accum<accfloat, 64> accB =
      mul_elem_64(vecBT, concat(broadcast_one_to_v32bfloat16(),
                                broadcast_one_to_v32bfloat16()));

  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> outStreamA(outA);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> outStreamB(outB);
  outStreamA << accA.to_vector<bfp16ebs8>();
  outStreamB << accB.to_vector<bfp16ebs8>();
}

} // extern "C"
