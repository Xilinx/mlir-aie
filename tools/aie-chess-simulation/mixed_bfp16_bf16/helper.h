//===- helper.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//



#include "io_helpers.h"


#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <string_view>
#include "aie_api/aie.hpp"

// block - block size
// size  - length of the input array
// array - the array
// returnArray - the array to be filled with the quantized values
// rounding - 0 for zero, 1 for nearest (tie to even)
// verbose - make some noise
// Quantization of an array of floats to bfp16.
// The return array is structured as follows:
// 1. The first byte is the shared exponent (max exponent of the block).
// 2. The next *block* bytes are the quantized values.
inline std::vector<uint8_t> floatToBfp16(int block, int size, float *array, int rounding = 0) {
  std::vector<uint8_t> res(size * 1.125);

  int mbits = 7;
  int start = 0, end, i, currentIndex = 1;
  unsigned int sign, exp, maxExp;
  unsigned int *p, mantissa;
  uint8_t valueInt8;

  while (true) {
    // decide on the block (starting and ending point)
    end = start + block;
    end = end > size ? size : end;

    // Find max exp
    maxExp = 0;
    for (i = start; i < end; i++) {
      p = (unsigned int *)(array + i);
      exp = *p >> 23;    // Get rid of mantissa
      exp &= 0x000000FF; // Keep the last 8 bit exponent (remove sign)

      maxExp = maxExp < exp ? exp : maxExp;
    }

    // Round each number
    for (i = start; i < end; i++) {
      p = (unsigned int *)(array + i);

      sign = *p & 0x80000000;     // Sign
      exp = *p >> 23;             // Get rid of mantissa
      exp &= 0x000000FF;          // Keep the last 8 bit exponent (remove sign)
      mantissa = *p & 0x007FFFFF; // 23-bit mantissa
      if (exp)
        mantissa |= 0x00800000; // add the implicit for normal value

      if (exp >= 255)
        continue; // Infinity or NaN remains

      // The rouding mode for the mantissa in AIE2p is always truncation
      // Each scalar value is stored in two's complement representation
      mantissa = sign ? ~mantissa + 1 : mantissa;
      // At least erase 23 - mbits + 1 (+1 is for making the implicit bit
      // explicit)
      valueInt8 = mantissa >> (23 - mbits + 1);

      // Note that shifting by more than 32 bits is undefined behavior in C++
      if (maxExp - exp >= 32) {
        valueInt8 = sign ? 0xff : 0x00;
      } else {
        // Perform an arithmetic right shift
        // Again, the rounding mode is truncation for AIE2p
        valueInt8 = static_cast<int8_t>(valueInt8) >> (maxExp - exp);
      }

      res[currentIndex] = valueInt8;
      currentIndex++;
    }
    res[currentIndex - 9] = (uint8_t)maxExp;
    currentIndex++;
    start = end;
    if (start >= size)
      break;
  }

  return res;
}


// Helper to print matrix in required format using C-style FILE*
void print_matrix_float(const char* filename, float* data, int rows, int cols) {
  FILE* fp = open_file(filename, "w+");
  fprintf(fp, "(%d, %d)\n", rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      fprintf(fp, "%f", (float)data[i * cols + j]);
      if (j < cols - 1) fprintf(fp, " ");
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

// Helper to print matrix in required format using C-style FILE*
void print_matrix_bfloat16(const char* filename, bfloat16* data, int rows, int cols) {
  FILE* fp = open_file(filename, "w+");
  fprintf(fp, "(%d, %d)\n", rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      fprintf(fp, "%f", (float)data[i * cols + j]);
      if (j < cols - 1) fprintf(fp, " ");
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}

// Golden result calculation: naive matrix multiplication (float)
void calc_golden_result(const float* A, const float* B, float* C, int M, int K, int N) {
  // C[M x N] = A[M x K] * B[K x N]
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a_val = (float)A[i * K + k];
        float b_val = (float)B[k * N + j];
        if (i == 0 && j == 0 && k < 8) {
          printf("DEBUG: A[0][%d]=%f, B[%d][0]=%f\n", k, a_val, k, b_val);
        }
        sum += a_val * b_val;
      }
      if (i == 0 && j < 8) {
        printf("DEBUG: gold[%d] sum = %f\n", j, sum);
      }
      C[i * N + j] = (float)sum;
    }
  }
}

// Layout transpose function: reorganize 8x8 matrix from row-major to column-major layout
// Input: 8x8 float array, row-major
// Output: 8x8 array in column-major layout
void layout_transpose_8x8block(float* input, float* output, int rows, int cols) {
  
  int output_idx = 0;

  // Process the single 8x8 block in column-major order
  for (int col = 0; col < 8; col++) {
    for (int row = 0; row < 8; row++) {
      // Calculate the position in the original row-major matrix
      int orig_idx = row * 8 + col;
      
      // Copy to output in column-major layout
      output[output_idx++] = input[orig_idx];
    }
  }
}


