//===- helper.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <bitset>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <vector>

inline std::string toBinaryString(int8_t n) {
  std::bitset<8> bits(static_cast<uint8_t>(n));
  std::string binary_str = bits.to_string();
  binary_str.insert(4, " ");

  return binary_str;
}

// Helper function to generate random floating point numbers with high exponent
// variance (useful for blocked datatypes). Exponents are interpreted as base 2
inline float generateRandomFloatingPoint(std::mt19937 &eng, double minExp,
                                         double maxExp) {
  std::uniform_real_distribution<float> distrExp(minExp, maxExp);
  float exponent = distrExp(eng);

  std::uniform_real_distribution<float> distrMantissa(0.0, 1.0);
  float mantissa = distrMantissa(eng);

  return mantissa * std::pow(2.0, exponent);
}

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
inline std::vector<uint8_t> floatToBfp16(int block, int size, float *array,
                                         int rounding = 0) {
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

// Convert a bfp16 array to a float.
// Size should be the number of bytes in the input bfp16 array
inline std::vector<float> bfp16ebs8ToFloat(int size, uint8_t *array,
                                           int verbose = 0) {
  std::vector<float> res(size / 1.125);

  int block = 8;
  int tempIndx = 0;
  for (int i = 0; i < size; i += block + 1) {
    uint8_t sharedExponent = (uint8_t)array[i];
    float multiplier;
    if (sharedExponent >= 127) {
      multiplier = 1.0 * (1 << (sharedExponent - 127));
    } else {
      multiplier = 1.0 / (1 << (127 - sharedExponent));
    }
    multiplier /= 64.0;
    if (verbose) {
      printf("shared_exponent = %d\n", sharedExponent);
      printf("multiplier = %f\n", multiplier);
    }
    for (int j = 1; j < block + 1; j++) {
      bool negative = array[i + j] & 0x80;
      if (negative) {
        // Two's complement for negative numbers
        uint8_t decoded = ~(array[i + j] - 1);
        res[tempIndx] = float(decoded) * multiplier;
      } else {
        res[tempIndx] = float(array[i + j] * multiplier);
      }
      res[tempIndx] = negative ? -res[tempIndx] : res[tempIndx];
      if (verbose) {
        printf("return_array[%d] = %f\n", tempIndx, res[tempIndx]);
      }
      tempIndx++;
    }
  }

  return res;
}

// Shuffle tiles of 64x64 elements for the matrix
// Width and height are expected to be the number of scalar elements in the
// matrix This function rearranges the 8x8 subtiles into rows so that a single
// subtile is contiguous in memory within each tile.
inline std::vector<uint8_t> shuffleMatrixForBfp16ebs8(
    size_t matrixWidth, size_t matrixHeight, size_t tileWidth,
    size_t tileHeight, std::vector<uint8_t> bfpMatrix, bool unshuffle = false) {
  assert(matrixWidth % tileWidth == 0 &&
         "Matrix width must be divisible by tile width");
  assert(matrixHeight % tileHeight == 0 &&
         "Matrix height must be divisible by tile height");
  assert(tileWidth % 64 == 0 && "Tile width must be a multiple of 64");
  assert(tileHeight % 8 == 0 && "Tile height must be a multiple of 8");
  assert(bfpMatrix.size() == (size_t)matrixWidth * matrixHeight * 1.125 &&
         "Matrix size must be width*height*1.125");

  matrixWidth = matrixWidth * 1.125;
  std::vector<uint8_t> res(matrixWidth * matrixHeight);

  tileWidth = tileWidth * 1.125;

  size_t subtileWidth = 8 * 1.125;
  size_t subtileHeight = 8;

  // The main idea is that inputGlobal X and Y are traversing the input matrix
  // in the order we want the elements to be accessed by the core, while
  // outputGlobal X and Y are traversing the tiles in the way they are going to
  // be sent to the accelerator. Essentially, outputGlobal X and Y are just
  // traversing the tiles themselves as if they were contiguous and then going
  // to the next tile.

  // Iterate over the tiles in the matrix
  for (size_t tileStartY = 0; tileStartY < matrixHeight;
       tileStartY += tileHeight) {
    for (size_t tileStartX = 0; tileStartX < matrixWidth;
         tileStartX += tileWidth) {

      size_t tileCountingIndex = 0;
      // Iterate over the subtiles in each tile
      for (size_t subtileStartY = 0; subtileStartY < tileHeight;
           subtileStartY += subtileHeight) {
        for (size_t subtileStartX = 0; subtileStartX < tileWidth;
             subtileStartX += subtileWidth) {

          // Iterate over the elements in each subtile
          for (size_t i = 0; i < subtileHeight; ++i) {
            for (size_t j = 0; j < subtileWidth; ++j) {
              size_t inputGlobalX = tileStartX + subtileStartX + j;
              size_t inputGlobalY = tileStartY + subtileStartY + i;
              size_t inputIndex = inputGlobalY * matrixWidth + inputGlobalX;

              size_t outputGlobalX = tileStartX + tileCountingIndex % tileWidth;
              size_t outputGlobalY = tileStartY + tileCountingIndex / tileWidth;
              size_t outputIndex = outputGlobalY * matrixWidth + outputGlobalX;

              if (!unshuffle) {
                res[outputIndex] = bfpMatrix[inputIndex];
              } else {
                res[inputIndex] = bfpMatrix[outputIndex];
              }
              tileCountingIndex++;
            }
          }
        }
      }
    }
  }

  return res;
}

// Pretty print to ostream a bfp16ebs8 array
inline void printBfp16ebs8Array(int arraySize, std::vector<uint8_t> array,
                                int blocksPerLine = 4,
                                int blocksBeforeEmptyLine = 8,
                                std::ostream &ostream = std::cout,
                                int width = 3,
                                const std::string &blockSeparatorStart = " | B",
                                const std::string &blockSeparatorEnd = " - ") {
  for (int i = 0; i < arraySize; i++) {
    if (i % (blocksPerLine * 9) == 0) {
      ostream << "\n";
      if (i % (blocksBeforeEmptyLine * 9) == 0) {
        ostream << "\n";
      }
    }

    if (i % 9 == 0) {
      ostream << blockSeparatorStart << std::setw(width) << i / 9 << " - ";
    }

    ostream << std::setw(4) << int(array[i]);
  }

  ostream << std::endl;
}
