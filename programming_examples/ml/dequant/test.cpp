//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "test_utils.h"
#include "xrt_test_wrapper.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
// ------------------------------------------------------
// Configure this to match your buffer data type
// ------------------------------------------------------
using DATATYPE_IN1 = std::uint8_t;    // Packed int4s and bfloat16 scales
using DATATYPE_OUT = std::bfloat16_t; // Dequantized bfloat16 output
#endif

// Group size (aka block size) is configurable via env var or default 32.
#ifndef SF_BLOCK_SIZE
#define SF_BLOCK_SIZE 32
#endif
constexpr int TILE_SIZE = 1024; // elements per tile

constexpr int IN1_VOLUME = IN1_SIZE;
constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE_OUT);

// For per-tile layout
constexpr int scale_size = (TILE_SIZE / SF_BLOCK_SIZE) * 2; // 2 bytes per scale
constexpr int per_tile_bytes =
    scale_size + (TILE_SIZE / 2); // 2 int4 elements per byte
constexpr int num_scales = TILE_SIZE / SF_BLOCK_SIZE;

// Initialize Input buffer 1 with packed int4 values and per-block bfloat16
// scales, with scales placed immediately after each tile's packed int4s
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int in_bytes) {
  int num_tiles = in_bytes / per_tile_bytes;

  for (int t = 0; t < num_tiles; t++) {
    int tile_base = t * per_tile_bytes;
    for (int pr = 0; pr < TILE_SIZE / 2; pr++) {
      std::int8_t lower = (std::rand()) & 0xF;
      std::int8_t upper = (std::rand()) & 0xF;
      bufIn1[tile_base + pr] =
          static_cast<std::uint8_t>((upper << 4) | (lower & 0xF));
    }
    int scale_base = tile_base + (TILE_SIZE / 2);
    for (int isf = 0; isf < num_scales; isf++) {
      float sf_val =
          4.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
      std::bfloat16_t sf = static_cast<std::bfloat16_t>(sf_val);
      std::uint16_t bits = *reinterpret_cast<std::uint16_t *>(&sf);
      std::uint8_t lo = static_cast<std::uint8_t>(bits & 0x00FF);
      std::uint8_t hi = static_cast<std::uint8_t>((bits >> 8) & 0x00FF);
      bufIn1[scale_base + isf * 2] = lo;
      bufIn1[scale_base + isf * 2 + 1] = hi;
    }
  }
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int out_elements) {
  std::memset(bufOut, 0, out_elements * sizeof(DATATYPE_OUT));
}

int verify_dequant_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                          int in_bytes, int out_elements,
                          bool verbose = false) {
  int errors = 0;
  int num_tiles = out_elements / TILE_SIZE;
  int samples_shown = 0;
  const int max_samples = 10; // Limit output to first 10 samples when verbose

  for (int t = 0; t < num_tiles; t++) {
    int tile_base = t * per_tile_bytes;
    int scale_base = tile_base + (TILE_SIZE / 2);

    // Preload scales for this tile
    std::vector<std::bfloat16_t> scales(num_scales);
    for (int isf = 0; isf < num_scales; isf++) {
      std::uint16_t bits =
          static_cast<std::uint16_t>(bufIn1[scale_base + isf * 2]) |
          (static_cast<std::uint16_t>(bufIn1[scale_base + isf * 2 + 1]) << 8);
      std::bfloat16_t sf = *reinterpret_cast<std::bfloat16_t *>(&bits);
      scales[isf] = sf;
    }

    // Walk all outputs in this tile
    for (int pr = 0; pr < TILE_SIZE; pr++) {
      int byte_index = tile_base + (pr / 2); // two int4 per byte
      bool use_upper = (pr % 2) != 0;        // odd index uses upper nibble
      std::uint8_t packed = bufIn1[byte_index];
      std::int8_t nibble = use_upper ? (packed >> 4) & 0xF : (packed & 0xF);
      // Convert 4-bit two's complement to signed int in [-8,7]
      int val = (nibble >= 8) ? (static_cast<int>(nibble) - 16)
                              : static_cast<int>(nibble);
      std::bfloat16_t sf = scales[pr / SF_BLOCK_SIZE];
      std::bfloat16_t expected = static_cast<std::bfloat16_t>(
          static_cast<float>(sf) * static_cast<float>(val));

      int out_idx = t * TILE_SIZE + pr;
      if (out_idx >= out_elements)
        continue;
      std::bfloat16_t hw = bufOut[out_idx];

      float f_expected = static_cast<float>(expected);
      float f_hw = static_cast<float>(hw);
      float denom = std::fabs(f_hw);
      float rel_err = denom > 0.0f ? std::fabs(f_hw - f_expected) / denom
                                   : std::fabs(f_hw - f_expected);

      // Show output for first few samples or any error
      if ((verbose && samples_shown < max_samples) || rel_err > 0.01f) {
        std::uint16_t hw_raw = *reinterpret_cast<std::uint16_t *>(&hw);
        std::uint16_t exp_raw = *reinterpret_cast<std::uint16_t *>(&expected);
        std::cout << "Idx " << out_idx << " Got: " << std::hex << hw_raw << " ("
                  << std::dec << f_hw << "), Expected: " << std::hex << exp_raw
                  << " (" << std::dec << f_expected << ") Error: " << rel_err;
        if (rel_err > 0.01f) {
          std::cout << " [ERROR]";
          errors++;
        }
        std::cout << std::endl;
        if (verbose && samples_shown < max_samples)
          samples_shown++;
      }
    }
  }
  return errors;
}

//*****************************************************************************
// Should not need to modify below section
//*****************************************************************************

int main(int argc, const char *argv[]) {

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_OUT, initialize_bufIn1,
                              initialize_bufOut,
                              [](DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                                 int volume, int verbosity) {
                                return verify_dequant_kernel(bufIn1, bufOut,
                                                             volume, OUT_VOLUME,
                                                             verbosity > 0);
                              }>(IN1_VOLUME, OUT_VOLUME, myargs);
  return res;
}