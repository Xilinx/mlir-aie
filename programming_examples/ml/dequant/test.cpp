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

// Helper to compute number of output elements from input size
inline int get_num_outputs(int in_bytes) {
  double denom = 0.5 + 2.0 / SF_BLOCK_SIZE;
  return static_cast<int>(in_bytes / denom);
}

// Helper to compute input size from number of outputs
inline int get_in_bytes(int out_elements) {
  return (out_elements / 2) + (out_elements / SF_BLOCK_SIZE) * 2;
}

// Initialize Input buffer 1 with packed int4 values and per-block bfloat16
// scales
void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int in_bytes) {
  int out_elements = get_num_outputs(in_bytes);
  int num_blocks = out_elements / SF_BLOCK_SIZE;
  int packed_int4_bytes = out_elements / 2;
  // Fill packed int4 values
  for (int pr_byte = 0; pr_byte < packed_int4_bytes; pr_byte++) {
    std::int8_t lower = (std::rand()) & 0xF; // 0..15, later interpreted as int4
    std::int8_t upper = (std::rand()) & 0xF;
    bufIn1[pr_byte] = static_cast<std::uint8_t>((upper << 4) | (lower & 0xF));
  }
  // bfloat16 scales per block of SF_BLOCK_SIZE outputs (2 bytes per scale)
  for (int isf = 0; isf < num_blocks; isf++) {
    float sf_val =
        4.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    std::bfloat16_t sf = static_cast<std::bfloat16_t>(sf_val);
    std::uint16_t bits = *reinterpret_cast<std::uint16_t *>(&sf);
    std::uint8_t lo = static_cast<std::uint8_t>(bits & 0x00FF);
    std::uint8_t hi = static_cast<std::uint8_t>((bits >> 8) & 0x00FF);
    int base = packed_int4_bytes + isf * 2;
    bufIn1[base] = lo;
    bufIn1[base + 1] = hi;
  }
}

// Initialize Output buffer
void initialize_bufOut(DATATYPE_OUT *bufOut, int out_elements) {
  std::memset(bufOut, 0, out_elements * sizeof(DATATYPE_OUT));
}

// Functional correctness verifier for int4 -> bfloat16 dequantization
int verify_dequant_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                          int in_bytes, int out_elements) {
  int errors = 0;

  int num_blocks = out_elements / SF_BLOCK_SIZE;
  int packed_int4_bytes = out_elements / 2;

  // Preload scales
  std::vector<std::bfloat16_t> scales(num_blocks);
  for (int isf = 0; isf < num_blocks; isf++) {
    int base = packed_int4_bytes + isf * 2;
    std::uint16_t bits = static_cast<std::uint16_t>(bufIn1[base]) |
                         (static_cast<std::uint16_t>(bufIn1[base + 1]) << 8);
    std::bfloat16_t sf = *reinterpret_cast<std::bfloat16_t *>(&bits);
    scales[isf] = sf;
  }

  // Walk all outputs
  for (int pr = 0; pr < out_elements; pr++) {
    int byte_index = pr / 2;        // two int4 per byte
    bool use_upper = (pr % 2) != 0; // odd index uses upper nibble
    std::uint8_t packed = bufIn1[byte_index];
    std::int8_t nibble = use_upper ? (packed >> 4) & 0xF : (packed & 0xF);
    // Convert 4-bit two's complement to signed int in [-8,7]
    int val = (nibble >= 8) ? (static_cast<int>(nibble) - 16)
                            : static_cast<int>(nibble);
    std::bfloat16_t sf = scales[pr / SF_BLOCK_SIZE];
    std::bfloat16_t expected = static_cast<std::bfloat16_t>(
        static_cast<float>(sf) * static_cast<float>(val));

    int out_idx = pr;
    if (out_idx >= out_elements)
      continue;
    std::bfloat16_t hw = bufOut[out_idx];

    float f_expected = static_cast<float>(expected);
    float f_hw = static_cast<float>(hw);
    float denom = std::fabs(f_hw);
    float rel_err = denom > 0.0f ? std::fabs(f_hw - f_expected) / denom
                                 : std::fabs(f_hw - f_expected);
    if (rel_err > 0.01f) {
      std::uint16_t hw_raw = *reinterpret_cast<std::uint16_t *>(&hw);
      std::uint16_t exp_raw = *reinterpret_cast<std::uint16_t *>(&expected);
      std::cout << "Idx " << pr << " HW " << std::hex << hw_raw << " ("
                << std::dec << f_hw << ") REF " << std::hex << exp_raw << " ("
                << std::dec << f_expected << ") sf=" << static_cast<float>(sf)
                << " val=" << val << std::endl;
      errors++;
    }
  }
  return errors;
}

//*****************************************************************************
// Should not need to modify below section
//*****************************************************************************

int main(int argc, const char *argv[]) {

  // Total number of dequantized outputs
  constexpr int OUT_VOLUME = 4096;
  // Packed int4 bytes + bfloat16 scales bytes
  constexpr int IN1_VOLUME =
      (OUT_VOLUME / 2) + (OUT_VOLUME / SF_BLOCK_SIZE) * 2;

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_OUT, initialize_bufIn1,
                              initialize_bufOut, verify_dequant_kernel>(
      IN1_VOLUME, OUT_VOLUME, myargs);
  return res;
}