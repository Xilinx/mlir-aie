//===- test_utils.h ----------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This file contains common helper functions for the generic host code

#ifndef _TEST_UTILS_H_
#define _TEST_UTILS_H_

#include "cxxopts.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#if defined(__STDCPP_BFLOAT16_T__)
#include <stdfloat>
#endif
#include <string>
#include <vector>

namespace xrt {
class device;
class kernel;
} // namespace xrt

namespace test_utils {

void check_arg_file_exists(const cxxopts::ParseResult &result,
                           std::string name);

void add_default_options(cxxopts::Options &options);

void parse_options(int argc, const char *argv[], cxxopts::Options &options,
                   cxxopts::ParseResult &result);

std::vector<uint32_t> load_instr_sequence(std::string instr_path);
std::vector<uint32_t> load_instr_binary(std::string instr_path);

void init_xrt_load_kernel(xrt::device &device, xrt::kernel &kernel,
                          int verbosity, std::string xclbinFileName,
                          std::string kernelNameInXclbin);

static inline std::int16_t random_int16_t(int32_t range = 0x10000) {
  return (std::int16_t)rand() % range;
}

static inline std::int32_t random_int32_t(int32_t range = 0x10000) {
  return (std::int32_t)rand() % range;
}

// The Linux toolchain has std::bfloat16_t. MSVC does not.
//
// Use this host-side helper for bfloat16 XRT buffers and reference checks.
// Device code should use the AIE bfloat16 types and APIs.
#if defined(__STDCPP_BFLOAT16_T__)
using bfloat16_t = std::bfloat16_t;

static inline bfloat16_t bfloat16_from_float(float value) {
  return bfloat16_t(value);
}

static inline bfloat16_t bfloat16_from_bits(std::uint16_t bits) {
  bfloat16_t value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline float bfloat16_to_float(bfloat16_t value) {
  return static_cast<float>(value);
}
#else
using bfloat16_t = std::uint16_t;

static inline bfloat16_t bfloat16_from_bits(std::uint16_t bits) { return bits; }

static inline float bfloat16_to_float(bfloat16_t bits) {
  const std::uint32_t expanded_bits = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &expanded_bits, sizeof(value));
  return value;
}

static inline bfloat16_t bfloat16_from_float(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  // Round to nearest-even instead of truncating.
  const std::uint32_t lsb = (bits >> 16) & 1U;
  const std::uint32_t rounding_bias = 0x7FFFU + lsb;
  return static_cast<bfloat16_t>((bits + rounding_bias) >> 16);
}
#endif

static inline bfloat16_t random_bfloat16_t(bfloat16_t scale, bfloat16_t bias) {
  const float scale_value = bfloat16_to_float(scale);
  const float bias_value = bfloat16_to_float(bias);
  return bfloat16_from_float((scale_value * (float)rand() / (float)(RAND_MAX)) +
                             bias_value);
}

bool nearly_equal(float a, float b, float epsilon = 128 * FLT_EPSILON,
                  float abs_th = FLT_MIN);

static inline bool nearly_equal_bfloat16(bfloat16_t a, bfloat16_t b,
                                         float epsilon = 128 * FLT_EPSILON,
                                         float abs_th = FLT_MIN) {
  return nearly_equal(bfloat16_to_float(a), bfloat16_to_float(b), epsilon,
                      abs_th);
}

static inline bfloat16_t bfloat16_add(bfloat16_t lhs, bfloat16_t rhs) {
  return bfloat16_from_float(bfloat16_to_float(lhs) + bfloat16_to_float(rhs));
}

static inline bfloat16_t bfloat16_mul(bfloat16_t lhs, bfloat16_t rhs) {
  return bfloat16_from_float(bfloat16_to_float(lhs) * bfloat16_to_float(rhs));
}

static inline bfloat16_t bfloat16_div(bfloat16_t lhs, bfloat16_t rhs) {
  return bfloat16_from_float(bfloat16_to_float(lhs) / bfloat16_to_float(rhs));
}

static inline bfloat16_t bfloat16_tanh(bfloat16_t value) {
  return bfloat16_from_float(std::tanh(bfloat16_to_float(value)));
}

template <typename T>
void print_matrix(const std::vector<T> matrix, int n_cols,
                  int n_printable_rows = 10, int n_printable_cols = 10,
                  std::ostream &ostream = std::cout,
                  const char col_sep[] = "  ", const char elide_sym[] = " ... ",
                  int w = -1) {
  assert(matrix.size() % n_cols == 0);

  auto maxima = std::minmax_element(matrix.begin(), matrix.end());
  T max_val = std::max(*maxima.first, std::abs(*maxima.second));
  size_t n_digits = log10(max_val);
  if (w == -1) {
    w = n_digits;
  }
  int n_rows = matrix.size() / n_cols;

  n_printable_rows = std::min(n_rows, n_printable_rows);
  n_printable_cols = std::min(n_cols, n_printable_cols);

  const bool elide_rows = n_printable_rows < n_rows;
  const bool elide_cols = n_printable_cols < n_cols;

  if (elide_rows || elide_cols) {
    w = std::max((int)w, (int)strlen(elide_sym));
  }

  w += 3; // for decimal point and two decimal digits
  ostream << std::fixed << std::setprecision(2);

#define print_row(what)                                                        \
  for (int col = 0; col < n_printable_cols / 2; col++) {                       \
    ostream << std::right << std::setw(w) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }                                                                            \
  if (elide_cols) {                                                            \
    ostream << std::setw(0) << elide_sym;                                      \
  }                                                                            \
  for (int col = n_printable_cols / 2 + 1; col < n_printable_cols; col++) {    \
    ostream << std::right << std::setw(w) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }

  for (int row = 0; row < n_printable_rows / 2; row++) {
    print_row(matrix[row * n_rows + col]);
    ostream << std::endl;
  }
  if (elide_rows) {
    print_row(elide_sym);
    ostream << std::endl;
  }
  for (int row = n_printable_rows / 2 + 1; row < n_printable_rows; row++) {
    print_row(matrix[row * n_rows + col]);
    ostream << std::endl;
  }

#undef print_row
}

void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path);

} // namespace test_utils

#endif // _TEST_UTILS_H_
