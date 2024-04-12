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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <cmath>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace po = boost::program_options;

namespace test_utils {

void check_arg_file_exists(po::variables_map &vm_in, std::string name);

void add_default_options(po::options_description &desc);

void parse_options(int argc, const char *argv[], po::options_description &desc,
                   po::variables_map &vm);

std::vector<uint32_t> load_instr_sequence(std::string instr_path);

void init_xrt_load_kernel(xrt::device &device, xrt::kernel &kernel,
                          int verbosity, std::string xclbinFileName,
                          std::string kernelNameInXclbin);

static inline std::int16_t random_int16_t();

static inline std::bfloat16_t random_bfloat16_t(std::bfloat16_t scale,
                                                std::bfloat16_t bias) {
  return std::bfloat16_t((scale * (float)rand() / (float)(RAND_MAX)) + bias);
}

bool nearly_equal(float a, float b, float epsilon = 128 * FLT_EPSILON,
                  float abs_th = FLT_MIN);

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
