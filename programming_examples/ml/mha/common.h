//===- matrix_multiplication.h ----------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This file contains common helper functions for the matrix multiplication
// host code, such as verifying and printing matrices.

#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <fstream>
#include <optional>
#include <ostream>
#include <stdfloat>

#include "test_utils.h"

namespace matmul_common {

// --------------------------------------------------------------------------
// Command Line Argument Handling
// --------------------------------------------------------------------------

void add_default_options(cxxopts::Options &options) {
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())
      ("kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)", cxxopts::value<std::string>())
      ("verbosity,v", "the verbosity of the output", cxxopts::value<int>()->default_value("0"))
      ("instr,i", "path of file containing userspace instructions sent to the NPU", cxxopts::value<std::string>())
      ("heads", "Number of heads", cxxopts::value<int>()->default_value("1"))
      ("S_q", "Sequence Length of Queries", cxxopts::value<int>()->default_value("512"))
      ("d", "Embedding Dimension", cxxopts::value<int>()->default_value("512"))
      ("S_kv", "Sequence Length of Keys and Values", cxxopts::value<int>()->default_value("512"))
      ("iters", "number of iterations", cxxopts::value<int>()->default_value("1"))
      ("warmup", "number of warmup iterations",cxxopts::value<int>()->default_value("0"))
      ("trace_sz,t", "trace size", cxxopts::value<int>()->default_value("0"))
      ("trace_file", "where to store trace output", cxxopts::value<std::string>()->default_value("trace.txt"));
}

void parse_options(int argc, const char *argv[], cxxopts::Options &options,
                   cxxopts::ParseResult &result) {
  try {
    result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << "\n";
      std::exit(1);
    }

    // Check required options
    if (!result.count("xclbin") || !result.count("kernel") ||
        !result.count("instr")) {
      std::cerr << "Error: Required options missing\n\n";
      std::cerr << "Usage:\n" << options.help() << "\n";
      std::exit(1);
    }

  } catch (const cxxopts::exceptions::parsing &e) {
    std::cerr << e.what() << "\n\n";
    std::cerr << "Usage:\n" << options.help() << "\n";
    std::exit(1);
  }
}

// nearly_equal function adapted from Stack Overflow, License CC BY-SA 4.0
// Original author: P-Gn
// Source: https://stackoverflow.com/a/32334103
bool nearly_equal(float a, float b, float epsilon = 128 * FLT_EPSILON,
                  float abs_th = FLT_MIN)
// those defaults are arbitrary and could be removed
{
  assert(std::numeric_limits<float>::epsilon() <= epsilon);
  assert(epsilon < 1.f);

  if (a == b)
    return true;

  auto diff = std::abs(a - b);
  auto norm =
      std::min((std::abs(a) + std::abs(b)), std::numeric_limits<float>::max());
  // or even faster: std::min(std::abs(a + b),
  // std::numeric_limits<float>::max()); keeping this commented out until I
  // update figures below
  return diff < std::max(abs_th, epsilon * norm);
}

template <typename T>
static inline float get_abs_tol();
template <typename T>
static inline float get_rel_tol();

template <>
float get_abs_tol<std::int16_t>() {
  return 0.0;
}

template <>
float get_abs_tol<std::int32_t>() {
  return 0.0;
}

template <>
float get_abs_tol<std::bfloat16_t>() {
  return 0.5;
}

template <>
float get_abs_tol<float>() {
  return 0.5;
}

template <>
float get_abs_tol<int8_t>() {
  return 0;
}

template <>
float get_rel_tol<std::int16_t>() {
  return 0.0;
}

template <>
float get_rel_tol<std::int32_t>() {
  return 0.0;
}

template <>
float get_rel_tol<std::bfloat16_t>() {
  return 0.05;
}

template <>
float get_rel_tol<float>() {
  return 0.05;
}

template <>
float get_rel_tol<int8_t>() {
  return 0;
}

template <typename T>
void print_matrix(const std::vector<T> matrix, int n_cols,
                  int n_printable_rows = 10, int n_printable_cols = 10,
                  std::ostream &ostream = std::cout,
                  const char col_sep[] = "  ", const char elide_sym[] = " ... ",
                  int w = -1) {
  assert(matrix.size() % n_cols == 0);

  auto maxima = std::minmax_element(matrix.begin(), matrix.end());
  T max_val = std::max(*maxima.first, (T)std::abs(*maxima.second));
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
  ostream << std::fixed << std::setprecision(5);

#define print_row(what)                                                        \
  for (int col = 0; col < (n_printable_cols + 1) / 2; col++) {                 \
    ostream << std::right << std::setw(w) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }                                                                            \
  if (elide_cols) {                                                            \
    ostream << std::setw(0) << elide_sym;                                      \
  }                                                                            \
  for (int i = 0; i < n_printable_cols / 2; i++) {                             \
    int col = n_cols - n_printable_cols / 2 + i;                               \
    ostream << std::right << std::setw(w) << (what);                           \
    ostream << std::setw(0) << col_sep;                                        \
  }

  for (int row = 0; row < (n_printable_rows + 1) / 2; row++) {
    print_row(matrix[row * n_cols + col]);
    ostream << std::endl;
  }
  if (elide_rows) {
    print_row(elide_sym);
    ostream << std::endl;
  }
  for (int i = 0; i < n_printable_rows / 2; i++) {
    int row = n_rows - n_printable_rows / 2 + i;
    print_row(matrix[row * n_cols + col]);
    ostream << std::endl;
  }

#undef print_row
}

// int8_t aka char will not print as a number but as a character; specialize
// print_matrix<int8_t> to cast to int16_t first so everything prints as numbers
template <>
void print_matrix(const std::vector<int8_t> matrix, int n_cols,
                  int n_printable_rows, int n_printable_cols,
                  std::ostream &ostream, const char col_sep[],
                  const char elide_sym[], int w) {
  std::vector<int16_t> cast_matrix(matrix.size());
  for (int i = 0; i < matrix.size(); i++) {
    cast_matrix[i] = (int16_t)matrix[i];
  }
  print_matrix(cast_matrix, n_cols, n_printable_rows, n_printable_cols, ostream,
               col_sep, elide_sym, w);
}

constexpr int max_printable_errors = 32;

template <typename Tout>
struct error {
  int head;
  int row;
  int col;
  Tout expected;
  Tout actual;
};

template <typename Tout>
std::optional<struct error<Tout>> // TODO
verify_single(std::ostream &os, int head, int row, int col, Tout expected, Tout actual,
              float abs_tol, float rel_tol) {
  bool match = expected == actual;
  if (abs_tol > 0 || rel_tol > 0) {
    // Allow for some tolerance for float data types
    match = nearly_equal(expected, actual, rel_tol, abs_tol);
  }
  if (!match) {
    return (struct error<Tout>){head, row, col, expected, actual};
  }
  return std::nullopt;
}

template <typename Tout>
void print_error_summary(std::ostream &os, int n_errors, int total_elements,
                         std::vector<struct error<Tout>> &errors,
                         Tout max_rel_error) {
  os << "Number of errors: " << n_errors << "/" << total_elements << std::endl;

  for (struct error<Tout> &err : errors) {
    os << "[" << std::setw(5) << err.head << ", " << std::setw(5) << err.row << ", " << std::setw(5) << err.col
       << "] " << std::setw(4) << std::setprecision(2) << std::fixed
       << (float)err.actual << " =!= " << std::setw(4) << std::setprecision(2)
       << std::fixed << (float)err.expected << std::endl;
  }

  if (n_errors > max_printable_errors) {
    os << "...and " << std::setw(0) << n_errors - max_printable_errors
       << " further errors." << std::endl;
  }
  
  if (n_errors > 0) {
    os << "Maximum relative error: " << std::setw(3) << std::setprecision(0)
       << max_rel_error * 100 << "%" << std::endl;
  }
}

void print_progress_bar(std::ostream &os, double progress, int len = 75) {
  os << "\r" << std::string((int)(progress * len), '|')
     << std::string(len - (int)(progress * len), ' ') << std::setw(4)
     << std::fixed << std::setprecision(0) << progress * 100 << "%"
     << "\r";
}

// --------------------------------------------------------------------------
// Tracing
// --------------------------------------------------------------------------
void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path) {
  std::ofstream fout(path);
  uint32_t *traceOut = (uint32_t *)traceOutPtr;
  for (int i = 0; i < trace_size / sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
}

} // namespace matmul_common

#endif
