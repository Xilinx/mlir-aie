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
#include <iostream>
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
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions sent to the NPU",
      cxxopts::value<std::string>())(
      "verify", "whether to verify the AIE computed output",
      cxxopts::value<bool>()->default_value("true"))(
      "rows,M", "Matrix size M", cxxopts::value<int>()->default_value("512"))(
      "inner,K", "Matrix size K", cxxopts::value<int>()->default_value("512"))(
      "columns,N", "Matrix size N",
      cxxopts::value<int>()->default_value("512"))(
      "iters", "number of iterations",
      cxxopts::value<int>()->default_value("1"))(
      "warmup", "number of warmup iterations",
      cxxopts::value<int>()->default_value("0"))(
      "trace_sz,t", "trace size", cxxopts::value<int>()->default_value("0"))(
      "trace_file", "where to store trace output",
      cxxopts::value<std::string>()->default_value("trace.txt"))(
      "b_col_maj", "Is B matrix in colum-major format?",
      cxxopts::value<int>()->default_value("0"))(
      "c_col_maj", "Is C matrix in colum-major format?",
      cxxopts::value<int>()->default_value("0"));
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

// --------------------------------------------------------------------------
// Matrix / Float / Math
// --------------------------------------------------------------------------

template <typename T>
static inline T get_random();

template <>
std::int16_t get_random<std::int16_t>() {
  return (std::int16_t)rand() % 0x10000;
}

template <>
int8_t get_random<int8_t>() {
  return (int8_t)rand() % 0x100;
}

template <>
std::bfloat16_t get_random<std::bfloat16_t>() {
  // Random numbers should NOT be uniformly between 0 and 1, because that
  // would make the matrix product AB always close to 1.
  return std::bfloat16_t(4.0 * (float)rand() / (float)(RAND_MAX));
}

template <typename Tin, typename Tout, typename Tacc>
void matmul(int M, int N, int K, const std::vector<Tin> A,
            const std::vector<Tin> B, std::vector<Tout> &C, int b_col_maj,
            int c_col_maj) {
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      Tacc running_sum = 0;
      for (int k = 0; k < K; k++) {
        if (!b_col_maj) {
          running_sum += Tacc(A[row * K + k] * B[k * N + col]);
        } else {
          running_sum += Tacc(A[row * K + k] * B[k + col * K]);
        }
      }
      if (!c_col_maj) {
        C[row * N + col] = Tout(running_sum);
      } else {
        C[row + col * M] = Tout(running_sum);
      }
    }
  }
}

template <typename Tin, typename Tout, typename Tacc>
Tout mul_acc(int M, int N, int K, int row, int col, const std::vector<Tin> A,
             const std::vector<Tin> B, int b_col_maj) {
  Tacc running_sum = 0;
  for (int k = 0; k < K; k++) {
    if (!b_col_maj) {
      running_sum += Tacc(A[row * K + k] * B[k * N + col]);
    } else {
      running_sum += Tacc(A[row * K + k] * B[k + col * K]);
    }
  }
  return (Tout)running_sum;
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
  ostream << std::fixed << std::setprecision(2);

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
  for (uint i = 0; i < matrix.size(); i++) {
    cast_matrix[i] = (int16_t)matrix[i];
  }
  print_matrix(cast_matrix, n_cols, n_printable_rows, n_printable_cols, ostream,
               col_sep, elide_sym, w);
}

constexpr int max_printable_errors = 32;

template <typename Tout>
struct error {
  int row;
  int col;
  Tout expected;
  Tout actual;
};

template <typename Tout>
std::optional<struct error<Tout>>
verify_single(std::ostream &os, int row, int col, Tout expected, Tout actual,
              float abs_tol, float rel_tol) {
  bool match = expected == actual;
  if (abs_tol > 0 || rel_tol > 0) {
    // Allow for some tolerance for float data types
    match = nearly_equal(expected, actual, rel_tol, abs_tol);
  }
  if (!match) {
    return (struct error<Tout>){row, col, expected, actual};
  }
  return std::nullopt;
}

template <typename Tout>
void print_error_summary(std::ostream &os, int n_errors,
                         std::vector<struct error<Tout>> &errors,
                         Tout max_rel_error) {
  for (struct error<Tout> &err : errors) {
    os << "[" << std::setw(5) << err.row << ", " << std::setw(5) << err.col
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
       << max_rel_error * 100 << "%"
       << " (last above)" << std::endl;
  }
}

void print_progress_bar(std::ostream &os, double progress, int len = 75) {
  os << "\r" << std::string((int)(progress * len), '|')
     << std::string(len - (int)(progress * len), ' ') << std::setw(4)
     << std::fixed << std::setprecision(0) << progress * 100 << "%"
     << "\r";
}

template <typename Tin, typename Tout, typename Tacc>
int verify(int M, int N, int K, std::vector<Tin> A, std::vector<Tin> B,
           std::vector<Tout> C, int verbosity = 0, float abs_tol = 0.5,
           float rel_tol = 0.05, int b_col_maj = 0, int c_col_maj = 0) {
  int n_errors = 0;
  std::vector<struct error<Tout>> errors;
  Tout max_rel_error = (Tout)0.0f;
  struct error<Tout> max_error;

  std::vector<Tout> CRef(M * N);
  matmul<Tin, Tout, Tacc>(M, N, K, A, B, CRef, b_col_maj, c_col_maj);

  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      std::optional<struct error<Tout>> error =
          verify_single(std::cout, row, col, CRef[row * N + col],
                        C[row * N + col], abs_tol, rel_tol);
      if (error.has_value()) {
        if (n_errors < max_printable_errors) {
          errors.push_back(*error);
        }
        Tout rel_error =
            std::abs(error->actual - error->expected) /
            std::max(std::abs(error->actual), std::abs(error->expected));
        if (rel_error > max_rel_error) {
          max_rel_error = rel_error;
          max_error = *error;
        }
        n_errors++;
      }
    }
  }
  if (n_errors)
    errors.push_back(max_error);
  print_error_summary(std::cout, n_errors, errors, max_rel_error);

  if (n_errors > 0) {
    std::cout << std::endl << "Reference:" << std::endl;
    matmul_common::print_matrix(CRef, N);
    std::cout << std::endl << "Output:" << std::endl;
    matmul_common::print_matrix(C, N);
  }

  return n_errors;
}

template <typename Tin, typename Tout, typename Tacc>
int verify_stochastic(int M, int N, int K, std::vector<Tin> A,
                      std::vector<Tin> B, std::vector<Tout> C, int n_samples,
                      int verbosity = 0, float abs_tol = 0.5,
                      float rel_tol = 0.05, int b_col_maj = 0,
                      int c_col_maj = 0) {
  std::mt19937 rng;
  auto rows = std::views::iota(0, M);
  auto cols = std::views::iota(0, N);
  auto sampled_rows = std::vector<int>(n_samples);
  auto sampled_cols = std::vector<int>(n_samples);

  std::ranges::sample(rows, sampled_rows.begin(), n_samples, rng);
  std::ranges::sample(cols, sampled_cols.begin(), n_samples, rng);

  int n_errors = 0;
  std::vector<struct error<Tout>> errors;
  Tout max_rel_error = (Tout)0.0f;
  double progress = 0;
  for (std::tuple<size_t, std::tuple<int &, int &>> cell :
       std::views::enumerate(std::views::zip(sampled_rows, sampled_cols))) {
    int i = std::get<0>(cell);
    int row = std::get<0>(std::get<1>(cell));
    int col = std::get<1>(std::get<1>(cell));
    if (verbosity >= 1 &&
        (int)(progress * 100) < (int)((double)i / n_samples * 100)) {
      // Only print progress bar if percentage changed
      progress = (double)i / n_samples;
      print_progress_bar(std::cerr, progress);
    }
    Tout ref = mul_acc<Tin, Tout, Tacc>(M, N, K, row, col, A, B, b_col_maj);
    Tout observed;
    if (!c_col_maj) {
      observed = C[row * N + col];
    } else {
      observed = C[row + col * M];
    }
    std::optional<struct error<Tout>> error =
        verify_single(std::cout, row, col, ref, observed, abs_tol, rel_tol);
    if (error.has_value()) {
      if (n_errors < max_printable_errors) {
        errors.push_back(*error);
      }
      Tout rel_error =
          std::abs(error->actual - error->expected) /
          std::max(std::abs(error->actual), std::abs(error->expected));
      if (rel_error > max_rel_error) {
        max_rel_error = rel_error;
      }
      n_errors++;
    }
  }
  std::cout << std::endl;

  print_error_summary(std::cout, n_errors, errors, max_rel_error);
  return n_errors;
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
