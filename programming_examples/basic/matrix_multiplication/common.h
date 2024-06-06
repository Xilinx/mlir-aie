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
#include <boost/program_options.hpp>
#include <cmath>
#include <optional>
#include <ostream>

namespace matmul_common {

namespace po = boost::program_options;

// --------------------------------------------------------------------------
// Command Line Argument Handling
// --------------------------------------------------------------------------

void check_arg_file_exists(po::variables_map &vm_in, std::string name) {
  if (!vm_in.count(name)) {
    throw std::runtime_error("Error: no " + name + " file was provided\n");
  } else {
    std::ifstream test(vm_in[name].as<std::string>());
    if (!test) {
      throw std::runtime_error("The " + name + " file " +
                               vm_in[name].as<std::string>() +
                               " does not exist.\n");
    }
  }
}

void add_default_options(po::options_description &desc) {
  desc.add_options()("help,h", "produce help message")(
      "xclbin,x", po::value<std::string>()->required(),
      "the input xclbin path")(
      "kernel,k", po::value<std::string>()->required(),
      "the kernel name in the XCLBIN (for instance PP_PRE_FD)")(
      "verbosity,v", po::value<int>()->default_value(0),
      "the verbosity of the output")(
      "instr,i", po::value<std::string>()->required(),
      "path of file containing userspace instructions sent to the NPU")(
      "verify", po::value<bool>()->default_value(true),
      "whether to verify the AIE computed output")(
      "M,M", po::value<int>()->default_value(512), "Matrix size M")(
      "K,K", po::value<int>()->default_value(512), "Matrix size K")(
      "N,N", po::value<int>()->default_value(512),
      "Matrix size N")("iters", po::value<int>()->default_value(1))(
      "warmup", po::value<int>()->default_value(0))(
      "trace_sz,t", po::value<int>()->default_value(0))(
      "trace_file", po::value<std::string>()->default_value("trace.txt"),
      "where to store trace output");
}

void parse_options(int argc, const char *argv[], po::options_description &desc,
                   po::variables_map &vm) {
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      std::exit(1);
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
    std::cerr << "Usage:\n" << desc << "\n";
    std::exit(1);
  }

  check_arg_file_exists(vm, "xclbin");
  check_arg_file_exists(vm, "instr");
}

// --------------------------------------------------------------------------
// AIE Specifics
// --------------------------------------------------------------------------

std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

// --------------------------------------------------------------------------
// Matrix / Float / Math
// --------------------------------------------------------------------------

static inline std::int16_t random_int16_t() {
  return (std::int16_t)rand() % 0x10000;
}

static inline std::bfloat16_t random_bfloat16_t() {
  // Random numbers should NOT be uniformly between 0 and 1, because that
  // would make the matrix product AB always close to 1.
  return std::bfloat16_t(4.0 * (float)rand() / (float)(RAND_MAX));
}

template <typename Tin, typename Tout>
void matmul_naive(int M, int N, int K, const std::vector<Tin> A,
                  const std::vector<Tin> B, std::vector<Tout> &C) {
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      Tout running_sum = 0;
      for (int k = 0; k < K; k++) {
        running_sum += Tout(A[row * K + k] * B[k * N + col]);
      }
      C[row * N + col] = Tout(running_sum);
    }
  }
}

template <typename Tin, typename Tout>
void matmul(int M, int N, int K, const std::vector<Tin> A,
            const std::vector<Tin> B, std::vector<Tout> &C) {
  // A is an  MxK matrix
  // B is a   KxN matrix
  // C is the MxN output matrix, assumed to be zeroed out

  constexpr int K_block_size = 64;
  const int n_K_blocks = K / K_block_size;

  const Tin *B_origin = B.data(); /* Avoid a calls to B.data() within the loop
                                     with this const variable. B does not get
                                     resized, so the pointer remains valid. */

  const Tin *A_base = A.data(); /* Points to start of current row of A,
                                   monotonically increasing by K. */
  const Tin *B_base = B_origin; /* Points to start of current column of B;
                                   increases by 1 in each inner loop, resets
                                   to B_origin (0) at the start of a new row
                                   (outer loop). */

  const Tin *A_ptr = A_base;
  const Tin *B_ptr = B_base;
  Tout *C_ptr = C.data(); /* Monotonically increasing by 1. */

  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      A_ptr = A_base;
      B_ptr = B_base;
      Tout running_sum = 0;
      for (int k = 0; k < n_K_blocks; k++) {
        for (int i = 0; i < K_block_size; i++) {
          running_sum += Tout(*A_ptr) * Tout(*B_ptr);
          A_ptr += 1; // Advance to right neighbor; next value in this row
          B_ptr += N; // Advance to bottom neighbor; next value in this column
        }
      }
      *C_ptr = Tout(running_sum);
      C_ptr += 1;
      B_base += 1; /* Next iteration: same row of A (A_base unchanged),
                      next column of B (B_base increases by 1) */
    }
    A_base += K;       // Advance to next row of A
    B_base = B_origin; /* Next row of A means we need to restart at the first
                          column of B. */
  }
}

template <typename Tin, typename Tout>
Tout mul_acc(int M, int N, int K, int row, int col, const std::vector<Tin> A,
             const std::vector<Tin> B) {
  Tout running_sum = 0;
  for (int k = 0; k < K; k++) {
    running_sum += Tout(A[row * K + k] * B[k * N + col]);
  }
  return running_sum;
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
  for (int col = 0; col < (n_printable_cols + 1) / 2; col++) {                 \
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

  for (int row = 0; row < (n_printable_rows + 1) / 2; row++) {
    print_row(matrix[row * n_cols + col]);
    ostream << std::endl;
  }
  if (elide_rows) {
    print_row(elide_sym);
    ostream << std::endl;
  }
  for (int row = n_printable_rows / 2 + 1; row < n_printable_rows; row++) {
    print_row(matrix[row * n_cols + col]);
    ostream << std::endl;
  }

#undef print_row
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
verify_single(std::ostream &os, int row, int col, Tout expected, Tout actual) {
  const float absTol = 0.5;
  const float relTol = 0.15;
  if (!nearly_equal(expected, actual, relTol, absTol)) {
    return (struct error<Tout>){row, col, expected, actual};
  }
  return std::nullopt;
}

template <typename Tout>
void print_error_summary(std::ostream &os, int n_errors,
                         std::vector<struct error<Tout>> &errors) {
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
}

void print_progress_bar(std::ostream &os, double progress, int len = 75) {
  os << "\r" << std::string((int)(progress * len), '|')
     << std::string(len - (int)(progress * len), ' ') << std::setw(4)
     << std::fixed << std::setprecision(0) << progress * 100 << "%"
     << "\r";
}

template <typename Tin, typename Tout>
int verify(int M, int N, int K, std::vector<Tin> A, std::vector<Tin> B,
           std::vector<Tout> C, int verbosity = 0) {
  int n_errors = 0;
  std::vector<struct error<Tout>> errors;

  std::vector<Tout> CRef(M * N);
  matmul(M, N, K, A, B, CRef);

  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      std::optional<struct error<Tout>> error = verify_single(
          std::cout, row, col, CRef[row * N + col], C[row * N + col]);
      if (error.has_value()) {
        if (n_errors < max_printable_errors) {
          errors.push_back(*error);
        }
        n_errors++;
      }
    }
  }
  print_error_summary(std::cout, n_errors, errors);

  if (n_errors > 0) {
    std::cout << std::endl << "Reference:" << std::endl;
    matmul_common::print_matrix(CRef, N);
    std::cout << std::endl << "Output:" << std::endl;
    matmul_common::print_matrix(C, N);
  }

  return n_errors;
}

template <typename Tin, typename Tout>
int verify_stochastic(int M, int N, int K, std::vector<Tin> A,
                      std::vector<Tin> B, std::vector<Tout> C, int n_samples,
                      int verbosity = 0) {
  std::mt19937 rng;
  auto rows = std::views::iota(0, M);
  auto cols = std::views::iota(0, N);
  auto sampled_rows = std::vector<int>(n_samples);
  auto sampled_cols = std::vector<int>(n_samples);

  std::ranges::sample(rows, sampled_rows.begin(), n_samples, rng);
  std::ranges::sample(cols, sampled_cols.begin(), n_samples, rng);

  int n_errors = 0;
  std::vector<struct error<Tout>> errors;
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
    Tout ref = mul_acc<Tin, Tout>(M, N, K, row, col, A, B);
    std::optional<struct error<Tout>> error =
        verify_single(std::cout, row, col, ref, C[row * N + col]);
    if (error.has_value()) {
      if (n_errors < max_printable_errors) {
        errors.push_back(*error);
      }
      n_errors++;
    }
  }
  std::cout << std::endl;

  print_error_summary(std::cout, n_errors, errors);
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