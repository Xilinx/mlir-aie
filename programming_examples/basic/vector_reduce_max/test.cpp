//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#ifndef DTYPE
#define DTYPE test_utils::bfloat16_t
#endif

using DATATYPE = DTYPE;

template <typename T> T random_input_value() {
  if constexpr (std::is_same_v<T, test_utils::bfloat16_t>) {
    return test_utils::random_bfloat16_t(test_utils::bfloat16_from_float(-4.0f),
                                         test_utils::bfloat16_from_float(8.0f));
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return test_utils::random_int32_t(100000);
  } else {
    std::cerr << "Unsupported data type" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <typename T> T lowest_value() {
  if constexpr (std::is_same_v<T, test_utils::bfloat16_t>) {
    return test_utils::bfloat16_from_float(
        -std::numeric_limits<float>::infinity());
  } else {
    return std::numeric_limits<T>::lowest();
  }
}

template <typename T> bool less_than(T lhs, T rhs) {
  if constexpr (std::is_same_v<T, test_utils::bfloat16_t>) {
    return test_utils::bfloat16_to_float(lhs) <
           test_utils::bfloat16_to_float(rhs);
  } else {
    return lhs < rhs;
  }
}

template <typename T> bool values_equal(T lhs, T rhs) {
  if constexpr (std::is_same_v<T, test_utils::bfloat16_t>) {
    return test_utils::nearly_equal_bfloat16(lhs, rhs);
  } else {
    return lhs == rhs;
  }
}

template <typename T> auto printable_value(T value) {
  if constexpr (std::is_same_v<T, test_utils::bfloat16_t>) {
    return test_utils::bfloat16_to_float(value);
  } else {
    return value;
  }
}

void initialize_bufIn1(DATATYPE *bufIn1, int SIZE) {
  DATATYPE max = lowest_value<DATATYPE>();
  for (int i = 0; i < SIZE; i++) {
    DATATYPE next = random_input_value<DATATYPE>();
    if (less_than(max, next))
      max = next;
    bufIn1[i] = next;
  }
}

void initialize_bufOut(DATATYPE *bufOut, int SIZE) {
  std::memset(bufOut, 0, SIZE);
}

int verify_vector_reduce_max(DATATYPE *bufIn1, DATATYPE *bufOut, int SIZE,
                             int verbosity) {
  int errors = 0;

  DATATYPE max = lowest_value<DATATYPE>();
  for (int i = 0; i < SIZE; i++) {
    if (less_than(max, bufIn1[i]))
      max = bufIn1[i];
  }

  if (!values_equal(bufOut[0], max)) {
    errors++;
    std::cout << "max is " << printable_value(max) << " calc "
              << printable_value(bufOut[0]) << std::endl;
  } else if (verbosity >= 1) {
    std::cout << "max is " << printable_value(max) << " calc "
              << printable_value(bufOut[0]) << std::endl;
  }
  return errors;
}

int main(int argc, const char *argv[]) {
  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(DATATYPE);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE);

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE, DATATYPE, initialize_bufIn1,
                              initialize_bufOut, verify_vector_reduce_max>(
      IN1_VOLUME, OUT_VOLUME, myargs);
  return res;
}
