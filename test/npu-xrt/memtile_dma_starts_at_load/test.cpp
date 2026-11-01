//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cstdint>

constexpr int OUT_VOLUME = 256;

void initialize_bufOut(std::int32_t *bufOut, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufOut[i] = -1;
}

int verify(std::int32_t *bufOut, int, int verbosity) {
  int errors = 0;
  for (int i = 0; i < OUT_VOLUME; i++) {
    if (bufOut[i] != i) {
      if (verbosity >= 1 || errors < 16)
        std::cout << "Error at " << i << ": " << bufOut[i] << " != " << i
                  << std::endl;
      errors++;
    }
  }
  return errors;
}

int main(int argc, const char *argv[]) {
  args myargs = parse_args(argc, argv);
  return setup_and_run_aie(
      verify, std::tuple<>(),
      make_out<std::int32_t>(OUT_VOLUME, initialize_bufOut), myargs);
}
