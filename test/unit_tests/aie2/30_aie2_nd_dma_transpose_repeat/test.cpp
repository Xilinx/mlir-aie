//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "test_library.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <xaiengine.h>

#include "aie_inc.cpp"

void populate_expected(int *buf) {
  const int stepsizes[3] = {8, 1, 1};
  const int wraps[3] = {8, 8, 2};
  int l = 0;
  for (int i = 0; i < wraps[2]; i++) {
    for (int j = 0; j < wraps[1]; j++) {
      for (int k = 0; k < wraps[0]; k++) {
        int i_ = i * stepsizes[2];
        int j_ = j * stepsizes[1];
        int k_ = k * stepsizes[0];
        buf[l] = i_ + j_ + k_;
        l++;
      }
    }
  }
}

void read_into(aie_libxaie_ctx_t *_xaie, int *buf) {
  for (int i = 0; i < 128; i++) {
    buf[i] = mlir_aie_read_buffer_buf34(_xaie, i);
  }
}

void print_buf(int *buf) {
  for (int i = 0; i < 128; i++) {
    printf("%3d ", buf[i]);
    if (0 == (i + 1) % 8) {
      printf("\n");
    }
  }
}

int main(int argc, char *argv[]) {
  int errors = 0;
  int seen[128];
  int expected[128];

  // Boilerplate setup code
  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_start_cores(_xaie);

  // After this lock is acquired, the kernels have completed and buf34
  // is populated.
  assert(XAIE_OK == mlir_aie_acquire_lock34_recv(_xaie, -1, 5000));

  // Read buf34 into seen.
  read_into(_xaie, seen);

  // Tear down.
  mlir_aie_deinit_libxaie(_xaie);

  // Compare results to expected results.
  populate_expected(expected);
  for (int i = 0; i < 128; i++) {
    if (expected[i] != seen[i]) {
      errors = 1;
      printf("Mismatch at index %d: %d != %d.\n", i, expected[i], seen[i]);
    }
  }

  if (0 == errors) {
    print_buf(seen);
    printf("PASS!\n");
  } else {
    printf("Expected:\n");
    print_buf(expected);
    printf("But got:\n");
    print_buf(seen);
    printf("FAIL.\n");
  }

  return errors;
}
