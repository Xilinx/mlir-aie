//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

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
#include "test_library.h"
#include "aie_inc.cpp"

#define BUF_SIZE (4 * 10) // # ints

void read_all_into(aie_libxaie_ctx_t *_xaie, int *buf) {
  for (int i = 0; i < BUF_SIZE; i++) {
    buf[i] = mlir_aie_read_buffer_buf83(_xaie, i);
  }
}

void populate_expected(int *buf) {
  int sum = 0;
  for (int i = 0; i < BUF_SIZE; i++) {
    switch (i % 10) {
    case 0:
      sum = i;
      break;
    case 1:
      sum += i;
      buf[i - 1] = sum;
      buf[i] = sum;
      break;
    case 2:
      sum = i;
      break;
    case 3:
      sum += i;
      break;
    case 4:
      sum += i;
      buf[i - 2] = sum;
      buf[i - 1] = sum;
      buf[i] = sum;
      break;
    case 5:
      sum = i;
      break;
    case 6:
      sum += i;
      break;
    case 7:
      sum += i;
      buf[i - 2] = sum;
      buf[i - 1] = sum;
      buf[i] = sum;
      break;
    case 8:
      sum = i;
      break;
    case 9:
      sum += i;
      buf[i - 1] = sum;
      buf[i] = sum;
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  int errors = 0;
  int seen[BUF_SIZE];
  int expected[BUF_SIZE];

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  assert(NULL != _xaie);
  mlir_aie_init_device(_xaie);
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_start_cores(_xaie);

  // After this lock is acquired, the kernels have completed and buf83
  // is populated.
  assert(XAIE_OK == mlir_aie_acquire_lock83(_xaie, -1, 5000));

  read_all_into(_xaie, seen);

  mlir_aie_deinit_libxaie(_xaie);

  populate_expected(expected);

  for (int i = 0; i < BUF_SIZE; i++) {
    printf("%04d=?=%04d ", seen[i], expected[i]);
    if ((i + 1) % 6 == 0) {
      printf("\n");
    }
    if (seen[i] != expected[i]) {
      printf("\nFAIL at index %d: %d != %d.\n", i, seen[i], expected[i]);
      return 1;
    }
  }

  printf("\nPASS!\n");
  return 0;
}