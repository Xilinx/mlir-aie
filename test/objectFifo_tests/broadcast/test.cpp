//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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

#define LOCK_TIMEOUT 100

#define LINE_WIDTH 16
#define HEIGHT 4

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_clear_tile_memory(_xaie, 1, 2);
  mlir_aie_clear_tile_memory(_xaie, 1, 3);
  mlir_aie_clear_tile_memory(_xaie, 1, 4);
  mlir_aie_clear_tile_memory(_xaie, 3, 3);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_start_cores(_xaie);

  int errors = 0;

  printf("Waiting to acquire output lock for read ...\n");
  if (mlir_aie_acquire_lock_out12(_xaie, 1, LOCK_TIMEOUT)) {
    printf("ERROR: did not acquire lock!\n");
    errors++;
  }
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < LINE_WIDTH; j++) {
      mlir_aie_check("After exchange. Check [i * LINE_WIDTH + j] = j",
                     mlir_aie_read_buffer_out12(_xaie, i * LINE_WIDTH + j), (j),
                     errors);
    }
  }
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < LINE_WIDTH; j++)
      printf("%d ", mlir_aie_read_buffer_out12(_xaie, i * LINE_WIDTH + j));
    printf("\n");
  }

  printf("Waiting to acquire output lock for read ...\n");
  if (mlir_aie_acquire_lock_out14(_xaie, 1, LOCK_TIMEOUT)) {
    printf("ERROR: did not acquire lock!\n");
    errors++;
  }
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < LINE_WIDTH; j++) {
      mlir_aie_check("After exchange. Check [i * LINE_WIDTH + j] = j",
                     mlir_aie_read_buffer_out14(_xaie, i * LINE_WIDTH + j), (j),
                     errors);
    }
  }
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < LINE_WIDTH; j++)
      printf("%d ", mlir_aie_read_buffer_out14(_xaie, i * LINE_WIDTH + j));
    printf("\n");
  }

  printf("Waiting to acquire output lock for read ...\n");
  if (mlir_aie_acquire_lock_out33(_xaie, 1, LOCK_TIMEOUT)) {
    printf("ERROR: did not acquire lock!\n");
    errors++;
  }
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < LINE_WIDTH; j++) {
      int test_value = 2 * j;
      if (i == HEIGHT - 1)
        test_value = j;
      mlir_aie_check("After exchange. Check [i * LINE_WIDTH + j] = test_value",
                     mlir_aie_read_buffer_out33(_xaie, i * LINE_WIDTH + j),
                     (test_value), errors);
    }
  }
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < LINE_WIDTH; j++)
      printf("%d ", mlir_aie_read_buffer_out33(_xaie, i * LINE_WIDTH + j));
    printf("\n");
  }

  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }
  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.\n");
  return res;
}
