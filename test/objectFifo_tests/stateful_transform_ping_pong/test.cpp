//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
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
#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)
#define mlir_aie_STACK_OFFSET 4096

#define LINE_WIDTH 16
#define HEIGHT 10

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_clear_tile_memory(_xaie, 1, 2);
  mlir_aie_clear_tile_memory(_xaie, 3, 3);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_start_cores(_xaie);

  int errors = 0;

  printf("Waiting to acquire output lock for read ...\n");
  if (!mlir_aie_acquire_lock(_xaie, 3, 3, 0, 1, LOCK_TIMEOUT)) {
    printf("ERROR: timeout hit!\n");
  }

  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < LINE_WIDTH; j++) {
      mlir_aie_check("After exchange. Check [i * LINE_WIDTH + j] = i + j",
                     mlir_aie_read_buffer_out(_xaie, i * LINE_WIDTH + j),
                     (i + j), errors);
    }
  }

  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < LINE_WIDTH; j++)
      printf("%d ", mlir_aie_read_buffer_out(_xaie, i * LINE_WIDTH + j));
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
