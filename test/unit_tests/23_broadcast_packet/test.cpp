//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
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
#define MAP_SIZE 16UL
#define MAP_MASK (MAP_SIZE - 1)

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  printf("before configure cores.\n");
  mlir_aie_configure_cores(_xaie);

  printf("before configure sw.\n");
  mlir_aie_configure_switchboxes(_xaie);

  printf("before DMA config\n");
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);
  int errors = 0;

  printf("Finish configure\n");
  #define DMA_COUNT 1024

  mlir_aie_clear_tile_memory(_xaie, 7, 2);
  mlir_aie_clear_tile_memory(_xaie, 7, 3);
  mlir_aie_clear_tile_memory(_xaie, 7, 4);
  mlir_aie_clear_tile_memory(_xaie, 6, 3);
  mlir_aie_clear_tile_memory(_xaie, 6, 4);

  for (int bd = 0; bd < DMA_COUNT; bd++) {
    mlir_aie_write_buffer_buf72_0(_xaie, bd, 720);
    mlir_aie_write_buffer_buf72_1(_xaie, bd, 721);
  }

  printf("before core start\n");
  mlir_aie_start_cores(_xaie);

  mlir_aie_release_lock72_4(_xaie, 1, 0);
  mlir_aie_release_lock72_5(_xaie, 1, 0);

  if (mlir_aie_acquire_lock63_0(_xaie, 1, 1000) == XAIE_OK)
    printf("Acquired lock63_0 (1) in tile (6,3). Done.\n");
  else {
    errors++;
    printf("Timed out while trying to acquire lock63_0.\n");
  }
  if (mlir_aie_acquire_lock64_0(_xaie, 1, 1000) == XAIE_OK)
    printf("Acquired lock64_0 (1) in tile (6,4). Done.\n");
  else {
    errors++;
    printf("Timed out while trying to acquire lock64_0.\n");
  }
  if (mlir_aie_acquire_lock73_0(_xaie, 1, 1000) == XAIE_OK)
    printf("Acquired lock73_0 (1) in tile (7,3). Done.\n");
  else {
    errors++;
    printf("Timed out while trying to acquire lock73_0.\n");
  }
  if (mlir_aie_acquire_lock74_0(_xaie, 1, 1000) == XAIE_OK)
    printf("Acquired lock74_0 (1) in tile (7,4). Done.\n");
  else {
    errors++;
    printf("Timed out while trying to acquire lock74_0.\n");
  }

  for (int bd = 0; bd < DMA_COUNT; bd++) {
    mlir_aie_check("After release lock:",
                   mlir_aie_read_buffer_buf72_0(_xaie, bd), 720, errors);
    mlir_aie_check("After release lock:",
                   mlir_aie_read_buffer_buf72_1(_xaie, bd), 721, errors);
    mlir_aie_check("After release lock:",
                   mlir_aie_read_buffer_buf63_0(_xaie, bd), 720, errors);
    mlir_aie_check("After release lock:",
                   mlir_aie_read_buffer_buf73_0(_xaie, bd), 720, errors);
    mlir_aie_check("After release lock:",
                   mlir_aie_read_buffer_buf64_0(_xaie, bd), 721, errors);
    mlir_aie_check("After release lock:",
                   mlir_aie_read_buffer_buf74_0(_xaie, bd), 721, errors);
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
