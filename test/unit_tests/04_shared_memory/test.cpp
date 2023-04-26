//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <xaiengine.h>
#include "test_library.h"

#define LOCK_TIMEOUT 100

#include "aie_inc.cpp"

int
main(int argc, char *argv[])
{
    printf("test start.\n");

    aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
    mlir_aie_init_device(_xaie);

    mlir_aie_clear_tile_memory(_xaie, 1, 3);
    mlir_aie_clear_tile_memory(_xaie, 1, 4);

    mlir_aie_configure_cores(_xaie);
    mlir_aie_configure_switchboxes(_xaie);
    mlir_aie_configure_dmas(_xaie);
    mlir_aie_initialize_locks(_xaie);

    int errors = 0;

    printf("Acquire input buffer lock first.\n");
    if (mlir_aie_acquire_input_lock(_xaie, 0, LOCK_TIMEOUT) == XAIE_OK)
      printf("Acquired input_lock (0) in tile (1,3). Done.\n");
    else {
      errors++;
      printf("Timed out while trying to acquire input_lock.\n");
    }
    mlir_aie_write_buffer_a(_xaie, 3, 7);

    mlir_aie_check("Before", mlir_aie_read_buffer_a(_xaie, 3), 7, errors);
    mlir_aie_check("Before", mlir_aie_read_buffer_b(_xaie, 5), 0, errors);
    mlir_aie_check("Before", mlir_aie_read_buffer_c(_xaie, 5), 0, errors);

    printf("Starting cores\n");
    mlir_aie_start_cores(_xaie);

    mlir_aie_check("Before and started", mlir_aie_read_buffer_a(_xaie, 3), 7,
                   errors);
    mlir_aie_check("Before and started", mlir_aie_read_buffer_b(_xaie, 5), 0,
                   errors);
    mlir_aie_check("Before and started", mlir_aie_read_buffer_c(_xaie, 5), 0,
                   errors);

    printf("Release input buffer lock.\n");
    mlir_aie_release_input_lock(_xaie, 1, 0);
    printf("Waiting to acquire output lock for read ...\n");
    if (mlir_aie_acquire_output_lock(_xaie, 1, LOCK_TIMEOUT) == XAIE_OK)
      printf("Acquired output_lock (1) in tile (1,4). Done.\n");
    else {
      errors++;
      printf("Timed out while trying to acquire output_lock.\n");
    }

    mlir_aie_check("After", mlir_aie_read_buffer_a(_xaie, 3), 7, errors);
    mlir_aie_check("After", mlir_aie_read_buffer_b(_xaie, 5), 35, errors);
    mlir_aie_check("After", mlir_aie_read_buffer_c(_xaie, 5), 175, errors);

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
