//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_clear_tile_memory(_xaie, 1, 3);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);

  int errors = 0;
  mlir_aie_write_buffer_a(_xaie, 3, 7);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  sleep(1);

  mlir_aie_check("Before release lock:", mlir_aie_read_buffer_b(_xaie, 5), 0,
                 errors);

  printf("Release input buffer lock.\n");
  mlir_aie_release_lock_a(_xaie, 1, 0);

  printf("Waiting to acquire output lock for read ...\n");
  if (mlir_aie_acquire_lock_b(_xaie, 1, LOCK_TIMEOUT)) {
    errors++;
    printf("ERROR: timeout hit!\n");
  }
  mlir_aie_check("After acquire lock:", mlir_aie_read_buffer_b(_xaie, 5), 35,
                 errors);

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
