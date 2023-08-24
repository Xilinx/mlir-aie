//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
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

#include "memory_allocator.h"
#include "test_library.h"

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

  printf("before DMA config\n");
  mlir_aie_configure_dmas(_xaie);

  int errors = 0;

  printf("before core start\n");
  mlir_aie_print_memtiledma_status(_xaie, 6, 1);
  mlir_aie_print_memtiledma_status(_xaie, 7, 1);
  mlir_aie_print_memtiledma_status(_xaie, 8, 1);

  mlir_aie_write_buffer_west(_xaie, 0, 0xDEAD);
  mlir_aie_write_buffer_west(_xaie, 4, 0xBEEF);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);
  mlir_aie_release_start_lock_1(_xaie, 2, 10000);
  mlir_aie_release_start_lock_2(_xaie, 2, 10000);
  mlir_aie_acquire_done_lock_1(_xaie, -2, 10000);
  mlir_aie_acquire_done_lock_2(_xaie, -2, 10000);

  printf("after core start\n");
  mlir_aie_print_memtiledma_status(_xaie, 6, 1);
  mlir_aie_print_memtiledma_status(_xaie, 7, 1);
  mlir_aie_print_memtiledma_status(_xaie, 8, 1);
  mlir_aie_check("After", mlir_aie_read_buffer_east(_xaie, 8), 0xDEAD, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_east(_xaie, 12), 0xBEEF, errors);

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
