//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("Test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);

  mlir_aie_init_mems(_xaie, 4);

  int *mem_ptr_in_0 = mlir_aie_mem_alloc(_xaie, 0, 32);
  int *mem_ptr_in_1 = mlir_aie_mem_alloc(_xaie, 0, 32);
  int *mem_ptr_out_0 = mlir_aie_mem_alloc(_xaie, 1, 32);
  int *mem_ptr_out_1 = mlir_aie_mem_alloc(_xaie, 1, 32);

  mlir_aie_external_set_addr_ext_buffer_in_0((u64)mem_ptr_in_0);
  mlir_aie_external_set_addr_ext_buffer_in_1((u64)mem_ptr_in_1);
  mlir_aie_external_set_addr_ext_buffer_out_0((u64)mem_ptr_out_0);
  mlir_aie_external_set_addr_ext_buffer_out_1((u64)mem_ptr_out_1);
  mlir_aie_configure_shimdma_20(_xaie);

  int errors = 0;

  mlir_aie_clear_tile_memory(_xaie, 2, 3);
  mlir_aie_clear_tile_memory(_xaie, 2, 5);

  // Helper function to enable all AIE cores
  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  int i = 0;
  while (i < 25) {
    mlir_aie_acquire_of_in_lock_0(_xaie, 0, 10000);
    for (int j = 0; j < 256; j++)
      mem_ptr_in[j] = i;
    mlir_aie_sync_mem_dev(_xaie, 0);
    mlir_aie_release_of_in_lock_0(_xaie, 1, 0);

    // acquire output shim
    if (mlir_aie_acquire_of_out_cons_lock_0(_xaie, 1, 10000) == XAIE_OK)
      printf("Acquired objFifo 3 lock 0 for read\n");
    else
      printf("ERROR: timed out on objFifo 3 lock 0 for read\n");

    // check output DDR
    mlir_aie_sync_mem_cpu(_xaie, 1);
    for (int j = 0; j < 256; j++)
      mlir_aie_check("After start cores:", mem_ptr_out[j], mem_ptr_in[j],
                     errors);

    // release output shim
    if (mlir_aie_release_of_out_cons_lock_0(_xaie, 0, 10000) == XAIE_OK)
      printf("Released objFifo 3 lock 0 for write\n");
    else
      printf("ERROR: timed out on objFifo 3 lock 0 for write\n");

    i++;
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

  return res;
}
