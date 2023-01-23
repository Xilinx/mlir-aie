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
  printf("Tutorial-5 test start.\n");

  // Standard helper function for initializing and configuring AIE array.
  // The host is used to initialize/ configure/ program the AIE array.
  // ------------------------------------------------------------------------
  // aie_libxaie_ctx_t - AIE config struct
  // mlir_aie_init_device ##### TODO #######
  // mlir_aie_configure_cores - Reset cores and locks. Load elfs.
  // mlir_aie_configure_switchboxes - Switchboxes not used in this example.
  // mlir_aie_configure_dmas - TileDMAs not used in this example.
  // mlir_aie_initialize_locks - Locks not used in this example.
  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);

  // mlir_aie_release_ddr_test_buffer_lock(_xaie, 0, 0);

  // Configure the number of DDR memory regions we plan to sync with design
  mlir_aie_init_mems(_xaie, 2);

  // Allocate buffer and return virtual pointer to memory
  int *mem_ptr_in = mlir_aie_mem_alloc(_xaie, 0, 256);
  int *mem_ptr_out = mlir_aie_mem_alloc(_xaie, 1, 256);

  // Set virtual pointer used to configure
  mlir_aie_external_set_addr_ddr_test_buffer_in((u64)mem_ptr_in);
  mlir_aie_external_set_addr_ddr_test_buffer_out((u64)mem_ptr_out);
  mlir_aie_configure_shimdma_70(_xaie);
  mem_ptr_in[3] = 14;

  mlir_aie_sync_mem_dev(_xaie, 0);
  mlir_aie_configure_shimdma_70(_xaie);

  int errors = 0;

  // Helper function to clear tile data memory
  mlir_aie_clear_tile_memory(_xaie, 3, 4);

  // Check the buffer value at index 3 to ensure it is zeroed out
  // prior to running our simple kernel.
  // ------------------------------------------------------------------------
  // mlir_aie_read_buffer_a34 - helper function to read tile local
  // memory at an offset (offset=3 in this case). _a34 maps to the
  // symbolic buffer name defined in aie.mlir.
  //
  // mlir_aie_check - helper function to compare values to expected
  // golden value and print error message to stdout and increment
  // "errors" variable if mismatch occurs.
  mlir_aie_check("Before start cores:", mlir_aie_read_buffer_a34(_xaie, 5), 0,
                 errors);

  // Helper function to enable all AIE cores
  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  mlir_aie_release_ddr_test_buffer_in_lock(_xaie, 1, 0);
  mlir_aie_release_ddr_test_buffer_out_lock(_xaie, 1, 0);

  // Wait time for cores to run. Number used here is much larger than needed.
  usleep(100);

  mlir_aie_release_ddr_test_buffer_out_lock(_xaie, 0, 0);
  mlir_aie_sync_mem_cpu(_xaie, 1); // Sync output buffer back to DDR/cache

  // Check buffer at index 3 again for expected value of 14 for tile(3,4)
  mlir_aie_check("After start cores:", mlir_aie_read_buffer_a34(_xaie, 3), 14,
                 errors);
  // Check buffer at index 5 again for expected value of 114 for tile(3,4)
  mlir_aie_check("After start cores:", mlir_aie_read_buffer_a34(_xaie, 5), 114,
                 errors);
  mlir_aie_check("After start cores:", mem_ptr_out[5], 114, errors);

  // Print Pass/Fail result of our test
  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }

  // Teardown and cleanup of AIE array
  mlir_aie_deinit_libxaie(_xaie);

  printf("Tutorial-5 test done.\n");
  return res;
}
