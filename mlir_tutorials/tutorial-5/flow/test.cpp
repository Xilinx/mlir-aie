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
#include "memory_allocator.h"

int main(int argc, char *argv[]) {
  printf("Tutorial-5 test start.\n");

  // Standard helper function for initializing and configuring AIE array.
  // The host is used to initialize/ configure/ program the AIE array.
  // ------------------------------------------------------------------------
  // aie_libxaie_ctx_t - AIE config struct
  // mlir_aie_init_device - Alloc AIE config struct
  // mlir_aie_configure_cores - Reset cores and locks. Load elfs.
  // mlir_aie_configure_switchboxes - Configure switchboxes.
  // mlir_aie_configure_dmas - Configure tile DMAs.
  // mlir_aie_initialize_locks - placeholder
  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);

  // mlir_aie_release_ddr_test_buffer_lock(_xaie, 0, 0);

  // Allocate buffer and return virtual pointer to memory
  ext_mem_model_t buf0, buf1;
  int *mem_ptr_in = mlir_aie_mem_alloc(_xaie, buf0, 256);
  int *mem_ptr_out = mlir_aie_mem_alloc(_xaie, buf1, 256);

  // Set virtual pointer used to configure
  mlir_aie_external_set_addr_ddr_test_buffer_in(_xaie, (u64)mem_ptr_in);
  mlir_aie_external_set_addr_ddr_test_buffer_out(_xaie, (u64)mem_ptr_out);
  mlir_aie_configure_shimdma_70(_xaie);
  mem_ptr_in[3] = 14;

  mlir_aie_sync_mem_dev(buf0);
  mlir_aie_configure_shimdma_70(_xaie);

  int errors = 0;

  // Clear buffer data memory
  for (int i = 0; i < 256; i++) {
    mlir_aie_write_buffer_a34(_xaie, i, 0);
  }

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

  printf("Release ddr input/output locks(1) to enable them\n");
  mlir_aie_release_ddr_test_buffer_in_lock(_xaie, 1, 0);
  mlir_aie_release_ddr_test_buffer_out_lock(_xaie, 1, 0);

  if (mlir_aie_acquire_ddr_test_buffer_out_lock(_xaie, 0, 1000) == XAIE_OK)
    printf("Acquired ddr output lock(0). Output shim dma done.\n");
  else
    printf("Timed out (1000) while trying to acquire ddr output lock (0).\n");

  mlir_aie_sync_mem_cpu(buf1); // Sync output buffer back to DDR/cache

  // Check buffer at index 3 again for expected value of 14 for tile(3,4)
  printf("Checking buf[3] = 14.\n");
  mlir_aie_check("After start cores:", mlir_aie_read_buffer_a34(_xaie, 3), 14,
                 errors);
  // Check buffer at index 5 again for expected value of 114 for tile(3,4)
  printf("Checking buf[5] = 114.\n");
  mlir_aie_check("After start cores:", mlir_aie_read_buffer_a34(_xaie, 5), 114,
                 errors);
  printf("Checking ddr_ptr[5] = 114.\n");
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
