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
  printf("Tutorial-2c test start.\n");

  int errors = 0;
  int num_iter = 1;
  u32 pc0_times[num_iter]; // track timer values

  // Standard helper function for initializing and configuring AIE array.
  // The host is used to initialize/ configure/ program the AIE array.
  // ------------------------------------------------------------------------
  // aie_libxaie_ctx_t - AIE config struct
  // mlir_aie_init_device - Alloc AIE config struct
  // mlir_aie_configure_cores - Reset cores and locks. Load elfs.
  // mlir_aie_configure_switchboxes - Switchboxes not used in this example.
  // mlir_aie_configure_dmas - TileDMAs not used in this example.
  // mlir_aie_initialize_locks - placeholder
  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);

  // Clear buffer data memory
  for (int i = 0; i < 256; i++) {
    mlir_aie_write_buffer_a14(_xaie, i, 0);
  }

  // Check the buffer value at index 3 to ensure it is zeroed out
  // prior to running our simple kernel.
  // ------------------------------------------------------------------------
  // mlir_aie_read_buffer_a14 - helper function to read tile local
  // memory at an offset (offset=3 in this case). _a14 maps to the
  // symbolic buffer name defined in aie.mlir.
  //
  // mlir_aie_check - helper function to compare values to expected
  // golden value and print error message to stdout and increment
  // "errors" variable if mismatch occurs.
  mlir_aie_check("Before start cores:", mlir_aie_read_buffer_a14(_xaie, 3), 0,
                 errors);

  // Performance counters
  // Trigger off start (0x00) of an AIE program
  XAie_EventPCEnable(&(_xaie->DevInst), XAie_TileLoc(1, 4), 0, 0);
  // Trigger off done (0x088) of an AIE program
  XAie_EventPCEnable(&(_xaie->DevInst), XAie_TileLoc(1, 4), 1, 240);

  // Define custom EventMonitor class to track event triggers for program
  // counter
  EventMonitor pc0(_xaie, 1, 4, 1, XAIE_EVENT_PC_0_CORE, XAIE_EVENT_PC_1_CORE,
                   XAIE_EVENT_NONE_CORE, XAIE_CORE_MOD);
  pc0.set();

  // Helper function to enable all AIE cores
  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  // Wait for lock14_0 to indicate tile(1,4) is done
  if (mlir_aie_acquire_lock14_0(_xaie, 1, 1000) == XAIE_OK)
    printf("Acquired lock14_0 (1) in tile (1,4). Done.\n");
  else
    printf("Timed out (1000) while trying to acquire lock14_0 (1).\n");

  pc0_times[0] = pc0.diff(); // store program counter value (0th iteration)

  // Check buffer at index 3 again for expected value of 14
  printf("Checking buf[3] = 14.\n");
  mlir_aie_check("After start cores:", mlir_aie_read_buffer_a14(_xaie, 3), 14,
                 errors);

  // Print Pass/Fail result of our test
  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }

  printf("\nProgram cycle counts:\n");
  // Output the timer values (average, standard deviation) for 1 iteration
  computeStats(pc0_times, 1);

  // Teardown and cleanup of AIE array
  mlir_aie_deinit_libxaie(_xaie);

  printf("Tutorial-2c test done.\n");
  return res;
}
