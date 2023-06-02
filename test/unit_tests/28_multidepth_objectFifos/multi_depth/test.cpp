//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
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
  
  u32 pc0_times[0]; // track timer values

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);

  mlir_aie_acquire_lock_pc(_xaie, 0, 10000);
  mlir_aie_acquire_of_in_lock_0(_xaie, 0, 10000);

  mlir_aie_init_mems(_xaie, 2);

  int *mem_ptr_in_0 = mlir_aie_mem_alloc(_xaie, 0, 32);
  int *mem_ptr_in_1 = mlir_aie_mem_alloc(_xaie, 1, 32);

  mlir_aie_external_set_addr_ext_buffer_in_0((u64)mem_ptr_in_0);
  mlir_aie_external_set_addr_ext_buffer_in_1((u64)mem_ptr_in_1);
  mlir_aie_configure_shimdma_20(_xaie);

  int errors = 0;

  mlir_aie_clear_tile_memory(_xaie, 2, 3);
  mlir_aie_clear_tile_memory(_xaie, 2, 5);

  // Define custom EventMonitor class to track event triggers for program counter
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(2,0), 
                        XAIE_PL_MOD, 3,
                        XAIE_EVENT_LOCK_0_RELEASED_PL);
  EventMonitor pc0(_xaie, 2, 5, 0, XAIE_EVENT_BROADCAST_3_MEM,
                   XAIE_EVENT_LOCK_0_REL_MEM,
                   XAIE_EVENT_NONE_MEM, XAIE_MEM_MOD); // device, tile, PC_number, start event, stop event, reset event, mode
  pc0.set();

  // Helper function to enable all AIE cores
  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  mlir_aie_release_lock_pc(_xaie, 0, 10000);
  mlir_aie_release_of_in_lock_0(_xaie, 0, 10000);

  int i = 0;
  while (i < 4) {
    // ping
    mlir_aie_acquire_of_in_lock_0(_xaie, 0, 10000);
    for (int j = 0; j < 32; j++)
      mem_ptr_in_0[j] = i;
    mlir_aie_sync_mem_dev(_xaie, 0);
    mlir_aie_release_of_in_lock_0(_xaie, 1, 0);
    i++;

    // pong
    mlir_aie_acquire_of_in_lock_1(_xaie, 0, 10000);
    for (int j = 0; j < 32; j++)
      mem_ptr_in_1[j] = i;
    mlir_aie_sync_mem_dev(_xaie, 1);
    mlir_aie_release_of_in_lock_1(_xaie, 1, 0);
    i++;
  }

  // acquire output lock
  if (mlir_aie_acquire_lock_out(_xaie, 1, 10000) == XAIE_OK)
    printf("Acquired lock_out for read\n");
  else
    printf("ERROR: timed out on lock_out for read\n");

  pc0_times[0] = pc0.diff(); // store program counter value

  // check output
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 32; j++)
      mlir_aie_check("Output:",
                 mlir_aie_read_buffer_buff_out(_xaie, i * 32 + j), (2 * i + 1), errors);

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
  mlir_aie_deinit_libxaie(_xaie);

  return res;
}
