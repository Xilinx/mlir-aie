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

#define MLIR_STACK_OFFSET 4096

#define MAP_SIZE 16UL
#define MAP_MASK (MAP_SIZE - 1)

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {

  int n = 1;
  u32 pc0_times[n];
  u32 pc1_times[n];
  u32 pc2_times[n];
  u32 pc3_times[n];

  printf("12_Stream_Delay test start.\n");
  printf("Running %d times ...\n", n);

  for (int iters = 0; iters < n; iters++) {

    aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
    mlir_aie_init_device(_xaie);
    mlir_aie_configure_cores(_xaie);
    mlir_aie_configure_switchboxes(_xaie);

    printf("Acquire input buffer lock first.\n");
    mlir_aie_acquire_lock(_xaie, 1, 3, 5, 0, 0); 

    mlir_aie_configure_dmas(_xaie);
    mlir_aie_initialize_locks(_xaie);

    mlir_aie_clear_tile_memory(_xaie, 1, 3);
    mlir_aie_clear_tile_memory(_xaie, 2, 3);

#define DMA_COUNT 512

    for (int i = 0; i < DMA_COUNT; i++) {
      mlir_aie_write_buffer_a13(_xaie, i, i + 1);
      mlir_aie_write_buffer_a43(_xaie, i, 0xdeadbeef);
    }

    // Destination Tile
    EventMonitor pc0(_xaie, 4, 3, 0, XAIE_EVENT_BROADCAST_2_MEM,
                 XAIE_EVENT_DMA_S2MM_1_FINISHED_BD_MEM,
                 XAIE_EVENT_NONE_MEM, XAIE_MEM_MOD);
    pc0.set();
    EventMonitor pc1(_xaie, 4, 3, 1, XAIE_EVENT_BROADCAST_2_MEM,
                 XAIE_EVENT_LOCK_6_REL_MEM,
                 XAIE_EVENT_NONE_MEM, XAIE_MEM_MOD);
    pc1.set();

    // Source Tile
    EventMonitor pc2(_xaie, 1, 3, 0, XAIE_EVENT_LOCK_5_ACQ_MEM,
                 XAIE_EVENT_DMA_MM2S_0_FINISHED_BD_MEM,
                 XAIE_EVENT_NONE_MEM, XAIE_MEM_MOD);
    pc2.set();
    EventMonitor pc3(_xaie, 1, 3, 1, XAIE_EVENT_LOCK_5_ACQ_MEM,
                 XAIE_EVENT_LOCK_5_REL_MEM,
                 XAIE_EVENT_NONE_MEM, XAIE_MEM_MOD);
    pc3.set();

    XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(1,3), 
                        XAIE_MEM_MOD, 2,
                        XAIE_EVENT_LOCK_5_ACQ_MEM); // Start

    mlir_aie_release_lock(_xaie, 1, 3, 5, 1, 0);

    usleep(100);

    pc0_times[iters] = pc0.diff();
    pc1_times[iters] = pc1.diff();
    pc2_times[iters] = pc2.diff();
    pc3_times[iters] = pc3.diff();

    for (int i = 0; i < DMA_COUNT; i++) {
      uint32_t d = mlir_aie_read_buffer_a13(_xaie, i);
      if (d != (i + 1)) {
        printf("Not Matched");
      }
    }

    int errors = 0;
    for (int i = 0; i < DMA_COUNT; i++) {
      uint32_t d = mlir_aie_read_buffer_a43(_xaie, i);
      if (d != (i + 1)) {
        errors++;
        printf("mismatch %x != 1 + %x\n", d, i);
        break;
      }
    }
    printf("\nHello\n");

    printf("Destination Tile: %u, %u\n", pc0.diff(), pc1.diff());

    printf("Source Tile: %u, %u\n", pc2.diff(), pc3.diff());

    mlir_aie_deinit_libxaie(_xaie);
  }
  computeStats(pc0_times, n);
  computeStats(pc1_times, n);
  computeStats(pc2_times, n);
  computeStats(pc3_times, n);
}
