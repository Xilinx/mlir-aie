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

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)

#define MAP_SIZE 16UL
#define MAP_MASK (MAP_SIZE - 1)

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  int n = 1;
  u32 pc2_times[n];

  printf("01_DDR_SHIM_LM_FillRate test start\n");
  printf("NEW\n");

  for (int iters = 0; iters < n; iters++) {

    aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
    mlir_aie_init_device(_xaie);

    mlir_aie_print_tile_status(_xaie, 7, 1);

    // Run auto generated config functions
    mlir_aie_configure_cores(_xaie);
    mlir_aie_configure_switchboxes(_xaie);
    mlir_aie_initialize_locks(_xaie);

    #define DMA_COUNT 7168

    mlir_aie_configure_dmas(_xaie);
    mlir_aie_init_mems(_xaie, 1);

    int *ddr_ptr = mlir_aie_mem_alloc(_xaie, 0, 0x4000 + 0x020100000000LL, DMA_COUNT);
    for(int i=0; i<DMA_COUNT; i++) {
      *(ddr_ptr + i) = i + 1;
    }
    mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2

    // XAie_DmaDesc dma_tile70_bd0;
    // XAie_DmaDescInit(&(_xaie->DevInst), &(dma_tile70_bd0), XAie_TileLoc(7,0));
    // XAie_DmaSetLock(&(dma_tile70_bd0), XAie_LockInit(1,1),XAie_LockInit(1,0));
    // XAie_DmaSetAddrLen(&(dma_tile70_bd0),  /* addr */ (u64)(ddr_ptr),  /* len */ 7168 * 4);
    // XAie_DmaSetAxi(&(dma_tile70_bd0), /* smid */ 0, /* burstlen */ 16, /* QoS */ 0 , /* Cache */ 1, /* Secure */ XAIE_DISABLE);
    // XAie_DmaSetNextBd(&(dma_tile70_bd0),  /* nextbd */ 0,  /* enableNextBd */ 0);
    // XAie_DmaEnableBd(&(dma_tile70_bd0));
    // XAie_DmaWriteBd(&(_xaie->DevInst), &(dma_tile70_bd0), XAie_TileLoc(7,0),  /* bd */ 0);
    // XAie_DmaChannelPushBdToQueue(&(_xaie->DevInst), XAie_TileLoc(7,0), /* ChNum */0, /* dmaDir */ DMA_MM2S, /* BdNum */0);
    // XAie_DmaChannelEnable(&(_xaie->DevInst), XAie_TileLoc(7,0), /* ChNum */ 0, /* dmaDir */ DMA_MM2S);
    // XAie_EnableShimDmaToAieStrmPort(&(_xaie->DevInst), XAie_TileLoc(7,0), 3);

    mlir_aie_external_set_addr_myBuffer_70_0((u64)ddr_ptr);
    mlir_aie_configure_shimdma_70(_xaie);

    mlir_aie_start_cores(_xaie);

    // We're going to stamp over the memory
    for (int i = 0; i < DMA_COUNT; i++) {
      mlir_aie_write_buffer_buf71_0(_xaie, i, 0xdeadbeef);
    }

    XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(7,0), 
                        /*XAie_ModuleType*/ XAIE_PL_MOD, /*u8 BroadcastId*/ 2, 
                        /*XAie_Events*/ XAIE_EVENT_LOCK_1_ACQUIRED_PL);

    XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(7,0), 
                        /*XAie_ModuleType*/ XAIE_PL_MOD, /*u8 BroadcastId*/ 3, 
                        /*XAie_Events*/ XAIE_EVENT_LOCK_1_RELEASED_PL);

    EventMonitor pc2(_xaie, 7, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                 XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                 XAIE_MEM_MOD);
    pc2.set();

    // iterate over the buffer
    usleep(1000);
    mlir_aie_release_lock(_xaie, 7, 0, 1, 1, 0); // Release lock for reading from DDR

    usleep(2000);

    pc2_times[iters] = pc2.diff();

    int errors = 0;
    for (int i = 0; i < DMA_COUNT; i++) {
      uint32_t d = mlir_aie_read_buffer_buf71_0(_xaie, i);
      if (d != (i + 1)) {
        errors++;
        printf("mismatch %x != 1 + %x\n", d, i);
      }
    }
    mlir_aie_deinit_libxaie(_xaie);
  }

  computeStats(pc2_times, n);
}
