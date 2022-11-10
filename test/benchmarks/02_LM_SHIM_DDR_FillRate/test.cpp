
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
  int n = 100;
  u32 pc0_times[n];
  u32 pc1_times[n];

  printf("02_LM_SHIM_DDR_FillRate test start.\n");
  printf("Running %d times ...\n", n);

  for (int iters = 0; iters < n; iters++) {


    aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
    mlir_aie_init_device(_xaie);
    mlir_aie_configure_cores(_xaie);
    mlir_aie_configure_switchboxes(_xaie);
    mlir_aie_initialize_locks(_xaie);
    mlir_aie_acquire_lock(_xaie, 7, 0, 2, 0, 0); 
    mlir_aie_acquire_lock(_xaie, 7, 1, 3, 0, 0); 

    #define DMA_COUNT 7168

    // printf("Acquired before\n");
    mlir_aie_configure_dmas(_xaie);
    mlir_aie_init_mems(_xaie, 1);

    int *ddr_ptr = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT);
    for(int i=0; i<DMA_COUNT; i++) {
      *(ddr_ptr + i) = 0xdeadbeef;
    }
    mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2

    // printf("Start address of ddr buffer = %p\n", ddr_ptr);

    for (int i = 0; i < DMA_COUNT; i++) {
      mlir_aie_write_buffer_buf71_0(_xaie, i, i+1);
    }

    // XAie_DmaDesc dma_tile70_bd0;
    // XAie_DmaDescInit(&(_xaie->DevInst), &(dma_tile70_bd0), XAie_TileLoc(7,0));
    // XAie_DmaSetLock(&(dma_tile70_bd0), XAie_LockInit(2,1),XAie_LockInit(2,0));
    // XAie_DmaSetAddrLen(&(dma_tile70_bd0),  /* addr */ (u64)(ddr_ptr),  /* len */ 7168 * 4);
    // XAie_DmaSetAxi(&(dma_tile70_bd0), /* smid */ 0, /* burstlen */ 16, /* QoS */ 0 , /* Cache */ 0, /* Secure */ XAIE_ENABLE);
    // XAie_DmaSetNextBd(&(dma_tile70_bd0),  /* nextbd */ 0,  /* enableNextBd */ 0);
    // XAie_DmaEnableBd(&(dma_tile70_bd0));
    // XAie_DmaWriteBd(&(_xaie->DevInst), &(dma_tile70_bd0), XAie_TileLoc(7,0),  /* bd */ 0);
    // XAie_DmaChannelPushBdToQueue(&(_xaie->DevInst), XAie_TileLoc(7,0), /* ChNum */0, /* dmaDir */ DMA_S2MM, /* BdNum */0);
    // XAie_DmaChannelEnable(&(_xaie->DevInst), XAie_TileLoc(7,0), /* ChNum */ 0, /* dmaDir */ DMA_S2MM);
    // XAie_EnableAieToShimDmaStrmPort(&(_xaie->DevInst), XAie_TileLoc(7,0), 2);

    mlir_aie_external_set_addr_myBuffer_70_0((u64)ddr_ptr);
    mlir_aie_configure_shimdma_70(_xaie);

    mlir_aie_start_cores(_xaie);

    int errors = 0;

    mlir_aie_check("Before", mlir_aie_read_buffer_buf71_0(_xaie, 3), 4, errors);

    XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(7,1), XAIE_MEM_MOD, 2,
                               XAIE_EVENT_LOCK_3_ACQ_MEM); // Start

    EventMonitor pc0(_xaie, 7, 0, 0, XAIE_EVENT_LOCK_2_ACQUIRED_PL,
                 XAIE_EVENT_LOCK_2_RELEASED_PL,
                 XAIE_EVENT_NONE_PL, XAIE_PL_MOD);
    pc0.set();

    mlir_aie_release_lock(_xaie, 7, 1, 3, 0, 0);
    usleep(1000);
    mlir_aie_release_lock(_xaie, 7, 0, 2, 1,
                         0); // for triggering the timer
    usleep(2000);
    pc0_times[iters] = pc0.diff();

    // printf("After Lock Release \n");
    // mlir_aie_print_tile_status(_xaie, 7, 1);

    mlir_aie_sync_mem_cpu(_xaie, 0); // only used in libaiev2

    for (int i = 0; i < DMA_COUNT; i++) {
      // int i = 0
      uint32_t d = *(ddr_ptr + i);
      if(d != (i + 1)) {
        printf("mismatch %x != 1 + %x\n", d, i);
      }
    }

    mlir_aie_deinit_libxaie(_xaie);
  }

  computeStats(pc0_times, n);
}
