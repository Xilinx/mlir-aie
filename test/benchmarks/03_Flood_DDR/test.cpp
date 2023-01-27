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

#include "aie_inc.cpp"

#define DMA_COUNT 7168

int main(int argc, char *argv[]) {

  int n = 1;
  u32 pc7_times[n];
  u32 pc6_times[n];
  u32 pc2_times[n];
  u32 pc3_times[n];
  u32 pc10_times[n];
  u32 pc11_times[n];
  u32 pc18_times[n];
  u32 pc19_times[n];
  u32 pc26_times[n];
  u32 pc27_times[n];
  u32 pc34_times[n];
  u32 pc35_times[n];
  u32 pc42_times[n];
  u32 pc43_times[n];
  u32 pc46_times[n];
  u32 pc47_times[n];

  printf("03_Flood_DDR test start.\n");
  printf("Running %d times ...\n", n);

  for (int iters = 0; iters < n; iters++) {

    // Run auto generated config functions
    aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
    mlir_aie_init_device(_xaie);

    // Run auto generated config functions
    mlir_aie_configure_cores(_xaie);
    mlir_aie_configure_switchboxes(_xaie);
    mlir_aie_initialize_locks(_xaie);

    mlir_aie_acquire_lock(_xaie, 7, 0, 2, 0, 0); 
    mlir_aie_acquire_lock(_xaie, 7, 1, 3, 0, 0); 

    mlir_aie_configure_dmas(_xaie);
    mlir_aie_init_mems(_xaie, 1);

    int *ddr_ptr = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT * 16);
    // int *ddr_ptr7 = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT);
    // int *phy_addr_ptr = (int*)XAie_MemGetPAddr(_xaie->buffers[0]);
    printf("virtual addr is %x\n", (u64)ddr_ptr);

    int *ddr_ptr2  = ddr_ptr;
    int *ddr_ptr3  = ddr_ptr + 1*DMA_COUNT;
    int *ddr_ptr6  = ddr_ptr + 2*DMA_COUNT;
    int *ddr_ptr7  = ddr_ptr + 3*DMA_COUNT;
    int *ddr_ptr10 = ddr_ptr + 4*DMA_COUNT;
    int *ddr_ptr11 = ddr_ptr + 5*DMA_COUNT;
    int *ddr_ptr18 = ddr_ptr + 6*DMA_COUNT;
    int *ddr_ptr19 = ddr_ptr + 7*DMA_COUNT;
    int *ddr_ptr26 = ddr_ptr + 8*DMA_COUNT;
    int *ddr_ptr27 = ddr_ptr + 9*DMA_COUNT;
    int *ddr_ptr34 = ddr_ptr + 10*DMA_COUNT;
    int *ddr_ptr35 = ddr_ptr + 11*DMA_COUNT;
    int *ddr_ptr42 = ddr_ptr + 12*DMA_COUNT;
    int *ddr_ptr43 = ddr_ptr + 13*DMA_COUNT;
    int *ddr_ptr46 = ddr_ptr + 14*DMA_COUNT;
    int *ddr_ptr47 = ddr_ptr + 15*DMA_COUNT;

    // int pos = 1;
    // int *ddr_ptr3  = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr6  = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr7  = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr10 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr11 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr18 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr19 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr26 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr27 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr34 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr35 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr42 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr43 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr46 = ddr_ptr + pos*DMA_COUNT;
    // int *ddr_ptr47 = ddr_ptr + pos*DMA_COUNT;

    for(int i=0; i<DMA_COUNT; i++) {
      *(ddr_ptr2 + i) = i + 1;
      *(ddr_ptr3 + i) = i + 1;
      *(ddr_ptr6 + i) = i + 1;
      *(ddr_ptr7 + i) = i + 1;
      *(ddr_ptr10 + i) = i + 1;
      *(ddr_ptr11 + i) = i + 1;
      *(ddr_ptr18 + i) = i + 1;
      *(ddr_ptr19 + i) = i + 1;
      *(ddr_ptr26 + i) = i + 1;
      *(ddr_ptr27 + i) = i + 1;
      *(ddr_ptr34 + i) = i + 1;
      *(ddr_ptr35 + i) = i + 1;
      *(ddr_ptr42 + i) = i + 1;
      *(ddr_ptr43 + i) = i + 1;
      *(ddr_ptr46 + i) = i + 1;
      *(ddr_ptr47 + i) = i + 1;
    }

    mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2

    mlir_aie_external_set_addr_myBuffer_20_0((u64)ddr_ptr2);
    mlir_aie_configure_shimdma_20(_xaie);
    mlir_aie_external_set_addr_myBuffer_30_0((u64)ddr_ptr3);
    mlir_aie_configure_shimdma_30(_xaie);
    mlir_aie_external_set_addr_myBuffer_60_0((u64)ddr_ptr6);
    mlir_aie_configure_shimdma_60(_xaie);
    mlir_aie_external_set_addr_myBuffer_70_0((u64)ddr_ptr7);
    mlir_aie_configure_shimdma_70(_xaie);
    mlir_aie_external_set_addr_myBuffer_100_0((u64)ddr_ptr10);
    mlir_aie_configure_shimdma_100(_xaie);
    mlir_aie_external_set_addr_myBuffer_110_0((u64)ddr_ptr11);
    mlir_aie_configure_shimdma_110(_xaie);
    mlir_aie_external_set_addr_myBuffer_180_0((u64)ddr_ptr18);
    mlir_aie_configure_shimdma_180(_xaie);
    mlir_aie_external_set_addr_myBuffer_190_0((u64)ddr_ptr19);
    mlir_aie_configure_shimdma_190(_xaie);
    mlir_aie_external_set_addr_myBuffer_260_0((u64)ddr_ptr26);
    mlir_aie_configure_shimdma_260(_xaie);
    mlir_aie_external_set_addr_myBuffer_270_0((u64)ddr_ptr27);
    mlir_aie_configure_shimdma_270(_xaie);
    mlir_aie_external_set_addr_myBuffer_340_0((u64)ddr_ptr34);
    mlir_aie_configure_shimdma_340(_xaie);
    mlir_aie_external_set_addr_myBuffer_350_0((u64)ddr_ptr35);
    mlir_aie_configure_shimdma_350(_xaie);
    mlir_aie_external_set_addr_myBuffer_420_0((u64)ddr_ptr42);
    mlir_aie_configure_shimdma_420(_xaie);
    mlir_aie_external_set_addr_myBuffer_430_0((u64)ddr_ptr43);
    mlir_aie_configure_shimdma_430(_xaie);
    mlir_aie_external_set_addr_myBuffer_460_0((u64)ddr_ptr46);
    mlir_aie_configure_shimdma_460(_xaie);
    mlir_aie_external_set_addr_myBuffer_470_0((u64)ddr_ptr47);
    mlir_aie_configure_shimdma_470(_xaie);


    // printf("Start cores\n");
    mlir_aie_start_cores(_xaie);


    // We're going to stamp over the memory
    for (int i = 0; i < DMA_COUNT; i++) {
      mlir_aie_write_buffer_buf21_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf31_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf61_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf71_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf101_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf111_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf181_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf191_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf261_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf271_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf341_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf351_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf421_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf431_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf461_0(_xaie, i, 0xdeadbeef);
      mlir_aie_write_buffer_buf471_0(_xaie, i, 0xdeadbeef);
    }

    XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(2,0), XAIE_PL_MOD, 2,
                               XAIE_EVENT_LOCK_1_ACQUIRED_PL); // Start

    XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(7,0), XAIE_PL_MOD, 2,
                               XAIE_EVENT_LOCK_1_ACQUIRED_PL); // Start

    std::vector<int> shim_cols = {2,  3,  6,  7,  10, 11, 18,
                                  19, 26, 27, 34, 35, 42, 46};

    EventMonitor pc2(_xaie, 2, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                 XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                 XAIE_MEM_MOD);
    pc2.set();
    EventMonitor pc3(_xaie, 3, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                 XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                 XAIE_MEM_MOD);
    pc3.set();
    EventMonitor pc6(_xaie, 6, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                 XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                 XAIE_MEM_MOD);
    pc6.set();
    EventMonitor pc7(_xaie, 7, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                 XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                 XAIE_MEM_MOD);
    pc7.set();
    EventMonitor pc10(_xaie, 10, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc10.set();
    EventMonitor pc11(_xaie, 11, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc11.set();
    EventMonitor pc18(_xaie, 18, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc18.set();
    EventMonitor pc19(_xaie, 19, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc19.set();
    EventMonitor pc26(_xaie, 26, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc26.set();
    EventMonitor pc27(_xaie, 27, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc27.set();
    EventMonitor pc34(_xaie, 34, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc34.set();
    EventMonitor pc35(_xaie, 35, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc35.set();
    EventMonitor pc42(_xaie, 42, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc42.set();
    EventMonitor pc43(_xaie, 43, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc43.set();
    EventMonitor pc46(_xaie, 46, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc46.set();
    EventMonitor pc47(_xaie, 47, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                  XAIE_EVENT_LOCK_0_REL_MEM, XAIE_EVENT_NONE_MEM,
                  XAIE_MEM_MOD);
    pc47.set();

    // iterate over the buffer
    usleep(1000);
    // XAie_StartTransaction(&(_xaie->DevInst),
    // XAIE_TRANSACTION_DISABLE_AUTO_FLUSH); Release shim_locks for reading from
    // DDR
    for (int col : shim_cols)
      mlir_aie_release_lock(_xaie, col, 0, 1, 1, 0);
    // XAie_SubmitTransaction(&(_xaie->DevInst), NULL);

    usleep(5000);
    pc2_times[iters] = pc2.diff();
    pc3_times[iters] = pc3.diff();
    pc6_times[iters] = pc6.diff();
    pc7_times[iters] = pc7.diff();
    pc10_times[iters] = pc10.diff();
    pc11_times[iters] = pc11.diff();
    pc18_times[iters] = pc18.diff();
    pc19_times[iters] = pc19.diff();
    pc26_times[iters] = pc26.diff();
    pc27_times[iters] = pc27.diff();
    pc34_times[iters] = pc34.diff();
    pc35_times[iters] = pc35.diff();
    pc42_times[iters] = pc42.diff();
    pc43_times[iters] = pc43.diff();
    pc46_times[iters] = pc46.diff();
    pc47_times[iters] = pc47.diff();

    int errors = 0;

    mlir_aie_deinit_libxaie(_xaie);
  }

  computeStats(pc2_times, n);
  computeStats(pc3_times, n);
  computeStats(pc6_times, n);
  computeStats(pc7_times, n);
  computeStats(pc10_times, n);
  computeStats(pc11_times, n);
  computeStats(pc18_times, n);
  computeStats(pc19_times, n);
  computeStats(pc26_times, n);
  computeStats(pc27_times, n);
  computeStats(pc34_times, n);
  computeStats(pc35_times, n);
  computeStats(pc42_times, n);
  computeStats(pc43_times, n);
  computeStats(pc46_times, n);
  computeStats(pc47_times, n);
}
