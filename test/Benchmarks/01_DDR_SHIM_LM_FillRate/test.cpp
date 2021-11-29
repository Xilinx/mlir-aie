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

#define XAIE_NUM_ROWS 8
#define XAIE_NUM_COLS 50
#define XAIE_ADDR_ARRAY_OFF 0x800

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)

#define MAP_SIZE 16UL
#define MAP_MASK (MAP_SIZE - 1)

namespace {

XAieGbl_Config *AieConfigPtr; /**< AIE configuration pointer */
XAieGbl AieInst;              /**< AIE global instance */
XAieGbl_HwCfg AieConfig;      /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS +
                                     1]; /**< Instantiates AIE array of
                                            [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS + 1];

#include "aie_inc.cpp"

} // namespace

int main(int argc, char *argv[]) {
  int n = 1;
  u32 pc2_times[n];

  for (int iters = 0; iters < n; iters++) {

    auto col = 7;

    size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
    XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS,
                             XAIE_ADDR_ARRAY_OFF);
    XAieGbl_HwInit(&AieConfig);
    AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
    XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

    ACDC_print_tile_status(TileInst[7][1]);

    // Run auto generated config functions

    mlir_configure_cores();
    mlir_configure_switchboxes();
    mlir_initialize_locks();

    static XAieGbl_MemInst *IO_Mem;
    u32 *ddr_ptr;

#define DMA_COUNT 7168

    IO_Mem = XAieGbl_MemInit(0);
    ddr_ptr = (u32 *)XAieGbl_MemGetPaddr(IO_Mem);
    printf("Start address of ddr buffer = %p\n", ddr_ptr);
    for (int i = 0; i < DMA_COUNT; i++) {
      // if ( i < 10){
      //   printf("%p\n", ddr_ptr + i);
      // }
      XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr + i), i + 1);
    }

    u32 *ddr_ptr2 = ddr_ptr + DMA_COUNT;
    printf("DDR_PTR NEW, %p \n", ddr_ptr2);

    printf("Acquired before");
    // XAieTile_LockAcquire(&(TileInst[7][1]), 0, 0, 0);
    mlir_configure_dmas();

    XAieDma_Shim ShimDMAInst_7_0;
    XAieDma_ShimInitialize(&(TileInst[7][0]), &ShimDMAInst_7_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_7_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0, /* bd */ 0, HIGH_ADDR((u64)ddr_ptr),
                          LOW_ADDR((u64)ddr_ptr), /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_7_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_7_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_7_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_7_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_7_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    // We're going to stamp over the memory
    for (int i = 0; i < DMA_COUNT; i++) {
      mlir_write_buffer_buf71_0(i, 0xdeadbeef);
    }

    XAieTilePl_EventBroadcast(&TileInst[7][0], 2,
                              XAIETILE_EVENT_SHIM_LOCK_1_ACQUIRED_NOC); // Start
    XAieTilePl_EventBroadcast(&TileInst[7][0], 3,
                              XAIETILE_EVENT_SHIM_LOCK_1_RELEASE_NOC); // Stop

    EventMon pc2(&TileInst[7][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                 XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                 MODE_MEM);
    pc2.set();

    // iterate over the buffer
    usleep(1000);
    XAieTile_LockRelease(&(TileInst[7][0]), 1, 1,
                         0); // Release lock for reading from DDR

    usleep(2000);

    pc2_times[iters] = pc2.diff();

    int errors = 0;
    for (int i = 0; i < DMA_COUNT; i++) {
      uint32_t d = mlir_read_buffer_buf71_0(i);
      if (d != (i + 1)) {
        errors++;
        printf("mismatch %x != 1 + %x\n", d, i);
      }
    }
  }

  computeStats(pc2_times, n);
}
