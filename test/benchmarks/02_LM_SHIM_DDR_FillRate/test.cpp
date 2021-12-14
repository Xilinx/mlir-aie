
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
  u32 pc0_times[n];
  u32 pc1_times[n];

  for (int iters = 0; iters < n; iters++) {

    printf("test start.\n");

    size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
    XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS,
                             XAIE_ADDR_ARRAY_OFF);
    XAieGbl_HwInit(&AieConfig);
    AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
    XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

    mlir_configure_cores();
    mlir_configure_switchboxes();
    mlir_initialize_locks();
    XAieTile_LockAcquire(&(TileInst[7][0]), 2, 0, 0);
    XAieTile_LockAcquire(&(TileInst[7][1]), 3, 0, 0);

#define DMA_COUNT 7168

    static XAieGbl_MemInst *IO_Mem;
    u32 *ddr_ptr;
    IO_Mem = XAieGbl_MemInit(0);
    ddr_ptr = (u32 *)XAieGbl_MemGetPaddr(IO_Mem);
    printf("Start address of ddr buffer = %p\n", ddr_ptr);
    for (int i = 0; i < DMA_COUNT; i++) {
      XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr + i), i + 1);
    }

    for (int i = 0; i < DMA_COUNT; i++) {
      mlir_write_buffer_buf71_0(i, 0x04);
    }

    mlir_configure_dmas();

    XAieDma_Shim ShimDMAInst_7_0;
    XAieDma_ShimInitialize(&(TileInst[7][0]), &ShimDMAInst_7_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_7_0, /* bd */ 0, /* lockID */ 2,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100006000), LOW_ADDR((u64)0x20100006000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0, /* bd */ 0, HIGH_ADDR((u64)ddr_ptr),
                          LOW_ADDR((u64)ddr_ptr), /* len */ 7168 * 4);

    XAieDma_ShimBdSetAxi(&ShimDMAInst_7_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 0,
                         /* secure */ XAIE_ENABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_7_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_7_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_7_0, XAIEDMA_SHIM_CHNUM_S2MM0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_7_0, XAIEDMA_TILE_CHNUM_S2MM0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    int errors = 0;

    ACDC_check("Before", mlir_read_buffer_buf71_0(3), 4, errors);

    XAieTileMem_EventBroadcast(&TileInst[7][1], 2,
                               XAIETILE_EVENT_MEM_LOCK_3_ACQUIRED); // Start

    EventMonitor pc0(&TileInst[7][0], 0, XAIETILE_EVENT_SHIM_LOCK_2_ACQUIRED_NOC,
                 XAIETILE_EVENT_SHIM_LOCK_2_RELEASE_NOC,
                 XAIETILE_EVENT_SHIM_NONE, MODE_PL);
    pc0.set();

    XAieTile_LockRelease(&(TileInst[7][1]), 3, 0, 0);
    usleep(1000);
    XAieTile_LockRelease(&(TileInst[7][0]), 2, 1,
                         0); // for triggering the timer
    usleep(2000);
    pc0_times[iters] = pc0.diff();

    printf("After Lock Release \n");
    ACDC_print_tile_status(TileInst[7][1]);

    for (int i = 0; i < DMA_COUNT; i++) {
      printf("%x \n", XAieGbl_MemRead32(IO_Mem, (u64)(ddr_ptr + i)));
      break;
    }
  }

  computeStats(pc0_times, n);
}
