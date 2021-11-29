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

  static XAieGbl_MemInst *IO_Mem;
  IO_Mem = XAieGbl_MemInit(0);

  u32 *ddr_ptr2;
  ddr_ptr2 = (u32 *)XAieGbl_MemGetPaddr(IO_Mem);
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr2 + i), i + 1);
  }

  u32 *ddr_ptr3 = ddr_ptr2 + DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr3 + i), i + 1);
  }

  u32 *ddr_ptr6 = ddr_ptr2 + 2 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr6 + i), i + 1);
  }

  u32 *ddr_ptr7 = ddr_ptr2 + 3 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr7 + i), i + 1);
  }

  u32 *ddr_ptr10 = ddr_ptr2 + 4 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr10 + i), i + 1);
  }

  u32 *ddr_ptr11 = ddr_ptr2 + 5 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr11 + i), i + 1);
  }

  u32 *ddr_ptr18 = ddr_ptr2 + 6 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr18 + i), i + 1);
  }

  u32 *ddr_ptr19 = ddr_ptr2 + 7 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr19 + i), i + 1);
  }

  u32 *ddr_ptr26 = ddr_ptr2 + 8 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr26 + i), i + 1);
  }

  u32 *ddr_ptr27 = ddr_ptr2 + 9 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr27 + i), i + 1);
  }

  u32 *ddr_ptr34 = ddr_ptr2 + 10 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr34 + i), i + 1);
  }

  u32 *ddr_ptr35 = ddr_ptr2 + 11 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr35 + i), i + 1);
  }

  u32 *ddr_ptr42 = ddr_ptr2 + 12 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr42 + i), i + 1);
  }

  u32 *ddr_ptr43 = ddr_ptr2 + 13 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr43 + i), i + 1);
  }

  u32 *ddr_ptr46 = ddr_ptr2 + 14 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr46 + i), i + 1);
  }

  u32 *ddr_ptr47 = ddr_ptr2 + 15 * DMA_COUNT;
  for (int i = 0; i < DMA_COUNT; i++) {
    XAieGbl_MemWrite32(IO_Mem, (u64)(ddr_ptr47 + i), i + 1);
  }

  for (int iters = 0; iters < n; iters++) {

    size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
    XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS,
                             XAIE_ADDR_ARRAY_OFF);
    XAieGbl_HwInit(&AieConfig);
    AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
    XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

    // Run auto generated config functions

    mlir_configure_cores();
    mlir_configure_switchboxes();
    mlir_initialize_locks();
    mlir_configure_dmas();

    XAieDma_Shim ShimDMAInst_2_0;
    XAieDma_ShimInitialize(&(TileInst[2][0]), &ShimDMAInst_2_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_2_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_2_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr2), LOW_ADDR((u64)ddr_ptr2),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_2_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_2_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_2_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_2_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_2_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_3_0;
    XAieDma_ShimInitialize(&(TileInst[3][0]), &ShimDMAInst_3_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_3_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_3_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr3), LOW_ADDR((u64)ddr_ptr3),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_3_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_3_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_3_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_3_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_3_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_6_0;
    XAieDma_ShimInitialize(&(TileInst[6][0]), &ShimDMAInst_6_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_6_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_6_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr6), LOW_ADDR((u64)ddr_ptr6),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_6_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_6_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_6_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_6_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_6_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_7_0;
    XAieDma_ShimInitialize(&(TileInst[7][0]), &ShimDMAInst_7_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_7_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr7), LOW_ADDR((u64)ddr_ptr7),
                          /* len */ 7168 * 4);
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

    XAieDma_Shim ShimDMAInst_10_0;
    XAieDma_ShimInitialize(&(TileInst[10][0]), &ShimDMAInst_10_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_10_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_10_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr10), LOW_ADDR((u64)ddr_ptr10),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_10_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_10_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_10_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_10_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_10_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_11_0;
    XAieDma_ShimInitialize(&(TileInst[11][0]), &ShimDMAInst_11_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_11_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_11_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr11), LOW_ADDR((u64)ddr_ptr11),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_11_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_11_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_11_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_11_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_11_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_18_0;
    XAieDma_ShimInitialize(&(TileInst[18][0]), &ShimDMAInst_18_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_18_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_18_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr18), LOW_ADDR((u64)ddr_ptr18),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_18_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_18_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_18_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_18_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_18_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_19_0;
    XAieDma_ShimInitialize(&(TileInst[19][0]), &ShimDMAInst_19_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_19_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_19_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr19), LOW_ADDR((u64)ddr_ptr19),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_19_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_19_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_19_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_19_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_19_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_26_0;
    XAieDma_ShimInitialize(&(TileInst[26][0]), &ShimDMAInst_26_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_26_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_26_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr26), LOW_ADDR((u64)ddr_ptr26),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_26_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_26_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_26_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_26_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_26_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_27_0;
    XAieDma_ShimInitialize(&(TileInst[27][0]), &ShimDMAInst_27_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_27_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_27_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr27), LOW_ADDR((u64)ddr_ptr27),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_27_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_27_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_27_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_27_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_27_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_34_0;
    XAieDma_ShimInitialize(&(TileInst[34][0]), &ShimDMAInst_34_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_34_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_34_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr34), LOW_ADDR((u64)ddr_ptr34),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_34_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_34_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_34_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_34_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_34_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_35_0;
    XAieDma_ShimInitialize(&(TileInst[35][0]), &ShimDMAInst_35_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_35_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_35_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr35), LOW_ADDR((u64)ddr_ptr35),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_35_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_35_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_35_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_35_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_35_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_42_0;
    XAieDma_ShimInitialize(&(TileInst[42][0]), &ShimDMAInst_42_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_42_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_42_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr42), LOW_ADDR((u64)ddr_ptr42),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_42_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_42_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_42_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_42_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_42_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_43_0;
    XAieDma_ShimInitialize(&(TileInst[43][0]), &ShimDMAInst_43_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_43_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_43_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr43), LOW_ADDR((u64)ddr_ptr43),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_43_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_43_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_43_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_43_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_43_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_46_0;
    XAieDma_ShimInitialize(&(TileInst[46][0]), &ShimDMAInst_46_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_46_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_46_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr46), LOW_ADDR((u64)ddr_ptr46),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_46_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_46_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_46_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_46_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_46_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    XAieDma_Shim ShimDMAInst_47_0;
    XAieDma_ShimInitialize(&(TileInst[47][0]), &ShimDMAInst_47_0);
    XAieDma_ShimBdSetLock(&ShimDMAInst_47_0, /* bd */ 0, /* lockID */ 1,
                          XAIE_ENABLE, /* release */ 0, XAIE_ENABLE,
                          /* acquire */ 1);
    // XAieDma_ShimBdSetAddr(&ShimDMAInst_7_0,  /* bd */ 0,
    // HIGH_ADDR((u64)0x20100004000), LOW_ADDR((u64)0x20100004000),  /* len */
    // 256 * 4);
    XAieDma_ShimBdSetAddr(&ShimDMAInst_47_0, /* bd */ 0,
                          HIGH_ADDR((u64)ddr_ptr47), LOW_ADDR((u64)ddr_ptr47),
                          /* len */ 7168 * 4);
    XAieDma_ShimBdSetAxi(&ShimDMAInst_47_0, /* bd */ 0, /* smid */ 0,
                         /* burstlen */ 16, /* QOS */ 0, /* Cache */ 1,
                         /* secure */ XAIE_DISABLE);
    XAieDma_ShimBdSetNext(&ShimDMAInst_47_0, /* bd */ 0, /* nextbd */ 0);
    XAieDma_ShimBdWrite(&ShimDMAInst_47_0, /* bd */ 0);
    XAieDma_ShimSetStartBd(&ShimDMAInst_47_0, XAIEDMA_SHIM_CHNUM_MM2S0,
                           /* bd */ 0);
    XAieDma_ShimChControl(&ShimDMAInst_47_0, XAIEDMA_TILE_CHNUM_MM2S0,
                          /* PauseStream */ XAIE_DISABLE,
                          /* PauseMM */ XAIE_DISABLE, /* Enable */ XAIE_ENABLE);

    // We're going to stamp over the memory
    for (int i = 0; i < DMA_COUNT; i++) {
      mlir_write_buffer_buf21_0(i, 0xdeadbeef);
      mlir_write_buffer_buf31_0(i, 0xdeadbeef);
      mlir_write_buffer_buf61_0(i, 0xdeadbeef);
      mlir_write_buffer_buf71_0(i, 0xdeadbeef);
      mlir_write_buffer_buf101_0(i, 0xdeadbeef);
      mlir_write_buffer_buf111_0(i, 0xdeadbeef);
      mlir_write_buffer_buf181_0(i, 0xdeadbeef);
      mlir_write_buffer_buf191_0(i, 0xdeadbeef);
      mlir_write_buffer_buf261_0(i, 0xdeadbeef);
      mlir_write_buffer_buf271_0(i, 0xdeadbeef);
      mlir_write_buffer_buf341_0(i, 0xdeadbeef);
      mlir_write_buffer_buf351_0(i, 0xdeadbeef);
      mlir_write_buffer_buf421_0(i, 0xdeadbeef);
      mlir_write_buffer_buf431_0(i, 0xdeadbeef);
      mlir_write_buffer_buf461_0(i, 0xdeadbeef);
      mlir_write_buffer_buf471_0(i, 0xdeadbeef);
    }

    XAieTilePl_EventBroadcast(&TileInst[2][0], 2,
                              XAIETILE_EVENT_SHIM_LOCK_1_ACQUIRED_NOC); // Start

    EventMon pc2(&TileInst[2][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                 XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                 MODE_MEM);
    pc2.set();
    EventMon pc3(&TileInst[3][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                 XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                 MODE_MEM);
    pc3.set();
    EventMon pc6(&TileInst[6][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                 XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                 MODE_MEM);
    pc6.set();
    EventMon pc7(&TileInst[7][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                 XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                 MODE_MEM);
    pc7.set();
    EventMon pc10(&TileInst[10][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc10.set();
    EventMon pc11(&TileInst[11][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc11.set();
    EventMon pc18(&TileInst[18][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc18.set();
    EventMon pc19(&TileInst[19][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc19.set();
    EventMon pc26(&TileInst[26][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc26.set();
    EventMon pc27(&TileInst[27][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc27.set();
    EventMon pc34(&TileInst[34][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc34.set();
    EventMon pc35(&TileInst[35][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc35.set();
    EventMon pc42(&TileInst[42][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc42.set();
    EventMon pc43(&TileInst[43][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc43.set();
    EventMon pc46(&TileInst[46][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc46.set();
    EventMon pc47(&TileInst[47][1], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                  XAIETILE_EVENT_MEM_LOCK_0_RELEASE, XAIETILE_EVENT_MEM_NONE,
                  MODE_MEM);
    pc47.set();

    // iterate over the buffer
    usleep(1000);
    XAieTile_LockRelease(&(TileInst[2][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[3][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[6][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[7][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[10][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[11][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[18][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[19][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[26][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[27][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[34][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[35][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[42][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[43][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[46][0]), 1, 1,
                         0); // Release lock for reading from DDR
    XAieTile_LockRelease(&(TileInst[47][0]), 1, 1,
                         0); // Release lock for reading from DDR

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
