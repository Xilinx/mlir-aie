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

#define BRAM_ADDR (0x4000 + 0x020100000000LL)
#define DMA_COUNT 512

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
  u32 pc2_times[n];
  u32 pc3_times[n];

  int total_errors = 0;

  auto col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS,
                           XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  for (int iters = 0; iters < n; iters++) {

    mlir_configure_cores();
    mlir_configure_switchboxes();
    mlir_initialize_locks();
    mlir_configure_dmas();

    XAieTileCore_EventBroadcast(&TileInst[7][3], 2,
                                XAIETILE_EVENT_CORE_FP_OVERFLOW); // Start
    XAieTileCore_EventBroadcast(&TileInst[7][4], 3,
                                XAIETILE_EVENT_CORE_FP_UNDERFLOW); // Stop

    // Track time between two broadcast events in destination tiles (7,3) and
    // (7,4)
    EventMonitor pc0(&TileInst[7][4], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                 XAIETILE_EVENT_MEM_BROADCAST_3, XAIETILE_EVENT_MEM_NONE,
                 MODE_MEM);
    pc0.set();

    EventMonitor pc1(&TileInst[7][2], 1, XAIETILE_EVENT_MEM_BROADCAST_2,
                 XAIETILE_EVENT_MEM_BROADCAST_3, XAIETILE_EVENT_MEM_NONE,
                 MODE_MEM);
    pc1.set();

    usleep(100);

    // Start Test by generating events in Source Tile
    XAieTileCore_EventGenerate(&TileInst[7][3],
                               XAIETILE_EVENT_CORE_FP_OVERFLOW);
    XAieTileCore_EventGenerate(&TileInst[7][4],
                               XAIETILE_EVENT_CORE_FP_UNDERFLOW);

    ACDC_print_tile_status(TileInst[7][3]);

    pc0_times[iters] = pc0.diff();
    pc1_times[iters] = pc1.diff();
  }
  printf("\nTime of First Signal: ");
  computeStats(pc1_times, n);
  printf("\nTime of Second Signal: ");
  computeStats(pc0_times, n);
}