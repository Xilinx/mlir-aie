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

#define LOCK_TIMEOUT 100

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)

#define MLIR_STACK_OFFSET 4096

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
  printf("test start.\n");

  int n = 1;
  u32 pc0_times[n];
  u32 pc1_times[n];
  u32 pc2_times[n];
  u32 pc3_times[n];

  for (int iters = 0; iters < n; iters++) {

    size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
    XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS,
                             XAIE_ADDR_ARRAY_OFF);
    XAieGbl_HwInit(&AieConfig);
    AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
    XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

    mlir_configure_cores();
    mlir_configure_switchboxes();

    printf("Acquire input buffer lock first.\n");
    XAieTile_LockAcquire(&(TileInst[1][3]), 5, 0, 0);

    mlir_configure_dmas();
    mlir_initialize_locks();

    ACDC_clear_tile_memory(TileInst[1][3]);
    ACDC_clear_tile_memory(TileInst[1][4]);

#define DMA_COUNT 512

    for (int i = 0; i < DMA_COUNT; i++) {
      mlir_write_buffer_a13(i, i + 1);
      mlir_write_buffer_a14(i, 0xdeadbeef);
    }

    // Destination Tile
    EventMon pc0(&TileInst[1][4], 0, XAIETILE_EVENT_MEM_BROADCAST_2,
                 XAIETILE_EVENT_MEM_DMA_S2MM_1_FINISHED_BD,
                 XAIETILE_EVENT_MEM_NONE, MODE_MEM);
    pc0.set();
    EventMon pc1(&TileInst[1][4], 1, XAIETILE_EVENT_MEM_BROADCAST_2,
                 XAIETILE_EVENT_MEM_LOCK_6_RELEASE, XAIETILE_EVENT_MEM_NONE,
                 MODE_MEM);
    pc1.set();

    // Source Tile
    EventMon pc2(&TileInst[1][3], 0, XAIETILE_EVENT_MEM_LOCK_5_ACQUIRED,
                 XAIETILE_EVENT_MEM_DMA_MM2S_0_FINISHED_BD,
                 XAIETILE_EVENT_MEM_NONE, MODE_MEM);
    pc2.set();
    EventMon pc3(&TileInst[1][3], 1, XAIETILE_EVENT_MEM_LOCK_5_ACQUIRED,
                 XAIETILE_EVENT_MEM_LOCK_5_RELEASE, XAIETILE_EVENT_MEM_NONE,
                 MODE_MEM);
    pc3.set();

    XAieTileMem_EventBroadcast(&TileInst[1][3], 2,
                               XAIETILE_EVENT_MEM_LOCK_5_ACQUIRED); // Start

    XAieTile_LockRelease(&(TileInst[1][3]), 5, 1, 0);
    usleep(100);

    pc0_times[iters] = pc0.diff();
    pc1_times[iters] = pc1.diff();
    pc2_times[iters] = pc2.diff();
    pc3_times[iters] = pc3.diff();

    for (int i = 0; i < DMA_COUNT; i++) {
      uint32_t d = mlir_read_buffer_a13(i);
      if (d != (i + 1)) {
        printf("Not Matched");
      }
    }

    int errors = 0;
    for (int i = 0; i < DMA_COUNT; i++) {
      uint32_t d = mlir_read_buffer_a14(i);
      if (d != (i + 1)) {
        errors++;
        printf("mismatch %x != 1 + %x\n", d, i);
        break;
      }
    }
  }

  printf("\nSource MM2S Finished ");
  computeStats(pc2_times, n);
  printf("Source Lock Released ");
  computeStats(pc3_times, n);
  printf("Destination S2MM Finished ");
  computeStats(pc0_times, n);
  printf("Destination Lock Released ");
  computeStats(pc1_times, n);
}
