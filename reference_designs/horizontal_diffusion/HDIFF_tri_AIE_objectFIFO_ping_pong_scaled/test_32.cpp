//===- test_32.cpp ----------------------------------------------*- C++ -*-===//
//
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
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
#include <time.h>
#include <unistd.h>
#include <xaiengine.h>
#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)
#define MLIR_STACK_OFFSET 4096
#define B_BLOCK_DEPTH 4 // set how many rows
#define HDIFF_COL 3     // columns
#define START_ROW 1
#define INPUT_ROWS 9

#define TOTAL_B_BLOCK 32
#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.");
  clock_t t;

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  u32 sleep_u = 100000;
  usleep(sleep_u);
  printf("before configure cores.");
  for (int b = 0; b < TOTAL_B_BLOCK; b++) {
    for (int i = 0; i < HDIFF_COL; i++) {
      for (int j = START_ROW; j < START_ROW + B_BLOCK_DEPTH; j++)
        mlir_aie_clear_tile_memory(_xaie, i, j);
    }
  }

  //   mlir_aie_clear_tile_memory(_xaie, 6, 4);
  mlir_aie_configure_cores(_xaie);

  usleep(sleep_u);
  printf("before configure switchboxes.");
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

  mlir_aie_acquire_lock(_xaie, 0, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(0, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc0(_xaie, 2, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc0.set();

  mlir_aie_acquire_lock(_xaie, 0, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(0, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc1(_xaie, 2, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc1.set();

  mlir_aie_acquire_lock(_xaie, 3, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(3, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc2(_xaie, 5, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc2.set();

  mlir_aie_acquire_lock(_xaie, 3, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(3, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc3(_xaie, 5, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc3.set();

  mlir_aie_acquire_lock(_xaie, 6, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(6, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc4(_xaie, 8, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc4.set();

  mlir_aie_acquire_lock(_xaie, 6, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(6, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc5(_xaie, 8, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc5.set();

  mlir_aie_acquire_lock(_xaie, 9, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(9, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc6(_xaie, 11, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc6.set();

  mlir_aie_acquire_lock(_xaie, 9, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(9, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc7(_xaie, 11, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc7.set();

  mlir_aie_acquire_lock(_xaie, 12, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(12, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc8(_xaie, 14, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc8.set();

  mlir_aie_acquire_lock(_xaie, 12, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(12, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc9(_xaie, 14, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc9.set();

  mlir_aie_acquire_lock(_xaie, 15, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(15, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc10(_xaie, 17, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc10.set();

  mlir_aie_acquire_lock(_xaie, 15, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(15, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc11(_xaie, 17, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc11.set();

  mlir_aie_acquire_lock(_xaie, 18, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(18, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc12(_xaie, 20, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc12.set();

  mlir_aie_acquire_lock(_xaie, 18, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(18, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc13(_xaie, 20, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc13.set();

  mlir_aie_acquire_lock(_xaie, 21, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(21, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc14(_xaie, 23, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc14.set();

  mlir_aie_acquire_lock(_xaie, 21, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(21, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc15(_xaie, 23, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc15.set();

  mlir_aie_acquire_lock(_xaie, 24, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(24, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc16(_xaie, 26, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc16.set();

  mlir_aie_acquire_lock(_xaie, 24, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(24, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc17(_xaie, 26, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc17.set();

  mlir_aie_acquire_lock(_xaie, 27, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(27, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc18(_xaie, 29, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc18.set();

  mlir_aie_acquire_lock(_xaie, 27, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(27, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc19(_xaie, 29, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc19.set();

  mlir_aie_acquire_lock(_xaie, 30, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(30, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc20(_xaie, 32, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc20.set();

  mlir_aie_acquire_lock(_xaie, 30, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(30, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc21(_xaie, 32, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc21.set();

  mlir_aie_acquire_lock(_xaie, 33, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(33, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc22(_xaie, 35, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc22.set();

  mlir_aie_acquire_lock(_xaie, 33, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(33, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc23(_xaie, 35, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc23.set();

  mlir_aie_acquire_lock(_xaie, 36, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(36, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc24(_xaie, 38, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc24.set();

  mlir_aie_acquire_lock(_xaie, 36, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(36, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc25(_xaie, 38, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc25.set();

  mlir_aie_acquire_lock(_xaie, 39, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(39, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc26(_xaie, 41, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc26.set();

  mlir_aie_acquire_lock(_xaie, 39, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(39, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc27(_xaie, 41, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc27.set();

  mlir_aie_acquire_lock(_xaie, 42, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(42, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc28(_xaie, 44, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc28.set();

  mlir_aie_acquire_lock(_xaie, 42, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(42, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc29(_xaie, 44, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc29.set();

  mlir_aie_acquire_lock(_xaie, 45, 2, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(45, 2), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc30(_xaie, 47, 2, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc30.set();

  mlir_aie_acquire_lock(_xaie, 45, 6, 14, 0, 0); // for timing
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(45, 6), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  EventMonitor pc31(_xaie, 47, 6, 0, XAIE_EVENT_BROADCAST_2_MEM,
                    XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                    XAIE_MEM_MOD);
  pc31.set();

  usleep(sleep_u);
  printf("before configure DMA");
  mlir_aie_configure_dmas(_xaie);
  int errors = 0;
  mlir_aie_init_mems(_xaie, 64);

  printf("Finish configure");
#define DMA_COUNT_IN 256 * INPUT_ROWS
#define DMA_COUNT_OUT 256 * 2 * B_BLOCK_DEPTH

  int *ddr_ptr_in_0 = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT_IN);
  int *ddr_ptr_in_1 = mlir_aie_mem_alloc(_xaie, 1, DMA_COUNT_IN);
  int *ddr_ptr_in_2 = mlir_aie_mem_alloc(_xaie, 2, DMA_COUNT_IN);
  int *ddr_ptr_in_3 = mlir_aie_mem_alloc(_xaie, 3, DMA_COUNT_IN);
  int *ddr_ptr_in_4 = mlir_aie_mem_alloc(_xaie, 4, DMA_COUNT_IN);
  int *ddr_ptr_in_5 = mlir_aie_mem_alloc(_xaie, 5, DMA_COUNT_IN);
  int *ddr_ptr_in_6 = mlir_aie_mem_alloc(_xaie, 6, DMA_COUNT_IN);
  int *ddr_ptr_in_7 = mlir_aie_mem_alloc(_xaie, 7, DMA_COUNT_IN);
  int *ddr_ptr_in_8 = mlir_aie_mem_alloc(_xaie, 8, DMA_COUNT_IN);
  int *ddr_ptr_in_9 = mlir_aie_mem_alloc(_xaie, 9, DMA_COUNT_IN);
  int *ddr_ptr_in_10 = mlir_aie_mem_alloc(_xaie, 10, DMA_COUNT_IN);
  int *ddr_ptr_in_11 = mlir_aie_mem_alloc(_xaie, 11, DMA_COUNT_IN);
  int *ddr_ptr_in_12 = mlir_aie_mem_alloc(_xaie, 12, DMA_COUNT_IN);
  int *ddr_ptr_in_13 = mlir_aie_mem_alloc(_xaie, 13, DMA_COUNT_IN);
  int *ddr_ptr_in_14 = mlir_aie_mem_alloc(_xaie, 14, DMA_COUNT_IN);
  int *ddr_ptr_in_15 = mlir_aie_mem_alloc(_xaie, 15, DMA_COUNT_IN);
  int *ddr_ptr_in_16 = mlir_aie_mem_alloc(_xaie, 16, DMA_COUNT_IN);
  int *ddr_ptr_in_17 = mlir_aie_mem_alloc(_xaie, 17, DMA_COUNT_IN);
  int *ddr_ptr_in_18 = mlir_aie_mem_alloc(_xaie, 18, DMA_COUNT_IN);
  int *ddr_ptr_in_19 = mlir_aie_mem_alloc(_xaie, 19, DMA_COUNT_IN);
  int *ddr_ptr_in_20 = mlir_aie_mem_alloc(_xaie, 20, DMA_COUNT_IN);
  int *ddr_ptr_in_21 = mlir_aie_mem_alloc(_xaie, 21, DMA_COUNT_IN);
  int *ddr_ptr_in_22 = mlir_aie_mem_alloc(_xaie, 22, DMA_COUNT_IN);
  int *ddr_ptr_in_23 = mlir_aie_mem_alloc(_xaie, 23, DMA_COUNT_IN);
  int *ddr_ptr_in_24 = mlir_aie_mem_alloc(_xaie, 24, DMA_COUNT_IN);
  int *ddr_ptr_in_25 = mlir_aie_mem_alloc(_xaie, 25, DMA_COUNT_IN);
  int *ddr_ptr_in_26 = mlir_aie_mem_alloc(_xaie, 26, DMA_COUNT_IN);
  int *ddr_ptr_in_27 = mlir_aie_mem_alloc(_xaie, 27, DMA_COUNT_IN);
  int *ddr_ptr_in_28 = mlir_aie_mem_alloc(_xaie, 28, DMA_COUNT_IN);
  int *ddr_ptr_in_29 = mlir_aie_mem_alloc(_xaie, 29, DMA_COUNT_IN);
  int *ddr_ptr_in_30 = mlir_aie_mem_alloc(_xaie, 30, DMA_COUNT_IN);
  int *ddr_ptr_in_31 = mlir_aie_mem_alloc(_xaie, 31, DMA_COUNT_IN);
  int *ddr_ptr_out_0 = mlir_aie_mem_alloc(_xaie, 32, DMA_COUNT_OUT);
  int *ddr_ptr_out_1 = mlir_aie_mem_alloc(_xaie, 33, DMA_COUNT_OUT);
  int *ddr_ptr_out_2 = mlir_aie_mem_alloc(_xaie, 34, DMA_COUNT_OUT);
  int *ddr_ptr_out_3 = mlir_aie_mem_alloc(_xaie, 35, DMA_COUNT_OUT);
  int *ddr_ptr_out_4 = mlir_aie_mem_alloc(_xaie, 36, DMA_COUNT_OUT);
  int *ddr_ptr_out_5 = mlir_aie_mem_alloc(_xaie, 37, DMA_COUNT_OUT);
  int *ddr_ptr_out_6 = mlir_aie_mem_alloc(_xaie, 38, DMA_COUNT_OUT);
  int *ddr_ptr_out_7 = mlir_aie_mem_alloc(_xaie, 39, DMA_COUNT_OUT);
  int *ddr_ptr_out_8 = mlir_aie_mem_alloc(_xaie, 40, DMA_COUNT_OUT);
  int *ddr_ptr_out_9 = mlir_aie_mem_alloc(_xaie, 41, DMA_COUNT_OUT);
  int *ddr_ptr_out_10 = mlir_aie_mem_alloc(_xaie, 42, DMA_COUNT_OUT);
  int *ddr_ptr_out_11 = mlir_aie_mem_alloc(_xaie, 43, DMA_COUNT_OUT);
  int *ddr_ptr_out_12 = mlir_aie_mem_alloc(_xaie, 44, DMA_COUNT_OUT);
  int *ddr_ptr_out_13 = mlir_aie_mem_alloc(_xaie, 45, DMA_COUNT_OUT);
  int *ddr_ptr_out_14 = mlir_aie_mem_alloc(_xaie, 46, DMA_COUNT_OUT);
  int *ddr_ptr_out_15 = mlir_aie_mem_alloc(_xaie, 47, DMA_COUNT_OUT);
  int *ddr_ptr_out_16 = mlir_aie_mem_alloc(_xaie, 48, DMA_COUNT_OUT);
  int *ddr_ptr_out_17 = mlir_aie_mem_alloc(_xaie, 49, DMA_COUNT_OUT);
  int *ddr_ptr_out_18 = mlir_aie_mem_alloc(_xaie, 50, DMA_COUNT_OUT);
  int *ddr_ptr_out_19 = mlir_aie_mem_alloc(_xaie, 51, DMA_COUNT_OUT);
  int *ddr_ptr_out_20 = mlir_aie_mem_alloc(_xaie, 52, DMA_COUNT_OUT);
  int *ddr_ptr_out_21 = mlir_aie_mem_alloc(_xaie, 53, DMA_COUNT_OUT);
  int *ddr_ptr_out_22 = mlir_aie_mem_alloc(_xaie, 54, DMA_COUNT_OUT);
  int *ddr_ptr_out_23 = mlir_aie_mem_alloc(_xaie, 55, DMA_COUNT_OUT);
  int *ddr_ptr_out_24 = mlir_aie_mem_alloc(_xaie, 56, DMA_COUNT_OUT);
  int *ddr_ptr_out_25 = mlir_aie_mem_alloc(_xaie, 57, DMA_COUNT_OUT);
  int *ddr_ptr_out_26 = mlir_aie_mem_alloc(_xaie, 58, DMA_COUNT_OUT);
  int *ddr_ptr_out_27 = mlir_aie_mem_alloc(_xaie, 59, DMA_COUNT_OUT);
  int *ddr_ptr_out_28 = mlir_aie_mem_alloc(_xaie, 60, DMA_COUNT_OUT);
  int *ddr_ptr_out_29 = mlir_aie_mem_alloc(_xaie, 61, DMA_COUNT_OUT);
  int *ddr_ptr_out_30 = mlir_aie_mem_alloc(_xaie, 62, DMA_COUNT_OUT);
  int *ddr_ptr_out_31 = mlir_aie_mem_alloc(_xaie, 63, DMA_COUNT_OUT);
  for (int i = 0; i < DMA_COUNT_IN; i++) {
    *(ddr_ptr_in_0 + i) = i;
    *(ddr_ptr_in_1 + i) = i;
    *(ddr_ptr_in_2 + i) = i;
    *(ddr_ptr_in_3 + i) = i;
    *(ddr_ptr_in_4 + i) = i;
    *(ddr_ptr_in_5 + i) = i;
    *(ddr_ptr_in_6 + i) = i;
    *(ddr_ptr_in_7 + i) = i;
    *(ddr_ptr_in_8 + i) = i;
    *(ddr_ptr_in_9 + i) = i;
    *(ddr_ptr_in_10 + i) = i;
    *(ddr_ptr_in_11 + i) = i;
    *(ddr_ptr_in_12 + i) = i;
    *(ddr_ptr_in_13 + i) = i;
    *(ddr_ptr_in_14 + i) = i;
    *(ddr_ptr_in_15 + i) = i;
    *(ddr_ptr_in_16 + i) = i;
    *(ddr_ptr_in_17 + i) = i;
    *(ddr_ptr_in_18 + i) = i;
    *(ddr_ptr_in_19 + i) = i;
    *(ddr_ptr_in_20 + i) = i;
    *(ddr_ptr_in_21 + i) = i;
    *(ddr_ptr_in_22 + i) = i;
    *(ddr_ptr_in_23 + i) = i;
    *(ddr_ptr_in_24 + i) = i;
    *(ddr_ptr_in_25 + i) = i;
    *(ddr_ptr_in_26 + i) = i;
    *(ddr_ptr_in_27 + i) = i;
    *(ddr_ptr_in_28 + i) = i;
    *(ddr_ptr_in_29 + i) = i;
    *(ddr_ptr_in_30 + i) = i;
    *(ddr_ptr_in_31 + i) = i;
  }
  for (int i = 0; i < DMA_COUNT_OUT; i++) {
    *(ddr_ptr_out_0 + i) = i;
    *(ddr_ptr_out_1 + i) = i;
    *(ddr_ptr_out_2 + i) = i;
    *(ddr_ptr_out_3 + i) = i;
    *(ddr_ptr_out_4 + i) = i;
    *(ddr_ptr_out_5 + i) = i;
    *(ddr_ptr_out_6 + i) = i;
    *(ddr_ptr_out_7 + i) = i;
    *(ddr_ptr_out_8 + i) = i;
    *(ddr_ptr_out_9 + i) = i;
    *(ddr_ptr_out_10 + i) = i;
    *(ddr_ptr_out_11 + i) = i;
    *(ddr_ptr_out_12 + i) = i;
    *(ddr_ptr_out_13 + i) = i;
    *(ddr_ptr_out_14 + i) = i;
    *(ddr_ptr_out_15 + i) = i;
    *(ddr_ptr_out_16 + i) = i;
    *(ddr_ptr_out_17 + i) = i;
    *(ddr_ptr_out_18 + i) = i;
    *(ddr_ptr_out_19 + i) = i;
    *(ddr_ptr_out_20 + i) = i;
    *(ddr_ptr_out_21 + i) = i;
    *(ddr_ptr_out_22 + i) = i;
    *(ddr_ptr_out_23 + i) = i;
    *(ddr_ptr_out_24 + i) = i;
    *(ddr_ptr_out_25 + i) = i;
    *(ddr_ptr_out_26 + i) = i;
    *(ddr_ptr_out_27 + i) = i;
    *(ddr_ptr_out_28 + i) = i;
    *(ddr_ptr_out_29 + i) = i;
    *(ddr_ptr_out_30 + i) = i;
    *(ddr_ptr_out_31 + i) = i;
  }
  mlir_aie_sync_mem_dev(_xaie, 0);
  mlir_aie_sync_mem_dev(_xaie, 1);
  mlir_aie_sync_mem_dev(_xaie, 2);
  mlir_aie_sync_mem_dev(_xaie, 3);
  mlir_aie_sync_mem_dev(_xaie, 4);
  mlir_aie_sync_mem_dev(_xaie, 5);
  mlir_aie_sync_mem_dev(_xaie, 6);
  mlir_aie_sync_mem_dev(_xaie, 7);
  mlir_aie_sync_mem_dev(_xaie, 8);
  mlir_aie_sync_mem_dev(_xaie, 9);
  mlir_aie_sync_mem_dev(_xaie, 10);
  mlir_aie_sync_mem_dev(_xaie, 11);
  mlir_aie_sync_mem_dev(_xaie, 12);
  mlir_aie_sync_mem_dev(_xaie, 13);
  mlir_aie_sync_mem_dev(_xaie, 14);
  mlir_aie_sync_mem_dev(_xaie, 15);
  mlir_aie_sync_mem_dev(_xaie, 16);
  mlir_aie_sync_mem_dev(_xaie, 17);
  mlir_aie_sync_mem_dev(_xaie, 18);
  mlir_aie_sync_mem_dev(_xaie, 19);
  mlir_aie_sync_mem_dev(_xaie, 20);
  mlir_aie_sync_mem_dev(_xaie, 21);
  mlir_aie_sync_mem_dev(_xaie, 22);
  mlir_aie_sync_mem_dev(_xaie, 23);
  mlir_aie_sync_mem_dev(_xaie, 24);
  mlir_aie_sync_mem_dev(_xaie, 25);
  mlir_aie_sync_mem_dev(_xaie, 26);
  mlir_aie_sync_mem_dev(_xaie, 27);
  mlir_aie_sync_mem_dev(_xaie, 28);
  mlir_aie_sync_mem_dev(_xaie, 29);
  mlir_aie_sync_mem_dev(_xaie, 30);
  mlir_aie_sync_mem_dev(_xaie, 31);
  mlir_aie_sync_mem_dev(_xaie, 32);
  mlir_aie_sync_mem_dev(_xaie, 33);
  mlir_aie_sync_mem_dev(_xaie, 34);
  mlir_aie_sync_mem_dev(_xaie, 35);
  mlir_aie_sync_mem_dev(_xaie, 36);
  mlir_aie_sync_mem_dev(_xaie, 37);
  mlir_aie_sync_mem_dev(_xaie, 38);
  mlir_aie_sync_mem_dev(_xaie, 39);
  mlir_aie_sync_mem_dev(_xaie, 40);
  mlir_aie_sync_mem_dev(_xaie, 41);
  mlir_aie_sync_mem_dev(_xaie, 42);
  mlir_aie_sync_mem_dev(_xaie, 43);
  mlir_aie_sync_mem_dev(_xaie, 44);
  mlir_aie_sync_mem_dev(_xaie, 45);
  mlir_aie_sync_mem_dev(_xaie, 46);
  mlir_aie_sync_mem_dev(_xaie, 47);
  mlir_aie_sync_mem_dev(_xaie, 48);
  mlir_aie_sync_mem_dev(_xaie, 49);
  mlir_aie_sync_mem_dev(_xaie, 50);
  mlir_aie_sync_mem_dev(_xaie, 51);
  mlir_aie_sync_mem_dev(_xaie, 52);
  mlir_aie_sync_mem_dev(_xaie, 53);
  mlir_aie_sync_mem_dev(_xaie, 54);
  mlir_aie_sync_mem_dev(_xaie, 55);
  mlir_aie_sync_mem_dev(_xaie, 56);
  mlir_aie_sync_mem_dev(_xaie, 57);
  mlir_aie_sync_mem_dev(_xaie, 58);
  mlir_aie_sync_mem_dev(_xaie, 59);
  mlir_aie_sync_mem_dev(_xaie, 60);
  mlir_aie_sync_mem_dev(_xaie, 61);
  mlir_aie_sync_mem_dev(_xaie, 62);
  mlir_aie_sync_mem_dev(_xaie, 63);
  mlir_aie_external_set_addr_ddr_buffer_in_0((u64)ddr_ptr_in_0);
  mlir_aie_external_set_addr_ddr_buffer_in_1((u64)ddr_ptr_in_1);
  mlir_aie_external_set_addr_ddr_buffer_in_2((u64)ddr_ptr_in_2);
  mlir_aie_external_set_addr_ddr_buffer_in_3((u64)ddr_ptr_in_3);
  mlir_aie_external_set_addr_ddr_buffer_in_4((u64)ddr_ptr_in_4);
  mlir_aie_external_set_addr_ddr_buffer_in_5((u64)ddr_ptr_in_5);
  mlir_aie_external_set_addr_ddr_buffer_in_6((u64)ddr_ptr_in_6);
  mlir_aie_external_set_addr_ddr_buffer_in_7((u64)ddr_ptr_in_7);
  mlir_aie_external_set_addr_ddr_buffer_in_8((u64)ddr_ptr_in_8);
  mlir_aie_external_set_addr_ddr_buffer_in_9((u64)ddr_ptr_in_9);
  mlir_aie_external_set_addr_ddr_buffer_in_10((u64)ddr_ptr_in_10);
  mlir_aie_external_set_addr_ddr_buffer_in_11((u64)ddr_ptr_in_11);
  mlir_aie_external_set_addr_ddr_buffer_in_12((u64)ddr_ptr_in_12);
  mlir_aie_external_set_addr_ddr_buffer_in_13((u64)ddr_ptr_in_13);
  mlir_aie_external_set_addr_ddr_buffer_in_14((u64)ddr_ptr_in_14);
  mlir_aie_external_set_addr_ddr_buffer_in_15((u64)ddr_ptr_in_15);
  mlir_aie_external_set_addr_ddr_buffer_in_16((u64)ddr_ptr_in_16);
  mlir_aie_external_set_addr_ddr_buffer_in_17((u64)ddr_ptr_in_17);
  mlir_aie_external_set_addr_ddr_buffer_in_18((u64)ddr_ptr_in_18);
  mlir_aie_external_set_addr_ddr_buffer_in_19((u64)ddr_ptr_in_19);
  mlir_aie_external_set_addr_ddr_buffer_in_20((u64)ddr_ptr_in_20);
  mlir_aie_external_set_addr_ddr_buffer_in_21((u64)ddr_ptr_in_21);
  mlir_aie_external_set_addr_ddr_buffer_in_22((u64)ddr_ptr_in_22);
  mlir_aie_external_set_addr_ddr_buffer_in_23((u64)ddr_ptr_in_23);
  mlir_aie_external_set_addr_ddr_buffer_in_24((u64)ddr_ptr_in_24);
  mlir_aie_external_set_addr_ddr_buffer_in_25((u64)ddr_ptr_in_25);
  mlir_aie_external_set_addr_ddr_buffer_in_26((u64)ddr_ptr_in_26);
  mlir_aie_external_set_addr_ddr_buffer_in_27((u64)ddr_ptr_in_27);
  mlir_aie_external_set_addr_ddr_buffer_in_28((u64)ddr_ptr_in_28);
  mlir_aie_external_set_addr_ddr_buffer_in_29((u64)ddr_ptr_in_29);
  mlir_aie_external_set_addr_ddr_buffer_in_30((u64)ddr_ptr_in_30);
  mlir_aie_external_set_addr_ddr_buffer_in_31((u64)ddr_ptr_in_31);
  mlir_aie_external_set_addr_ddr_buffer_out_0((u64)ddr_ptr_out_0);
  mlir_aie_external_set_addr_ddr_buffer_out_1((u64)ddr_ptr_out_1);
  mlir_aie_external_set_addr_ddr_buffer_out_2((u64)ddr_ptr_out_2);
  mlir_aie_external_set_addr_ddr_buffer_out_3((u64)ddr_ptr_out_3);
  mlir_aie_external_set_addr_ddr_buffer_out_4((u64)ddr_ptr_out_4);
  mlir_aie_external_set_addr_ddr_buffer_out_5((u64)ddr_ptr_out_5);
  mlir_aie_external_set_addr_ddr_buffer_out_6((u64)ddr_ptr_out_6);
  mlir_aie_external_set_addr_ddr_buffer_out_7((u64)ddr_ptr_out_7);
  mlir_aie_external_set_addr_ddr_buffer_out_8((u64)ddr_ptr_out_8);
  mlir_aie_external_set_addr_ddr_buffer_out_9((u64)ddr_ptr_out_9);
  mlir_aie_external_set_addr_ddr_buffer_out_10((u64)ddr_ptr_out_10);
  mlir_aie_external_set_addr_ddr_buffer_out_11((u64)ddr_ptr_out_11);
  mlir_aie_external_set_addr_ddr_buffer_out_12((u64)ddr_ptr_out_12);
  mlir_aie_external_set_addr_ddr_buffer_out_13((u64)ddr_ptr_out_13);
  mlir_aie_external_set_addr_ddr_buffer_out_14((u64)ddr_ptr_out_14);
  mlir_aie_external_set_addr_ddr_buffer_out_15((u64)ddr_ptr_out_15);
  mlir_aie_external_set_addr_ddr_buffer_out_16((u64)ddr_ptr_out_16);
  mlir_aie_external_set_addr_ddr_buffer_out_17((u64)ddr_ptr_out_17);
  mlir_aie_external_set_addr_ddr_buffer_out_18((u64)ddr_ptr_out_18);
  mlir_aie_external_set_addr_ddr_buffer_out_19((u64)ddr_ptr_out_19);
  mlir_aie_external_set_addr_ddr_buffer_out_20((u64)ddr_ptr_out_20);
  mlir_aie_external_set_addr_ddr_buffer_out_21((u64)ddr_ptr_out_21);
  mlir_aie_external_set_addr_ddr_buffer_out_22((u64)ddr_ptr_out_22);
  mlir_aie_external_set_addr_ddr_buffer_out_23((u64)ddr_ptr_out_23);
  mlir_aie_external_set_addr_ddr_buffer_out_24((u64)ddr_ptr_out_24);
  mlir_aie_external_set_addr_ddr_buffer_out_25((u64)ddr_ptr_out_25);
  mlir_aie_external_set_addr_ddr_buffer_out_26((u64)ddr_ptr_out_26);
  mlir_aie_external_set_addr_ddr_buffer_out_27((u64)ddr_ptr_out_27);
  mlir_aie_external_set_addr_ddr_buffer_out_28((u64)ddr_ptr_out_28);
  mlir_aie_external_set_addr_ddr_buffer_out_29((u64)ddr_ptr_out_29);
  mlir_aie_external_set_addr_ddr_buffer_out_30((u64)ddr_ptr_out_30);
  mlir_aie_external_set_addr_ddr_buffer_out_31((u64)ddr_ptr_out_31);
  mlir_aie_configure_shimdma_20(_xaie);
  mlir_aie_configure_shimdma_30(_xaie);
  mlir_aie_configure_shimdma_60(_xaie);
  mlir_aie_configure_shimdma_70(_xaie);
  mlir_aie_configure_shimdma_100(_xaie);
  mlir_aie_configure_shimdma_110(_xaie);
  mlir_aie_configure_shimdma_180(_xaie);
  mlir_aie_configure_shimdma_190(_xaie);
  mlir_aie_configure_shimdma_260(_xaie);
  mlir_aie_configure_shimdma_270(_xaie);
  mlir_aie_configure_shimdma_340(_xaie);
  mlir_aie_configure_shimdma_350(_xaie);
  mlir_aie_configure_shimdma_420(_xaie);
  mlir_aie_configure_shimdma_430(_xaie);
  mlir_aie_configure_shimdma_460(_xaie);
  mlir_aie_configure_shimdma_470(_xaie);

  printf("before core start");
  // mlir_aie_print_tile_status(_xaie, 7, 3);

  printf("Release lock for accessing DDR.");
  mlir_aie_release_of_0_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_15_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_23_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_38_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_46_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_61_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_69_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_84_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_92_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_107_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_115_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_130_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_138_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_153_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_161_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_176_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_184_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_199_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_207_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_222_lock_0(_xaie, 0, 0);
  /*ADDD ALL THE LOCKS*/
  mlir_aie_release_of_230_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_245_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_253_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_268_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_276_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_291_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_299_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_314_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_322_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_337_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_345_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_360_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_368_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_383_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_392_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_407_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_414_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_429_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_437_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_452_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_460_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_475_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_483_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_498_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_506_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_521_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_529_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_544_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_552_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_567_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_575_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_590_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_598_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_613_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_621_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_636_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_644_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_659_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_667_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_682_lock_0(_xaie, 0, 0);

  mlir_aie_release_of_690_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_705_lock_0(_xaie, 0, 0);
  mlir_aie_release_of_713_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_728_lock_0(_xaie, 0, 0);
  /*ADDD ALL THE LOCKS*/

  /*ADDD ALL THE LOCKS*/

  printf("Start cores");
  ///// --- start counter-----
  t = clock();
  mlir_aie_start_cores(_xaie);
  mlir_aie_release_lock(_xaie, 0, 2, 14, 0, 0);  // for timing
  mlir_aie_release_lock(_xaie, 0, 6, 14, 0, 0);  // for timing
  mlir_aie_release_lock(_xaie, 3, 2, 14, 0, 0);  // for timing
  mlir_aie_release_lock(_xaie, 3, 6, 14, 0, 0);  // for timing
  mlir_aie_release_lock(_xaie, 6, 2, 14, 0, 0);  // for timing
  mlir_aie_release_lock(_xaie, 6, 6, 14, 0, 0);  // for timing
  mlir_aie_release_lock(_xaie, 9, 2, 14, 0, 0);  // for timing
  mlir_aie_release_lock(_xaie, 9, 6, 14, 0, 0);  // for timing
  mlir_aie_release_lock(_xaie, 12, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 12, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 15, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 15, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 18, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 18, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 21, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 21, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 24, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 24, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 27, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 27, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 30, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 30, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 33, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 33, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 36, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 36, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 39, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 39, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 42, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 42, 6, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 45, 2, 14, 0, 0); // for timing
  mlir_aie_release_lock(_xaie, 45, 6, 14, 0, 0); // for timing

  t = clock() - t;

  printf("It took %ld clicks (%f seconds).", t, ((float)t) / CLOCKS_PER_SEC);

  usleep(sleep_u);
  printf("after core start");
  // mlir_aie_print_tile_status(_xaie, 7, 3);

  usleep(sleep_u);
  mlir_aie_sync_mem_cpu(_xaie,
                        32); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        33); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        34); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        35); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        36); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        37); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        38); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        39); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        40); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        41); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        42); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        43); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        44); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        45); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        46); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        47); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        48); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        49); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        50); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        51); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        52); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        53); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        54); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        55); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        56); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        57); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        58); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        59); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        60); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        61); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        62); //// only used in libaiev2 //sync up with output
  mlir_aie_sync_mem_cpu(_xaie,
                        63); //// only used in libaiev2 //sync up with output

  for (int i = 0; i < 512; i++) {
    printf("Location %d:  %d", i, ddr_ptr_out_0[i]);
  }

  int res = 0;
  if (!errors) {
    printf("PASS!");
    res = 0;
  } else {
    printf("Fail!");
    res = -1;
  }
  printf("PC0 cycles: %d", pc0.diff());
  printf("PC1 cycles: %d", pc1.diff());
  printf("PC2 cycles: %d", pc2.diff());
  printf("PC3 cycles: %d", pc3.diff());
  printf("PC4 cycles: %d", pc4.diff());
  printf("PC5 cycles: %d", pc5.diff());
  printf("PC6 cycles: %d", pc6.diff());
  printf("PC7 cycles: %d", pc7.diff());
  printf("PC8 cycles: %d", pc8.diff());
  printf("PC9 cycles: %d", pc9.diff());
  printf("PC10 cycles: %d", pc10.diff());
  printf("PC11 cycles: %d", pc11.diff());
  printf("PC12 cycles: %d", pc12.diff());
  printf("PC13 cycles: %d", pc13.diff());
  printf("PC14 cycles: %d", pc14.diff());
  printf("PC15 cycles: %d", pc15.diff());
  printf("PC16 cycles: %d", pc16.diff());
  printf("PC17 cycles: %d", pc17.diff());
  printf("PC18 cycles: %d", pc18.diff());
  printf("PC19 cycles: %d", pc19.diff());
  printf("PC20 cycles: %d", pc20.diff());
  printf("PC21 cycles: %d", pc21.diff());
  printf("PC22 cycles: %d", pc22.diff());
  printf("PC23 cycles: %d", pc23.diff());
  printf("PC24 cycles: %d", pc24.diff());
  printf("PC25 cycles: %d", pc25.diff());
  printf("PC26 cycles: %d", pc26.diff());
  printf("PC27 cycles: %d", pc27.diff());
  printf("PC28 cycles: %d", pc28.diff());
  printf("PC29 cycles: %d", pc29.diff());
  printf("PC30 cycles: %d", pc30.diff());
  printf("PC31 cycles: %d", pc31.diff());

  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.");

  return res;
}
