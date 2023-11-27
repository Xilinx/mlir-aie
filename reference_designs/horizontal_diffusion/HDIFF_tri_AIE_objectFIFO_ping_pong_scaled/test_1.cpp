//===- test_1.cpp -----------------------------------------------*- C++ -*-===//
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

#define TOTAL_B_BLOCK 1 // only 1
#define B_BLOCK_DEPTH 4 // set how many rows
#define HDIFF_COL 3     // columns
#define START_ROW 1
#define INPUT_ROWS 9

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");
  clock_t t;

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  u32 sleep_u = 100000;
  usleep(sleep_u);
  printf("before configure cores.\n");
  for (int b = 0; b < TOTAL_B_BLOCK; b++) {
    for (int i = 0; i < HDIFF_COL; i++) {
      for (int j = START_ROW; j < START_ROW + B_BLOCK_DEPTH; j++)
        mlir_aie_clear_tile_memory(_xaie, i, j);
    }
  }

  //   mlir_aie_clear_tile_memory(_xaie, 6, 4);
  mlir_aie_configure_cores(_xaie);

  usleep(sleep_u);
  printf("before configure switchboxes.\n");
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

  mlir_aie_acquire_lock(_xaie, 0, 1, 14, 0, 0); // for timing
  // When lock 14 is acquired, we broadcast event 2 from tile t73
  XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(0, 1), XAIE_MEM_MOD, 2,
                      XAIE_EVENT_LOCK_14_ACQ_MEM);
  // we use broadcast event 2 at tile t75 to start the timer, and the local
  // event lock 14 acquire to stop the timer
  EventMonitor pc0(_xaie, 2, 1, 0, XAIE_EVENT_BROADCAST_2_MEM,
                   XAIE_EVENT_LOCK_14_ACQ_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc0.set();

  usleep(sleep_u);
  printf("before configure DMA\n");
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_init_mems(_xaie, 2);
  int errors = 0;

  printf("Finish configure\n");
#define DMA_COUNT_IN 256 * INPUT_ROWS
#define DMA_COUNT_OUT 256 * 2 * B_BLOCK_DEPTH
  int *ddr_ptr_in_0 = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT_IN);
  int *ddr_ptr_out_0 = mlir_aie_mem_alloc(_xaie, 1, DMA_COUNT_OUT);
  //   int *mem_ptr2 = mlir_aie_mem_alloc(_xaie, 2, DMA_COUNT);
  //   int *mem_ptr3 = mlir_aie_mem_alloc(_xaie, 3, DMA_COUNT);
  //   int *mem_ptr4 = mlir_aie_mem_alloc(_xaie, 4, DMA_COUNT);
  //   int *mem_ptr5 = mlir_aie_mem_alloc(_xaie, 5, DMA_COUNT);
  //   int *mem_ptr6 = mlir_aie_mem_alloc(_xaie, 6, DMA_COUNT);
  //   int *mem_ptr7 = mlir_aie_mem_alloc(_xaie, 7, DMA_COUNT);

  // initialize the external buffers
  for (int i = 0; i < DMA_COUNT_IN; i++) {
    *(ddr_ptr_in_0 + i) = i; // input
  }

  for (int i = 0; i < DMA_COUNT_OUT; i++) {
    *(ddr_ptr_out_0 + i) = 0; // input
  }

  mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 1); // only used in libaiev2
  //   mlir_aie_sync_mem_dev(_xaie, 2); // only used in libaiev2
  //   mlir_aie_sync_mem_dev(_xaie, 3); // only used in libaiev2
  //   mlir_aie_sync_mem_dev(_xaie, 4); // only used in libaiev2
  //   mlir_aie_sync_mem_dev(_xaie, 5); // only used in libaiev2
  //   mlir_aie_sync_mem_dev(_xaie, 6); // only used in libaiev2
  //   mlir_aie_sync_mem_dev(_xaie, 7); // only used in libaiev2

  mlir_aie_external_set_addr_ddr_buffer_in_0(
      (u64)ddr_ptr_in_0); // external set address
  mlir_aie_external_set_addr_ddr_buffer_out_0((u64)ddr_ptr_out_0);
  //   mlir_aie_external_set_addr_RHS_tile0((u64)mem_ptr2);
  //   mlir_aie_external_set_addr_RHS_tile1((u64)mem_ptr3);
  //   mlir_aie_external_set_addr_RHS_tile2((u64)mem_ptr4);
  //   mlir_aie_external_set_addr_RHS_tile3((u64)mem_ptr5);
  //   mlir_aie_external_set_addr_Out_tile0((u64)mem_ptr6);
  //   mlir_aie_external_set_addr_Out_tile1((u64)mem_ptr7);
  //   mlir_aie_configure_shimdma_60(_xaie);
  mlir_aie_configure_shimdma_20(_xaie);
  //   mlir_aie_configure_shimdma_100(_xaie);

  printf("before core start\n");
  // mlir_aie_print_tile_status(_xaie, 7, 3);

  printf("Release lock for accessing DDR.\n");
  mlir_aie_release_of_0_lock_0(_xaie, 1, 0); // (_xaie,release_value,time_out)
  mlir_aie_release_of_15_lock_0(_xaie, 0, 0);

  printf("Start cores\n");
  ///// --- start counter-----
  t = clock();
  mlir_aie_start_cores(_xaie);
  mlir_aie_release_lock(_xaie, 0, 1, 14, 0, 0); // for timing
  t = clock() - t;

  printf("It took %ld clicks (%f seconds).\n", t, ((float)t) / CLOCKS_PER_SEC);

  usleep(sleep_u);
  printf("after core start\n");
  // mlir_aie_print_tile_status(_xaie, 7, 3);

  usleep(sleep_u);

  //   mlir_aie_acquire_of_17_lock_0(_xaie, 1, 0);
  //   mlir_aie_acquire_of_15_lock_0(_xaie, 1, 0);

  mlir_aie_sync_mem_cpu(_xaie,
                        1); // only used in libaiev2 //sync up with output
  ///// --- end counter-----
  for (int i = 0; i < 512; i++) {
    printf("Location %d:  %d\n", i, ddr_ptr_out_0[i]);
  }

  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }
  printf("PC0 cycles: %d\n", pc0.diff());
  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.\n");

  return res;
}
