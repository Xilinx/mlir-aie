//===- test.cpp -------------------------------------------------*- C++ -*-===//
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

#define AIE_COL 32
#define BROAD_CORES 1
#define START_ROW 2
#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");
  clock_t t;

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  u32 sleep_u = 100000;
  usleep(sleep_u);
  printf("before configure cores.\n");

  for (int i = 0; i < AIE_COL; i++) {
    for (int j = START_ROW; j < START_ROW + BROAD_CORES; j++)
      mlir_aie_clear_tile_memory(_xaie, i, j);
  }

  mlir_aie_configure_cores(_xaie);

  usleep(sleep_u);
  printf("before configure switchboxes.\n");
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

  mlir_aie_acquire_lock(_xaie, 7, 3, 14, 0, 0); // for timing
  EventMonitor pc0(_xaie, 7, 3, 0, XAIE_EVENT_LOCK_14_ACQ_MEM,
                   XAIE_EVENT_LOCK_14_REL_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  pc0.set();

  usleep(sleep_u);
  printf("before configure DMA\n");
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_init_mems(_xaie, 64);
  int errors = 0;

  printf("Finish configure\n");

#define DMA_COUNT_IN 1536
#define DMA_COUNT_OUT 512
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

  int *ddr_ptr_out_0_2 = mlir_aie_mem_alloc(_xaie, 32, DMA_COUNT_OUT);
  int *ddr_ptr_out_1_2 = mlir_aie_mem_alloc(_xaie, 33, DMA_COUNT_OUT);
  int *ddr_ptr_out_2_2 = mlir_aie_mem_alloc(_xaie, 34, DMA_COUNT_OUT);
  int *ddr_ptr_out_3_2 = mlir_aie_mem_alloc(_xaie, 35, DMA_COUNT_OUT);
  int *ddr_ptr_out_4_2 = mlir_aie_mem_alloc(_xaie, 36, DMA_COUNT_OUT);
  int *ddr_ptr_out_5_2 = mlir_aie_mem_alloc(_xaie, 37, DMA_COUNT_OUT);
  int *ddr_ptr_out_6_2 = mlir_aie_mem_alloc(_xaie, 38, DMA_COUNT_OUT);
  int *ddr_ptr_out_7_2 = mlir_aie_mem_alloc(_xaie, 39, DMA_COUNT_OUT);
  int *ddr_ptr_out_8_2 = mlir_aie_mem_alloc(_xaie, 40, DMA_COUNT_OUT);
  int *ddr_ptr_out_9_2 = mlir_aie_mem_alloc(_xaie, 41, DMA_COUNT_OUT);
  int *ddr_ptr_out_10_2 = mlir_aie_mem_alloc(_xaie, 42, DMA_COUNT_OUT);
  int *ddr_ptr_out_11_2 = mlir_aie_mem_alloc(_xaie, 43, DMA_COUNT_OUT);
  int *ddr_ptr_out_12_2 = mlir_aie_mem_alloc(_xaie, 44, DMA_COUNT_OUT);
  int *ddr_ptr_out_13_2 = mlir_aie_mem_alloc(_xaie, 45, DMA_COUNT_OUT);
  int *ddr_ptr_out_14_2 = mlir_aie_mem_alloc(_xaie, 46, DMA_COUNT_OUT);
  int *ddr_ptr_out_15_2 = mlir_aie_mem_alloc(_xaie, 47, DMA_COUNT_OUT);
  int *ddr_ptr_out_16_2 = mlir_aie_mem_alloc(_xaie, 48, DMA_COUNT_OUT);
  int *ddr_ptr_out_17_2 = mlir_aie_mem_alloc(_xaie, 49, DMA_COUNT_OUT);
  int *ddr_ptr_out_18_2 = mlir_aie_mem_alloc(_xaie, 50, DMA_COUNT_OUT);
  int *ddr_ptr_out_19_2 = mlir_aie_mem_alloc(_xaie, 51, DMA_COUNT_OUT);
  int *ddr_ptr_out_20_2 = mlir_aie_mem_alloc(_xaie, 52, DMA_COUNT_OUT);
  int *ddr_ptr_out_21_2 = mlir_aie_mem_alloc(_xaie, 53, DMA_COUNT_OUT);
  int *ddr_ptr_out_22_2 = mlir_aie_mem_alloc(_xaie, 54, DMA_COUNT_OUT);
  int *ddr_ptr_out_23_2 = mlir_aie_mem_alloc(_xaie, 55, DMA_COUNT_OUT);
  int *ddr_ptr_out_24_2 = mlir_aie_mem_alloc(_xaie, 56, DMA_COUNT_OUT);
  int *ddr_ptr_out_25_2 = mlir_aie_mem_alloc(_xaie, 57, DMA_COUNT_OUT);
  int *ddr_ptr_out_26_2 = mlir_aie_mem_alloc(_xaie, 58, DMA_COUNT_OUT);
  int *ddr_ptr_out_27_2 = mlir_aie_mem_alloc(_xaie, 59, DMA_COUNT_OUT);
  int *ddr_ptr_out_28_2 = mlir_aie_mem_alloc(_xaie, 60, DMA_COUNT_OUT);
  int *ddr_ptr_out_29_2 = mlir_aie_mem_alloc(_xaie, 61, DMA_COUNT_OUT);
  int *ddr_ptr_out_30_2 = mlir_aie_mem_alloc(_xaie, 62, DMA_COUNT_OUT);
  int *ddr_ptr_out_31_2 = mlir_aie_mem_alloc(_xaie, 63, DMA_COUNT_OUT);

  // initialize the external buffers
  for (int i = 0; i < DMA_COUNT_IN; i++) {
    *(ddr_ptr_in_0 + i) = i;  // input
    *(ddr_ptr_in_1 + i) = i;  // input
    *(ddr_ptr_in_2 + i) = i;  // input
    *(ddr_ptr_in_3 + i) = i;  // input
    *(ddr_ptr_in_4 + i) = i;  // input
    *(ddr_ptr_in_5 + i) = i;  // input
    *(ddr_ptr_in_6 + i) = i;  // input
    *(ddr_ptr_in_7 + i) = i;  // input
    *(ddr_ptr_in_8 + i) = i;  // input
    *(ddr_ptr_in_9 + i) = i;  // input
    *(ddr_ptr_in_10 + i) = i; // input
    *(ddr_ptr_in_11 + i) = i; // input
    *(ddr_ptr_in_12 + i) = i; // input
    *(ddr_ptr_in_13 + i) = i; // input
    *(ddr_ptr_in_14 + i) = i; // input
    *(ddr_ptr_in_15 + i) = i; // input
    *(ddr_ptr_in_16 + i) = i; // input
    *(ddr_ptr_in_17 + i) = i; // input
    *(ddr_ptr_in_18 + i) = i; // input
    *(ddr_ptr_in_19 + i) = i; // input
    *(ddr_ptr_in_20 + i) = i; // input
    *(ddr_ptr_in_21 + i) = i; // input
    *(ddr_ptr_in_22 + i) = i; // input
    *(ddr_ptr_in_23 + i) = i; // input
    *(ddr_ptr_in_24 + i) = i; // input
    *(ddr_ptr_in_25 + i) = i; // input
    *(ddr_ptr_in_26 + i) = i; // input
    *(ddr_ptr_in_27 + i) = i; // input
    *(ddr_ptr_in_28 + i) = i; // input
    *(ddr_ptr_in_29 + i) = i; // input
    *(ddr_ptr_in_30 + i) = i; // input
    *(ddr_ptr_in_31 + i) = i; // input
  }

  for (int i = 0; i < DMA_COUNT_OUT; i++) {
    *(ddr_ptr_out_0_2 + i) = 0;
    *(ddr_ptr_out_1_2 + i) = 0;
    *(ddr_ptr_out_2_2 + i) = 0;
    *(ddr_ptr_out_3_2 + i) = 0;
    *(ddr_ptr_out_4_2 + i) = 0;
    *(ddr_ptr_out_5_2 + i) = 0;
    *(ddr_ptr_out_6_2 + i) = 0;
    *(ddr_ptr_out_7_2 + i) = 0;
    *(ddr_ptr_out_8_2 + i) = 0;
    *(ddr_ptr_out_9_2 + i) = 0;
    *(ddr_ptr_out_10_2 + i) = 0;
    *(ddr_ptr_out_11_2 + i) = 0;
    *(ddr_ptr_out_12_2 + i) = 0;
    *(ddr_ptr_out_13_2 + i) = 0;
    *(ddr_ptr_out_14_2 + i) = 0;
    *(ddr_ptr_out_15_2 + i) = 0;
    *(ddr_ptr_out_16_2 + i) = 0;
    *(ddr_ptr_out_17_2 + i) = 0;
    *(ddr_ptr_out_18_2 + i) = 0;
    *(ddr_ptr_out_19_2 + i) = 0;
    *(ddr_ptr_out_20_2 + i) = 0;
    *(ddr_ptr_out_21_2 + i) = 0;
    *(ddr_ptr_out_22_2 + i) = 0;
    *(ddr_ptr_out_23_2 + i) = 0;
    *(ddr_ptr_out_24_2 + i) = 0;
    *(ddr_ptr_out_25_2 + i) = 0;
    *(ddr_ptr_out_26_2 + i) = 0;
    *(ddr_ptr_out_27_2 + i) = 0;
    *(ddr_ptr_out_28_2 + i) = 0;
    *(ddr_ptr_out_29_2 + i) = 0;
    *(ddr_ptr_out_30_2 + i) = 0;
    *(ddr_ptr_out_31_2 + i) = 0;
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

  mlir_aie_external_set_addr_ddr_buffer_out_0_2((u64)ddr_ptr_out_0_2);
  mlir_aie_external_set_addr_ddr_buffer_out_1_2((u64)ddr_ptr_out_1_2);
  mlir_aie_external_set_addr_ddr_buffer_out_2_2((u64)ddr_ptr_out_2_2);
  mlir_aie_external_set_addr_ddr_buffer_out_3_2((u64)ddr_ptr_out_3_2);
  mlir_aie_external_set_addr_ddr_buffer_out_4_2((u64)ddr_ptr_out_4_2);
  mlir_aie_external_set_addr_ddr_buffer_out_5_2((u64)ddr_ptr_out_5_2);
  mlir_aie_external_set_addr_ddr_buffer_out_6_2((u64)ddr_ptr_out_6_2);
  mlir_aie_external_set_addr_ddr_buffer_out_7_2((u64)ddr_ptr_out_7_2);
  mlir_aie_external_set_addr_ddr_buffer_out_8_2((u64)ddr_ptr_out_8_2);
  mlir_aie_external_set_addr_ddr_buffer_out_9_2((u64)ddr_ptr_out_9_2);
  mlir_aie_external_set_addr_ddr_buffer_out_10_2((u64)ddr_ptr_out_10_2);
  mlir_aie_external_set_addr_ddr_buffer_out_11_2((u64)ddr_ptr_out_11_2);
  mlir_aie_external_set_addr_ddr_buffer_out_12_2((u64)ddr_ptr_out_12_2);
  mlir_aie_external_set_addr_ddr_buffer_out_13_2((u64)ddr_ptr_out_13_2);
  mlir_aie_external_set_addr_ddr_buffer_out_14_2((u64)ddr_ptr_out_14_2);
  mlir_aie_external_set_addr_ddr_buffer_out_15_2((u64)ddr_ptr_out_15_2);
  mlir_aie_external_set_addr_ddr_buffer_out_16_2((u64)ddr_ptr_out_16_2);
  mlir_aie_external_set_addr_ddr_buffer_out_17_2((u64)ddr_ptr_out_17_2);
  mlir_aie_external_set_addr_ddr_buffer_out_18_2((u64)ddr_ptr_out_18_2);
  mlir_aie_external_set_addr_ddr_buffer_out_19_2((u64)ddr_ptr_out_19_2);
  mlir_aie_external_set_addr_ddr_buffer_out_20_2((u64)ddr_ptr_out_20_2);
  mlir_aie_external_set_addr_ddr_buffer_out_21_2((u64)ddr_ptr_out_21_2);
  mlir_aie_external_set_addr_ddr_buffer_out_22_2((u64)ddr_ptr_out_22_2);
  mlir_aie_external_set_addr_ddr_buffer_out_23_2((u64)ddr_ptr_out_23_2);
  mlir_aie_external_set_addr_ddr_buffer_out_24_2((u64)ddr_ptr_out_24_2);
  mlir_aie_external_set_addr_ddr_buffer_out_25_2((u64)ddr_ptr_out_25_2);
  mlir_aie_external_set_addr_ddr_buffer_out_26_2((u64)ddr_ptr_out_26_2);
  mlir_aie_external_set_addr_ddr_buffer_out_27_2((u64)ddr_ptr_out_27_2);
  mlir_aie_external_set_addr_ddr_buffer_out_28_2((u64)ddr_ptr_out_28_2);
  mlir_aie_external_set_addr_ddr_buffer_out_29_2((u64)ddr_ptr_out_29_2);
  mlir_aie_external_set_addr_ddr_buffer_out_30_2((u64)ddr_ptr_out_30_2);
  mlir_aie_external_set_addr_ddr_buffer_out_31_2((u64)ddr_ptr_out_31_2);

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

  printf("Release lock for accessing DDR.\n");

  mlir_aie_release_obj_in_1_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_1_2_cons_lock_0(_xaie, 0,
                                           0); // (_xaie,release_value,time_out)
  mlir_aie_release_obj_in_0_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_0_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_11_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_10_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_10_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_11_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_31_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_30_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_30_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_31_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_9_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_8_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_8_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_9_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_18_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_18_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_19_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_19_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_27_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_27_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_26_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_26_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_4_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_5_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_out_4_2_cons_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_in_5_lock_0(_xaie, 1, 0);

  // /33
  mlir_aie_release_obj_in_7_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_7_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_6_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_6_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_3_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_3_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_2_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_2_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_24_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_25_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_25_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_24_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_14_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_14_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_15_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_15_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_12_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_12_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_13_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_13_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_17_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_16_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_16_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_17_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_28_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_28_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_29_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_29_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_21_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_20_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_20_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_21_2_cons_lock_0(_xaie, 0, 0);

  mlir_aie_release_obj_in_22_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_22_2_cons_lock_0(_xaie, 0, 0);
  mlir_aie_release_obj_in_23_lock_0(_xaie, 1, 0);
  mlir_aie_release_obj_out_23_2_cons_lock_0(_xaie, 0, 0);

  printf("Start cores\n");
  t = clock();
  ///// --- start counter-----
  mlir_aie_start_cores(_xaie);

  mlir_aie_release_lock(_xaie, 7, 3, 14, 0, 0); // for timing
  t = clock() - t;

  printf("It took %ld clicks (%f seconds).\n", t, ((float)t) / CLOCKS_PER_SEC);

  usleep(sleep_u);
  printf("after core start\n");
  // mlir_aie_print_tile_status(_xaie, 7, 3);

  usleep(sleep_u);
  // /mnt/scratch/gagsingh/mlir-aie/install/bin/aie-opt
  // --aie-objectFifo-stateful-transform aie.mlir

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
  ///// --- end counter-----
  // for (int i =0; i < 256; i ++ ){
  //       printf("Location %d:  %d\n", i, ddr_ptr_out[i]);
  //   }

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
