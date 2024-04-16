//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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

int main(int argc, char *argv[]) {
  printf("test start.\n");

  // int n = 1;
  // u32 pc0_times[n];
  // u32 pc1_times[n];
  // u32 pc2_times[n];
  // u32 pc3_times[n];
  // u32 pc4_times[n];
  // u32 pc5_times[n];
  // u32 pc6_times[n];
  // u32 pc7_times[n];

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_clear_tile_memory(_xaie, 7, 3);
  mlir_aie_clear_tile_memory(_xaie, 7, 4);
  mlir_aie_clear_tile_memory(_xaie, 7, 5);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

  u32 sleep_u = 100000;
  usleep(sleep_u);
  printf("before DMA config\n");
  mlir_aie_configure_dmas(_xaie);

  usleep(sleep_u);
  printf("after DMA config\n");

  int errors = 0;

#define DMA_COUNT 512

  // Load IDCT Data
  FILE *file = fopen("image.txt", "r");
  if (file == NULL) {
    perror("Error opening file: ");
    return 1;
  }
  int image[DMA_COUNT];
  int num;
  int i = 0;
  while (fscanf(file, "%d\n", &num) > 0 && i < DMA_COUNT) {
    image[i] = num;
    i++;
  }
  fclose(file);
  printf("IDCT data loaded.\n");

  mlir_aie_init_mems(_xaie, 2);
  int16_t *ddr_ptr_in = (int16_t *)mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT);
  int16_t *ddr_ptr_out = (int16_t *)mlir_aie_mem_alloc(_xaie, 1, DMA_COUNT);
  for (int16_t i = 0; i < DMA_COUNT; i++) {
    *(ddr_ptr_in + i) = image[i];
    *(ddr_ptr_out + i) = 0;
  }
  mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 1); // only used in libaiev2

  mlir_aie_external_set_addr_buffer_in((u64)ddr_ptr_in);
  mlir_aie_external_set_addr_buffer_out((u64)ddr_ptr_out);
  mlir_aie_configure_shimdma_70(_xaie);

  // EventMonitor pc0(_xaie, 7, 3, 0, XAIE_EVENT_LOCK_3_ACQ_MEM,
  //                  XAIE_EVENT_LOCK_3_REL_MEM, XAIE_EVENT_NONE_MEM,
  //                  XAIE_MEM_MOD);
  // EventMonitor pc1(_xaie, 7, 3, 1, XAIE_EVENT_LOCK_5_ACQ_MEM,
  //                  XAIE_EVENT_LOCK_5_REL_MEM, XAIE_EVENT_NONE_MEM,
  //                  XAIE_MEM_MOD);

  // EventMonitor pc2(_xaie, 6, 3, 0, XAIE_EVENT_LOCK_3_ACQ_MEM,
  //                  XAIE_EVENT_LOCK_3_REL_MEM, XAIE_EVENT_NONE_MEM,
  //                  XAIE_MEM_MOD);
  // EventMonitor pc3(_xaie, 6, 3, 1, XAIE_EVENT_LOCK_5_ACQ_MEM,
  //                  XAIE_EVENT_LOCK_5_REL_MEM, XAIE_EVENT_NONE_MEM,
  //                  XAIE_MEM_MOD);

  // EventMonitor pc4(_xaie, 5, 3, 0, XAIE_EVENT_LOCK_3_ACQ_MEM,
  //                  XAIE_EVENT_LOCK_3_REL_MEM, XAIE_EVENT_NONE_MEM,
  //                  XAIE_MEM_MOD);
  // EventMonitor pc5(_xaie, 5, 3, 1, XAIE_EVENT_LOCK_5_ACQ_MEM,
  //                  XAIE_EVENT_LOCK_5_REL_MEM, XAIE_EVENT_NONE_MEM,
  //                  XAIE_MEM_MOD);

  // EventMonitor pc6(_xaie, 7, 0, 0, XAIE_EVENT_LOCK_1_ACQUIRED_PL,
  //                  XAIE_EVENT_LOCK_2_RELEASED_PL, XAIE_EVENT_NONE_PL,
  //                  XAIE_PL_MOD);
  // EventMonitor pc7(_xaie, 7, 0, 1, XAIE_EVENT_LOCK_2_ACQUIRED_PL,
  //                  XAIE_EVENT_LOCK_2_RELEASED_PL, XAIE_EVENT_NONE_PL,
  //                  XAIE_PL_MOD);

  // pc0.set();
  // pc1.set();
  // pc2.set();
  // pc3.set();
  // pc4.set();
  // pc5.set();
  // pc6.set();
  // pc7.set();

  // for (int i=0; i<DMA_COUNT; i++) {
  //     int16_t d = ddr_ptr_out[i];
  //     printf("ddr_ptr_out[%d] = %d\n", i, d);
  // }

  printf("before core start\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  printf("Release lock for accessing DDR.\n");
  mlir_aie_release_buffer_in_lock(_xaie, 1, 0);
  mlir_aie_release_buffer_out_lock(_xaie, 1, 0);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  usleep(sleep_u);
  printf("after core start\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);
  // u32 locks70;
  // locks70 = XAieGbl_Read32(TileInst[7][0].TileAddr + 0x00014F00);
  // printf("Locks70 = %08X\n", locks70);

  usleep(1000);
  // pc0_times[0] = pc0.diff();
  // pc1_times[0] = pc1.diff();
  // pc2_times[0] = pc2.diff();
  // pc3_times[0] = pc3.diff();
  // pc4_times[0] = pc4.diff();
  // pc5_times[0] = pc5.diff();
  // pc6_times[0] = pc6.diff();
  // pc7_times[0] = pc7.diff();
  // usleep(sleep_u);

  // mlir_aie_check("After", mlir_read_buffer_a_ping(0), 384, errors);
  // mlir_aie_check("After", mlir_read_buffer_a_pong(0), 448, errors);
  // mlir_aie_check("After", mlir_read_buffer_b_ping(0), 385, errors);
  // mlir_aie_check("After", mlir_read_buffer_b_pong(0), 449, errors);

  // Dump contents of ddr_ptr_out
  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_a73_ping(i);
  //         printf("buffer out a ping 73 [%d] = %d\n", i, d);
  //     }
  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_a73_pong(i);
  //         printf("buffer out a pong 73 [%d] = %d\n", i, d);
  //     }

  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_b73_ping(i);
  //         printf("buffer out b ping 73 [%d] = %d\n", i, d);
  //     }
  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_b73_pong(i);
  //         printf("buffer out b pong 73 [%d] = %d\n", i, d);
  //     }

  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_a74_ping(i);
  //         printf("buffer out a ping 74 [%d] = %d\n", i, d);
  //     }
  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_a74_pong(i);
  //         printf("buffer out a pong 74 [%d] = %d\n", i, d);
  //     }

  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_b74_ping(i);
  //         printf("buffer out b ping 74 [%d] = %d\n", i, d);
  //     }
  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_b74_pong(i);
  //         printf("buffer out b pong 74 [%d] = %d\n", i, d);
  //     }

  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_a75_ping(i);
  //         printf("buffer out a ping 75 [%d] = %d\n", i, d);
  //     }
  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_a75_pong(i);
  //         printf("buffer out a pong 75 [%d] = %d\n", i, d);
  //     }

  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_b75_ping(i);
  //         printf("buffer out b ping 75 [%d] = %d\n", i, d);
  //     }
  // for (int i=0; i<16; i++) {
  //         uint32_t d = mlir_read_buffer_b75_ping(i);
  //         printf("buffer out b pong 75 [%d] = %d\n", i, d);
  //     }

  mlir_aie_acquire_buffer_out_lock(_xaie, 0, 0);
  mlir_aie_sync_mem_cpu(_xaie, 1); // only used in libaiev2

  for (int i = 0; i < DMA_COUNT; i++)
    mlir_aie_check("DDR out", ddr_ptr_out[i], image[i], errors);

  // computeStats(pc0_times, n);
  // computeStats(pc1_times, n);
  // computeStats(pc2_times, n);
  // computeStats(pc3_times, n);
  // computeStats(pc4_times, n);
  // computeStats(pc5_times, n);
  // computeStats(pc6_times, n);
  // computeStats(pc7_times, n);

  int res = 0;

  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }

  mlir_aie_deinit_libxaie(_xaie);
  printf("test done.\n");
  return res;
}