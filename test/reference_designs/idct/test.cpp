//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <xaiengine.h>
#include "test_library.h"

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)
#define MLIR_STACK_OFFSET 4096

#include "aie_inc.cpp"

// Taken from /reference_designs/MM_2x2/test.cpp.
#define MAP_SIZE 16UL
#define MAP_MASK (MAP_SIZE - 1)
void devmemRW32(uint32_t address, uint32_t value, bool write) {
  int fd;
  uint32_t *map_base;
  uint32_t read_result;
  uint32_t offset = address - 0xF70A0000;

  if ((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1)
    printf("ERROR!!!! open(devmem)\n");
  printf("\n/dev/mem opened.\n");
  fflush(stdout);

  map_base = (uint32_t *)mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,
                              fd, 0xF70A0000);
  if (map_base == (void *)-1)
    printf("ERROR!!!! map_base\n");
  printf("Memory mapped at address %p.\n", map_base);
  fflush(stdout);

  read_result = map_base[uint32_t(offset / 4)];
  printf("Value at address 0x%X: 0x%X\n", address, read_result);
  fflush(stdout);

  if (write) {
    map_base[uint32_t(offset / 4)] = value;
    // msync(map_base, MAP_SIZE, MS_SYNC);
    read_result = map_base[uint32_t(offset / 4)];
    printf("Written 0x%X; readback 0x%X\n", value, read_result);
    fflush(stdout);
  }

  // msync(map_base, MAP_SIZE, MS_SYNC);
  if (munmap(map_base, MAP_SIZE) == -1)
    printf("ERROR!!!! unmap_base\n");
  printf("/dev/mem closed.\n");
  fflush(stdout);
  close(fd);
}

int
main(int argc, char *argv[])
{
  devmemRW32(0xF70A000C, 0xF9E8D7C6, true);
  devmemRW32(0xF70A0000, 0x04000000, true);
  devmemRW32(0xF70A0004, 0x040381B1, true);
  devmemRW32(0xF70A0000, 0x04000000, true);
  devmemRW32(0xF70A0004, 0x000381B1, true);
  devmemRW32(0xF70A000C, 0x12341234, true);

  printf("test start.\n");

  int n = 1;
  u32 pc0_times[n];
  u32 pc1_times[n];
  u32 pc2_times[n];
  u32 pc3_times[n];
  u32 pc4_times[n];
  u32 pc5_times[n];
  u32 pc6_times[n];
  u32 pc7_times[n];

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

  u32 sleep_u = 100000;
  usleep(sleep_u);
  printf("before DMA config\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  mlir_aie_configure_dmas(_xaie);

  usleep(sleep_u);
  printf("after DMA config\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  int errors = 0;

  // // Load IDCT Data:
  // File *file = fopen("image.txt")
  // int image[512];
  // int num;
  // while(fscanf(file, "%d", &num) > 0){
  //     image[i] = num;
  //     i++;
  // }
  // fclose(file);

    #define DMA_COUNT 512

  mlir_aie_init_mems(_xaie, 2);
  u_int16_t *ddr_ptr_in = (u_int16_t *)mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT);
  u_int16_t *ddr_ptr_out = (u_int16_t *)mlir_aie_mem_alloc(_xaie, 1, DMA_COUNT);
  for (u_int16_t i = 0; i < DMA_COUNT; i++) {
    *(ddr_ptr_in + i) = i;
    *(ddr_ptr_out + i) = 0;
  }
  mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 1); // only used in libaiev2

#ifdef LIBXAIENGINEV2
  mlir_aie_external_set_addr_myBuffer_70_0((u64)ddr_ptr_in);
  mlir_aie_external_set_addr_myBuffer_70_1((u64)ddr_ptr_out);
  mlir_aie_configure_shimdma_70(_xaie);
#endif

  EventMonitor pc0(_xaie, 7, 3, 0, XAIE_EVENT_LOCK_3_ACQ_MEM,
                   XAIE_EVENT_LOCK_3_REL_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  EventMonitor pc1(_xaie, 7, 3, 1, XAIE_EVENT_LOCK_5_ACQ_MEM,
                   XAIE_EVENT_LOCK_5_REL_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);

  EventMonitor pc2(_xaie, 6, 3, 0, XAIE_EVENT_LOCK_3_ACQ_MEM,
                   XAIE_EVENT_LOCK_3_REL_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  EventMonitor pc3(_xaie, 6, 3, 1, XAIE_EVENT_LOCK_5_ACQ_MEM,
                   XAIE_EVENT_LOCK_5_REL_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);

  EventMonitor pc4(_xaie, 5, 3, 0, XAIE_EVENT_LOCK_3_ACQ_MEM,
                   XAIE_EVENT_LOCK_3_REL_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);
  EventMonitor pc5(_xaie, 5, 3, 1, XAIE_EVENT_LOCK_5_ACQ_MEM,
                   XAIE_EVENT_LOCK_5_REL_MEM, XAIE_EVENT_NONE_MEM,
                   XAIE_MEM_MOD);

  EventMonitor pc6(_xaie, 7, 0, 0, XAIE_EVENT_LOCK_1_ACQUIRED_PL,
                   XAIE_EVENT_LOCK_2_RELEASED_PL, XAIE_EVENT_NONE_PL,
                   XAIE_PL_MOD);
  EventMonitor pc7(_xaie, 7, 0, 1, XAIE_EVENT_LOCK_2_ACQUIRED_PL,
                   XAIE_EVENT_LOCK_2_RELEASED_PL, XAIE_EVENT_NONE_PL,
                   XAIE_PL_MOD);

  pc0.set();
  pc1.set();
  pc2.set();
  pc3.set();
  pc4.set();
  pc5.set();
  pc6.set();
  pc7.set();

  // for (int i=0; i<DMA_COUNT; i++) {
  //     uint32_t d = ddr_ptr_in[i];
  //     printf("ddr_ptr_in[%d] = %d\n", i, d);
  // }

  // for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_a73_ping(i, 0x0);
  // }

  // for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_a73_pong(i, 0x0);
  // }

  // for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_b73_ping(i, 0x0);
  // }

  // for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_b73_pong(i, 0x0);
  // }

  //   for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_a74_ping(i, 0x0);
  // }

  // for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_a74_pong(i, 0x0);
  // }

  // for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_b74_ping(i, 0x0);
  // }

  // for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_b74_pong(i, 0x0);
  // }

  //     for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_a75_ping(i, 0x0);
  // }

  // for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_a75_pong(i, 0x0);
  // }

  // for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_b75_ping(i, 0x0);
  // }

  // for (int i=0; i<DMA_COUNT; i++) {
  //     mlir_write_buffer_b75_pong(i, 0x0);
  // }

  mlir_aie_clear_tile_memory(_xaie, 7, 3);
  mlir_aie_clear_tile_memory(_xaie, 7, 4);
  mlir_aie_clear_tile_memory(_xaie, 7, 5);

  printf("before core start\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  // usleep(sleep_u);
  // printf("after core start\n");
  // ACDC_print_tile_status(TileInst[7][3]);
  // u32 locks70;
  // locks70 = XAieGbl_Read32(TileInst[7][0].TileAddr + 0x00014F00);
  // printf("Locks70 = %08X\n", locks70);

  // printf("Release lock for accessing DDR.\n");
  mlir_aie_release_lock(_xaie, 7, 0, 1, 1, 0);
  mlir_aie_release_lock(_xaie, 7, 0, 2, 1, 0);

  usleep(1000);
  pc0_times[0] = pc0.diff();
  pc1_times[0] = pc1.diff();
  pc2_times[0] = pc2.diff();
  pc3_times[0] = pc3.diff();
  pc4_times[0] = pc4.diff();
  pc5_times[0] = pc5.diff();
  pc6_times[0] = pc6.diff();
  pc7_times[0] = pc7.diff();
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

  printf("reached1: ");
  mlir_aie_acquire_lock(_xaie, 7, 0, 2, 0, 0);

  mlir_aie_sync_mem_cpu(_xaie, 1); // only used in libaiev2

  // for (uint16_t i=0; i<DMA_COUNT; i++) {
  //     uint16_t d = ddr_ptr_out[i];
  //     printf("ddr_ptr_out[%d] = %d\n", i, d);
  // }

  for (int i = 0; i < 512; i++)
    mlir_aie_check("DDR out", ddr_ptr_out[i], i, errors);

  // computeStats(pc0_times, n);
  computeStats(pc1_times, n);
  // computeStats(pc2_times, n);
  computeStats(pc3_times, n);
  // computeStats(pc4_times, n);
  computeStats(pc5_times, n);
  computeStats(pc6_times, n);
  // computeStats(pc7_times, n);

  int res = 0;

  printf("reached2: ");

  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }

  printf("reached3: ");

  mlir_aie_deinit_libxaie(_xaie);
  printf("test done.\n");
  return res;
}