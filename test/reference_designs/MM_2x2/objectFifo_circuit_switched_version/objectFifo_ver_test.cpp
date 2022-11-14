//===- objectFifo_ver_test.cpp
//-------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
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

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)
#define MLIR_STACK_OFFSET 4096

#include "aie_inc.cpp"
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
int main(int argc, char *argv[]) {
  devmemRW32(0xF70A000C, 0xF9E8D7C6, true);
  devmemRW32(0xF70A0000, 0x04000000, true);
  devmemRW32(0xF70A0004, 0x040381B1, true);
  devmemRW32(0xF70A0000, 0x04000000, true);
  devmemRW32(0xF70A0004, 0x000381B1, true);
  devmemRW32(0xF70A000C, 0x12341234, true);
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  u32 sleep_u = 100000;
  usleep(sleep_u);
  printf("before configure cores.\n");

  mlir_aie_clear_tile_memory(_xaie, 7, 3);
  mlir_aie_clear_tile_memory(_xaie, 7, 4);
  mlir_aie_clear_tile_memory(_xaie, 6, 3);
  mlir_aie_clear_tile_memory(_xaie, 6, 4);
  mlir_aie_configure_cores(_xaie);

  usleep(sleep_u);
  printf("before configure switchboxes.\n");

  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

  mlir_aie_release_lock(_xaie, 6, 0, 0, 0, 0);
  mlir_aie_release_lock(_xaie, 6, 0, 1, 0, 0);
  mlir_aie_release_lock(_xaie, 6, 0, 2, 0, 0);
  mlir_aie_release_lock(_xaie, 6, 0, 3, 0, 0);
  mlir_aie_release_lock(_xaie, 7, 0, 0, 0, 0);
  mlir_aie_release_lock(_xaie, 7, 0, 1, 0, 0);

  usleep(sleep_u);
  printf("before configure DMA\n");

  mlir_aie_configure_dmas(_xaie);
  mlir_aie_init_mems(_xaie, 8);
  int errors = 0;

  printf("Finish configure\n");
#define DMA_COUNT 1024
  int *mem_ptr0 = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT);
  int *mem_ptr1 = mlir_aie_mem_alloc(_xaie, 1, DMA_COUNT);
  int *mem_ptr2 = mlir_aie_mem_alloc(_xaie, 2, DMA_COUNT);
  int *mem_ptr3 = mlir_aie_mem_alloc(_xaie, 3, DMA_COUNT);
  int *mem_ptr4 = mlir_aie_mem_alloc(_xaie, 4, DMA_COUNT);
  int *mem_ptr5 = mlir_aie_mem_alloc(_xaie, 5, DMA_COUNT);
  int *mem_ptr6 = mlir_aie_mem_alloc(_xaie, 6, DMA_COUNT + 1);
  int *mem_ptr7 = mlir_aie_mem_alloc(_xaie, 7, DMA_COUNT + 1);

  // initialize the external buffers
  for (int i = 0; i < DMA_COUNT + 1; i++) {
    if (i == 0) {
      *(mem_ptr6 + i) = 99;
      *(mem_ptr7 + i) = 99;
    } else {
      *(mem_ptr0 + i - 1) = 1; // LHS_tile0
      *(mem_ptr1 + i - 1) = 2; // LHS_tile1
      *(mem_ptr2 + i - 1) = 3; // RHS_tile0
      *(mem_ptr3 + i - 1) = 4; // RHS_tile1
      *(mem_ptr4 + i - 1) = 5; // RHS_tile2
      *(mem_ptr5 + i - 1) = 6; // RHS_tile3
      *(mem_ptr6 + i) = 99;    // Out_tile0
      *(mem_ptr7 + i) = 99;    // Out_tile1
    }
  }

  mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 1); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 2); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 3); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 4); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 5); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 6); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 7); // only used in libaiev2

#ifdef LIBXAIENGINEV2
  mlir_aie_external_set_addr_myBuffer_60_0((u64)mem_ptr0);
  mlir_aie_external_set_addr_myBuffer_60_1((u64)mem_ptr1);
  mlir_aie_external_set_addr_myBuffer_60_2((u64)mem_ptr6);
  mlir_aie_external_set_addr_myBuffer_60_3((u64)mem_ptr7);
  mlir_aie_external_set_addr_myBuffer_70_0((u64)mem_ptr2);
  mlir_aie_external_set_addr_myBuffer_70_1((u64)mem_ptr3);
  mlir_aie_external_set_addr_myBuffer_100_0((u64)mem_ptr4);
  mlir_aie_external_set_addr_myBuffer_100_1((u64)mem_ptr5);
  mlir_aie_configure_shimdma_60(_xaie);
  mlir_aie_configure_shimdma_70(_xaie);
  mlir_aie_configure_shimdma_100(_xaie);
#endif

  printf("before core start\n");

  mlir_aie_start_cores(_xaie);

  mlir_aie_release_lock(_xaie, 6, 0, 0, 1, 0);
  mlir_aie_release_lock(_xaie, 6, 0, 1, 1, 0);
  mlir_aie_release_lock(_xaie, 7, 0, 0, 1, 0);
  mlir_aie_release_lock(_xaie, 7, 0, 1, 1, 0);
  mlir_aie_release_lock(_xaie, 10, 0, 0, 1, 0);
  mlir_aie_release_lock(_xaie, 10, 0, 1, 1, 0);

  usleep(sleep_u);

  for (int idx0 = 1; idx0 < 1025; ++idx0) {
    if (mem_ptr6[idx0] != 352) {
      printf("Out_tile0[%d]=%d\n", idx0 - 1, mem_ptr6[idx0]);
      errors++;
    }
    if (mem_ptr7[idx0] != 544) {
      printf("Out_tile1[%d]=%d\n", idx0 - 1, mem_ptr7[idx0]);
      errors++;
    }
  }

  // if (Header0 == 7 && Header1 == 6) {
  //   for (int idx0 = 1; idx0 < 1025; ++idx0) {
  //     if (mem_ptr6[idx0] != 544) {
  //       printf("Out_tile0[%d]=%d\n", idx0 - 1, mem_ptr6[idx0]);
  //       errors++;
  //     }
  //     if (mem_ptr7[idx0] != 352) {
  //       printf("Out_tile1[%d]=%d\n", idx0 - 1, mem_ptr7[idx0]);
  //       errors++;
  //     }
  //   }
  // }

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
