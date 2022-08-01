//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
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

  mlir_aie_configure_cores(_xaie);

  usleep(sleep_u);
  printf("before configure sw.\n");

  mlir_aie_configure_switchboxes(_xaie);

  usleep(sleep_u);
  printf("before DMA config\n");

  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);
  int errors = 0;

  printf("Finish configure\n");
#define DMA_COUNT 1024

  mlir_aie_clear_tile_memory(_xaie, 7, 2);
  mlir_aie_clear_tile_memory(_xaie, 7, 3);
  mlir_aie_clear_tile_memory(_xaie, 7, 4);
  mlir_aie_clear_tile_memory(_xaie, 6, 3);
  mlir_aie_clear_tile_memory(_xaie, 6, 4);

  usleep(sleep_u);

  for (int bd = 0; bd < DMA_COUNT; bd++) {
    mlir_aie_write_buffer_buf72_0(_xaie, bd, 720);
    mlir_aie_write_buffer_buf72_1(_xaie, bd, 721);
  }

  printf("before core start\n");
  usleep(sleep_u);

  mlir_aie_start_cores(_xaie);

  mlir_aie_release_lock(_xaie, 7, 2, 4, 1, 0);
  mlir_aie_release_lock(_xaie, 7, 2, 5, 1, 0);

  usleep(sleep_u);

  for (int bd = 0; bd < DMA_COUNT; bd++) {
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf72_0(_xaie, bd), 720, errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf72_1(_xaie, bd), 721, errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf63_0(_xaie, bd), 720, errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf73_0(_xaie, bd), 720, errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf64_0(_xaie, bd), 721, errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf74_0(_xaie, bd), 721, errors);
  }

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
