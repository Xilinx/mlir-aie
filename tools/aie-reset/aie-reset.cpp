//===- aie-reset.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// This binary is useful for triggering a soft reset of the AIEngine
// array through the NPI interface.  The soft reset reinitializes all
// registers in the array, including data registers in the stream
// switches.  The intention is that under normal circumstances, this
// is not necessary and executing designs consume all of the data
// produced.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

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

void devmemRW(uint32_t address, uint32_t value, bool write) {
  int fd;
  void *map_base, *virt_addr;
  uint32_t read_result;
  uint64_t read_64;

  if ((fd = open("/dev/mem", O_RDWR | O_SYNC)) == -1)
    printf("ERROR!!!! open(devmem)\n");
  printf("\n/dev/mem opened.\n");
  fflush(stdout);

  map_base = mmap(0, MAP_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                  address & ~MAP_MASK);
  if (map_base == (void *)-1)
    printf("ERROR!!!! map_base\n");
  printf("Memory mapped at address %p.\n", map_base);
  fflush(stdout);

  if ((address % 8) == 4) {
    virt_addr = (char *)map_base + ((address - 4) & MAP_MASK);
    read_64 = *((volatile unsigned long long *)virt_addr);
    read_result = read_64 >> 32;
    printf("Value at address 0x%X (%p+4): 0x%X\n", address, virt_addr,
           read_result);
  } else {
    virt_addr = (char *)map_base + (address & MAP_MASK);
    read_result = *((volatile unsigned long *)virt_addr);
    printf("Value at address 0x%X (%p): 0x%X\n", address, virt_addr,
           read_result);
  }
  fflush(stdout);

  if (write) {
    if ((address % 8) == 4) {
      // printf("DEBUG: read64: 0x%llX\n", read_64);
      uint64_t write_64 =
          (read_64 & 0x00000000FFFFFFFF) | ((uint64_t)value << 32);
      // printf("DEBUG: write64: 0x%llX\n", write_64);
      *((volatile unsigned long long *)virt_addr) = write_64;
      // msync(map_base, MAP_SIZE, MS_SYNC);
      read_64 = *((volatile unsigned long long *)virt_addr);
      // printf("DEBUG: read64: 0x%llX\n", read_64);
      read_result = read_64 >> 32;
    } else {
      *((volatile unsigned long *)virt_addr) = value;
      // msync(map_base, MAP_SIZE, MS_SYNC);
      read_result = *((volatile unsigned long *)virt_addr);
    }
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

  return 0;
}