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

//#define XAIE_NUM_ROWS 8
//#define XAIE_NUM_COLS 50
//#define XAIE_ADDR_ARRAY_OFF 0x800

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  auto col = 7;

  mlir_aie_print_tile_status(_xaie, 7, 2);

  // Run auto generated config functions

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);

  // XAieDma_Shim ShimDmaInst1;
  /*  uint32_t *bram_ptr;

  #define BRAM_ADDR (0x4000 + 0x020100000000LL)
  #define DMA_COUNT 512

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd != -1) {
      bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd, BRAM_ADDR);
      for (int i = 0; i < DMA_COUNT; i++) {
        bram_ptr[i] = i + 1;
        // printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);
      }
    }
  */
  mlir_aie_init_mems(_xaie, 1);
#define DMA_COUNT 512
  int *ddr_ptr_in =
      mlir_aie_mem_alloc(_xaie, 0, 0x4000 + 0x020100000000LL, DMA_COUNT);
  for (int i = 0; i < DMA_COUNT; i++) {
    *(ddr_ptr_in + i) = i + 1;
  }
  mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2

#ifdef LIBXAIENGINEV2
  mlir_aie_external_set_addr_myBuffer_70_0((u64)ddr_ptr_in);
  mlir_aie_configure_shimdma_70(_xaie);
#endif

  // We're going to stamp over the memory
  for (int i = 0; i < DMA_COUNT; i++) {
    mlir_aie_write_buffer_buf72_0(_xaie, i, 0xdeadbeef);
  }

  mlir_aie_release_lock(_xaie, 7, 0, 1, 1,
                        0); // Release lock for reading from DDR

  mlir_aie_print_tile_status(_xaie, 7, 2);

  int errors = 0;
  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t d = mlir_aie_read_buffer_buf72_0(_xaie, i);
    if (d != (i + 1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, i);
    }
  }

  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("fail %d/%d.\n", (DMA_COUNT - errors), DMA_COUNT);
    res = -1;
  }
  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.\n");
  return res;
}
