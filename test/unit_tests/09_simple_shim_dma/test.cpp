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

#include "aie_inc.cpp"

int
main(int argc, char *argv[])
{
  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  // Run auto generated config functions
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);

  mlir_aie_init_mems(_xaie, 1);

  #define DMA_COUNT 512
  int *mem_ptr = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT);
  for (int i = 0; i < DMA_COUNT; i++) {
    *(mem_ptr + i) = i + 1;
  }
  mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2

  // We're going to stamp over the memory
  for (int i=0; i<DMA_COUNT; i++) {
    mlir_aie_write_buffer_buf72_0(_xaie, i, 0xdeadbeef);
  }

#ifdef LIBXAIENGINEV2
  mlir_aie_external_set_addr_myBuffer_70_0((u64)mem_ptr);
  mlir_aie_configure_shimdma_70(_xaie);
#endif

  printf("\nAfter configure shimDMAs:\n");
  mlir_aie_print_tile_status(_xaie, 7, 2);
  mlir_aie_print_dma_status(_xaie, 7, 2);
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  // Release lock for reading from DDR
  mlir_aie_release_lock(_xaie, 7, 0, 1, 1, 0);

  printf("\nAfter release lock:\n");
  mlir_aie_print_tile_status(_xaie, 7, 2);
  mlir_aie_print_dma_status(_xaie, 7, 2);
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = 0;
    if (i < DMA_COUNT / 2)
      d = mlir_aie_read_buffer_buf72_0(_xaie, i);
    else
      d = mlir_aie_read_buffer_buf72_1(_xaie, i - DMA_COUNT / 2);
    if (d != (i+1)) {
      errors++;
      printf("mismatch %x != 1 + %d\n", d, i);
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
