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
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <xaiengine.h>

#include "memory_allocator.h"
#include "test_library.h"

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_print_tile_status(_xaie, 7, 2);

  // Run auto generated config functions
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);

#define DMA_COUNT 512
  ext_mem_model_t buf0;
  int *ddr_ptr_in = mlir_aie_mem_alloc(_xaie, buf0, DMA_COUNT);
  for (int i = 0; i < DMA_COUNT; i++) {
    *(ddr_ptr_in + i) = i + 1;
  }
  mlir_aie_sync_mem_dev(buf0);

  mlir_aie_external_set_addr_input_buffer(_xaie, (u64)ddr_ptr_in);
  mlir_aie_configure_shimdma_70(_xaie);

  // We're going to stamp over the memory
  for (int i = 0; i < DMA_COUNT; i++) {
    mlir_aie_write_buffer_buf72_0(_xaie, i, 0xdeadbeef);
  }

  // Release lock for reading from DDR
  mlir_aie_release_input_lock(_xaie, 1, 0);
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
