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

#include "aie_inc.cpp"
#define MAP_SIZE 16UL
#define MAP_MASK (MAP_SIZE - 1)

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

  printf("before configure cores.\n");
  mlir_aie_configure_cores(_xaie);

  printf("before configure sw.\n");
  mlir_aie_configure_switchboxes(_xaie);

  printf("before DMA config\n");
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_init_mems(_xaie, 2);
  int errors = 0;

  printf("Finish configure\n");

  mlir_aie_clear_tile_memory(_xaie, 7, 2);

  #define DMA_COUNT 256
  mlir_aie_init_mems(_xaie, 2);
  int *mem_ptr0 = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT);
  int *mem_ptr1 = mlir_aie_mem_alloc(_xaie, 1, DMA_COUNT + 1);

  for (int i = 0; i < DMA_COUNT + 1; i++) {
    if (i == 0) {
      mem_ptr1[0] = 1;
    } else {
      mem_ptr0[i - 1] = 72;
      mem_ptr1[i] = 1;
    }
  }
  mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 1); // only used in libaiev2

#ifdef LIBXAIENGINEV2
  mlir_aie_external_set_addr_input((u64)mem_ptr0);
  mlir_aie_external_set_addr_output((u64)mem_ptr1);
  mlir_aie_configure_shimdma_70(_xaie);
#endif

  printf("before core start\n");

  mlir_aie_release_output_lock(_xaie, 0, 0);
  mlir_aie_release_inter_lock(_xaie, 0, 0);

  mlir_aie_start_cores(_xaie);

  mlir_aie_release_input_lock(_xaie, 1, 0);

  if (mlir_aie_acquire_output_lock(_xaie, 1, 1000) == XAIE_OK)
    printf("Acquired output_lock (1) in tile (7,0). Done.\n");
  else {
    errors++;
    printf("Timed out while trying to acquire output_lock.\n");
  }

  mlir_aie_print_dma_status(_xaie, 7, 2);
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  mlir_aie_sync_mem_cpu(_xaie, 1); // only used in libaiev2

  for (int bd = 0; bd < DMA_COUNT + 1; bd++) {
    if (bd == 0) {
      printf("External memory1[0]=%x\n", mem_ptr1[0]);
    } else if (mem_ptr1[bd] != 72) {
      printf("External memory1[%d]=%d\n", bd, mem_ptr1[bd]);
      errors++;
    }
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
