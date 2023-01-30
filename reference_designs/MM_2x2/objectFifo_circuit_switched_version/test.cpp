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
  int *mem_ptr6 = mlir_aie_mem_alloc(_xaie, 6, DMA_COUNT);
  int *mem_ptr7 = mlir_aie_mem_alloc(_xaie, 7, DMA_COUNT);

  // initialize the external buffers
  for (int i = 0; i < DMA_COUNT; i++) {
    *(mem_ptr0 + i) = 1;  // LHS_tile0
    *(mem_ptr1 + i) = 2;  // LHS_tile1
    *(mem_ptr2 + i) = 3;  // RHS_tile0
    *(mem_ptr3 + i) = 4;  // RHS_tile1
    *(mem_ptr4 + i) = 5;  // RHS_tile2
    *(mem_ptr5 + i) = 6;  // RHS_tile3
    *(mem_ptr6 + i) = 99; // Out_tile0
    *(mem_ptr7 + i) = 99; // Out_tile1
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
  mlir_aie_external_set_addr_LHS_tile0((u64)mem_ptr0);
  mlir_aie_external_set_addr_LHS_tile1((u64)mem_ptr1);
  mlir_aie_external_set_addr_RHS_tile0((u64)mem_ptr2);
  mlir_aie_external_set_addr_RHS_tile1((u64)mem_ptr3);
  mlir_aie_external_set_addr_RHS_tile2((u64)mem_ptr4);
  mlir_aie_external_set_addr_RHS_tile3((u64)mem_ptr5);
  mlir_aie_external_set_addr_Out_tile0((u64)mem_ptr6);
  mlir_aie_external_set_addr_Out_tile1((u64)mem_ptr7);
  mlir_aie_configure_shimdma_60(_xaie);
  mlir_aie_configure_shimdma_70(_xaie);
  mlir_aie_configure_shimdma_100(_xaie);
#endif

  printf("before core start\n");

  mlir_aie_release_producer_objFifo_LHS0_0(_xaie, 10000);
  mlir_aie_release_producer_objFifo_LHS1_0(_xaie, 10000);
  mlir_aie_release_producer_objFifo_RHS0_0(_xaie, 10000);
  mlir_aie_release_producer_objFifo_RHS1_0(_xaie, 10000);
  mlir_aie_release_producer_objFifo_RHS2_0(_xaie, 10000);
  mlir_aie_release_producer_objFifo_RHS3_0(_xaie, 10000);

  mlir_aie_start_cores(_xaie);

  usleep(sleep_u);

  mlir_aie_acquire_consumer_objFifo_out0_0(_xaie, 10000);
  mlir_aie_acquire_consumer_objFifo_out1_0(_xaie, 10000);
  mlir_aie_sync_mem_cpu(_xaie, 6); // only used in libaiev2
  mlir_aie_sync_mem_cpu(_xaie, 7); // only used in libaiev2

  for (int idx0 = 0; idx0 < 1024; ++idx0) {
    if (mem_ptr6[idx0] != 352) {
      printf("Out_tile0[%d]=%d\n", idx0, mem_ptr6[idx0]);
      errors++;
    }
    if (mem_ptr7[idx0] != 544) {
      printf("Out_tile1[%d]=%d\n", idx0, mem_ptr7[idx0]);
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
