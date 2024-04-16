//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "memory_allocator.h"
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
  int errors = 0;

  printf("Finish configure\n");
#define DMA_COUNT 1024
  ext_mem_model_t buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7;
  int *mem_ptr0 = mlir_aie_mem_alloc(_xaie, buf0, DMA_COUNT);
  int *mem_ptr1 = mlir_aie_mem_alloc(_xaie, buf1, DMA_COUNT);
  int *mem_ptr2 = mlir_aie_mem_alloc(_xaie, buf2, DMA_COUNT);
  int *mem_ptr3 = mlir_aie_mem_alloc(_xaie, buf3, DMA_COUNT);
  int *mem_ptr4 = mlir_aie_mem_alloc(_xaie, buf4, DMA_COUNT);
  int *mem_ptr5 = mlir_aie_mem_alloc(_xaie, buf5, DMA_COUNT);
  int *mem_ptr6 = mlir_aie_mem_alloc(_xaie, buf6, DMA_COUNT);
  int *mem_ptr7 = mlir_aie_mem_alloc(_xaie, buf7, DMA_COUNT);

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

  mlir_aie_sync_mem_dev(buf0); // only used in libaiev2
  mlir_aie_sync_mem_dev(buf1); // only used in libaiev2
  mlir_aie_sync_mem_dev(buf2); // only used in libaiev2
  mlir_aie_sync_mem_dev(buf3); // only used in libaiev2
  mlir_aie_sync_mem_dev(buf4); // only used in libaiev2
  mlir_aie_sync_mem_dev(buf5); // only used in libaiev2
  mlir_aie_sync_mem_dev(buf6); // only used in libaiev2
  mlir_aie_sync_mem_dev(buf7); // only used in libaiev2

  mlir_aie_external_set_addr_LHS_tile0(_xaie, (u64)mem_ptr0);
  mlir_aie_external_set_addr_LHS_tile1(_xaie, (u64)mem_ptr1);
  mlir_aie_external_set_addr_RHS_tile0(_xaie, (u64)mem_ptr2);
  mlir_aie_external_set_addr_RHS_tile1(_xaie, (u64)mem_ptr3);
  mlir_aie_external_set_addr_RHS_tile2(_xaie, (u64)mem_ptr4);
  mlir_aie_external_set_addr_RHS_tile3(_xaie, (u64)mem_ptr5);
  mlir_aie_external_set_addr_Out_tile0(_xaie, (u64)mem_ptr6);
  mlir_aie_external_set_addr_Out_tile1(_xaie, (u64)mem_ptr7);
  mlir_aie_configure_shimdma_60(_xaie);
  mlir_aie_configure_shimdma_70(_xaie);
  mlir_aie_configure_shimdma_100(_xaie);

  printf("before core start\n");

  mlir_aie_release_of_LHS0_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_LHS1_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_RHS0_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_RHS1_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_RHS2_lock_0(_xaie, 1, 0);
  mlir_aie_release_of_RHS3_lock_0(_xaie, 1, 0);

  mlir_aie_start_cores(_xaie);

  usleep(sleep_u);

  mlir_aie_acquire_of_out0_cons_lock_0(_xaie, 1, 0);
  mlir_aie_acquire_of_out1_cons_lock_0(_xaie, 1, 0);
  mlir_aie_sync_mem_cpu(buf6); // only used in libaiev2
  mlir_aie_sync_mem_cpu(buf7); // only used in libaiev2

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
