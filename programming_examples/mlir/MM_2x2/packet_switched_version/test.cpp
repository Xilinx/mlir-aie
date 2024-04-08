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

  mlir_aie_release_LHS_tile0_lock(_xaie, 0, 0);
  mlir_aie_release_LHS_tile1_lock(_xaie, 0, 0);
  mlir_aie_release_RHS_tile0_lock(_xaie, 0, 0);
  mlir_aie_release_RHS_tile1_lock(_xaie, 0, 0);
  mlir_aie_release_RHS_tile2_lock(_xaie, 0, 0);
  mlir_aie_release_RHS_tile3_lock(_xaie, 0, 0);

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
  int *mem_ptr6 = mlir_aie_mem_alloc(_xaie, buf6, DMA_COUNT + 1);
  int *mem_ptr7 = mlir_aie_mem_alloc(_xaie, buf7, DMA_COUNT + 1);

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
  mlir_aie_configure_shimdma_70(_xaie);
  mlir_aie_configure_shimdma_60(_xaie);

  printf("before core start\n");

  mlir_aie_release_LHS_tile0_lock(_xaie, 1, 0);
  mlir_aie_release_LHS_tile1_lock(_xaie, 1, 0);
  mlir_aie_release_RHS_tile0_lock(_xaie, 1, 0);
  mlir_aie_release_RHS_tile1_lock(_xaie, 1, 0);
  mlir_aie_release_RHS_tile2_lock(_xaie, 1, 0);
  mlir_aie_release_RHS_tile3_lock(_xaie, 1, 0);

  mlir_aie_start_cores(_xaie);

  usleep(sleep_u);
  // Check if the local buffer contain the correct data
  for (int bd = 0; bd < 4; bd++) {
    mlir_aie_check("Before release lock LHS:",
                   mlir_aie_read_buffer_buf63_0(_xaie, bd), 1, errors);
    mlir_aie_check("Before release lock RHS:",
                   mlir_aie_read_buffer_buf63_1(_xaie, bd), 3, errors);
    mlir_aie_check("Before release lock ACC:",
                   mlir_aie_read_buffer_buf63_3(_xaie, bd), 96, // Sub_sum0
                   errors);
    mlir_aie_check("Before release lock LHS:",
                   mlir_aie_read_buffer_buf64_0(_xaie, bd), 2, errors);
    mlir_aie_check("Before release lock RHS:",
                   mlir_aie_read_buffer_buf64_1(_xaie, bd), 4, errors);
    mlir_aie_check("Before release lock Out:",
                   mlir_aie_read_buffer_buf64_2(_xaie, bd), 352, // Out_tile0
                   errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf73_0(_xaie, bd), 1, errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf73_1(_xaie, bd), 5, errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf73_3(_xaie, bd), 160, // Sub_sum1
                   errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf74_0(_xaie, bd), 2, errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf74_1(_xaie, bd), 6, errors);
    mlir_aie_check("Before release lock:",
                   mlir_aie_read_buffer_buf74_2(_xaie, bd), 544, // Out_tile1
                   errors);
  }

  mlir_aie_sync_mem_cpu(buf6); // only used in libaiev2
  mlir_aie_sync_mem_cpu(buf7); // only used in libaiev2

  // Check if the external buffer receives the correct result
  int Header0 = mem_ptr6[0] & 31;
  int Header1 = mem_ptr7[0] & 31;

  printf("Header 0 = %d\n", Header0);
  printf("Header 1 = %d\n", Header1);

  // Compare the result according to the header since the order of the result is
  // not known
  if (Header0 == 6 && Header1 == 7) {
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
  } else if (Header0 == 7 && Header1 == 6) {
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
  } else {
    errors++;
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
