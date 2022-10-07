//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2020 Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// phy-opt --convert-layout-to-physical="device=aie" \
//    --convert-physical-to-aie > mm_2x2-aie.mlir
// FIXME: Manually add { link_with="kernel.o" }, which should ba part of
// aiecc.py aiecc.py --sysroot=.. -v mm_2x2-aie.mlir \
//    mlir-aie/runtime_lib/test_library.cpp mm_2x2_driver.cpp \
//    -Imlir-aie/runtime_lib/ -o test.elf

#include "test_library.h"
#include <cassert>
#include <cstdio>
#include <unistd.h>

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)
#include "aie_inc.cpp"

#define DMA_COUNT 1024

int main(int argc, char *argv[]) {
  // Initialize device
  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);
  mlir_aie_clear_tile_memory(_xaie, 7, 3);
  mlir_aie_clear_tile_memory(_xaie, 6, 3);
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

  // Allocate BRAM
  mlir_aie_init_mems(_xaie, 8);
  int *mem_ptr0 = mlir_aie_mem_alloc(_xaie, 0, 0x020100000000LL, DMA_COUNT);
  int *mem_ptr1 = mlir_aie_mem_alloc(_xaie, 1, 0x020100001000LL, DMA_COUNT);
  int *mem_ptr2 = mlir_aie_mem_alloc(_xaie, 2, 0x020100002000LL, DMA_COUNT);
  int *mem_ptr3 = mlir_aie_mem_alloc(_xaie, 3, 0x020100003000LL, DMA_COUNT);
  int *mem_ptr4 = mlir_aie_mem_alloc(_xaie, 4, 0x020100004000LL, DMA_COUNT);
  int *mem_ptr5 = mlir_aie_mem_alloc(_xaie, 5, 0x020100005000LL, DMA_COUNT);
  int *mem_ptr6 = mlir_aie_mem_alloc(_xaie, 6, 0x020100006000LL, DMA_COUNT);
  int *mem_ptr7 = mlir_aie_mem_alloc(_xaie, 7, 0x020100007000LL, DMA_COUNT);

  // Start AIE device
  mlir_aie_configure_dmas(_xaie);
  mlir_aie_start_cores(_xaie);

  // Pass a set of data
  for (int i = 0; i < DMA_COUNT; i++) {
    mem_ptr0[i] = 1;
    mem_ptr1[i] = 2;
    mem_ptr2[i] = 3;
    mem_ptr3[i] = 4;
    mem_ptr4[i] = 5;
    mem_ptr5[i] = 6;
  }
  mlir_aie_sync_mem_dev(_xaie, 0);
  mlir_aie_sync_mem_dev(_xaie, 1);
  mlir_aie_sync_mem_dev(_xaie, 2);
  mlir_aie_sync_mem_dev(_xaie, 3);
  mlir_aie_sync_mem_dev(_xaie, 4);
  mlir_aie_sync_mem_dev(_xaie, 5);

  // and notify the AIE kernel
  mlir_aie_release_lock(_xaie, 6, 0, 0, 1, 0);
  mlir_aie_release_lock(_xaie, 6, 0, 1, 1, 0);
  mlir_aie_release_lock(_xaie, 6, 0, 2, 1, 0);
  mlir_aie_release_lock(_xaie, 6, 0, 3, 1, 0);
  mlir_aie_release_lock(_xaie, 7, 0, 0, 1, 0);
  mlir_aie_release_lock(_xaie, 7, 0, 1, 1, 0);

  // FIXME: this should be a spatial.push() in the host code
  mlir_aie_release_lock(_xaie, 6, 3, 2, 1, 0);
  mlir_aie_release_lock(_xaie, 7, 3, 2, 1, 0);

  // Wait for AIE kernel to finish
  usleep(100000);

  // and sync data back to CPU
  mlir_aie_sync_mem_cpu(_xaie, 6);
  mlir_aie_sync_mem_cpu(_xaie, 7);

  int errors = 0;
  // FIXME: remove DMA header
  // https://github.com/Xilinx/mlir-aie/pull/133/commits/0e5c6f9165014fcaec2acbe674ac6bca01f0726c
  for (int idx0 = 1; idx0 < 1024; ++idx0) {
    mlir_aie_check("C0", mem_ptr6[idx0], 352, errors);
    mlir_aie_check("C1", mem_ptr7[idx0], 544, errors);
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

  return res;
}