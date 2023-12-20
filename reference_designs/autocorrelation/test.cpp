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

  mlir_aie_clear_tile_memory(_xaie, 1, 3);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);

  int errors = 0;

#define DMA_COUNT 1024
  ext_mem_model_t buf0, buf1;
  int *input = mlir_aie_mem_alloc(_xaie, buf0, DMA_COUNT);
  int *output = mlir_aie_mem_alloc(_xaie, buf1, DMA_COUNT);

  mlir_aie_external_set_addr_input(_xaie, (u64)input);
  mlir_aie_external_set_addr_output(_xaie, (u64)output);

  mlir_aie_configure_shimdma_70(_xaie);
  mlir_aie_start_cores(_xaie);

  for (int i = 0; i < 64; i++) {
    input[i] = rand() & 0xFF;
  }

  mlir_aie_sync_mem_dev(buf0); // only used in libaiev2
  mlir_aie_sync_mem_dev(buf1); // only used in libaiev2

  mlir_aie_release_input_lock(_xaie, 1, 10000);

  printf("Waiting for the result ...\n");
  if (mlir_aie_acquire_output_lock(_xaie, 1, 100000)) {
    printf("ERROR: timeout hit!\n");
  }

  mlir_aie_sync_mem_dev(buf0); // only used in libaiev2
  mlir_aie_sync_mem_dev(buf1); // only used in libaiev2

  for (int i = 0; i < 64; i++) {
    printf("output[%d] = %d\n", i, output[i]);
  }

  mlir_aie_check("output[63]", output[63], 13905, errors);

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
