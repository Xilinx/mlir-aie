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

#include "aie_inc.cpp"

constexpr int bufferSize = 64;
constexpr int numberOfSubBuffers = 4;

int main(int argc, char *argv[])
{
  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  // Run auto generated config functions
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);

  mlir_aie_init_mems(_xaie, 1);


  int *mem_ptr = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT);
  for (int i = 0; i < bufferSize; i++) {
    mem_ptr[i] = -1;
  }
  mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2

  mlir_aie_external_set_addr_buffer((u64)mem_ptr);

  if (mlir_aie_release_shimLock_0(_xaie, 0, 10000) == XAIE_OK)
    printf("Pre-Released shimlock for write\n");
  else
    printf("ERROR: timed shimLock for write\n");

  mlir_aie_configure_shimdma_70(_xaie);

  printf("\nAfter configure shimDMAs:\n");
  mlir_aie_print_tile_status(_xaie, 7, 2);
  mlir_aie_print_dma_status(_xaie, 7, 2);
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  int errors = 0;
  for (int j= 0; j < numberOfSubBuffers; j++){
    printf("Receiving sub-block: %d\n",j);
    // acquire output shim
    if (mlir_aie_acquire_shimLock_0(_xaie, 1, 10000) == XAIE_OK)
      printf("Acquired shimLock for read\n");
    else
      printf("ERROR: timed out on shimLock for read\n");

    // check output DDR
    mlir_aie_sync_mem_cpu(_xaie, 1);
    printSublock(mem_ptr, bufferSize/numberOfSubBuffers);
    for (int i = 0; i < bufferSize/numberOfSubBuffers; i++)
      mlir_aie_check("After start cores:", mem_ptr[i], j, errors);

    // release output shim
    if (mlir_aie_release_shimLock(_xaie, 0, 10000) == XAIE_OK)
      printf("Released shimLock for write\n");
    else
      printf("ERROR: timed out shimLock for write\n");

    

    }
  }

  
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
