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

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

  printf("before DMA config\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  mlir_aie_configure_dmas(_xaie);

  printf("after DMA config\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  int errors = 0;
#define DMA_COUNT 512
  ext_mem_model_t buf0, buf1;
  int *ddr_ptr_in = mlir_aie_mem_alloc(buf0, DMA_COUNT);
  int *ddr_ptr_out = mlir_aie_mem_alloc(buf1, DMA_COUNT);
  for (int i = 0; i < DMA_COUNT; i++) {
    *(ddr_ptr_in + i) = i;
    *(ddr_ptr_out + i) = 0;
  }
  mlir_aie_sync_mem_dev(buf0);
  mlir_aie_sync_mem_dev(buf1);

  mlir_aie_print_shimdma_status(_xaie, 7, 0);

#ifdef __AIESIM__
    mlir_aie_external_set_addr_input_buffer(buf0.physicalAddr);
    mlir_aie_external_set_addr_output_buffer(buf1.physicalAddr);
#else
    mlir_aie_external_set_addr_input_buffer((u64)ddr_ptr_in);
    mlir_aie_external_set_addr_output_buffer((u64)ddr_ptr_out);
#endif
  mlir_aie_configure_shimdma_70(_xaie);

  mlir_aie_clear_tile_memory(_xaie, 7, 3);

  mlir_aie_acquire_input_lock_write(_xaie, -1, 0);

  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  printf("before core start\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  printf("after core start\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  printf("Release lock for accessing DDR.\n");
  mlir_aie_release_input_lock_read(_xaie, 1, 0);

  if (mlir_aie_acquire_output_lock_read(_xaie, -1, 0)) {
    errors++;
  }
  
  mlir_aie_check("After", mlir_aie_read_buffer_a_ping(_xaie, 0), 384, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_a_pong(_xaie, 0), 448, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_b_ping(_xaie, 0), 385, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_b_pong(_xaie, 0), 449, errors);

  mlir_aie_sync_mem_cpu(buf1);

  // Dump contents of ddr_ptr_out
  for (int i = 0; i < 16; i++) {
    uint32_t d = ddr_ptr_out[i];
    printf("ddr_ptr_out[%d] = %d\n", i, d);
  }

  for (int i = 0; i < 512; i++)
    mlir_aie_check("DDR out", ddr_ptr_out[i], i + 1, errors);

  /*
  XAieDma_Shim ShimDmaInst1;
  XAieDma_ShimSoftInitialize(&(TileInst[7][0]), &ShimDmaInst1);
  XAieDma_ShimBdClearAll((&ShimDmaInst1));
  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, XAIE_DISABLE,
  XAIE_DISABLE, XAIE_DISABLE); XAieDma_ShimChControl((&ShimDmaInst1),
  XAIEDMA_SHIM_CHNUM_S2MM0, XAIE_DISABLE, XAIE_DISABLE, XAIE_DISABLE);
  */

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
