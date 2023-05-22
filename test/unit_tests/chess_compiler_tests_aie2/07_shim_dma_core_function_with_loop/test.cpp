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

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);

  u32 sleep_u = 100000;
  usleep(sleep_u);
  printf("before DMA config\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  mlir_aie_configure_dmas(_xaie);

  usleep(sleep_u);
  printf("after DMA config\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  int errors = 0;
  mlir_aie_init_mems(_xaie, 2);
#define DMA_COUNT 512
  int *ddr_ptr_in = mlir_aie_mem_alloc(_xaie, 0, DMA_COUNT);
  int *ddr_ptr_out = mlir_aie_mem_alloc(_xaie, 1, DMA_COUNT);
  for (int i = 0; i < DMA_COUNT; i++) {
    *(ddr_ptr_in + i) = i;
    *(ddr_ptr_out + i) = 0;
  }
  mlir_aie_sync_mem_dev(_xaie, 0); // only used in libaiev2
  mlir_aie_sync_mem_dev(_xaie, 1); // only used in libaiev2

  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  mlir_aie_external_set_addr_input_buffer((u64)ddr_ptr_in);
  mlir_aie_external_set_addr_output_buffer((u64)ddr_ptr_out);
  mlir_aie_configure_shimdma_70(_xaie);

  mlir_aie_clear_tile_memory(_xaie, 7, 3);

  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  usleep(sleep_u);
  printf("before core start\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  usleep(sleep_u);
  printf("after core start\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);
  u32 locks70;
  locks70 =
      mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 7, 0) + 0x00014F00);
  printf("Locks70 = %08X\n", locks70);

  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  printf("Release lock for accessing DDR.\n");
  mlir_aie_release_input_lock(_xaie, /*r/w*/ 1, 0);
  mlir_aie_release_output_lock(_xaie, /*r/w*/ 1, 0);

  usleep(sleep_u);
  printf("after lock release\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  locks70 =
      mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 7, 0) + 0x00014F00);
  printf("Locks70 = %08X\n", locks70);

  mlir_aie_check("After", mlir_aie_read_buffer_a_ping(_xaie, 0), 384, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_a_pong(_xaie, 0), 448, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_b_ping(_xaie, 0), 385, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_b_pong(_xaie, 0), 449, errors);

  mlir_aie_dump_tile_memory(_xaie, 7, 3);

  mlir_aie_sync_mem_dev(_xaie, 1); // only used in libaiev2
  mlir_aie_sync_mem_cpu(_xaie, 1); // only used in libaiev2

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
