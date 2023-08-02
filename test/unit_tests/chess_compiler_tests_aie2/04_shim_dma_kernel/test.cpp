//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
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

  u32 sleep_u = 100000;
  printf("before DMA config\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);

  mlir_aie_configure_dmas(_xaie);

  printf("after DMA config\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);
  mlir_aie_print_dma_status(_xaie, 7, 3);
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  int errors = 0;

  /*
      uint32_t *ddr_ptr_in, *ddr_ptr_out;
      #define DDR_ADDR_IN  (0x4000+0x020100000000LL)
      #define DDR_ADDR_OUT (0x6000+0x020100000000LL)
      #define DMA_COUNT 512

      int fd = open("/dev/mem", O_RDWR | O_SYNC);
      if (fd != -1) {
          ddr_ptr_in  = (uint32_t *)mmap(NULL, 0x800, PROT_READ|PROT_WRITE,
     MAP_SHARED, fd, DDR_ADDR_IN); ddr_ptr_out = (uint32_t *)mmap(NULL, 0x800,
     PROT_READ|PROT_WRITE, MAP_SHARED, fd, DDR_ADDR_OUT); for (int i=0;
     i<DMA_COUNT; i++) { ddr_ptr_in[i] = i+1; ddr_ptr_out[i] = 0;
          }
      }
  */
#define DMA_COUNT 32
  ext_mem_model_t buf0, buf1;
  int *ddr_ptr_in = mlir_aie_mem_alloc(buf0, DMA_COUNT);
  int *ddr_ptr_out = mlir_aie_mem_alloc(buf1, DMA_COUNT);
  for (int i = 0; i < DMA_COUNT; i++) {
    *(ddr_ptr_in + i) = i + 1;
    *(ddr_ptr_out + i) = i + 2;
  }
  mlir_aie_sync_mem_dev(buf0);
  mlir_aie_sync_mem_dev(buf1);

#ifdef __AIESIM__
  mlir_aie_external_set_addr_input_buffer(buf0.physicalAddr);
  mlir_aie_external_set_addr_output_buffer(buf1.physicalAddr);
#else
  mlir_aie_external_set_addr_input_buffer((u64)ddr_ptr_in);
  mlir_aie_external_set_addr_output_buffer((u64)ddr_ptr_out);
#endif
  mlir_aie_configure_shimdma_70(_xaie);

  // mlir_aie_clear_tile_memory(_xaie, 7, 3);

  // Set iteration to 2 TODO: fix this
  // XAieTile_DmWriteWord(&(TileInst[7][3]), 5120 , 2);

  for (int i = 0; i < DMA_COUNT / 2; i++) {
    mlir_aie_write_buffer_a_ping(_xaie, i, 0x4);
    mlir_aie_write_buffer_a_pong(_xaie, i, 0x4);
    mlir_aie_write_buffer_b_ping(_xaie, i, 0x4);
    mlir_aie_write_buffer_b_pong(_xaie, i, 0x4);
  }

  mlir_aie_check("Before", mlir_aie_read_buffer_a_ping(_xaie, 3), 4, errors);
  mlir_aie_check("Before", mlir_aie_read_buffer_a_pong(_xaie, 3), 4, errors);
  mlir_aie_check("Before", mlir_aie_read_buffer_b_ping(_xaie, 5), 4, errors);
  mlir_aie_check("Before", mlir_aie_read_buffer_b_pong(_xaie, 5), 4, errors);

  //    mlir_aie_dump_tile_memory(TileInst[7][3]);

  /*
      // TODO Check for completion of shimdma
      int shimdma_stat_mm2s0, shimdma_stat_s2mm0;
      XAieDma_Shim ShimDMAInst_7_0;
      XAieDma_ShimInitialize(&(TileInst[7][0]), &ShimDMAInst_7_0);
      shimdma_stat_mm2s0 = XAieDma_ShimPendingBdCount(&ShimDMAInst_7_0,
     XAIEDMA_SHIM_CHNUM_MM2S0); shimdma_stat_s2mm0 =
     XAieDma_ShimPendingBdCount(&ShimDMAInst_7_0, XAIEDMA_SHIM_CHNUM_S2MM0);
      printf("shimdma_stat_mm2s0/s2mm0 = %d/ %d\n",shimdma_stat_mm2s0,
     shimdma_stat_s2mm0);
  */

  printf("before core start\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);
  mlir_aie_print_dma_status(_xaie, 7, 3);
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  printf("after core start\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);
  mlir_aie_print_dma_status(_xaie, 7, 3);
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  printf("Release lock for accessing DDR.\n");
  mlir_aie_release_input_lock_read(_xaie, 1, 10);

  // usleep(sleep_u);

  if (mlir_aie_acquire_output_lock_read(_xaie, -1, 200)) {
    errors++;
  }

  printf("after lock release\n");
  mlir_aie_print_tile_status(_xaie, 7, 3);
  mlir_aie_print_dma_status(_xaie, 7, 3);
  mlir_aie_print_shimdma_status(_xaie, 7, 0);

  mlir_aie_check("After", mlir_aie_read_buffer_a_ping(_xaie, 0), 1, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_a_ping(_xaie, 1), 2, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_a_ping(_xaie, 2), 3, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_a_ping(_xaie, 3), 4, errors);
  // mlir_aie_check("After", mlir_aie_read_buffer_a_ping(_xaie, 127), 128,
  // errors); mlir_aie_check("After", mlir_aie_read_buffer_a_ping(_xaie, 255),
  // 16, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_a_pong(_xaie, 3), 16 + 4,
                 errors);
  mlir_aie_check("After", mlir_aie_read_buffer_b_ping(_xaie, 5), 20, errors);
  mlir_aie_check("After", mlir_aie_read_buffer_b_pong(_xaie, 5), (16 + 4) * 5,
                 errors);
  printf("after lock release2\n");

  /*
      // Dump contents of ddr_ptr_out
      for (int i=0; i<DMA_COUNT; i++) {
          uint32_t d = ddr_ptr_out[i];
          if(d != 0)
              printf("ddr_ptr_out[%d] = %d\n", i, d);
      }
  */
  mlir_aie_sync_mem_cpu(buf1);
  mlir_aie_check("DDR out", ddr_ptr_out[5], 20, errors);
  mlir_aie_check("DDR out", ddr_ptr_out[16 + 5], (16 + 4) * 5, errors);

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
