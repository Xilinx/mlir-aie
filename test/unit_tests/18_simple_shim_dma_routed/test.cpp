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

#define XAIE_NUM_ROWS 8
#define XAIE_NUM_COLS 50
#define XAIE_ADDR_ARRAY_OFF 0x800

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)

namespace {

XAieGbl_Config *AieConfigPtr; /**< AIE configuration pointer */
XAieGbl AieInst;              /**< AIE global instance */
XAieGbl_HwCfg AieConfig;      /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS +
                                     1]; /**< Instantiates AIE array of
                                            [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS + 1];

#include "aie_inc.cpp"

} // namespace

int main(int argc, char *argv[]) {
  auto col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS,
                           XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  ACDC_print_tile_status(TileInst[7][2]);

  // Run auto generated config functions

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();

  // XAieDma_Shim ShimDmaInst1;
  uint32_t *bram_ptr;

#define BRAM_ADDR (0x4000 + 0x020100000000LL)
#define DMA_COUNT 512

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ | PROT_WRITE,
                                MAP_SHARED, fd, BRAM_ADDR);
    for (int i = 0; i < DMA_COUNT; i++) {
      bram_ptr[i] = i + 1;
      // printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);
    }
  }

  // We're going to stamp over the memory
  for (int i = 0; i < DMA_COUNT; i++) {
    mlir_write_buffer_buf72_0(i, 0xdeadbeef);
  }

  XAieTile_LockRelease(&(TileInst[7][0]), 1, 1,
                       0); // Release lock for reading from DDR

  ACDC_print_tile_status(TileInst[7][2]);

  int errors = 0;
  for (int i = 0; i < DMA_COUNT; i++) {
    uint32_t d = mlir_read_buffer_buf72_0(i);
    if (d != (i + 1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, i);
    }
  }

  if (!errors) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail %d/%d.\n", (DMA_COUNT - errors), DMA_COUNT);
    return -1;
  }
}
