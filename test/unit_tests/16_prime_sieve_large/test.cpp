//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
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

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

#define MLIR_STACK_OFFSET 4096

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie_inc.cpp"

}

int
main(int argc, char *argv[])
{
    printf("test start.\n");

    size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
    XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
    XAieGbl_HwInit(&AieConfig);
    AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
    XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

    ACDC_clear_tile_memory(TileInst[1][3]);

    mlir_configure_cores();
    mlir_configure_switchboxes();
    mlir_initialize_locks();
    mlir_configure_dmas();
    mlir_start_cores();

    int errors = 0;

    printf("Waiting for the result ...\n");
    if(!XAieTile_LockAcquire(&(TileInst[1][6]), 0, 1, 100)) {
        printf("ERROR: timeout hit!\n");
    }

    ACDC_dump_tile_memory(TileInst[1][6]);
    ACDC_check("prime5[0]", mlir_read_buffer_prime5(0),7, errors);
    ACDC_check("prime5[1]", mlir_read_buffer_prime5(1),11, errors);
    ACDC_check("prime5[2]", mlir_read_buffer_prime5(2),13, errors);

    if (!errors) {
        printf("PASS!\n"); return 0;
    } else {
        printf("Fail!\n"); return -1;
    }
    printf("test done.\n");
}
