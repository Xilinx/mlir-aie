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
    
    mlir_configure_cores();    
    mlir_configure_switchboxes();
    mlir_initialize_locks();

    u32 sleep_u = 100000; 
    usleep(sleep_u);
    printf("before DMA config\n");
    ACDC_print_tile_status(TileInst[7][3]);
    
    mlir_configure_dmas();

    usleep(sleep_u);
    printf("after DMA config\n");
    ACDC_print_tile_status(TileInst[7][3]);

    int errors = 0;

    uint32_t *ddr_ptr_in, *ddr_ptr_out;
    #define DDR_ADDR_IN  (0x4000+0x020100000000LL)
    #define DDR_ADDR_OUT (0x6000+0x020100000000LL)
    #define DMA_COUNT 512

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd != -1) {
        ddr_ptr_in  = (uint32_t *)mmap(NULL, 0x800, PROT_READ|PROT_WRITE, MAP_SHARED, fd, DDR_ADDR_IN);
        ddr_ptr_out = (uint32_t *)mmap(NULL, 0x800, PROT_READ|PROT_WRITE, MAP_SHARED, fd, DDR_ADDR_OUT);
        for (int i=0; i<DMA_COUNT; i++) {
            ddr_ptr_in[i] = i;
            ddr_ptr_out[i] = 0;
        }
    }

    ACDC_clear_tile_memory(TileInst[7][3]);  
/*
    // TODO Check for completion of shimdma
    int shimdma_stat_mm2s0, shimdma_stat_s2mm0;
    XAieDma_Shim ShimDMAInst_7_0;
    XAieDma_ShimInitialize(&(TileInst[7][0]), &ShimDMAInst_7_0);
    shimdma_stat_mm2s0 = XAieDma_ShimPendingBdCount(&ShimDMAInst_7_0, XAIEDMA_SHIM_CHNUM_MM2S0);
    shimdma_stat_s2mm0 = XAieDma_ShimPendingBdCount(&ShimDMAInst_7_0, XAIEDMA_SHIM_CHNUM_S2MM0);
    printf("shimdma_stat_mm2s0/s2mm0 = %d/ %d\n",shimdma_stat_mm2s0, shimdma_stat_s2mm0);
*/

    usleep(sleep_u);
    printf("before core start\n");
    ACDC_print_tile_status(TileInst[7][3]);

    printf("Start cores\n");
    mlir_start_cores();

    usleep(sleep_u);
    printf("after core start\n");
    ACDC_print_tile_status(TileInst[7][3]);
    u32 locks70;
    locks70 = XAieGbl_Read32(TileInst[7][0].TileAddr + 0x00014F00);
    printf("Locks70 = %08X\n", locks70);

    printf("Release lock for accessing DDR.\n");
    XAieTile_LockRelease(&(TileInst[7][0]), /*lockid*/ 1, /*r/w*/ 1, 0); 
    XAieTile_LockRelease(&(TileInst[7][0]), /*lockid*/ 2, /*r/w*/ 1, 0); 

    usleep(sleep_u);
    printf("after lock release\n");
    ACDC_print_tile_status(TileInst[7][3]);
    locks70 = XAieGbl_Read32(TileInst[7][0].TileAddr + 0x00014F00);
    printf("Locks70 = %08X\n", locks70);

    ACDC_check("After", mlir_read_buffer_a_ping(0), 384, errors);
    ACDC_check("After", mlir_read_buffer_a_pong(0), 448, errors);
    ACDC_check("After", mlir_read_buffer_b_ping(0), 385, errors);
    ACDC_check("After", mlir_read_buffer_b_pong(0), 449, errors);    

    // Dump contents of ddr_ptr_out
    for (int i=0; i<16; i++) {
        uint32_t d = ddr_ptr_out[i];
        printf("ddr_ptr_out[%d] = %d\n", i, d);
    }

    for (int i=0; i<512; i++)
        ACDC_check("DDR out",ddr_ptr_out[i],i+1, errors);

    /*
    XAieDma_Shim ShimDmaInst1;
    XAieDma_ShimSoftInitialize(&(TileInst[7][0]), &ShimDmaInst1);
    XAieDma_ShimBdClearAll((&ShimDmaInst1));
    XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S0, XAIE_DISABLE, XAIE_DISABLE, XAIE_DISABLE);
    XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_S2MM0, XAIE_DISABLE, XAIE_DISABLE, XAIE_DISABLE);
    */

    if (!errors) {
        printf("PASS!\n"); return 0;
    } else {
        printf("Fail!\n"); return -1;
    }
}