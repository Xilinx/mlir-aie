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


      devmemRW32(0xF70A000C, 0xF9E8D7C6, true);
    devmemRW32(0xF70A0000, 0x04000000, true);
    devmemRW32(0xF70A0004, 0x040381B1, true);
    devmemRW32(0xF70A0000, 0x04000000, true);
    devmemRW32(0xF70A0004, 0x000381B1, true);
    devmemRW32(0xF70A000C, 0x12341234, true);

    printf("test start.\n");

    int n = 1;
    u32 pc0_times[n];
    u32 pc1_times[n];
    u32 pc2_times[n];
    u32 pc3_times[n];
    u32 pc4_times[n];
    u32 pc5_times[n];
    u32 pc6_times[n];
    u32 pc7_times[n];

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

    // // Load IDCT Data:
    // File *file = fopen("image.txt")
    // int image[512];
    // int num;
    // while(fscanf(file, "%d", &num) > 0){
    //     image[i] = num;
    //     i++;
    // }
    // fclose(file);


    u_int16_t *ddr_ptr_in, *ddr_ptr_out;
    #define DDR_ADDR_IN  (0x4000+0x020100000000LL)
    #define DDR_ADDR_OUT (0x6000+0x020100000000LL)
    #define DMA_COUNT 512

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd != -1) {
        ddr_ptr_in  = (u_int16_t *)mmap(NULL, 0x800, PROT_READ|PROT_WRITE, MAP_SHARED, fd, DDR_ADDR_IN);
        ddr_ptr_out = (u_int16_t *)mmap(NULL, 0x800, PROT_READ|PROT_WRITE, MAP_SHARED, fd, DDR_ADDR_OUT);
        for (u_int16_t i=0; i<DMA_COUNT; i++) {
            ddr_ptr_in[i] = i;
            // ddr_ptr_in[i] = rand() % 255;
            ddr_ptr_out[i] = 0;
        }
    }


    EventMonitor pc0(&TileInst[7][3], 0, XAIETILE_EVENT_MEM_LOCK_3_ACQUIRED, XAIETILE_EVENT_MEM_LOCK_3_RELEASE, XAIETILE_EVENT_MEM_NONE, MODE_MEM);
    EventMonitor pc1(&TileInst[7][3], 1, XAIETILE_EVENT_MEM_LOCK_5_ACQUIRED, XAIETILE_EVENT_MEM_LOCK_5_RELEASE, XAIETILE_EVENT_MEM_NONE, MODE_MEM);

    EventMonitor pc2(&TileInst[6][3], 0, XAIETILE_EVENT_MEM_LOCK_3_ACQUIRED, XAIETILE_EVENT_MEM_LOCK_3_RELEASE, XAIETILE_EVENT_MEM_NONE, MODE_MEM);
    EventMonitor pc3(&TileInst[6][3], 1, XAIETILE_EVENT_MEM_LOCK_5_ACQUIRED, XAIETILE_EVENT_MEM_LOCK_5_RELEASE, XAIETILE_EVENT_MEM_NONE, MODE_MEM);


    EventMonitor pc4(&TileInst[5][3], 0, XAIETILE_EVENT_MEM_LOCK_3_ACQUIRED, XAIETILE_EVENT_MEM_LOCK_3_RELEASE, XAIETILE_EVENT_MEM_NONE, MODE_MEM);
    EventMonitor pc5(&TileInst[5][3], 1, XAIETILE_EVENT_MEM_LOCK_5_ACQUIRED, XAIETILE_EVENT_MEM_LOCK_5_RELEASE, XAIETILE_EVENT_MEM_NONE, MODE_MEM);


    EventMonitor pc6(&TileInst[7][0], 0, XAIETILE_EVENT_SHIM_LOCK_1_ACQUIRED_NOC, XAIETILE_EVENT_SHIM_LOCK_2_RELEASE_NOC, XAIETILE_EVENT_SHIM_NONE, MODE_PL);
    EventMonitor pc7(&TileInst[7][0], 1, XAIETILE_EVENT_SHIM_LOCK_2_ACQUIRED_NOC, XAIETILE_EVENT_SHIM_LOCK_2_RELEASE_NOC, XAIETILE_EVENT_SHIM_NONE, MODE_PL);

    pc0.set();
    pc1.set();
    pc2.set();
    pc3.set();
    pc4.set();
    pc5.set();
    pc6.set();
    pc7.set();

    // for (int i=0; i<DMA_COUNT; i++) {
    //     uint32_t d = ddr_ptr_in[i];
    //     printf("ddr_ptr_in[%d] = %d\n", i, d);
    // }

    
    // for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_a73_ping(i, 0x0);
    // }
    
    // for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_a73_pong(i, 0x0);
    // }

    
    // for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_b73_ping(i, 0x0);
    // }

    
    // for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_b73_pong(i, 0x0);
    // }

    //   for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_a74_ping(i, 0x0);
    // }
    
    // for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_a74_pong(i, 0x0);
    // }

    
    // for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_b74_ping(i, 0x0);
    // }

    
    // for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_b74_pong(i, 0x0);
    // }

    //     for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_a75_ping(i, 0x0);
    // }
    
    // for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_a75_pong(i, 0x0);
    // }

    
    // for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_b75_ping(i, 0x0);
    // }

    
    // for (int i=0; i<DMA_COUNT; i++) {
    //     mlir_write_buffer_b75_pong(i, 0x0);
    // }

    ACDC_clear_tile_memory(TileInst[7][3]);  
    ACDC_clear_tile_memory(TileInst[7][4]);  
    ACDC_clear_tile_memory(TileInst[7][5]);  

    printf("before core start\n");
    ACDC_print_tile_status(TileInst[7][3]);

    printf("Start cores\n");
    mlir_start_cores();

    // usleep(sleep_u);
    // printf("after core start\n");
    // ACDC_print_tile_status(TileInst[7][3]);
    // u32 locks70;
    // locks70 = XAieGbl_Read32(TileInst[7][0].TileAddr + 0x00014F00);
    // printf("Locks70 = %08X\n", locks70);

    // printf("Release lock for accessing DDR.\n");
    XAieTile_LockRelease(&(TileInst[7][0]), /*lockid*/ 1, /*r/w*/ 1, 0); 
    XAieTile_LockRelease(&(TileInst[7][0]), /*lockid*/ 2, /*r/w*/ 1, 0); 

    usleep(1000);
    pc0_times[0] = pc0.diff();
    pc1_times[0] = pc1.diff();
    pc2_times[0] = pc2.diff();
    pc3_times[0] = pc3.diff();
    pc4_times[0] = pc4.diff();
    pc5_times[0] = pc5.diff();
    pc6_times[0] = pc6.diff();
    pc7_times[0] = pc7.diff();
    // usleep(sleep_u);
    
    // ACDC_check("After", mlir_read_buffer_a_ping(0), 384, errors);
    // ACDC_check("After", mlir_read_buffer_a_pong(0), 448, errors);
    // ACDC_check("After", mlir_read_buffer_b_ping(0), 385, errors);
    // ACDC_check("After", mlir_read_buffer_b_pong(0), 449, errors);    

    // Dump contents of ddr_ptr_out
    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_a73_ping(i);
    //         printf("buffer out a ping 73 [%d] = %d\n", i, d);
    //     }
    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_a73_pong(i);
    //         printf("buffer out a pong 73 [%d] = %d\n", i, d);
    //     }

    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_b73_ping(i);
    //         printf("buffer out b ping 73 [%d] = %d\n", i, d);
    //     }
    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_b73_pong(i);
    //         printf("buffer out b pong 73 [%d] = %d\n", i, d);
    //     }

    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_a74_ping(i);
    //         printf("buffer out a ping 74 [%d] = %d\n", i, d);
    //     }
    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_a74_pong(i);
    //         printf("buffer out a pong 74 [%d] = %d\n", i, d);
    //     }

    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_b74_ping(i);
    //         printf("buffer out b ping 74 [%d] = %d\n", i, d);
    //     }
    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_b74_pong(i);
    //         printf("buffer out b pong 74 [%d] = %d\n", i, d);
    //     }

    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_a75_ping(i);
    //         printf("buffer out a ping 75 [%d] = %d\n", i, d);
    //     }
    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_a75_pong(i);
    //         printf("buffer out a pong 75 [%d] = %d\n", i, d);
    //     }

    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_b75_ping(i);
    //         printf("buffer out b ping 75 [%d] = %d\n", i, d);
    //     }
    // for (int i=0; i<16; i++) {
    //         uint32_t d = mlir_read_buffer_b75_ping(i);
    //         printf("buffer out b pong 75 [%d] = %d\n", i, d);
    //     }



    printf("reached1: ");

    for (uint16_t i=0; i<DMA_COUNT; i++) {
        uint16_t d = ddr_ptr_out[i];
        printf("ddr_ptr_out[%d] = %d\n", i, d);
    }

    // for (int i=0; i<512; i++)
    //     ACDC_check("DDR out",ddr_ptr_out[i],i+1, errors);
    // computeStats(pc0_times, n);
    computeStats(pc1_times, n);
    // computeStats(pc2_times, n);
    computeStats(pc3_times, n);
    // computeStats(pc4_times, n);
    computeStats(pc5_times, n);
    computeStats(pc6_times, n);
    // computeStats(pc7_times, n);
    
    printf("reached2: ");

    if (!errors) {
        printf("PASS!\n"); return 0;
    } else {
        printf("Fail!\n"); return -1;
    }

    printf("reached3: ");


}