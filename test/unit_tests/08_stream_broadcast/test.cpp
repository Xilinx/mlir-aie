// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

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
    ACDC_clear_tile_memory(TileInst[3][2]);
    ACDC_clear_tile_memory(TileInst[3][3]);
    ACDC_clear_tile_memory(TileInst[3][4]);

    mlir_configure_cores();
    mlir_configure_switchboxes();
    mlir_configure_dmas();
    mlir_initialize_locks();

    printf("Acquire input buffer lock first.\n");
    XAieTile_LockAcquire(&(TileInst[1][3]), 3, 0, 0); // Should this part of setup???
    XAieTile_DmWriteWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+(3*4), 0); // reset output to 0
    XAieTile_DmWriteWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+1024+(5*4), 0); // reset output to 0
    XAieTile_DmWriteWord(&(TileInst[3][2]), MLIR_STACK_OFFSET+(5*4), 0); // reset output to 0
    XAieTile_DmWriteWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+(5*4), 0); // reset output to 0
    XAieTile_DmWriteWord(&(TileInst[3][4]), MLIR_STACK_OFFSET+(5*4), 0); // reset output to 0
    XAieTile_DmWriteWord(&(TileInst[3][2]), MLIR_STACK_OFFSET+1024+(5*4), 0); // reset output to 0
    XAieTile_DmWriteWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+1024+(5*4), 0); // reset output to 0
    XAieTile_DmWriteWord(&(TileInst[3][4]), MLIR_STACK_OFFSET+1024+(5*4), 0); // reset output to 0

    XAieTile_DmWriteWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+(3*4), 7); // set input value

    uint32_t tmp;
    ACDC_dump_tile_memory(TileInst[1][3]);
    ACDC_dump_tile_memory(TileInst[3][2]);
    ACDC_dump_tile_memory(TileInst[3][3]);
    ACDC_dump_tile_memory(TileInst[3][4]);

    tmp = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+(3*4));
    printf("Tile[1][3]: a[%d] = %d\n",3,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[1][3]: b[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][2]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][2]: a[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][3]: a[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][4]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][4]: a[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][2]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][2]: b[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][3]: b[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][4]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][4]: b[%d] = %d\n",5,tmp);

    XAieLib_usleep(1000);
    ACDC_print_tile_status(TileInst[1][3]);
    ACDC_print_dma_status(TileInst[1][3]);
    ACDC_print_tile_status(TileInst[3][2]);
    ACDC_print_dma_status(TileInst[3][2]);
    ACDC_print_tile_status(TileInst[3][3]);
    ACDC_print_dma_status(TileInst[3][3]);
    ACDC_print_tile_status(TileInst[3][4]);
    ACDC_print_dma_status(TileInst[3][4]);

    printf("Start cores\n");
    mlir_start_cores();

    ACDC_dump_tile_memory(TileInst[1][3]);
    ACDC_dump_tile_memory(TileInst[3][2]);
    ACDC_dump_tile_memory(TileInst[3][3]);
    ACDC_dump_tile_memory(TileInst[3][4]);
    tmp = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+(3*4));
    printf("Tile[1][3]: a[%d] = %d\n",3,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[1][3]: b[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][2]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][2]: a[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][3]: a[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][4]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][4]: a[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][2]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][2]: b[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][3]: b[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][4]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][4]: b[%d] = %d\n",5,tmp);

    ACDC_print_tile_status(TileInst[1][3]);
    ACDC_print_dma_status(TileInst[1][3]);
    ACDC_print_tile_status(TileInst[3][2]);
    ACDC_print_dma_status(TileInst[3][2]);
    ACDC_print_tile_status(TileInst[3][3]);
    ACDC_print_dma_status(TileInst[3][3]);
    ACDC_print_tile_status(TileInst[3][4]);
    ACDC_print_dma_status(TileInst[3][4]);

    XAieLib_usleep(1000);
//    ACDC_print_tile_status(TileInst[1][3]);

    ACDC_dump_tile_memory(TileInst[1][3]);
    ACDC_dump_tile_memory(TileInst[3][2]);
    ACDC_dump_tile_memory(TileInst[3][3]);
    ACDC_dump_tile_memory(TileInst[3][4]);
    
    uint32_t d1_32 = XAieTile_DmReadWord(&(TileInst[3][2]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][2]: b[%d] = %d\n",5,d1_32);
    uint32_t d1_33 = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][3]: b[%d] = %d\n",5,d1_33);
    uint32_t d1_34 = XAieTile_DmReadWord(&(TileInst[3][4]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][4]: b[%d] = %d\n",5,d1_34);

    printf("Release input buffer lock.\n");
    XAieTile_LockRelease(&(TileInst[1][3]), 3, 1, 0);

    ACDC_dump_tile_memory(TileInst[1][3]);
    ACDC_dump_tile_memory(TileInst[3][2]);
    ACDC_dump_tile_memory(TileInst[3][3]);
    ACDC_dump_tile_memory(TileInst[3][4]);

    tmp = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+(3*4));
    printf("Tile[1][3]: a[%d] = %d\n",3,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[1][3]: b[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][2]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][2]: a[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][3]: a[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][4]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][4]: a[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][2]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][2]: b[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][3]: b[%d] = %d\n",5,tmp);
    tmp = XAieTile_DmReadWord(&(TileInst[3][4]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][4]: b[%d] = %d\n",5,tmp);

    XAieLib_usleep(1000);
    ACDC_print_tile_status(TileInst[1][3]);
    ACDC_print_dma_status(TileInst[1][3]);
    ACDC_print_tile_status(TileInst[3][2]);
    ACDC_print_dma_status(TileInst[3][2]);
    ACDC_print_tile_status(TileInst[3][3]);
    ACDC_print_dma_status(TileInst[3][3]);
    ACDC_print_tile_status(TileInst[3][4]);
    ACDC_print_dma_status(TileInst[3][4]);

    XAieTile_LockAcquire(&(TileInst[3][3]), 7, 0, 0); // Should this part of setup???

    ACDC_dump_tile_memory(TileInst[1][3]);
    ACDC_dump_tile_memory(TileInst[3][2]);
    ACDC_dump_tile_memory(TileInst[3][3]);
    ACDC_dump_tile_memory(TileInst[3][4]);
    
    uint32_t d2_32 = XAieTile_DmReadWord(&(TileInst[3][2]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][2]: b[%d] = %d\n",5,d2_32);
    uint32_t d2_33 = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][3]: b[%d] = %d\n",5,d2_33);
    uint32_t d2_34 = XAieTile_DmReadWord(&(TileInst[3][4]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][4]: b[%d] = %d\n",5,d2_34);

    // 7+7+21 = 35
    int errors = 0;
    //if(d1 == 35 || d2 != 35) errors++;
    if(d1_32 != 0 || d1_33 != 0 || d1_34 != 0 || d2_32 != 105 || d2_33 != 140 || d2_34 != 175) errors++;

    if (!errors) {
        printf("PASS!\n");
    } else {
        printf("Fail!\n");
    }
    printf("test done.\n");
}
