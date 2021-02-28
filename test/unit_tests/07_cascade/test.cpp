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
    ACDC_clear_tile_memory(TileInst[2][3]);

    mlir_configure_cores();
    mlir_configure_switchboxes();
    mlir_configure_dmas();
    mlir_initialize_locks();

    printf("Acquire input buffer lock first.\n");
    XAieTile_LockAcquire(&(TileInst[1][3]), 3, 0, 0); // Should this part of setup???
    XAieTile_DmWriteWord(&(TileInst[2][3]), MLIR_STACK_OFFSET+(5*4), 0); // reset output to 0
    XAieTile_DmWriteWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+(3*4), 7); // set input value

//    XAieLib_usleep(1000);
//    ACDC_print_tile_status(TileInst[2][3]);

    printf("Start cores\n");
    mlir_start_cores();

//    XAieLib_usleep(1000);
//    ACDC_print_tile_status(TileInst[2][3]);

    uint32_t d1 = XAieTile_DmReadWord(&(TileInst[2][3]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[2][3]: data[%d] = %d\n",7,d1);

    printf("Release input buffer lock.\n");
    XAieTile_LockRelease(&(TileInst[1][3]), 3, 1, 0);

//    XAieLib_usleep(1000);
//    ACDC_print_tile_status(TileInst[2][3]);

    printf("Waiting to acquire output lock for read ...\n");
    while(!XAieTile_LockAcquire(&(TileInst[2][3]), 7, 1, 0)) {} // Should this part of setup???
    uint32_t d2 = XAieTile_DmReadWord(&(TileInst[2][3]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[2][3]: data[%d] = %d\n",7,d2);

    // 7*5 = 35, 35*5 = 175
    int errors = 0;
    //if(d1 == 35 || d2 != 35) errors++;
    if(d1 != 0 || d2 != 175) errors++;

    if (!errors) {
        printf("PASS!\n");
    } else {
        printf("Fail!\n");
    }
    printf("test done.\n");
}
