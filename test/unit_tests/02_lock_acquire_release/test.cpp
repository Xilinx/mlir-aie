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

    mlir_configure_cores();
    mlir_configure_switchboxes();
    mlir_initialize_locks();
    mlir_configure_dmas();

    //XAieLib_usleep(1000);

    int errors = 0;

    // XAieTile_LockRelease(&(TileInst[j][i]), l, val, timeout)
    // XAIeTile_LockAcquire(&(TileInst[j][i]), l, val, timeout)
    //
    XAieTile_LockAcquire(&(TileInst[1][3]), 3, 0, 0);
    XAieLib_usleep(1000);
    u32 l = XAieGbl_Read32(TileInst[1][3].TileAddr + 0x0001EF00);
    u32 s = (l >> 6) & 0x3;
    printf("Lock acquire 3: 0 is %x\n",s);
    XAieTile_LockAcquire(&(TileInst[1][3]), 5, 0, 0);
    XAieLib_usleep(1000);
    l = XAieGbl_Read32(TileInst[1][3].TileAddr + 0x0001EF00);
    s = (l >> 10) & 0x3;
    printf("Lock acquire 5: 0 is %x\n",s);
    XAieTile_LockRelease(&(TileInst[1][3]), 5, 1, 0);
    XAieLib_usleep(1000);
    l = XAieGbl_Read32(TileInst[1][3].TileAddr + 0x0001EF00);
    s = (l >> 10) & 0x3;
    printf("Lock release 5: 0 is %x\n",s);

    u32 locks = XAieGbl_Read32(TileInst[1][3].TileAddr + 0x0001EF00);
    for (int lock=0;lock<16;lock++) {
        u32 two_bits = (locks >> (lock*2)) & 0x3;
        if (two_bits) {
            printf("Lock %d: ", lock);
            u32 acquired = two_bits & 0x1;
            u32 value = two_bits & 0x2;
            if (acquired)
                printf("Acquired ");
            printf(value?"1":"0");
            printf("\n");
            if(((lock == 3) && (acquired != 1)) || ((lock == 5) && (acquired != 0) && (value != 0)))
                errors++;
        }
    }

    mlir_configure_cores();
    mlir_start_cores();
    XAieLib_usleep(1000);

    locks = XAieGbl_Read32(TileInst[1][3].TileAddr + 0x0001EF00);
    for (int lock=0;lock<16;lock++) {
        u32 two_bits = (locks >> (lock*2)) & 0x3;
        if (two_bits) {
            printf("Lock %d: ", lock);
            u32 acquired = two_bits & 0x1;
            u32 value = two_bits & 0x2;
            if (acquired)
                printf("Acquired ");
            printf(value?"1":"0");
            printf("\n");
            if(((lock == 3) && (acquired != 1)) || ((lock == 5) && (acquired != 0) && (value != 0)))
                errors++;
        }
    }

    if (!errors) {
        printf("PASS!\n");
    } else {
        printf("Fail!\n");
    }
    printf("test done.\n");
}
