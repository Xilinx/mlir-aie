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

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie_inc.cpp"

}

void printCoreStatus(int col, int row) {

	
	u32 status, coreTimerLow, locks;
	status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x032004);
	coreTimerLow = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0340F8);
	locks = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001EF00);
	printf("Core [%d, %d] status is %08X, timer is %u, locks are %08X\n",col, row, status, coreTimerLow, locks);
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
		}
	}
}

int
main(int argc, char *argv[])
{
  auto col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  printCoreStatus(col, 2);

  // Run auto generated config functions

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();

  uint32_t *bram_ptr;

  #define BRAM_ADDR (0x4000+0x020100000000LL)
  #define DMA_COUNT 512

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
    for (int i=0; i<DMA_COUNT; i++) {
      bram_ptr[i] = i+1;
      //printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);
    }
  }

  // We're going to stamp over the memory
  for (int i=0; i<DMA_COUNT; i++) {
    XAieTile_DmWriteWord(&(TileInst[col][2]), 0x1000+i*4, 0xdeadbeef);
  }

// Release lock for reading from DDR
  XAieTile_LockRelease(&(TileInst[7][0]), 1, 1, 0); 

  XAieLib_usleep(1000);

  printCoreStatus(col, 2);
  
  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = XAieTile_DmReadWord(&(TileInst[col][2]), 0x1000+i*4);
    if (d != (i+1)) {
      errors++;
      printf("mismatch %x != 1 + %x\n", d, i);
    }
  }

  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %d/%d.\n", (DMA_COUNT-errors), DMA_COUNT);
  }

}
