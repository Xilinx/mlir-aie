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
  auto col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  // Run auto generated config functions

  mlir_configure_cores();

  // get locks
  XAieTile_LockAcquire(&(TileInst[7][3]), 0, 0, 0);
  XAieTile_LockAcquire(&(TileInst[7][1]), 0, 0, 0);

  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();

  usleep(10000);

  uint32_t bd_ctrl, bd_pckt;
  bd_ctrl = XAieTile_DmReadWord(&(TileInst[7][1]), 0x0001D018); 
  bd_pckt = XAieTile_DmReadWord(&(TileInst[7][1]), 0x0001D010); 
  printf("BD0_71: pckt: %x, ctrl: %x \n", bd_pckt, bd_ctrl);
  bd_ctrl = XAieTile_DmReadWord(&(TileInst[7][3]), 0x0001D018); 
  bd_pckt = XAieTile_DmReadWord(&(TileInst[7][3]), 0x0001D010); 
  printf("BD0_73: pckt: %x, ctrl: %x \n", bd_pckt, bd_ctrl);

  int count = 256;

  // We're going to stamp over the memory
  for (int i=0; i<count; i++) {
      mlir_write_buffer_buf73(i, 73);
      mlir_write_buffer_buf71(i, 71);
      mlir_write_buffer_buf62(i, 1);
      mlir_write_buffer_buf62(i+count, 1);
  }

  usleep(10000);

  XAieTile_LockRelease(&(TileInst[7][3]), 0, 0, 0); // Release lock
  XAieTile_LockRelease(&(TileInst[7][1]), 0, 0, 0); // Release lock

  int errors = 0;
  for (int i=0; i<count; i++) {
    uint32_t d73 = mlir_read_buffer_buf62(i);
    uint32_t d71 = mlir_read_buffer_buf62(i+count);
    printf("73[%d]: %x\n", i, d73);
    printf("71[%d]: %x\n", i, d71);
  }

  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %d/%d.\n", (count-errors), count);
  }

}
