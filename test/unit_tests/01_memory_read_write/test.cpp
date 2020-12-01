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

int
main(int argc, char *argv[])
{
  printf("test start.\n");
  auto col = 1;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  // reset cores and locks
//  mlir_configure_cores();
/*
  for (int i = 1; i <= XAIE_NUM_ROWS; i++) {
    for (int j = 0; j < XAIE_NUM_COLS; j++) {
      XAieTile_CoreControl(&(TileInst[j][i]), XAIE_DISABLE, XAIE_ENABLE);
      for (int l=0; l<16; l++)
        XAieTile_LockRelease(&(TileInst[j][i]), l, 0x0, 0);
    }
  }
*/
  // cores
  //
  //  mlir_initialize_cores();
//  XAieTile_CoreControl(&(TileInst[col][1]), XAIE_DISABLE, XAIE_DISABLE);
  XAieTile_CoreControl(&(TileInst[col][2]), XAIE_DISABLE, XAIE_DISABLE);

  XAieTile_ShimColumnReset(&(TileInst[col][0]), XAIE_RESETENABLE);
  XAieTile_ShimColumnReset(&(TileInst[col][0]), XAIE_RESETDISABLE);

  // configure switchboxes 
  mlir_configure_switchboxes();
//  XAieTile_ShimStrmMuxConfig(&(TileInst[col][0]), XAIETILE_SHIM_STRM_MUX_SOUTH7, XAIETILE_SHIM_STRM_MUX_DMA);

  // locks
  mlir_initialize_locks();

  // dmas
  mlir_configure_dmas();
/*
  XAieDma_Shim ShimDmaInst1;
  uint32_t *bram_ptr;

  #define BRAM_ADDR 0x020100000000LL
  #define DMA_COUNT 256

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, BRAM_ADDR);
    for (int i=0; i<DMA_COUNT; i++) {
      bram_ptr[i] = i+1;
      //printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);
    }
  }

  auto burstlen = 4;
  XAieDma_ShimInitialize(&(TileInst[col][0]), &ShimDmaInst1);
  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 1, HIGH_ADDR((u64)BRAM_ADDR), LOW_ADDR((u64)BRAM_ADDR), sizeof(u32) * DMA_COUNT);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 1 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 1);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S1, 1);

  auto ret = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S1);
  if (ret)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, ret);

  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S1, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S1)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }
*/
XAieGbl_LoadElf(&(TileInst[1][2]),(u8*)("aie.elf"),XAIE_ENABLE);

//  mlir_start_cores();
XAieTile_CoreControl(&(TileInst[1][2]), XAIE_ENABLE, XAIE_DISABLE);

  int errors = 0;
//  for (int i=0; i<DMA_COUNT; i++) {
  for (int i=0; i<32; i++) {
    uint32_t d = XAieTile_DmReadWord(&(TileInst[col][2]), 16384+(i*4));
    if (d != (i+1)) {
      errors++;
      printf("mismatch at %d: %d != %d\n", i, d, i);
    }
  }
/*
  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %d/%d.\n", (DMA_COUNT-errors), DMA_COUNT);
  }
*/
  printf("test done.\n");
}
