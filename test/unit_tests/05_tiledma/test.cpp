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

#define MLIR_STACK_OFFSET 4096

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie_inc.cpp"

}

void print_core_status(u32 col, u32 row)
{
    u32 status, coreTimerLow, PC, LR, SP, locks, R0, R4;

    status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x032004);
    coreTimerLow = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0340F8);
    PC = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x00030280);
    LR = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x000302B0);
    SP = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x000302A0);
    locks = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001EF00);
    u32 trace_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x000140D8);


    R0 = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x00030000);
    R4 = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x00030040);
    printf("Core [%d, %d] status is %08X, timer is %u, PC is %d, locks are %08X, LR is %08X, SP is %08X, R0 is %08X,R4 is %08X\n",col, row, status, coreTimerLow, PC, locks, LR, SP, R0, R4);
    printf("Core [%d, %d] trace status is %08X\n",col, row, trace_status);

    //printf("Check locks.\n");
    //locks = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001EF00);
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

void print_dma_status(int col, int row) {


  u32 dma_mm2s_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF10);
  u32 dma_s2mm_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF00);
  u32 dma_mm2s_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DE10);
  u32 dma_s2mm_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DE00);
  u32 dma_bd0_a       = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D000); 
  u32 dma_bd0_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D018);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  printf("DMA [%d, %d] mm2s_status/ctrl is %08X %08X, s2mm_status is %08X %08X, BD0_Addr_A is %08X, BD0_control is %08X\n",col, row, dma_mm2s_status, dma_mm2s_control, dma_s2mm_status, dma_s2mm_control, dma_bd0_a, dma_bd0_control);
  for (int bd=0;bd<8;bd++) {
      u32 dma_bd_addr_a        = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D000 + (0x20*bd));
      u32 dma_bd_control       = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D018 + (0x20*bd));
    if (dma_bd_control & 0x80000000) {
      printf("BD %d valid\n",bd);
      int current_s2mm_ch0 = (dma_s2mm_status >> 16) & 0xf;  
      int current_s2mm_ch1 = (dma_s2mm_status >> 20) & 0xf;  
      int current_mm2s_ch0 = (dma_mm2s_status >> 16) & 0xf;  
      int current_mm2s_ch1 = (dma_mm2s_status >> 20) & 0xf;  

      if (s2mm_ch0_running && bd == current_s2mm_ch0) {
        printf(" * Current BD for s2mm channel 0\n");
      }
      if (s2mm_ch1_running && bd == current_s2mm_ch1) {
        printf(" * Current BD for s2mm channel 1\n");
      }
      if (mm2s_ch0_running && bd == current_mm2s_ch0) {
        printf(" * Current BD for mm2s channel 0\n");
      }
      if (mm2s_ch1_running && bd == current_mm2s_ch1) {
        printf(" * Current BD for mm2s channel 1\n");
      }

      if (dma_bd_control & 0x08000000) {
        u32 dma_packet = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D010 + (0x20*bd));
        printf("   Packet mode: %02X\n",dma_packet & 0x1F);
      }
      int words_to_transfer = 1+(dma_bd_control & 0x1FFF);
      int base_address = dma_bd_addr_a  & 0x1FFF;
//      printf("   Transfering %d 32 bit words to/from %05X\n",words_to_transfer, base_address);
      printf("   Transfering %d 32 bit words to/from %06X\n",words_to_transfer, base_address);

      printf("   ");
//      for (int w=0;w<4; w++) {
      for (int w=0;w<7; w++) {
        printf("%08X ",XAieTile_DmReadWord(&(TileInst[col][row]), (base_address+w) * 4));
      }
      printf("\n");
      if (dma_bd_addr_a & 0x40000) {
        u32 lock_id = (dma_bd_addr_a >> 22) & 0xf;
        printf("   Acquires lock %d ",lock_id);
        if (dma_bd_addr_a & 0x10000) 
          printf("with value %d ",(dma_bd_addr_a >> 17) & 0x1);

        printf("currently ");
        u32 locks = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001EF00);
        u32 two_bits = (locks >> (lock_id*2)) & 0x3;
        if (two_bits) {
          u32 acquired = two_bits & 0x1;
          u32 value = two_bits & 0x2;
          if (acquired)
            printf("Acquired ");
          printf(value?"1":"0");
        }
        else printf("0");
        printf("\n");

      }
      if (dma_bd_control & 0x30000000) { // FIFO MODE
        int FIFO = (dma_bd_control >> 28) & 0x3;
          u32 dma_fifo_counter = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF20);				
        printf("   Using FIFO Cnt%d : %08X\n",FIFO, dma_fifo_counter);
      }
    }

  }

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
    mlir_configure_dmas();
    mlir_initialize_locks();

    printf("Acquire input buffer lock first.\n");
    XAieTile_LockAcquire(&(TileInst[1][3]), 3, 0, 0); // Should this part of setup???
    XAieTile_DmWriteWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+1024+(5*4), 0); // reset output to 0
    XAieTile_DmWriteWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+(5*4), 0);      // reset input to 0
    XAieTile_DmWriteWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+1024+(5*4), 0); // reset output to 0
    XAieTile_DmWriteWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+(3*4), 7); // set input value

    uint32_t d0;
    d0 = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+(3*4));
    printf("Tile[1][3]: data_in[%d] = %d\n",3,d0);
    d0 = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[1][3]: data_out[%d] = %d\n",5,d0);
    d0 = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][3]: data_in[%d] = %d\n",5,d0);
    d0 = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][3]: data_out[%d] = %d\n",5,d0);

    XAieLib_usleep(1000);
    print_core_status(1,3);
    print_dma_status(1,3);
    print_core_status(3,3);
    print_dma_status(3,3);

    printf("Start cores\n");
    mlir_start_cores();

    XAieLib_usleep(1000);
    print_core_status(1,3);
    print_dma_status(1,3);
    print_core_status(3,3);
    print_dma_status(3,3);

//    uint32_t d0;
    d0 = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+(3*4));
    printf("Tile[1][3]: data_in[%d] = %d\n",3,d0);
    d0 = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[1][3]: data_out[%d] = %d\n",5,d0);
    d0 = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][3]: data_in[%d] = %d\n",5,d0);

    uint32_t d1 = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][3]: data_out[%d] = %d\n",5,d1);

    printf("Release input buffer lock.\n");
    XAieTile_LockRelease(&(TileInst[1][3]), 3, 1, 0); 

    d0 = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+(3*4));
    printf("Tile[1][3]: data_in[%d] = %d\n",3,d0);
    d0 = XAieTile_DmReadWord(&(TileInst[1][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[1][3]: data_out[%d] = %d\n",5,d0);
    d0 = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+(5*4));
    printf("Tile[3][3]: data_in[%d] = %d\n",5,d0);

    XAieLib_usleep(1000);
    print_core_status(1,3);
    print_dma_status(1,3);
    print_core_status(3,3);
    print_dma_status(3,3);

    printf("Waiting to acquire output lock for read ...\n");
    while(!XAieTile_LockAcquire(&(TileInst[3][3]), 7, 1, 0)) {} // Should this part of setup???
//    int lock_acq = XAieTile_LockAcquire(&(TileInst[3][3]), 7, 1, 0);
//    printf("lock acquire status: %d\n",lock_acq);
    uint32_t d2 = XAieTile_DmReadWord(&(TileInst[3][3]), MLIR_STACK_OFFSET+1024+(5*4));
    printf("Tile[3][3]: dat_out[%d] = %d\n",5,d2);

    // 7+7+21 = 35
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
