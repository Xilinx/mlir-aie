//===- test_library.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// This file contains common libraries used for testing.

#include "test_library.h"
#include <stdio.h>

void ACDC_dump_tile_memory(struct XAieGbl_Tile &tile) {
    int col = tile.ColId;
    int row = tile.RowId;
    for (int i=0; i<0x2000; i++) {
        uint32_t d = XAieTile_DmReadWord(&(tile), (i*4));
        if(d != 0)
            printf("Tile[%d][%d]: mem[%d] = %d\n", col, row, i, d);
    }
}

void ACDC_clear_tile_memory(struct XAieGbl_Tile &tile) {
    int col = tile.ColId;
    int row = tile.RowId;
    for (int i=0; i<0x2000; i++) {
        XAieTile_DmWriteWord(&(tile), (i*4), 0);
    }
}

void ACDC_print_dma_status(struct XAieGbl_Tile &tile) {
  int col = tile.ColId;
  int row = tile.RowId;
    
  u32 dma_mm2s_status = XAieGbl_Read32(tile.TileAddr + 0x0001DF10);
  u32 dma_s2mm_status = XAieGbl_Read32(tile.TileAddr + 0x0001DF00);
  u32 dma_mm2s_control = XAieGbl_Read32(tile.TileAddr + 0x0001DE10);
  u32 dma_s2mm_control = XAieGbl_Read32(tile.TileAddr + 0x0001DE00);
  u32 dma_bd0_a       = XAieGbl_Read32(tile.TileAddr + 0x0001D000); 
  u32 dma_bd0_control = XAieGbl_Read32(tile.TileAddr + 0x0001D018);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  printf("DMA [%d, %d] mm2s_status/ctrl is %08X %08X, s2mm_status is %08X %08X, BD0_Addr_A is %08X, BD0_control is %08X\n",col, row, dma_mm2s_status, dma_mm2s_control, dma_s2mm_status, dma_s2mm_control, dma_bd0_a, dma_bd0_control);
  for (int bd=0;bd<8;bd++) {
      u32 dma_bd_addr_a        = XAieGbl_Read32(tile.TileAddr + 0x0001D000 + (0x20*bd));
      u32 dma_bd_control       = XAieGbl_Read32(tile.TileAddr + 0x0001D018 + (0x20*bd));
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
        u32 dma_packet = XAieGbl_Read32(tile.TileAddr + 0x0001D010 + (0x20*bd));
        printf("   Packet mode: %02X\n",dma_packet & 0x1F);
      }
      int words_to_transfer = 1+(dma_bd_control & 0x1FFF);
      int base_address = dma_bd_addr_a  & 0x1FFF;
      printf("   Transfering %d 32 bit words to/from %06X\n",words_to_transfer, base_address);

      printf("   ");
      for (int w=0;w<7; w++) {
        printf("%08X ",XAieTile_DmReadWord(&(tile), (base_address+w) * 4));
      }
      printf("\n");
      if (dma_bd_addr_a & 0x40000) {
        u32 lock_id = (dma_bd_addr_a >> 22) & 0xf;
        printf("   Acquires lock %d ",lock_id);
        if (dma_bd_addr_a & 0x10000) 
          printf("with value %d ",(dma_bd_addr_a >> 17) & 0x1);

        printf("currently ");
        u32 locks = XAieGbl_Read32(tile.TileAddr + 0x0001EF00);
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
          u32 dma_fifo_counter = XAieGbl_Read32(tile.TileAddr + 0x0001DF20);				
        printf("   Using FIFO Cnt%d : %08X\n",FIFO, dma_fifo_counter);
      }
    }
  }
}


/// Print the status of a core represented by the given tile, at the given coordinates.
void ACDC_print_tile_status(struct XAieGbl_Tile &tile) {
    int col = tile.ColId;
    int row = tile.RowId;
    u32 status, coreTimerLow, PC, LR, SP, locks, R0, R4;

    status = XAieGbl_Read32(tile.TileAddr + 0x032004);
    coreTimerLow = XAieGbl_Read32(tile.TileAddr + 0x0340F8);
    PC = XAieGbl_Read32(tile.TileAddr + 0x00030280);
    LR = XAieGbl_Read32(tile.TileAddr + 0x000302B0);
    SP = XAieGbl_Read32(tile.TileAddr + 0x000302A0);
    locks = XAieGbl_Read32(tile.TileAddr + 0x0001EF00);
    u32 trace_status = XAieGbl_Read32(tile.TileAddr + 0x000140D8);

    R0 = XAieGbl_Read32(tile.TileAddr + 0x00030000);
    R4 = XAieGbl_Read32(tile.TileAddr + 0x00030040);
    printf("Core [%d, %d] status is %08X, timer is %u, PC is %08X, locks are %08X, LR is %08X, SP is %08X, R0 is %08X,R4 is %08X\n",col, row, status, coreTimerLow, PC, locks, LR, SP, R0, R4);
    printf("Core [%d, %d] trace status is %08X\n",col, row, trace_status);

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

    const char * core_status_strings[] = {"Enabled", "In Reset",
      "Memory Stall S", "Memory Stall W", "Memory Stall N", "Memory Stall E",
      "Lock Stall S", "Lock Stall W", "Lock Stall N", "Lock Stall E",
      "Stream Stall S", "Stream Stall W", "Stream Stall N", "Stream Stall E",
      "Cascade Stall Master", "Cascade Stall Slave", "Debug Halt", "ECC Error",
      "ECC Scrubbing", "Error Halt", "Core Done"};
    printf("Core Status: ");
    for(int i = 0; i <= 20; i++) {
      if((status >> i) & 0x1)
        printf("%s ", core_status_strings[i]);
    }
    printf("\n");
}

static void clear_range(u64 TileAddr, u64 low, u64 high) {
  for (int i=low; i<=high; i+=4) {
    XAieGbl_Write32(TileAddr+i, 0);
    // int x = XAieGbl_Read32(TileAddr+i);
    // if(x != 0) {
    //   printf("@%x = %x\n", i, x);
    //   XAieGbl_Write32(TileAddr+i, 0);
    // }
  }
}
void ACDC_clear_config(struct XAieGbl_Tile &tile) {
    int col = tile.ColId;
    int row = tile.RowId;
  	u64 TileAddr = tile.TileAddr;

    // Put the core in reset first, otherwise bus collisions
    // result in arm bus errors.
    XAieTile_CoreControl(&tile, XAIE_DISABLE, XAIE_ENABLE);

    // Program Memory
    clear_range(TileAddr, 0x20000, 0x200FF);
    // TileDMA
    clear_range(TileAddr, 0x1D000, 0x1D1F8);
    XAieGbl_Write32(TileAddr+0x1DE00, 0);
    XAieGbl_Write32(TileAddr+0x1DE08, 0);
    XAieGbl_Write32(TileAddr+0x1DE10, 0);
    XAieGbl_Write32(TileAddr+0x1DE08, 0);
    // Stream Switch master config
    clear_range(TileAddr, 0x3F000, 0x3F060);
    // Stream Switch slave config
    clear_range(TileAddr, 0x3F100, 0x3F168);
    // Stream Switch slave slot config
    clear_range(TileAddr, 0x3F200, 0x3F3AC);
}

void ACDC_clear_shim_config(struct XAieGbl_Tile &tile) {
    int col = tile.ColId;
    int row = tile.RowId;
	u64 TileAddr = tile.TileAddr;

    // ShimDMA
    clear_range(TileAddr, 0x1D000, 0x1D13C);
    XAieGbl_Write32(TileAddr+0x1D140, 0);
    XAieGbl_Write32(TileAddr+0x1D148, 0);
    XAieGbl_Write32(TileAddr+0x1D150, 0);
    XAieGbl_Write32(TileAddr+0x1D158, 0);

    // Stream Switch master config
    clear_range(TileAddr, 0x3F000, 0x3F058);
    // Stream Switch slave config
    clear_range(TileAddr, 0x3F100, 0x3F15C);
    // Stream Switch slave slot config
    clear_range(TileAddr, 0x3F200, 0x3F37C);
}