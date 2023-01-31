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
#include "math.h"
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>

extern "C" {
extern aie_libxaie_ctx_t *ctx /* = nullptr*/;
}

/*
 ******************************************************************************
 * LIBXAIENGIENV1
 ******************************************************************************
 */
#ifdef LIBXAIENGINEV1

aie_libxaie_ctx_t *mlir_aie_init_libxaie() {
  aie_libxaie_ctx_t *ctx =
      (aie_libxaie_ctx_t *)malloc(sizeof(aie_libxaie_ctx_t));
  if (!ctx)
    return 0;

  return ctx;
}

void mlir_aie_deinit_libxaie(aie_libxaie_ctx_t *ctx) { free(ctx); }

// Return zero on success
int mlir_aie_init_device(aie_libxaie_ctx_t *ctx) {
  XAIEGBL_HWCFG_SET_CONFIG((&(ctx->AieConfig)), XAIE_NUM_ROWS, XAIE_NUM_COLS,
                           0x800);
  XAieGbl_HwInit(&(ctx->AieConfig));
  ctx->AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&(ctx->AieInst), &(ctx->TileInst[0][0]),
                        ctx->AieConfigPtr);

#if defined(__AIESIM__) && !defined(__CDO__)
  printf("Turning ecc off\n");
  XAie_TurnEccOff(&(ctx->DevInst));
#endif

  return 0;
}

// Return non-zero on success.
int mlir_aie_acquire_lock(aie_libxaie_ctx_t *ctx, int col, int row, int lockid,
                          int lockval, int timeout) {
  return XAieTile_LockAcquire(&(ctx->TileInst[col][row]), lockid, lockval,
                              timeout);
}

// Return non-zero on success.
int mlir_aie_release_lock(aie_libxaie_ctx_t *ctx, int col, int row, int lockid,
                          int lockval, int timeout) {
  return XAieTile_LockRelease(&(ctx->TileInst[col][row]), lockid, lockval,
                              timeout);
}

// Return a 2-bit lock status.
// The low-order bit is 1 if the lock has been acquired.
// The higher-order bit has the lock value.
int mlir_aie_get_lock_status(aie_libxaie_ctx_t *ctx, int col, int row, int lockid) {
  u64 tileAddr = _XAie_GetTileAddr(&(ctx->DevInst), row, col);
  u32 locks;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001EF00, &locks);
  u32 two_bits = (locks >> (lock_id * 2)) & 0x3;
  return two_bits;
}

// Return the read value.
u32 mlir_aie_read32(aie_libxaie_ctx_t *ctx, u64 addr) {
  return XAieGbl_Read32(addr);
}

void mlir_aie_write32(aie_libxaie_ctx_t *ctx, u64 addr, u32 val) {
  XAieGbl_Write32(addr, val);
}

// Return the read value.
u32 mlir_aie_data_mem_rd_word(aie_libxaie_ctx_t *ctx, int col, int row,
                              u64 addr) {
  return XAieTile_DmReadWord(&(ctx->TileInst[col][row]), addr);
}

void mlir_aie_data_mem_wr_word(aie_libxaie_ctx_t *ctx, int col, int row,
                               u64 addr, u32 data) {
  XAieTile_DmWriteWord(&(ctx->TileInst[col][row]), addr, data);
}

// Return the tile address.
u64 mlir_aie_get_tile_addr(aie_libxaie_ctx_t *ctx, int col, int row) {
  struct XAieGbl_Tile *tile = &(ctx->TileInst[col][row]);
  return (u64)(tile->TileAddr);
}

// Print the (non-zero) contents of the data memory of the given core.
void mlir_aie_dump_tile_memory(aie_libxaie_ctx_t *ctx, int col, int row) {
  struct XAieGbl_Tile *tile = &(ctx->TileInst[col][row]);

  for (int i = 0; i < 0x2000; i++) {
    uint32_t d = XAieTile_DmReadWord(tile, (i * 4));
    if (d != 0)
      printf("Tile[%d][%d]: mem[%d] = %d\n", col, row, i, d);
  }
}

// Zero the data memory of the given core.
void mlir_aie_clear_tile_memory(aie_libxaie_ctx_t *ctx, int col, int row) {
  struct XAieGbl_Tile *tile = &(ctx->TileInst[col][row]);

  // FIXME: how do we make this architecture-independent?
  for (int i = 0; i < 0x2000; i++) {
    XAieTile_DmWriteWord(tile, (i * 4), 0);
  }
}

// Print the status of the given Tile DMA.  (row must be greater than zero)
void mlir_aie_print_dma_status(aie_libxaie_ctx_t *ctx, int col, int row) {
  struct XAieGbl_Tile *tile = &(ctx->TileInst[col][row]);

  u32 dma_mm2s_status = XAieGbl_Read32(tile->TileAddr + 0x0001DF10);
  u32 dma_s2mm_status = XAieGbl_Read32(tile->TileAddr + 0x0001DF00);
  u32 dma_mm2s_control = XAieGbl_Read32(tile->TileAddr + 0x0001DE10);
  u32 dma_s2mm_control = XAieGbl_Read32(tile->TileAddr + 0x0001DE00);
  u32 dma_bd0_a = XAieGbl_Read32(tile->TileAddr + 0x0001D000);
  u32 dma_bd0_control = XAieGbl_Read32(tile->TileAddr + 0x0001D018);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  printf("DMA [%d, %d] mm2s_status/ctrl is %08X %08X, s2mm_status is %08X "
         "%08X, BD0_Addr_A is %08X, BD0_control is %08X\n",
         col, row, dma_mm2s_status, dma_mm2s_control, dma_s2mm_status,
         dma_s2mm_control, dma_bd0_a, dma_bd0_control);
  for (int bd = 0; bd < 8; bd++) {
    u32 dma_bd_addr_a =
        XAieGbl_Read32(tile->TileAddr + 0x0001D000 + (0x20 * bd));
    u32 dma_bd_control =
        XAieGbl_Read32(tile->TileAddr + 0x0001D018 + (0x20 * bd));
    if (dma_bd_control & 0x80000000) {
      printf("BD %d valid\n", bd);
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
        u32 dma_packet =
            XAieGbl_Read32(tile->TileAddr + 0x0001D010 + (0x20 * bd));
        printf("   Packet mode: %02X\n", dma_packet & 0x1F);
      }
      int words_to_transfer = 1 + (dma_bd_control & 0x1FFF);
      int base_address = dma_bd_addr_a & 0x1FFF;
      printf("   Transfering %d 32 bit words to/from %06X\n", words_to_transfer,
             base_address);

      printf("   ");
      for (int w = 0; w < 7; w++) {
        printf("%08X ", XAieTile_DmReadWord(tile, (base_address + w) * 4));
      }
      printf("\n");
      if (dma_bd_addr_a & 0x40000) {
        u32 lock_id = (dma_bd_addr_a >> 22) & 0xf;
        printf("   Acquires lock %d ", lock_id);
        if (dma_bd_addr_a & 0x10000)
          printf("with value %d ", (dma_bd_addr_a >> 17) & 0x1);

        printf("currently ");
        u32 locks = XAieGbl_Read32(tile->TileAddr + 0x0001EF00);
        u32 two_bits = (locks >> (lock_id * 2)) & 0x3;
        if (two_bits) {
          u32 acquired = two_bits & 0x1;
          u32 value = two_bits & 0x2;
          if (acquired)
            printf("Acquired ");
          printf(value ? "1" : "0");
        } else
          printf("0");
        printf("\n");
      }
      if (dma_bd_control & 0x30000000) { // FIFO MODE
        int FIFO = (dma_bd_control >> 28) & 0x3;
        u32 dma_fifo_counter = XAieGbl_Read32(tile->TileAddr + 0x0001DF20);
        printf("   Using FIFO Cnt%d : %08X\n", FIFO, dma_fifo_counter);
      }
    }
  }
}

// Print the status of the given Shim DMA.  (row must be zero)
void mlir_aie_print_shimdma_status(aie_libxaie_ctx_t *ctx, int col, int row) {
  struct XAieGbl_Tile *tile = &(ctx->TileInst[col][row]);

  u32 dma_mm2s_status = XAieGbl_Read32(tile->TileAddr + 0x0001D164);
  u32 dma_s2mm_status = XAieGbl_Read32(tile->TileAddr + 0x0001D160);

  u32 dma_mm2s0_control = XAieGbl_Read32(tile->TileAddr + 0x0001D150);
  u32 dma_mm2s1_control = XAieGbl_Read32(tile->TileAddr + 0x0001D158);

  u32 dma_s2mm0_control = XAieGbl_Read32(tile->TileAddr + 0x0001D140);
  u32 dma_s2mm1_control = XAieGbl_Read32(tile->TileAddr + 0x0001D148);

  u32 dma_bd0_a = XAieGbl_Read32(tile->TileAddr + 0x0001D000);
  u32 dma_bd0_control = XAieGbl_Read32(tile->TileAddr + 0x0001D008);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  printf("DMA [%d, %d] mm2s_status/0ctrl/1ctrl is %08X %02X %02X, "
         "s2mm_status/0ctrl/1ctrl is %08X %02X %02X, BD0_Addr_A is %08X, "
         "BD0_control is %08X\n",
         col, row, dma_mm2s_status, dma_mm2s0_control, dma_mm2s1_control,
         dma_s2mm_status, dma_s2mm0_control, dma_s2mm1_control, dma_bd0_a,
         dma_bd0_control);
  for (int bd = 0; bd < 8; bd++) {
    u32 dma_bd_addr_a =
        XAieGbl_Read32(tile->TileAddr + 0x0001D000 + (0x14 * bd));
    u32 dma_bd_buffer_length =
        XAieGbl_Read32(tile->TileAddr + 0x0001D004 + (0x14 * bd));
    u32 dma_bd_control =
        XAieGbl_Read32(tile->TileAddr + 0x0001D008 + (0x14 * bd));
    if (dma_bd_control & 0x1) {
      printf("BD %d valid\n", bd);
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
      //      int words_to_transfer = 1 + (dma_bd_control & 0x1FFF);
      int words_to_transfer = dma_bd_buffer_length;
      //      int base_address = dma_bd_addr_a & 0x1FFF;
      u64 base_address =
          (u64)dma_bd_addr_a + ((u64)((dma_bd_control >> 16) & 0xFFFF) << 32);
      printf("   Transfering %d 32 bit words to/from %06X\n", words_to_transfer,
             (unsigned int)base_address);

      int use_next_bd = ((dma_bd_control >> 15) & 0x1);
      int next_bd = ((dma_bd_control >> 11) & 0xF);
      int lockID = ((dma_bd_control >> 7) & 0xF);
      int enable_lock_release = ((dma_bd_control >> 6) & 0x1);
      int lock_release_val = ((dma_bd_control >> 5) & 0x1);
      int use_release_val = ((dma_bd_control >> 4) & 0x1);
      int enable_lock_acquire = ((dma_bd_control >> 3) & 0x1);
      int lock_acquire_val = ((dma_bd_control >> 2) & 0x1);
      int use_acquire_val = ((dma_bd_control >> 1) & 0x1);

      printf("next_bd: %d, use_next_bd: %d\n", next_bd, use_next_bd);
      printf("lock: %d, acq(en: %d, val: %d, use: %d), rel(en: %d, val: %d, "
             "use: %d)\n",
             lockID, enable_lock_acquire, lock_acquire_val, use_acquire_val,
             enable_lock_release, lock_release_val, use_release_val);

      printf("   ");
    }
  }
}

/// Print the status of a core represented by the given tile, at the given
/// coordinates.
void mlir_aie_print_tile_status(aie_libxaie_ctx_t *ctx, int col, int row) {
  struct XAieGbl_Tile *tile = &(ctx->TileInst[col][row]);
  u32 status, coreTimerLow, PC, LR, SP, locks, R0, R4;

  status = XAieGbl_Read32(tile->TileAddr + 0x032004);
  coreTimerLow = XAieGbl_Read32(tile->TileAddr + 0x0340F8);
  PC = XAieGbl_Read32(tile->TileAddr + 0x00030280);
  LR = XAieGbl_Read32(tile->TileAddr + 0x000302B0);
  SP = XAieGbl_Read32(tile->TileAddr + 0x000302A0);
  locks = XAieGbl_Read32(tile->TileAddr + 0x0001EF00);
  u32 trace_status = XAieGbl_Read32(tile->TileAddr + 0x000140D8);

  R0 = XAieGbl_Read32(tile->TileAddr + 0x00030000);
  R4 = XAieGbl_Read32(tile->TileAddr + 0x00030040);
  printf("Core [%d, %d] status is %08X, timer is %u, PC is %08X, locks are "
         "%08X, LR is %08X, SP is %08X, R0 is %08X,R4 is %08X\n",
         col, row, status, coreTimerLow, PC, locks, LR, SP, R0, R4);
  printf("Core [%d, %d] trace status is %08X\n", col, row, trace_status);

  for (int lock = 0; lock < 16; lock++) {
    u32 two_bits = (locks >> (lock * 2)) & 0x3;
    if (two_bits) {
      printf("Lock %d: ", lock);
      u32 acquired = two_bits & 0x1;
      u32 value = two_bits & 0x2;
      if (acquired)
        printf("Acquired ");
      printf(value ? "1" : "0");
      printf("\n");
    }
  }

  const char *core_status_strings[] = {"Enabled",
                                       "In Reset",
                                       "Memory Stall S",
                                       "Memory Stall W",
                                       "Memory Stall N",
                                       "Memory Stall E",
                                       "Lock Stall S",
                                       "Lock Stall W",
                                       "Lock Stall N",
                                       "Lock Stall E",
                                       "Stream Stall S",
                                       "Stream Stall W",
                                       "Stream Stall N",
                                       "Stream Stall E",
                                       "Cascade Stall Master",
                                       "Cascade Stall Slave",
                                       "Debug Halt",
                                       "ECC Error",
                                       "ECC Scrubbing",
                                       "Error Halt",
                                       "Core Done"};
  printf("Core Status: ");
  for (int i = 0; i <= 20; i++) {
    if ((status >> i) & 0x1)
      printf("%s ", core_status_strings[i]);
  }
  printf("\n");
}

static void clear_range(u64 TileAddr, u64 low, u64 high) {
  for (int i = low; i <= high; i += 4) {
    XAieGbl_Write32(TileAddr + i, 0);
    // int x = XAieGbl_Read32(TileAddr+i);
    // if(x != 0) {
    //   printf("@%x = %x\n", i, x);
    //   XAieGbl_Write32(TileAddr+i, 0);
    // }
  }
}

// Disable and clear the configuration of the given tile (row must be greater
// than zero), The overwrites with zeros the accessible configuration including
// program memory, tile dma configuration, and stream switch configuration. Note
// that this does not completely reset the tile.  In particular, the tile DMA
// may be stalled waiting for a lock and data may remain resident in any queues.
// It also does not change the state of any locks.
void mlir_aie_clear_config(aie_libxaie_ctx_t *ctx, int col, int row) {
  u64 TileAddr = ctx->TileInst[col][row].TileAddr;
  struct XAieGbl_Tile *tile = &(ctx->TileInst[col][row]);

  // Put the core in reset first, otherwise bus collisions
  // result in arm bus errors.
  XAieTile_CoreControl(tile, XAIE_DISABLE, XAIE_ENABLE);

  // Program Memory
  clear_range(TileAddr, 0x20000, 0x200FF);
  // TileDMA
  clear_range(TileAddr, 0x1D000, 0x1D1F8);
  XAieGbl_Write32(TileAddr + 0x1DE00, 0);
  XAieGbl_Write32(TileAddr + 0x1DE08, 0);
  XAieGbl_Write32(TileAddr + 0x1DE10, 0);
  XAieGbl_Write32(TileAddr + 0x1DE08, 0);
  // Stream Switch master config
  clear_range(TileAddr, 0x3F000, 0x3F060);
  // Stream Switch slave config
  clear_range(TileAddr, 0x3F100, 0x3F168);
  // Stream Switch slave slot config
  clear_range(TileAddr, 0x3F200, 0x3F3AC);
}

// Disable and clear the configuration of the given tile (row must be zero),
// The overwrites with zeros the accessible configuration including program
// memory, tile dma configuration, and stream switch configuration.  Note that
// this does not completely reset the tile.  In particular, the tile DMA may be
// stalled waiting for a lock and data may remain resident in any queues. It
// also does not change the state of any locks.
void mlir_aie_clear_shim_config(aie_libxaie_ctx_t *ctx, int col, int row) {
  u64 TileAddr = ctx->TileInst[col][row].TileAddr;

  // ShimDMA
  clear_range(TileAddr, 0x1D000, 0x1D13C);
  XAieGbl_Write32(TileAddr + 0x1D140, 0);
  XAieGbl_Write32(TileAddr + 0x1D148, 0);
  XAieGbl_Write32(TileAddr + 0x1D150, 0);
  XAieGbl_Write32(TileAddr + 0x1D158, 0);

  // Stream Switch master config
  clear_range(TileAddr, 0x3F000, 0x3F058);
  // Stream Switch slave config
  clear_range(TileAddr, 0x3F100, 0x3F15C);
  // Stream Switch slave slot config
  clear_range(TileAddr, 0x3F200, 0x3F37C);
}

void mlir_aie_init_mems(aie_libxaie_ctx_t *ctx, int numBufs) {} // Placeholder

// Allocate a buffer of the given size.  Note that the allocation is not
// guaranteed to be aligned to any boundary.
int *mlir_aie_mem_alloc(aie_libxaie_ctx_t *ctx, int bufIdx, u64 addr,
                        int size) {
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  int *mem_ptr;
  if (fd != -1) {
    mem_ptr =
        (int *)mmap(NULL, 0x8000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, addr);
  }
  return mem_ptr;
}

void mlir_aie_sync_mem_cpu(aie_libxaie_ctx_t *ctx, int bufIdx) {} // Placeholder
void mlir_aie_sync_mem_dev(aie_libxaie_ctx_t *ctx, int bufIdx) {} // Placeholder

/*
 ******************************************************************************
 * LIBXAIENGINEV2
 ******************************************************************************
 */
#else

// namespace aie_device {
//}

aie_libxaie_ctx_t *mlir_aie_init_libxaie() {
  aie_libxaie_ctx_t *ctx =
      (aie_libxaie_ctx_t *)malloc(sizeof(aie_libxaie_ctx_t));
  if (!ctx)
    return 0;

  ctx->AieConfigPtr.AieGen = XAIE_DEV_GEN_AIE;
  ctx->AieConfigPtr.BaseAddr = XAIE_BASE_ADDR;
  ctx->AieConfigPtr.ColShift = XAIE_COL_SHIFT;
  ctx->AieConfigPtr.RowShift = XAIE_ROW_SHIFT;
  ctx->AieConfigPtr.NumRows = XAIE_NUM_ROWS;
  ctx->AieConfigPtr.NumCols = XAIE_NUM_COLS;
  ctx->AieConfigPtr.ShimRowNum = XAIE_SHIM_ROW;
  ctx->AieConfigPtr.MemTileRowStart = XAIE_RES_TILE_ROW_START;
  ctx->AieConfigPtr.MemTileNumRows = XAIE_RES_TILE_NUM_ROWS;
  //  ctx->AieConfigPtr.ReservedRowStart = XAIE_RES_TILE_ROW_START;
  //  ctx->AieConfigPtr.ReservedNumRows  = XAIE_RES_TILE_NUM_ROWS;
  ctx->AieConfigPtr.AieTileRowStart = XAIE_AIE_TILE_ROW_START;
  ctx->AieConfigPtr.AieTileNumRows = XAIE_AIE_TILE_NUM_ROWS;
  ctx->AieConfigPtr.PartProp = {0};
  ctx->DevInst = {0};

  /*
    XAIEGBL_HWCFG_SET_CONFIG((&xaie->AieConfig),
                             XAIE_NUM_ROWS, XAIE_NUM_COLS, 0x800);
    XAieGbl_HwInit(&xaie->AieConfig);
    xaie->AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
    XAieGbl_CfgInitialize(&xaie->AieInst,
                          &xaie->TileInst[0][0], xaie->AieConfigPtr);

    _air_host_active_libxaie1 = xaie;
  */

  return ctx;
}

void mlir_aie_deinit_libxaie(aie_libxaie_ctx_t *ctx) {
  //  if (xaie == _air_host_active_libxaie1)
  //    _air_host_active_libxaie1 = nullptr;
  AieRC RC = XAie_Finish(&(ctx->DevInst));
  if (RC != XAIE_OK) {
    printf("Failed to finish tiles.\n");
  }
  free(ctx);
}

// Return zero on success
int mlir_aie_init_device(aie_libxaie_ctx_t *ctx) {
  AieRC RC = XAIE_OK;

  RC = XAie_CfgInitialize(&(ctx->DevInst), &(ctx->AieConfigPtr));
  if (RC != XAIE_OK) {
    printf("Driver initialization failed.\n");
    return -1;
  }

  RC = XAie_PmRequestTiles(&(ctx->DevInst), NULL, 0);
  if (RC != XAIE_OK) {
    printf("Failed to request tiles.\n");
    return -1;
  }

  // TODO Extra code to really teardown the partitions
  RC = XAie_Finish(&(ctx->DevInst));
  if (RC != XAIE_OK) {
    printf("Failed to finish tiles.\n");
    return -1;
  }
  RC = XAie_CfgInitialize(&(ctx->DevInst), &(ctx->AieConfigPtr));
  if (RC != XAIE_OK) {
    printf("Driver initialization failed.\n");
    return -1;
  }
  RC = XAie_PmRequestTiles(&(ctx->DevInst), NULL, 0);
  if (RC != XAIE_OK) {
    printf("Failed to request tiles.\n");
    return -1;
  }

#if defined(__AIESIM__) && !defined(__CDO__)
  printf("Turning ecc off\n");
  XAie_TurnEccOff(&(ctx->DevInst));
#endif

  return 0;
}

// Return non-zero on success
int mlir_aie_acquire_lock(aie_libxaie_ctx_t *ctx, int col, int row, int lockid,
                          int lockval, int timeout) {
  return (XAie_LockAcquire(&(ctx->DevInst), XAie_TileLoc(col, row),
                           XAie_LockInit(lockid, lockval), timeout) == XAIE_OK);
}

// Return non-zero on success
int mlir_aie_release_lock(aie_libxaie_ctx_t *ctx, int col, int row, int lockid,
                          int lockval, int timeout) {
  return (XAie_LockRelease(&(ctx->DevInst), XAie_TileLoc(col, row),
                           XAie_LockInit(lockid, lockval), timeout) == XAIE_OK);
}

u32 mlir_aie_read32(aie_libxaie_ctx_t *ctx, u64 addr) {
  u32 val;
  XAie_Read32(&(ctx->DevInst), addr, &val);
  return val;
}

void mlir_aie_write32(aie_libxaie_ctx_t *ctx, u64 addr, u32 val) {
  XAie_Write32(&(ctx->DevInst), addr, val);
}

u32 mlir_aie_data_mem_rd_word(aie_libxaie_ctx_t *ctx, int col, int row,
                              u64 addr) {
  u32 data;
  XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(col, row), addr, &data);
  return data;
}

void mlir_aie_data_mem_wr_word(aie_libxaie_ctx_t *ctx, int col, int row,
                               u64 addr, u32 data) {
  XAie_DataMemWrWord(&(ctx->DevInst), XAie_TileLoc(col, row), addr, data);
}

u64 mlir_aie_get_tile_addr(aie_libxaie_ctx_t *ctx, int col, int row) {
  return _XAie_GetTileAddr(&(ctx->DevInst), row, col);
}

void mlir_aie_dump_tile_memory(aie_libxaie_ctx_t *ctx, int col, int row) {
  for (int i = 0; i < 0x2000; i++) {
    uint32_t d;
    AieRC rc = XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(col, row),
                                  (i * 4), &d);
    if (rc == XAIE_OK && d != 0)
      printf("Tile[%d][%d]: mem[%d] = %d\n", col, row, i, d);
  }
}

void mlir_aie_clear_tile_memory(aie_libxaie_ctx_t *ctx, int col, int row) {
  for (int i = 0; i < 0x2000; i++) {
    XAie_DataMemWrWord(&(ctx->DevInst), XAie_TileLoc(col, row), (i * 4), 0);
  }
}

void mlir_aie_print_dma_status(aie_libxaie_ctx_t *ctx, int col, int row) {
  u64 tileAddr = _XAie_GetTileAddr(&(ctx->DevInst), row, col);

  u32 dma_mm2s_status;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001DF10, &dma_mm2s_status);
  u32 dma_s2mm_status;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001DF00, &dma_s2mm_status);
  u32 dma_mm2s0_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001DE10, &dma_mm2s0_control);
  u32 dma_mm2s1_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001DE18, &dma_mm2s1_control);
  u32 dma_s2mm0_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001DE00, &dma_s2mm0_control);
  u32 dma_s2mm1_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001DE08, &dma_s2mm1_control);
  u32 dma_bd0_a;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D000, &dma_bd0_a);
  u32 dma_bd0_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D018, &dma_bd0_control);
  u32 dma_bd1_a;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D020, &dma_bd1_a);
  u32 dma_bd1_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D038, &dma_bd1_control);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  printf("DMA [%d, %d] mm2s_status/0ctrl/1ctrl is %08X %02X %02X, "
         "s2mm_status/0ctrl/1ctrl is %08X %02X %02X, BD0_Addr_A is %08X, "
         "BD0_control is %08X, BD1_Addr_A is %08X, BD1_control is %08X\n",
         col, row, dma_mm2s_status, dma_mm2s0_control, dma_mm2s1_control,
         dma_s2mm_status, dma_s2mm0_control, dma_s2mm1_control, dma_bd0_a,
         dma_bd0_control, dma_bd1_a, dma_bd1_control);
  for (int bd = 0; bd < 8; bd++) {
    u32 dma_bd_addr_a;
    XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D000 + (0x20 * bd),
                &dma_bd_addr_a);
    u32 dma_bd_control;
    XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D018 + (0x20 * bd),
                &dma_bd_control);
    if (dma_bd_control & 0x80000000) {
      printf("BD %d valid\n", bd);
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
        u32 dma_packet;
        XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D010 + (0x20 * bd),
                    &dma_packet);
        printf("   Packet mode: %02X\n", dma_packet & 0x1F);
      }
      int words_to_transfer = 1 + (dma_bd_control & 0x1FFF);
      int base_address = dma_bd_addr_a & 0x1FFF;
      printf("   Transfering %d 32 bit words to/from %06X\n", words_to_transfer,
             base_address);

      printf("   ");
      for (int w = 0; w < 7; w++) {
        u32 tmpd;
        XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(col, row),
                           (base_address + w) * 4, &tmpd);
        printf("%08X ", tmpd);
      }
      printf("\n");
      if (dma_bd_addr_a & 0x40000) {
        u32 lock_id = (dma_bd_addr_a >> 22) & 0xf;
        printf("   Acquires lock %d ", lock_id);
        if (dma_bd_addr_a & 0x10000)
          printf("with value %d ", (dma_bd_addr_a >> 17) & 0x1);

        printf("currently ");
        u32 locks;
        XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001EF00, &locks);
        u32 two_bits = (locks >> (lock_id * 2)) & 0x3;
        if (two_bits) {
          u32 acquired = two_bits & 0x1;
          u32 value = two_bits & 0x2;
          if (acquired)
            printf("Acquired ");
          printf(value ? "1" : "0");
        } else
          printf("0");
        printf("\n");
      }
      if (dma_bd_control & 0x30000000) { // FIFO MODE
        int FIFO = (dma_bd_control >> 28) & 0x3;
        u32 dma_fifo_counter;
        XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001DF20, &dma_fifo_counter);
        printf("   Using FIFO Cnt%d : %08X\n", FIFO, dma_fifo_counter);
      }
      u32 nextBd = ((dma_bd_control >> 13) & 0xF);
      u32 useNextBd = ((dma_bd_control >> 17) & 0x1);
      printf("   Next BD: %d, Use next BD: %d\n", nextBd, useNextBd);
    }
  }
}

void mlir_aie_print_shimdma_status(aie_libxaie_ctx_t *ctx, int col, int row) {
  // int col = loc.Col;
  // int row = loc.Row;
  u64 tileAddr = _XAie_GetTileAddr(&(ctx->DevInst), row, col);

  u32 dma_mm2s_status;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D164, &dma_mm2s_status);
  u32 dma_s2mm_status;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D160, &dma_s2mm_status);

  u32 dma_mm2s0_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D150, &dma_mm2s0_control);
  u32 dma_mm2s1_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D158, &dma_mm2s1_control);

  u32 dma_s2mm0_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D140, &dma_s2mm0_control);
  u32 dma_s2mm1_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D148, &dma_s2mm1_control);

  u32 dma_bd0_a;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D000, &dma_bd0_a);
  u32 dma_bd0_control;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D008, &dma_bd0_control);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  printf("DMA [%d, %d] mm2s_status/0ctrl/1ctrl is %08X %02X %02X, "
         "s2mm_status/0ctrl/1ctrl is %08X %02X %02X, BD0_Addr_A is %08X, "
         "BD0_control is %08X\n",
         col, row, dma_mm2s_status, dma_mm2s0_control, dma_mm2s1_control,
         dma_s2mm_status, dma_s2mm0_control, dma_s2mm1_control, dma_bd0_a,
         dma_bd0_control);
  for (int bd = 0; bd < 8; bd++) {
    u32 dma_bd_addr_a;
    XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D000 + (0x14 * bd),
                &dma_bd_addr_a);
    u32 dma_bd_buffer_length;
    XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D004 + (0x14 * bd),
                &dma_bd_buffer_length);
    u32 dma_bd_control;
    XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D008 + (0x14 * bd),
                &dma_bd_control);
    if (dma_bd_control & 0x1) {
      printf("BD %d valid\n", bd);
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
      /*
            if (dma_bd_control & 0x08000000) {
              u32 dma_packet;
              XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001D010 + (0x14 * bd),
                          &dma_packet);
              printf("   Packet mode: %02X\n", dma_packet & 0x1F);
            }
      */
      //      int words_to_transfer = 1 + (dma_bd_control & 0x1FFF);
      int words_to_transfer = dma_bd_buffer_length;
      //      int base_address = dma_bd_addr_a & 0x1FFF;
      u64 base_address =
          (u64)dma_bd_addr_a + ((u64)((dma_bd_control >> 16) & 0xFFFF) << 32);
      printf("   Transfering %d 32 bit words to/from %06X\n", words_to_transfer,
             (unsigned int)base_address);

      int use_next_bd = ((dma_bd_control >> 15) & 0x1);
      int next_bd = ((dma_bd_control >> 11) & 0xF);
      int lockID = ((dma_bd_control >> 7) & 0xF);
      int enable_lock_release = ((dma_bd_control >> 6) & 0x1);
      int lock_release_val = ((dma_bd_control >> 5) & 0x1);
      int use_release_val = ((dma_bd_control >> 4) & 0x1);
      int enable_lock_acquire = ((dma_bd_control >> 3) & 0x1);
      int lock_acquire_val = ((dma_bd_control >> 2) & 0x1);
      int use_acquire_val = ((dma_bd_control >> 1) & 0x1);

      printf("next_bd: %d, use_next_bd: %d\n", next_bd, use_next_bd);
      printf("lock: %d, acq(en: %d, val: %d, use: %d), rel(en: %d, val: %d, "
             "use: %d)\n",
             lockID, enable_lock_acquire, lock_acquire_val, use_acquire_val,
             enable_lock_release, lock_release_val, use_release_val);

      printf("   ");
      /*
            for (int w = 0; w < 7; w++) {
              u32 tmpd;
              XAie_DataMemRdWord(&(ctx->DevInst), XAie_TileLoc(col, row),
                                 (base_address + w) * 4, &tmpd);
              printf("%08X ", tmpd);
            }
            printf("\n");
            if (dma_bd_addr_a & 0x40000) {
              u32 lock_id = (dma_bd_addr_a >> 22) & 0xf;
              printf("   Acquires lock %d ", lock_id);
              if (dma_bd_addr_a & 0x10000)
                printf("with value %d ", (dma_bd_addr_a >> 17) & 0x1);

              printf("currently ");
              u32 locks;
              XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001EF00, &locks);
              u32 two_bits = (locks >> (lock_id * 2)) & 0x3;
              if (two_bits) {
                u32 acquired = two_bits & 0x1;
                u32 value = two_bits & 0x2;
                if (acquired)
                  printf("Acquired ");
                printf(value ? "1" : "0");
              } else
                printf("0");
              printf("\n");
            }
            if (dma_bd_control & 0x30000000) { // FIFO MODE
              int FIFO = (dma_bd_control >> 28) & 0x3;
              u32 dma_fifo_counter;
              XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001DF20,
         &dma_fifo_counter); printf("   Using FIFO Cnt%d : %08X\n", FIFO,
         dma_fifo_counter);
            }
      */
    }
  }
}

/// Print the status of a core represented by the given tile, at the given
/// coordinates.
void mlir_aie_print_tile_status(aie_libxaie_ctx_t *ctx, int col, int row) {
  // int col = loc.Col;
  // int row = loc.Row;
  u64 tileAddr = _XAie_GetTileAddr(&(ctx->DevInst), row, col);
  u32 status, coreTimerLow, PC, LR, SP, locks, R0, R4;

  XAie_Read32(&(ctx->DevInst), tileAddr + 0x032004, &status);
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0340F8, &coreTimerLow);
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x00030280, &PC);
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x000302B0, &LR);
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x000302A0, &SP);
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x0001EF00, &locks);
  u32 trace_status;
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x000140D8, &trace_status);

  XAie_Read32(&(ctx->DevInst), tileAddr + 0x00030000, &R0);
  XAie_Read32(&(ctx->DevInst), tileAddr + 0x00030040, &R4);
  printf("Core [%d, %d] status is %08X, timer is %u, PC is %08X, locks are "
         "%08X, LR is %08X, SP is %08X, R0 is %08X,R4 is %08X\n",
         col, row, status, coreTimerLow, PC, locks, LR, SP, R0, R4);
  printf("Core [%d, %d] trace status is %08X\n", col, row, trace_status);

  for (int lock = 0; lock < 16; lock++) {
    u32 two_bits = (locks >> (lock * 2)) & 0x3;
    if (two_bits) {
      printf("Lock %d: ", lock);
      u32 acquired = two_bits & 0x1;
      u32 value = two_bits & 0x2;
      if (acquired)
        printf("Acquired ");
      printf(value ? "1" : "0");
      printf("\n");
    }
  }

  const char *core_status_strings[] = {"Enabled",
                                       "In Reset",
                                       "Memory Stall S",
                                       "Memory Stall W",
                                       "Memory Stall N",
                                       "Memory Stall E",
                                       "Lock Stall S",
                                       "Lock Stall W",
                                       "Lock Stall N",
                                       "Lock Stall E",
                                       "Stream Stall S",
                                       "Stream Stall W",
                                       "Stream Stall N",
                                       "Stream Stall E",
                                       "Cascade Stall Master",
                                       "Cascade Stall Slave",
                                       "Debug Halt",
                                       "ECC Error",
                                       "ECC Scrubbing",
                                       "Error Halt",
                                       "Core Done"};
  printf("Core Status: ");
  for (int i = 0; i <= 20; i++) {
    if ((status >> i) & 0x1)
      printf("%s ", core_status_strings[i]);
  }
  printf("\n");
}

static void clear_range(XAie_DevInst *devInst, u64 tileAddr, u64 low,
                        u64 high) {
  for (int i = low; i <= high; i += 4) {
    XAie_Write32(devInst, tileAddr + i, 0);
    // int x = XAie_Read32(ctx->DevInst,tileAddr+i);
    // if(x != 0) {
    //   printf("@%x = %x\n", i, x);
    //   XAie_Write32(ctx->DevInst,tileAddr+i, 0);
    // }
  }
}
void mlir_aie_clear_config(aie_libxaie_ctx_t *ctx, int col, int row) {
  // int col = loc.Col;
  // int row = loc.Row;
  u64 tileAddr = _XAie_GetTileAddr(&(ctx->DevInst), row, col);

  // Put the core in reset first, otherwise bus collisions
  // result in arm bus errors.
  // TODO Check if this works
  XAie_CoreDisable(&(ctx->DevInst), XAie_TileLoc(col, row));

  // Program Memory
  clear_range(&(ctx->DevInst), tileAddr, 0x20000, 0x200FF);
  // TileDMA
  clear_range(&(ctx->DevInst), tileAddr, 0x1D000, 0x1D1F8);
  XAie_Write32(&(ctx->DevInst), tileAddr + 0x1DE00, 0);
  XAie_Write32(&(ctx->DevInst), tileAddr + 0x1DE08, 0);
  XAie_Write32(&(ctx->DevInst), tileAddr + 0x1DE10, 0);
  XAie_Write32(&(ctx->DevInst), tileAddr + 0x1DE08, 0);
  // Stream Switch master config
  clear_range(&(ctx->DevInst), tileAddr, 0x3F000, 0x3F060);
  // Stream Switch slave config
  clear_range(&(ctx->DevInst), tileAddr, 0x3F100, 0x3F168);
  // Stream Switch slave slot config
  clear_range(&(ctx->DevInst), tileAddr, 0x3F200, 0x3F3AC);

  // TODO Check if this works
  XAie_CoreEnable(&(ctx->DevInst), XAie_TileLoc(col, row));
}

void mlir_aie_clear_shim_config(aie_libxaie_ctx_t *ctx, int col, int row) {
  // int col = loc.Col;
  // int row = loc.Row;
  u64 tileAddr = _XAie_GetTileAddr(&(ctx->DevInst), row, col);

  // ShimDMA
  clear_range(&(ctx->DevInst), tileAddr, 0x1D000, 0x1D13C);
  XAie_Write32(&(ctx->DevInst), tileAddr + 0x1D140, 0);
  XAie_Write32(&(ctx->DevInst), tileAddr + 0x1D148, 0);
  XAie_Write32(&(ctx->DevInst), tileAddr + 0x1D150, 0);
  XAie_Write32(&(ctx->DevInst), tileAddr + 0x1D158, 0);

  // Stream Switch master config
  clear_range(&(ctx->DevInst), tileAddr, 0x3F000, 0x3F058);
  // Stream Switch slave config
  clear_range(&(ctx->DevInst), tileAddr, 0x3F100, 0x3F15C);
  // Stream Switch slave slot config
  clear_range(&(ctx->DevInst), tileAddr, 0x3F200, 0x3F37C);
}

void mlir_aie_init_mems(aie_libxaie_ctx_t *ctx, int numBufs) {
#if defined(__AIESIM__)
  ctx->buffers =
      (ext_mem_model_t **)malloc(numBufs * sizeof(ext_mem_model_t *));
#else
  ctx->buffers = (XAie_MemInst **)malloc(numBufs * sizeof(XAie_MemInst *));
#endif
}

int *mlir_aie_mem_alloc(aie_libxaie_ctx_t *ctx, int bufIdx, int size) {
#if defined(__AIESIM__)
  ctx->buffers[bufIdx] = new ext_mem_model_t;
  (ctx->buffers[bufIdx])->virtualAddr = std::malloc(size * sizeof(int));
  if ((ctx->buffers[bufIdx])->virtualAddr) {
    (ctx->buffers[bufIdx])->size = size;
    // assign physical space in SystemC DDR memory controller
    (ctx->buffers[bufIdx])->physicalAddr = nextAlignedAddr;
    // adjust nextAlignedAddr to the next 128-bit aligned address
    nextAlignedAddr = nextAlignedAddr + size;
    uint64_t gapToAligned = nextAlignedAddr % 16; // 16byte (128bit)
    if (gapToAligned > 0)
      nextAlignedAddr += (16 - gapToAligned);
  } else {
    printf("ExtMemModel: Failed to allocate %d memory.\n", size);
  }

  std::cout << "ExtMemModel constructor: virutal address " << std::hex
            << (ctx->buffers[bufIdx])->virtualAddr << ", physical address "
            << (ctx->buffers[bufIdx])->physicalAddr << ", size " << std::dec
            << (ctx->buffers[bufIdx])->size << std::endl;

  return (ctx->buffers[bufIdx])->virtualAddr;
#else
  //  ctx->InBuffers = (XAie_MemInst**)malloc(sizeof(XAie_MemInst*));
  //  XAie_MemInst *IN;
  ctx->buffers[bufIdx] =
      XAie_MemAllocate(&(ctx->DevInst), size * sizeof(int), XAIE_MEM_CACHEABLE);
  int *mem_ptr = (int *)XAie_MemGetVAddr(ctx->buffers[bufIdx]);
  XAie_MemSyncForCPU(ctx->buffers[bufIdx]);
  return mem_ptr;
#endif
}

void mlir_aie_sync_mem_cpu(aie_libxaie_ctx_t *ctx, int bufIdx) {
#if defined(__AIESIM__)
  aiesim_ReadGM((ctx->buffers[bufIdx])->physicalAddr,
                (ctx->buffers[bufIdx])->virtualAddr,
                (ctx->buffers[bufIdx])->size);
#else
  XAie_MemSyncForCPU(ctx->buffers[bufIdx]);
#endif
}

void mlir_aie_sync_mem_dev(aie_libxaie_ctx_t *ctx, int bufIdx) {
#if defined(__AIESIM__)
  aiesim_WriteGM((ctx->buffers[bufIdx])->physicalAddr,
                 (ctx->buffers[bufIdx])->virtualAddr,
                 (ctx->buffers[bufIdx])->size);
#else
  XAie_MemSyncForDev(ctx->buffers[bufIdx]);
#endif
}

#endif

/*
 ******************************************************************************
 * COMMON
 ******************************************************************************
 */

void computeStats(u32 performance_counter[], int n) {
  u32 total_0 = 0;

  for (int i = 0; i < n; i++) {
    total_0 += performance_counter[i];
  }

  float mean_0 = (float)total_0 / n;

  float sdev_0 = 0;

  for (int i = 0; i < n; i++) {
    float x = (float)performance_counter[i] - mean_0;
    sdev_0 += x * x;
  }

  sdev_0 = sqrtf(sdev_0 / n);

  printf("Mean and Standard Devation: %f, %f \n", mean_0, sdev_0);
}
