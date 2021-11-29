//===- test_library.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <xaiengine.h>

#define mlir_aie_check(s, r, v, errors)                                        \
  if (r != v) {                                                                \
    printf("ERROR %s: %s expected %d, but was %d!\n", s, #r, v, r);            \
    errors++;                                                                  \
  }
#define mlir_aie_check_float(s, r, v, errors)                                  \
  if (r != v) {                                                                \
    printf("ERROR %s: %s expected %f, but was %f!\n", s, #r, v, r);            \
    errors++;                                                                  \
  }

/*
 ******************************************************************************
 * LIBXAIENGIENV2
 ******************************************************************************
 */
#ifdef LIBXAIENGINEV2

#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_NUM_ROWS 9
#define XAIE_NUM_COLS 50
#define XAIE_COL_SHIFT 23
#define XAIE_ROW_SHIFT 18
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 0
#define XAIE_RES_TILE_NUM_ROWS 0
#define XAIE_AIE_TILE_ROW_START 1
#define XAIE_AIE_TILE_NUM_ROWS 8

// XAie_SetupConfig(ConfigPtr, XAIE_DEV_GEN_AIE, XAIE_BASE_ADDR,
//        XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
//        XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
//        XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS,
//        XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);
/*
XAie_Config ConfigPtr = {XAIE_DEV_GEN_AIE, XAIE_BASE_ADDR,
        XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
        XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_SHIM_ROW,
        XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS,
        XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS, {0}, };
//XAie_InstDeclare(DevInst, &ConfigPtr);
XAie_DevInst DevInst = { 0 };
*/

struct aie_libxaie_ctx_t {
  XAie_Config AieConfigPtr;
  XAie_DevInst DevInst;
  /*
    XAieGbl_Config *AieConfigPtr;
    XAieGbl AieInst;
    XAieGbl_HwCfg AieConfig;
    XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];
    XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];
  */
};

/*
 ******************************************************************************
 * LIBXAIENGIENV1
 ******************************************************************************
 */
#else

#define XAIE_NUM_ROWS 8
#define XAIE_NUM_COLS 50
#define XAIE_ADDR_ARRAY_OFF 0x800

//#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
//#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

//#define MLIR_STACK_OFFSET 4096

struct aie_libxaie_ctx_t {
  XAieGbl_Config *AieConfigPtr;
  XAieGbl AieInst;
  XAieGbl_HwCfg AieConfig;
  XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS + 1];
  XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS + 1];
};
#endif

aie_libxaie_ctx_t *mlir_aie_init_libxaie();
void mlir_aie_deinit_libxaie(aie_libxaie_ctx_t *);

int mlir_aie_init_device(aie_libxaie_ctx_t *ctx);

int mlir_aie_acquire_lock(aie_libxaie_ctx_t *ctx, int col, int row, int lockid,
                          int lockval, int timeout);
int mlir_aie_release_lock(aie_libxaie_ctx_t *ctx, int col, int row, int lockid,
                          int lockval, int timeout);
u32 mlir_aie_read32(aie_libxaie_ctx_t *ctx, u64 addr);
void mlir_aie_write32(aie_libxaie_ctx_t *ctx, u64 addr, u32 val);
u32 mlir_aie_data_mem_rd_word(aie_libxaie_ctx_t *ctx, int col, int row,
                              u64 addr);
void mlir_aie_data_mem_wr_word(aie_libxaie_ctx_t *ctx, int col, int row,
                               u64 addr, u32 data);

u64 mlir_aie_get_tile_addr(aie_libxaie_ctx_t *ctx, int col, int row);

/// Dump the contents of the memory associated with the given tile.
void mlir_aie_dump_tile_memory(aie_libxaie_ctx_t *ctx, int col, int row);

/// Clear the contents of the memory associated with the given tile.
void mlir_aie_clear_tile_memory(aie_libxaie_ctx_t *ctx, int col, int row);

/// Print the status of a dma represented by the given tile.
void mlir_aie_print_dma_status(aie_libxaie_ctx_t *ctx, int col, int row);

/// Print the status of a core represented by the given tile.
void mlir_aie_print_tile_status(aie_libxaie_ctx_t *ctx, int col, int row);

/// Zero out the program and configuration memory of the tile.
void mlir_aie_clear_config(aie_libxaie_ctx_t *ctx, int col, int row);

/// Zero out the configuration memory of the shim tile.
void mlir_aie_clear_shim_config(aie_libxaie_ctx_t *ctx, int col, int row);
