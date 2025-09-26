//===- test_library.h -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
#ifndef AIE_TEST_LIBRARY_H
#define AIE_TEST_LIBRARY_H

#include "target.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef HSA_RUNTIME
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa_ext_air.h"
#endif

#ifdef HSA_RUNTIME
template <typename T>
inline void mlir_aie_write_pkt(hsa_queue_t *q, uint32_t packet_id, T *pkt) {
  reinterpret_cast<T *>(q->base_address)[packet_id] = *pkt;
}
#endif

extern "C" {

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

// class for using events and PF cpounters
class EventMonitor {
public:
  EventMonitor(aie_libxaie_ctx_t *_xaie, u32 _col, u32 _row, u32 _pfc,
               XAie_Events _startE, XAie_Events _endE, XAie_Events _resetE,
               XAie_ModuleType _module) {
    //  EventMonitor(struct XAieGbl_Tile *_tilePtr, u32 _pfc, u32 _startE, u32
    //  _endE,
    //           u32 _resetE, XAie_ModuleType _module) {
    // tilePtr = _tilePtr;
    devInst = _xaie->XAieDevInst;
    row = _row;
    col = _col;
    pfc = _pfc;
    mode = _module;
    XAie_PerfCounterControlSet(devInst, XAie_TileLoc(col, row), mode, pfc,
                               _startE, _endE);

    // mode = _mode; // 0: Core, 1: PL, 2, Mem
    // if (mode == MODE_CORE) {
    //   XAieTileCore_PerfCounterControl(tilePtr, pfc, _startE, _endE, _resetE);
    // } else if (mode == MODE_PL) {
    //   XAieTilePl_PerfCounterControl(tilePtr, pfc, _startE, _endE, _resetE);
    // } else {
    //   XAieTileMem_PerfCounterControl(tilePtr, pfc, _startE, _endE, _resetE);
    // }
  }
  void set() {
    // XAie_PerfCounterSet(devInst, XAie_TileLoc(col,row), mode, pfc, val);
    XAie_PerfCounterGet(devInst, XAie_TileLoc(col, row), mode, pfc, &start);
    // if (mode == MODE_CORE) {
    //   start = XAieTileCore_PerfCounterGet(tilePtr, pfc);
    // } else if (mode == MODE_PL) {
    //   start = XAieTilePl_PerfCounterGet(tilePtr, pfc);
    // } else {
    //   start = XAieTileMem_PerfCounterGet(tilePtr, pfc);
    // }
  }
  u32 read() {
    u32 val;
    XAie_PerfCounterGet(devInst, XAie_TileLoc(col, row), mode, pfc, &val);
    return val;
    // if (mode == MODE_CORE) {
    //   return XAieTileCore_PerfCounterGet(tilePtr, pfc);
    // } else if (mode == MODE_PL) {
    //   return XAieTilePl_PerfCounterGet(tilePtr, pfc);
    // } else {
    //   return XAieTileMem_PerfCounterGet(tilePtr, pfc);
    // }
  }
  u32 diff() {
    u32 end;
    XAie_PerfCounterGet(devInst, XAie_TileLoc(col, row), mode, pfc, &end);
    // if (mode == MODE_CORE) {
    //   end = XAieTileCore_PerfCounterGet(tilePtr, pfc);
    // } else if (mode == MODE_PL) {
    //   end = XAieTilePl_PerfCounterGet(tilePtr, pfc);
    // } else {
    //   end = XAieTileMem_PerfCounterGet(tilePtr, pfc);
    // }
    if (end < start) {
      printf("WARNING: EventMonitor: performance counter wrapped!\n");
      return 0; // TODO: fix this
    } else {
      return end - start;
    }
  }

private:
  u32 start;
  u32 pfc;
  XAie_ModuleType mode;
  u8 col, row;
  XAie_DevInst *devInst;
};

/*
 ******************************************************************************
 * Common functions
 ******************************************************************************
 */

// This is a more elegant solution
#ifdef HSA_RUNTIME
hsa_status_t mlir_aie_packet_req_translation(hsa_agent_dispatch_packet_t *pkt,
                                             uint64_t va);

hsa_status_t mlir_aie_packet_nd_memcpy(
    hsa_agent_dispatch_packet_t *pkt, uint16_t herd_id, uint8_t col,
    uint8_t direction, uint8_t channel, uint8_t burst_len, uint8_t memory_space,
    uint64_t phys_addr, uint32_t transfer_length1d, uint32_t transfer_length2d,
    uint32_t transfer_stride2d, uint32_t transfer_length3d,
    uint32_t transfer_stride3d, uint32_t transfer_length4d,
    uint32_t transfer_stride4d);

hsa_status_t mlir_aie_queue_dispatch_and_wait(
    hsa_agent_t *agent, hsa_queue_t *q, uint64_t packet_id, uint64_t doorbell,
    hsa_agent_dispatch_packet_t *pkt, bool destroy_signal = true);

#endif

/// @brief  Initialize libXAIE and allocate a new context object.
/// @return A pointer to the context
aie_libxaie_ctx_t *mlir_aie_init_libxaie();
void mlir_aie_deinit_libxaie(aie_libxaie_ctx_t *);

int mlir_aie_init_device(aie_libxaie_ctx_t *ctx, uint32_t device_id = 0);

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

/// Print the status of a memtiledma represented by the given location.
void mlir_aie_print_memtiledma_status(aie_libxaie_ctx_t *ctx, int col, int row);

/// Print the status of a shimdma represented by the given location (row = 0).
void mlir_aie_print_shimdma_status(aie_libxaie_ctx_t *ctx, int col, int row);

/// Print the status of a core represented by the given tile.
void mlir_aie_print_tile_status(aie_libxaie_ctx_t *ctx, int col, int row);

/// Zero out the program and configuration memory of the tile.
void mlir_aie_clear_config(aie_libxaie_ctx_t *ctx, int col, int row);

/// Zero out the configuration memory of the shim tile.
void mlir_aie_clear_shim_config(aie_libxaie_ctx_t *ctx, int col, int row);

void computeStats(u32 performance_counter[], int n);

} // extern "C"

#endif
