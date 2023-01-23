//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "test_library.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <xaiengine.h>

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {
  int n = 1;
  u32 pc0_times[n];
  u32 pc1_times[n];
  u32 pc2_times[n];
  u32 pc3_times[n];

  printf("09_Shim_Broadcast_Horizontal test start.\n");
  printf("Running %d times ...\n", n);

  int total_errors = 0;

  auto col = 7;

  for (int iters = 0; iters < n; iters++) {

    aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
    mlir_aie_init_device(_xaie);
    mlir_aie_configure_cores(_xaie);
    mlir_aie_configure_switchboxes(_xaie);
    mlir_aie_initialize_locks(_xaie);

    mlir_aie_acquire_lock1(_xaie, 0, 0);
    mlir_aie_acquire_lock2(_xaie, 0, 0);

    mlir_aie_configure_dmas(_xaie);

    XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(6,0), 
                        XAIE_PL_MOD, 2,
                        XAIE_EVENT_DMA_S2MM_ERROR_PL); // Start

    XAie_EventBroadcast(&(_xaie->DevInst), XAie_TileLoc(7,0), 
                        XAIE_PL_MOD, 3,
                        XAIE_EVENT_LOCK_0_RELEASED_PL); // Stop

    // Track time between two broadcast events in destination tiles (7,3) and
    // (7,4)
    EventMonitor pc0(_xaie, 5, 0, 0, XAIE_EVENT_BROADCAST_A_2_PL,
                 XAIE_EVENT_BROADCAST_A_3_PL, XAIE_EVENT_NONE_PL,
                 XAIE_PL_MOD);
    pc0.set();

    EventMonitor pc1(_xaie, 7, 0, 1, XAIE_EVENT_BROADCAST_A_2_PL,
                 XAIE_EVENT_LOCK_0_RELEASED_PL, XAIE_EVENT_NONE_PL,
                 XAIE_PL_MOD);
    pc1.set();

    // Start Test by generating events in Source Tile
    XAie_EventGenerate(&(_xaie->DevInst), XAie_TileLoc(6,0),
                       XAIE_PL_MOD, XAIE_EVENT_DMA_S2MM_0_ERROR_PL);
    XAie_EventGenerate(&(_xaie->DevInst), XAie_TileLoc(7,0),
                       XAIE_PL_MOD, XAIE_EVENT_LOCK_0_RELEASED_PL);

    usleep(200);

    mlir_aie_print_tile_status(_xaie, 7, 3);

    pc0_times[iters] = pc0.diff();
    pc1_times[iters] = pc1.diff();

    mlir_aie_deinit_libxaie(_xaie);
  }

  computeStats(pc0_times, n);
  computeStats(pc1_times, n);
}