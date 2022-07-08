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

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)

#define MLIR_STACK_OFFSET 4096

#define MAP_SIZE 16UL
#define MAP_MASK (MAP_SIZE - 1)

#include "aie_inc.cpp"

int main(int argc, char *argv[]) {

  int n = 1;
  u32 pc0_times[n];
  u32 pc1_times[n];

  printf("07_Lock_Acquire test start.\n");
  printf("Running %d times ...\n", n);

  for (int iters = 0; iters < n; iters++) {

    aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();

    mlir_aie_clear_tile_memory(_xaie, 1, 3);

    mlir_aie_init_device(_xaie);
    mlir_aie_configure_cores(_xaie);
    mlir_aie_configure_switchboxes(_xaie);
    mlir_aie_initialize_locks(_xaie);
    mlir_aie_configure_dmas(_xaie);


    EventMonitor pc0(_xaie, 1, 3, 0, XAIE_EVENT_ACTIVE_CORE,
                 XAIE_EVENT_DISABLED_CORE, XAIE_EVENT_NONE_CORE,
                 XAIE_CORE_MOD);

    pc0.set();

    mlir_aie_start_cores(_xaie);
    usleep(100);
    pc0_times[iters] = pc0.diff();

    int errors = 0;

    mlir_aie_deinit_libxaie(_xaie);
  }
  computeStats(pc0_times, n);
}