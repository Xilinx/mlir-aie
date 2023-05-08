//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
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

constexpr int numberOfLoops = 4;
constexpr int golden[4] = {7, 13, 43, 47};

#ifdef __AIEARCH__
unsigned short aieArch = __AIEARCH__;
#else
unsigned short aieArch = 10;
#endif

int main(int argc, char *argv[]) {
  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  // Run auto generated config functions
  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);

  mlir_aie_init_mems(_xaie, 1);

  mlir_aie_clear_tile_memory(_xaie, 7, 2);

  printf("Start cores\n");
  mlir_aie_start_cores(_xaie);

  int errors = 0;

  for (int j = 0; j < numberOfLoops; j++) {
    printf("Receiving sub-block: %d\n", j);

    // acquire core lock
    if (mlir_aie_acquire_coreLock(_xaie, 1, 10000) == XAIE_OK)
      printf("Acquired coreLock for read\n");
    else
      printf("ERROR: timed out on acquire coreLock for read\n");

    // check aie L1 content
    mlir_aie_check("After start cores:", mlir_aie_read_buffer_aieL1(_xaie, 0),
                   golden[j], errors);

    // release core lock
    if (mlir_aie_release_coreLock(_xaie, (aieArch == 20) ? -1 : 0, 10000) ==
        XAIE_OK)
      printf("Released coreLock for write\n");
    else
      printf("ERROR: timed out release coreLock for write\n");
  }

  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("FAILED: %d wrong of %d.\n", (errors), numberOfLoops);
    res = -1;
  }

  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.\n");
  return res;
}
