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

int main(int argc, char *argv[]) {
  printf("test start.\n");

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  mlir_aie_clear_tile_memory(_xaie, 1, 3);

  mlir_aie_configure_cores(_xaie);
  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);

  int errors = 0;

  mlir_aie_write_buffer_b(_xaie, 0, 11);
  mlir_aie_check("Before memory writes: ", mlir_aie_read_buffer_b(_xaie, 0), 11,
                 errors);
  mlir_aie_check("Before memory writes: ", mlir_aie_read_buffer_a(_xaie, 0), 0,
                 errors);

  mlir_aie_start_cores(_xaie);

  mlir_aie_check("After memory writes: ", mlir_aie_read_buffer_a(_xaie, 0), 11,
                 errors);

  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }

  mlir_aie_deinit_libxaie(_xaie);
  printf("test done.\n");

  return res;
}
