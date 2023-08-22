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

  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);

  int errors = 0;

  // This should timeout and return an error condition.
  if (!mlir_aie_acquire_lock1(_xaie, 1, 0)) {
    // Succeeded!  Which is an error.
    printf("Error: Lock acquired successfully in wrong state!\n");
    errors++;
  }

  mlir_aie_acquire_lock1(_xaie, 0, 0);
  usleep(1000);
  u32 l =
      mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 1, 3) + 0x0001EF00);
  u32 s = (l >> 6) & 0x3;
  printf("Lock acquire 3: 0 is %x\n", s);
  mlir_aie_acquire_lock2(_xaie, 0, 0);
  usleep(1000);
  l = mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 1, 3) + 0x0001EF00);
  s = (l >> 10) & 0x3;
  printf("Lock acquire 5: 0 is %x\n", s);
  mlir_aie_release_lock2(_xaie, 1, 0);
  usleep(1000);
  l = mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 1, 3) + 0x0001EF00);
  s = (l >> 10) & 0x3;
  printf("Lock release 5: 0 is %x\n", s);

  u32 locks =
      mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 1, 3) + 0x0001EF00);
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
      if (((lock == 3) && (acquired != 1)) ||
          ((lock == 5) && (acquired != 0) && (value != 0)))
        errors++;
    }
  }

  mlir_aie_configure_cores(_xaie);
  mlir_aie_start_cores(_xaie);
  usleep(1000);

  locks =
      mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 1, 3) + 0x0001EF00);
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
      if (((lock == 3) && (acquired != 1)) ||
          ((lock == 5) && (acquired != 0) && (value != 0)))
        errors++;
    }
  }

  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    printf("%d Errors\n", errors);
    res = -1;
  }
  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.\n");
  return res;
}
