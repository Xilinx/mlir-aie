//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

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
#include "test_library.h"

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

#include "aie_inc.cpp"

int
main(int argc, char *argv[])
{
  auto col = 7;

  aie_libxaie_ctx_t *_xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(_xaie);

  // Run auto generated config functions

  mlir_aie_configure_cores(_xaie);

  // get locks
  mlir_aie_acquire_lock73(_xaie, 0, 0);
  mlir_aie_acquire_lock71(_xaie, 0, 0);

  mlir_aie_configure_switchboxes(_xaie);
  mlir_aie_initialize_locks(_xaie);
  mlir_aie_configure_dmas(_xaie);

  usleep(10000);

  uint32_t bd_ctrl, bd_pckt;
  bd_ctrl =
      mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 7, 1) + 0x0001D018);
  bd_pckt =
      mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 7, 1) + 0x0001D010);
  printf("BD0_71: pckt: %x, ctrl: %x \n", bd_pckt, bd_ctrl);
  bd_ctrl =
      mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 7, 3) + 0x0001D018);
  bd_pckt =
      mlir_aie_read32(_xaie, mlir_aie_get_tile_addr(_xaie, 7, 3) + 0x0001D010);
  printf("BD0_73: pckt: %x, ctrl: %x \n", bd_pckt, bd_ctrl);

  int count = 256;

  // We're going to stamp over the memory
  for (int i=0; i<count; i++) {
    mlir_aie_write_buffer_buf73(_xaie, i, 73);
    mlir_aie_write_buffer_buf71(_xaie, i, 71);
    mlir_aie_write_buffer_buf62(_xaie, i, 1);
    mlir_aie_write_buffer_buf62(_xaie, i + count, 1);
  }

  usleep(10000);

  mlir_aie_release_lock73(_xaie, 0, 0); // Release lock
  mlir_aie_release_lock71(_xaie, 0, 0); // Release lock

  int errors = 0;
  for (int i=0; i<count; i++) {
    uint32_t d73 = mlir_aie_read_buffer_buf62(_xaie, i);
    uint32_t d71 = mlir_aie_read_buffer_buf62(_xaie, i + count);
    if (d73 != 73)
      errors++;
    if (d71 != 71)
      errors++;
    printf("73[%d]: %x\n", i, d73);
    printf("71[%d]: %x\n", i, d71);
  }

  int res = 0;
  if (!errors) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("fail %d/%d.\n", (count * 2 - errors), count * 2);
    res = -1;
  }
  mlir_aie_deinit_libxaie(_xaie);

  printf("test done.\n");
  return res;
}
