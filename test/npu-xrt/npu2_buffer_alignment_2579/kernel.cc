//===- kernel.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Mirrors the kernel from issue #2579: 32-element int32 add-1 implemented
// with explicit aie::load_v / aie::store_v on a 16-element (512-bit) vector.
// On AIE2P (NPU2) this load width requires 64-byte buffer alignment; the
// regression test uses stack_size=1028 to push the first compute-tile buffer
// off a 64B boundary unless the buffer allocator pads correctly.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#define TILE_WIDTH 32
#define VECT_FACTOR 16

extern "C" {

void alignment_test_kernel(int32_t *bufin, int32_t *bufout) {
  auto to_add = aie::broadcast<int32_t, VECT_FACTOR>(1);
  for (int i = 0; i < TILE_WIDTH; i += VECT_FACTOR) {
    aie::vector<int32_t, VECT_FACTOR> vin = aie::load_v<VECT_FACTOR>(bufin + i);
    auto vout = aie::add(vin, to_add);
    aie::store_v(bufout + i, vout);
  }
}

}  // extern "C"
