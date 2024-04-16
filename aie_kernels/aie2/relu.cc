//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

void relu(bfloat16 *restrict a, bfloat16 *restrict c, const int TILE_SIZE) {
  const int v_factor = 32;
  v32bfloat16 zeroes = broadcast_zero_bfloat16();

  event0();
  for (size_t i = 0; i < TILE_SIZE; i += v_factor)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32bfloat16 input = *(v32bfloat16 *)(a + i);
      v32bfloat16 output = max(input, zeroes);
      *(v32bfloat16 *)(c + i) = output;
    }
  event1();
  return;
}

extern "C" {

void bf16_relu(bfloat16 *a_in, bfloat16 *c_out) { relu(a_in, c_out, 1024); }

} // extern "C"
