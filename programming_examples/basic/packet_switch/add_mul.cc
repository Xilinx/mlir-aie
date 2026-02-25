//===- add_mul.cc -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <typename T, int W = 128>
void mul_aie(T *restrict in0, T *restrict out) {
  event0();
  const int vec_factor = 16;

  aie::vector<T, vec_factor> In0;
  aie::accum<acc32, vec_factor> Out;

  const int F = W / vec_factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(6, ) {
      In0 = aie::load_v<vec_factor>(in0);
      Out = aie::mul(In0, (T)2);
      aie::store_v(out, Out.template to_vector<T>());
      in0 += vec_factor;
      out += vec_factor;
    }
  event1();
}

template <typename T, int W = 128>
void add_aie(T *restrict in0, T *restrict out) {
  event0();
  const int vec_factor = 16;

  aie::vector<T, vec_factor> In0;
  aie::vector<T, vec_factor> Out;

  const int F = W / vec_factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(6, ) {
      In0 = aie::load_v<vec_factor>(in0);
      Out = aie::add(In0, (T)2);
      aie::store_v(out, Out);
      in0 += vec_factor;
      out += vec_factor;
    }
  event1();
}

extern "C" {

#ifndef DIM_S
#define DIM_S 256
#endif

void add(int8 *restrict In0, int8 *restrict y) { add_aie<int8, DIM_S>(In0, y); }

void mul(int8 *restrict In0, int8 *restrict y) { mul_aie<int8, DIM_S>(In0, y); }
} // extern "C"
