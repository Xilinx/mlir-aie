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

template <typename T_in, typename T_sf, typename T_out, const int N>
void expand(T_in *in, T_out *out) {

  constexpr int block_size = 32;
  constexpr int number_of_blocks = 8;
  // Super block size = block_size x number_of_blocks

  T_in *__restrict pI = in; // Input pointer
  T_in *__restrict pSFb = in + N/2 ; // The scale factors are after the inputs
  T_sf *__restrict pSF = (T_sf *)pSFb; // But we only advance by the number of bytes not elements
  T_out *__restrict pO = out;
  const int F = N / (block_size * number_of_blocks); // number of super blocks
  event0();
  event0();
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(F, ) { // 16 -> F
      aie::vector<T_sf, number_of_blocks> sfV = aie::load_v<number_of_blocks>(pSF); // Load number_of_blocks scale factors
      pSF += number_of_blocks; // Advance by the number of bytes
      event0();
      for (int k = 0; k < number_of_blocks; k++){
          aie::vector<T_in, block_size> I0 = aie::load_v<block_size>(pI); // Load one block of input
          pI += block_size/2; // Advance by the number of bytes

          int16 sf = sfV[k]; // sfV[k % number_of_blocks]

          aie::vector<int16, block_size> sf_broadcast = aie::broadcast(sf);

          // Upsize these to 8 bits -> 16
          aie::vector<int8, block_size> asInt8 = aie::unpack(I0); // Unpack the 4 bit values to 8 bits
          aie::vector<int16, block_size> asInt16 = aie::unpack(asInt8); // Unpack the 8 bit values to 16 bits
          aie::vector<int16, block_size> scaled_int16 = aie::mul(asInt16, sf_broadcast).to_vector<int16>(0); // Scale the int16 values
          aie::store_v(pO, scaled_int16); // Write the scaled int16 values to output
          pO += block_size; // Advance by the number of bytes
        }
      event1();
    }
  event1();
  event1();
}

extern "C" {

void expand_int4_to_int16(int4 *a_in, int16 *c_out) {
  expand<int4, int16, int16, 1024>(a_in, c_out);
}

} // extern "C"