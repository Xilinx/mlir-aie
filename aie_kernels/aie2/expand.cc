//===- expand.cc -------------------------------------------------*- C++
//-*-===//
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

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

template <typename T_in, typename T_sf, typename T_out, const int N>
void expand(T_in *in, T_out *out) {

// Keep vector width constant; group size can vary as a multiple of 32
#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif
  static_assert((GROUP_SIZE % 32) == 0, "GROUP_SIZE must be a multiple of 32");
  constexpr int block_size = 32;
  constexpr int blocks_per_group = GROUP_SIZE / block_size;
  constexpr int groups_per_tile = N / GROUP_SIZE;
  // Super block size = block_size x number_of_blocks

  T_in *__restrict pI = in;           // Input pointer
  T_in *__restrict pSFb = in + N / 2; // The scale factors are after the inputs
  T_sf *__restrict pSF =
      (T_sf *)pSFb; // But we only advance by the number of bytes not elements
  T_out *__restrict pO = out;
  const int F = groups_per_tile; // iterate over groups of size GROUP_SIZE
  event0();
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(F, ) { // 16 -> F
      // Load one scale per group (scalar load)
      T_sf sf = *pSF;
      pSF += 1;
      event0();
      for (int k = 0; k < blocks_per_group; k++) {
        aie::vector<T_in, block_size> I0 =
            aie::load_v<block_size>(pI); // Load one block of input (32 int4s)
        pI += block_size / 2;            // Advance by the number of bytes

        bfloat16 sf_bf16 = sf;

        aie::vector<bfloat16, block_size> sf_broadcast =
            aie::broadcast(sf_bf16);

        // Upsize these to 8 bits -> 16 -> bfloat16
        aie::vector<uint8, block_size> asInt8 =
            aie::unpack(I0); // Unpack the 4 bit values to 8 bits
        aie::vector<uint16, block_size> asInt16 =
            aie::unpack(asInt8); // Unpack the 8 bit values to 16 bits
        aie::vector<bfloat16, block_size> as_bf16 =
            aie::to_float<bfloat16>(asInt16, 0); // Convert to bfloat16
        aie::vector<bfloat16, block_size> scaled_bf16 =
            aie::mul(as_bf16, sf_broadcast); // Scale the bfloat16 values
        aie::store_v(pO,
                     scaled_bf16); // Write the scaled bfloat16 values to output
        pO += block_size;          // Advance by the number of bytes
      }
      event1();
    }
  event1();
}

extern "C" {

void expand_int4_to_bfloat16(uint4 *a_in, bfloat16 *c_out) {
  expand<uint4, bfloat16, bfloat16, 1024>(a_in, c_out);
}

} // extern "C"
