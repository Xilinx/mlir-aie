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

template <typename T_in, typename T_b, typename T_sb, typename T_out, const int N>
void expand(T_in *in, T_out *out) {

  constexpr int block_size = 32;
  constexpr int number_of_blocks = 16;
  const int n_blocks = N/block_size;
  // Super block size = block_size x number_of_blocks

  T_in *__restrict pI = in; // Input pointer

  T_in *__restrict pBlock = in + N/2 ; // The scale factors are after the inputs
  T_b *__restrict pSF_b = (T_b *)pBlock; // But we only advance by the number of bytes not elements
  T_b *__restrict pMIN_b = pSF_b + n_blocks; // But we only advance by the number of bytes not elements
  
  T_in *__restrict pSBlock = in + N/2 + 2*n_blocks; // The scale factors are after the inputs
  T_sb *__restrict pSF_sb = (T_sb *)pSBlock; // But we only advance by the number of bytes not elements
  
  T_out *__restrict pO = out;

  const int F = N / (block_size * number_of_blocks); // N/1024 = number of super blocks
  event0();
  event0();
  //aie::vector<T_sb, 4> sfSB = aie::load_v<4>(pSF_sb); // Load 1 bf16 scale factors
  T_sb sf_sb = *pSF_sb++;
  T_sb min_sb = *pSF_sb++;
  aie::vector<T_sb, block_size> min_broadcast_sb = aie::broadcast(sf_sb);

  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(F, ) { 
      aie::vector<T_b, number_of_blocks> sfV = aie::load_v<number_of_blocks>(pSF_b); // Load number_of_blocks scale factors
      pSF_b += number_of_blocks; // Advance by the number of bytes
      aie::vector<T_b, number_of_blocks> minV = aie::load_v<number_of_blocks>(pMIN_b); // Load number_of_blocks mins
      pMIN_b += number_of_blocks; // Advance by the number of bytes
      event0();
      for (int k = 0; k < number_of_blocks; k++)
      {
          aie::vector<T_in, block_size> I0 = aie::load_v<block_size>(pI); // Load one block of input 
          pI += block_size/2; // Advance by the number of bytes

          T_b min_b = minV[k];
          aie::vector<T_b, 2*block_size> min_broadcast_b2 = aie::broadcast(min_b);
          aie::vector<T_b, block_size> min_broadcast_b = min_broadcast_b2.template extract<block_size>(0);
          aie::accum<acc32, block_size> acc_32;
          acc_32.from_vector(min_broadcast_b);

          aie::vector<uint8, block_size> asUint8 = aie::unpack(I0); // Unpack the 4 bit values to 8 bits
          aie::vector<int8, block_size> asInt8 = asUint8.cast_to<int8>();
          acc_32 = aie::mac(acc_32, asInt8, sfV[k]);
          aie::vector<int16, block_size> asInt16 = acc_32.to_vector<int16>(0); // Convert to int16

          aie::vector<bfloat16, block_size> as_bf16 = aie::to_float<bfloat16>(asInt16, 0); // Convert to bfloat16
          
          aie::accum<accfloat, block_size> accf;
          accf.from_vector(min_broadcast_sb);
          accf = aie::mac(accf, as_bf16, sf_sb);
          aie::store_v(pO, accf.to_vector<bfloat16>(0)); // Write the scaled bfloat16 values to output
          //aie::store_v(pO, as_bf16);
          pO += block_size; // Advance by the number of bytes
        }
      event1();
    }
  event1();
  event1();
}

extern "C" {

void expand_uint4_to_bfloat16(uint4 *a_in, bfloat16 *c_out) {
  expand<uint4, int8, bfloat16, bfloat16, 1024>(a_in, c_out);
}

} // extern "C"