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
void expand_intrinsics(T_in *in, T_out *out) {

  constexpr int block_size = 32;
  constexpr int number_of_blocks = 8;
  // Super block size = block_size x number_of_blocks

  v64int4 *pI = (v64int4 *)in;
  T_in *__restrict pSFb = in + N/2; // The scale factors are after the inputs
  v32bfloat16 *pSF = (v32bfloat16 *)pSFb;
  v16bfloat16 *pO = (v16bfloat16 *)out;

  constexpr int F = N / (block_size * number_of_blocks); // number of super blocks
  event0();
  event0();
  v32bfloat16 sfV = *pSF++; // Load 32 scale factors
  
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(F, ) { // Ali modified 16 -> F
      event0();
      for (int k = 0; k < number_of_blocks; k+=2){
        v64int4 I0 = *pI++; // Load two blocks
        v64int8 asInt8 = unpack(I0); // Unpack the 4 bit values to 8 bits for two blocks
        for (int j = 0; j < 2; j++) {
          //v32bfloat16 sf_broadcast = broadcast_elem(sfV, i*number_of_blocks+k+j); // Load sf for one block --> does not do what I though it would!!!!!!
          v32bfloat16 sf_broadcast = broadcast_bfloat16(ext_elem(sfV, i*number_of_blocks+k+j)); // Load sf for one block
          v32int16 asInt16 = unpack(extract_v32int8(asInt8, j)); // Unpack the 8 bit values to 16 bits for one block
          // v32bfloat16 as_bf16 = v32bfloat16(asInt16); // Convert to bfloat16 for one block --> does not do what I though it would!!!!!!
          v32acc32 as_acc32 = mul_elem_32(asInt16, broadcast_one_to_v32int16());
          v32bfloat16 as_bf16 = to_v32bfloat16(v32accfloat(as_acc32)); // This is wrong

          // Prepare for 1x2x1 multiplication using mul_elem_16_2
          // First half of the data
          v16bfloat16 as_bf16_0 = extract_v16bfloat16(as_bf16, 0); 
          v32bfloat16 as_bf16_0padded = insert(broadcast_zero_bfloat16(), 0, as_bf16_0); // 16 data + 16 zero
          // First half of the sf
          v16bfloat16 sf_broadcast_0 = extract_v16bfloat16(sf_broadcast, 0);
          v32bfloat16 sf_broadcast_0padded = insert(broadcast_zero_bfloat16(), 0, sf_broadcast_0); // 16 sf + 16 zero
          // multiply and store first 16 values
          v16accfloat scaled_bf16_0 = mul_elem_16_2(as_bf16_0padded, sf_broadcast_0padded);
          *pO++ = srs(scaled_bf16_0);
          // Second half of the data
          v16bfloat16 as_bf16_1 = extract_v16bfloat16(as_bf16, 1);
          v32bfloat16 as_bf16_1padded = insert(broadcast_zero_bfloat16(), 0, as_bf16_1); // 16 data + 16 zero 
          //  Second half of the sf
          v16bfloat16 sf_broadcast_1 = extract_v16bfloat16(sf_broadcast, 1);
          v32bfloat16 sf_broadcast_1padded = insert(broadcast_zero_bfloat16(), 0, sf_broadcast_1); // 16 sf + 16 zero
          // multiply and store second 16 values
          v16accfloat scaled_bf16_1 = mul_elem_16_2(as_bf16_1padded, sf_broadcast_1padded);
          *pO++ = srs(scaled_bf16_1);
        }
      }
      event1();
    }
  event1();
  event1();
}

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
    chess_prepare_for_pipelining chess_loop_range(F, ) { // Ali modified 16 -> F
      aie::vector<T_sf, number_of_blocks> sfV = aie::load_v<number_of_blocks>(pSF); // Load number_of_blocks scale factors
      pSF += number_of_blocks; // Advance by the number of bytes
      event0();
      for (int k = 0; k < number_of_blocks; k++){
          aie::vector<T_in, block_size> I0 = aie::load_v<block_size>(pI); // Load one block of input
          pI += block_size/2; // Advance by the number of bytes

          bfloat16 sf = sfV[k]; // Ali modified from sfV[k % number_of_blocks]

          aie::vector<bfloat16, block_size> sf_broadcast = aie::broadcast(sf);

          // Upsize these to 8 bits -> 16 -> bfloat16
          aie::vector<int8, block_size> asInt8 = aie::unpack(I0); // Unpack the 4 bit values to 8 bits
          aie::vector<int16, block_size> asInt16 = aie::unpack(asInt8); // Unpack the 8 bit values to 16 bits
          aie::vector<bfloat16, block_size> as_bf16 = aie::to_float<bfloat16>(asInt16, 0); // Convert to bfloat16
          aie::vector<bfloat16, block_size> scaled_bf16 = aie::mul(as_bf16, sf_broadcast); // Scale the bfloat16 values
          aie::store_v(pO, scaled_bf16); // Write the scaled bfloat16 values to output
          pO += block_size; // Advance by the number of bytes
        }
      event1();
    }
  event1();
  event1();
}

extern "C" {

void expand_int4_to_bfloat16(int4 *a_in, bfloat16 *c_out) {
  expand<int4, bfloat16, bfloat16, 1024>(a_in, c_out);
}

} // extern "C"