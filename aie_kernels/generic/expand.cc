//===- expand.cc ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T_in, typename T_sf, typename T_out, const int N,
          const int G>
// in and out are distinct objects in every design that binds this, and saying
// so is what lets the loop below overlap iterations: without it the scheduler
// has to assume each store may feed the next block's load and settles for a
// single-stage schedule at II 26, against a six-stage one at II 4.
void expand(T_in *__restrict in, T_out *__restrict out) {
  // Keep vector width constant; group size can vary as a multiple of 32
  constexpr int block_size = 32;
  constexpr int blocks_per_group = G / block_size;
  constexpr int groups_per_tile = N / G;
  // Super block size = block_size x blocks_per_group
  static_assert((G % block_size) == 0, "GROUP_SIZE must be a multiple of 32");

  T_in *__restrict pI = in; // Input pointer
  T_in *pSFb = in + N / 2;  // The scale factors are after the inputs
  T_sf *pSF =
      (T_sf *)pSFb; // But we only advance by the number of bytes not elements
  T_out *__restrict pO = out;
  const int F = groups_per_tile; // iterate over groups of size GROUP_SIZE
  const aie::vector<uint8, block_size> hi_byte =
      aie::broadcast<uint8, block_size>(0x43);
  const aie::vector<bfloat16, block_size> bias =
      aie::broadcast<bfloat16, block_size>(bfloat16(128.0f));
  event0();
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(F, ) { // 16 -> F
      // Load one scale per group (scalar load)
      T_sf sf = *pSF;
      pSF += 1;
      for (int k = 0; k < blocks_per_group; k++) {
        aie::vector<T_in, block_size> I0 =
            aie::load_v<block_size>(pI); // Load one block of input (32 uint4s)
        pI += block_size / 2;            // Advance by the number of bytes

        bfloat16 sf_bf16 = sf;

        aie::vector<bfloat16, block_size> sf_broadcast =
            aie::broadcast(sf_bf16);

        aie::vector<uint8, block_size> asInt8 =
            aie::unpack(I0); // Unpack the 4 bit values to 8 bits
        // 0x43nn is the bfloat16 for 128 + nn whenever nn < 128, so pairing
        // each nibble with a 0x43 byte lands the value in bfloat16 with one
        // shuffle, where to_float spends a ups/add/sub/conv chain on it.
        auto [lo, hi] = aie::interleave_zip(asInt8, hi_byte, 1);
        aie::vector<bfloat16, block_size> biased =
            aie::concat(lo, hi).cast_to<bfloat16>();
        // (128 + nn) * sf - 128 * sf is nn * sf exactly: both products widen a
        // bfloat16 pair into the f32 accumulator, so neither one rounds.
        aie::vector<bfloat16, block_size> scaled_bf16 =
            aie::msc(aie::mul(biased, sf_broadcast), bias, sf_broadcast)
                .to_vector<bfloat16>();
        aie::store_v(pO,
                     scaled_bf16); // Write the scaled bfloat16 values to output
        pO += block_size;          // Advance by the number of bytes
      }
    }
  event1();
}

extern "C" {

#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 1024
#endif

void expand_uint4_to_bfloat16(uint4 *a_in, bfloat16 *c_out) {
  expand<uint4, bfloat16, bfloat16, TILE_SIZE, GROUP_SIZE>(a_in, c_out);
}

} // extern "C"
