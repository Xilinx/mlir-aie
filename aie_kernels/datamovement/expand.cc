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

#if AIE_TUNED_AIE2
// AIE2 has no 32-lane bf16 multiply: a bf16 vmac.f sums, per lane i of 16,
// a[i] * b[i] and a[16 + i] * b[16 + i]. A block's 32 biased values are one
// operand, and the scale (0 in the other half) picks the 16 lanes an output
// takes. The accumulator starts at -128 * scale, so (128 + nn) * scale lands
// on nn * scale exactly.
template <typename T_in, typename T_sf, typename T_out, const int N,
          const int G>
void expand(T_in *__restrict in, T_out *__restrict out) {
  constexpr int block_size = 32;
  constexpr int blocks_per_group = G / block_size;
  constexpr int blocks = N / block_size;
  static_assert((G % block_size) == 0, "GROUP_SIZE must be a multiple of 32");
  // An odd block count makes the payload 2 mod 4 bytes, which no DMA moves.
  static_assert(blocks % 2 == 0, "TILE_SIZE must be a multiple of 64");

  T_in *__restrict pI = in;
  T_in *pSFb = in + N / 2;
  const T_sf *__restrict pSF = (const T_sf *)pSFb;
  T_out *__restrict pO = out;
  const aie::vector<uint8, 64> hi_byte = aie::broadcast<uint8, 64>(0x43);
  const aie::vector<bfloat16, 32> neg_bias =
      aie::broadcast<bfloat16, 32>(bfloat16(-128.0f));
  const aie::vector<bfloat16, 32> zero = aie::zeros<bfloat16, 32>();
  const aie::mask<32> lo_half = aie::mask<32>::from_uint32(0x0000ffffu);

  struct scale {
    v32bfloat16 lo, hi;
    v16accfloat bias;
  };
  auto make_scale = [&](bfloat16 sf) __attribute__((always_inline)) {
    const aie::vector<bfloat16, 32> s = aie::broadcast<bfloat16, 32>(sf);
    scale c;
    c.lo = aie::select(zero, s, lo_half);
    c.hi = aie::select(s, zero, lo_half);
    c.bias = mul_elem_16_2(neg_bias, c.lo);
    return c;
  };
  auto block = [&](const aie::vector<uint8, 64> &biased_bytes,
                   const scale &c) __attribute__((always_inline)) {
    const v32bfloat16 biased = biased_bytes.cast_to<bfloat16>();
    aie::store_v(pO,
                 aie::accum<accfloat, 16>(mac_elem_16_2(biased, c.lo, c.bias))
                     .to_vector<bfloat16>());
    aie::store_v(pO + 16,
                 aie::accum<accfloat, 16>(mac_elem_16_2(biased, c.hi, c.bias))
                     .to_vector<bfloat16>());
    pO += block_size;
  };
  // Two blocks from one 64-nibble load.
  auto two_blocks = [&](const scale &c0,
                        const scale &c1) __attribute__((always_inline)) {
    const aie::vector<uint8, 64> nibbles =
        aie::unpack(aie::load_v<2 * block_size>(pI));
    pI += block_size;
    const auto [lo, hi] = aie::interleave_zip(nibbles, hi_byte, 1);
    block(lo, c0);
    block(hi, c1);
  };

  event0();
  if constexpr (blocks_per_group % 2) {
    // Two groups at a time keep every 64-nibble load 32-byte aligned; the
    // block count is even, so the group count is too.
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < blocks / blocks_per_group; g += 2) {
      const scale c0 = make_scale(pSF[0]);
      const scale c1 = make_scale(pSF[1]);
      pSF += 2;
      AIE_LOOP_UNROLL_FULL
      for (int k = 1; k < blocks_per_group; k += 2)
        two_blocks(c0, c0);
      two_blocks(c0, c1);
      AIE_LOOP_UNROLL_FULL
      for (int k = 1; k < blocks_per_group; k += 2)
        two_blocks(c1, c1);
    }
  } else {
    AIE_LOOP_NO_UNROLL
    for (int g = 0; g < blocks / blocks_per_group; g++) {
      const scale c = make_scale(*pSF++);
      AIE_LOOP_UNROLL_FULL
      for (int k = 0; k < blocks_per_group; k += 2)
        two_blocks(c, c);
    }
  }
  event1();
}
#else
template <typename T_in, typename T_sf, typename T_out, const int N,
          const int G>
// in and out are distinct in every design that binds this; without
// __restrict the loop below cannot overlap iterations.
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
#if AIE_TUNED_AIE2P
  // A block's mul -> msc chain holds its accumulator for the whole latency;
  // two blocks in flight fill the gap.
  AIE_LOOP_UNROLL(2)
#endif
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
        // each nibble with a 0x43 byte converts it in one shuffle.
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
#endif

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
