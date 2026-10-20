//===- table_i8.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stores a vector of int8 values mapped through a 256-entry int8 table
// indexed by value + 128 (a quantized activation such as SiLU). AIE2P has no
// vector gather: each byte is a lane extract and a table load, and four of
// them are packed into a word store, as byte stores issue only every few
// cycles.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_COMMON_TABLE_I8_H
#define AIE_KERNELS_COMMON_TABLE_I8_H

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../aie_kernel_utils.h"

// `dst` 4-byte aligned.
template <unsigned N>
static inline __attribute__((always_inline)) void
store_table_i8(int8_t *__restrict dst, aie::vector<int8, N> values,
               const int8_t *__restrict table) {
  // value + 128 is the value's byte with its top bit flipped.
  const aie::vector<uint8, N> index =
      aie::bit_xor(values, aie::broadcast<int8, N>(-128))
          .template cast_to<uint8>();
  const uint8_t *__restrict t = reinterpret_cast<const uint8_t *>(table);
  uint32_t *__restrict out = reinterpret_cast<uint32_t *>(dst);
  AIE_LOOP_UNROLL_FULL
  for (unsigned i = 0; i < N / 4; i++)
    out[i] = (uint32_t)t[index[4 * i]] | (uint32_t)t[index[4 * i + 1]] << 8 |
             (uint32_t)t[index[4 * i + 2]] << 16 |
             (uint32_t)t[index[4 * i + 3]] << 24;
}

// Maps N bytes in place, 64 at a time; `data` 32-byte aligned.
template <unsigned N>
static inline __attribute__((always_inline)) void
map_table_i8(int8_t *data, const int8_t *__restrict table) {
  static_assert(N % 32 == 0, "the table maps 32 bytes at a time");
  AIE_LOOP_NO_UNROLL
  for (unsigned i = 0; i < N / 64 * 64; i += 64)
    store_table_i8(
        data + i,
        aie::concat(aie::load_v<32>(data + i), aie::load_v<32>(data + i + 32)),
        table);
  if constexpr (N % 64)
    store_table_i8(data + N - 32, aie::load_v<32>(data + N - 32), table);
}

#endif // AIE_KERNELS_COMMON_TABLE_I8_H
