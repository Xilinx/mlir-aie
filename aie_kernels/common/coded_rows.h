//===- coded_rows.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decodes aie.utils.coded rows back to bf16, for any kernel whose bf16
// operand arrives coded. A row of K weights is K low bytes, K / 2 code
// nibbles, then at a 64-byte boundary base_pos, base_neg and the exception
// count, then `capacity` exceptions of (index, u16; high byte; pad); the row
// is a multiple of 64 bytes, so every row's planes stay 64-byte aligned.

#ifndef AIE_KERNELS_COMMON_CODED_ROWS_H
#define AIE_KERNELS_COMMON_CODED_ROWS_H

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../aie_kernel_utils.h"

template <uint32_t k>
constexpr uint32_t coded_header_at = (3 * k / 2 + 63) / 64 * 64;

template <uint32_t k, uint32_t capacity>
constexpr uint32_t coded_row_bytes =
    coded_header_at<k> + (4 + 4 * capacity + 63) / 64 * 64;

// m coded rows of k weights from `a` into m bf16 rows at `out`:
// hi = code < 8 ? base_pos + code : base_neg + code - 8, interleaved with the
// low bytes into little-endian bf16 words, then the exceptions patched in.
template <uint32_t k, uint32_t capacity>
static inline void decode_coded_rows(uint32_t m, const uint8_t *__restrict a,
                                     bfloat16 *__restrict out) {
  static_assert(k % 64 == 0);
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *__restrict lo = a + row * coded_row_bytes<k, capacity>;
    const uint8_t *__restrict header = lo + coded_header_at<k>;
    uint8_t *__restrict o = (uint8_t *)(out + row * k);
    const aie::vector<uint8, 64> pos = aie::broadcast<uint8, 64>(header[0]);
    const aie::vector<uint8, 64> neg =
        aie::broadcast<uint8, 64>((uint8_t)(header[1] - 8));
    // Stepped pointers, not offsets from j: Peano then folds the code load
    // into vldb.unpack and pipelines 64 weights in 5 cycles.
    const uint8_t *__restrict pl = lo;
    const uint8_t *__restrict pc = lo + k;
    uint8_t *__restrict po = o;
    AIE_LOOP_MIN_ITERATION_COUNT(k / 64)
    for (uint32_t j = 0; j < k / 64; j++, pl += 64, pc += 32, po += 128) {
      aie::vector<uint8, 64> c = aie::load_v<64>((const uint4 *)pc).unpack();
      aie::vector<uint8, 64> hi =
          aie::add(c, aie::select(neg, pos, aie::lt(c, (uint8_t)8)));
      auto [w0, w1] = aie::interleave_zip(aie::load_v<64>(pl), hi, 1);
      aie::store_v(po, w0);
      aie::store_v(po + 64, w1);
    }
    const uint32_t *__restrict e = (const uint32_t *)(header + 4);
    for (uint32_t n = header[2]; n > 0; n--, e++)
      o[2 * (*e & 0xFFFF) + 1] = (uint8_t)(*e >> 16);
  }
}

#endif // AIE_KERNELS_COMMON_CODED_ROWS_H
