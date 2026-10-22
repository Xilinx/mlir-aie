//===- table_i8.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Maps int8 values through a 256-entry int8 table indexed by value + 128 (a
// quantized activation such as SiLU). On AIE2P the 4-way LUT read takes 32
// lookups a fetch; elsewhere each byte is a lane extract and a table load,
// four of them packed into a word store, as byte stores issue only every few
// cycles.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_COMMON_TABLE_I8_H
#define AIE_KERNELS_COMMON_TABLE_I8_H

#include <stdint.h>

#include <aie_api/aie.hpp>

#include "../aie_kernel_utils.h"

#if __AIE_ARCH__ >= 21
#include "aie_bank_placement.h"

#ifndef TABLE_I8_COPIES
#define TABLE_I8_COPIES 2
#endif
static_assert(TABLE_I8_COPIES == 1 || TABLE_I8_COPIES == 2,
              "the table map keeps one or two copies of the table");

// The table as the LUT read takes it: an int16 per entry, and each run of 16
// entries twice over, as the read takes a lane's parity for address bit 5. A
// stack array is not reachable by the LUT read, so it lives here. The read
// fetches its two halves from two tables: copies in two banks are read in
// parallel, one copy twice serially, for 1 KB less. Every table kernel on a
// core shares its copies, as each call writes them before it reads them.
#if TABLE_I8_COPIES == 2
AIE_BANK_A alignas(64) inline int16_t table_i8_runs[512];
AIE_BANK_B alignas(64) inline int16_t table_i8_runs_b[512];
#else
alignas(64) inline int16_t table_i8_runs_one[512];
#endif

// The entries for 32 int8 indices from runs whose entry 128 is at `base` and
// `base_b` (two copies, read in parallel, or one copy twice), each lane's
// offset moved on by `first` (lanes 0 to 15) or `second` (16 to 31) bytes.
static inline __attribute__((always_inline)) aie::vector<int16, 32>
read_runs(const int8_t *base, const int8_t *base_b, aie::vector<int8, 32> index,
          aie::vector<int32, 16> first = aie::zeros<int32, 16>(),
          aie::vector<int32, 16> second = aie::zeros<int32, 16>()) {
  // 4 bytes of offset an entry.
  const aie::vector<int32, 32> wide = index.unpack().unpack();
  const aie::vector<int32, 16> lanes0 =
      aie::add(aie::upshift(wide.template extract<16>(0), 2), first);
  const aie::vector<int32, 16> lanes1 =
      aie::add(aie::upshift(wide.template extract<16>(1), 2), second);
  v64int8 r0, r1, r2, r3;
  load_lut_2x_int8(base, base_b, lanes0, r0, r1);
  load_lut_2x_int8(base, base_b, lanes1, r2, r3);
  return aie::vector<int16, 32>(
      (v32int16)::shuffle(::shuffle(r0, r1, T16_16x4_lo),
                          ::shuffle(r2, r3, T16_16x4_lo), T256_2x2_lo));
}

static inline __attribute__((always_inline)) aie::vector<int16, 32>
read_runs(const int8_t *base, aie::vector<int8, 32> index,
          aie::vector<int32, 16> first = aie::zeros<int32, 16>(),
          aie::vector<int32, 16> second = aie::zeros<int32, 16>()) {
  return read_runs(base, base, index, first, second);
}

// Maps K vectors of 32 bytes at `data` in place through the runs, whose entry
// 128 is at `base` and `base_b` in the two copies. The LUT reads cannot be told
// apart from the stores, so all K are read before any is stored.
template <unsigned K>
static inline __attribute__((always_inline)) void
map_runs_i8(int8_t *data, const int8_t *base, const int8_t *base_b) {
  aie::vector<int8, 32> x[K];
  AIE_LOOP_UNROLL_FULL
  for (unsigned k = 0; k < K; k++)
    x[k] = aie::load_v<32>(data + 32 * k);
  aie::vector<int16, 32> read[K];
  AIE_LOOP_UNROLL_FULL
  for (unsigned k = 0; k < K; k++)
    read[k] = read_runs(base, base_b, x[k]);
  AIE_LOOP_UNROLL_FULL
  for (unsigned k = 0; k < K; k++)
    aie::store_v(data + 32 * k, read[k].pack());
}

// Maps N bytes in place, 128 at a time; `data` and `table` 32-byte aligned.
template <unsigned N>
static inline __attribute__((always_inline)) void
map_table_i8(int8_t *data, const int8_t *__restrict table) {
  static_assert(N % 32 == 0, "the table maps 32 bytes at a time");
#if TABLE_I8_COPIES == 2
  int16_t *runs[2] = {table_i8_runs, table_i8_runs_b};
#else
  int16_t *runs[1] = {table_i8_runs_one};
#endif
  AIE_LOOP_UNROLL_FULL
  for (unsigned r = 0; r < 16; r++) {
    const aie::vector<int16, 16> run = aie::load_v<16>(table + 16 * r).unpack();
    AIE_LOOP_UNROLL_FULL
    for (unsigned c = 0; c < TABLE_I8_COPIES; c++) {
      aie::store_v(runs[c] + 32 * r, run);
      aie::store_v(runs[c] + 32 * r + 16, run);
    }
  }
  int8_t *base = reinterpret_cast<int8_t *>(runs[0]) + 512;
  int8_t *base_b = reinterpret_cast<int8_t *>(runs[TABLE_I8_COPIES - 1]) + 512;
  for (unsigned i = 0; i < N / 128 * 128; i += 128)
    map_runs_i8<4>(data + i, base, base_b);
  if constexpr (N % 128)
    map_runs_i8<N % 128 / 32>(data + N / 128 * 128, base, base_b);
}
#else
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
  using word = uint32_t __attribute__((may_alias));
  word *__restrict out = reinterpret_cast<word *>(dst);
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
#endif

#endif // AIE_KERNELS_COMMON_TABLE_I8_H
