//===- row_addresses.cc -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef ROWS
#define ROWS 8
#endif
#ifndef TABLE_ROWS
#define TABLE_ROWS 1024
#endif
#ifndef ROW_BYTES
#define ROW_BYTES 4096
#endif
#ifndef LOW_BITS
#define LOW_BITS 29
#endif
#ifndef APERTURE
#define APERTURE 0x80000000
#endif

// A 64-bit multiply is a runtime-library call; a table under 4 GiB needs none.
#if (TABLE_ROWS - 1) * ROW_BYTES < 0x100000000
typedef uint32_t offset_t;
#else
typedef uint64_t offset_t;
#endif

// The address words of each id's row of a table, as a shim buffer descriptor
// holds them. A core reads 30 bits of a runtime value, so the table's address
// arrives split at LOW_BITS.
extern "C" void row_addresses(int32_t *ids, uint32_t *out, int32_t lo,
                              int32_t hi) {
  event0();
  uint64_t base =
      ((uint64_t)(uint32_t)hi << LOW_BITS) + (uint32_t)lo + (uint64_t)APERTURE;
  for (int r = 0; r < ROWS; r++) {
    int32_t id = ids[r] < -TABLE_ROWS ? -TABLE_ROWS : ids[r];
    id = id > TABLE_ROWS - 1 ? TABLE_ROWS - 1 : id;
    if (id < 0)
      id += TABLE_ROWS;
    uint64_t address = base + (offset_t)id * (offset_t)ROW_BYTES;
    out[2 * r] = (uint32_t)address & 0xFFFFFFFCu;
    out[2 * r + 1] = (uint32_t)(address >> 32) & 0xFFFFu;
  }
  event1();
}
