//===- AIECoreMemory.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Data memory layout of a core tile. The buffer allocator (AIEAssignBuffers)
// and the linker script emitter (AIETargetLdScript) both describe the layout
// with these types, so they compute the same region.
//
// AIECoreSymbols.h declares the symbol names that the dialect and the aiecc
// driver share.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_IR_AIECOREMEMORY_H
#define AIE_DIALECT_AIE_IR_AIECOREMEMORY_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

namespace xilinx::AIE {

// A half-open [start, start + size) run of bytes. Banks, free runs and a core's
// data region all use this type.
struct MemoryRun {
  int64_t start = 0;
  int64_t size = 0;

  int64_t end() const { return start + size; }
  bool contains(int64_t addr) const { return addr >= start && addr < end(); }
};

// Largest run of free bytes in [0, memSize) whose start is a multiple of
// `alignBytes`. The occupied half-open intervals may be unsorted and may
// overlap.
//
// AIETargetLdScript grants the core compiler one region for its .data, .rodata
// and .bss: this run. The allocator's reserved_data_size check and the linker
// script both call this, so they compute the same size.
//
// The linker starts .data at a multiple of its strongest section alignment, so
// an unaligned start loses the difference to padding. A reservation of the
// exact size then overflows the region.
inline MemoryRun
largestFreeRun(int64_t memSize,
               llvm::SmallVector<std::pair<int64_t, int64_t>> occupied,
               int64_t alignBytes = 1) {
  assert(alignBytes > 0 && "alignBytes must be positive");
  llvm::sort(occupied);
  MemoryRun best;
  int64_t cursor = 0;
  auto consider = [&](int64_t start, int64_t end) {
    start = llvm::alignTo(start, alignBytes);
    if (start < end && end - start > best.size)
      best = {start, end - start};
  };
  for (auto &interval : occupied) {
    // A zero-length interval occupies no bytes, so it must not split a run.
    if (interval.first == interval.second)
      continue;
    consider(cursor, std::min(interval.first, memSize));
    cursor = std::max(cursor, interval.second);
    if (cursor >= memSize)
      return best;
  }
  consider(cursor, memSize);
  return best;
}

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_IR_AIECOREMEMORY_H
