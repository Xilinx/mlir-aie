//===- AIECoreMemory.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared vocabulary for a core tile's data memory and object layout, used by
// the several places that have to agree about them: the buffer allocator
// (AIEAssignBuffers), the linker script emitter (AIETargetLdScript), core
// outlining (AIECoreToStandard), and aiecc's stack analysis.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_IR_AIECOREMEMORY_H
#define AIE_DIALECT_AIE_IR_AIECOREMEMORY_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace xilinx::AIE {

// Stack requirement aiecc's call-graph analysis computed for a core, stamped
// on the CoreOp so the buffer allocator's memory-map diagnostics can show it
// next to the stack region. aiecc erases it again before the module is handed
// on, so it never reaches user-visible IR.
inline constexpr llvm::StringLiteral kComputedStackRequirementAttrName =
    "aiecc.computed_stack_requirement";

// Name of the top-level function AIECoreToStandard outlines a CoreOp's body
// into. The canonical definition is there (where the function is actually
// created); every other reader of a compiled core object -- aiecc's
// post-build stack-size check reads this function's own frame size back out
// of the object -- must agree on the same name, so they share this rather
// than each formatting "core_<col>_<row>" independently.
inline std::string coreFrameSymbolName(int col, int row) {
  std::string name;
  llvm::raw_string_ostream(name) << "core_" << col << "_" << row;
  return name;
}

// A half-open [start, start + size) run of bytes.
struct MemoryRun {
  int64_t start = 0;
  int64_t size = 0;
};

// Largest run of free bytes in [0, memSize), given occupied half-open
// intervals that need not be sorted or disjoint.
//
// This is the number that decides whether a core links. AIETargetLdScript
// hands the core compiler exactly one region for its own .data/.rodata/.bss --
// this run -- so the allocator's reserved_data_size acceptance test and the
// linker script must agree byte for byte. Both call this rather than sweeping
// the intervals themselves.
inline MemoryRun
largestFreeRun(int64_t memSize,
               llvm::SmallVector<std::pair<int64_t, int64_t>> occupied) {
  llvm::sort(occupied);
  MemoryRun best;
  int64_t cursor = 0;
  for (auto &interval : occupied) {
    int64_t gapEnd = std::min(interval.first, memSize);
    if (gapEnd - cursor > best.size)
      best = {cursor, gapEnd - cursor};
    cursor = std::max(cursor, interval.second);
    if (cursor >= memSize)
      return best;
  }
  if (memSize - cursor > best.size)
    best = {cursor, memSize - cursor};
  return best;
}

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_IR_AIECOREMEMORY_H
