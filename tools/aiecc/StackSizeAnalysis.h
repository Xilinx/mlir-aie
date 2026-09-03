//===- StackSizeAnalysis.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes a core's stack requirement from its linked ELF: the maximum over
// the frame-weighted paths that leave the ELF entry point, following the call
// graph that the `.stack_sizes` data and the relocations describe. One call
// chain is live at a time, so the maximum bounds the requirement. A symbol the
// analysis cannot measure, and a symbol that recurses, both end the walk with
// a failure, so the result is an upper bound.
//
// The linker decides what a core contains, so measuring its output covers the
// toolchain's own startup code.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_STACKSIZEANALYSIS_H
#define AIECC_STACKSIZEANALYSIS_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xilinx::aiecc {

// Cycle: the requirement is unbounded, so the design must declare a
// stack_size_override. Unmeasurable: the ELF is unreadable, or its
// `.stack_sizes` data is malformed. The driver warns for this case.
enum class StackRequirementFailure { Cycle, Unmeasurable };

struct StackRequirementResult {
  std::optional<int64_t> bytes;
  std::string error;
  StackRequirementFailure failureKind = StackRequirementFailure::Unmeasurable;
  // Functions the ELF holds no `.stack_sizes` entry for. Their frames count as
  // 0, so `bytes` is a lower bound while this list is non-empty. Peano's aie2
  // crt1.o puts `_main_init` here; a kernel compiled without
  // -fstack-size-section puts its own functions here.
  std::vector<std::string> unmeasured;
};

// Measures the stack that the linked core ELF at `elfPath` needs. `overrides`
// maps a function name to a declared requirement for its whole subtree, and
// the walk stops at such a function.
//
// The link must keep the relocations (`-Wl,--emit-relocs`), which carry the
// call edges.
StackRequirementResult
computeStackRequirement(llvm::StringRef elfPath,
                        const llvm::StringMap<int64_t> &overrides);

} // namespace xilinx::aiecc

#endif // AIECC_STACKSIZEANALYSIS_H
