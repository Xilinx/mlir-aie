//===- StackSizeAnalysis.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes a core's stack requirement from its call graph: the MAX over
// root-to-leaf paths through `.stack_sizes` data and call/data relocations
// in its `link_files` objects (never the sum -- only one call chain is
// live at a time). Must never undercount; an unmeasurable or recursive
// symbol is reported, not silently assumed safe.
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_STACKSIZEANALYSIS_H
#define AIECC_STACKSIZEANALYSIS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <optional>
#include <string>

namespace xilinx::aiecc {

struct StackGraphNode {
  int64_t frameSize = -1; // -1 = not yet measured
  llvm::SmallVector<std::string, 4> callees;
};

// The merged call graph across every object linked into one core, keyed by
// symbol name. dataReferences/dataEscapes back the indirect-call heuristic
// in addObjectToStackGraph, consumed by resolveIndirectCallEdges.
struct StackGraph {
  llvm::StringMap<StackGraphNode> nodes;
  llvm::StringMap<llvm::StringSet<>> dataReferences;
  llvm::StringMap<llvm::SmallVector<std::string, 2>> dataEscapes;
};

// Collects every function symbol `path` defines into `names`. False means
// `path` isn't a plain relocatable object.
bool collectDefinedFunctionNames(llvm::StringRef path,
                                 llvm::StringSet<> &names);

// Parses `path`'s `.stack_sizes` entries and call/data relocations into
// `graph`. `knownFunctions` must list every function defined anywhere in
// the core's object set, so a relocation against an undefined symbol
// (unreliable st_type) can still be recognized as a cross-object call.
//
// False means `path` could not be fully attributed, but `graph` is still
// safe to use: every such case only skips a write (leaving frameSize at
// its -1 sentinel), never writes a wrong value.
bool addObjectToStackGraph(llvm::StringRef path,
                           const llvm::StringSet<> &knownFunctions,
                           StackGraph &graph);

// Folds the conservative indirect-call edges collected across every object
// into `graph`, then normalizes callee lists. Call once per core, after
// every link_files object has been added and before computeStackRequirement.
void resolveIndirectCallEdges(StackGraph &graph);

// Cycle: unbounded, always requires stack_size_override. Unmeasurable:
// missing .stack_sizes data (e.g. a pre-existing or archived object) -- a
// warning, not a build failure.
enum class StackRequirementFailure { Cycle, Unmeasurable };

struct StackRequirementResult {
  std::optional<int64_t> bytes;
  std::string error;
  StackRequirementFailure failureKind = StackRequirementFailure::Unmeasurable;
};

// The max over `roots` (symbols the core body calls directly) of each
// root's longest frame-weighted path in `graph`. `overrides` names symbols
// whose subtree is a declared leaf rather than analyzed.
StackRequirementResult
computeStackRequirement(const StackGraph &graph,
                        llvm::ArrayRef<llvm::StringRef> roots,
                        const llvm::StringMap<int64_t> &overrides);

// A function's own frame size, without a call-graph walk -- for reading
// back a core's compiled top-level frame after codegen, which
// computeStackRequirement cannot see.
std::optional<int64_t> measureFunctionFrameSize(llvm::StringRef path,
                                                llvm::StringRef symbol);

} // namespace xilinx::aiecc

#endif // AIECC_STACKSIZEANALYSIS_H
