//===- StackSizeAnalysis.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes a core's stack requirement from its call graph: the maximum over
// root-to-leaf paths through the `.stack_sizes` data and the call and data
// relocations of its `link_files` objects. One call chain is live at a time,
// so the maximum bounds the requirement. A symbol the analysis cannot
// measure, and a symbol that recurses, both end the walk with a failure, so
// the result is an upper bound.
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
  int64_t frameSize = -1; // -1 marks an unmeasured frame
  llvm::SmallVector<std::string, 4> callees;
};

// The call graph across every object linked into one core, keyed by graph key
// (see the comment on SectionOwners in the implementation).
// addObjectToStackGraph fills dataReferences and dataEscapes for the
// indirect-call heuristic, and resolveIndirectCallEdges turns them into call
// edges.
struct StackGraph {
  llvm::StringMap<StackGraphNode> nodes;
  llvm::StringMap<llvm::StringSet<>> dataReferences;
  llvm::StringMap<llvm::SmallVector<std::string, 2>> dataEscapes;
};

// Collects every function symbol that `path` defines into `names`. Returns
// false when `path` holds something other than a plain relocatable object.
bool collectDefinedFunctionNames(llvm::StringRef path,
                                 llvm::StringSet<> &names);

// Parses `path`'s `.stack_sizes` entries and its call and data relocations
// into `graph`. `knownFunctions` must name every function that any object of
// the core defines, so that a relocation against an undefined symbol counts
// as a cross-object call. The st_type of an undefined symbol is unreliable.
//
// Returns false when the attribution of `path` is incomplete. `graph` stays
// usable: an incomplete attribution skips a write and leaves frameSize at the
// -1 sentinel.
bool addObjectToStackGraph(llvm::StringRef path,
                           const llvm::StringSet<> &knownFunctions,
                           StackGraph &graph);

// Adds the conservative indirect-call edges to `graph` and sorts each callee
// list. Call once per core, after addObjectToStackGraph runs on every
// link_files object, and before computeStackRequirement.
void resolveIndirectCallEdges(StackGraph &graph);

// Cycle: the requirement is unbounded, so the design must declare a
// stack_size_override. Unmeasurable: an object carries no `.stack_sizes`
// data, for example an archive member. The driver warns for this case.
enum class StackRequirementFailure { Cycle, Unmeasurable };

struct StackRequirementResult {
  std::optional<int64_t> bytes;
  std::string error;
  StackRequirementFailure failureKind = StackRequirementFailure::Unmeasurable;
};

// Returns the maximum, over `roots` (the symbols the core body calls
// directly), of each root's longest frame-weighted path in `graph`.
// `overrides` maps a symbol to a declared requirement for its whole subtree,
// and the walk stops at such a symbol.
StackRequirementResult
computeStackRequirement(const StackGraph &graph,
                        llvm::ArrayRef<llvm::StringRef> roots,
                        const llvm::StringMap<int64_t> &overrides);

// Returns a function's own frame size, without a call-graph walk. The driver
// reads a core's top-level frame this way after codegen.
// computeStackRequirement covers the callees of that frame.
std::optional<int64_t> measureFunctionFrameSize(llvm::StringRef path,
                                                llvm::StringRef symbol);

} // namespace xilinx::aiecc

#endif // AIECC_STACKSIZEANALYSIS_H
