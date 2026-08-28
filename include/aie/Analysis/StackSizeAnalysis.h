//===- StackSizeAnalysis.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes how much stack a core's call tree actually needs, by walking the
// `.stack_sizes` section (see aiecc.cpp's core-object `llc` invocation) and
// the call/data relocations of a core's `link_files` objects.
//
// Unlike reserved_data_size (a sum -- static buffers coexist regardless of
// control flow), a call tree's requirement is the MAX over root-to-leaf
// paths, since only one call chain is live at a time. An unbroken recursive
// cycle is a hard failure unless the caller overrides the affected root --
// undercounting would corrupt memory, the one direction this analysis must
// never be wrong in. A missing measurement (an archive/bitcode input, or a
// pre-existing kernel object) is instead just a warning, the same "haven't
// measured this yet" gap reserved_data_size already treats as non-fatal.
//
// Building the graph takes two passes because a cross-object call leaves, in
// the caller's object, a relocation against an *undefined* symbol whose
// st_type is unreliable (this toolchain emits NOTYPE for it) -- nothing in
// that object alone says whether the name is a callee or an ordinary extern
// data symbol. Pass 1 (collectDefinedFunctionNames) gathers every function
// name any of a core's objects defines; pass 2 (addObjectToStackGraph) then
// treats a relocation target as a call if it's a function locally, or its
// name is in that closure.
//
// Indirect calls (through a function pointer) leave no relocation naming a
// callee. The only available signal is the reverse direction: some function
// F's address is taken from a data symbol D, and some other function C
// references D -- treated as a conservative edge C -> F, folding F's frame
// into C's path. This can only overcount, the safe direction, and is
// resolved once across a core's whole object set (resolveIndirectCallEdges),
// since C, D, and F may live in different objects.
//
// Attributing a `.stack_sizes` entry to a function: this toolchain places
// each function in its own section, paired with a `.stack_sizes` entry
// relocated against a local marker symbol in that same section. So
// attribution follows the entry's relocation to its target's *section* (not
// name -- the target is usually the marker, not the function symbol) and
// looks up the one function this object defines there. A section shared by
// more than one function symbol makes the whole object unmeasurable.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_ANALYSIS_STACKSIZEANALYSIS_H
#define AIE_ANALYSIS_STACKSIZEANALYSIS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <optional>
#include <string>

namespace xilinx::aiecc {

// One function's contribution to the call graph: its own frame size and the
// symbols it (directly, or conservatively via an indirect-call data
// reference) calls.
struct StackGraphNode {
  int64_t frameSize = -1; // -1 = not yet measured
  llvm::SmallVector<std::string, 4> callees;
};

// The merged call graph across every object linked into one core. Symbols
// are looked up by name, so a callee defined in a sibling object resolves
// naturally once both objects have been added. `dataReferences` and
// `dataEscapes` are bookkeeping for the indirect-call resolution described
// in the file comment, consumed once by resolveIndirectCallEdges after every
// object has contributed.
struct StackGraph {
  llvm::StringMap<StackGraphNode> nodes;
  llvm::StringMap<llvm::StringSet<>>
      dataReferences; // func -> data syms it touches
  llvm::StringMap<llvm::SmallVector<std::string, 2>>
      dataEscapes; // data sym -> funcs stored there
};

// Pass 1: collect every function symbol `path` *defines* (not merely
// references) into `names`. Returns false if `path` isn't a plain
// relocatable object (archive, bitcode, missing file); `names` is unioned
// across every call regardless, so one unmeasurable object doesn't lose
// what the others defined.
bool collectDefinedFunctionNames(llvm::StringRef path,
                                 llvm::StringSet<> &names);

// Pass 2: parse `path`'s `.stack_sizes` entries and its functions' call/data
// relocations into `graph`, merging with whatever it already contains (so a
// multi-object core builds one shared graph). `knownFunctions` must already
// contain every function name defined anywhere in the core's object set
// (see collectDefinedFunctionNames).
//
// Returns false if `path` cannot be fully attributed (not a plain
// relocatable object, an ambiguous section, an entry that doesn't resolve to
// a known function, a malformed ULEB128); each such case only skips a write,
// leaving frameSize at its -1 sentinel, so `graph` stays safe to use as-is.
bool addObjectToStackGraph(llvm::StringRef path,
                           const llvm::StringSet<> &knownFunctions,
                           StackGraph &graph);

// Wire in the conservative indirect-call edges recorded across however many
// objects contributed to `graph`, then normalize every callee list. Call
// once per core after all of its link_files objects have been added via
// addObjectToStackGraph, and before computeStackRequirement.
void resolveIndirectCallEdges(StackGraph &graph);

// Cycle: the requirement is unbounded, so an override is always required.
// Unmeasurable: the same "haven't measured this yet" gap reserved_data_size
// treats as a warning (an archive/bitcode input, or a pre-existing kernel
// object) -- failing outright would break every such design on first
// contact with a newer aiecc.
enum class StackRequirementFailure { Cycle, Unmeasurable };

// Result of computing one core's stack requirement: either the computed
// byte count, or a human-readable reason it could not be computed, tagged
// with whether that reason is a hard failure (Cycle) or merely incomplete
// information (Unmeasurable) -- see StackRequirementFailure.
struct StackRequirementResult {
  std::optional<int64_t> bytes;
  std::string error;
  StackRequirementFailure failureKind = StackRequirementFailure::Unmeasurable;
};

// Compute a core's stack requirement as the max over `roots` (the symbols
// directly called from the core body) of each root's longest
// frame-size-weighted root-to-leaf path in `graph`. `overrides` names
// symbols (typically the roots themselves -- the kernel entry points MLIR
// declared) whose subtree is a declared leaf rather than analyzed.
StackRequirementResult
computeStackRequirement(const StackGraph &graph,
                        llvm::ArrayRef<llvm::StringRef> roots,
                        const llvm::StringMap<int64_t> &overrides);

// Measure one function's own frame size from a compiled object, without any
// call-graph walk -- used to read back the core body's own top-level frame
// (symbol `core_<col>_<row>`, see AIECoreToStandard.cpp), which
// computeStackRequirement above cannot see since it only knows a core's
// *callees*. Returns nullopt if `path` is not a plain relocatable object or
// `symbol` has no `.stack_sizes` entry.
std::optional<int64_t> measureFunctionFrameSize(llvm::StringRef path,
                                                llvm::StringRef symbol);

} // namespace xilinx::aiecc

#endif // AIE_ANALYSIS_STACKSIZEANALYSIS_H
