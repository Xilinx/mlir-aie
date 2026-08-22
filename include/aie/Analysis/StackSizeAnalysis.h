//===- StackSizeAnalysis.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes how much stack a core's call tree actually needs, by walking the
// `.stack_sizes` section (see aiecc.cpp's core-object `llc` invocation and
// PEANOWRAP*_FLAGS/compile_cxx_core_function for where kernel objects get it
// too) and the call/data relocations of a core's `link_files` objects.
//
// Unlike reserved_data_size (a sum -- static buffers coexist regardless of
// control flow), a call tree's requirement is the MAX over root-to-leaf
// paths: only one call chain is live at a time. Recursion is not statically
// boundable, so an unbroken cycle is a hard failure unless the caller
// supplies a declared override for the affected root symbol -- silently
// assuming 0/unbounded would be unsafe in the wrong direction (undercounting
// corrupts memory; overcounting only wastes it), the reverse of
// reserved_data_size's "unmeasurable -> warn and reserve nothing".
//
// Two passes are required, not one, because of how relocatable objects
// represent cross-object references: a call from object A to a function
// defined in sibling object B leaves, inside A, a relocation against an
// *undefined* symbol named for the callee -- and an undefined symbol's
// st_type is unreliable (this toolchain emits NOTYPE for it), so nothing in
// A alone says whether that undefined name is a function in B or an
// ordinary extern data symbol. Only once every object a core links has been
// inspected for what it *defines* can an undefined reference elsewhere be
// resolved. So: pass 1 (collectDefinedFunctionNames) gathers every function
// name any of a core's objects actually defines; pass 2
// (addObjectToStackGraph, called once per object with that closure) treats
// a relocation target as "a function" if it is one locally (reliable -- the
// object defines it) or its name is in the closure (the only case an
// undefined reference can safely be treated as a call).
//
// Indirect calls (through a function pointer) leave no relocation at the
// call site naming a callee at all -- the target is a runtime register
// load. The only signal available without disassembly is the reverse
// direction: some function F's address is itself taken from a data symbol D
// (a vtable-like structure or function-pointer table), and separately some
// other function C references D. That is treated as a conservative call
// edge C -> F: C might invoke F, so F's frame (and whatever F itself might
// reach) is folded into C's path like any other callee. This can only
// overcount (C is credited with a possible call it may never make at
// runtime), the safe direction; it still only becomes a hard failure when
// the *resulting* graph has a cycle or reaches an unmeasurable symbol --
// indirect calls are not a separately-special-cased failure mode, they
// simply widen the graph conservatively before the same cycle/measurability
// checks run. This resolution is also necessarily deferred across a core's
// whole object set (resolveIndirectCallEdges, called once after every
// object has contributed): C and D can live in different objects than F.
//
// How a `.stack_sizes` entry is attributed to a function: every AIE kernel
// object observed from this toolchain places each function in its own
// section (`.text.<mangled name>`), and pairs it with its own `.stack_sizes`
// section carrying exactly one relocation, against a local `.Lfunc_beginN`
// marker symbol defined in that same function's section. So resolving "which
// function does this .stack_sizes entry describe" reduces to: follow the
// entry's relocation to its target symbol, read that symbol's *section*
// (not its name -- the target is often the anonymous marker, not the
// function symbol itself), and look up the one function symbol this object
// defines in that section. If a section ever contains more than one
// function symbol (this toolchain does not appear to produce that, but
// nothing guarantees it never will), attribution is ambiguous and the whole
// object is treated as unmeasurable, the same conservative direction as an
// archive or bitcode input in measureObjectDataSectionBytes.
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
// references) into `names`, across however many objects a core's
// link_files span -- the closure addObjectToStackGraph needs to tell a
// cross-object call from an ordinary extern data reference. Returns false
// if `path` isn't a plain relocatable object (archive, bitcode, missing
// file); `names` is unioned across every call regardless, so one
// unmeasurable object among several doesn't lose what the others defined.
bool collectDefinedFunctionNames(llvm::StringRef path,
                                 llvm::StringSet<> &names);

// Pass 2: parse `path`'s `.stack_sizes` entries and its functions' call/data
// relocations into `graph`, merging with whatever `graph` already contains
// (so a multi-object core builds one shared graph across all its
// link_files). `knownFunctions` must already contain every function name
// defined anywhere in the core's object set (see collectDefinedFunctionNames)
// so an undefined-here reference can be recognized as a cross-object call.
//
// Returns false if `path` cannot be fully attributed: not a plain
// relocatable object, a section shared by more than one function symbol, a
// `.stack_sizes` entry whose relocation doesn't resolve to a known
// function, or an unterminated/oversized ULEB128. `graph` may still contain
// partial data from what parsed successfully before the failure; callers
// must treat any `false` as "this whole object is unmeasurable", not merely
// skip the unattributed part.
bool addObjectToStackGraph(llvm::StringRef path,
                           const llvm::StringSet<> &knownFunctions,
                           StackGraph &graph);

// Wire in the conservative indirect-call edges recorded across however many
// objects contributed to `graph` (see the file comment), then normalize
// every callee list. Call once per core after all of its link_files objects
// have been added via addObjectToStackGraph, and before
// computeStackRequirement -- which relies on the normalization for
// reproducible diagnostics.
void resolveIndirectCallEdges(StackGraph &graph);

// Why a cycle and a missing measurement get different severities: a cycle
// means the requirement is fundamentally unbounded -- there is no safe
// number to assume, so this always demands an override. Missing
// `.stack_sizes` data, by contrast, is the same "haven't measured this
// artifact yet" gap reserved_data_size already treats as a warning (an
// archive, bitcode, or -- overwhelmingly the common case during rollout --
// a kernel object compiled before this analysis existed, or by a toolchain
// this analysis doesn't yet cover): failing every such core outright would
// break essentially every pre-existing design on first contact with a
// newer aiecc, for a gap that is often not even reachable at runtime.
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
// call-graph walk -- used to read back the MLIR-generated core object's own
// top-level frame (symbol `core_<col>_<row>`, see AIECoreToStandard.cpp)
// after it has been compiled, the one number StackGraph/computeStackRequirement
// above cannot see (it only knows about a core's *callees*, not the core
// body's own code). Returns nullopt if `path` is not a plain relocatable
// object or `symbol` has no `.stack_sizes` entry -- the same unmeasurable
// bucket as everywhere else in this analysis.
std::optional<int64_t> measureFunctionFrameSize(llvm::StringRef path,
                                                llvm::StringRef symbol);

} // namespace xilinx::aiecc

#endif // AIE_ANALYSIS_STACKSIZEANALYSIS_H
