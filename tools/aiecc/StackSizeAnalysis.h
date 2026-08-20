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

#ifndef AIECC_STACKSIZEANALYSIS_H
#define AIECC_STACKSIZEANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

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

namespace detail {

// Find the one symbol of the requested type this object defines in section
// `secIdx`. Returns empty if none, and marks `ambiguous` if more than one --
// multiple same-typed symbols sharing a section breaks the section-index
// attribution this analysis relies on throughout.
struct SectionOwners {
  llvm::DenseMap<uint64_t, llvm::StringRef> bySection;
  llvm::DenseMap<uint64_t, bool> ambiguous;

  llvm::StringRef lookup(uint64_t secIdx) const {
    if (ambiguous.lookup(secIdx))
      return {};
    auto it = bySection.find(secIdx);
    return it == bySection.end() ? llvm::StringRef() : it->second;
  }
};

inline SectionOwners buildSectionOwners(llvm::object::ObjectFile &obj,
                                        llvm::object::SymbolRef::Type wanted) {
  SectionOwners result;
  for (const llvm::object::SymbolRef &sym : obj.symbols()) {
    auto typeOrErr = sym.getType();
    if (!typeOrErr) {
      llvm::consumeError(typeOrErr.takeError());
      continue;
    }
    if (*typeOrErr != wanted)
      continue;
    auto secOrErr = sym.getSection();
    if (!secOrErr) {
      llvm::consumeError(secOrErr.takeError());
      continue;
    }
    if (*secOrErr == obj.section_end())
      continue;
    auto nameOrErr = sym.getName();
    if (!nameOrErr) {
      llvm::consumeError(nameOrErr.takeError());
      continue;
    }
    uint64_t secIdx = (*secOrErr)->getIndex();
    if (result.bySection.count(secIdx)) {
      result.ambiguous[secIdx] = true;
      continue;
    }
    result.bySection[secIdx] = *nameOrErr;
  }
  return result;
}

// Resolve a relocation's target symbol to the function symbol that owns its
// section (see the file comment: the target is often an anonymous marker in
// the same section as the actual function, not the function symbol itself).
inline llvm::StringRef
resolveToOwningFunction(const llvm::object::SymbolRef &sym,
                        const SectionOwners &funcs,
                        llvm::object::ObjectFile &obj) {
  auto secOrErr = sym.getSection();
  if (!secOrErr) {
    llvm::consumeError(secOrErr.takeError());
    return {};
  }
  if (*secOrErr == obj.section_end())
    return {};
  return funcs.lookup((*secOrErr)->getIndex());
}

} // namespace detail

// Pass 1: collect every function symbol `path` *defines* (not merely
// references) into `names`, across however many objects a core's
// link_files span -- the closure addObjectToStackGraph needs to tell a
// cross-object call from an ordinary extern data reference. Returns false
// if `path` isn't a plain relocatable object (archive, bitcode, missing
// file); `names` is unioned across every call regardless, so one
// unmeasurable object among several doesn't lose what the others defined.
inline bool collectDefinedFunctionNames(llvm::StringRef path,
                                        llvm::StringSet<> &names) {
  auto binOrErr = llvm::object::ObjectFile::createObjectFile(path);
  if (!binOrErr) {
    llvm::consumeError(binOrErr.takeError());
    return false;
  }
  llvm::object::ObjectFile &obj = *binOrErr->getBinary();
  if (!obj.isRelocatableObject())
    return false;
  for (const llvm::object::SymbolRef &sym : obj.symbols()) {
    auto typeOrErr = sym.getType();
    if (!typeOrErr) {
      llvm::consumeError(typeOrErr.takeError());
      continue;
    }
    if (*typeOrErr != llvm::object::SymbolRef::ST_Function)
      continue;
    auto secOrErr = sym.getSection();
    if (!secOrErr) {
      llvm::consumeError(secOrErr.takeError());
      continue;
    }
    if (*secOrErr == obj.section_end())
      continue; // still just a declaration in this object.
    auto nameOrErr = sym.getName();
    if (!nameOrErr) {
      llvm::consumeError(nameOrErr.takeError());
      continue;
    }
    names.insert(*nameOrErr);
  }
  return true;
}

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
// function, or an unterminated ULEB128. `graph` may still contain partial
// data from what parsed successfully before the failure; callers must treat
// any `false` as "this whole object is unmeasurable", not merely skip the
// unattributed part.
inline bool addObjectToStackGraph(llvm::StringRef path,
                                  const llvm::StringSet<> &knownFunctions,
                                  StackGraph &graph) {
  auto binOrErr = llvm::object::ObjectFile::createObjectFile(path);
  if (!binOrErr) {
    llvm::consumeError(binOrErr.takeError());
    return false;
  }
  llvm::object::ObjectFile &obj = *binOrErr->getBinary();
  if (!obj.isRelocatableObject())
    return false;

  detail::SectionOwners funcs =
      detail::buildSectionOwners(obj, llvm::object::SymbolRef::ST_Function);
  detail::SectionOwners dataSyms =
      detail::buildSectionOwners(obj, llvm::object::SymbolRef::ST_Data);
  // Every defined function gets a node, even one this core never calls --
  // harmless, and needed so a function with a `.stack_sizes` entry but zero
  // observed call edges (a leaf) still measures correctly.
  for (auto &kv : funcs.bySection)
    if (!funcs.ambiguous.lookup(kv.first))
      graph.nodes.try_emplace(kv.second.str());

  bool ok = true;

  // Relocations are entries of a separate `.rela.X` section, not of the `X`
  // section they modify -- `sec.relocations()` is only non-empty when `sec`
  // itself is that relocation-holding section, and `sec.getRelocatedSection()`
  // is what maps it back to `X` (its offsets, name, and executability all
  // belong to `X`, not to `sec`). So the outer loop below walks every
  // section looking for ones that hold relocations at all, then dispatches
  // on what section those relocations modify.
  for (const llvm::object::SectionRef &sec : obj.sections()) {
    if (sec.relocation_begin() == sec.relocation_end())
      continue;
    auto modifiedOrErr = sec.getRelocatedSection();
    if (!modifiedOrErr) {
      llvm::consumeError(modifiedOrErr.takeError());
      ok = false;
      continue;
    }
    if (*modifiedOrErr == obj.section_end())
      continue;
    llvm::object::SectionRef modified = **modifiedOrErr;
    auto modifiedNameOrErr = modified.getName();
    if (!modifiedNameOrErr) {
      llvm::consumeError(modifiedNameOrErr.takeError());
      ok = false;
      continue;
    }

    if (*modifiedNameOrErr == ".stack_sizes") {
      // Attribute each entry in the modified `.stack_sizes` section to its
      // owning function, via the relocation at that entry's address field.
      auto contentsOrErr = modified.getContents();
      if (!contentsOrErr) {
        llvm::consumeError(contentsOrErr.takeError());
        ok = false;
        continue;
      }
      llvm::StringRef contents = *contentsOrErr;
      llvm::DenseMap<uint64_t, llvm::object::SymbolRef> relocAtOffset;
      for (const llvm::object::RelocationRef &rel : sec.relocations())
        relocAtOffset.try_emplace(rel.getOffset(), *rel.getSymbol());

      const unsigned addrSize = obj.getBytesInAddress();
      const auto *base = reinterpret_cast<const uint8_t *>(contents.data());
      const uint8_t *end = base + contents.size();
      size_t offset = 0;
      while (offset < contents.size()) {
        if (offset + addrSize > contents.size()) {
          ok = false;
          break;
        }
        uint64_t entryOffset = offset;
        offset += addrSize;
        unsigned lebLen = 0;
        const char *lebErr = nullptr;
        uint64_t frameSize =
            llvm::decodeULEB128(base + offset, &lebLen, end, &lebErr);
        if (lebErr || lebLen == 0) {
          ok = false;
          break;
        }
        offset += lebLen;

        auto relocIt = relocAtOffset.find(entryOffset);
        if (relocIt == relocAtOffset.end()) {
          ok = false;
          continue;
        }
        llvm::StringRef funcName =
            detail::resolveToOwningFunction(relocIt->second, funcs, obj);
        if (funcName.empty()) {
          ok = false;
          continue;
        }
        graph.nodes[funcName].frameSize = static_cast<int64_t>(frameSize);
      }
      continue;
    }

    // Otherwise: `modified` is the section these relocations patch --
    // executable (code referencing something) or data (something's address
    // stored here). For each relocation, decide whether its target is "a
    // function": reliably so if defined here with ST_Function, or (since an
    // undefined symbol's type is not trustworthy) if its name is in the
    // whole core's function-name closure.
    llvm::StringRef ownerName = funcs.lookup(modified.getIndex());
    bool modifiedIsText = modified.isText();
    for (const llvm::object::RelocationRef &rel : sec.relocations()) {
      llvm::object::SymbolRef relSym = *rel.getSymbol();
      auto nameOrErr = relSym.getName();
      if (!nameOrErr) {
        llvm::consumeError(nameOrErr.takeError());
        continue;
      }
      llvm::StringRef relName = *nameOrErr;

      auto typeOrErr = relSym.getType();
      bool typeIsFunction =
          typeOrErr && *typeOrErr == llvm::object::SymbolRef::ST_Function;
      if (!typeOrErr)
        llvm::consumeError(typeOrErr.takeError());

      auto homeSecOrErr = relSym.getSection();
      bool isUndefinedHere =
          !homeSecOrErr || *homeSecOrErr == obj.section_end();
      if (!homeSecOrErr)
        llvm::consumeError(homeSecOrErr.takeError());
      bool homeIsKnownText = !isUndefinedHere && (*homeSecOrErr)->isText();

      bool isFunctionRef = typeIsFunction || (isUndefinedHere &&
                                              knownFunctions.contains(relName));

      if (modifiedIsText) {
        if (ownerName.empty())
          continue; // relocation in a text section this object doesn't own
                    // a function for (shouldn't happen; be conservative
                    // and simply not record an edge rather than guess).
        if (isFunctionRef)
          graph.nodes[ownerName].callees.push_back(relName.str());
        else if (!homeIsKnownText)
          // Not a same-object branch-target label, and not (yet) known to
          // be a function -- record as a potential data reference in case
          // it later turns out to be a function-pointer table (resolved by
          // resolveIndirectCallEdges once every object has contributed).
          graph.dataReferences[ownerName].insert(relName.str());
      } else if (isFunctionRef) {
        // `modified` is data and the referenced symbol is a function: its
        // address escapes here. Key by the data symbol that owns `modified`
        // (the same name other code's data references above resolve
        // against), not by section index -- section indices aren't
        // meaningful across different objects.
        llvm::StringRef dataOwner = dataSyms.lookup(modified.getIndex());
        if (!dataOwner.empty())
          graph.dataEscapes[dataOwner].push_back(relName.str());
      }
    }
  }

  return ok;
}

// Wire in the conservative indirect-call edges recorded across however many
// objects contributed to `graph` (see the file comment) -- call once per
// core after every one of its link_files objects has been added via
// addObjectToStackGraph, before computeStackRequirement.
inline void resolveIndirectCallEdges(StackGraph &graph) {
  for (auto &entry : graph.dataReferences) {
    llvm::StringRef funcName = entry.first();
    for (auto &dataEntry : entry.second) {
      llvm::StringRef dataName = dataEntry.first();
      auto it = graph.dataEscapes.find(dataName);
      if (it == graph.dataEscapes.end())
        continue;
      for (const std::string &escaped : it->second)
        graph.nodes[funcName].callees.push_back(escaped);
    }
  }
}

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

namespace detail {

enum class VisitState { Unvisited, InProgress, Done };

inline std::optional<int64_t>
maxPathFrom(llvm::StringRef sym, const StackGraph &graph,
            const llvm::StringMap<int64_t> &overrides,
            llvm::StringMap<VisitState> &state, llvm::StringMap<int64_t> &memo,
            llvm::SmallVectorImpl<llvm::StringRef> &pathStack,
            std::string &error, StackRequirementFailure &failureKind) {
  // An override cuts the subtree here: the analysis never looks past a
  // symbol the user has explicitly sized, mirroring reserved_data_size's
  // "explicit skips measurement entirely" rule. This is also how a
  // recursive/indirectly-called symbol becomes resolvable at all: the user
  // only ever needs to override the kernel entry point they already
  // declared, not whatever internal symbol MLIR never saw.
  if (auto it = overrides.find(sym); it != overrides.end())
    return it->second;

  if (auto it = memo.find(sym); it != memo.end())
    return it->second;

  auto nodeIt = graph.nodes.find(sym);
  if (nodeIt == graph.nodes.end() || nodeIt->second.frameSize < 0) {
    error = ("no stack size information for '" + sym.str() +
             "' -- it is called (directly, or conservatively through a "
             "function-pointer reference) but its defining object was not "
             "measurable (missing .stack_sizes, an archive/bitcode input, "
             "or not part of this core's link_files)");
    failureKind = StackRequirementFailure::Unmeasurable;
    return std::nullopt;
  }
  const StackGraphNode &node = nodeIt->second;

  VisitState &st = state[sym];
  if (st == VisitState::InProgress) {
    std::string cycle;
    for (llvm::StringRef s : pathStack)
      cycle += s.str() + " -> ";
    cycle += sym.str();
    error = "recursion detected: " + cycle;
    failureKind = StackRequirementFailure::Cycle;
    return std::nullopt;
  }

  st = VisitState::InProgress;
  pathStack.push_back(sym);
  int64_t best = 0;
  for (const std::string &callee : node.callees) {
    auto sub = maxPathFrom(callee, graph, overrides, state, memo, pathStack,
                           error, failureKind);
    if (!sub)
      return std::nullopt;
    best = std::max(best, *sub);
  }
  pathStack.pop_back();
  state[sym] = VisitState::Done;

  int64_t total = node.frameSize + best;
  memo[sym] = total;
  return total;
}

} // namespace detail

// Compute a core's stack requirement as the max over `roots` (the symbols
// directly called from the core body) of each root's longest
// frame-size-weighted root-to-leaf path in `graph`. `overrides` names
// symbols (typically the roots themselves -- the kernel entry points MLIR
// declared) whose subtree is a declared leaf rather than analyzed.
inline StackRequirementResult
computeStackRequirement(const StackGraph &graph,
                        llvm::ArrayRef<llvm::StringRef> roots,
                        const llvm::StringMap<int64_t> &overrides) {
  llvm::StringMap<detail::VisitState> state;
  llvm::StringMap<int64_t> memo;
  int64_t best = 0;
  for (llvm::StringRef root : roots) {
    llvm::SmallVector<llvm::StringRef, 8> pathStack;
    std::string error;
    StackRequirementFailure failureKind = StackRequirementFailure::Unmeasurable;
    auto v = detail::maxPathFrom(root, graph, overrides, state, memo, pathStack,
                                 error, failureKind);
    if (!v)
      return {std::nullopt, std::move(error), failureKind};
    best = std::max(best, *v);
  }
  return {best, {}};
}

} // namespace xilinx::aiecc

#endif // AIECC_STACKSIZEANALYSIS_H
