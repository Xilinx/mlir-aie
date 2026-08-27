//===- StackSizeAnalysis.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Analysis/StackSizeAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"

#include <algorithm>
#include <limits>
#include <utility>

using namespace xilinx::aiecc;

namespace {

// Find the one symbol of the requested type this object defines in section
// `secIdx`. Returns empty if none, and marks `ambiguous` if more than one --
// multiple same-typed symbols sharing a section breaks the section-index
// attribution this analysis relies on throughout.
//
// Values are graph keys, not raw names: a `static` symbol is only visible in
// its own object, so two objects can define an unrelated symbol under the
// same name -- a bare-name key would alias them. Such a symbol's key is
// qualified by its defining object's path (see graphKeyFor); a globally
// bound symbol keeps its bare name, since cross-object calls resolve by name.
struct SectionOwners {
  llvm::DenseMap<uint64_t, std::string> bySection;
  llvm::DenseMap<uint64_t, bool> ambiguous;

  llvm::StringRef lookup(uint64_t secIdx) const {
    if (ambiguous.lookup(secIdx))
      return {};
    auto it = bySection.find(secIdx);
    return it == bySection.end() ? llvm::StringRef()
                                 : llvm::StringRef(it->second);
  }
};

// No ELF symbol name can contain this byte (names are C strings), so
// prefixing a locally-bound symbol's key with its defining object's path can
// never collide with any globally-bound symbol's bare name.
constexpr char kLocalKeySep = '\x01';

std::string graphKeyFor(llvm::StringRef path, llvm::StringRef name,
                        bool isLocal) {
  if (!isLocal)
    return name.str();
  return (path + llvm::Twine(kLocalKeySep) + name).str();
}

SectionOwners buildSectionOwners(llvm::object::ObjectFile &obj,
                                 llvm::object::SymbolRef::Type wanted,
                                 llvm::StringRef path) {
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
    auto flagsOrErr = sym.getFlags();
    if (!flagsOrErr) {
      llvm::consumeError(flagsOrErr.takeError());
      continue;
    }
    bool isLocal = !(*flagsOrErr & llvm::object::SymbolRef::SF_Global);
    uint64_t secIdx = (*secOrErr)->getIndex();
    if (result.bySection.count(secIdx)) {
      result.ambiguous[secIdx] = true;
      continue;
    }
    result.bySection[secIdx] = graphKeyFor(path, *nameOrErr, isLocal);
  }
  return result;
}

// Resolve a relocation's target symbol to the graph key of the function (or
// data symbol) that owns its section (see the header comment: the target is
// often an anonymous marker in the same section, not the symbol itself).
// Empty means either undefined here or an ambiguous section.
llvm::StringRef resolveToOwningSymbol(const llvm::object::SymbolRef &sym,
                                      const SectionOwners &owners,
                                      llvm::object::ObjectFile &obj) {
  auto secOrErr = sym.getSection();
  if (!secOrErr) {
    llvm::consumeError(secOrErr.takeError());
    return {};
  }
  if (*secOrErr == obj.section_end())
    return {};
  return owners.lookup((*secOrErr)->getIndex());
}

// Key a reference (a direct call target, or a function/data-table touch for
// the indirect-call heuristic) by the same graph key its target's own
// definition was registered under: undefined here must be globally bound
// (only a global resolves across objects), so `bareName` is already right;
// defined here resolves through its home section instead, giving a
// locally-bound target its object-qualified key. Falls back to `bareName`
// when that section is ambiguous, same as before this resolution existed.
llvm::StringRef resolveReferenceKey(const llvm::object::SymbolRef &sym,
                                    bool isUndefinedHere,
                                    llvm::StringRef bareName,
                                    const SectionOwners &owners,
                                    llvm::object::ObjectFile &obj) {
  if (isUndefinedHere)
    return bareName;
  llvm::StringRef resolved = resolveToOwningSymbol(sym, owners, obj);
  return resolved.empty() ? bareName : resolved;
}

// Diagnostics must never show a path-qualified internal key to the user --
// strip back to the plain symbol name.
llvm::StringRef displayName(llvm::StringRef key) {
  size_t sep = key.find(kLocalKeySep);
  return sep == llvm::StringRef::npos ? key : key.substr(sep + 1);
}

enum class VisitState { Unvisited, InProgress, Done };

std::optional<int64_t>
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
    error = ("no stack size information for '" + displayName(sym).str() +
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
      cycle += displayName(s).str() + " -> ";
    cycle += displayName(sym).str();
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

  // Frames come from an object file, so a malformed or hostile input could
  // make this sum wrap negative and silently *undercount* -- the one
  // direction this analysis must never be wrong in.
  if (node.frameSize > std::numeric_limits<int64_t>::max() - best) {
    error = ("stack requirement for '" + displayName(sym).str() +
             "' overflows a signed 64-bit byte count; its object's "
             ".stack_sizes data is not believable");
    failureKind = StackRequirementFailure::Unmeasurable;
    return std::nullopt;
  }
  int64_t total = node.frameSize + best;
  memo[sym] = total;
  return total;
}

} // namespace

bool xilinx::aiecc::collectDefinedFunctionNames(llvm::StringRef path,
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

bool xilinx::aiecc::addObjectToStackGraph(
    llvm::StringRef path, const llvm::StringSet<> &knownFunctions,
    StackGraph &graph) {
  auto binOrErr = llvm::object::ObjectFile::createObjectFile(path);
  if (!binOrErr) {
    llvm::consumeError(binOrErr.takeError());
    return false;
  }
  llvm::object::ObjectFile &obj = *binOrErr->getBinary();
  if (!obj.isRelocatableObject())
    return false;

  SectionOwners funcs =
      buildSectionOwners(obj, llvm::object::SymbolRef::ST_Function, path);
  SectionOwners dataSyms =
      buildSectionOwners(obj, llvm::object::SymbolRef::ST_Data, path);
  // Every defined function gets a node, even one this core never calls --
  // harmless, and needed so a function with a `.stack_sizes` entry but zero
  // observed call edges (a leaf) still measures correctly.
  for (auto &kv : funcs.bySection)
    if (!funcs.ambiguous.lookup(kv.first))
      graph.nodes.try_emplace(kv.second);

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
      for (const llvm::object::RelocationRef &rel : sec.relocations()) {
        // A relocation against the null symbol index (R_*_NONE, or a purely
        // section-relative one) has no symbol to dereference.
        llvm::object::symbol_iterator relSym = rel.getSymbol();
        if (relSym == obj.symbol_end())
          continue;
        relocAtOffset.try_emplace(rel.getOffset(), *relSym);
      }

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
        // A frame that does not fit a signed byte count is malformed; taking
        // it would wrap negative and undercount.
        if (frameSize >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          ok = false;
          break;
        }

        auto relocIt = relocAtOffset.find(entryOffset);
        if (relocIt == relocAtOffset.end()) {
          ok = false;
          continue;
        }
        llvm::StringRef funcName =
            resolveToOwningSymbol(relocIt->second, funcs, obj);
        if (funcName.empty()) {
          ok = false;
          continue;
        }
        // Distinct local (`static`) functions no longer collide here (their
        // keys are qualified by defining object, see graphKeyFor), but two
        // objects can still define the same *global* name (e.g. two weak
        // definitions, where only one wins at link time) and this graph has
        // no way to know which. Keeping the larger frame can only overcount,
        // whereas letting the later object win could silently undercount
        // whichever definition actually links in.
        int64_t &slot = graph.nodes[funcName].frameSize;
        slot = std::max(slot, static_cast<int64_t>(frameSize));
      }
      continue;
    }

    // Otherwise: `modified` is the section these relocations patch --
    // executable (code referencing something) or data (something's address
    // stored here). For each relocation, decide whether its target is "a
    // function": reliably so if defined here with ST_Function, or (since an
    // undefined symbol's type is not trustworthy) if its name is in the
    // whole core's function-name closure. An undefined *data* symbol that
    // happens to share a name with a function defined elsewhere is therefore
    // treated as a call; that only ever adds an edge, so it overcounts.
    llvm::StringRef ownerName = funcs.lookup(modified.getIndex());
    bool modifiedIsText = modified.isText();
    for (const llvm::object::RelocationRef &rel : sec.relocations()) {
      llvm::object::symbol_iterator relSymIt = rel.getSymbol();
      if (relSymIt == obj.symbol_end())
        continue; // no symbol to attribute this relocation to.
      llvm::object::SymbolRef relSym = *relSymIt;
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
          graph.nodes[ownerName].callees.push_back(
              resolveReferenceKey(relSym, isUndefinedHere, relName, funcs, obj)
                  .str());
        else if (!homeIsKnownText)
          // Not a same-object branch-target label, and not (yet) known to
          // be a function -- record as a potential data reference in case
          // it later turns out to be a function-pointer table (resolved by
          // resolveIndirectCallEdges once every object has contributed).
          graph.dataReferences[ownerName].insert(
              resolveReferenceKey(relSym, isUndefinedHere, relName, dataSyms,
                                  obj)
                  .str());
      } else if (isFunctionRef) {
        // `modified` is data and the referenced symbol is a function: its
        // address escapes here. Key by the data symbol that owns `modified`
        // (the same key other code's data references above resolve
        // against), not by section index -- section indices aren't
        // meaningful across different objects.
        llvm::StringRef dataOwner = dataSyms.lookup(modified.getIndex());
        if (!dataOwner.empty())
          graph.dataEscapes[dataOwner].push_back(
              resolveReferenceKey(relSym, isUndefinedHere, relName, funcs, obj)
                  .str());
      }
    }
  }

  return ok;
}

void xilinx::aiecc::resolveIndirectCallEdges(StackGraph &graph) {
  for (auto &entry : graph.dataReferences) {
    llvm::StringRef funcName = entry.first();
    for (auto &dataEntry : entry.second) {
      auto it = graph.dataEscapes.find(dataEntry.first());
      if (it == graph.dataEscapes.end())
        continue;
      for (const std::string &escaped : it->second)
        graph.nodes[funcName].callees.push_back(escaped);
    }
  }
  // Callee lists are accumulated from hash-ordered maps and name the same
  // symbol once per call site, plus once per function-pointer table it
  // escapes through. Sorting and deduplicating makes the walk -- and so the
  // diagnostic naming the first offending path -- identical from run to run,
  // and keeps the search proportional to distinct callees rather than to
  // call sites.
  for (auto &node : graph.nodes) {
    auto &callees = node.second.callees;
    llvm::sort(callees);
    callees.erase(std::unique(callees.begin(), callees.end()), callees.end());
  }
}

StackRequirementResult xilinx::aiecc::computeStackRequirement(
    const StackGraph &graph, llvm::ArrayRef<llvm::StringRef> roots,
    const llvm::StringMap<int64_t> &overrides) {
  llvm::StringMap<VisitState> state;
  llvm::StringMap<int64_t> memo;
  int64_t best = 0;
  for (llvm::StringRef root : roots) {
    llvm::SmallVector<llvm::StringRef, 8> pathStack;
    std::string error;
    StackRequirementFailure failureKind = StackRequirementFailure::Unmeasurable;
    auto v = maxPathFrom(root, graph, overrides, state, memo, pathStack, error,
                         failureKind);
    if (!v)
      return {std::nullopt, std::move(error), failureKind};
    best = std::max(best, *v);
  }
  return {best, {}};
}

std::optional<int64_t>
xilinx::aiecc::measureFunctionFrameSize(llvm::StringRef path,
                                        llvm::StringRef symbol) {
  llvm::StringSet<> known;
  if (!collectDefinedFunctionNames(path, known))
    return std::nullopt;
  StackGraph graph;
  if (!addObjectToStackGraph(path, known, graph))
    return std::nullopt;
  auto it = graph.nodes.find(symbol);
  if (it == graph.nodes.end() || it->second.frameSize < 0)
    return std::nullopt;
  return it->second.frameSize;
}
