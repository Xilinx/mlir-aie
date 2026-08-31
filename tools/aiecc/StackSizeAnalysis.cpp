//===- StackSizeAnalysis.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StackSizeAnalysis.h"

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

// Maps a section index to the one symbol of the requested type that this
// object defines in that section. Two such symbols in one section mark the
// section ambiguous, because this analysis attributes data by section index.
//
// A value is a graph key. A locally bound symbol is visible only inside its
// own object, so graphKeyFor qualifies its key with that object's path. Two
// objects can otherwise define unrelated symbols under one name. A globally
// bound symbol keeps its bare name, because a cross-object call resolves by
// name.
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

// An ELF symbol name never contains this byte, so a key that joins an object
// path and a locally bound symbol name never equals the bare name of a
// globally bound symbol.
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

// Returns the graph key of the function or data symbol that owns `sym`'s
// section. A relocation often targets an anonymous marker inside that
// section, so the owner is the symbol of interest. The result is empty when
// this object leaves `sym` undefined, or when the section is ambiguous.
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

// Returns the graph key that the definition of the target uses. A symbol this
// object leaves undefined is globally bound, so `bareName` is its key. A
// symbol this object defines resolves through its home section, which yields
// the object-qualified key of a locally bound target. An ambiguous section
// yields `bareName`.
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

// Strips the object path from a graph key. A diagnostic names the plain
// symbol.
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
  // An override ends the walk at this symbol. A recursive symbol, and a
  // symbol reached through a function pointer, become measurable this way:
  // the design declares the override on the kernel entry point it names, and
  // the walk stops above the internal symbol that the MLIR declaration cannot
  // name.
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

  // A frame size comes from an object file. A malformed input makes this sum
  // wrap negative, which undercounts the requirement.
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
      continue; // a declaration in this object
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
  // Every function this object defines gets a node, including one the core
  // never calls. A leaf function has a `.stack_sizes` entry and no call edge,
  // and its node holds that entry.
  for (auto &kv : funcs.bySection)
    if (!funcs.ambiguous.lookup(kv.first))
      graph.nodes.try_emplace(kv.second);

  bool ok = true;

  // A `.rela.X` section holds the relocations that apply to section `X`. The
  // name, the offsets and the executability of `X` drive the code below, so
  // each iteration resolves `X` through `getRelocatedSection()`.
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
      // Attributes each entry of the `.stack_sizes` section to its owning
      // function through the relocation at the address field of that entry.
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
        // section-relative one) has no symbol to resolve.
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
        // A frame size larger than a signed byte count is malformed. Taking
        // it would wrap negative and undercount the requirement.
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
        // Two objects can define one global name, for example through two
        // weak definitions. The linker picks one of them. Keeping the larger
        // frame overcounts at worst.
        int64_t &slot = graph.nodes[funcName].frameSize;
        slot = std::max(slot, static_cast<int64_t>(frameSize));
      }
      continue;
    }

    // Here `modified` holds code or data. A relocation target counts as a
    // function when this object defines it with ST_Function, or when
    // `knownFunctions` names it. The type of an undefined symbol is
    // unreliable, so the name test also matches an undefined data symbol that
    // shares a name with a function elsewhere, which overcounts.
    llvm::StringRef ownerName = funcs.lookup(modified.getIndex());
    bool modifiedIsText = modified.isText();
    for (const llvm::object::RelocationRef &rel : sec.relocations()) {
      llvm::object::symbol_iterator relSymIt = rel.getSymbol();
      if (relSymIt == obj.symbol_end())
        continue; // no symbol to attribute this relocation to
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
          continue; // no function owns this text section; record no edge
        if (isFunctionRef)
          graph.nodes[ownerName].callees.push_back(
              resolveReferenceKey(relSym, isUndefinedHere, relName, funcs, obj)
                  .str());
        else if (!homeIsKnownText)
          // A target outside the text sections of this object can name a
          // function-pointer table. Record it as a data reference;
          // resolveIndirectCallEdges turns it into a call edge once every
          // object has contributed.
          graph.dataReferences[ownerName].insert(
              resolveReferenceKey(relSym, isUndefinedHere, relName, dataSyms,
                                  obj)
                  .str());
      } else if (isFunctionRef) {
        // `modified` holds data and names a function, so the address of that
        // function escapes into the data. Key the record by the data symbol
        // that owns `modified`, which is the key the data references above
        // resolve to. A section index has no meaning across objects.
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
  // A callee list grows from hash-ordered maps, and names one symbol once per
  // call site and once per function-pointer table. Sorting and deduplicating
  // fixes the order of the walk, so the diagnostic names the same path on
  // every run, and bounds the search by the number of distinct callees.
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
