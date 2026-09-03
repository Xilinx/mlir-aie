//===- StackSizeAnalysis.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StackSizeAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"

#include <algorithm>
#include <limits>

using namespace xilinx::aiecc;
using namespace llvm::object;

namespace {

// The address ranges of one symbol type, sorted so that an address resolves to
// the symbol that owns it. A linked ELF gives every definition a unique
// address, so this analysis keys its nodes by address. An alias resolves to
// the range it shares.
struct SymbolRanges {
  struct Entry {
    uint64_t addr;
    uint64_t size;
    llvm::StringRef name;
  };
  llvm::SmallVector<Entry, 32> entries;

  const Entry *owner(uint64_t addr) const {
    auto it = llvm::upper_bound(
        entries, addr, [](uint64_t a, const Entry &e) { return a < e.addr; });
    if (it == entries.begin())
      return nullptr;
    --it;
    return addr < it->addr + it->size ? &*it : nullptr;
  }

  // Selects the relocations that are calls. A call targets a function's entry.
  // The linker also emits a relocation for a branch between the basic blocks of
  // one function. Such a branch targets an address inside the function.
  const Entry *startsAt(uint64_t addr) const {
    const Entry *e = owner(addr);
    return e && e->addr == addr ? e : nullptr;
  }
};

struct Node {
  int64_t frameSize = -1; // -1 marks a function with no `.stack_sizes` entry
  llvm::SmallVector<uint64_t, 4> callees;
};

struct Graph {
  llvm::DenseMap<uint64_t, Node> nodes;
  llvm::DenseMap<uint64_t, llvm::DenseSet<uint64_t>> dataReferences;
  llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 2>> dataEscapes;
  SymbolRanges funcs;
  llvm::DenseMap<uint64_t, int64_t> overridesByAddr;

  llvm::StringRef nameOf(uint64_t addr) const {
    const SymbolRanges::Entry *e = funcs.owner(addr);
    return e ? e->name : llvm::StringRef("<unknown>");
  }
};

// Collects the address ranges of every symbol of type `wanted`. `addrByName`
// records each symbol's address under its name. An override names a function,
// so that map resolves it, and it resolves an alias of that function too.
SymbolRanges collectRanges(ObjectFile &obj, SymbolRef::Type wanted,
                           llvm::StringMap<uint64_t> *addrByName = nullptr) {
  SymbolRanges ranges;
  for (const SymbolRef &sym : obj.symbols()) {
    auto type = sym.getType();
    auto addr = sym.getAddress();
    auto name = sym.getName();
    if (!type || !addr || !name) {
      llvm::consumeError(type.takeError());
      llvm::consumeError(addr.takeError());
      llvm::consumeError(name.takeError());
      continue;
    }
    if (*type != wanted)
      continue;
    if (addrByName)
      addrByName->try_emplace(*name, *addr);
    if (uint64_t size = ELFSymbolRef(sym).getSize())
      ranges.entries.push_back({*addr, size, *name});
  }
  llvm::sort(ranges.entries,
             [](const auto &a, const auto &b) { return a.addr < b.addr; });
  return ranges;
}

// Reads the `.stack_sizes` entries, each an address followed by a ULEB128
// frame size. The linker resolves the addresses in place, so each entry names
// its function directly.
bool readFrameSizes(ObjectFile &obj, SectionRef sec, Graph &graph) {
  auto contents = sec.getContents();
  if (!contents) {
    llvm::consumeError(contents.takeError());
    return false;
  }
  const unsigned addrSize = obj.getBytesInAddress();
  const llvm::endianness order =
      obj.isLittleEndian() ? llvm::endianness::little : llvm::endianness::big;
  const auto *cursor = reinterpret_cast<const uint8_t *>(contents->data());
  const uint8_t *end = cursor + contents->size();
  while (cursor < end) {
    if (static_cast<size_t>(end - cursor) < addrSize)
      return false;
    uint64_t funcAddr =
        addrSize == 8 ? llvm::support::endian::read<uint64_t>(cursor, order)
                      : llvm::support::endian::read<uint32_t>(cursor, order);
    cursor += addrSize;
    unsigned lebLen = 0;
    const char *lebErr = nullptr;
    uint64_t frameSize = llvm::decodeULEB128(cursor, &lebLen, end, &lebErr);
    if (lebErr || lebLen == 0)
      return false;
    cursor += lebLen;
    // A frame size past a signed byte count is malformed. Taking it would wrap
    // negative and undercount the requirement.
    if (frameSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return false;
    if (const SymbolRanges::Entry *fn = graph.funcs.owner(funcAddr))
      graph.nodes[fn->addr].frameSize = std::max<int64_t>(
          graph.nodes[fn->addr].frameSize, static_cast<int64_t>(frameSize));
  }
  return true;
}

// Records one call edge, or one half of the function-pointer heuristic, per
// relocation. `patched` is the address the relocation writes, `target` the
// address it writes there.
void addEdge(Graph &graph, const SymbolRanges &data, bool patchedIsText,
             uint64_t patched, uint64_t target) {
  const SymbolRanges::Entry *targetFunc = graph.funcs.startsAt(target);
  if (patchedIsText) {
    const SymbolRanges::Entry *owner = graph.funcs.owner(patched);
    if (!owner)
      return;
    if (targetFunc)
      graph.nodes[owner->addr].callees.push_back(targetFunc->addr);
    else if (const SymbolRanges::Entry *targetData = data.owner(target))
      // A data object that a function reads can hold a function-pointer table.
      graph.dataReferences[owner->addr].insert(targetData->addr);
    return;
  }
  // The address of a function escapes into data, so any function that reads
  // that data may call it.
  if (targetFunc)
    if (const SymbolRanges::Entry *owner = data.owner(patched))
      graph.dataEscapes[owner->addr].push_back(targetFunc->addr);
}

enum class VisitState { Unvisited, InProgress, Done };

std::optional<int64_t> maxPathFrom(uint64_t sym, const Graph &graph,
                                   llvm::DenseMap<uint64_t, VisitState> &state,
                                   llvm::DenseMap<uint64_t, int64_t> &memo,
                                   llvm::SmallVectorImpl<uint64_t> &pathStack,
                                   std::string &error,
                                   StackRequirementFailure &failureKind,
                                   llvm::DenseSet<uint64_t> &unmeasured) {
  // An override ends the walk at this function. A recursive function, and a
  // function reached through a function pointer, become measurable this way:
  // the design declares the override on the kernel entry point it names, and
  // the walk stops above the internal function that the MLIR declaration
  // cannot name.
  if (auto it = graph.overridesByAddr.find(sym);
      it != graph.overridesByAddr.end())
    return it->second;

  if (auto it = memo.find(sym); it != memo.end())
    return it->second;

  auto nodeIt = graph.nodes.find(sym);
  // A frame the ELF holds no entry for counts as 0, which makes the total a
  // lower bound. The walk continues, so one such function costs only its own
  // frame. Peano's aie2 crt1.o carries no `.stack_sizes` at all, so this keeps
  // the check alive on that target.
  int64_t frameSize = 0;
  if (nodeIt == graph.nodes.end() || nodeIt->second.frameSize < 0)
    unmeasured.insert(sym);
  else
    frameSize = nodeIt->second.frameSize;
  static const Node emptyNode;
  const Node &node = nodeIt == graph.nodes.end() ? emptyNode : nodeIt->second;

  VisitState &st = state[sym];
  if (st == VisitState::InProgress) {
    std::string cycle;
    for (uint64_t s : pathStack)
      cycle += graph.nameOf(s).str() + " -> ";
    cycle += graph.nameOf(sym).str();
    error = "recursion detected: " + cycle;
    failureKind = StackRequirementFailure::Cycle;
    return std::nullopt;
  }

  st = VisitState::InProgress;
  pathStack.push_back(sym);
  int64_t best = 0;
  for (uint64_t callee : node.callees) {
    auto sub = maxPathFrom(callee, graph, state, memo, pathStack, error,
                           failureKind, unmeasured);
    if (!sub)
      return std::nullopt;
    best = std::max(best, *sub);
  }
  pathStack.pop_back();
  state[sym] = VisitState::Done;

  // A frame size comes from the ELF. A malformed input makes this sum wrap
  // negative, which undercounts the requirement.
  if (frameSize > std::numeric_limits<int64_t>::max() - best) {
    error = ("stack requirement for '" + graph.nameOf(sym).str() +
             "' overflows a signed 64-bit byte count; the core's .stack_sizes "
             "data is not believable");
    failureKind = StackRequirementFailure::Unmeasurable;
    return std::nullopt;
  }
  int64_t total = frameSize + best;
  memo[sym] = total;
  return total;
}

StackRequirementResult
fail(std::string error,
     StackRequirementFailure kind = StackRequirementFailure::Unmeasurable) {
  return {std::nullopt, std::move(error), kind};
}

} // namespace

StackRequirementResult xilinx::aiecc::computeStackRequirement(
    llvm::StringRef elfPath, const llvm::StringMap<int64_t> &overrides) {
  auto binOrErr = ObjectFile::createObjectFile(elfPath);
  if (!binOrErr) {
    llvm::consumeError(binOrErr.takeError());
    return fail("cannot read the linked core '" + elfPath.str() + "'");
  }
  ObjectFile &obj = *binOrErr->getBinary();

  auto entry = obj.getStartAddress();
  if (!entry) {
    llvm::consumeError(entry.takeError());
    return fail("the linked core '" + elfPath.str() + "' has no entry point");
  }

  Graph graph;
  llvm::StringMap<uint64_t> funcAddrByName;
  graph.funcs = collectRanges(obj, SymbolRef::ST_Function, &funcAddrByName);
  SymbolRanges data = collectRanges(obj, SymbolRef::ST_Data);
  for (const auto &kv : overrides)
    if (auto it = funcAddrByName.find(kv.first()); it != funcAddrByName.end())
      graph.overridesByAddr[it->second] = kv.second;

  bool complete = true;
  for (const SectionRef &sec : obj.sections()) {
    auto name = sec.getName();
    if (!name) {
      llvm::consumeError(name.takeError());
      continue;
    }
    if (*name == ".stack_sizes") {
      complete &= readFrameSizes(obj, sec, graph);
      continue;
    }
    if (sec.relocation_begin() == sec.relocation_end())
      continue;
    // A `.rela.X` section holds the relocations that apply to section `X`, so
    // the executability of `X` says whether a relocation patches code or data.
    auto patchedSec = sec.getRelocatedSection();
    if (!patchedSec) {
      llvm::consumeError(patchedSec.takeError());
      continue;
    }
    if (*patchedSec == obj.section_end())
      continue;
    bool patchedIsText = (*patchedSec)->isText();
    for (const RelocationRef &rel : sec.relocations()) {
      symbol_iterator target = rel.getSymbol();
      if (target == obj.symbol_end())
        continue; // no symbol to attribute this relocation to
      auto targetAddr = target->getAddress();
      if (!targetAddr) {
        llvm::consumeError(targetAddr.takeError());
        continue;
      }
      addEdge(graph, data, patchedIsText, rel.getOffset(), *targetAddr);
    }
  }

  for (const auto &ref : graph.dataReferences)
    for (uint64_t dataAddr : ref.second)
      if (auto it = graph.dataEscapes.find(dataAddr);
          it != graph.dataEscapes.end())
        llvm::append_range(graph.nodes[ref.first].callees, it->second);
  // A callee list grows from hash-ordered maps. It names one function once per
  // call site and once per function-pointer table. This loop fixes the order of
  // the walk, so the diagnostic names the same path on every run. It also
  // bounds the search by the number of distinct callees.
  for (auto &node : graph.nodes) {
    auto &callees = node.second.callees;
    llvm::sort(callees);
    callees.erase(std::unique(callees.begin(), callees.end()), callees.end());
  }

  const SymbolRanges::Entry *root = graph.funcs.owner(*entry);
  if (!root)
    return fail("the linked core '" + elfPath.str() +
                "' declares no function at its entry point");
  // The entry point establishes SP. The stack this measures starts there, so
  // the entry point's own frame counts as 0.
  Node &rootNode = graph.nodes[root->addr];
  if (rootNode.frameSize < 0)
    rootNode.frameSize = 0;
  if (!complete)
    return fail("the linked core '" + elfPath.str() +
                "' carries malformed .stack_sizes data");

  llvm::DenseMap<uint64_t, VisitState> state;
  llvm::DenseMap<uint64_t, int64_t> memo;
  llvm::SmallVector<uint64_t, 8> pathStack;
  llvm::DenseSet<uint64_t> unmeasured;
  std::string error;
  StackRequirementFailure failureKind = StackRequirementFailure::Unmeasurable;
  auto bytes = maxPathFrom(root->addr, graph, state, memo, pathStack, error,
                           failureKind, unmeasured);
  if (!bytes)
    return {std::nullopt, std::move(error), failureKind};

  StackRequirementResult result{*bytes, {}, failureKind, {}};
  for (uint64_t addr : unmeasured)
    result.unmeasured.push_back(graph.nameOf(addr).str());
  llvm::sort(result.unmeasured);
  return result;
}
