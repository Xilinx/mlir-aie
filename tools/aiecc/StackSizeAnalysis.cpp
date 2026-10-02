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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <algorithm>
#include <limits>
#include <tuple>

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
    if (it == entries.begin()) {
      return nullptr;
    }
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

  std::string describe(uint64_t addr) const {
    return nameOf(addr).str() + "@0x" + llvm::utohexstr(addr);
  }
};

// Collects the address ranges of every symbol of type `wanted`. `addrByName`
// maps a name to an address. A `stack_size_override` arrives from the MLIR
// under a function name. This analysis keys its nodes by address. That map
// bridges the two.
//
// It holds only globally bindable symbols. A declaration in the MLIR binds by
// the linker's rules, and a `static` function of the same name belongs to the
// object that defines it. It holds unsized symbols too, so an override on an
// alias reaches the node the alias shares.
SymbolRanges collectRanges(ObjectFile &obj, SymbolRef::Type wanted,
                           llvm::StringMap<uint64_t> *addrByName = nullptr) {
  SymbolRanges ranges;
  for (const SymbolRef &sym : obj.symbols()) {
    auto type = sym.getType();
    auto addr = sym.getAddress();
    auto name = sym.getName();
    auto flags = sym.getFlags();
    if (!type || !addr || !name || !flags) {
      llvm::consumeError(type.takeError());
      llvm::consumeError(addr.takeError());
      llvm::consumeError(name.takeError());
      llvm::consumeError(flags.takeError());
      continue;
    }
    if (*type != wanted) {
      continue;
    }
    if (addrByName &&
        (*flags & (SymbolRef::SF_Global | SymbolRef::SF_Weak)) != 0) {
      addrByName->try_emplace(*name, *addr);
    }
    if (uint64_t size = ELFSymbolRef(sym).getSize()) {
      ranges.entries.push_back({*addr, size, *name});
    }
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
    if (static_cast<size_t>(end - cursor) < addrSize) {
      return false;
    }
    uint64_t funcAddr =
        addrSize == 8 ? llvm::support::endian::read<uint64_t>(cursor, order)
                      : llvm::support::endian::read<uint32_t>(cursor, order);
    cursor += addrSize;
    unsigned lebLen = 0;
    const char *lebErr = nullptr;
    uint64_t frameSize = llvm::decodeULEB128(cursor, &lebLen, end, &lebErr);
    if (lebErr || lebLen == 0) {
      return false;
    }
    cursor += lebLen;
    // A frame size past a signed byte count is malformed. Taking it would wrap
    // negative and undercount the requirement.
    if (frameSize >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    if (const SymbolRanges::Entry *fn = graph.funcs.owner(funcAddr)) {
      graph.nodes[fn->addr].frameSize = std::max<int64_t>(
          graph.nodes[fn->addr].frameSize, static_cast<int64_t>(frameSize));
    }
  }
  return true;
}

bool isZeroSizedFunctionSymbol(const SymbolRef &sym) {
  auto type = sym.getType();
  if (!type) {
    llvm::consumeError(type.takeError());
    return false;
  }
  return *type == SymbolRef::ST_Function && ELFSymbolRef(sym).getSize() == 0;
}

bool isAieDataWordRelocation(const ObjectFile &obj, const RelocationRef &rel) {
  // These ABI constants live in Peano's ELF.h, not the host LLVM headers.
  constexpr unsigned aieElfMachine = 264;
  constexpr unsigned aieElfFlagMask = 0x7;
  constexpr unsigned aie1ElfFlag = 0x1;
  constexpr unsigned aie2ElfFlag = 0x2;
  constexpr unsigned aie2pElfFlag = 0x3;
  constexpr unsigned aie2psElfFlag = 0x4;
  constexpr uint64_t aie1DataWordRelocation = 72;
  constexpr uint64_t aie2DataWordRelocation = 50;
  constexpr uint64_t aie2pDataWordRelocation = 62;
  constexpr uint64_t aie2psDataWordRelocation = 135;

  const auto *elf = llvm::dyn_cast<ELFObjectFileBase>(&obj);
  if (!elf || elf->getEMachine() != aieElfMachine) {
    return false;
  }
  // Peano uses one relocation number per AIE variant for the plain 32-bit
  // address literal (`FK_Data_4`). That literal is not a call, even in `.text`.
  switch (elf->getPlatformFlags() & aieElfFlagMask) {
  case aie1ElfFlag:
    return rel.getType() == aie1DataWordRelocation;
  case aie2ElfFlag:
    return rel.getType() == aie2DataWordRelocation;
  case aie2pElfFlag:
    return rel.getType() == aie2pDataWordRelocation;
  case aie2psElfFlag:
    return rel.getType() == aie2psDataWordRelocation;
  default:
    return false;
  }
}

bool isCallLikeRelocation(const ObjectFile &obj, const RelocationRef &rel) {
  if (isAieDataWordRelocation(obj, rel)) {
    return false;
  }
  llvm::SmallString<32> typeNameStorage;
  rel.getTypeName(typeNameStorage);
  llvm::StringRef typeName(typeNameStorage);
  if (typeName.contains("CALL") || typeName.contains("JUMP") ||
      typeName.contains("BRANCH") || typeName.contains("PLT32")) {
    return true;
  }
  if (typeName == "R_X86_64_64" || typeName == "R_X86_64_32" ||
      typeName == "R_X86_64_32S" || typeName.contains("ABS") ||
      typeName.contains("ADDR")) {
    return false;
  }
  // Keep unknown relocation kinds conservative: dropping them can disconnect
  // the call graph and undercount the stack requirement.
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
    if (!owner) {
      return;
    }
    if (targetFunc) {
      graph.nodes[owner->addr].callees.push_back(targetFunc->addr);
    } else if (const SymbolRanges::Entry *targetData = data.owner(target)) {
      // A data object that a function reads can hold a function-pointer table.
      graph.dataReferences[owner->addr].insert(targetData->addr);
    }
    return;
  }
  // The address of a function escapes into data, so any function that reads
  // that data may call it.
  if (targetFunc) {
    if (const SymbolRanges::Entry *owner = data.owner(patched)) {
      graph.dataEscapes[owner->addr].push_back(targetFunc->addr);
    }
  }
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
      it != graph.overridesByAddr.end()) {
    return it->second;
  }

  if (auto it = memo.find(sym); it != memo.end()) {
    return it->second;
  }

  auto nodeIt = graph.nodes.find(sym);
  // A frame the ELF does not report counts as 0, which makes the total a lower
  // bound. The walk continues, so one such function costs only its own frame.
  int64_t frameSize = 0;
  if (nodeIt != graph.nodes.end() && nodeIt->second.frameSize >= 0) {
    frameSize = nodeIt->second.frameSize;
  } else {
    unmeasured.insert(sym);
  }
  static const Node emptyNode;
  const Node &node = nodeIt == graph.nodes.end() ? emptyNode : nodeIt->second;

  VisitState &st = state[sym];
  if (st == VisitState::InProgress) {
    std::string cycle;
    for (uint64_t s : pathStack) {
      cycle += graph.describe(s) + " -> ";
    }
    cycle += graph.describe(sym);
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
    if (!sub) {
      return std::nullopt;
    }
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
  for (const auto &kv : overrides) {
    if (auto it = funcAddrByName.find(kv.first()); it != funcAddrByName.end()) {
      graph.overridesByAddr[it->second] = kv.second;
    }
  }

  bool complete = true;
  bool sawRelocations = false;
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
    if (sec.relocation_begin() == sec.relocation_end()) {
      continue;
    }
    sawRelocations = true;
    // A `.rela.X` section holds the relocations that apply to section `X`, so
    // the executability of `X` says whether a relocation patches code or data.
    auto patchedSec = sec.getRelocatedSection();
    if (!patchedSec) {
      llvm::consumeError(patchedSec.takeError());
      continue;
    }
    if (*patchedSec == obj.section_end()) {
      continue;
    }
    bool patchedIsText = (*patchedSec)->isText();
    for (const RelocationRef &rel : sec.relocations()) {
      symbol_iterator target = rel.getSymbol();
      if (target == obj.symbol_end()) {
        continue; // no symbol to attribute this relocation to
      }
      // A fully inlined entry point can survive the link as a zero-sized FUNC
      // symbol at another function's address. Only call-like relocations should
      // close a call edge through that alias; an address constant in .text
      // would otherwise invent a callee that the code never executes.
      if (patchedIsText && isZeroSizedFunctionSymbol(*target) &&
          !isCallLikeRelocation(obj, rel)) {
        continue;
      }
      auto targetAddr = target->getAddress();
      if (!targetAddr) {
        llvm::consumeError(targetAddr.takeError());
        continue;
      }
      addEdge(graph, data, patchedIsText, rel.getOffset(), *targetAddr);
    }
  }

  for (const auto &ref : graph.dataReferences) {
    for (uint64_t dataAddr : ref.second) {
      if (auto it = graph.dataEscapes.find(dataAddr);
          it != graph.dataEscapes.end()) {
        llvm::append_range(graph.nodes[ref.first].callees, it->second);
      }
    }
  }
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
  if (!root) {
    return fail("the linked core '" + elfPath.str() +
                "' declares no function at its entry point");
  }
  // Relocations carry every call edge. A link that drops them leaves the graph
  // with no edges. The walk would then report the entry point's own frame as
  // the whole requirement. The chess/BCF link produces such an ELF.
  if (!sawRelocations) {
    return fail("the linked core '" + elfPath.str() +
                "' retains no relocations, so its call graph is unavailable");
  }
  // The entry point establishes SP. The stack this measures starts there, so
  // the entry point's own frame counts as 0.
  Node &rootNode = graph.nodes[root->addr];
  if (rootNode.frameSize < 0) {
    rootNode.frameSize = 0;
  }
  if (!complete) {
    return fail("the linked core '" + elfPath.str() +
                "' carries malformed .stack_sizes data");
  }

  llvm::DenseMap<uint64_t, VisitState> state;
  llvm::DenseMap<uint64_t, int64_t> memo;
  llvm::SmallVector<uint64_t, 8> pathStack;
  llvm::DenseSet<uint64_t> unmeasured;
  std::string error;
  StackRequirementFailure failureKind = StackRequirementFailure::Unmeasurable;
  auto bytes = maxPathFrom(root->addr, graph, state, memo, pathStack, error,
                           failureKind, unmeasured);
  if (!bytes) {
    return {std::nullopt, std::move(error), failureKind};
  }

  StackRequirementResult result{*bytes, {}, failureKind, {}};
  for (uint64_t addr : unmeasured) {
    result.unmeasured.push_back(graph.nameOf(addr).str());
  }
  llvm::sort(result.unmeasured);
  return result;
}

std::optional<int64_t>
xilinx::aiecc::measureDataSectionBytes(llvm::StringRef elfPath) {
  auto sections = readCoreDataSections(elfPath);
  if (sections.empty() && !llvm::sys::fs::exists(elfPath)) {
    return std::nullopt;
  }
  int64_t total = 0;
  for (const CoreDataSection &sec : sections) {
    // The program memory holds .text; bank-pinned sections have regions of
    // their own. Only unpinned data competes for the `data` region.
    if (!sec.bank) {
      total += sec.size;
    }
  }
  return total;
}

std::optional<xilinx::aiecc::BankSectionSize>
xilinx::aiecc::measureDataSectionDemand(llvm::StringRef elfPath) {
  auto sections = readCoreDataSections(elfPath);
  if (sections.empty() && !llvm::sys::fs::exists(elfPath)) {
    return std::nullopt;
  }
  llvm::stable_sort(sections,
                    [](const CoreDataSection &a, const CoreDataSection &b) {
                      return a.address < b.address;
                    });
  BankSectionSize demand;
  for (const CoreDataSection &sec : sections) {
    if (sec.bank || sec.size == 0) {
      continue;
    }
    demand.size = llvm::alignTo(demand.size, sec.align) + sec.size;
    demand.align = std::max(demand.align, sec.align);
  }
  return demand;
}

namespace {

// Bank letters run in their natural order, matching the `a, b, c, d` order of
// Peano's `aie_dm_resource` and the `_ab` / `_cd` table pairs the AIE runtime
// library ships. One definition, so the section-name and address-space readers
// below cannot drift apart.
constexpr int kMaxBankLetters = 4;

std::optional<int> bankFromLetter(char c) {
  if (c < 'A' || c >= 'A' + kMaxBankLetters) {
    return std::nullopt;
  }
  return c - 'A';
}

// `DM_bank` in a section name, followed by the letters of the banks it asks
// for: `.bss.DM_bankA.4`, or `.data.DM_bankAB`. Returns the banks, or nothing
// when the name carries no such request.
llvm::SmallVector<int, 2> banksFromSectionName(llvm::StringRef name) {
  llvm::SmallVector<int, 2> banks;
  if (name.consume_front(".aie.bank") && !name.empty() && name.front() >= '0' &&
      name.front() <= '3' && (name.size() == 1 || name[1] == '.')) {
    banks.push_back(name.front() - '0');
    return banks;
  }
  size_t pos = name.find("DM_bank");
  if (pos == llvm::StringRef::npos) {
    return banks;
  }
  for (char c : name.drop_front(pos + strlen("DM_bank"))) {
    auto bank = bankFromLetter(c);
    if (!bank) {
      break; // The letters end; what follows is the compiler's own suffix.
    }
    banks.push_back(*bank);
  }
  llvm::sort(banks);
  banks.erase(std::unique(banks.begin(), banks.end()), banks.end());
  return banks;
}

void sortAssertions(std::vector<BankAssertion> &assertions) {
  llvm::sort(assertions, [](const BankAssertion &a, const BankAssertion &b) {
    return std::tie(a.symbol, a.origin) < std::tie(b.symbol, b.origin);
  });
  assertions.erase(
      std::unique(assertions.begin(), assertions.end(),
                  [](const BankAssertion &a, const BankAssertion &b) {
                    return a.symbol == b.symbol && a.origin == b.origin;
                  }),
      assertions.end());
}

} // namespace

llvm::SmallVector<CoreDataSection>
xilinx::aiecc::readCoreDataSections(llvm::StringRef elfPath,
                                    int64_t tileBaseAddress) {
  llvm::SmallVector<CoreDataSection> sections;
  auto binary = llvm::object::createBinary(elfPath);
  if (!binary) {
    llvm::consumeError(binary.takeError());
    return sections;
  }
  auto *obj = llvm::dyn_cast<ObjectFile>(binary->getBinary());
  if (!obj) {
    return sections;
  }
  auto *elf = llvm::dyn_cast<llvm::object::ELFObjectFileBase>(obj);
  for (const SectionRef &sec : obj->sections()) {
    auto name = sec.getName();
    if (!name) {
      llvm::consumeError(name.takeError());
      continue;
    }
    // The linked section name, not a request: chess merges several requests
    // into one output section, and its name is the bank that won.
    auto banks = banksFromSectionName(*name);
    bool isData = name->starts_with(".data") || name->starts_with(".rodata") ||
                  name->starts_with(".bss");
    if (banks.empty() && !isData) {
      continue; // .text and the rest live in program memory.
    }
    CoreDataSection out;
    out.name = name->str();
    out.address = static_cast<int64_t>(sec.getAddress()) - tileBaseAddress;
    out.size = static_cast<int64_t>(sec.getSize());
    out.align = elf ? static_cast<int64_t>(sec.getAlignment().value()) : 1;
    if (banks.size() == 1) {
      out.bank = banks[0];
    }
    sections.push_back(std::move(out));
  }
  return sections;
}

std::vector<BankAssertion> xilinx::aiecc::readBankAssertionsFromObjects(
    llvm::ArrayRef<std::string> objectPaths) {
  std::vector<BankAssertion> assertions;
  llvm::StringSet<> seenObjects, definitions, ambiguous;
  auto inspectObject = [&](const ObjectFile &obj) {
    for (const SymbolRef &sym : obj.symbols()) {
      auto name = sym.getName();
      auto section = sym.getSection();
      auto type = sym.getType();
      auto flags = sym.getFlags();
      if (!name || !section || !type || !flags) {
        llvm::consumeError(name.takeError());
        llvm::consumeError(section.takeError());
        llvm::consumeError(type.takeError());
        llvm::consumeError(flags.takeError());
        continue;
      }
      if (name->empty() || *type != SymbolRef::ST_Data ||
          (*flags & SymbolRef::SF_Undefined)) {
        continue;
      }
      // Count unpinned definitions too. GC can leave only one same-named
      // local in the ELF, without identifying which input request it owns.
      if (!definitions.insert(*name).second) {
        ambiguous.insert(*name);
      }
      if (*section == obj.section_end()) {
        continue;
      }
      auto sectionName = (*section)->getName();
      if (!sectionName) {
        llvm::consumeError(sectionName.takeError());
        continue;
      }
      auto banks = banksFromSectionName(*sectionName);
      if (banks.empty()) {
        continue;
      }
      assertions.push_back({name->str(), sectionName->str(), banks});
    }
  };
  for (llvm::StringRef path : objectPaths) {
    if (!seenObjects.insert(path).second) {
      continue;
    }
    auto binary = llvm::object::createBinary(path);
    if (!binary) {
      llvm::consumeError(binary.takeError());
      continue;
    }
    if (auto *obj = llvm::dyn_cast<ObjectFile>(binary->getBinary())) {
      inspectObject(*obj);
    } else if (auto *archive = llvm::dyn_cast<Archive>(binary->getBinary())) {
      llvm::Error error = llvm::Error::success();
      for (const auto &child : archive->children(error)) {
        auto member = child.getAsBinary();
        if (!member) {
          llvm::consumeError(member.takeError());
          continue;
        }
        if (auto *obj = llvm::dyn_cast<ObjectFile>(member->get())) {
          inspectObject(*obj);
        }
      }
      llvm::consumeError(std::move(error));
    }
  }
  llvm::erase_if(assertions, [&](const BankAssertion &assertion) {
    return ambiguous.contains(assertion.symbol);
  });
  sortAssertions(assertions);
  return assertions;
}

llvm::SmallVector<xilinx::aiecc::BankSectionSize>
xilinx::aiecc::measureBankSectionBytes(llvm::StringRef elfPath, int numBanks) {
  llvm::SmallVector<BankSectionSize> sizes(std::max(numBanks, 0));
  auto sections = readCoreDataSections(elfPath);
  llvm::stable_sort(sections,
                    [](const CoreDataSection &a, const CoreDataSection &b) {
                      return a.address < b.address;
                    });
  for (const CoreDataSection &sec : sections) {
    if (!sec.bank || *sec.bank >= static_cast<int>(sizes.size()) ||
        sec.size == 0) {
      continue;
    }
    auto &demand = sizes[*sec.bank];
    demand.size = llvm::alignTo(demand.size, sec.align) + sec.size;
    demand.align = std::max(demand.align, sec.align);
  }
  return sizes;
}

std::vector<BankViolation> xilinx::aiecc::checkBankPlacements(
    llvm::StringRef elfPath, llvm::ArrayRef<BankAssertion> assertions,
    int64_t tileBaseAddress, int64_t bankSize, int numBanks) {
  std::vector<BankViolation> violations;
  if (assertions.empty() || bankSize <= 0 || numBanks <= 0) {
    return violations;
  }
  llvm::StringMap<uint64_t> sizes;
  auto addrByName = readDataSymbolAddresses(elfPath, tileBaseAddress, &sizes);

  for (const BankAssertion &assertion : assertions) {
    auto it = addrByName.find(assertion.symbol);
    if (it == addrByName.end()) {
      // GC removes unused symbols. Duplicate local names are ambiguous: this
      // default check reports only placements it can prove contradictory.
      continue;
    }
    int64_t offset = it->second;
    if (offset < 0 || offset >= bankSize * numBanks) {
      continue; // Not in this tile's data memory; not ours to judge.
    }
    int actualBank = static_cast<int>(offset / bankSize);
    uint64_t size = sizes.lookup(assertion.symbol);
    bool crossesBank =
        size > static_cast<uint64_t>(bankSize - offset % bankSize);
    if (llvm::is_contained(assertion.banks, actualBank) && !crossesBank) {
      continue;
    }
    violations.push_back({assertion, offset, actualBank, size, crossesBank});
  }
  return violations;
}

namespace {

llvm::StringRef aieIntrinsicName(const llvm::CallBase *call) {
  const llvm::Function *callee = call->getCalledFunction();
  if (!callee) {
    return {};
  }
  llvm::StringRef name = callee->getName();
  if (name.consume_front("llvm.aie2.") || name.consume_front("llvm.aie2p.")) {
    return name;
  }
  return {};
}

// The gather that reads an `aie::lut<4>`. Its addresses arrive as one vector,
// so finding it identifies the pair without knowing anything about the kernel.
bool isLutGather(const llvm::CallBase *call) {
  llvm::StringRef name = aieIntrinsicName(call);
  if (name.consume_front("load.4x16.") || name.consume_front("load.4x32.") ||
      name.consume_front("load.4x64.")) {
    return name == "lo" || name == "hi";
  }
  return false;
}

// An offset into a table is classified by the object that contains it. That is
// sound only because `checkLutBankSeparation` assigns a bank to an object
// solely when the object's whole extent provably lies in one bank, so every
// in-bounds offset shares that bank. `inbounds` also admits one-past-the-end,
// which is the next bank when an object ends flush with a boundary, so a
// constant offset is additionally bounded against the object's size. A variable
// offset rides on `inbounds` alone, and an object of unknown size (an `extern
// tbl[]` declaration) has nothing to bound against.
bool offsetStaysInObject(const llvm::GEPOperator *gep,
                         const llvm::DataLayout &layout) {
  if (gep->hasAllZeroIndices()) {
    return true;
  }
  if (!gep->isInBounds()) {
    return false;
  }
  llvm::APInt offset(layout.getIndexTypeSizeInBits(gep->getType()), 0);
  if (!gep->accumulateConstantOffset(layout, offset)) {
    return true; // Variable offset; `inbounds` is the only guarantee available.
  }
  if (offset.isNegative()) {
    return false;
  }
  const auto *global = llvm::dyn_cast<llvm::GlobalVariable>(
      gep->getPointerOperand()->stripPointerCasts());
  if (!global) {
    return true;
  }
  uint64_t allocSize = layout.getTypeAllocSize(global->getValueType());
  return allocSize == 0 || offset.ult(allocSize);
}

// Walks a broadcast vector of one pointer back to the object it addresses.
// aie_api builds it as splat(zext(ptrtoint(base))), which reaches the IR as a
// shufflevector over an insertelement, and as a constant expression when the
// base is a global.
const llvm::Value *resolveBroadcastBase(const llvm::Value *v,
                                        const llvm::DataLayout &layout) {
  bool splattedInsert = false;
  for (int hop = 0; hop < 16 && v; ++hop) {
    if (const auto *shuf = llvm::dyn_cast<llvm::ShuffleVectorInst>(v)) {
      if (!llvm::all_of(shuf->getShuffleMask(),
                        [](int index) { return index == 0; })) {
        return nullptr;
      }
      splattedInsert = true;
      v = shuf->getOperand(0);
    } else if (const auto *ins = llvm::dyn_cast<llvm::InsertElementInst>(v)) {
      const auto *index = llvm::dyn_cast<llvm::ConstantInt>(ins->getOperand(2));
      if (!splattedInsert || !index || !index->isZero()) {
        return nullptr;
      }
      splattedInsert = false;
      v = ins->getOperand(1);
    } else if (const auto *gep = llvm::dyn_cast<llvm::GEPOperator>(v)) {
      // `aie::linear_approx` folds a nonzero bias into the table address, so
      // rejecting every offset here rejects the common case.
      if (!offsetStaysInObject(gep, layout)) {
        return nullptr;
      }
      v = gep->getPointerOperand();
    } else if (const auto *op = llvm::dyn_cast<llvm::Operator>(v);
               op && (llvm::isa<llvm::CastInst>(v) ||
                      llvm::isa<llvm::ConstantExpr>(v))) {
      if (op->getOpcode() != llvm::Instruction::ZExt &&
          op->getOpcode() != llvm::Instruction::PtrToInt &&
          op->getOpcode() != llvm::Instruction::BitCast &&
          op->getOpcode() != llvm::Instruction::AddrSpaceCast) {
        return nullptr;
      }
      v = op->getOperand(0);
    } else if (const auto *constant = llvm::dyn_cast<llvm::Constant>(v);
               constant && constant->getType()->isVectorTy()) {
      v = constant->getSplatValue();
    } else if (const auto *call = llvm::dyn_cast<llvm::CallBase>(v);
               call && aieIntrinsicName(call) == "vbroadcast32.I512" &&
               call->arg_size() == 1) {
      v = call->getArgOperand(0);
    } else {
      return v;
    }
  }
  return nullptr;
}

std::optional<LutOperand> asLutOperand(const llvm::Value *v) {
  LutOperand op;
  if (const auto *global = llvm::dyn_cast_or_null<llvm::GlobalVariable>(v)) {
    op.kind = LutOperand::Kind::Symbol;
    op.symbol = global->getName().str();
    return op;
  }
  if (const auto *arg = llvm::dyn_cast_or_null<llvm::Argument>(v);
      arg && arg->getType()->isPointerTy()) {
    op.kind = LutOperand::Kind::Param;
    op.paramIndex = static_cast<int>(arg->getArgNo());
    return op;
  }
  if (llvm::isa_and_nonnull<llvm::AllocaInst>(v)) {
    op.kind = LutOperand::Kind::Stack;
    return op;
  }
  return std::nullopt;
}

std::optional<std::vector<LutPair>> readLutPairs(llvm::MemoryBufferRef buffer) {
  if (buffer.getBuffer().empty()) {
    return std::nullopt;
  }
  llvm::LLVMContext context;
  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> module = llvm::parseIR(buffer, err, context);
  if (!module) {
    return std::nullopt;
  }

  std::vector<LutPair> pairs;
  const llvm::DataLayout &layout = module->getDataLayout();
  for (const llvm::Function &fn : *module) {
    for (const llvm::Instruction &inst : llvm::instructions(fn)) {
      const auto *gather = llvm::dyn_cast<llvm::CallBase>(&inst);
      if (!gather || !isLutGather(gather)) {
        continue;
      }
      // Only the gather's address expression identifies its table pair. A
      // vsel elsewhere in the function can be unrelated ordinary blending.
      llvm::SmallVector<const llvm::Value *, 16> worklist;
      llvm::DenseSet<const llvm::Value *> visited;
      if (!gather->arg_empty()) {
        worklist.push_back(gather->getArgOperand(0));
      }
      bool found = false;
      while (!worklist.empty()) {
        const llvm::Value *value = worklist.pop_back_val();
        if (!visited.insert(value).second) {
          continue;
        }
        if (const auto *call = llvm::dyn_cast<llvm::CallBase>(value)) {
          llvm::StringRef name = aieIntrinsicName(call);
          if (name == "vsel32" && call->arg_size() >= 2) {
            auto a = asLutOperand(
                resolveBroadcastBase(call->getArgOperand(0), layout));
            auto b = asLutOperand(
                resolveBroadcastBase(call->getArgOperand(1), layout));
            // The index expression may select ordinary integers too. Require
            // pointer evidence for a candidate; if none is found anywhere,
            // the gather still gets an Unknown pair below.
            if (!a && !b) {
              continue;
            }
            pairs.push_back({fn.getName().str(), a.value_or(LutOperand{}),
                             b.value_or(LutOperand{})});
            found = true;
          } else if (name == "ext.I256.I512" && call->arg_size() == 2) {
            worklist.push_back(call->getArgOperand(0));
          }
          // Never follow an arbitrary call (in particular, a load) into its
          // arguments: that is not evidence of the returned address's origin.
          continue;
        }
        if (const auto *op = llvm::dyn_cast<llvm::Instruction>(value)) {
          if (const auto *shuffle =
                  llvm::dyn_cast<llvm::ShuffleVectorInst>(op)) {
            unsigned width =
                llvm::cast<llvm::VectorType>(shuffle->getOperand(0)->getType())
                    ->getElementCount()
                    .getKnownMinValue();
            bool first = false, second = false;
            for (int index : shuffle->getShuffleMask()) {
              first |= index >= 0 && static_cast<unsigned>(index) < width;
              second |= index >= 0 && static_cast<unsigned>(index) >= width;
            }
            if (first != second) {
              worklist.push_back(shuffle->getOperand(second ? 1 : 0));
            } else {
              pairs.push_back({fn.getName().str(), {}, {}});
              found = true;
            }
          } else if (llvm::isa<llvm::PHINode, llvm::SelectInst>(op)) {
            // One resolved arm is not evidence for the other arms.
            pairs.push_back({fn.getName().str(), {}, {}});
            found = true;
          } else if (op->getOpcode() == llvm::Instruction::Add ||
                     op->getOpcode() == llvm::Instruction::BitCast) {
            for (const llvm::Value *operand : op->operand_values()) {
              worklist.push_back(operand);
            }
          }
        }
      }
      if (!found) {
        pairs.push_back({fn.getName().str(), {}, {}});
      }
    }
  }
  return pairs;
}

} // namespace

std::optional<std::vector<LutPair>>
xilinx::aiecc::readLutPairsFromIR(llvm::StringRef irPath) {
  auto buffer = llvm::MemoryBuffer::getFile(irPath);
  if (!buffer) {
    return std::nullopt;
  }
  // LTO can inline a live source function and remove its symbol, so absence
  // from the final ELF cannot prove a raw IR function's gathers are dead.
  return readLutPairs((*buffer)->getMemBufferRef());
}

std::optional<std::vector<LutPair>>
xilinx::aiecc::readLutPairsFromObject(llvm::StringRef objectPath,
                                      llvm::StringRef elfPath) {
  if (objectPath.ends_with(".ll") || objectPath.ends_with(".bc")) {
    return readLutPairsFromIR(objectPath);
  }
  auto binary = llvm::object::createBinary(objectPath);
  if (!binary) {
    llvm::consumeError(binary.takeError());
    return std::nullopt;
  }
  auto *obj = llvm::dyn_cast<ObjectFile>(binary->getBinary());
  if (!obj) {
    return std::nullopt;
  }
  llvm::StringSet<> objectFunctions, liveFunctions;
  if (!elfPath.empty()) {
    for (const SymbolRef &sym : obj->symbols()) {
      auto name = sym.getName();
      auto type = sym.getType();
      auto flags = sym.getFlags();
      if (!name || !type || !flags) {
        llvm::consumeError(name.takeError());
        llvm::consumeError(type.takeError());
        llvm::consumeError(flags.takeError());
        return std::nullopt;
      }
      if (*type == SymbolRef::ST_Function &&
          !(*flags & SymbolRef::SF_Undefined)) {
        objectFunctions.insert(*name);
      }
    }
    auto linkedBinary = llvm::object::createBinary(elfPath);
    if (!linkedBinary) {
      llvm::consumeError(linkedBinary.takeError());
      return std::nullopt;
    }
    auto *linkedObject = llvm::dyn_cast<ObjectFile>(linkedBinary->getBinary());
    if (!linkedObject) {
      return std::nullopt;
    }
    for (const SymbolRef &sym : linkedObject->symbols()) {
      auto name = sym.getName();
      auto type = sym.getType();
      auto flags = sym.getFlags();
      if (!name || !type || !flags) {
        llvm::consumeError(name.takeError());
        llvm::consumeError(type.takeError());
        llvm::consumeError(flags.takeError());
        return std::nullopt;
      }
      if (*type == SymbolRef::ST_Function &&
          !(*flags & SymbolRef::SF_Undefined)) {
        liveFunctions.insert(*name);
      }
    }
  }
  std::optional<std::vector<LutPair>> pairs;
  for (const SectionRef &sec : obj->sections()) {
    auto name = sec.getName();
    if (!name) {
      llvm::consumeError(name.takeError());
      return std::nullopt;
    }
    if (*name != ".llvmbc") {
      continue;
    }
    auto contents = sec.getContents();
    if (!contents) {
      llvm::consumeError(contents.takeError());
      return std::nullopt;
    }
    auto sectionPairs =
        readLutPairs(llvm::MemoryBufferRef(*contents, objectPath));
    if (!sectionPairs) {
      return std::nullopt;
    }
    if (!pairs) {
      pairs.emplace();
    }
    llvm::append_range(*pairs, *sectionPairs);
  }
  if (pairs && !elfPath.empty()) {
    llvm::erase_if(*pairs, [&](const LutPair &pair) {
      // Embedded IR can precede codegen inlining. Only a function emitted in
      // the input object and absent from the ELF is known to have been removed.
      return objectFunctions.contains(pair.function) &&
             !liveFunctions.contains(pair.function);
    });
  }
  return pairs;
}

llvm::StringMap<int64_t>
xilinx::aiecc::readDataSymbolAddresses(llvm::StringRef elfPath,
                                       int64_t tileBaseAddress,
                                       llvm::StringMap<uint64_t> *sizes) {
  llvm::StringMap<int64_t> addrByName;
  if (sizes) {
    sizes->clear();
  }
  if (tileBaseAddress < 0) {
    return addrByName;
  }
  auto binary = llvm::object::createBinary(elfPath);
  if (!binary) {
    llvm::consumeError(binary.takeError());
    return addrByName;
  }
  auto *obj = llvm::dyn_cast<ObjectFile>(binary->getBinary());
  if (!obj) {
    return addrByName;
  }
  llvm::StringSet<> ambiguous;
  for (const SymbolRef &sym : obj->symbols()) {
    auto name = sym.getName();
    auto addr = sym.getAddress();
    auto type = sym.getType();
    auto flags = sym.getFlags();
    if (!name || !addr || !type || !flags) {
      llvm::consumeError(name.takeError());
      llvm::consumeError(addr.takeError());
      llvm::consumeError(type.takeError());
      llvm::consumeError(flags.takeError());
      continue;
    }
    if (*type != SymbolRef::ST_Data || (*flags & SymbolRef::SF_Undefined) ||
        *addr > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        ambiguous.contains(*name)) {
      continue;
    }
    int64_t offset = static_cast<int64_t>(*addr) - tileBaseAddress;
    auto inserted = addrByName.try_emplace(*name, offset);
    if (!inserted.second && inserted.first->second != offset) {
      // Distinct local symbols may have the same name after linking. The IR
      // alone cannot tell which definition the object contributed.
      addrByName.erase(*name);
      ambiguous.insert(*name);
      if (sizes) {
        sizes->erase(*name);
      }
      continue;
    }
    if (sizes && llvm::isa<ELFObjectFileBase>(obj)) {
      uint64_t size = ELFSymbolRef(sym).getSize();
      auto insertedSize = sizes->try_emplace(*name, size);
      if (!insertedSize.second) {
        insertedSize.first->second = std::max(insertedSize.first->second, size);
      }
    }
  }
  return addrByName;
}

std::optional<int64_t>
xilinx::aiecc::parseLinkOverflowBytes(llvm::StringRef log,
                                      llvm::StringRef region) {
  // Anchored on the region and confined to its own line: a link can overflow
  // several regions, and reading another one's number here would misreport the
  // shortfall of whichever the caller asked about.
  std::string needle = ("will not fit in region '" + region + "'").str();
  size_t pos = log.find(needle);
  if (pos == llvm::StringRef::npos) {
    return std::nullopt;
  }
  llvm::StringRef line =
      log.drop_front(pos).take_until([](char c) { return c == '\n'; });
  size_t at = line.find("overflowed by ");
  if (at == llvm::StringRef::npos) {
    return std::nullopt;
  }
  int64_t bytes = 0;
  if (line.drop_front(at + strlen("overflowed by "))
          .consumeInteger(10, bytes)) {
    return std::nullopt;
  }
  return bytes;
}
