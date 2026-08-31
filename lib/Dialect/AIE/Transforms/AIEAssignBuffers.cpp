//===- AIEAssignBuffers.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEASSIGNBUFFERADDRESSES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-assign-buffers"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

static bool isBufferPreAllocated(BufferOp buffer) {
  auto addr = buffer.getAddress();
  auto memBank = buffer.getMemBank();
  return (addr != std::nullopt || memBank != std::nullopt);
}

// Return an address that is aligned to tile's load/store bus
// NOTE: assume address are byte address
static int64_t getAlignedAddress(int64_t address, uint32_t alignBitWidth) {
  assert(alignBitWidth != 0 && alignBitWidth % 8 == 0 &&
         "alignBitWidth must be a non-zero multiple of 8");
  uint32_t alignByteWidth = alignBitWidth / 8;
  if (address % alignByteWidth == 0) {
    return address;
  }
  return ((address / alignByteWidth) + 1) * alignByteWidth;
}

// Return the alignment (in bits) `buffer` must satisfy.
//
// The load/store bus width is not sufficient on its own: from AIE2P on, a
// full-width vector access needs 512-bit alignment while the bus is 256 bits
// (see AIETargetModel::getComputeTileMaxVectorAlignBits and aie_api's
// vector_ldst_align). A buffer big enough to hold such a vector may be accessed
// by one -- we cannot see inside externally-compiled kernels -- so it gets the
// stricter alignment. Buffers too small to hold one cannot be accessed that way
// without going out of bounds, so the bus width still applies and they cost no
// extra padding.
static uint32_t getRequiredAlignBits(BufferOp buffer, uint32_t busAlignBits,
                                     uint32_t maxVecAlignBits) {
  if (maxVecAlignBits <= busAlignBits)
    return busAlignBits;
  int64_t sizeBits = static_cast<int64_t>(buffer.getAllocationSize()) * 8;
  return sizeBits >= static_cast<int64_t>(maxVecAlignBits) ? maxVecAlignBits
                                                           : busAlignBits;
}

// Check that every buffer in the list is properly aligned (when its
// `aligned` attribute is set) and that no two buffers overlap. The input
// vector must be sorted by ascending address. Emits an error on the first
// offending buffer and returns false; returns true otherwise.
static bool checkAndPrintBufferOverlap(SmallVector<BufferOp> &sortedBuffers,
                                       uint32_t tileAlignBitWidth,
                                       uint32_t maxVecAlignBits) {
  for (size_t i = 0; i < sortedBuffers.size(); ++i) {
    auto cur = sortedBuffers[i];
    auto curAddrOpt = cur.getAddress();
    assert(curAddrOpt.has_value() && "buffer must have address assigned");
    int64_t curAddr = *curAddrOpt;

    // Alignment check. Buffers the user pinned are held only to the bus width
    // (see checkAndAddBufferWithAddress); the stricter vector requirement
    // applies to addresses this pass chose.
    uint32_t reqAlignBits =
        isBufferPreAllocated(cur)
            ? tileAlignBitWidth
            : getRequiredAlignBits(cur, tileAlignBitWidth, maxVecAlignBits);
    uint32_t alignByteWidth = reqAlignBits / 8;
    if (cur.getAligned() && alignByteWidth != 0 &&
        curAddr % alignByteWidth != 0) {
      cur.emitOpError("buffer '")
          << cur.name() << "' at address 0x" << llvm::utohexstr(curAddr)
          << " is not aligned to the required " << reqAlignBits << " bits";
      return false;
    }

    // Overlap check against every EARLIER buffer this one may not share with.
    //
    // Comparing only against sortedBuffers[i - 1] is sound only while no pair
    // is exempt: a buffer nested inside a larger earlier one is caught because
    // the two are adjacent in address order. An exempt pair breaks that. A
    // large buffer, a small same-group buffer overlaid on it, then a third
    // buffer starting at the small one's end is adjacent only to the small one,
    // and its overlap with the large one is never examined. So track the
    // furthest end among the buffers `cur` must stay clear of.
    //
    // Buffers in DIFFERENT alloc_groups are overlaid on purpose -- that is the
    // whole contract -- so a shared address between them is not a defect. Two
    // members of the SAME group are laid out sequentially and must not overlap.
    // A grouped buffer overlapping an UNGROUPED one still is. Both groups have
    // to be present for the exemption, or two UNGROUPED buffers would compare
    // equal as absent and exempt every ordinary overlap.
    BufferOp blocker;
    int64_t blockerEnd = 0;
    for (size_t j = 0; j < i; ++j) {
      auto other = sortedBuffers[j];
      if (auto curGroup = cur.getAllocGroup())
        if (auto otherGroup = other.getAllocGroup())
          if (*curGroup != *otherGroup)
            continue;
      auto otherAddrOpt = other.getAddress();
      assert(otherAddrOpt.has_value() && "buffer must have address assigned");
      int64_t end = *otherAddrOpt + other.getAllocationSize();
      if (end > blockerEnd) {
        blockerEnd = end;
        blocker = other;
      }
    }
    if (blocker && curAddr < blockerEnd) {
      cur.emitOpError("buffer '")
          << cur.name() << "' at address 0x" << llvm::utohexstr(curAddr)
          << " overlaps with '" << blocker.name() << "' at address 0x"
          << llvm::utohexstr(blocker.getAddress().value())
          << " (size: " << blocker.getAllocationSize() << " bytes)";
      return false;
    }
  }
  return true;
}

// Check if there is any overlap between the stack and the allocated buffers.
static bool checkAndPrintOverlapStackframe(int stacksize,
                                           SmallVector<BufferOp> &buffers) {
  for (auto buf : buffers) {
    auto bufAddrOpt = buf.getAddress();
    assert(bufAddrOpt.has_value() && "buffer must have address assigned");
    int64_t bufAddr = *bufAddrOpt;
    if (bufAddr < stacksize) {
      buf.emitOpError("buffer '")
          << buf.name() << "' at address 0x" << llvm::utohexstr(bufAddr)
          << " overlaps with stack (size: " << stacksize << " bytes)";
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// BasicAllocation : sequential alloc from largest to smallest
//===----------------------------------------------------------------------===//
static bool checkAndPrintOverflow(TileOp tile, int address,
                                  int maxDataMemorySize, int stacksize,
                                  SmallVector<BufferOp> &buffers) {
  if (address > maxDataMemorySize) {
    InFlightDiagnostic error =
        tile.emitOpError("allocated buffers exceeded available memory\n");
    auto &note = error.attachNote() << "MemoryMap:\n";
    auto printbuffer = [&](StringRef name, int address, int size) {
      note << "\t" << name << " \t" << ": 0x" << llvm::utohexstr(address)
           << "-0x" << llvm::utohexstr(address + size - 1) << " \t(" << size
           << " bytes)\n";
    };
    if (stacksize > 0)
      printbuffer("(stack)", 0, stacksize);
    else
      error << "(no stack allocated)\n";

    for (auto buffer : buffers) {
      auto bufferAddrOpt = buffer.getAddress();
      assert(bufferAddrOpt.has_value() && "buffer must have address assigned");
      printbuffer(buffer.name(), *bufferAddrOpt, buffer.getAllocationSize());
    }
    return false;
  }
  return true;
}

// An allocation unit is either one ordinary buffer or every `alloc_group`
// on the tile, overlaid.
// Members of a group are asserted never to be live simultaneously, so they
// overlay: the unit costs its LARGEST member, not the sum. See AIE_BufferOp's
// description for what is asserted and what is checked.
namespace {
struct AllocUnit {
  // Each entry is one alloc_group, laid out sequentially from the unit's base.
  // Entries overlay: they all start at the same base, so the unit costs the
  // largest group's total, not the sum of every group. An ungrouped buffer is a
  // unit holding one group of one member, which is the pre-existing behaviour.
  SmallVector<SmallVector<BufferOp>> groups;
  int64_t size = 0;     // max over groups of that group's summed extent
  bool aligned = false; // any member aligned => align the shared base
};
} // namespace

// Group this tile's unallocated buffers into allocation units, preserving the
// pass's existing largest-first order across units.
static SmallVector<AllocUnit> buildAllocUnits(ArrayRef<BufferOp> buffers) {
  SmallVector<AllocUnit> units;
  // Every alloc_group on this tile overlays every other, so they share ONE
  // unit. groupIndex maps a group name to its slot within that unit's group
  // list.
  llvm::MapVector<StringRef, unsigned> groupIndex;
  std::optional<unsigned> overlayUnit;
  for (auto buffer : buffers) {
    auto group = buffer.getAllocGroup();
    if (!group) {
      AllocUnit u;
      u.groups.push_back({buffer});
      u.size = buffer.getAllocationSize();
      u.aligned = buffer.getAligned();
      units.push_back(std::move(u));
      continue;
    }
    if (!overlayUnit) {
      overlayUnit = units.size();
      units.push_back(AllocUnit{});
    }
    AllocUnit &u = units[*overlayUnit];
    auto [it, inserted] = groupIndex.try_emplace(*group, u.groups.size());
    if (inserted)
      u.groups.push_back({});
    u.groups[it->second].push_back(buffer);
    u.aligned |= buffer.getAligned();
    // A group's extent is its members' sum; the unit's is the largest such sum.
    int64_t extent = 0;
    for (auto member : u.groups[it->second])
      extent += member.getAllocationSize();
    u.size = std::max(u.size, extent);
  }
  std::stable_sort(
      units.begin(), units.end(),
      [](const AllocUnit &a, const AllocUnit &b) { return a.size > b.size; });
  return units;
}

// The decidable half of the alloc_group assertion: two buffers from different
// groups that one core references without a selector between them are
// simultaneously live by construction, so they can never be overlaid whatever
// the author intended. Catching this here turns a silent memory-aliasing bug
// into a compile error.
// Does any buffer on this tile ask for an overlay?
static bool tileHasAllocGroup(DeviceOp device, TileOp tile) {
  bool found = false;
  device.walk([&](BufferOp buffer) {
    if (buffer.getTileOp() == tile && buffer.getAllocGroup())
      found = true;
  });
  return found;
}

// Whether two ops sit in branches of one branching op that selects exactly one
// of them -- the only ground on which two references inside one core body may
// be called mutually exclusive. Decided at the INNERMOST COMMON ANCESTOR: two
// ops in different regions of one scf.if / scf.index_switch are exclusive;
// anything else (different ops, an scf.while whose regions both run, plain
// sequence) is not.
//
// Scoping this at the block rather than the ancestor does not work: the
// objectFifo lowering hands every buffer reference its own scf.index_switch
// case, so on a lowered design no two buffers ever share a block and a
// block-scoped check silently passes everything.
static bool inExclusiveBranches(Operation *a, Operation *b) {
  llvm::DenseMap<Operation *, Region *> aChain;
  for (Region *r = a->getParentRegion(); r; r = r->getParentRegion())
    if (Operation *parent = r->getParentOp())
      aChain[parent] = r;
    else
      break;

  for (Region *r = b->getParentRegion(); r; r = r->getParentRegion()) {
    Operation *parent = r->getParentOp();
    if (!parent)
      break;
    auto it = aChain.find(parent);
    if (it == aChain.end())
      continue;
    // Innermost common ancestor. Only a one-of-N selector makes its regions
    // exclusive; scf.for/scf.while run theirs unconditionally.
    if (!isa<scf::IfOp, scf::IndexSwitchOp>(parent))
      return false;
    return it->second != r;
  }
  return false;
}

static bool checkAllocGroups(DeviceOp device) {
  bool ok = true;
  // Collect every reference to a grouped buffer from inside a core. Two
  // references conflict when they name DIFFERENT groups, since different groups
  // are the ones the allocator overlays; same-group members get distinct
  // storage and may be live together, which is the point of the group.
  device.walk([&](CoreOp core) {
    SmallVector<std::tuple<BufferOp, Operation *, StringRef>> refs;
    core.walk([&](Operation *op) {
      for (Value operand : op->getOperands())
        if (auto buffer = dyn_cast_or_null<BufferOp>(operand.getDefiningOp()))
          if (auto group = buffer.getAllocGroup())
            refs.emplace_back(buffer, op, *group);
    });

    for (size_t i = 0; i < refs.size(); ++i)
      for (size_t j = i + 1; j < refs.size(); ++j) {
        auto [bufI, opI, groupI] = refs[i];
        auto [bufJ, opJ, groupJ] = refs[j];
        if (groupI == groupJ)
          continue;
        if (inExclusiveBranches(opI, opJ))
          continue;
        ok = false;
        bufJ.emitOpError()
            << "is in alloc_group '" << groupJ
            << "' while this aie.core also references a buffer in alloc_group '"
            << groupI
            << "', and no selector separates the two references, so they are "
               "simultaneously live and cannot be overlaid";
        bufI.emitRemark() << "member of alloc_group '" << groupI << "'";
        return;
      }
  });
  return ok;
}

static bool basicAllocation(TileOp tile) {
  auto device = tile->getParentOfType<AIE::DeviceOp>();
  if (!device)
    return false;

  const auto &targetModel = getTargetModel(tile);
  int maxDataMemorySize = 0;
  uint32_t tileAlignBitWidth = 0;
  // MemTile buffers are reached by DMA rather than by core vector load/stores,
  // so only the bus width applies there.
  uint32_t maxVecAlignBitWidth = 0;
  if (tile.isMemTile()) {
    maxDataMemorySize = targetModel.getMemTileSize();
    tileAlignBitWidth = targetModel.getMemTileLoadStoreBusWidth();
    maxVecAlignBitWidth = tileAlignBitWidth;
  } else {
    maxDataMemorySize = targetModel.getLocalMemorySize();
    tileAlignBitWidth = targetModel.getComputeTileLoadStoreBusWidth();
    maxVecAlignBitWidth = targetModel.getComputeTileMaxVectorAlignBits();
  }

  SmallVector<BufferOp> buffers;
  SmallVector<BufferOp> allocated_buffers;
  SmallVector<BufferOp> allBuffers_on_tile;
  // Collect all the buffers for this tile. If the buffer has an address, add
  // it to allocated_buffers. Otherwise, add it to buffers.
  device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
    if (buffer.getTileOp() == tile) {
      if (buffer.getAddress()) {
        allocated_buffers.push_back(
            buffer); // TODO: Right now, this ignore all buffer with only
                     // mem_bank attribute.
      } else {
        buffers.push_back(buffer);
      }
      allBuffers_on_tile.push_back(buffer);
    }
  });

  // Sort buffers by allocation size.
  std::sort(buffers.begin(), buffers.end(), [](BufferOp a, BufferOp b) {
    return a.getAllocationSize() > b.getAllocationSize();
  });

  // Sort allocated_buffers by address
  std::sort(allocated_buffers.begin(), allocated_buffers.end(),
            [](BufferOp a, BufferOp b) {
              return a.getAddress().value() < b.getAddress().value();
            });

  // Address range owned by the MemTile is 0x80000.
  // Address range owned by the tile is 0x8000 in
  // AIE1 and 0x10000 in AIE2, but we need room at
  // the bottom for stack.
  int64_t stacksize = 0;
  int64_t address = 0;
  if (auto core = tile.getCoreOp()) {
    stacksize = core.getEffectiveStackSize();
    address += stacksize;
  }

  // Ensure alignment of preallocated buffer
  for (auto buffer : allocated_buffers) {
    auto bufferAddrOpt = buffer.getAddress();
    assert(bufferAddrOpt.has_value() &&
           "allocated_buffers only holds buffers with an address");
    // An explicitly pinned address is the user's assertion, often dictated by
    // an external ABI (e.g. RTP buffers a host writes at a fixed address). Hold
    // it only to the bus width, as before: the allocator's job is to avoid
    // *creating* misalignment, not to veto a supplied address on a requirement
    // inferred from the buffer's size.
    if (buffer.getAligned() && *bufferAddrOpt % (tileAlignBitWidth / 8) != 0) {
      buffer.emitOpError("pre-allocated address must be aligned to ")
          << tileAlignBitWidth << " bits when the aligned attribute is set";
      return false;
    }
  }

  // As the next address to allocate is assigned, skip over any buffers
  // from the allocated_buffers list.
  // Note: alignment must be applied *before* (and after each skip in) the
  // overlap-skip loop below, so the loop reasons about the buffer's actual
  // placement. Otherwise an unaligned candidate address can appear to fit
  // before the next pre-allocated buffer, but get bumped forward by
  // getAlignedAddress and silently alias that pre-allocated buffer.
  auto *current_alloc = allocated_buffers.begin();
  for (const AllocUnit &unit : buildAllocUnits(buffers)) {
    // Every group starts at the unit's base, so the base has to satisfy the
    // strictest requirement any member has -- a per-buffer figure since
    // getRequiredAlignBits keys on the buffer's own size.
    uint32_t unitAlignBits = tileAlignBitWidth;
    for (const auto &group : unit.groups)
      for (auto buffer : group)
        if (buffer.getAligned())
          unitAlignBits = std::max(
              unitAlignBits, getRequiredAlignBits(buffer, tileAlignBitWidth,
                                                  maxVecAlignBitWidth));
    if (unit.aligned)
      address = getAlignedAddress(address, unitAlignBits);
    while (current_alloc != allocated_buffers.end() &&
           address + unit.size > current_alloc->getAddress().value()) {
      address = current_alloc->getAddress().value() +
                current_alloc->getAllocationSize();
      if (unit.aligned)
        address = getAlignedAddress(address, unitAlignBits);
      current_alloc++;
    }

    // Groups overlay: each starts at the unit's base. Members within a group do
    // not, so they follow one another. The unit advances by the largest group's
    // total, which is what makes N modes cost max(sum) rather than sum(max).
    for (const auto &group : unit.groups) {
      int64_t offset = address;
      for (auto buffer : group) {
        assert(!buffer.getAddress());
        if (buffer.getAligned())
          offset = getAlignedAddress(
              offset, getRequiredAlignBits(buffer, tileAlignBitWidth,
                                           maxVecAlignBitWidth));
        buffer.setAddress(offset);
        offset += buffer.getAllocationSize();
      }
    }
    address += unit.size;
  }

  // Sort by smallest address before printing memory map and running the
  // overlap / overflow checks below.
  std::sort(allBuffers_on_tile.begin(), allBuffers_on_tile.end(),
            [](BufferOp a, BufferOp b) {
              assert(a.getAddress().has_value() &&
                     "buffer must have address assigned");
              assert(b.getAddress().has_value() &&
                     "buffer must have address assigned");
              return a.getAddress().value() < b.getAddress().value();
            });

  // Compute the true high-water mark across *all* buffers (including
  // pre-allocated ones above the dynamic-allocation cursor) so that
  // checkAndPrintOverflow sees a correct memory bound.
  int64_t highWater = address;
  if (!allBuffers_on_tile.empty()) {
    auto &last = allBuffers_on_tile.back();
    auto lastAddrOpt = last.getAddress();
    assert(lastAddrOpt.has_value() && "buffer must have address assigned");
    highWater =
        std::max<int64_t>(highWater, *lastAddrOpt + last.getAllocationSize());
  }

  // Check if memory was exceeded or buffers overlap, and print debug info.
  return (checkAndPrintOverlapStackframe(stacksize, allBuffers_on_tile) &&
          checkAndPrintBufferOverlap(allBuffers_on_tile, tileAlignBitWidth,
                                     maxVecAlignBitWidth) &&
          checkAndPrintOverflow(tile, highWater, maxDataMemorySize, stacksize,
                                allBuffers_on_tile));
}

//===----------------------------------------------------------------------===//
// SimpleBankAwareAllocation : round-robin each alloc over available banks
//===----------------------------------------------------------------------===//
using BankLimits = struct BankLimits {
  int64_t startAddr;
  int64_t endAddr;
};

// Function that given a number of banks and their size, computes
// the start and end addresses for each bank and fills in the entry
// in the bankLimits vector.
static void fillBankLimits(int numBanks, int bankSize,
                           std::vector<BankLimits> &bankLimits) {
  for (int i = 0; i < numBanks; i++) {
    auto startAddr = bankSize * i;
    auto endAddr = bankSize * (i + 1);
    bankLimits.push_back({startAddr, endAddr});
  }
}

// Function that sets the address attribute of the given buffer to
// the given start_addr. It also updates the entry in the
// nextAddrInBanks for the corresponding bank.
static void setAndUpdateAddressInBank(BufferOp buffer, int64_t start_addr,
                                      int64_t end_addr,
                                      std::vector<int64_t> &nextAddrInBanks) {

  buffer.setAddress(start_addr);
  auto memBankOpt = buffer.getMemBank();
  assert(memBankOpt.has_value() &&
         "callers must set mem_bank before updating its bank cursor");
  nextAddrInBanks[*memBankOpt] = end_addr;
}

// Function that checks whether the given buffer already has a set address
// attribute. If it does, it finds in which bank the buffer is and checks
// whether there is enough space left for it (and ensure the bank match to
// the mem_bank attribute if given).
// If there is the function
// returns true and if not, the function emits a warning that the address
// will be overwritten and returns false (which will cause the buffer to be
// added to the list of buffers without addresses, to be completed later on).
static FailureOr<bool>
checkAndAddBufferWithAddress(BufferOp buffer, int numBanks,
                             uint32_t tileAlignBitWidth,
                             [[maybe_unused]] uint32_t maxVecAlignBits,
                             std::vector<int64_t> &nextAddrInBanks,
                             std::vector<BankLimits> &bankLimits) {
  auto addrAttr = buffer->getAttrOfType<IntegerAttr>("address");
  if (!addrAttr)
    return false;
  // it is fine if mem_bank is not set
  auto memBankAttr = buffer->getAttrOfType<IntegerAttr>("mem_bank");

  int addr = addrAttr.getInt();
  // As in basicAllocation: an explicitly pinned address is held only to the bus
  // width, since it may be fixed by an external ABI.
  if (buffer.getAligned() && addr % (tileAlignBitWidth / 8) != 0) {
    return buffer->emitOpError("address attribute value must be aligned to ")
           << tileAlignBitWidth << " bits when the aligned attribute is set";
  }
  for (int i = 0; i < numBanks; i++) {
    // if the address is not within the bank, continue
    if (addr < bankLimits[i].startAddr || addr >= bankLimits[i].endAddr)
      continue;

    // if the allocator already allocated this address, fail
    if (addr < nextAddrInBanks[i])
      return buffer->emitOpError("would override allocated address");

    // the allocator can accomadate this existing allocation
    nextAddrInBanks[i] = addr + buffer.getAllocationSize();
    if (memBankAttr) {
      // specified both mem_bank and address, check if they are consistent
      int mem_bank = memBankAttr.getInt();
      if (mem_bank != i)
        return buffer->emitOpError(
            "mem_bank attribute is inconsistent with address attribute");
    }
    buffer.setMemBank(i);
    return true;
  }
  return buffer->emitOpError(
      "address attribute does not fall within any bank range");
}

// Function that checks whether the given buffer already has a set mem_bank
// attribute. If it does, it checks whether there is enough space left for
// it. If there is, it sets the buffer's address field and if not, the
// function emits a warning that the mem_bank will be overwritten and returns
// false (which will cause the buffer to be added to the list of buffers
// without addresses, to be completed later on).
static FailureOr<bool> checkAndAddBufferWithMemBank(
    BufferOp buffer, int numBanks, uint32_t tileAlignBitWidth,
    uint32_t maxVecAlignBits, std::vector<int64_t> &nextAddrInBanks,
    std::vector<BankLimits> &bankLimits) {
  auto memBankAttr = buffer->getAttrOfType<IntegerAttr>("mem_bank");
  if (!memBankAttr)
    return false;

  int mem_bank = memBankAttr.getInt();
  if (mem_bank < 0 || mem_bank >= numBanks) {
    return buffer->emitOpError("mem_bank attribute value is out of range");
  }

  int64_t startAddr = nextAddrInBanks[mem_bank];
  if (buffer.getAligned()) {
    startAddr = getAlignedAddress(
        startAddr,
        getRequiredAlignBits(buffer, tileAlignBitWidth, maxVecAlignBits));
  }

  int64_t endAddr = startAddr + buffer.getAllocationSize();
  if (endAddr > bankLimits[mem_bank].endAddr)
    return buffer->emitOpError("would override existing mem_bank");
  setAndUpdateAddressInBank(buffer, startAddr, endAddr, nextAddrInBanks);
  return true;
}

// Prints the memory map across banks
static void printMemMap(TileOp tile, SmallVector<BufferOp> &allocatedBuffers,
                        SmallVector<BufferOp> &preAllocatedBuffers,
                        int numBanks, std::vector<BankLimits> &bankLimits,
                        int stacksize) {
  InFlightDiagnostic error = tile.emitWarning(
      "Not all requested buffers fit in the available memory.\n");
  auto &note = error.attachNote()
               << "Current configuration of buffers in bank(s) : ";
  note << "MemoryMap:\n";
  auto printbuffer = [&](StringRef name, int address, int size) {
    note << "\t" << "\t" << name << " \t" << ": 0x" << llvm::utohexstr(address)
         << "-0x" << llvm::utohexstr(address + size - 1) << " \t(" << size
         << " bytes)\n";
  };
  for (int i = 0; i < numBanks; i++) {
    if (i == 0) {
      if (stacksize > 0)
        printbuffer("(stack)", 0, stacksize);
      else
        note << "(no stack allocated)\n";
    }
    note << "\t" << "bank : " << i << "\t" << "0x"
         << llvm::utohexstr(bankLimits[i].startAddr) << "-0x"
         << llvm::utohexstr(bankLimits[i].endAddr - 1) << "\n";
    for (auto buffer : preAllocatedBuffers) {
      auto addrOpt = buffer.getAddress();
      auto memBankOpt = buffer.getMemBank();
      assert(addrOpt.has_value() && memBankOpt.has_value() &&
             "pre-allocated buffers have both address and mem_bank set");
      if (*memBankOpt == i)
        printbuffer(buffer.name(), *addrOpt, buffer.getAllocationSize());
    }
    for (auto buffer : allocatedBuffers) {
      auto addrOpt = buffer.getAddress();
      auto memBankOpt = buffer.getMemBank();
      assert(addrOpt.has_value() && memBankOpt.has_value() &&
             "allocated buffers have both address and mem_bank set");
      if (*memBankOpt == i)
        printbuffer(buffer.name(), *addrOpt, buffer.getAllocationSize());
    }
  }
}

// Function that given a buffer will iterate over all the memory banks
// starting from the given index to try and find a bank with enough
// space. If it does, it will set the buffer's address and mem_bank
// attributes and update the nextAddrInBanks vector.
// If it does not find one with enough space, it will throw an error.
// Returns true if the buffer was successfully allocated, false otherwise.
// If no bank has enough space to accommodate the buffer, an error is emitted.

static int setBufferAddress(BufferOp buffer, int numBanks,
                            uint32_t tileAlignBitWidth,
                            uint32_t maxVecAlignBits, int &startBankIndex,
                            std::vector<int64_t> &nextAddrInBanks,
                            std::vector<BankLimits> &bankLimits) {
  assert(startBankIndex < numBanks &&
         "Unexpected input value for startBankIndex");
  int bankIndex = startBankIndex;
  bool allocated = false;
  for (int i = 0; i < numBanks; i++) {
    int64_t startAddr = nextAddrInBanks[bankIndex];

    if (buffer.getAligned()) {
      startAddr = getAlignedAddress(
          startAddr,
          getRequiredAlignBits(buffer, tileAlignBitWidth, maxVecAlignBits));
    }

    int64_t endAddr = startAddr + buffer.getAllocationSize();
    if (endAddr <= bankLimits[bankIndex].endAddr) {
      buffer.setMemBank(bankIndex);
      setAndUpdateAddressInBank(buffer, startAddr, endAddr, nextAddrInBanks);
      allocated = true;
      bankIndex = (bankIndex + 1) % numBanks;
      startBankIndex = bankIndex;
      break;
    }
    // Move to the next bank
    bankIndex = (bankIndex + 1) % numBanks;
  }
  // If no bank has enough space, throws error
  if (!allocated) {
    buffer.emitWarning("Failed to allocate buffer: ")
        << buffer.name() << " with size: " << buffer.getAllocationSize()
        << " bytes.";
    return false;
  }
  return true;
}

static bool checkAndPrintOverflow(TileOp tile, int numBanks, int stacksize,
                                  SmallVector<BufferOp> &allBuffers,
                                  std::vector<int64_t> &nextAddrInBanks,
                                  std::vector<BankLimits> &bankLimits) {
  bool foundOverflow = false;
  std::vector<int> overflow_banks;
  for (int i = 0; i < numBanks; i++) {
    if (nextAddrInBanks[i] > bankLimits[i].endAddr) {
      foundOverflow = true;
      overflow_banks.push_back(i);
    }
  }
  if (foundOverflow) {
    InFlightDiagnostic error =
        tile.emitWarning("allocated buffers exceeded available memory\n");
    auto &note = error.attachNote() << "Error in bank(s) : ";
    for (auto bank : overflow_banks)
      note << bank << " ";
    note << "\n";
    note << "MemoryMap:\n";
    auto printbuffer = [&](StringRef name, int address, int size) {
      note << "\t" << "\t" << name << " \t" << ": 0x"
           << llvm::utohexstr(address) << "-0x"
           << llvm::utohexstr(address + size - 1) << " \t(" << size
           << " bytes)\n";
    };
    for (int i = 0; i < numBanks; i++) {
      note << "\t" << "bank : " << i << "\t" << "0x"
           << llvm::utohexstr(bankLimits[i].startAddr) << "-0x"
           << llvm::utohexstr(bankLimits[i].endAddr - 1) << "\n";
      if (i == 0) {
        if (stacksize > 0)
          printbuffer("(stack)", 0, stacksize);
        else
          error << "(no stack allocated)\n";
      }
      for (auto buffer : allBuffers) {
        auto addrOpt = buffer.getAddress();
        auto memBankOpt = buffer.getMemBank();
        assert(addrOpt.has_value() && memBankOpt.has_value() &&
               "every allocated buffer has both address and mem_bank set");
        if (*memBankOpt == i)
          printbuffer(buffer.name(), *addrOpt, buffer.getAllocationSize());
      }
    }
    return false;
  }
  return true;
}

// Function to deallocate attributes of buffers in case of a failure
static void deAllocationBuffers(SmallVector<BufferOp> &buffers) {
  for (auto buffer : buffers) {
    buffer->removeAttr("address");
    buffer->removeAttr("mem_bank");
  }
}

static bool simpleBankAwareAllocation(TileOp tile) {
  auto device = tile->getParentOfType<AIE::DeviceOp>();
  if (!device)
    return false;

  std::vector<int64_t>
      nextAddrInBanks; // each entry is the next address available for use
                       // for that bank
                       // (e.g. nextAddrInBanks[tile_0][1] = next available
                       // address in bank 1 for tile_0)
  std::vector<BankLimits> bankLimits; // the entries contain pairs of start and
                                      // end addresses for each bank

  const auto &targetModel = getTargetModel(tile);
  int maxDataMemorySize = 0;
  uint32_t tileAlignBitWidth = 0;
  // MemTile buffers are reached by DMA rather than by core vector load/stores,
  // so only the bus width applies there.
  uint32_t maxVecAlignBitWidth = 0;
  if (tile.isMemTile()) {
    maxDataMemorySize = targetModel.getMemTileSize();
    tileAlignBitWidth = targetModel.getMemTileLoadStoreBusWidth();
    maxVecAlignBitWidth = tileAlignBitWidth;
  } else {
    maxDataMemorySize = targetModel.getLocalMemorySize();
    tileAlignBitWidth = targetModel.getComputeTileLoadStoreBusWidth();
    maxVecAlignBitWidth = targetModel.getComputeTileMaxVectorAlignBits();
  }

  int numBanks = targetModel.getNumBanks(tile.getCol(), tile.getRow());
  int bankSize = maxDataMemorySize / numBanks;

  // Address range owned by the MemTile is 0x80000.
  // Address range owned by the tile is 0x8000 in
  // AIE1 and 0x10000 in AIE2, but we need room at
  // the bottom for stack.
  int stacksize = 0;
  nextAddrInBanks.reserve(numBanks);

  for (int i = 0; i < numBanks; i++)
    nextAddrInBanks.push_back(bankSize * i);
  if (auto core = tile.getCoreOp()) {
    stacksize = core.getEffectiveStackSize();

    if (stacksize >= maxDataMemorySize) {
      tile->emitOpError("stack size exceeds local memory size");
      return false;
    }

    // The stack occupies the bottom of the tile's memory. When it is larger
    // than a single bank, spill it across consecutive banks so each bank's
    // next-free address accounts for the portion of the stack it holds.
    int remainStacksize = stacksize;
    for (int bank_idx = 0; bank_idx < numBanks && remainStacksize > 0;
         bank_idx++) {
      if (remainStacksize >= bankSize) {
        nextAddrInBanks[bank_idx] += bankSize;
        remainStacksize -= bankSize;
      } else {
        nextAddrInBanks[bank_idx] += remainStacksize;
        remainStacksize = 0;
      }
    }
  }
  fillBankLimits(numBanks, bankSize, bankLimits);

  SmallVector<BufferOp> preAllocatedBuffers;
  SmallVector<BufferOp> buffersToAlloc;
  SmallVector<BufferOp> allBuffers_on_tile;
  // Collect all the buffers for this tile.
  device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
    if (buffer.getTileOp() == tile) {
      if (!isBufferPreAllocated(buffer)) {
        buffersToAlloc.push_back(buffer);
      } else {
        preAllocatedBuffers.push_back(buffer);
      }
      allBuffers_on_tile.push_back(buffer);
    }
  });

  // First, allocate the buffer with pre-allocated address
  // Then, allocated the buffer with pre-allocated mem_bank t
  // Do it by doing a sort preAllocatedBuffers to have buffer with address
  // first, then buffer with only mem_bank
  std::sort(preAllocatedBuffers.begin(), preAllocatedBuffers.end(),
            [](BufferOp a, BufferOp b) -> bool {
              auto a_addr = a.getAddress();
              auto b_addr = b.getAddress();
              if (a_addr.has_value() && b_addr.has_value()) {
                return a_addr.value() <
                       b_addr.value(); // ascending address order
                                       // within same bank
              }
              // Address buffers before mem_bank-only buffers; otherwise
              // stable.
              return a_addr.has_value() && !b_addr.has_value();
            });

  for (auto buffer : preAllocatedBuffers) {

    auto has_addr = checkAndAddBufferWithAddress(
        buffer, numBanks, tileAlignBitWidth, maxVecAlignBitWidth,
        nextAddrInBanks, bankLimits);
    if (failed(has_addr))
      return false;
    // NOLINTNEXTLINE
    if (*has_addr)
      continue;
    auto has_bank = checkAndAddBufferWithMemBank(
        buffer, numBanks, tileAlignBitWidth, maxVecAlignBitWidth,
        nextAddrInBanks, bankLimits);
    if (failed(has_bank))
      return false;
  }

  // Sort by largest allocation size before allocating.
  std::sort(buffersToAlloc.begin(), buffersToAlloc.end(),
            [](BufferOp a, BufferOp b) {
              return a.getAllocationSize() > b.getAllocationSize();
            });

  // Set addresses for remaining buffers.
  SmallVector<BufferOp>
      allocatedBuffers; // keep track of buffers allocated in this function to
                        // be able to deallocate in case of failure and print
                        // helpful debug info about them. This does not include
                        // the pre-allocated buffers.
  int bankIndex = 0;
  for (auto buffer : buffersToAlloc) {
    // If the buffer doesn't fit in any of the bank space then
    // it prints the current memory map of the banks,
    // deallocates all the buffers, and
    // returns a failure.
    if (!setBufferAddress(buffer, numBanks, tileAlignBitWidth,
                          maxVecAlignBitWidth, bankIndex, nextAddrInBanks,
                          bankLimits)) {

      printMemMap(tile, allocatedBuffers, preAllocatedBuffers, numBanks,
                  bankLimits, stacksize);
      deAllocationBuffers(allocatedBuffers);
      return false;
    }
    allocatedBuffers.push_back(buffer);
  }
  assert(allocatedBuffers.size() == buffersToAlloc.size());

  // Sort by smallest address before printing memory map.
  std::sort(allBuffers_on_tile.begin(), allBuffers_on_tile.end(),
            [](BufferOp a, BufferOp b) {
              assert(a.getAddress().has_value() &&
                     "buffer must have address assigned");
              assert(b.getAddress().has_value() &&
                     "buffer must have address assigned");
              return a.getAddress().value() < b.getAddress().value();
            });
  // Check if memory was exceeded on any bank and print debug info.
  return checkAndPrintOverlapStackframe(stacksize, allBuffers_on_tile) &&
         checkAndPrintBufferOverlap(allBuffers_on_tile, tileAlignBitWidth,
                                    maxVecAlignBitWidth) &&
         checkAndPrintOverflow(tile, numBanks, stacksize, allBuffers_on_tile,
                               nextAddrInBanks, bankLimits);
}

static LogicalResult checkBufferScope(BufferOp buffer, DeviceOp device) {
  // Buffers are not allowed to be inside the core without being statically
  // initialized.
  Operation *parent = buffer->getParentOp();
  // Allowed to be in MemTile
  if (!isa<DeviceOp>(parent) && !isa<MemTileDMAOp>(parent) &&
      !buffer.getInitialValue().has_value()) {
    auto tile = buffer.getTileOp();
    tile->emitOpError("Buffer '")
        << buffer.name()
        << "' must be defined directly under the device scope. Currently it "
           "is nested inside a core tile.";
    return failure();
  }
  return success();
}

struct AIEAssignBufferAddressesPass
    : xilinx::AIE::impl::AIEAssignBufferAddressesBase<
          AIEAssignBufferAddressesPass> {

  AIEAssignBufferAddressesPass() = default;

  AIEAssignBufferAddressesPass(const AIEAssignBufferAddressesOptions &options) {
    clAllocScheme = options.clAllocScheme;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
    // Ensure all BufferOps are globally defined at the device level.
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (failed(checkBufferScope(buffer, device)))
        return signalPassFailure();
    });
    // Make sure all the buffers have a name
    int counter = 0;
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (!buffer.hasName()) {
        std::string name = "_anonymous";
        name += std::to_string(counter++);
        buffer->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
      }
    });

    // alloc_group overlays buffers, so validate the part of its contract that
    // is decidable before any address is handed out.
    if (!checkAllocGroups(device))
      return signalPassFailure();

    // Select allocation scheme per tile
    for (auto tile : device.getOps<TileOp>()) {
      auto tileAllocationScheme = tile.getAllocationScheme();

      if (!tileAllocationScheme)
        tileAllocationScheme = clAllocScheme;

      if (tileAllocationScheme == "basic-sequential") {
        if (!basicAllocation(tile)) {
          tile.emitOpError("Basic sequential allocation failed.");
          return signalPassFailure();
        }
      } else if (tileAllocationScheme == "bank-aware") {
        if (tileHasAllocGroup(device, tile)) {
          tile.emitOpError(
              "bank-aware allocation does not implement alloc_group "
              "overlays; use basic-sequential on this tile");
          return signalPassFailure();
        }
        if (!simpleBankAwareAllocation(tile)) {
          tile.emitOpError("Bank-aware allocation failed.");
          return signalPassFailure();
        }
      } else if (tileHasAllocGroup(device, tile)) {
        // Only basic-sequential implements overlays, so do not try bank-aware
        // first and silently drop them.
        if (!basicAllocation(tile)) {
          tile.emitOpError("Basic sequential allocation failed.");
          return signalPassFailure();
        }
      } else {
        if (!simpleBankAwareAllocation(tile)) {
          tile.emitWarning("Bank-aware allocation failed, trying basic "
                           "sequential allocation.");
          if (!basicAllocation(tile)) {
            tile.emitOpError("Basic sequential allocation also failed.");
            return signalPassFailure();
          }
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferAddressesPass() {
  return std::make_unique<AIEAssignBufferAddressesPass>();
}

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferAddressesPass(
    const AIEAssignBufferAddressesOptions &options) {
  return std::make_unique<AIEAssignBufferAddressesPass>(options);
}
