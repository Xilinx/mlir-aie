//===- AIEAssignBuffers.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIECoreMemory.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"

#include "llvm/ADT/BitVector.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEASSIGNBUFFERADDRESSES
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-assign-buffers"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

// Surfaced in the memory-map diagnostics below so a user sees the call-graph
// analysis's number alongside the stack region, reserved_data_size, and free
// space in one place, instead of cross-referencing a warning emitted earlier
// in the build log. Absent unless aiecc ran its stack analysis.
static std::optional<int64_t> getComputedStackRequirement(TileOp tile) {
  CoreOp core = tile.getCoreOp();
  if (!core)
    return std::nullopt;
  if (auto attr = core->getAttrOfType<IntegerAttr>(
          kComputedStackRequirementAttrName))
    return attr.getInt();
  return std::nullopt;
}

// Free run left for the core's own sections, computed the way
// AIETargetLdScript computes the `data` region it hands the core compiler.
static int64_t coreFreeRun(int64_t memSize, int64_t stackSize,
                           ArrayRef<BufferOp> buffers) {
  SmallVector<std::pair<int64_t, int64_t>> occupied;
  occupied.emplace_back(0, stackSize);
  for (auto buffer : buffers)
    if (auto addr = buffer.getAddress())
      occupied.emplace_back(*addr, *addr + buffer.getAllocationSize());
  return largestFreeRun(memSize, std::move(occupied)).size;
}

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

// Check that every buffer in the list is properly aligned (when its
// `aligned` attribute is set) and that no two buffers overlap. The input
// vector must be sorted by ascending address. Emits an error on the first
// offending buffer and returns false; returns true otherwise.
static bool checkAndPrintBufferOverlap(ArrayRef<BufferOp> sortedBuffers,
                                       uint32_t tileAlignBitWidth) {
  uint32_t alignByteWidth = tileAlignBitWidth / 8;
  BufferOp prev = nullptr;
  for (auto cur : sortedBuffers) {
    auto curAddrOpt = cur.getAddress();
    assert(curAddrOpt.has_value() && "buffer must have address assigned");
    int64_t curAddr = *curAddrOpt;

    // Alignment check.
    if (cur.getAligned() && alignByteWidth != 0 &&
        curAddr % alignByteWidth != 0) {
      cur.emitOpError("buffer '")
          << cur.name() << "' at address 0x" << llvm::utohexstr(curAddr)
          << " is not aligned to tile load/store bus width ("
          << tileAlignBitWidth << " bits)";
      return false;
    }

    // A zero-sized buffer covers no bytes, so it cannot overlap anything; it
    // would otherwise report a false overlap when it shares an address.
    if (cur.getAllocationSize() == 0)
      continue;

    // Overlap check against the closest buffer below in address order.
    if (prev) {
      auto prevAddrOpt = prev.getAddress();
      assert(prevAddrOpt.has_value() && "buffer must have address assigned");
      int64_t prevAddr = *prevAddrOpt;
      int64_t prevEnd = prevAddr + prev.getAllocationSize();
      if (curAddr < prevEnd) {
        cur.emitOpError("buffer '")
            << cur.name() << "' at address 0x" << llvm::utohexstr(curAddr)
            << " overlaps with '" << prev.name() << "' at address 0x"
            << llvm::utohexstr(prevAddr)
            << " (size: " << prev.getAllocationSize() << " bytes)";
        return false;
      }
    }
    prev = cur;
  }
  return true;
}

// Check if there is any overlap between the stack and the allocated buffers.
static bool checkAndPrintOverlapStackframe(int64_t stacksize,
                                           ArrayRef<BufferOp> buffers) {
  for (auto buf : buffers) {
    // A zero-sized buffer covers no bytes, so it cannot overlap the stack.
    if (buf.getAllocationSize() == 0)
      continue;
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

// A reservation the packing can't honour must fail here too, not just in
// bank-aware: a design whose bank-aware attempt fails for an unrelated reason
// falls back to basic-sequential, and the shortfall would otherwise surface
// much later as a linker region-overflow error naming neither the tile nor
// the buffers responsible.
static bool checkAndPrintReservedData(TileOp tile, int64_t freeRun,
                                      int64_t reservedData) {
  if (freeRun >= reservedData)
    return true;
  tile.emitWarning("buffers leave only ")
      << freeRun << " contiguous bytes for the core's data sections, which need "
      << reservedData << " bytes.";
  return false;
}

//===----------------------------------------------------------------------===//
// BasicAllocation : sequential alloc from largest to smallest
//===----------------------------------------------------------------------===//
static bool checkAndPrintOverflow(TileOp tile, int64_t address,
                                  int64_t maxDataMemorySize, int64_t stacksize,
                                  ArrayRef<BufferOp> buffers) {
  if (address > maxDataMemorySize) {
    InFlightDiagnostic error =
        tile.emitOpError("allocated buffers exceeded available memory\n");
    auto &note = error.attachNote() << "MemoryMap:\n";
    auto printbuffer = [&](StringRef name, int64_t address, int64_t size) {
      note << "\t" << name << " \t"
           << ": 0x" << llvm::utohexstr(address) << "-0x"
           << llvm::utohexstr(address + size - 1) << " \t(" << size
           << " bytes)\n";
    };
    if (stacksize > 0)
      printbuffer("(stack)", 0, stacksize);
    else
      note << "\t(no stack allocated)\n";
    if (auto computed = getComputedStackRequirement(tile))
      note << "\t(call-graph analysis computed this core's callees need >= "
           << *computed << " bytes of stack)\n";

    for (auto buffer : buffers) {
      auto bufferAddrOpt = buffer.getAddress();
      assert(bufferAddrOpt.has_value() && "buffer must have address assigned");
      printbuffer(buffer.name(), *bufferAddrOpt, buffer.getAllocationSize());
    }
    return false;
  }
  return true;
}

static bool basicAllocation(TileOp tile) {
  auto device = tile->getParentOfType<AIE::DeviceOp>();
  if (!device)
    return false;

  const auto &targetModel = getTargetModel(tile);
  int maxDataMemorySize = 0;
  uint32_t tileAlignBitWidth = 0;
  if (tile.isMemTile()) {
    maxDataMemorySize = targetModel.getMemTileSize();
    tileAlignBitWidth = targetModel.getMemTileLoadStoreBusWidth();
  } else {
    maxDataMemorySize = targetModel.getLocalMemorySize();
    tileAlignBitWidth = targetModel.getComputeTileLoadStoreBusWidth();
  }

  SmallVector<BufferOp> buffers;
  SmallVector<BufferOp> allocated_buffers;
  SmallVector<BufferOp> allBuffers_on_tile;
  // Collect all the buffers for this tile. If the buffer has an address, add
  // it to allocated_buffers. Otherwise, add it to buffers.
  device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
    if (buffer.getTileOp() == tile) {
      if (buffer.getAddress()) {
        allocated_buffers.push_back(buffer);
      } else {
        // Basic-sequential ignores mem_bank entirely (the bit-packed bump
        // pointer below has no notion of banks), so a mem_bank left over
        // here -- whether requested directly under this scheme, or rolled
        // back by a bank-aware attempt that failed and fell back to this one
        // -- describes a placement this scheme is not going to produce.
        // Clear it rather than let the address this scheme does assign carry
        // a bank label that may no longer be true.
        buffer->removeAttr("mem_bank");
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
  int64_t reservedData = 0;
  if (auto core = tile.getCoreOp()) {
    stacksize = core.getEffectiveStackSize();
    address += stacksize;
    // Space the core's own compiled sections need; see reserved_data_size.
    reservedData = core.getReservedDataSize().value_or(0);
  }

  // Ensure alignment of preallocated buffer
  for (auto buffer : allocated_buffers) {
    auto bufferAddrOpt = buffer.getAddress();
    assert(bufferAddrOpt.has_value() &&
           "allocated_buffers only holds buffers with an address");
    if (buffer.getAligned() && *bufferAddrOpt % (tileAlignBitWidth / 8) != 0) {
      buffer.emitOpError("pre-allocated address must be aligned to tile "
                         "load/store bus width when aligned attribute is set");
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
  for (auto buffer : buffers) {
    assert(!buffer.getAddress());
    if (buffer.getAligned())
      address = getAlignedAddress(address, tileAlignBitWidth);
    while (current_alloc != allocated_buffers.end() &&
           address + buffer.getAllocationSize() >
               current_alloc->getAddress().value()) {
      address = current_alloc->getAddress().value() +
                current_alloc->getAllocationSize();
      if (buffer.getAligned())
        address = getAlignedAddress(address, tileAlignBitWidth);
      current_alloc++;
    }

    buffer.setAddress(address);
    address += buffer.getAllocationSize();
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
          checkAndPrintBufferOverlap(allBuffers_on_tile, tileAlignBitWidth) &&
          checkAndPrintOverflow(tile, highWater, maxDataMemorySize, stacksize,
                                allBuffers_on_tile) &&
          checkAndPrintReservedData(
              tile,
              coreFreeRun(maxDataMemorySize, stacksize, allBuffers_on_tile),
              reservedData));
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

namespace {
// Byte-granular map of which bytes of one tile's data memory are taken.
//
// This replaces the "next free address" watermark this allocator kept per
// bank. A watermark is a bump pointer and cannot represent a hole, so a buffer
// pinned at a fixed address stranded every free byte below it. A tile is at
// most 512 kB, so tracking occupancy exactly costs at most a 64 kB bitmap.
// Granularity is one byte rather than the load/store bus width because buffers
// marked `aligned = false` are packed at unaligned offsets on purpose.
class MemoryOccupancy {
public:
  explicit MemoryOccupancy(int64_t size) : occupied(size, false) {}

  int64_t size() const { return occupied.size(); }

  // True when [start, end) lies inside the tile and no byte of it is taken.
  bool isRangeFree(int64_t start, int64_t end) const {
    if (start < 0 || end > size() || start > end)
      return false;
    return start == end || occupied.find_first_in(start, end) == -1;
  }

  void markOccupied(int64_t start, int64_t end) {
    assert(start >= 0 && end <= size() && start <= end &&
           "range must lie inside the tile");
    if (start < end)
      occupied.set(start, end);
  }

  // Start of the tightest gap in [lo, hi) that holds `size` bytes, or nullopt.
  // Ties go to the lowest address, so placement is deterministic. Candidate
  // starts are aligned up *before* the fit test, so a gap is never rejected
  // just because its first free byte is misaligned.
  std::optional<int64_t> findGap(int64_t lo, int64_t hi, int64_t size,
                                 int64_t alignBytes) const {
    assert(alignBytes > 0 && "alignment must be positive");
    lo = std::max<int64_t>(lo, 0);
    hi = std::min(hi, this->size());
    std::optional<int64_t> best;
    int64_t bestSlack = 0;
    for (int64_t cursor = lo; cursor < hi;) {
      int gapStart = occupied.find_first_unset_in(cursor, hi);
      if (gapStart == -1)
        break;
      int nextTaken = occupied.find_first_in(gapStart, hi);
      // find_first_in cannot return gapStart (it is clear), so gapEnd >
      // gapStart >= cursor and the cursor always advances.
      int64_t gapEnd = nextTaken == -1 ? hi : nextTaken;

      int64_t start = llvm::alignTo(gapStart, alignBytes);
      if (start + size <= gapEnd) {
        // Waste is measured across the whole gap, so a hole that would lose a
        // lot of its front to alignment padding is not mistaken for a tight
        // fit and preferred over one that genuinely wastes less.
        int64_t slack = (gapEnd - gapStart) - size;
        if (!best || slack < bestSlack) {
          best = start;
          bestSlack = slack;
        }
      }
      cursor = gapEnd;
    }
    return best;
  }

private:
  llvm::BitVector occupied;
};
} // namespace

// Alignment a buffer must be placed at, in bytes. Buffers marked
// `aligned = false` may start anywhere.
static int64_t getBufferAlignBytes(BufferOp buffer,
                                   uint32_t tileAlignBitWidth) {
  if (!buffer.getAligned())
    return 1;
  return std::max<int64_t>(tileAlignBitWidth / 8, 1);
}

// Index of the bank owning `addr`, or -1 when it falls outside every bank.
static int getBankContaining(int64_t addr, int numBanks,
                             ArrayRef<BankLimits> bankLimits) {
  for (int i = 0; i < numBanks; i++)
    if (addr >= bankLimits[i].startAddr && addr < bankLimits[i].endAddr)
      return i;
  return -1;
}

// Sets the buffer's address and mem_bank attributes and marks the bytes it
// covers as taken.
static void placeBuffer(BufferOp buffer, int64_t startAddr, int bank,
                        MemoryOccupancy &occupancy) {
  buffer.setAddress(startAddr);
  buffer.setMemBank(bank);
  occupancy.markOccupied(startAddr, startAddr + buffer.getAllocationSize());
}

// Places a buffer carrying an explicit `address`, checking that the space it
// asks for is free and that any `mem_bank` it also carries agrees. Returns
// false when the buffer has no address at all, leaving it to the mem_bank or
// free-placement path; returns failure when the address is unusable.
//
// Every failure here has already emitted an error, so the caller must treat it
// as terminal rather than falling back to another scheme: basic-sequential
// rejects the same pins for its own reasons except the mem_bank/address
// disagreement, which it would "honour" by silently ignoring the requested
// bank.
static FailureOr<bool>
checkAndAddBufferWithAddress(BufferOp buffer, int numBanks,
                             uint32_t tileAlignBitWidth,
                             MemoryOccupancy &occupancy,
                             ArrayRef<BankLimits> bankLimits) {
  auto addrOpt = buffer.getAddress();
  if (!addrOpt)
    return false;
  // it is fine if mem_bank is not set
  auto memBankOpt = buffer.getMemBank();

  int64_t addr = *addrOpt;
  if (buffer.getAligned() &&
      addr % getBufferAlignBytes(buffer, tileAlignBitWidth) != 0) {
    return buffer->emitOpError(
        "address attribute value must be aligned to tile load/store bus width "
        "when aligned attribute is set");
  }

  int bank = getBankContaining(addr, numBanks, bankLimits);
  if (bank < 0)
    return buffer->emitOpError(
        "address attribute does not fall within any bank range");

  int64_t endAddr = addr + buffer.getAllocationSize();
  if (endAddr > occupancy.size())
    return buffer->emitOpError("address attribute would place the buffer past "
                               "the end of the tile's memory");

  // A pinned address is only invalid when it actually collides with something
  // already placed. Note the buffer is deliberately *not* required to stay
  // inside `bank`: the hardware places no natural-size or bank alignment
  // requirement on a buffer, so a pinned buffer may straddle a bank boundary.
  if (!occupancy.isRangeFree(addr, endAddr))
    return buffer->emitOpError("would override allocated address");

  if (memBankOpt && *memBankOpt != bank)
    return buffer->emitOpError(
        "mem_bank attribute is inconsistent with address attribute");
  placeBuffer(buffer, addr, bank, occupancy);
  return true;
}

// Bank a buffer is required to live in because the user asked for it. This is
// recorded separately from the `mem_bank` attribute because the allocator also
// writes that attribute for buffers it places itself, and clears it when
// rolling back a failed attempt.
using RequiredBanks = DenseMap<Operation *, int>;

// A buffer carrying only a `mem_bank` still has to be given an address, so it
// is placed with the other unplaced buffers rather than ahead of them; this
// just records the request and rejects a bank that does not exist.
static LogicalResult recordRequiredBank(BufferOp buffer, int numBanks,
                                        RequiredBanks &requiredBanks) {
  auto memBankOpt = buffer.getMemBank();
  if (!memBankOpt)
    return success();
  if (*memBankOpt < 0 || *memBankOpt >= numBanks)
    return buffer->emitOpError("mem_bank attribute value is out of range");
  requiredBanks[buffer] = *memBankOpt;
  return success();
}

// Prints the memory map across banks
static void printMemMap(TileOp tile, ArrayRef<BufferOp> allocatedBuffers,
                        ArrayRef<BufferOp> preAllocatedBuffers, int numBanks,
                        ArrayRef<BankLimits> bankLimits, int64_t stacksize) {
  InFlightDiagnostic error = tile.emitWarning(
      "Not all requested buffers fit in the available memory.\n");
  auto &note = error.attachNote()
               << "Current configuration of buffers in bank(s) : ";
  note << "MemoryMap:\n";
  auto printbuffer = [&](StringRef name, int64_t address, int64_t size) {
    note << "\t"
         << "\t" << name << " \t"
         << ": 0x" << llvm::utohexstr(address) << "-0x"
         << llvm::utohexstr(address + size - 1) << " \t(" << size
         << " bytes)\n";
  };
  for (int i = 0; i < numBanks; i++) {
    if (i == 0) {
      if (stacksize > 0)
        printbuffer("(stack)", 0, stacksize);
      else
        note << "\t(no stack allocated)\n";
      if (auto computed = getComputedStackRequirement(tile))
        note << "\t(call-graph analysis computed this core's callees need >= "
             << *computed << " bytes of stack)\n";
    }
    note << "\t"
         << "bank : " << i << "\t"
         << "0x" << llvm::utohexstr(bankLimits[i].startAddr) << "-0x"
         << llvm::utohexstr(bankLimits[i].endAddr - 1) << "\n";
    // This runs on the failure path, so some buffers have no address yet.
    auto printPlaced = [&](ArrayRef<BufferOp> buffers) {
      for (auto buffer : buffers) {
        auto addrOpt = buffer.getAddress();
        auto memBankOpt = buffer.getMemBank();
        if (addrOpt && memBankOpt && *memBankOpt == i)
          printbuffer(buffer.name(), *addrOpt, buffer.getAllocationSize());
      }
    };
    printPlaced(preAllocatedBuffers);
    printPlaced(allocatedBuffers);
  }
}

// Places a buffer in the first bank, starting round-robin from the given
// index, that has a hole big enough, taking the tightest such hole so that
// large clean regions stay available for large buffers.
//
// Spreading over banks limits DMA bank contention; it is not a bound on how
// large a buffer may be. So a buffer that fits in no single bank straddles
// bank boundaries instead of being rejected, preferring a bank-aligned start
// because that touches the fewest banks for a given size.
//
// `preferBankAligned` asks a straddling buffer to start on a bank boundary.
// That is the placement touching the fewest banks, but it can strand up to a
// bank of space ahead of the buffer, so the caller retries without it rather
// than reporting a tile full.
//
// Returns false if there is no room for the buffer at all; the caller reports
// which buffer failed.
static bool setBufferAddress(BufferOp buffer, int numBanks,
                             uint32_t tileAlignBitWidth, int &startBankIndex,
                             bool preferBankAligned, bool spreadAcrossBanks,
                             const RequiredBanks &requiredBanks,
                             MemoryOccupancy &occupancy,
                             ArrayRef<BankLimits> bankLimits) {
  assert(startBankIndex < numBanks &&
         "Unexpected input value for startBankIndex");
  int64_t size = buffer.getAllocationSize();
  int64_t alignBytes = getBufferAlignBytes(buffer, tileAlignBitWidth);

  auto place = [&](int64_t startAddr, int bank) {
    placeBuffer(buffer, startAddr, bank, occupancy);
    startBankIndex = (bank + 1) % numBanks;
    return true;
  };

  // A requested mem_bank is a hard constraint: no other bank, and no
  // straddling. It may use any hole inside its own bank. It also leaves the
  // round-robin cursor alone, since the user chose this bank, not the spread.
  auto required = requiredBanks.find(buffer);
  if (required != requiredBanks.end()) {
    int bank = required->second;
    if (auto startAddr =
            occupancy.findGap(bankLimits[bank].startAddr,
                              bankLimits[bank].endAddr, size, alignBytes)) {
      placeBuffer(buffer, *startAddr, bank, occupancy);
      return true;
    }
    // A zero-sized buffer covers no bytes, so no bank can be too full to hold
    // it; findGap only failed for want of a free byte it will never use.
    if (size == 0) {
      placeBuffer(buffer, bankLimits[bank].startAddr, bank, occupancy);
      return true;
    }
    return false;
  }

  // Spreading buffers over banks limits DMA contention, but it also chops the
  // free space into per-bank holes. When the core needs a large contiguous run
  // for its own data, the caller turns spreading off and everything packs from
  // the bottom instead.
  if (spreadAcrossBanks) {
    for (int i = 0; i < numBanks; i++) {
      int bank = (startBankIndex + i) % numBanks;
      if (auto startAddr =
              occupancy.findGap(bankLimits[bank].startAddr,
                                bankLimits[bank].endAddr, size, alignBytes))
        return place(*startAddr, bank);
    }
  }

  // No single bank can hold it; straddle banks rather than give up. Search
  // only the banked region so the result always maps back to a bank.
  int64_t bankedEnd = bankLimits.back().endAddr;
  int64_t bankSize = bankedEnd / numBanks;
  std::optional<int64_t> startAddr;
  // Only ask for bank-boundary starts when a bank boundary actually satisfies
  // the buffer's own alignment; otherwise this would silently search on a
  // stride that is neither a bank boundary nor what was intended.
  if (preferBankAligned && bankSize % alignBytes == 0)
    startAddr = occupancy.findGap(0, bankedEnd, size, bankSize);
  if (!startAddr)
    startAddr = occupancy.findGap(0, bankedEnd, size, alignBytes);
  if (startAddr) {
    int bank = getBankContaining(*startAddr, numBanks, bankLimits);
    assert(bank >= 0 && "a gap inside the banked region belongs to a bank");
    return place(*startAddr, bank);
  }

  // A zero-sized buffer covers no bytes, so a tile with no hole left in it can
  // still hold one. It is excluded from the overlap checks for the same
  // reason.
  if (size == 0)
    return place(0, 0);

  return false;
}

// Places every buffer in `buffersToAlloc`, in order, into the free space left
// by the pre-allocated ones. Returns the first buffer that did not fit, or
// nullptr when they all did; `placed` collects what was assigned so a failed
// attempt can be rolled back.
static BufferOp placeFreeBuffers(ArrayRef<BufferOp> buffersToAlloc,
                                 int numBanks, uint32_t tileAlignBitWidth,
                                 bool preferBankAligned, bool spreadAcrossBanks,
                                 const RequiredBanks &requiredBanks,
                                 MemoryOccupancy &occupancy,
                                 ArrayRef<BankLimits> bankLimits,
                                 SmallVectorImpl<BufferOp> &placed) {
  int startBankIndex = 0;
  for (auto buffer : buffersToAlloc) {
    if (!setBufferAddress(buffer, numBanks, tileAlignBitWidth, startBankIndex,
                          preferBankAligned, spreadAcrossBanks, requiredBanks,
                          occupancy, bankLimits))
      return buffer;
    placed.push_back(buffer);
  }
  return nullptr;
}

// Rolls back what the allocator wrote, leaving a mem_bank the user asked for
// in place.
static void deAllocationBuffers(SmallVectorImpl<BufferOp> &buffers,
                                const RequiredBanks &requiredBanks) {
  for (auto buffer : buffers) {
    buffer->removeAttr("address");
    if (!requiredBanks.count(buffer))
      buffer->removeAttr("mem_bank");
  }
}

// Why bank-aware allocation stopped. A design that simply does not fit may be
// worth retrying with another scheme; a constraint the user wrote and that
// cannot be honoured must not be, because the other scheme ignores mem_bank
// and would silently place the buffer in a different bank than was asked for.
enum class BankAwareResult { Success, OutOfMemory, ConstraintUnsatisfiable };

static BankAwareResult simpleBankAwareAllocation(TileOp tile) {
  auto device = tile->getParentOfType<AIE::DeviceOp>();
  if (!device)
    return BankAwareResult::OutOfMemory;

  std::vector<BankLimits> bankLimits; // the entries contain pairs of start and
                                      // end addresses for each bank

  const auto &targetModel = getTargetModel(tile);
  int maxDataMemorySize = 0;
  uint32_t tileAlignBitWidth = 0;
  if (tile.isMemTile()) {
    maxDataMemorySize = targetModel.getMemTileSize();
    tileAlignBitWidth = targetModel.getMemTileLoadStoreBusWidth();
  } else {
    maxDataMemorySize = targetModel.getLocalMemorySize();
    tileAlignBitWidth = targetModel.getComputeTileLoadStoreBusWidth();
  }

  int numBanks = targetModel.getNumBanks(tile.getCol(), tile.getRow());
  int bankSize = maxDataMemorySize / numBanks;

  // Address range owned by the MemTile is 0x80000.
  // Address range owned by the tile is 0x8000 in
  // AIE1 and 0x10000 in AIE2, but we need room at
  // the bottom for stack.
  int stacksize = 0;
  int64_t reservedData = 0;
  MemoryOccupancy occupancy(maxDataMemorySize);
  if (auto core = tile.getCoreOp()) {
    stacksize = core.getEffectiveStackSize();
    occupancy.markOccupied(0, std::min<int64_t>(stacksize, maxDataMemorySize));
    // Space the core's own compiled sections need; see reserved_data_size.
    reservedData = core.getReservedDataSize().value_or(0);
  }
  fillBankLimits(numBanks, bankSize, bankLimits);

  RequiredBanks requiredBanks;
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
        buffer, numBanks, tileAlignBitWidth, occupancy, bankLimits);
    // An error has already been emitted; retrying under another scheme would
    // either hit the same problem or, for a mem_bank/address disagreement,
    // "succeed" by ignoring the bank the design asked for.
    if (failed(has_addr))
      return BankAwareResult::ConstraintUnsatisfiable;
    // NOLINTNEXTLINE
    if (*has_addr)
      continue;
    // Only a mem_bank: the address is still ours to choose, so queue it with
    // the rest instead of letting a small constrained buffer carve up the free
    // space before the large buffers have had a chance at it.
    if (failed(recordRequiredBank(buffer, numBanks, requiredBanks)))
      return BankAwareResult::ConstraintUnsatisfiable;
    buffersToAlloc.push_back(buffer);
  }

  // Set addresses for remaining buffers.
  SmallVector<BufferOp>
      allocatedBuffers; // keep track of buffers allocated in this function to
                        // be able to deallocate in case of failure and print
                        // helpful debug info about them. This does not include
                        // the pre-allocated buffers.

  // Packing around fixed obstacles has no cheap optimal answer, so rather than
  // trusting one greedy order, try a few and keep the first that fits. Each
  // attempt is a handful of bitmap scans.
  //
  // Buffers always go largest first, so small ones cannot fragment the clean
  // regions large ones need. The two knobs are whether a buffer pinned to a
  // bank goes before the unconstrained ones -- placing it first guarantees it a
  // home, placing it later stops it bisecting a free run that spans banks --
  // and whether a straddling buffer starts on a bank boundary, which touches
  // the fewest banks but strands the space ahead of it.
  // The last entry gives up bank spreading entirely and packs from the bottom;
  // it is the only one that can leave a large contiguous run for the core's own
  // data, so it is what a tight `reserved_data_size` falls back to.
  static constexpr struct {
    bool bankConstrainedFirst;
    bool preferBankAligned;
    bool spreadAcrossBanks;
  } strategies[] = {{true, true, true},
                    {true, false, true},
                    {false, false, true},
                    {true, false, false}};

  MemoryOccupancy pinnedOnly = occupancy;
  BufferOp failed = nullptr;
  // The best free run any strategy managed to leave, and whether one placed
  // every buffer at all. A later strategy failing on placement must not hide
  // an earlier one that fit but left too little room for the core's own data:
  // that shortfall is the actionable diagnostic, and it is what the user has
  // to act on.
  bool placedEverything = false;
  int64_t bestFreeRun = 0;
  for (auto strategy : strategies) {
    deAllocationBuffers(allocatedBuffers, requiredBanks);
    allocatedBuffers.clear();
    occupancy = pinnedOnly;
    // Sorted into a fresh vector each time, so every strategy's order is
    // relative to the original walk order rather than to whatever the previous
    // strategy's sort happened to leave behind.
    SmallVector<BufferOp> order(buffersToAlloc);
    llvm::stable_sort(order, [&](BufferOp a, BufferOp b) {
      if (strategy.bankConstrainedFirst &&
          requiredBanks.count(a) != requiredBanks.count(b))
        return requiredBanks.count(a) > requiredBanks.count(b);
      return a.getAllocationSize() > b.getAllocationSize();
    });
    failed = placeFreeBuffers(order, numBanks, tileAlignBitWidth,
                              strategy.preferBankAligned,
                              strategy.spreadAcrossBanks, requiredBanks,
                              occupancy, bankLimits, allocatedBuffers);
    if (failed)
      continue;
    // Placing every buffer is not enough: the core compiler is handed the
    // largest gap left over, so a layout that fits but strands the core's own
    // data is no good either. Both must hold before a strategy is accepted.
    placedEverything = true;
    int64_t freeRun =
        coreFreeRun(maxDataMemorySize, stacksize, allBuffers_on_tile);
    bestFreeRun = std::max(bestFreeRun, freeRun);
    if (freeRun >= reservedData)
      break;
  }
  if (placedEverything && bestFreeRun < reservedData) {
    // Every buffer fit, but not with enough contiguous room left over for the
    // core's own data, which the linker script hands out as a single region.
    tile.emitWarning("buffers leave only ")
        << bestFreeRun << " contiguous bytes for the core's data sections, "
        << "which need " << reservedData
        << " bytes. Every buffer was placed, so the memory map is not the "
           "interesting part; the free space is simply too broken up.";
    deAllocationBuffers(allocatedBuffers, requiredBanks);
    return BankAwareResult::OutOfMemory;
  }
  if (failed) {
    // A buffer pinned to a bank that cannot hold it is a constraint the user
    // wrote, not a tile that ran out of room, so it gets its own error and no
    // memory map.
    if (requiredBanks.count(failed)) {
      failed->emitOpError("would override existing mem_bank");
      deAllocationBuffers(allocatedBuffers, requiredBanks);
      return BankAwareResult::ConstraintUnsatisfiable;
    }
    failed.emitWarning("Failed to allocate buffer: ")
        << failed.name() << " with size: " << failed.getAllocationSize()
        << " bytes.";
    // The memory map reads the addresses handed out, so print before rolling
    // them back.
    printMemMap(tile, allocatedBuffers, preAllocatedBuffers, numBanks,
                bankLimits, stacksize);
    deAllocationBuffers(allocatedBuffers, requiredBanks);
    return BankAwareResult::OutOfMemory;
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
  // Every placement above was taken from free space inside the tile, so a
  // bank/tile overflow is no longer representable here; the stack and overlap
  // checks remain as a backstop over the final addresses.
  if (!checkAndPrintOverlapStackframe(stacksize, allBuffers_on_tile) ||
      !checkAndPrintBufferOverlap(allBuffers_on_tile, tileAlignBitWidth))
    return BankAwareResult::OutOfMemory;
  return BankAwareResult::Success;
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
        if (simpleBankAwareAllocation(tile) != BankAwareResult::Success) {
          tile.emitOpError("Bank-aware allocation failed.");
          return signalPassFailure();
        }
      } else {
        switch (simpleBankAwareAllocation(tile)) {
        case BankAwareResult::Success:
          break;
        case BankAwareResult::ConstraintUnsatisfiable:
          // Basic sequential allocation ignores mem_bank, so retrying there
          // would "succeed" by putting the buffer in a bank the design did not
          // ask for. Report the constraint instead.
          tile.emitOpError("Bank-aware allocation failed.");
          return signalPassFailure();
        case BankAwareResult::OutOfMemory:
          tile.emitWarning("Bank-aware allocation failed, trying basic "
                           "sequential allocation.");
          if (!basicAllocation(tile)) {
            tile.emitOpError("Basic sequential allocation also failed.");
            return signalPassFailure();
          }
          break;
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
