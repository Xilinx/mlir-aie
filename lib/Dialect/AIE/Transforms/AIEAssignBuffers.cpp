//===- AIEAssignBuffers.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIECoreMemory.h"
#include "aie/Dialect/AIE/IR/AIECoreSymbols.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"

#include "llvm/ADT/BitVector.h"

#include <optional>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEASSIGNBUFFERADDRESSES
#define GEN_PASS_DEF_AIEPREPAREBUFFERS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-assign-buffers"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

// Absent unless aiecc measured the core.
static std::optional<int64_t> getMeasuredStackSize(TileOp tile) {
  CoreOp core = tile.getCoreOp();
  if (!core)
    return std::nullopt;
  if (auto measured = core.getMeasuredStackSize())
    return static_cast<int64_t>(*measured);
  return std::nullopt;
}

// A memtile is reached by DMA rather than by core vector load and store, so its
// bus width also covers the vector-alignment requirement.
struct TileMemoryLimits {
  int64_t maxDataMemorySize;
  uint32_t tileAlignBitWidth;
  uint32_t maxVecAlignBits;
};
static TileMemoryLimits tileMemoryLimits(TileOp tile,
                                         const AIETargetModel &targetModel) {
  if (tile.isMemTile())
    return {targetModel.getMemTileSize(),
            targetModel.getMemTileLoadStoreBusWidth(),
            targetModel.getMemTileLoadStoreBusWidth()};
  return {targetModel.getLocalMemorySize(),
          targetModel.getComputeTileLoadStoreBusWidth(),
          targetModel.getComputeTileMaxVectorAlignBits()};
}

// Every buffer must already have an address.
static void sortBuffersByAddress(SmallVectorImpl<BufferOp> &buffers) {
  llvm::sort(buffers, [](BufferOp a, BufferOp b) {
    assert(a.getAddress().has_value() && "buffer must have address assigned");
    assert(b.getAddress().has_value() && "buffer must have address assigned");
    return a.getAddress().value() < b.getAddress().value();
  });
}

// The region the core compiler receives for its own sections.
// AIETargetLdScript calls largestFreeRun on the same interval list, so the
// stamped value equals the emitted `data` region.
static MemoryRun coreDataRun(int64_t memSize, int64_t stackSize,
                             ArrayRef<BufferOp> buffers, int64_t alignBytes) {
  SmallVector<std::pair<int64_t, int64_t>> occupied;
  occupied.emplace_back(0, stackSize);
  for (auto buffer : buffers)
    if (auto addr = buffer.getAddress())
      occupied.emplace_back(*addr, *addr + buffer.getAllocationSize());
  return largestFreeRun(memSize, std::move(occupied), alignBytes);
}

// Record where the core's data region lies, for AIETargetLdScript. A memtile
// and a shim have no compiled sections and no linker script.
static void stampCoreDataRegion(TileOp tile, MemoryRun run) {
  CoreOp core = tile.getCoreOp();
  if (!core)
    return;
  Builder b(tile.getContext());
  core.setDataOriginAttr(b.getI32IntegerAttr(run.start));
  core.setDataLengthAttr(b.getI32IntegerAttr(run.size));
}

// Clear the stamp of an earlier run or of a failed attempt.
static void clearCoreDataStamp(TileOp tile) {
  if (CoreOp core = tile.getCoreOp()) {
    core->removeAttr("data_origin");
    core->removeAttr("data_length");
  }
}

static bool isBufferPreAllocated(BufferOp buffer) {
  auto addr = buffer.getAddress();
  auto memBank = buffer.getMemBank();
  return (addr != std::nullopt || memBank != std::nullopt);
}

// Round a byte address up to the given alignment, expressed in bits.
static int64_t getAlignedAddress(int64_t address, uint32_t alignBitWidth) {
  assert(alignBitWidth != 0 && alignBitWidth % 8 == 0 &&
         "alignBitWidth must be a non-zero multiple of 8");
  return llvm::alignTo(address, alignBitWidth / 8);
}

// Return the alignment (in bits) `buffer` must satisfy.
//
// Bus width alone is insufficient: from AIE2P on, a full-width vector access
// needs 512-bit alignment while the bus is 256 bits wide. An externally
// compiled kernel may perform such an access, so a buffer large enough to hold
// a full-width vector gets the stricter alignment. A smaller buffer keeps the
// bus width and costs no padding.
static uint32_t getRequiredAlignBits(BufferOp buffer, uint32_t busAlignBits,
                                     uint32_t maxVecAlignBits) {
  if (maxVecAlignBits <= busAlignBits)
    return busAlignBits;
  int64_t sizeBits = static_cast<int64_t>(buffer.getAllocationSize()) * 8;
  return sizeBits >= static_cast<int64_t>(maxVecAlignBits) ? maxVecAlignBits
                                                           : busAlignBits;
}

// Check alignment (when `aligned` is set) and that no two buffers overlap. The
// input vector must be sorted by ascending address. Returns false and emits an
// error on the first offending buffer; true otherwise.
static bool checkAndPrintBufferOverlap(ArrayRef<BufferOp> sortedBuffers,
                                       uint32_t tileAlignBitWidth,
                                       uint32_t maxVecAlignBits) {
  BufferOp prev = nullptr;
  for (auto cur : sortedBuffers) {
    auto curAddrOpt = cur.getAddress();
    assert(curAddrOpt.has_value() && "buffer must have address assigned");
    int64_t curAddr = *curAddrOpt;

    // A pinned buffer must satisfy the bus width. The stricter vector
    // requirement applies to the addresses this pass chooses.
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

    // A zero-sized buffer covers no bytes; skip it to avoid a false overlap
    // when it shares an address.
    if (cur.getAllocationSize() == 0)
      continue;

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

// Report a reservation shortfall here, because the linker's region-overflow
// error names neither the tile nor the buffers. A bank-aware attempt that falls
// back to basic-sequential reaches this.
static bool checkAndPrintReservedData(TileOp tile, int64_t freeRun,
                                      int64_t reservedData) {
  if (freeRun >= reservedData)
    return true;
  tile.emitWarning("buffers leave only ")
      << freeRun
      << " contiguous bytes for the core's data sections, which need "
      << reservedData << " bytes.";
  return false;
}

// One line of a memory-map diagnostic. Both diagnostics below call this, so
// they format their lines identically.
static void printMemoryMapEntry(Diagnostic &note, StringRef name,
                                int64_t address, int64_t size, int indent,
                                StringRef suffix = "") {
  for (int i = 0; i < indent; ++i)
    note << "\t";
  int64_t end = size == 0 ? address : address + size - 1;
  note << name << " \t"
       << ": 0x" << llvm::utohexstr(address) << "-0x" << llvm::utohexstr(end)
       << " \t(" << size << " bytes)" << suffix << "\n";
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
      printMemoryMapEntry(note, name, address, size, /*indent=*/1);
    };
    if (stacksize > 0)
      printbuffer("(stack)", 0, stacksize);
    else
      note << "\t(no stack allocated)\n";
    if (auto measured = getMeasuredStackSize(tile))
      note << "\t(aiecc measured this core's stack requirement as " << *measured
           << " bytes)\n";

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

  clearCoreDataStamp(tile);

  auto [maxDataMemorySize, tileAlignBitWidth, maxVecAlignBits] =
      tileMemoryLimits(tile, getTargetModel(tile));

  SmallVector<BufferOp> buffers;
  SmallVector<BufferOp> allocated_buffers;
  SmallVector<BufferOp> allBuffers_on_tile;
  device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
    if (buffer.getTileOp() == tile) {
      if (buffer.getAddress()) {
        allocated_buffers.push_back(buffer);
      } else {
        // This scheme packs from the bottom of the tile and has no notion of
        // banks, so it clears mem_bank: the label would name a bank the
        // address need not fall in. A user pin reaches this only under an
        // explicit basic-sequential request, because the fallback path refuses
        // to drop one (see BankAwareResult::OutOfMemory).
        if (buffer.getMemBank())
          buffer->emitWarning("basic-sequential allocation ignores mem_bank; "
                              "dropping the pin on: ")
              << buffer.name();
        buffer->removeAttr("mem_bank");
        buffers.push_back(buffer);
      }
      allBuffers_on_tile.push_back(buffer);
    }
  });

  // Stable, so buffers of equal size keep program order, as the bank-aware
  // path's placementOrder does.
  llvm::stable_sort(buffers, [](BufferOp a, BufferOp b) {
    return a.getAllocationSize() > b.getAllocationSize();
  });

  llvm::stable_sort(allocated_buffers, [](BufferOp a, BufferOp b) {
    return a.getAddress().value() < b.getAddress().value();
  });

  // Stack lives at the bottom of the tile's data memory.
  int64_t stacksize = 0;
  int64_t address = 0;
  int64_t reservedData = 0;
  if (auto core = tile.getCoreOp()) {
    stacksize = core.getEffectiveStackSize();
    address += stacksize;
    reservedData = core.getReservedDataSize().value_or(0);
  }

  for (auto buffer : allocated_buffers) {
    auto bufferAddrOpt = buffer.getAddress();
    assert(bufferAddrOpt.has_value() &&
           "allocated_buffers only holds buffers with an address");
    // A pinned address must satisfy the bus width. The stricter size-inferred
    // vector requirement applies to the addresses this pass chooses.
    if (buffer.getAligned() && *bufferAddrOpt % (tileAlignBitWidth / 8) != 0) {
      buffer.emitOpError("pre-allocated address must be aligned to ")
          << tileAlignBitWidth << " bits when the aligned attribute is set";
      return false;
    }
  }

  // Align the address before and after each skip, so the fit test runs on the
  // final placement. A misaligned candidate otherwise passes the fit test, and
  // getAlignedAddress then bumps it into a pre-allocated buffer.
  auto *current_alloc = allocated_buffers.begin();
  for (auto buffer : buffers) {
    assert(!buffer.getAddress());
    uint32_t reqAlignBits =
        getRequiredAlignBits(buffer, tileAlignBitWidth, maxVecAlignBits);
    if (buffer.getAligned())
      address = getAlignedAddress(address, reqAlignBits);
    while (current_alloc != allocated_buffers.end() &&
           address + buffer.getAllocationSize() >
               current_alloc->getAddress().value()) {
      address = current_alloc->getAddress().value() +
                current_alloc->getAllocationSize();
      if (buffer.getAligned())
        address = getAlignedAddress(address, reqAlignBits);
      current_alloc++;
    }

    buffer.setAddress(address);
    address += buffer.getAllocationSize();
  }

  sortBuffersByAddress(allBuffers_on_tile);

  // High-water mark across all buffers, including pre-allocated buffers above
  // the allocation cursor, so checkAndPrintOverflow reads a correct bound.
  int64_t highWater = address;
  if (!allBuffers_on_tile.empty()) {
    auto &last = allBuffers_on_tile.back();
    auto lastAddrOpt = last.getAddress();
    assert(lastAddrOpt.has_value() && "buffer must have address assigned");
    highWater =
        std::max<int64_t>(highWater, *lastAddrOpt + last.getAllocationSize());
  }

  MemoryRun dataRun =
      coreDataRun(maxDataMemorySize, stacksize, allBuffers_on_tile,
                  std::max<int64_t>(maxVecAlignBits / 8, 1));
  if (!checkAndPrintOverlapStackframe(stacksize, allBuffers_on_tile) ||
      !checkAndPrintBufferOverlap(allBuffers_on_tile, tileAlignBitWidth,
                                  maxVecAlignBits) ||
      !checkAndPrintOverflow(tile, highWater, maxDataMemorySize, stacksize,
                             allBuffers_on_tile) ||
      !checkAndPrintReservedData(tile, dataRun.size, reservedData))
    return false;
  stampCoreDataRegion(tile, dataRun);
  return true;
}

//===----------------------------------------------------------------------===//
// SimpleBankAwareAllocation : round-robin each alloc over available banks
//===----------------------------------------------------------------------===//
// Compute the extent of each bank.
static void fillBankLimits(int numBanks, int64_t bankSize,
                           std::vector<MemoryRun> &bankLimits) {
  for (int i = 0; i < numBanks; i++)
    bankLimits.push_back({bankSize * i, bankSize});
}

namespace {
// Which bytes of one tile's data memory are taken. Byte-granular, because a
// buffer marked `aligned = false` is packed at an unaligned offset.
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

  // Undo a markOccupied. The caller lifts the core's data-region placeholder
  // once every buffer is placed. No buffer lies inside the placeholder's range,
  // so clearing exactly that range is exact.
  void markFree(int64_t start, int64_t end) {
    assert(start >= 0 && end <= size() && start <= end &&
           "range must lie inside the tile");
    if (start < end)
      occupied.reset(start, end);
  }

  // Placement for `size` bytes in [lo, hi) that leaves the largest single free
  // run behind anywhere in [0, size()), or nullopt when nothing fits. An object
  // placed before the unconstrained buffers -- the core's data region, a
  // bank-pinned buffer -- keeps more contiguous space when it sits flush
  // against occupied space. Only the two flush positions of each hole are
  // candidates, because an interior placement leaves strictly less contiguous
  // space.
  std::optional<int64_t> findLeastFragmentingGap(int64_t lo, int64_t hi,
                                                 int64_t size,
                                                 int64_t alignBytes) const {
    assert(alignBytes > 0 && "alignment must be positive");
    std::optional<int64_t> best;
    int64_t bestRun = -1;
    auto consider = [&](int64_t start) {
      MemoryOccupancy trial = *this;
      trial.markOccupied(start, start + size);
      int64_t run = trial.largestGap(0, this->size());
      // Ties to the lowest address, for determinism.
      if (run > bestRun || (run == bestRun && best && start < *best)) {
        bestRun = run;
        best = start;
      }
    };
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      int64_t low = llvm::alignTo(gapStart, alignBytes);
      if (low + size > gapEnd)
        return;
      consider(low);
      // Flush to the top, aligned down but never below `low`.
      consider(
          std::max<int64_t>(llvm::alignDown(gapEnd - size, alignBytes), low));
    });
    return best;
  }

  // Start of the tightest gap in [lo, hi) that holds `size` bytes, or nullopt.
  // Each candidate start is aligned up before the fit test, so a misaligned
  // first free byte does not reject a gap.
  std::optional<int64_t> findGap(int64_t lo, int64_t hi, int64_t size,
                                 int64_t alignBytes) const {
    assert(alignBytes > 0 && "alignment must be positive");
    std::optional<int64_t> best;
    int64_t bestSlack = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      int64_t start = llvm::alignTo(gapStart, alignBytes);
      if (start + size <= gapEnd) {
        // Waste measured across the whole gap, so front-alignment padding
        // counts and a padded hole does not rank as a tight fit.
        int64_t slack = (gapEnd - gapStart) - size;
        if (!best || slack < bestSlack) {
          best = start;
          bestSlack = slack;
        }
      }
    });
    return best;
  }

  // Total free bytes in [lo, hi).
  int64_t freeBytes(int64_t lo, int64_t hi) const {
    int64_t total = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      total += gapEnd - gapStart;
    });
    return total;
  }

  // Bytes of the enclosing free run that `size` bytes at `addr` would leave
  // unused; placement scoring prefers a tight fit to keep large runs whole.
  int64_t slackAt(int64_t addr, int64_t size) const {
    int64_t slack = 0;
    forEachGap(0, this->size(), [&](int64_t gapStart, int64_t gapEnd) {
      if (addr >= gapStart && addr + size <= gapEnd)
        slack = (gapEnd - gapStart) - size;
    });
    return slack;
  }

  // The free run enclosing `addr`, or a zero-length run if it is taken.
  MemoryRun gapAt(int64_t addr) const {
    MemoryRun found;
    forEachGap(0, this->size(), [&](int64_t gapStart, int64_t gapEnd) {
      if (addr >= gapStart && addr < gapEnd)
        found = {gapStart, gapEnd - gapStart};
    });
    return found;
  }

  // The two largest free runs by size, duplicates kept, so two equal runs give
  // {n, n}. A placement consumes one run and leaves the other whole, which lets
  // a candidate score the run it leaves behind in constant time.
  std::pair<int64_t, int64_t> topTwoGaps(int64_t lo, int64_t hi) const {
    int64_t first = 0, second = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      int64_t run = gapEnd - gapStart;
      if (run > first) {
        second = first;
        first = run;
      } else if (run > second) {
        second = run;
      }
    });
    return {first, second};
  }

  // Largest single free run in [lo, hi). A reservation needs one contiguous
  // run, which freeBytes does not measure.
  int64_t largestGap(int64_t lo, int64_t hi) const {
    int64_t best = 0;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      best = std::max(best, gapEnd - gapStart);
    });
    return best;
  }

private:
  // Calls fn(gapStart, gapEnd) for every maximal free run in [lo, hi).
  template <typename Fn>
  void forEachGap(int64_t lo, int64_t hi, Fn fn) const {
    lo = std::max<int64_t>(lo, 0);
    hi = std::min(hi, size());
    for (int64_t cursor = lo; cursor < hi;) {
      int gapStart = occupied.find_first_unset_in(cursor, hi);
      if (gapStart == -1)
        break;
      int nextTaken = occupied.find_first_in(gapStart, hi);
      // gapStart is clear, so gapEnd > gapStart >= cursor: the cursor advances.
      int64_t gapEnd = nextTaken == -1 ? hi : nextTaken;
      fn(gapStart, gapEnd);
      cursor = gapEnd;
    }
  }

  llvm::BitVector occupied;
};
} // namespace

// Alignment a buffer must be placed at, in bytes. Buffers marked
// `aligned = false` may start anywhere.
static int64_t getBufferAlignBytes(BufferOp buffer, uint32_t tileAlignBitWidth,
                                   uint32_t maxVecAlignBits) {
  if (!buffer.getAligned())
    return 1;
  return std::max<int64_t>(
      getRequiredAlignBits(buffer, tileAlignBitWidth, maxVecAlignBits) / 8, 1);
}

// Index of the bank owning `addr`, or -1 when it falls outside every bank.
static int getBankContaining(int64_t addr, int numBanks,
                             ArrayRef<MemoryRun> bankLimits) {
  for (int i = 0; i < numBanks; i++)
    if (bankLimits[i].contains(addr))
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

// Tile-level constants shared by every bank-aware helper below.
struct BankAwareContext {
  int numBanks;
  uint32_t tileAlignBitWidth;
  uint32_t maxVecAlignBits;
  ArrayRef<MemoryRun> bankLimits;
  int64_t maxDataMemorySize;
  int64_t stacksize;
  int64_t reservedData;
  // Whether this tile has a core, and so compiled sections that need a
  // contiguous region. A memtile has none.
  bool hasCore;
};

// Places a buffer carrying an explicit `address`, and checks that the space is
// free and that any `mem_bank` agrees. Returns false when the buffer has no
// address, which leaves it to the mem_bank or free-placement path. Failure is
// terminal: this emits an error and the caller must not retry another scheme
// (see BankAwareResult below).
static FailureOr<bool>
checkAndAddBufferWithAddress(BufferOp buffer, const BankAwareContext &ctx,
                             MemoryOccupancy &occupancy) {
  auto addrOpt = buffer.getAddress();
  if (!addrOpt)
    return false;
  // it is fine if mem_bank is not set
  auto memBankOpt = buffer.getMemBank();

  int64_t addr = *addrOpt;
  // A pinned address must satisfy the bus width. The stricter vector alignment
  // applies to the addresses this pass chooses.
  if (buffer.getAligned() && addr % (ctx.tileAlignBitWidth / 8) != 0) {
    return buffer->emitOpError("address attribute value must be aligned to ")
           << ctx.tileAlignBitWidth
           << " bits when the aligned attribute is set";
  }

  int bank = getBankContaining(addr, ctx.numBanks, ctx.bankLimits);
  if (bank < 0) {
    // A zero-sized buffer pinned one past the last bank is legal. Assign it the
    // last bank, where nothing can conflict with it.
    if (buffer.getAllocationSize() == 0 && addr == ctx.bankLimits.back().end())
      bank = ctx.numBanks - 1;
    else
      return buffer->emitOpError(
          "address attribute does not fall within any bank range");
  }

  int64_t endAddr = addr + buffer.getAllocationSize();
  if (endAddr > occupancy.size())
    return buffer->emitOpError("address attribute would place the buffer past "
                               "the end of the tile's memory");

  // Only a real collision is invalid. The hardware lets a buffer straddle a
  // bank boundary, so a pinned buffer may extend past `bank`.
  if (!occupancy.isRangeFree(addr, endAddr))
    return buffer->emitOpError("would override allocated address");

  if (memBankOpt && *memBankOpt != bank)
    return buffer->emitOpError(
        "mem_bank attribute is inconsistent with address attribute");
  placeBuffer(buffer, addr, bank, occupancy);
  return true;
}

// Bank a buffer must live in because the user requested it. Held separately
// from the `mem_bank` attribute, which the allocator writes for the buffers it
// places and clears when it rolls back.
using RequiredBanks = DenseMap<Operation *, int>;

// Records a mem_bank request and rejects a bank that does not exist. A
// mem_bank-only buffer still needs an address, so it is placed with the other
// unplaced buffers.
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

static void printMemMap(TileOp tile, ArrayRef<BufferOp> allocatedBuffers,
                        ArrayRef<BufferOp> preAllocatedBuffers,
                        const BankAwareContext &ctx) {
  InFlightDiagnostic error = tile.emitWarning(
      "Not all requested buffers fit in the available memory.\n");
  auto &note = error.attachNote()
               << "Current configuration of buffers in bank(s) : ";
  note << "MemoryMap:\n";
  auto printbuffer = [&](StringRef name, int64_t address, int64_t size,
                         StringRef suffix = "") {
    printMemoryMapEntry(note, name, address, size, /*indent=*/2, suffix);
  };
  for (int i = 0; i < ctx.numBanks; i++) {
    if (i == 0) {
      if (ctx.stacksize > 0)
        printbuffer("(stack)", 0, ctx.stacksize);
      else
        note << "\t(no stack allocated)\n";
      if (auto measured = getMeasuredStackSize(tile))
        note << "\t(aiecc measured this core's stack requirement as "
             << *measured << " bytes)\n";
    }
    note << "\t"
         << "bank : " << i << "\t"
         << "0x" << llvm::utohexstr(ctx.bankLimits[i].start) << "-0x"
         << llvm::utohexstr(ctx.bankLimits[i].end() - 1) << "\n";
    // This runs on the failure path, where some buffers have no address yet.
    auto printPlaced = [&](ArrayRef<BufferOp> buffers) {
      for (auto buffer : buffers) {
        auto addrOpt = buffer.getAddress();
        auto memBankOpt = buffer.getMemBank();
        if (!addrOpt || !memBankOpt || *memBankOpt != i)
          continue;
        int64_t size = buffer.getAllocationSize();
        // Listed under its start bank. A buffer too big for one bank straddles,
        // so its range can extend past the bank printed above it.
        std::string suffix;
        if (*addrOpt + size > ctx.bankLimits[i].end())
          suffix = (" (straddles into bank " + llvm::Twine(i + 1) + ")").str();
        printbuffer(buffer.name(), *addrOpt, size, suffix);
      }
    };
    printPlaced(preAllocatedBuffers);
    printPlaced(allocatedBuffers);
  }
}

// Places a buffer in the tightest hole that fits, and spreads buffers over the
// banks round-robin to limit DMA contention. A buffer that fits no single bank
// straddles bank boundaries. Returns false when no room remains; the caller
// reports which buffer failed.
static bool setBufferAddress(BufferOp buffer, const BankAwareContext &ctx,
                             int &startBankIndex,
                             const RequiredBanks &requiredBanks,
                             MemoryOccupancy &occupancy) {
  assert(startBankIndex < ctx.numBanks &&
         "Unexpected input value for startBankIndex");
  int64_t size = buffer.getAllocationSize();
  int64_t alignBytes =
      getBufferAlignBytes(buffer, ctx.tileAlignBitWidth, ctx.maxVecAlignBits);

  auto place = [&](int64_t startAddr, int bank) {
    placeBuffer(buffer, startAddr, bank, occupancy);
    startBankIndex = (bank + 1) % ctx.numBanks;
    return true;
  };

  // A requested mem_bank is a hard constraint: one bank, no straddling, and the
  // round-robin cursor stays where it is. This placement precedes the
  // unconstrained buffers, so it takes the spot in its bank that leaves the
  // largest single run behind.
  auto required = requiredBanks.find(buffer);
  if (required != requiredBanks.end()) {
    int bank = required->second;
    if (auto startAddr = occupancy.findLeastFragmentingGap(
            ctx.bankLimits[bank].start, ctx.bankLimits[bank].end(), size,
            alignBytes)) {
      placeBuffer(buffer, *startAddr, bank, occupancy);
      return true;
    }
    // A zero-sized buffer needs no free byte, so every bank holds it.
    if (size == 0) {
      placeBuffer(buffer, ctx.bankLimits[bank].start, bank, occupancy);
      return true;
    }
    return false;
  }

  // Score every candidate placement and keep the best, ranked by:
  //  1. banks touched, fewest first (each spanned bank costs DMA bandwidth);
  //  2. largest free run left behind, biggest first;
  //  3. round-robin distance from the cursor, nearest first (spreads for DMA);
  //  4. slack, tightest first, which leaves the large runs unbroken;
  //  5. address, lowest first, for determinism.
  //
  // Criterion 2 gives the core's own .data, .rodata and .bss somewhere to live.
  // The linker script grants one region, the largest gap between the stack and
  // the buffers, so placement has to maximize that gap. Criterion 2 outranks
  // bank spreading because spreading costs contiguity, and a core whose
  // sections do not fit does not link at all.
  //
  // A memtile has no compiled sections, so criterion 2 is neutralized there and
  // round-robin governs, which spreads its buffers across banks for DMA
  // bandwidth.
  int64_t bankedEnd = ctx.bankLimits.back().end();
  int64_t bankSize = bankedEnd / ctx.numBanks;
  // Computed once per buffer, not per candidate: every candidate splits one
  // run, so the largest run left elsewhere is one of these two. Named
  // separately, because C++17 cannot capture a structured binding in the lambda
  // below.
  std::pair<int64_t, int64_t> topGaps =
      occupancy.topTwoGaps(0, ctx.maxDataMemorySize);
  int64_t largestRun = topGaps.first, secondRun = topGaps.second;
  struct Candidate {
    int64_t addr;
    int bank;
    int64_t banksTouched;
    int64_t negLargestRunLeft;
    int64_t rrDistance;
    int64_t slack;
  };
  std::optional<Candidate> best;
  auto consider = [&](std::optional<int64_t> addr) {
    if (!addr)
      return;
    int bank = getBankContaining(*addr, ctx.numBanks, ctx.bankLimits);
    if (bank < 0)
      return;
    int64_t touched =
        size == 0 ? 1 : (*addr + size - 1) / bankSize - *addr / bankSize + 1;
    // This placement splits one run into the piece below and the piece above.
    // Every other run stays whole, so the biggest of those is whichever of the
    // top two this candidate does not consume.
    MemoryRun gap = occupancy.gapAt(*addr);
    int64_t elsewhere = gap.size == largestRun ? secondRun : largestRun;
    int64_t leftPiece = *addr - gap.start;
    int64_t rightPiece = gap.end() - (*addr + size);
    int64_t runLeft = std::max({elsewhere, leftPiece, rightPiece});
    Candidate c{*addr,
                bank,
                touched,
                ctx.hasCore ? -runLeft : 0,
                (bank - startBankIndex + ctx.numBanks) % ctx.numBanks,
                occupancy.slackAt(*addr, size)};
    auto rank = [](const Candidate &x) {
      return std::tie(x.banksTouched, x.negLargestRunLeft, x.rrDistance,
                      x.slack, x.addr);
    };
    if (!best || rank(c) < rank(*best))
      best = c;
  };

  for (int i = 0; i < ctx.numBanks; i++)
    consider(occupancy.findGap(ctx.bankLimits[i].start, ctx.bankLimits[i].end(),
                               size, alignBytes));
  // Allow straddling when nothing fits one bank. The banked region maps to a
  // bank, so the result always has one.
  consider(occupancy.findGap(0, bankedEnd, size, alignBytes));
  // Try bank-boundary starts only when a boundary also satisfies the buffer's
  // own alignment, because otherwise this searches on an unintended stride.
  if (bankSize % alignBytes == 0)
    consider(occupancy.findGap(0, bankedEnd, size, bankSize));

  if (best)
    return place(best->addr, best->bank);

  // A zero-sized buffer needs no free byte, so a full tile still holds it.
  if (size == 0)
    return place(0, 0);

  return false;
}

// Places every buffer in `buffersToAlloc`, in order. Returns the first buffer
// that did not fit, or nullptr when they all did; `placed` collects what was
// assigned so a failed attempt can be rolled back.
static BufferOp placeFreeBuffers(ArrayRef<BufferOp> buffersToAlloc,
                                 const BankAwareContext &ctx,
                                 const RequiredBanks &requiredBanks,
                                 MemoryOccupancy &occupancy,
                                 SmallVectorImpl<BufferOp> &placed) {
  int startBankIndex = 0;
  for (auto buffer : buffersToAlloc) {
    if (!setBufferAddress(buffer, ctx, startBankIndex, requiredBanks,
                          occupancy))
      return buffer;
    placed.push_back(buffer);
  }
  return nullptr;
}

// Rolls back what the allocator wrote, and leaves a mem_bank the user requested
// in place.
static void deAllocationBuffers(SmallVectorImpl<BufferOp> &buffers,
                                const RequiredBanks &requiredBanks) {
  for (auto buffer : buffers) {
    buffer->removeAttr("address");
    if (!requiredBanks.count(buffer))
      buffer->removeAttr("mem_bank");
  }
}

// Why bank-aware allocation stopped. OutOfMemory permits a retry with another
// scheme. ConstraintUnsatisfiable forbids one, because basic-sequential ignores
// mem_bank and would silently place the buffer in a different bank.
namespace {
enum class BankAwareResult { Success, OutOfMemory, ConstraintUnsatisfiable };
} // namespace

// Places every buffer carrying an explicit `address`, address pins first, so a
// mem_bank-only buffer cannot carve up space an address pin needs. A
// mem_bank-only buffer enters `requiredBanks` and queues into
// `buffersToAlloc`. Failure is terminal (see BankAwareResult).
static LogicalResult placePreAllocatedBuffers(
    SmallVectorImpl<BufferOp> &preAllocatedBuffers, const BankAwareContext &ctx,
    MemoryOccupancy &occupancy, RequiredBanks &requiredBanks,
    SmallVectorImpl<BufferOp> &buffersToAlloc) {
  // Address buffers first (ascending), then mem_bank-only buffers. The sort is
  // stable, so the mem_bank-only tail, whose elements all compare equal, keeps
  // program order and places identically from one run to the next.
  llvm::stable_sort(preAllocatedBuffers, [](BufferOp a, BufferOp b) -> bool {
    auto a_addr = a.getAddress();
    auto b_addr = b.getAddress();
    if (a_addr.has_value() && b_addr.has_value())
      return a_addr.value() < b_addr.value();
    return a_addr.has_value() && !b_addr.has_value();
  });

  for (auto buffer : preAllocatedBuffers) {
    auto has_addr = checkAndAddBufferWithAddress(buffer, ctx, occupancy);
    if (failed(has_addr))
      return failure();
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (*has_addr)
      continue;
    // Only a mem_bank: this pass still chooses the address, so queue the
    // buffer with the rest.
    if (failed(recordRequiredBank(buffer, ctx.numBanks, requiredBanks)))
      return failure();
    buffersToAlloc.push_back(buffer);
  }
  return success();
}

// Order buffers for placement: bank-pinned first (most-constrained-variable, as
// a pin has one candidate bank), then largest first, so a small buffer does not
// split the run a large one needs.
static SmallVector<BufferOp>
placementOrder(ArrayRef<BufferOp> buffersToAlloc,
               const RequiredBanks &requiredBanks) {
  SmallVector<BufferOp> order(buffersToAlloc.begin(), buffersToAlloc.end());
  llvm::stable_sort(order, [&](BufferOp a, BufferOp b) {
    if (requiredBanks.count(a) != requiredBanks.count(b))
      return requiredBanks.count(a) > requiredBanks.count(b);
    return a.getAllocationSize() > b.getAllocationSize();
  });
  return order;
}

// Reserve `reservedData` contiguous bytes for the core's own sections, as a
// placeholder the free buffers cannot place into. This runs after the
// bank-pinned buffers, because a region placed first can take the only space a
// user's mem_bank pin needs and turn a working design into a constraint error.
// Returns the placeholder's range, or nullopt (with a diagnostic) when no run
// is large enough.
static std::optional<MemoryRun>
reserveCoreDataRegion(TileOp tile, const BankAwareContext &ctx,
                      MemoryOccupancy &occupancy) {
  if (ctx.reservedData <= 0)
    return MemoryRun{0, 0};
  // Align the origin: the linker starts `.data` at a multiple of its strongest
  // section alignment, so an unaligned origin loses that space to padding.
  int64_t alignBytes = std::max<int64_t>(ctx.maxVecAlignBits / 8, 1);
  if (auto start = occupancy.findLeastFragmentingGap(
          0, ctx.maxDataMemorySize, ctx.reservedData, alignBytes)) {
    occupancy.markOccupied(*start, *start + ctx.reservedData);
    return MemoryRun{*start, ctx.reservedData};
  }

  // This runs before any free buffer is placed, so only the stack and the
  // user's own pins can be in the way.
  tile.emitOpError("cannot reserve ")
      << ctx.reservedData
      << " contiguous bytes for this core's data sections "
         "(reserved_data_size); the largest free run is "
      << occupancy.largestGap(0, ctx.maxDataMemorySize)
      << " bytes. Only the stack and this tile's address- or bank-pinned "
         "buffers are placed at this point, so it is one of those, or the "
         "reservation itself, that has to give";
  return std::nullopt;
}

static BankAwareResult simpleBankAwareAllocation(TileOp tile) {
  auto device = tile->getParentOfType<AIE::DeviceOp>();
  if (!device)
    return BankAwareResult::OutOfMemory;

  clearCoreDataStamp(tile);

  std::vector<MemoryRun> bankLimits;

  const auto &targetModel = getTargetModel(tile);
  auto [maxDataMemorySize, tileAlignBitWidth, maxVecAlignBits] =
      tileMemoryLimits(tile, targetModel);

  int numBanks = targetModel.getNumBanks(tile.getCol(), tile.getRow());
  int64_t bankSize = maxDataMemorySize / numBanks;

  // Stack lives at the bottom of the tile's data memory.
  int64_t stacksize = 0;
  int64_t reservedData = 0;
  MemoryOccupancy occupancy(maxDataMemorySize);
  if (auto core = tile.getCoreOp()) {
    stacksize = core.getEffectiveStackSize();
    occupancy.markOccupied(0, std::min<int64_t>(stacksize, maxDataMemorySize));
    reservedData = core.getReservedDataSize().value_or(0);
  }
  fillBankLimits(numBanks, bankSize, bankLimits);
  BankAwareContext ctx{numBanks,     tileAlignBitWidth,     maxVecAlignBits,
                       bankLimits,   maxDataMemorySize,     stacksize,
                       reservedData, (bool)tile.getCoreOp()};

  RequiredBanks requiredBanks;
  SmallVector<BufferOp> preAllocatedBuffers;
  SmallVector<BufferOp> buffersToAlloc;
  SmallVector<BufferOp> allBuffers_on_tile;
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

  if (failed(placePreAllocatedBuffers(preAllocatedBuffers, ctx, occupancy,
                                      requiredBanks, buffersToAlloc)))
    return BankAwareResult::ConstraintUnsatisfiable;

  // Buffers this pass placed (not the pre-allocated ones), for rollback and
  // diagnostics on failure.
  SmallVector<BufferOp> allocatedBuffers;

  // One pass, three steps: place the bank-pinned prefix, reserve the core's
  // data region in what remains, then fill the unconstrained buffers around it.
  // The reservation is what makes one pass sufficient.
  SmallVector<BufferOp> order = placementOrder(buffersToAlloc, requiredBanks);
  auto *pinnedEnd = llvm::partition_point(
      order, [&](BufferOp b) { return requiredBanks.count(b) > 0; });
  ArrayRef<BufferOp> pinnedPrefix(order.begin(), pinnedEnd);
  ArrayRef<BufferOp> freeSuffix(pinnedEnd, order.end());

  BufferOp failedBuffer = placeFreeBuffers(pinnedPrefix, ctx, requiredBanks,
                                           occupancy, allocatedBuffers);

  std::optional<MemoryRun> dataPlaceholder;
  if (!failedBuffer) {
    dataPlaceholder = reserveCoreDataRegion(tile, ctx, occupancy);
    if (!dataPlaceholder) {
      // reserveCoreDataRegion already reported the obstacle. Continue to
      // basic-sequential, which packs from the bottom and may leave room.
      deAllocationBuffers(allocatedBuffers, requiredBanks);
      return BankAwareResult::OutOfMemory;
    }
    failedBuffer = placeFreeBuffers(freeSuffix, ctx, requiredBanks, occupancy,
                                    allocatedBuffers);
  }

  if (BufferOp failed = failedBuffer) {
    // A buffer pinned to a bank that cannot hold it is a user constraint, not
    // an out-of-room tile: give it its own error and no memory map.
    if (requiredBanks.count(failed)) {
      int bank = requiredBanks.lookup(failed);
      int64_t need = failed.getAllocationSize();
      int64_t bankCapacity = bankLimits[bank].size;
      if (need > bankCapacity)
        failed->emitOpError("requires ")
            << need << " bytes, which cannot fit in bank " << bank << " ("
            << bankCapacity << " bytes total)";
      else
        failed->emitOpError("requires ")
            << need << " bytes in bank " << bank << ", but only "
            << occupancy.freeBytes(bankLimits[bank].start,
                                   bankLimits[bank].end())
            << " of " << bankCapacity << " bytes are free there";
      deAllocationBuffers(allocatedBuffers, requiredBanks);
      return BankAwareResult::ConstraintUnsatisfiable;
    }
    failed.emitWarning("Failed to allocate buffer: ")
        << failed.name() << " with size: " << failed.getAllocationSize()
        << " bytes.";
    // The reservation is a bitmap placeholder, not a buffer, so the memory map
    // does not list it. Report whether lifting it would have been enough, which
    // tells the user whether to shrink reserved_data_size or free the tile.
    if (dataPlaceholder && dataPlaceholder->size) {
      MemoryOccupancy without = occupancy;
      without.markFree(dataPlaceholder->start,
                       dataPlaceholder->start + dataPlaceholder->size);
      int64_t need = failed.getAllocationSize();
      bool wouldFit =
          without.largestGap(0, ctx.maxDataMemorySize) >= need || need == 0;
      failed.emitRemark("this core reserves ")
          << dataPlaceholder->size
          << " bytes for its own data sections (reserved_data_size), placed at "
             "0x"
          << llvm::utohexstr(dataPlaceholder->start) << "; '"
          << failed.name().getValue() << "' "
          << (wouldFit ? "would have fit without that reservation"
                       : "would not have fit even without that reservation");
    }
    // Print before rollback, while the addresses are still set.
    printMemMap(tile, allocatedBuffers, preAllocatedBuffers, ctx);
    deAllocationBuffers(allocatedBuffers, requiredBanks);
    return BankAwareResult::OutOfMemory;
  }
  assert(allocatedBuffers.size() == buffersToAlloc.size());

  sortBuffersByAddress(allBuffers_on_tile);
  // Every placement came from free space in the tile, so overflow cannot happen
  // here. The stack and overlap checks remain as a backstop.
  if (!checkAndPrintOverlapStackframe(stacksize, allBuffers_on_tile) ||
      !checkAndPrintBufferOverlap(allBuffers_on_tile, tileAlignBitWidth,
                                  maxVecAlignBits))
    return BankAwareResult::OutOfMemory;
  // Lift the placeholder and stamp the run that survives. That run absorbs
  // whatever the buffers leave unused, so the grant is at least the request.
  if (dataPlaceholder && dataPlaceholder->size)
    occupancy.markFree(dataPlaceholder->start,
                       dataPlaceholder->start + dataPlaceholder->size);
  stampCoreDataRegion(
      tile, coreDataRun(maxDataMemorySize, stacksize, allBuffers_on_tile,
                        std::max<int64_t>(maxVecAlignBits / 8, 1)));
  return BankAwareResult::Success;
}

static LogicalResult checkBufferScope(BufferOp buffer, DeviceOp device) {
  // A buffer inside a core must be statically initialized; MemTileDMA is
  // allowed too.
  Operation *parent = buffer->getParentOp();
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

namespace {
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
          // Can't retry under basic-sequential (see BankAwareResult).
          tile.emitOpError("Bank-aware allocation failed.");
          return signalPassFailure();
        case BankAwareResult::OutOfMemory: {
          // A mem_bank-only buffer would silently lose its bank under
          // basic-sequential, even when its own pin is satisfiable and
          // bank-aware failed for an unrelated reason. Only an address pin is
          // safe to retry, because basic-sequential honours it.
          SmallVector<BufferOp> droppedPins;
          device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
            if (buffer.getTileOp() == tile && buffer.getMemBank() &&
                !buffer.getAddress())
              droppedPins.push_back(buffer);
          });
          if (!droppedPins.empty()) {
            InFlightDiagnostic diag = tile.emitOpError(
                "bank-aware allocation failed; falling back to "
                "basic-sequential would silently drop the mem_bank pin on: ");
            for (auto [i, buffer] : llvm::enumerate(droppedPins)) {
              if (i)
                diag << ", ";
              diag << buffer.name();
            }
            return signalPassFailure();
          }
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
  }
};
} // namespace

namespace {
struct AIEPrepareBuffersPass
    : xilinx::AIE::impl::AIEPrepareBuffersBase<AIEPrepareBuffersPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (failed(checkBufferScope(buffer, device)))
        return signalPassFailure();
    });
    // A buffer's name becomes a symbol in its core's object, so every buffer
    // needs a name before the cores are lowered.
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
    int counter = 0;
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (!buffer.hasName()) {
        std::string name = "_anonymous";
        name += std::to_string(counter++);
        buffer->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
      }
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEPrepareBuffersPass() {
  return std::make_unique<AIEPrepareBuffersPass>();
}

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferAddressesPass() {
  return std::make_unique<AIEAssignBufferAddressesPass>();
}

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferAddressesPass(
    const AIEAssignBufferAddressesOptions &options) {
  return std::make_unique<AIEAssignBufferAddressesPass>(options);
}
