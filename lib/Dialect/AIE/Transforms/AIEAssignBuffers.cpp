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
#include "aie/Dialect/AIE/Transforms/AIEBufferAllocation.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"

#include "llvm/ADT/BitVector.h"

#include <limits>
#include <optional>
#include <set>

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
  if (!core) {
    return std::nullopt;
  }
  if (auto measured = core.getMeasuredStackSize()) {
    return static_cast<int64_t>(*measured);
  }
  return std::nullopt;
}

// Every buffer must already have an address.
static void sortBuffersByAddress(SmallVectorImpl<BufferOp> &buffers) {
  llvm::sort(buffers, [](BufferOp a, BufferOp b) {
    assert(a.getAddress().has_value() && "buffer must have address assigned");
    assert(b.getAddress().has_value() && "buffer must have address assigned");
    return a.getAddress().value() < b.getAddress().value();
  });
}

// How a diagnostic names an extent. The allocator creates the core_data and
// bank_reserved buffers itself, so their symbols appear in no user source and
// naming them would send the reader looking for something they never wrote.
static std::string bufferLabel(BufferOp buffer) {
  if (buffer.getCoreData()) {
    return "this core's data sections (data_size)";
  }
  if (buffer.getBankReserved()) {
    return "this core's static data pinned to bank " +
           std::to_string(buffer.getMemBank().value_or(0));
  }
  // aie-prepare-buffers names every buffer, but this pass also runs alone.
  if (!buffer.hasName()) {
    return "an unnamed buffer";
  }
  return ("buffer \"" + buffer.name().getValue() + "\"").str();
}

// The name a memory map prints for an extent.
static StringRef mapLabel(BufferOp buffer) {
  if (buffer.getCoreData()) {
    return StringRef("(core data sections)");
  }
  if (buffer.getBankReserved()) {
    return StringRef("(bank-pinned static data)");
  }
  if (!buffer.hasName()) {
    return StringRef("(unnamed buffer)");
  }
  return buffer.name().getValue();
}

// Give each extent a core's own sections hold (CoreOp::getSectionReservations)
// an aie.buffer, so placement fits around it: the data region as one more
// extent; the room a bank's pinned static data will need, held before
// unconstrained buffers take it; and a prebaked `elf_file` core's runs, pinned
// at the addresses its image already uses. Idempotent: this pass runs
// standalone and inside larger pipelines.
static LogicalResult materializeCoreSections(DeviceOp device) {
  using Kind = CoreSectionReservation::Kind;
  OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
  for (auto core : llvm::to_vector(device.getOps<CoreOp>())) {
    SmallVector<CoreSectionReservation> reservations =
        core.getSectionReservations();
    if (reservations.empty()) {
      continue;
    }
    auto tile = dyn_cast_or_null<TileOp>(core.getTile().getDefiningOp());
    if (!tile) {
      return core.emitOpError(
          "reserves memory for its own sections, but its tile is not placed; "
          "run --aie-place-tiles first");
    }
    std::string at =
        std::to_string(tile.getCol()) + "_" + std::to_string(tile.getRow());
    SmallVector<BufferOp> holders =
        core.getSectionReservationBuffers(reservations);
    for (auto [held, holder] : llvm::zip_equal(reservations, holders)) {
      if (holder) {
        continue;
      }
      std::string name =
          held.kind == Kind::DataRegion ? "core_data_" + at
          : held.kind == Kind::Bank
              ? "bank_reserved_" + at + "_" + std::to_string(held.at)
              : "prebaked_" + at + "_" + std::to_string(held.at);
      // A buffer that is not the reservation's exact extent cannot stand in
      // for it, and a second one cannot be made beside it.
      BufferOp clash;
      bool dataRegion = held.kind == Kind::DataRegion;
      device.walk([&](BufferOp buffer) {
        if (buffer.getSymName() == StringRef(name) ||
            (dataRegion && buffer.getCoreData() &&
             buffer.getTile() == tile.getResult()))
          clash = buffer;
      });
      if (clash) {
        return clash.emitOpError("does not match the ")
               << held.bytes << "-byte reservation core (" << tile.getCol()
               << ", " << tile.getRow() << ") makes for its own sections";
      }
      builder.setInsertionPoint(core);
      auto type = MemRefType::get({held.bytes}, builder.getI8Type());
      auto buffer = BufferOp::create(
          builder, core.getLoc(), type, tile.getResult(),
          /*sym_name=*/nullptr,
          /*address=*/
          held.kind == Kind::Prebaked ? builder.getI32IntegerAttr(held.at)
                                      : nullptr,
          /*initial_value=*/nullptr,
          /*mem_bank=*/
          held.kind == Kind::Bank ? builder.getI32IntegerAttr(held.at)
                                  : nullptr,
          /*core_data=*/
          held.kind == Kind::DataRegion ? builder.getUnitAttr() : nullptr,
          /*bank_reserved=*/
          held.kind == Kind::DataRegion ? nullptr : builder.getUnitAttr(),
          /*aligned=*/held.aligned() ? nullptr : builder.getBoolAttr(false));
      buffer->setAttr(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
    }
  }
  return success();
}

// A singleton address space supplies the bank pin (BufferOp::verify has
// already checked an explicit pin against it); pairs stay flexible until
// placement.
//
// The buffer carries the space rather than the kernel's declaration doing so
// alone, because `func.call` already requires operand types to match the
// callee's signature exactly, so all that is left is to place the buffer where
// its type says it lives.
static void applySignatureBankConstraints(DeviceOp device) {
  device.walk([&](BufferOp buffer) {
    SmallVector<int, 2> banks = banksFromMemorySpace(buffer.getType());
    if (banks.size() == 1 && !buffer.getMemBank())
      buffer.setMemBankAttr(IntegerAttr::get(
          IntegerType::get(device.getContext(), 32), banks.front()));
  });
}

static bool isBufferPreAllocated(BufferOp buffer) {
  auto addr = buffer.getAddress();
  auto memBank = buffer.getMemBank();
  return (addr != std::nullopt || memBank != std::nullopt);
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
            : requiredBufferAlignBits(tileAlignBitWidth, maxVecAlignBits,
                                      cur.getAllocationSize());
    uint32_t alignByteWidth = reqAlignBits / 8;
    if (cur.getAligned() && alignByteWidth != 0 &&
        curAddr % alignByteWidth != 0) {
      cur.emitOpError("") << bufferLabel(cur) << " at address 0x"
                          << llvm::utohexstr(curAddr)
                          << " is not aligned to the required " << reqAlignBits
                          << " bits";
      return false;
    }

    // A zero-sized buffer covers no bytes; skip it to avoid a false overlap
    // when it shares an address.
    if (cur.getAllocationSize() == 0) {
      continue;
    }

    if (prev) {
      auto prevAddrOpt = prev.getAddress();
      assert(prevAddrOpt.has_value() && "buffer must have address assigned");
      int64_t prevAddr = *prevAddrOpt;
      int64_t prevEnd = prevAddr + prev.getAllocationSize();
      if (curAddr < prevEnd) {
        cur.emitOpError("")
            << bufferLabel(cur) << " at address 0x" << llvm::utohexstr(curAddr)
            << " overlaps with " << bufferLabel(prev) << " at address 0x"
            << llvm::utohexstr(prevAddr)
            << " (size: " << prev.getAllocationSize() << " bytes)";
        return false;
      }
    }
    prev = cur;
  }
  return true;
}

static bool checkAndPrintOverlapStackframe(MemoryRun stackRun,
                                           ArrayRef<BufferOp> buffers) {
  for (auto buf : buffers) {
    // A zero-sized buffer covers no bytes, so it cannot overlap the stack.
    if (buf.getAllocationSize() == 0) {
      continue;
    }
    auto bufAddrOpt = buf.getAddress();
    assert(bufAddrOpt.has_value() && "buffer must have address assigned");
    int64_t bufAddr = *bufAddrOpt;
    int64_t bufEnd = bufAddr + buf.getAllocationSize();
    if (bufAddr < stackRun.end() && stackRun.start < bufEnd) {
      buf.emitOpError("") << bufferLabel(buf) << " at address 0x"
                          << llvm::utohexstr(bufAddr) << " overlaps the stack ("
                          << stackRun.size << " bytes at 0x"
                          << llvm::utohexstr(stackRun.start) << ")";
      return false;
    }
  }
  return true;
}

// One line of a memory-map diagnostic. Both diagnostics below call this, so
// they format their lines identically.
static void printMemoryMapEntry(Diagnostic &note, StringRef name,
                                int64_t address, int64_t size, int indent,
                                StringRef suffix = "") {
  for (int i = 0; i < indent; ++i) {
    note << "\t";
  }
  int64_t end = size == 0 ? address : address + size - 1;
  note << name << " \t"
       << ": 0x" << llvm::utohexstr(address) << "-0x" << llvm::utohexstr(end)
       << " \t(" << size << " bytes)" << suffix << "\n";
}

//===----------------------------------------------------------------------===//
// SimpleBankAwareAllocation : round-robin each alloc over available banks
//===----------------------------------------------------------------------===//
// Compute the extent of each bank.
static void fillBankLimits(int numBanks, int64_t bankSize,
                           std::vector<MemoryRun> &bankLimits) {
  for (int i = 0; i < numBanks; i++) {
    bankLimits.push_back({bankSize * i, bankSize});
  }
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
    if (start < 0 || end > size() || start > end) {
      return false;
    }
    return start == end || occupied.find_first_in(start, end) == -1;
  }

  void markOccupied(int64_t start, int64_t end) {
    assert(start >= 0 && end <= size() && start <= end &&
           "range must lie inside the tile");
    if (start < end) {
      occupied.set(start, end);
    }
  }

  // Undo a markOccupied. The caller lifts the core's data-region placeholder
  // once every buffer is placed. No buffer lies inside the placeholder's range,
  // so clearing exactly that range is exact.
  void markFree(int64_t start, int64_t end) {
    assert(start >= 0 && end <= size() && start <= end &&
           "range must lie inside the tile");
    if (start < end) {
      occupied.reset(start, end);
    }
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
      if (low + size > gapEnd) {
        return;
      }
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
      if (addr >= gapStart && addr + size <= gapEnd) {
        slack = (gapEnd - gapStart) - size;
      }
    });
    return slack;
  }

  // The free run enclosing `addr`, or a zero-length run if it is taken.
  MemoryRun gapAt(int64_t addr) const {
    MemoryRun found;
    forEachGap(0, this->size(), [&](int64_t gapStart, int64_t gapEnd) {
      if (addr >= gapStart && addr < gapEnd) {
        found = {gapStart, gapEnd - gapStart};
      }
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

  // Every maximal free run in [lo, hi), lowest first. The search enumerates
  // candidate placements from these rather than taking the single best run.
  SmallVector<MemoryRun> gapsIn(int64_t lo, int64_t hi) const {
    SmallVector<MemoryRun> runs;
    forEachGap(lo, hi, [&](int64_t gapStart, int64_t gapEnd) {
      runs.push_back({gapStart, gapEnd - gapStart});
    });
    return runs;
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
      if (gapStart == -1) {
        break;
      }
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

// Index of the bank owning `addr`, or -1 when it falls outside every bank.
static int getBankContaining(int64_t addr, int numBanks,
                             ArrayRef<MemoryRun> bankLimits) {
  for (int i = 0; i < numBanks; i++) {
    if (bankLimits[i].contains(addr)) {
      return i;
    }
  }
  return -1;
}

// Tile-level constants shared by every bank-aware helper below.
struct BankAwareContext {
  int numBanks;
  uint32_t tileAlignBitWidth;
  uint32_t maxVecAlignBits;
  ArrayRef<MemoryRun> bankLimits;
  int64_t maxDataMemorySize;
  MemoryRun stackRun;
  // Whether this tile has a core, and so compiled sections that need a
  // contiguous region. A memtile has none.
  bool hasCore;
  ArrayRef<TileLayoutBuffer> buffers;
  // Bytes at the start of the tile an initialized buffer must not cover; see
  // allocateTile. Other buffers may still use them.
  int64_t initGuardBytes = 0;
};

// Records buffer `i` at `startAddr` in `bank` and marks the bytes it covers as
// taken.
static void placeBuffer(size_t i, int64_t startAddr, int bank,
                        const BankAwareContext &ctx, MemoryOccupancy &occupancy,
                        TileLayout &layout) {
  layout.at[i] = {startAddr, bank};
  occupancy.markOccupied(startAddr, startAddr + ctx.buffers[i].size);
}

// Fails `layout` on buffer `i` with `message`.
static void failPin(TileLayout &layout, size_t i, const llvm::Twine &message) {
  layout.status = TileLayout::Status::BadPin;
  layout.failed = i;
  layout.message = message.str();
}

// Places a buffer carrying an explicit `address`, and checks that the space is
// free and that any `mem_bank` agrees. Returns false when the buffer has no
// address, which leaves it to the mem_bank or free-placement path. Failure is
// terminal: it fails `layout` and the caller must not retry another scheme.
static FailureOr<bool> checkAndAddBufferWithAddress(size_t i,
                                                    const BankAwareContext &ctx,
                                                    MemoryOccupancy &occupancy,
                                                    TileLayout &layout) {
  const TileLayoutBuffer &buffer = ctx.buffers[i];
  if (!buffer.address) {
    return false;
  }

  int64_t addr = *buffer.address;
  int64_t measuredAlign = buffer.sectionAlignBytes;
  if (addr % measuredAlign != 0) {
    failPin(layout, i,
            "data reservation address must be aligned to " +
                llvm::Twine(measuredAlign) +
                " bytes to satisfy its measured section alignment");
    return failure();
  }

  // A pinned address must satisfy the bus width. The stricter vector alignment
  // applies to the addresses this pass chooses.
  if (buffer.aligned && addr % (ctx.tileAlignBitWidth / 8) != 0) {
    failPin(layout, i,
            "address attribute value must be aligned to " +
                llvm::Twine(ctx.tileAlignBitWidth) +
                " bits when the aligned attribute is set");
    return failure();
  }

  int bank = getBankContaining(addr, ctx.numBanks, ctx.bankLimits);
  if (bank < 0) {
    // A zero-sized buffer pinned one past the last bank is legal. Assign it the
    // last bank, where nothing can conflict with it.
    if (buffer.size == 0 && addr == ctx.bankLimits.back().end()) {
      bank = ctx.numBanks - 1;
    } else {
      failPin(layout, i,
              "address attribute does not fall within any bank range");
      return failure();
    }
  }

  int64_t endAddr = addr + buffer.size;
  if (endAddr > occupancy.size()) {
    failPin(layout, i,
            "address attribute would place the buffer past the end of the "
            "tile's memory");
    return failure();
  }

  int lastBank = buffer.size == 0 ? bank
                                  : getBankContaining(endAddr - 1, ctx.numBanks,
                                                      ctx.bankLimits);
  for (int touched = bank; !buffer.spaceBanks.empty() && touched <= lastBank;
       ++touched) {
    if (!llvm::is_contained(buffer.spaceBanks, touched)) {
      failPin(layout, i,
              "address attribute places the buffer in bank " +
                  llvm::Twine(touched) +
                  ", which is excluded by its memory space");
      return failure();
    }
  }
  if (buffer.memBank && *buffer.memBank != bank) {
    failPin(layout, i,
            "address attribute lies in bank " + llvm::Twine(bank) +
                ", but mem_bank requests bank " + llvm::Twine(*buffer.memBank));
    return failure();
  }

  // Only a real collision is invalid. The hardware lets a buffer straddle a
  // bank boundary, so a pinned buffer may extend past `bank`.
  if (!occupancy.isRangeFree(addr, endAddr)) {
    failPin(layout, i, "would override allocated address");
    return failure();
  }

  placeBuffer(i, addr, bank, ctx, occupancy, layout);
  return true;
}

// Banks a buffer may live in because the user requested them, by the
// buffer's index in the problem.
using RequiredBanks = DenseMap<size_t, SmallVector<int, 2>>;

// Record the allowed banks, intersected with any explicit mem_bank request.
// BufferOp::verify has already checked that the intersection is nonempty.
static LogicalResult recordRequiredBank(size_t i, const BankAwareContext &ctx,
                                        RequiredBanks &requiredBanks,
                                        TileLayout &layout) {
  const TileLayoutBuffer &buffer = ctx.buffers[i];
  if (buffer.memBank &&
      (*buffer.memBank < 0 || *buffer.memBank >= ctx.numBanks)) {
    failPin(layout, i, "mem_bank attribute value is out of range");
    return failure();
  }
  SmallVector<int, 2> banks = buffer.spaceBanks;
  if (buffer.memBank)
    banks = {*buffer.memBank};
  if (!banks.empty()) {
    if (banks.back() >= ctx.numBanks) {
      failPin(layout, i, "memory space requests a bank that does not exist");
      return failure();
    }
    requiredBanks[i] = std::move(banks);
  }
  return success();
}

static void printMemMap(TileOp tile, ArrayRef<BufferOp> allocatedBuffers,
                        ArrayRef<BufferOp> preAllocatedBuffers, int numBanks,
                        ArrayRef<MemoryRun> bankLimits, MemoryRun stackRun) {
  InFlightDiagnostic error = tile.emitWarning(
      "Not all requested buffers fit in the available memory.\n");
  auto &note = error.attachNote()
               << "Current configuration of buffers in bank(s) : ";
  note << "MemoryMap:\n";
  auto printbuffer = [&](StringRef name, int64_t address, int64_t size,
                         StringRef suffix = "") {
    printMemoryMapEntry(note, name, address, size, /*indent=*/2, suffix);
  };
  if (stackRun.size == 0) {
    note << "\t(no stack allocated)\n";
  }
  if (auto measured = getMeasuredStackSize(tile)) {
    note << "\t(aiecc measured this core's stack requirement as " << *measured
         << " bytes)\n";
  }
  for (int i = 0; i < numBanks; i++) {
    note << "\t" << "bank : " << i << "\t" << "0x"
         << llvm::utohexstr(bankLimits[i].start) << "-0x"
         << llvm::utohexstr(bankLimits[i].end() - 1) << "\n";
    // Under whichever bank holds it, rather than always under bank 0.
    if (stackRun.size > 0 && bankLimits[i].contains(stackRun.start)) {
      printbuffer("(stack)", stackRun.start, stackRun.size);
    }
    // This runs on the failure path, where some buffers have no address yet.
    // A mem_bank-only buffer appears in both lists -- it is pre-allocated in
    // the sense that its bank was given, but this pass still chose its address
    // -- so listing it once takes a filter.
    SmallPtrSet<Operation *, 8> listed;
    auto printPlaced = [&](ArrayRef<BufferOp> buffers) {
      for (auto buffer : buffers) {
        auto addrOpt = buffer.getAddress();
        auto memBankOpt = buffer.getMemBank();
        if (!addrOpt || !memBankOpt || *memBankOpt != i ||
            !listed.insert(buffer).second) {
          continue;
        }
        int64_t size = buffer.getAllocationSize();
        // Listed under its start bank. A buffer too big for one bank straddles,
        // so its range can extend past the bank printed above it.
        std::string suffix;
        if (*addrOpt + size > bankLimits[i].end()) {
          suffix = (" (straddles into bank " + llvm::Twine(i + 1) + ")").str();
        }
        printbuffer(mapLabel(buffer), *addrOpt, size, suffix);
      }
    };
    printPlaced(preAllocatedBuffers);
    printPlaced(allocatedBuffers);
  }
}

namespace {
// Sentinel first key for an unscored fallback, so it sorts behind every
// scored candidate however those score.
constexpr int64_t kFallbackRank = std::numeric_limits<int64_t>::max();

// Candidate address ordering, lower ranks first:
//  1. banks touched, fewest first (each spanned bank costs DMA bandwidth);
//  2. largest free run left behind, biggest first, capped by contiguityCap --
//     the bytes still to place. A run wider than that serves nothing, so past
//     the cap every candidate ties and criterion 3 takes over;
//  3. round-robin distance from the cursor, nearest first (spreads for DMA);
//  4. slack, tightest first, which leaves the large runs unbroken;
//  5. address, lowest first, for determinism.
// A memtile has no core, so criterion 2 is neutralized there and round-robin
// governs throughout.
struct Placement {
  int64_t addr;
  int bank;
  int64_t banksTouched;
  int64_t negLargestRunLeft;
  int64_t rrDistance;
  int64_t slack;

  std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> rank() const {
    return {banksTouched, negLargestRunLeft, rrDistance, slack, addr};
  }
};
} // namespace

// Every address `buffer` could take in the current occupancy, best first.
//
// The head of this list is the address a single-pass greedy would have chosen,
// so a design that placed before places identically; the tail is where the
// search goes when a later buffer is left with nowhere to sit.
static SmallVector<Placement>
rankedPlacements(size_t buffer, const BankAwareContext &ctx, int startBankIndex,
                 const RequiredBanks &requiredBanks,
                 const MemoryOccupancy &tileOccupancy, int64_t contiguityCap) {
  assert(startBankIndex < ctx.numBanks &&
         "Unexpected input value for startBankIndex");
  // An initialized buffer sees the guarded bytes as taken, so none of its
  // candidates cover them. Every other buffer may still be placed there.
  std::optional<MemoryOccupancy> guarded;
  if (ctx.initGuardBytes > 0 && ctx.buffers[buffer].initialized) {
    guarded = tileOccupancy;
    guarded->markOccupied(0, std::min(ctx.initGuardBytes, guarded->size()));
  }
  const MemoryOccupancy &occupancy = guarded ? *guarded : tileOccupancy;
  int64_t size = ctx.buffers[buffer].size;
  int64_t alignBytes = ctx.buffers[buffer].alignBytes;
  SmallVector<Placement> out;

  // A requested bank set is a hard constraint, and the round-robin cursor stays
  // put. Only the two flush ends of each hole are worth trying, because an
  // interior start leaves strictly less behind.
  auto required = requiredBanks.find(buffer);
  if (required != requiredBanks.end()) {
    auto &banks = required->second;
    SmallVector<MemoryRun> ranges;
    for (int bank : banks)
      ranges.push_back(ctx.bankLimits[bank]);
    // Adjacent allowed banks can hold one spanning buffer; a disjoint pair
    // cannot use the intervening, forbidden bank.
    if (banks.size() == 2 && banks.back() == banks.front() + 1)
      ranges.push_back({ctx.bankLimits[banks.front()].start,
                        ctx.bankLimits[banks.front()].size +
                            ctx.bankLimits[banks.back()].size});
    for (MemoryRun limits : ranges) {
      for (MemoryRun gap : occupancy.gapsIn(limits.start, limits.end())) {
        int64_t low = llvm::alignTo(gap.start, alignBytes);
        if (low + size > gap.end()) {
          continue;
        }
        for (int64_t addr :
             {low, std::max<int64_t>(
                       llvm::alignDown(gap.end() - size, alignBytes), low)}) {
          MemoryOccupancy trial = occupancy;
          trial.markOccupied(addr, addr + size);
          int bank = getBankContaining(addr, ctx.numBanks, ctx.bankLimits);
          if (!llvm::is_contained(banks, bank))
            continue;
          int lastBank = size == 0
                             ? bank
                             : getBankContaining(addr + size - 1, ctx.numBanks,
                                                 ctx.bankLimits);
          out.push_back({addr, bank, /*banksTouched=*/lastBank - bank + 1,
                         -trial.largestGap(0, trial.size()), /*rrDistance=*/0,
                         /*slack=*/0});
        }
      }
    }
    // A zero-sized buffer needs no free byte, so every bank holds it.
    if (out.empty() && size == 0) {
      int bank = banks.front();
      out.push_back({ctx.bankLimits[bank].start, bank, 1, 0, 0, 0});
    }
  } else {
    int64_t bankedEnd = ctx.bankLimits.back().end();
    int64_t bankSize = bankedEnd / ctx.numBanks;
    // Computed once per buffer, not per candidate: every candidate splits one
    // run, so the largest run left elsewhere is one of these two. Named
    // separately, because C++17 cannot capture a structured binding in the
    // lambda below.
    std::pair<int64_t, int64_t> topGaps =
        occupancy.topTwoGaps(0, ctx.maxDataMemorySize);
    int64_t largestRun = topGaps.first, secondRun = topGaps.second;
    auto consider = [&](std::optional<int64_t> addr) {
      if (!addr) {
        return;
      }
      int bank = getBankContaining(*addr, ctx.numBanks, ctx.bankLimits);
      if (bank < 0) {
        return;
      }
      int64_t touched =
          size == 0 ? 1 : (*addr + size - 1) / bankSize - *addr / bankSize + 1;
      // This placement splits one run into the piece below and the piece above.
      // Every other run stays whole, so the biggest of those is whichever of
      // the top two this candidate does not consume.
      MemoryRun gap = occupancy.gapAt(*addr);
      int64_t elsewhere = gap.size == largestRun ? secondRun : largestRun;
      int64_t leftPiece = *addr - gap.start;
      int64_t rightPiece = gap.end() - (*addr + size);
      int64_t runLeft =
          std::min(std::max({elsewhere, leftPiece, rightPiece}), contiguityCap);
      out.push_back({*addr, bank, touched, ctx.hasCore ? -runLeft : 0,
                     (bank - startBankIndex + ctx.numBanks) % ctx.numBanks,
                     occupancy.slackAt(*addr, size)});
    };

    for (int i = 0; i < ctx.numBanks; i++) {
      consider(occupancy.findGap(ctx.bankLimits[i].start,
                                 ctx.bankLimits[i].end(), size, alignBytes));
    }
    // Allow straddling when nothing fits one bank. The banked region maps to a
    // bank, so the result always has one.
    consider(occupancy.findGap(0, bankedEnd, size, alignBytes));
    // Try bank-boundary starts only when a boundary also satisfies the buffer's
    // own alignment, because otherwise this searches on an unintended stride.
    if (bankSize % alignBytes == 0) {
      consider(occupancy.findGap(0, bankedEnd, size, bankSize));
    }
    // A zero-sized buffer needs no free byte, so a full tile still holds it.
    if (out.empty() && size == 0) {
      out.push_back({0, 0, 1, 0, 0, 0});
    }
  }

  llvm::stable_sort(out, [](const Placement &a, const Placement &b) {
    return a.rank() < b.rank();
  });

  // The probes above ask each hole only for its tightest fit, so a layout
  // needing some other position in one is unreachable -- and unreachable reads
  // as "no room". Both flush ends of every hole cover that. Appended after the
  // sort, never scored: ranking them would move addresses that already work.
  // Unpinned only; the mem_bank branch above already does this within its bank,
  // and widening it here would place the buffer outside that bank.
  if (size > 0 && requiredBanks.find(buffer) == requiredBanks.end()) {
    for (MemoryRun gap : occupancy.gapsIn(0, ctx.maxDataMemorySize)) {
      int64_t low = llvm::alignTo(gap.start, alignBytes);
      if (low + size > gap.end()) {
        continue;
      }
      for (int64_t addr :
           {low, std::max<int64_t>(
                     llvm::alignDown(gap.end() - size, alignBytes), low)}) {
        int bank = getBankContaining(addr, ctx.numBanks, ctx.bankLimits);
        if (bank >= 0) {
          // Ranks last by construction, so it is only ever reached by
          // backtracking past every scored candidate.
          out.push_back({addr, bank, kFallbackRank, 0, 0, 0});
        }
      }
    }
  }

  // The probes overlap: a gap that is the tightest in its bank is often also
  // the tightest overall, and the fallbacks repeat whichever of those was a
  // flush end. Equal addresses are the same placement, and trying one twice
  // only costs budget. Stable, so the ranked entry survives its duplicate.
  llvm::stable_sort(out, [](const Placement &a, const Placement &b) {
    return a.addr < b.addr;
  });
  out.erase(
      llvm::unique(out, [](const Placement &a,
                           const Placement &b) { return a.addr == b.addr; }),
      out.end());
  llvm::stable_sort(out, [](const Placement &a, const Placement &b) {
    return a.rank() < b.rank();
  });
  return out;
}

// Depth-first placement over `order`, trying each buffer's addresses in rank
// order and undoing a choice that leaves a later buffer nowhere to go.
//
// The first descent is exactly what a single-pass greedy produces, so this only
// does extra work where that pass used to fail outright. On failure `deepest`
// names the buffer that ran out of addresses furthest into the order, which is
// the one whose demand the tile could not meet.
namespace {
struct PlacementSearch {
  ArrayRef<size_t> order;
  const BankAwareContext &ctx;
  const RequiredBanks &requiredBanks;
  MemoryOccupancy &occupancy;
  SmallVectorImpl<size_t> &placed;
  TileLayoutStats &stats;
  TileLayout &layout;
  // How many placements the search may still try on this tile before giving
  // up and reporting the deepest failure it reached; the pass's
  // placement-budget option. Packing around fixed obstacles is NP-hard, so the
  // tree has no useful worst-case bound; this keeps a design the search cannot
  // solve to bounded compile time instead of exponential. A node count rather
  // than a time limit, so a build stays reproducible.
  int64_t budget;
  // Set when the node budget ran out with candidates still untried, so the
  // caller can say "gave up" rather than "no such layout exists".
  bool exhausted = false;

  // The furthest the search reached, kept so a failure can report the layout it
  // got closest with rather than the empty tile backtracking leaves behind.
  // First arrangement to reach a given depth wins, not the last: the first
  // comes down the highest-ranked path, so the map a user sees is the one the
  // ranking preferred rather than whichever branch happened to be explored
  // last.
  size_t deepestIndex = 0;
  std::optional<size_t> deepest = std::nullopt;
  SmallVector<std::pair<size_t, Placement>> deepestLayout = {};

  // Subproblems already searched in full without a layout, keyed by the next
  // buffer's index and the free runs left. The candidates a buffer gets depend
  // on nothing else -- the cursor only orders them -- so one of these can never
  // succeed. Same-sized buffers reach the same free runs in every order they
  // are placed in; without this the search retries each order.
  std::set<SmallVector<int64_t>> failedStates = {};

  bool run(size_t index, int startBankIndex, int64_t remaining) {
    if (index == order.size()) {
      return true;
    }
    SmallVector<int64_t> state = {static_cast<int64_t>(index)};
    for (MemoryRun gap : occupancy.gapsIn(0, occupancy.size())) {
      state.append({gap.start, gap.size});
    }
    if (failedStates.count(state)) {
      return false;
    }
    size_t buffer = order[index];
    int64_t size = ctx.buffers[buffer].size;
    int64_t rest = remaining - size; // Placement's criterion 2 cap.

    SmallVector<Placement> candidates = rankedPlacements(
        buffer, ctx, startBankIndex, requiredBanks, occupancy, rest);
    if (candidates.empty() && (!deepest || index > deepestIndex)) {
      recordDeepest(index, buffer);
    }
    for (const Placement &candidate : candidates) {
      if (budget <= 0) {
        // Out of budget before this buffer had an address: it is where the
        // search stopped, so it is the one to report.
        if (!deepest || index > deepestIndex) {
          recordDeepest(index, buffer);
        }
        exhausted = true;
        break;
      }
      --budget;
      placeBuffer(buffer, candidate.addr, candidate.bank, ctx, occupancy,
                  layout);
      placed.push_back(buffer);
      // A pinned buffer leaves the cursor alone; see rankedPlacements.
      int nextBank = requiredBanks.count(buffer)
                         ? startBankIndex
                         : (candidate.bank + 1) % ctx.numBanks;
      if (run(index + 1, nextBank, rest)) {
        return true;
      }
      placed.pop_back();
      ++stats.backtracks;
      occupancy.markFree(candidate.addr, candidate.addr + size);
      layout.at[buffer] = std::nullopt;
    }
    if (!exhausted) {
      failedStates.insert(std::move(state));
    }
    return false;
  }

  // Re-apply the best partial layout so the caller's memory map describes the
  // closest the search came, not the tile it unwound to.
  void restoreDeepest() {
    for (auto &[buffer, at] : deepestLayout) {
      placeBuffer(buffer, at.addr, at.bank, ctx, occupancy, layout);
      placed.push_back(buffer);
    }
  }

private:
  void recordDeepest(size_t index, size_t buffer) {
    deepestIndex = index;
    deepest = buffer;
    deepestLayout.clear();
    for (size_t done : placed) {
      const auto &at = layout.at[done];
      assert(at && "placed buffer must have an address and bank");
      if (!at)
        continue;
      auto [address, bank] = *at;
      deepestLayout.push_back({done, Placement{address, bank, 0, 0, 0, 0}});
    }
  }
};
} // namespace

// Exhaustive placement, for when the ranked search fails. The ranked search
// only tries positions flush against what is already placed, so a buffer that
// must sit against one placed after it is out of its reach in any order.
//
// Any layout can be slid down, one buffer at a time in address order, until
// each buffer sits at the lowest address its alignment and banks allow above
// the one before it. Walking the free runs upward and choosing which buffer
// comes next therefore reaches every layout that exists, and a tile this
// rejects within budget has none. What can follow an address depends only on
// the address and the buffers left, so a failed pair is never tried twice,
// which collapses the orders of one set of buffers into a single visit.
namespace {
struct CompactionSearch {
  ArrayRef<size_t> order;
  const BankAwareContext &ctx;
  SmallVector<MemoryRun> gaps;
  // Per buffer in `order`: size, alignment, the ranges it may lie in, and a
  // class shared with every buffer it could swap with.
  SmallVector<int64_t> sizes, aligns;
  SmallVector<SmallVector<MemoryRun, 2>> ranges;
  SmallVector<unsigned> classes;
  SmallVector<int64_t> addrs;
  // Free bytes and the largest run from each gap to the top of the tile.
  SmallVector<int64_t> freeAbove, runAbove;
  llvm::DenseSet<std::pair<int64_t, llvm::BitVector>> failed;
  int64_t budget;
  bool exhausted = false;

  CompactionSearch(ArrayRef<size_t> order, const BankAwareContext &ctx,
                   const RequiredBanks &requiredBanks,
                   const MemoryOccupancy &occupancy, int64_t budget)
      : order(order), ctx(ctx),
        gaps(occupancy.gapsIn(0, ctx.maxDataMemorySize)),
        addrs(order.size(), -1), budget(budget) {
    for (size_t i = 0; i < order.size(); ++i) {
      size_t buffer = order[i];
      sizes.push_back(ctx.buffers[buffer].size);
      aligns.push_back(ctx.buffers[buffer].alignBytes);
      SmallVector<MemoryRun, 2> allowed;
      auto required = requiredBanks.find(buffer);
      if (required == requiredBanks.end()) {
        allowed.push_back({0, ctx.maxDataMemorySize});
      } else {
        auto &banks = required->second;
        for (int bank : banks)
          allowed.push_back(ctx.bankLimits[bank]);
        if (banks.size() == 2 && banks.back() == banks.front() + 1)
          allowed = {{ctx.bankLimits[banks.front()].start,
                      ctx.bankLimits[banks.front()].size +
                          ctx.bankLimits[banks.back()].size}};
      }
      // An initialized buffer must dodge the guarded bytes at the foot of
      // the tile, the same restriction rankedPlacements enforces (see the
      // npu2 column-0 memtile comment in allocateTile). Clamping the ranges
      // here, before the equivalence classes below are computed, keeps a
      // guarded buffer out of any class an unguarded one of the same size
      // could also join.
      if (ctx.initGuardBytes > 0 && ctx.buffers[buffer].initialized) {
        SmallVector<MemoryRun, 2> clamped;
        for (MemoryRun range : allowed) {
          int64_t start = std::max(range.start, ctx.initGuardBytes);
          if (start < range.end())
            clamped.push_back({start, range.end() - start});
        }
        allowed = std::move(clamped);
      }
      ranges.push_back(allowed);
      unsigned cls = i;
      for (size_t j = 0; j < i; ++j) {
        if (sizes[j] == sizes[i] && aligns[j] == aligns[i] &&
            llvm::equal(ranges[j], ranges[i], [](MemoryRun a, MemoryRun b) {
              return a.start == b.start && a.size == b.size;
            })) {
          cls = classes[j];
          break;
        }
      }
      classes.push_back(cls);
    }
    freeAbove.assign(gaps.size() + 1, 0);
    runAbove.assign(gaps.size() + 1, 0);
    for (size_t g = gaps.size(); g-- > 0;) {
      freeAbove[g] = freeAbove[g + 1] + gaps[g].size;
      runAbove[g] = std::max(runAbove[g + 1], gaps[g].size);
    }
  }

  // Lowest address at or above `from` in gap `g` where buffer `i` fits.
  std::optional<int64_t> lowest(size_t i, size_t g, int64_t from) const {
    std::optional<int64_t> best;
    for (MemoryRun range : ranges[i]) {
      int64_t at = llvm::alignTo(std::max(from, range.start), aligns[i]);
      if (at + sizes[i] <= std::min(gaps[g].end(), range.end()) &&
          getBankContaining(at, ctx.numBanks, ctx.bankLimits) >= 0 &&
          (!best || at < *best))
        best = at;
    }
    return best;
  }

  bool run() {
    llvm::BitVector left(order.size());
    int64_t bytes = 0;
    for (size_t i = 0; i < order.size(); ++i) {
      if (sizes[i] == 0) {
        // Covers no byte, so it goes in the first gap its banks reach.
        for (size_t g = 0; g < gaps.size() && addrs[i] < 0; ++g)
          if (auto at = lowest(i, g, gaps[g].start))
            addrs[i] = *at;
        if (addrs[i] < 0)
          addrs[i] = ranges[i].front().start;
        continue;
      }
      left.set(i);
      bytes += sizes[i];
    }
    return gaps.empty() ? left.none()
                        : place(0, gaps.front().start, left, bytes);
  }

private:
  bool place(size_t g, int64_t from, llvm::BitVector &left, int64_t bytes) {
    if (left.none())
      return true;
    int64_t largest = 0;
    for (int i : left.set_bits())
      largest = std::max(largest, sizes[i]);
    int64_t here = gaps[g].end() - from;
    if (bytes > here + freeAbove[g + 1] ||
        largest > std::max(here, runAbove[g + 1]) ||
        failed.contains({from, left}))
      return false;
    SmallVector<unsigned> tried;
    for (int i : left.set_bits()) {
      if (llvm::is_contained(tried, classes[i]))
        continue;
      tried.push_back(classes[i]);
      std::optional<int64_t> at = lowest(i, g, from);
      if (!at)
        continue;
      if (budget-- <= 0) {
        exhausted = true;
        return false;
      }
      addrs[i] = *at;
      left.reset(i);
      bool done = place(g, *at + sizes[i], left, bytes - sizes[i]);
      left.set(i);
      if (done)
        return true;
      if (exhausted)
        return false;
    }
    if (g + 1 < gaps.size() && place(g + 1, gaps[g + 1].start, left, bytes))
      return true;
    if (!exhausted)
      failed.insert({from, left});
    return false;
  }
};
} // namespace

// Places every buffer in `buffersToAlloc`. Returns the buffer that could not be
// placed, or nothing when they all were; `placed` collects what was assigned.
static std::optional<size_t>
placeFreeBuffers(ArrayRef<size_t> buffersToAlloc, const BankAwareContext &ctx,
                 const RequiredBanks &requiredBanks, MemoryOccupancy &occupancy,
                 SmallVectorImpl<size_t> &placed, int64_t budget,
                 bool &exhausted, TileLayoutStats &stats, TileLayout &layout) {
  int64_t remaining = 0;
  for (size_t buffer : buffersToAlloc) {
    remaining += ctx.buffers[buffer].size;
  }
  PlacementSearch search{buffersToAlloc, ctx,   requiredBanks, occupancy,
                         placed,         stats, layout,        budget};
  int64_t before = stats.backtracks;
  bool solved = search.run(/*index=*/0, /*startBankIndex=*/0, remaining);
  if (stats.backtracks > before) {
    ++stats.tilesSearched;
  }
  if (solved) {
    return std::nullopt;
  }
  // The ranked search unwinds everything it tried, so the tile is back to its
  // pinned buffers alone.
  CompactionSearch complete(buffersToAlloc, ctx, requiredBanks, occupancy,
                            budget);
  if (complete.run()) {
    ++stats.tilesCompacted;
    for (auto [buffer, addr] : llvm::zip(buffersToAlloc, complete.addrs)) {
      placeBuffer(buffer, addr,
                  getBankContaining(addr, ctx.numBanks, ctx.bankLimits), ctx,
                  occupancy, layout);
      placed.push_back(buffer);
    }
    return std::nullopt;
  }
  search.restoreDeepest();
  exhausted = complete.exhausted;
  if (complete.exhausted) {
    ++stats.budgetExhausted;
  }
  return search.deepest;
}

// Places every buffer carrying an explicit `address`, address pins first, so a
// mem_bank-only buffer cannot carve up space an address pin needs. A
// mem_bank-only buffer enters `requiredBanks` and queues into
// `buffersToAlloc`. Failure is terminal.
static LogicalResult placePreAllocatedBuffers(
    SmallVectorImpl<size_t> &preAllocatedBuffers, const BankAwareContext &ctx,
    MemoryOccupancy &occupancy, RequiredBanks &requiredBanks,
    SmallVectorImpl<size_t> &buffersToAlloc, TileLayout &layout) {
  // Address buffers first (ascending), then mem_bank-only buffers. The sort is
  // stable, so the mem_bank-only tail, whose elements all compare equal, keeps
  // program order and places identically from one run to the next.
  llvm::stable_sort(preAllocatedBuffers, [&](size_t a, size_t b) -> bool {
    auto a_addr = ctx.buffers[a].address;
    auto b_addr = ctx.buffers[b].address;
    if (a_addr.has_value() && b_addr.has_value()) {
      return a_addr.value() < b_addr.value();
    }
    return a_addr.has_value() && !b_addr.has_value();
  });

  for (size_t buffer : preAllocatedBuffers) {
    auto has_addr =
        checkAndAddBufferWithAddress(buffer, ctx, occupancy, layout);
    if (failed(has_addr)) {
      return failure();
    }
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (*has_addr) {
      continue;
    }
    // Only a mem_bank: this pass still chooses the address, so queue the
    // buffer with the rest.
    if (failed(recordRequiredBank(buffer, ctx, requiredBanks, layout))) {
      return failure();
    }
    buffersToAlloc.push_back(buffer);
  }
  return success();
}

// Order buffers for placement: fewest allowed banks first, then largest first,
// so a small buffer does not split the run a large one needs.
static SmallVector<size_t> placementOrder(ArrayRef<size_t> buffersToAlloc,
                                          const RequiredBanks &requiredBanks,
                                          const BankAwareContext &ctx) {
  SmallVector<size_t> order(buffersToAlloc.begin(), buffersToAlloc.end());
  llvm::stable_sort(order, [&](size_t a, size_t b) {
    if (requiredBanks.count(a) != requiredBanks.count(b)) {
      return requiredBanks.count(a) > requiredBanks.count(b);
    }
    if (requiredBanks.count(a) &&
        requiredBanks.lookup(a).size() != requiredBanks.lookup(b).size())
      return requiredBanks.lookup(a).size() < requiredBanks.lookup(b).size();
    return ctx.buffers[a].size > ctx.buffers[b].size;
  });
  return order;
}

TileLayoutBuffer xilinx::AIE::tileLayoutBuffer(BufferOp buffer,
                                               const TileMemoryLimits &limits) {
  return {buffer.getAllocationSize(),
          bufferAlignBytes(buffer, limits.tileAlignBitWidth,
                           limits.maxVecAlignBits),
          buffer.getSectionAlignment(),
          buffer.getAligned(),
          buffer.getAddress(),
          buffer.getMemBank(),
          banksFromMemorySpace(buffer.getType()),
          buffer.getInitialValue().has_value()};
}

// The buffer materializeCoreSections makes for `held`, as tileLayoutBuffer
// above reads it back.
TileLayoutBuffer
xilinx::AIE::tileLayoutBuffer(const CoreSectionReservation &held,
                              const TileMemoryLimits &limits) {
  using Kind = CoreSectionReservation::Kind;
  if (held.kind == Kind::Prebaked)
    return {held.bytes,   1,  1,    /*aligned=*/false, held.at,
            std::nullopt, {}, false};
  int64_t vectorAlign =
      requiredBufferAlignBits(limits.tileAlignBitWidth, limits.maxVecAlignBits,
                              held.bytes) /
      8;
  std::optional<int32_t> bank;
  if (held.kind == Kind::Bank)
    bank = held.at;
  return {held.bytes,
          std::max(vectorAlign, held.alignBytes),
          held.alignBytes,
          /*aligned=*/true,
          std::nullopt,
          bank,
          {},
          false};
}

TileLayout xilinx::AIE::layoutTile(const TileLayoutProblem &problem,
                                   TileLayoutStats &stats) {
  TileLayout layout;
  layout.at.resize(problem.buffers.size());

  std::vector<MemoryRun> bankLimits;
  int64_t bankSize = problem.maxDataMemorySize / problem.numBanks;
  fillBankLimits(problem.numBanks, bankSize, bankLimits);

  // A stack_bank-only core has no address yet. Its bank is a hard constraint,
  // so reserve inside that bank before anything unconstrained is placed, and
  // record the address the way a mem_bank-only buffer gets one.
  MemoryRun stackRun = problem.stackRun;
  MemoryOccupancy occupancy(problem.maxDataMemorySize);
  if (problem.hasCore) {
    if (std::optional<int> bank = problem.stackBank) {
      // Address-pinned buffers cannot move, whereas a bank-only stack can.
      // Account for those pins before choosing the stack's address. The normal
      // buffer placement below still validates each pin and records occupancy.
      MemoryOccupancy stackOccupancy(problem.maxDataMemorySize);
      for (const TileLayoutBuffer &buffer : problem.buffers) {
        if (!buffer.address)
          continue;
        int64_t start = *buffer.address;
        int64_t end = start + buffer.size;
        if (start >= 0 && end <= problem.maxDataMemorySize)
          stackOccupancy.markOccupied(start, end);
      }
      std::optional<int64_t> at = stackOccupancy.findLeastFragmentingGap(
          bankLimits[*bank].start, bankLimits[*bank].end(), stackRun.size,
          problem.stackAlignBytes);
      if (!at) {
        layout.status = TileLayout::Status::StackNoRoom;
        layout.message =
            ("requires a " + llvm::Twine(stackRun.size) +
             "-byte stack in bank " + llvm::Twine(*bank) +
             ", but no contiguous aligned space remains after address-pinned "
             "buffers")
                .str();
        return layout;
      }
      stackRun.start = *at;
      layout.stackStart = at;
    }
    occupancy.markOccupied(stackRun.start,
                           std::min(stackRun.end(), problem.maxDataMemorySize));
  }
  BankAwareContext ctx{problem.numBanks,          problem.tileAlignBitWidth,
                       problem.maxVecAlignBits,   bankLimits,
                       problem.maxDataMemorySize, stackRun,
                       problem.hasCore,           problem.buffers};

  RequiredBanks requiredBanks;
  SmallVector<size_t> preAllocatedBuffers;
  SmallVector<size_t> buffersToAlloc;
  for (auto [i, buffer] : llvm::enumerate(problem.buffers)) {
    if (!buffer.address && !buffer.memBank && buffer.spaceBanks.empty()) {
      buffersToAlloc.push_back(i);
    } else {
      preAllocatedBuffers.push_back(i);
    }
  }

  if (failed(placePreAllocatedBuffers(preAllocatedBuffers, ctx, occupancy,
                                      requiredBanks, buffersToAlloc, layout))) {
    return layout;
  }
  layout.preAllocated = preAllocatedBuffers;
  ctx.initGuardBytes = problem.initGuardBytes;

  // Bank-pinned buffers lead the order, then the core's data region, then the
  // unconstrained ones -- most-constrained first, since a pin has one candidate
  // bank. They are searched together rather than in separate passes: where a
  // pin's position inside its bank decides whether a later buffer fits, only a
  // single search can revisit it.
  SmallVector<size_t> order =
      placementOrder(buffersToAlloc, requiredBanks, ctx);
  std::optional<size_t> failedBuffer =
      placeFreeBuffers(order, ctx, requiredBanks, occupancy, layout.allocated,
                       problem.budget, layout.exhausted, stats, layout);
  if (!failedBuffer) {
    assert(layout.allocated.size() == buffersToAlloc.size());
    return layout;
  }
  layout.status = TileLayout::Status::NoRoom;
  layout.failed = *failedBuffer;
  if (auto required = requiredBanks.find(*failedBuffer);
      required != requiredBanks.end()) {
    layout.failedBanks = required->second;
    int bank = required->second.front();
    layout.freeInFailedBank =
        occupancy.freeBytes(bankLimits[bank].start, bankLimits[bank].end());
  }
  for (size_t buffer : buffersToAlloc)
    if (!layout.at[buffer])
      layout.unplacedBytes += problem.buffers[buffer].size;
  return layout;
}

// Rolls back what the allocator wrote, and leaves a mem_bank the user requested
// in place.
static void deAllocationBuffers(SmallVectorImpl<BufferOp> &buffers,
                                ArrayRef<TileLayoutBuffer> requested) {
  for (auto [buffer, request] : llvm::zip_equal(buffers, requested)) {
    buffer->removeAttr("address");
    if (request.memBank) {
      buffer.setMemBank(request.memBank);
    } else {
      buffer->removeAttr("mem_bank");
    }
  }
}

static LogicalResult allocateTile(TileOp tile, int64_t budget,
                                  TileLayoutStats &stats) {
  auto device = tile->getParentOfType<AIE::DeviceOp>();
  if (!device) {
    return failure();
  }

  const auto &targetModel = getTargetModel(tile);
  TileMemoryLimits limits = tileMemoryLimits(targetModel, tile.getTileType());
  TileLayoutProblem problem;
  problem.maxDataMemorySize = limits.maxDataMemorySize;
  problem.tileAlignBitWidth = limits.tileAlignBitWidth;
  problem.maxVecAlignBits = limits.maxVecAlignBits;
  problem.numBanks = targetModel.getNumBanks(tile.getCol(), tile.getRow());
  problem.budget = budget;
  CoreOp coreOp = tile.getCoreOp();
  problem.hasCore = static_cast<bool>(coreOp);
  if (coreOp) {
    problem.stackRun = coreOp.getStackRun();
    std::optional<uint32_t> stackBank = coreOp.getStackBank();
    if (!coreOp.getStackAddress() && stackBank) {
      problem.stackBank = static_cast<int>(*stackBank);
      problem.stackAlignBytes = targetModel.getCoreStackAlignment();
    }
  }
  // On npu2, the first word of the column-0 memtile reads back as 0x00CD0CD0
  // on every dispatch after the first, though nothing in the design writes it.
  // A buffer that DMA refills each dispatch is unaffected, but an initialized
  // one is loaded only once and keeps the bad word, so keep initialized buffers
  // off that word. Other buffers may still take it, and a buffer the user
  // pinned there keeps its address.
  //
  // TODO: Investigate. This is a low-confidence workaround for a cause nobody
  // has found: the word may be reserved by the firmware or driver, or clobbered
  // by something else. The value predates initialized buffers dropping their
  // locks (#3608 saw it first), so that change is not the cause.
  if (tile.isMemTile() && tile.getCol() == 0 &&
      targetModel.hasProperty(AIETargetModel::IsNPU) &&
      targetModel.getTargetArch() == AIEArch::AIE2p) {
    problem.initGuardBytes = problem.tileAlignBitWidth / 8;
  }

  SmallVector<BufferOp> buffers;
  device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
    if (buffer.getTileOp() != tile)
      return;
    buffers.push_back(buffer);
    problem.buffers.push_back(tileLayoutBuffer(buffer, limits));
  });

  TileLayout layout = layoutTile(problem, stats);
  if (layout.status == TileLayout::Status::StackNoRoom) {
    coreOp.emitOpError(layout.message);
    return failure();
  }
  if (layout.stackStart) {
    coreOp.setStackAddress(*layout.stackStart);
  }
  if (layout.status == TileLayout::Status::BadPin) {
    buffers[layout.failed].emitOpError(layout.message);
    return failure();
  }
  for (auto [buffer, at] : llvm::zip_equal(buffers, layout.at)) {
    if (at) {
      buffer.setAddress(at->first);
      buffer.setMemBank(at->second);
    }
  }
  auto opsOf = [&](ArrayRef<size_t> indices) {
    return llvm::map_to_vector(indices, [&](size_t i) { return buffers[i]; });
  };
  SmallVector<BufferOp> allocatedBuffers = opsOf(layout.allocated);
  SmallVector<TileLayoutBuffer> allocatedRequests = llvm::map_to_vector(
      layout.allocated, [&](size_t i) { return problem.buffers[i]; });
  MemoryRun stackRun = problem.stackRun;
  if (layout.stackStart)
    stackRun.start = *layout.stackStart;

  if (layout.status == TileLayout::Status::NoRoom) {
    BufferOp failed = buffers[layout.failed];
    std::vector<MemoryRun> bankLimits;
    fillBankLimits(problem.numBanks,
                   problem.maxDataMemorySize / problem.numBanks, bankLimits);
    // A buffer pinned to a bank that cannot hold it is a user constraint, not
    // an out-of-room tile: give it its own error and no memory map. Unless the
    // search merely ran out of budget, in which case the bank may have room.
    if (!layout.exhausted && !layout.failedBanks.empty()) {
      auto banks = layout.failedBanks;
      if (banks.size() > 1) {
        failed.emitOpError("")
            << bufferLabel(failed) << " requires " << failed.getAllocationSize()
            << " bytes, but no contiguous aligned space remains within allowed "
               "banks "
            << banks.front() << " and " << banks.back();
        deAllocationBuffers(allocatedBuffers, allocatedRequests);
        return failure();
      }
      int bank = banks.front();
      int64_t need = failed.getAllocationSize();
      int64_t bankCapacity = bankLimits[bank].size;
      if (need > bankCapacity) {
        failed->emitOpError("") << bufferLabel(failed) << " requires " << need
                                << " bytes, which cannot fit in bank " << bank
                                << " (" << bankCapacity << " bytes total)";
      } else {
        failed->emitOpError("")
            << bufferLabel(failed) << " requires " << need << " bytes in bank "
            << bank << ", but only " << layout.freeInFailedBank << " of "
            << bankCapacity << " bytes are free there";
      }
      deAllocationBuffers(allocatedBuffers, allocatedRequests);
      return failure();
    }
    InFlightDiagnostic diag = failed.emitOpError(kNoTileRoom)
                              << bufferLabel(failed) << " needs "
                              << failed.getAllocationSize()
                              << " bytes and this tile has no room left for it";
    if (CoreOp core = tile.getCoreOp())
      if (int64_t data = core.getDataSize().value_or(
              static_cast<int64_t>(core.getMeasuredDataSize().value_or(0)))) {
        auto device = core->getParentOfType<DeviceOp>();
        auto cores = device.getOps<CoreOp>();
        diag.attachNote() << kCoreSectionsNote << data << " bytes (core "
                          << std::distance(cores.begin(),
                                           llvm::find(cores, core))
                          << " of "
                          << FlatSymbolRefAttr::get(device.getSymNameAttr())
                          << ")";
      }
    if (layout.exhausted) {
      diag.attachNote() << "the search hit its " << budget
                        << "-placement budget with arrangements still untried, "
                           "so a layout may exist that it did not reach (raise "
                           "it with placement-budget)";
    }
    // Print before rollback, while the addresses are still set.
    printMemMap(tile, allocatedBuffers, opsOf(layout.preAllocated),
                problem.numBanks, bankLimits, stackRun);
    deAllocationBuffers(allocatedBuffers, allocatedRequests);
    return failure();
  }

  sortBuffersByAddress(buffers);
  // Every placement came from free space in the tile, so overflow cannot happen
  // here. The stack and overlap checks remain as a backstop.
  if (!checkAndPrintOverlapStackframe(stackRun, buffers) ||
      !checkAndPrintBufferOverlap(buffers, problem.tileAlignBitWidth,
                                  problem.maxVecAlignBits)) {
    return failure();
  }
  return success();
}

static LogicalResult checkBufferScope(BufferOp buffer, DeviceOp device) {
  // collectBuffers walks the device's own operations, so the linker script and
  // the BCF describe a buffer only at that depth. A buffer nested anywhere else
  // reaches the core as an undefined symbol. MemTileDMA is the exception: a
  // memtile links no core, so nothing refers to the buffer by name.
  Operation *parent = buffer->getParentOp();
  if (!isa<DeviceOp>(parent) && !isa<MemTileDMAOp>(parent)) {
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

  AIEAssignBufferAddressesPass(const AIEAssignBufferAddressesOptions &options)
      : AIEAssignBufferAddressesBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    int64_t budget = clPlacementBudget;
    if (budget <= 0) {
      device.emitError("placement-budget must be positive, got ") << budget;
      return signalPassFailure();
    }
    applySignatureBankConstraints(device);
    if (failed(materializeCoreSections(device))) {
      return signalPassFailure();
    }

    // One allocator, one answer. There used to be two schemes and a fallback
    // between them, which meant a tile that ran out of room reported twice and
    // explained itself once, and a `mem_bank` pin could be dropped on the way
    // from one scheme to the other. Placement now either finds a layout or says
    // why it could not.
    TileLayoutStats stats;
    bool ok = llvm::all_of(device.getOps<TileOp>(), [&](TileOp tile) {
      return succeeded(allocateTile(tile, budget, stats));
    });
    // Recorded on failure too, where budget-exhausted is the one that matters.
    numTilesSearched += stats.tilesSearched;
    numBacktracks += stats.backtracks;
    numBudgetExhausted += stats.budgetExhausted;
    numTilesCompacted += stats.tilesCompacted;
    if (!ok)
      signalPassFailure();
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
    // A buffer's name becomes a symbol in its core's object, so every buffer
    // needs a name before the cores are lowered. Naming runs first, because a
    // diagnostic below names the buffer it rejects.
    OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
    unsigned counter = 0;
    NamedOpTable names(device);
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (!buffer.hasName()) {
        buffer->setAttr(
            SymbolTable::getSymbolAttrName(),
            builder.getStringAttr(names.uniqueName("_anonymous", counter)));
      }
    });
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (failed(checkBufferScope(buffer, device))) {
        return signalPassFailure();
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
