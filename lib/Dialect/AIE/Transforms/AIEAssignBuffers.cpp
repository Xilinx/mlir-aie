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
#include "aie/Dialect/AIE/Transforms/AIETileMemoryFit.h"

#include "mlir/IR/Attributes.h"

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
  return buffer.name().getValue();
}

// Give each core's `data_size` an aie.buffer, so placement treats the core's
// own sections as one more extent to fit. Idempotent: this pass runs standalone
// and inside larger pipelines.
static void materializeCoreDataBuffers(DeviceOp device) {
  OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
  for (auto core : llvm::to_vector(device.getOps<CoreOp>())) {
    // A declared `data_size` still wins: it is a reservation the design asked
    // for, and may deliberately exceed what this build happens to measure.
    // Without one, the probe's measurement stands in, so the region is an
    // extent placement fits rather than whatever space is left over.
    int64_t size = core.getDataSize().value_or(
        static_cast<int64_t>(core.getMeasuredDataSize().value_or(0)));
    if (size <= 0) {
      continue;
    }
    auto tile = cast<TileOp>(core.getTile().getDefiningOp());
    bool present = false;
    device.walk([&](BufferOp buffer) {
      if (buffer.getCoreData() && buffer.getTileOp() == tile) {
        present = true;
      }
    });
    if (present) {
      continue;
    }
    builder.setInsertionPoint(core);
    auto type = MemRefType::get({size}, builder.getI8Type());
    auto buffer =
        BufferOp::create(builder, core.getLoc(), type, tile.getResult(),
                         /*sym_name=*/nullptr, /*address=*/nullptr,
                         /*initial_value=*/nullptr,
                         /*mem_bank=*/nullptr,
                         /*core_data=*/builder.getUnitAttr(),
                         /*bank_reserved=*/nullptr,
                         /*aligned=*/nullptr);
    buffer->setAttr(SymbolTable::getSymbolAttrName(),
                    builder.getStringAttr("core_data_" +
                                          std::to_string(tile.getCol()) + "_" +
                                          std::to_string(tile.getRow())));
  }
}

// Give each bank a core pins static data to an aie.buffer of its own, so the
// room the linker will need there is held before unconstrained buffers take it.
// Sizes come from `measured_bank_sizes`, which aiecc fills from a probe link.
// Idempotent, like materializeCoreDataBuffers.
static void materializeBankReservations(DeviceOp device) {
  OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
  for (auto core : llvm::to_vector(device.getOps<CoreOp>())) {
    auto sizes = core.getMeasuredBankSizes();
    if (!sizes) {
      continue;
    }
    auto tile = cast<TileOp>(core.getTile().getDefiningOp());
    for (auto [bank, size] : llvm::enumerate(*sizes)) {
      if (size <= 0) {
        continue;
      }
      std::string name = "bank_reserved_" + std::to_string(tile.getCol()) +
                         "_" + std::to_string(tile.getRow()) + "_" +
                         std::to_string(bank);
      bool present = false;
      device.walk([&](BufferOp buffer) {
        if (buffer.name() == name) {
          present = true;
        }
      });
      if (present) {
        continue;
      }
      builder.setInsertionPoint(core);
      auto type = MemRefType::get({size}, builder.getI8Type());
      auto buffer =
          BufferOp::create(builder, core.getLoc(), type, tile.getResult(),
                           /*sym_name=*/nullptr, /*address=*/nullptr,
                           /*initial_value=*/nullptr,
                           /*mem_bank=*/builder.getI32IntegerAttr(bank),
                           /*core_data=*/nullptr,
                           /*bank_reserved=*/builder.getUnitAttr(),
                           /*aligned=*/nullptr);
      buffer->setAttr(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
    }
  }
}

// Give a prebaked `elf_file` core's occupied extents an aie.buffer each, so
// placement keeps the tile's buffers clear of memory that ELF already holds.
// aiecc fills `measured_data_ranges` by reading the ELF; without it the
// allocator has no way to know a prebaked core owns anything at all.
//
// The same shape as materializeBankReservations, pinned by address rather than
// by bank: the addresses in a prebaked image are already final. Idempotent.
static void materializePrebakedRanges(DeviceOp device) {
  OpBuilder builder = OpBuilder::atBlockTerminator(device.getBody());
  for (auto core : llvm::to_vector(device.getOps<CoreOp>())) {
    auto ranges = core.getMeasuredDataRanges();
    if (!ranges) {
      continue;
    }
    auto tile = cast<TileOp>(core.getTile().getDefiningOp());
    for (size_t i = 0; i + 1 < ranges->size(); i += 2) {
      int32_t address = (*ranges)[i], size = (*ranges)[i + 1];
      if (size <= 0) {
        continue;
      }
      std::string name = "prebaked_" + std::to_string(tile.getCol()) + "_" +
                         std::to_string(tile.getRow()) + "_" +
                         std::to_string(address);
      bool present = false;
      device.walk([&](BufferOp buffer) {
        if (buffer.name() == name) {
          present = true;
        }
      });
      if (present) {
        continue;
      }
      builder.setInsertionPoint(core);
      auto type = MemRefType::get({size}, builder.getI8Type());
      auto buffer =
          BufferOp::create(builder, core.getLoc(), type, tile.getResult(),
                           /*sym_name=*/nullptr,
                           /*address=*/builder.getI32IntegerAttr(address),
                           /*initial_value=*/nullptr,
                           /*mem_bank=*/nullptr,
                           /*core_data=*/nullptr,
                           /*bank_reserved=*/builder.getUnitAttr(),
                           /*aligned=*/builder.getBoolAttr(false));
      buffer->setAttr(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
    }
  }
}

// Check a buffer's bank pin against its address space. A singleton space also
// supplies the pin; pairs stay flexible until placement.
//
// The buffer carries the space rather than the kernel's declaration doing so
// alone, because `func.call` already requires operand types to match the
// callee's signature exactly. Agreement between the buffer and the kernel it is
// passed to is therefore checked by the verifier, not here; all that is left is
// to place the buffer where its type says it lives.
static LogicalResult applySignatureBankConstraints(DeviceOp device) {
  auto result = success();
  device.walk([&](BufferOp buffer) {
    auto banks = banksFromMemorySpace(buffer.getType());
    if (banks.empty()) {
      return;
    }
    if (auto existing = buffer.getMemBank();
        existing && !llvm::is_contained(banks, *existing)) {
      auto diag = buffer.emitOpError("has address space for bank ");
      diag << banks.front();
      if (banks.size() > 1)
        diag << " or " << banks.back();
      diag << " but is pinned to bank " << *existing
           << ". These constraints disagree";
      result = failure();
      return;
    }
    if (banks.size() == 1)
      buffer.setMemBankAttr(IntegerAttr::get(
          IntegerType::get(device.getContext(), 32), banks.front()));
  });
  return result;
}

static bool isBufferPreAllocated(BufferOp buffer) {
  auto addr = buffer.getAddress();
  auto memBank = buffer.getMemBank();
  return (addr != std::nullopt || memBank != std::nullopt);
}

static int64_t getMeasuredDataAlignBytes(BufferOp buffer) {
  if (!buffer.getCoreData() && !buffer.getBankReserved())
    return 1;
  CoreOp core = buffer.getTileOp().getCoreOp();
  if (!core)
    return 1;
  if (buffer.getCoreData())
    return core.getMeasuredDataAlignment().value_or(1);
  auto bank = buffer.getMemBank();
  auto alignments = core.getMeasuredBankAlignments();
  if (bank && alignments && *bank >= 0 &&
      static_cast<size_t>(*bank) < alignments->size())
    return std::max<int64_t>((*alignments)[*bank], 1);
  return 1;
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

// Alignment a buffer must be placed at, in bytes.
static int64_t getBufferAlignBytes(BufferOp buffer, uint32_t tileAlignBitWidth,
                                   uint32_t maxVecAlignBits) {
  return fitAlignBytes(buffer.getAligned(), tileAlignBitWidth, maxVecAlignBits,
                       buffer.getAllocationSize(),
                       getMeasuredDataAlignBytes(buffer));
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
  const TileMemoryLayout &layout;
  uint32_t tileAlignBitWidth;
  uint32_t maxVecAlignBits;
  MemoryRun stackRun;
};

// Places a buffer carrying an explicit `address`, and checks that the space is
// free and that any `mem_bank` agrees. Returns false when the buffer has no
// address, which leaves it to the mem_bank or free-placement path. Failure is
// terminal: this emits an error and the caller must not retry another scheme
static FailureOr<bool>
checkAndAddBufferWithAddress(BufferOp buffer, const BankAwareContext &ctx,
                             MemoryOccupancy &occupancy) {
  auto addrOpt = buffer.getAddress();
  if (!addrOpt) {
    return false;
  }

  int64_t addr = *addrOpt;
  int64_t measuredAlign = getMeasuredDataAlignBytes(buffer);
  if (addr % measuredAlign != 0)
    return buffer.emitOpError("data reservation address must be aligned to ")
           << measuredAlign
           << " bytes to satisfy its measured section alignment";

  // A pinned address must satisfy the bus width. The stricter vector alignment
  // applies to the addresses this pass chooses.
  if (buffer.getAligned() && addr % (ctx.tileAlignBitWidth / 8) != 0) {
    return buffer->emitOpError("address attribute value must be aligned to ")
           << ctx.tileAlignBitWidth
           << " bits when the aligned attribute is set";
  }

  int bank = ctx.layout.bankContaining(addr);
  if (bank < 0) {
    // A zero-sized buffer pinned one past the last bank is legal. Assign it the
    // last bank, where nothing can conflict with it.
    if (buffer.getAllocationSize() == 0 &&
        addr == ctx.layout.bankLimits.back().end()) {
      bank = ctx.layout.numBanks - 1;
    } else {
      return buffer->emitOpError(
          "address attribute does not fall within any bank range");
    }
  }

  int64_t endAddr = addr + buffer.getAllocationSize();
  if (endAddr > occupancy.size()) {
    return buffer->emitOpError("address attribute would place the buffer past "
                               "the end of the tile's memory");
  }

  auto banks = banksFromMemorySpace(buffer.getType());
  int lastBank = buffer.getAllocationSize() == 0
                     ? bank
                     : ctx.layout.bankContaining(endAddr - 1);
  for (int touched = bank; !banks.empty() && touched <= lastBank; ++touched) {
    if (!llvm::is_contained(banks, touched))
      return buffer.emitOpError("address attribute places the buffer in bank ")
             << touched << ", which is excluded by its memory space";
  }
  if (auto requested = buffer.getMemBank(); requested && *requested != bank)
    return buffer.emitOpError("address attribute lies in bank ")
           << bank << ", but mem_bank requests bank " << *requested;

  // Only a real collision is invalid. The hardware lets a buffer straddle a
  // bank boundary, so a pinned buffer may extend past `bank`.
  if (!occupancy.isRangeFree(addr, endAddr)) {
    return buffer->emitOpError("would override allocated address");
  }

  placeBuffer(buffer, addr, bank, occupancy);
  return true;
}

// Banks a buffer may live in because the user requested them. Held separately
// from the `mem_bank` attribute, which the allocator writes for the buffers it
// places and clears when it rolls back.
using RequiredBanks = DenseMap<Operation *, SmallVector<int, 2>>;

// Record the allowed banks, intersected with any explicit mem_bank request.
// applySignatureBankConstraints has already checked that the intersection is
// nonempty.
static LogicalResult recordRequiredBank(BufferOp buffer, int numBanks,
                                        RequiredBanks &requiredBanks) {
  auto memBankOpt = buffer.getMemBank();
  if (memBankOpt && (*memBankOpt < 0 || *memBankOpt >= numBanks)) {
    return buffer->emitOpError("mem_bank attribute value is out of range");
  }
  auto banks = banksFromMemorySpace(buffer.getType());
  if (memBankOpt)
    banks = {*memBankOpt};
  if (!banks.empty()) {
    if (banks.back() >= numBanks)
      return buffer.emitOpError(
          "memory space requests a bank that does not exist");
    requiredBanks[buffer] = std::move(banks);
  }
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
  if (ctx.stackRun.size == 0) {
    note << "\t(no stack allocated)\n";
  }
  if (auto measured = getMeasuredStackSize(tile)) {
    note << "\t(aiecc measured this core's stack requirement as " << *measured
         << " bytes)\n";
  }
  for (int i = 0; i < ctx.layout.numBanks; i++) {
    note << "\t" << "bank : " << i << "\t" << "0x"
         << llvm::utohexstr(ctx.layout.bankLimits[i].start) << "-0x"
         << llvm::utohexstr(ctx.layout.bankLimits[i].end() - 1) << "\n";
    // Under whichever bank holds it, rather than always under bank 0.
    if (ctx.stackRun.size > 0 &&
        ctx.layout.bankLimits[i].contains(ctx.stackRun.start)) {
      printbuffer("(stack)", ctx.stackRun.start, ctx.stackRun.size);
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
        if (*addrOpt + size > ctx.layout.bankLimits[i].end()) {
          suffix = (" (straddles into bank " + llvm::Twine(i + 1) + ")").str();
        }
        printbuffer(mapLabel(buffer), *addrOpt, size, suffix);
      }
    };
    printPlaced(preAllocatedBuffers);
    printPlaced(allocatedBuffers);
  }
}

// Rolls back what the allocator wrote, and leaves a mem_bank the user requested
// in place.
static void deAllocationBuffers(SmallVectorImpl<BufferOp> &buffers,
                                const RequiredBanks &requiredBanks) {
  for (auto buffer : buffers) {
    buffer->removeAttr("address");
    auto required = requiredBanks.find(buffer);
    if (required == requiredBanks.end() || required->second.size() != 1) {
      buffer->removeAttr("mem_bank");
    } else {
      buffer.setMemBank(required->second.front());
    }
  }
}

// Places every buffer carrying an explicit `address`, address pins first, so a
// mem_bank-only buffer cannot carve up space an address pin needs. A
// mem_bank-only buffer enters `requiredBanks` and queues into
// `buffersToAlloc`. Failure is terminal.
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
    if (a_addr.has_value() && b_addr.has_value()) {
      return a_addr.value() < b_addr.value();
    }
    return a_addr.has_value() && !b_addr.has_value();
  });

  for (auto buffer : preAllocatedBuffers) {
    auto has_addr = checkAndAddBufferWithAddress(buffer, ctx, occupancy);
    if (failed(has_addr)) {
      return failure();
    }
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    if (*has_addr) {
      continue;
    }
    // Only a mem_bank: this pass still chooses the address, so queue the
    // buffer with the rest.
    if (failed(
            recordRequiredBank(buffer, ctx.layout.numBanks, requiredBanks))) {
      return failure();
    }
    buffersToAlloc.push_back(buffer);
  }
  return success();
}

static LogicalResult allocateTile(TileOp tile, int64_t budget,
                                  FitStats &stats) {
  auto device = tile->getParentOfType<AIE::DeviceOp>();
  if (!device) {
    return failure();
  }

  const auto &targetModel = getTargetModel(tile);
  auto [maxDataMemorySize, tileAlignBitWidth, maxVecAlignBits] =
      tileMemoryLimits(targetModel, tile.getTileType());
  TileMemoryLayout layout(targetModel, {tile.getCol(), tile.getRow()},
                          static_cast<bool>(tile.getCoreOp()));
  ArrayRef<MemoryRun> bankLimits = layout.bankLimits;

  // A stack_bank-only core has no address yet. Its bank is a hard constraint,
  // so reserve inside that bank before anything unconstrained is placed, and
  // record the address the way a mem_bank-only buffer gets one.
  MemoryRun stackRun;
  MemoryOccupancy occupancy(maxDataMemorySize);
  CoreOp coreOp = tile.getCoreOp();
  if (coreOp) {
    stackRun = coreOp.getStackRun();
    auto stackBank = coreOp.getStackBank();
    if (!coreOp.getStackAddress() && stackBank) {
      int bank = *stackBank;
      int64_t align = targetModel.getCoreStackAlignment();
      // Address-pinned buffers cannot move, whereas a bank-only stack can.
      // Account for those pins before choosing the stack's address. The normal
      // buffer placement below still validates each pin and records occupancy.
      MemoryOccupancy stackOccupancy(maxDataMemorySize);
      device.walk([&, maxDataMemorySize = maxDataMemorySize](BufferOp buffer) {
        if (buffer.getTileOp() != tile || !buffer.getAddress())
          return;
        int64_t start = *buffer.getAddress();
        int64_t end = start + buffer.getAllocationSize();
        if (start >= 0 && end <= maxDataMemorySize)
          stackOccupancy.markOccupied(start, end);
      });
      std::optional<int64_t> at = stackOccupancy.findLeastFragmentingGap(
          bankLimits[bank].start, bankLimits[bank].end(), stackRun.size, align);
      if (!at) {
        coreOp.emitOpError("requires a ")
            << stackRun.size << "-byte stack in bank " << bank
            << ", but no contiguous aligned space remains after address-pinned "
               "buffers";
        return failure();
      }
      stackRun.start = *at;
      coreOp.setStackAddress(*at);
    }
    occupancy.markOccupied(stackRun.start,
                           std::min(stackRun.end(), maxDataMemorySize));
  }
  BankAwareContext ctx{layout, tileAlignBitWidth, maxVecAlignBits, stackRun};

  RequiredBanks requiredBanks;
  SmallVector<BufferOp> preAllocatedBuffers;
  SmallVector<BufferOp> buffersToAlloc;
  SmallVector<BufferOp> allBuffers_on_tile;
  device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
    if (buffer.getTileOp() == tile) {
      if (!isBufferPreAllocated(buffer) &&
          banksFromMemorySpace(buffer.getType()).empty()) {
        buffersToAlloc.push_back(buffer);
      } else {
        preAllocatedBuffers.push_back(buffer);
      }
      allBuffers_on_tile.push_back(buffer);
    }
  });

  if (failed(placePreAllocatedBuffers(preAllocatedBuffers, ctx, occupancy,
                                      requiredBanks, buffersToAlloc))) {
    return failure();
  }

  // Buffers this pass placed (not the pre-allocated ones), for rollback and
  // diagnostics on failure.
  SmallVector<BufferOp> allocatedBuffers;

  // Bank-pinned buffers lead the order, then the core's data region, then the
  // unconstrained ones -- most-constrained first, since a pin has one candidate
  // bank. They are searched together rather than in separate passes: where a
  // pin's position inside its bank decides whether a later buffer fits, only a
  // single search can revisit it.
  SmallVector<FitBuffer> records;
  for (BufferOp buffer : buffersToAlloc)
    records.push_back(
        {buffer.getAllocationSize(),
         getBufferAlignBytes(buffer, tileAlignBitWidth, maxVecAlignBits),
         buffer.getInitialValue().has_value(), requiredBanks.lookup(buffer)});
  SmallVector<BufferOp> order;
  SmallVector<FitBuffer> fit;
  for (unsigned index : fitOrder(records)) {
    order.push_back(buffersToAlloc[index]);
    fit.push_back(records[index]);
  }
  FitResult result = fitTileBuffers(layout, fit, occupancy, budget, stats);
  for (auto &[index, at] : result.placed) {
    order[index].setAddress(at.addr);
    order[index].setMemBank(at.bank);
    allocatedBuffers.push_back(order[index]);
  }
  bool searchExhausted = result.exhausted;
  BufferOp failedBuffer = result.failed ? order[*result.failed] : nullptr;

  if (BufferOp failed = failedBuffer) {
    // A buffer pinned to a bank that cannot hold it is a user constraint, not
    // an out-of-room tile: give it its own error and no memory map. Unless the
    // search merely ran out of budget, in which case the bank may have room.
    if (!searchExhausted && requiredBanks.count(failed)) {
      auto banks = requiredBanks.lookup(failed);
      if (banks.size() > 1) {
        failed.emitOpError("")
            << bufferLabel(failed) << " requires " << failed.getAllocationSize()
            << " bytes, but no contiguous aligned space remains within allowed "
               "banks "
            << banks.front() << " and " << banks.back();
        deAllocationBuffers(allocatedBuffers, requiredBanks);
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
            << bank << ", but only "
            << occupancy.freeBytes(bankLimits[bank].start,
                                   bankLimits[bank].end())
            << " of " << bankCapacity << " bytes are free there";
      }
      deAllocationBuffers(allocatedBuffers, requiredBanks);
      return failure();
    }
    InFlightDiagnostic diag = failed.emitOpError("could not be placed: ")
                              << bufferLabel(failed) << " needs "
                              << failed.getAllocationSize()
                              << " bytes and this tile has no room left for it";
    if (searchExhausted) {
      diag.attachNote() << "the search hit its " << budget
                        << "-placement budget with arrangements still untried, "
                           "so a layout may exist that it did not reach (raise "
                           "it with placement-budget)";
    }
    // Print before rollback, while the addresses are still set.
    printMemMap(tile, allocatedBuffers, preAllocatedBuffers, ctx);
    deAllocationBuffers(allocatedBuffers, requiredBanks);
    return failure();
  }
  assert(allocatedBuffers.size() == buffersToAlloc.size());

  sortBuffersByAddress(allBuffers_on_tile);
  // Every placement came from free space in the tile, so overflow cannot happen
  // here. The stack and overlap checks remain as a backstop.
  if (!checkAndPrintOverlapStackframe(stackRun, allBuffers_on_tile) ||
      !checkAndPrintBufferOverlap(allBuffers_on_tile, tileAlignBitWidth,
                                  maxVecAlignBits)) {
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
    if (failed(applySignatureBankConstraints(device))) {
      return signalPassFailure();
    }
    materializeCoreDataBuffers(device);
    materializeBankReservations(device);
    materializePrebakedRanges(device);

    // One allocator, one answer. There used to be two schemes and a fallback
    // between them, which meant a tile that ran out of room reported twice and
    // explained itself once, and a `mem_bank` pin could be dropped on the way
    // from one scheme to the other. Placement now either finds a layout or says
    // why it could not.
    FitStats stats;
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
