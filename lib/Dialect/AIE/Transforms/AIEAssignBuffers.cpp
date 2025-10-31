//===- AIEAssignBuffers.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"

#define DEBUG_TYPE "aie-assign-buffers"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

//===----------------------------------------------------------------------===//
// BasicAllocation : sequential alloc from largest to smallest
//===----------------------------------------------------------------------===//
bool checkAndPrintOverflow(TileOp tile, int address, int maxDataMemorySize,
                           int stacksize, SmallVector<BufferOp> &buffers) {
  if (address > maxDataMemorySize) {
    InFlightDiagnostic error =
        tile.emitOpError("allocated buffers exceeded available memory\n");
    auto &note = error.attachNote() << "MemoryMap:\n";
    auto printbuffer = [&](StringRef name, int address, int size) {
      note << "\t" << name << " \t"
           << ": 0x" << llvm::utohexstr(address) << "-0x"
           << llvm::utohexstr(address + size - 1) << " \t(" << size
           << " bytes)\n";
    };
    if (stacksize > 0)
      printbuffer("(stack)", 0, stacksize);
    else
      error << "(no stack allocated)\n";

    for (auto buffer : buffers) {
      assert(buffer.getAddress().has_value() &&
             "buffer must have address assigned");
      printbuffer(buffer.name(), buffer.getAddress().value(),
                  buffer.getAllocationSize());
    }
    return false;
  }
  return true;
}

bool basicAllocation(TileOp tile) {
  auto device = tile->getParentOfType<AIE::DeviceOp>();
  if (!device)
    return false;

  const auto &targetModel = getTargetModel(tile);
  int maxDataMemorySize = 0;
  if (tile.isMemTile())
    maxDataMemorySize = targetModel.getMemTileSize();
  else
    maxDataMemorySize = targetModel.getLocalMemorySize();

  SmallVector<BufferOp> buffers;
  SmallVector<BufferOp> allocated_buffers;
  // Collect all the buffers for this tile. If the buffer has an address, add
  // it to allocated_buffers. Otherwise, add it to buffers.
  device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
    if (buffer.getTileOp() == tile) {
      if (buffer.getAddress())
        allocated_buffers.push_back(buffer);
      else
        buffers.push_back(buffer);
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
  int stacksize = 0;
  int address = 0;
  if (auto core = tile.getCoreOp()) {
    stacksize = core.getStackSize();
    address += stacksize;
  }

  // As the next address to allocate is assigned, skip over any buffers
  // from the allocated_buffers list.
  auto current_alloc = allocated_buffers.begin();
  for (auto buffer : buffers) {
    assert(!buffer.getAddress());
    while (current_alloc != allocated_buffers.end() &&
           address + buffer.getAllocationSize() >
               current_alloc->getAddress().value()) {
      address = current_alloc->getAddress().value() +
                current_alloc->getAllocationSize();
      current_alloc++;
    }
    buffer.setAddress(address);
    address += buffer.getAllocationSize();
  }

  // Sort by smallest address before printing memory map.
  std::sort(buffers.begin(), buffers.end(), [](BufferOp a, BufferOp b) {
    assert(a.getAddress().has_value() && "buffer must have address assigned");
    assert(b.getAddress().has_value() && "buffer must have address assigned");
    return a.getAddress().value() < b.getAddress().value();
  });
  // Check if memory was exceeded on any bank and print debug info.
  return checkAndPrintOverflow(tile, address, maxDataMemorySize, stacksize,
                               buffers);
}

//===----------------------------------------------------------------------===//
// SimpleBankAwareAllocation : round-robin each alloc over available banks
//===----------------------------------------------------------------------===//
typedef struct BankLimits {
  int64_t startAddr;
  int64_t endAddr;
} BankLimits;

// Function that given a number of banks and their size, computes
// the start and end addresses for each bank and fills in the entry
// in the bankLimits vector.
void fillBankLimits(int numBanks, int bankSize,
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
void setAndUpdateAddressInBank(BufferOp buffer, int64_t start_addr,
                               int64_t end_addr,
                               std::vector<int64_t> &nextAddrInBanks) {
  // Fixme: alignment
  buffer.setAddress(start_addr);
  nextAddrInBanks[buffer.getMemBank().value()] = end_addr;
}

// Function that checks whether the given buffer already has a set address
// attribute. If it does, it finds in which bank the buffer is and checks
// whether there is enough space left for it. If there is the function
// returns true and if not, the function emits a warning that the address
// will be overwritten and returns false (which will cause the buffer to be
// added to the list of buffers without addresses, to be completed later on).
FailureOr<bool>
checkAndAddBufferWithAddress(BufferOp buffer, int numBanks,
                             std::vector<int64_t> &nextAddrInBanks,
                             std::vector<BankLimits> &bankLimits) {
  auto addrAttr = buffer->getAttrOfType<IntegerAttr>("address");
  if (!addrAttr)
    return false;

  int addr = addrAttr.getInt();
  for (int i = 0; i < numBanks; i++) {
    // if the address is not within the bank, continue
    if (addr < bankLimits[i].startAddr || addr >= bankLimits[i].endAddr)
      continue;

    // if the allocator already allocated this address, fail
    if (addr < nextAddrInBanks[i])
      return buffer->emitOpError("would override allocated address");

    // the allocator can accomadate this existing allocation
    nextAddrInBanks[i] = addr + buffer.getAllocationSize();
    buffer.setMemBank(i);
  }
  return true;
}

// Function that checks whether the given buffer already has a set mem_bank
// attribute. If it does, it checks whether there is enough space left for
// it. If there is, it sets the buffer's address field and if not, the
// function emits a warning that the mem_bank will be overwritten and returns
// false (which will cause the buffer to be added to the list of buffers
// without addresses, to be completed later on).
FailureOr<bool>
checkAndAddBufferWithMemBank(BufferOp buffer, int numBanks,
                             std::vector<int64_t> &nextAddrInBanks,
                             std::vector<BankLimits> &bankLimits) {
  auto memBankAttr = buffer->getAttrOfType<IntegerAttr>("mem_bank");
  if (!memBankAttr)
    return false;

  int mem_bank = memBankAttr.getInt();
  int64_t startAddr = nextAddrInBanks[mem_bank];
  int64_t endAddr = startAddr + buffer.getAllocationSize();
  if (endAddr > bankLimits[mem_bank].endAddr)
    return buffer->emitOpError("would override existing mem_bank");
  setAndUpdateAddressInBank(buffer, startAddr, endAddr, nextAddrInBanks);
  return true;
}

// Prints the memory map across banks
void printMemMap(TileOp tile, SmallVector<BufferOp> &allocatedBuffers,
                 SmallVector<BufferOp> &preAllocatedBuffers, int numBanks,
                 std::vector<BankLimits> &bankLimits, int stacksize) {
  InFlightDiagnostic error = tile.emitWarning(
      "Not all requested buffers fit in the available memory.\n");
  auto &note = error.attachNote()
               << "Current configuration of buffers in bank(s) : ";
  note << "MemoryMap:\n";
  auto printbuffer = [&](StringRef name, int address, int size) {
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
        note << "(no stack allocated)\n";
    }
    note << "\t"
         << "bank : " << i << "\t"
         << "0x" << llvm::utohexstr(bankLimits[i].startAddr) << "-0x"
         << llvm::utohexstr(bankLimits[i].endAddr - 1) << "\n";
    for (auto buffer : preAllocatedBuffers) {
      auto addr = buffer.getAddress().value();
      auto mem_bank = buffer.getMemBank().value();
      if (mem_bank == i)
        printbuffer(buffer.name(), addr, buffer.getAllocationSize());
    }
    for (auto buffer : allocatedBuffers) {
      auto addr = buffer.getAddress().value();
      auto mem_bank = buffer.getMemBank().value();
      if (mem_bank == i)
        printbuffer(buffer.name(), addr, buffer.getAllocationSize());
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

int setBufferAddress(BufferOp buffer, int numBanks, int &startBankIndex,
                     std::vector<int64_t> &nextAddrInBanks,
                     std::vector<BankLimits> &bankLimits) {
  assert(startBankIndex < numBanks &&
         "Unexpected input value for startBankIndex");
  int bankIndex = startBankIndex;
  bool allocated = false;
  for (int i = 0; i < numBanks; i++) {
    int64_t startAddr = nextAddrInBanks[bankIndex];
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

bool checkAndPrintOverflow(TileOp tile, int numBanks, int stacksize,
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
      note << "\t"
           << "\t" << name << " \t"
           << ": 0x" << llvm::utohexstr(address) << "-0x"
           << llvm::utohexstr(address + size - 1) << " \t(" << size
           << " bytes)\n";
    };
    for (int i = 0; i < numBanks; i++) {
      note << "\t"
           << "bank : " << i << "\t"
           << "0x" << llvm::utohexstr(bankLimits[i].startAddr) << "-0x"
           << llvm::utohexstr(bankLimits[i].endAddr - 1) << "\n";
      if (i == 0) {
        if (stacksize > 0)
          printbuffer("(stack)", 0, stacksize);
        else
          error << "(no stack allocated)\n";
      }
      for (auto buffer : allBuffers) {
        auto addr = buffer.getAddress().value();
        auto mem_bank = buffer.getMemBank().value();
        if (mem_bank == i)
          printbuffer(buffer.name(), addr, buffer.getAllocationSize());
      }
    }
    return false;
  }
  return true;
}

// Function to deallocate attributes of buffers in case of a failure
void deAllocationBuffers(SmallVector<BufferOp> &buffers) {
  for (auto buffer : buffers) {
    buffer->removeAttr("address");
    buffer->removeAttr("mem_bank");
  }
}

bool simpleBankAwareAllocation(TileOp tile) {
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
  if (tile.isMemTile())
    maxDataMemorySize = targetModel.getMemTileSize();
  else
    maxDataMemorySize = targetModel.getLocalMemorySize();

  int numBanks = targetModel.getNumBanks(tile.getCol(), tile.getRow());
  int bankSize = maxDataMemorySize / numBanks;

  // Address range owned by the MemTile is 0x80000.
  // Address range owned by the tile is 0x8000 in
  // AIE1 and 0x10000 in AIE2, but we need room at
  // the bottom for stack.
  int stacksize = 0;
  for (int i = 0; i < numBanks; i++)
    nextAddrInBanks.push_back(bankSize * i);
  if (auto core = tile.getCoreOp()) {
    stacksize = core.getStackSize();
    nextAddrInBanks[0] += stacksize;
  }
  fillBankLimits(numBanks, bankSize, bankLimits);

  SmallVector<BufferOp> preAllocatedBuffers;
  SmallVector<BufferOp> buffersToAlloc;
  SmallVector<BufferOp> allBuffers;
  // Collect all the buffers for this tile.
  device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
    if (buffer.getTileOp() == tile)
      allBuffers.push_back(buffer);
  });
  // If possible, the buffers with an already specified address will not
  // be overwritten (the available address range of the bank the buffers
  // are in will start AFTER the specified adress + buffer size).
  // Buffers with a specified mem_bank will be assigned first, after
  // the above.
  for (auto buffer : allBuffers) {
    if (buffer.getTileOp() == tile) {
      auto has_addr = checkAndAddBufferWithAddress(buffer, numBanks,
                                                   nextAddrInBanks, bankLimits);
      auto has_bank = checkAndAddBufferWithMemBank(buffer, numBanks,
                                                   nextAddrInBanks, bankLimits);
      if (failed(has_addr) || failed(has_bank))
        return false;
      if (!has_addr.value() && !has_bank.value())
        buffersToAlloc.push_back(buffer);
      else
        preAllocatedBuffers.push_back(buffer);
    }
  }

  // Sort by largest allocation size before allocating.
  std::sort(buffersToAlloc.begin(), buffersToAlloc.end(),
            [](BufferOp a, BufferOp b) {
              return a.getAllocationSize() > b.getAllocationSize();
            });

  // Set addresses for remaining buffers.
  SmallVector<BufferOp> allocatedBuffers;
  int bankIndex = 0;
  for (auto buffer : buffersToAlloc) {
    // If the buffer doesn't fit in any of the bank space then
    // it prints the current memory map of the banks,
    // deallocates all the buffers, and
    // returns a failure.
    if (!setBufferAddress(buffer, numBanks, bankIndex, nextAddrInBanks,
                          bankLimits)) {

      printMemMap(tile, allocatedBuffers, preAllocatedBuffers, numBanks,
                  bankLimits, stacksize);
      deAllocationBuffers(allocatedBuffers);
      return false;
    } else {
      allocatedBuffers.push_back(buffer);
    }
  }

  // Sort by smallest address before printing memory map.
  std::sort(allBuffers.begin(), allBuffers.end(), [](BufferOp a, BufferOp b) {
    assert(a.getAddress().has_value() && "buffer must have address assigned");
    assert(b.getAddress().has_value() && "buffer must have address assigned");
    return a.getAddress().value() < b.getAddress().value();
  });
  // Check if memory was exceeded on any bank and print debug info.
  return checkAndPrintOverflow(tile, numBanks, stacksize, allBuffers,
                               nextAddrInBanks, bankLimits);
}

LogicalResult checkBufferScope(BufferOp buffer, DeviceOp device) {
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
    : AIEAssignBufferAddressesBase<AIEAssignBufferAddressesPass> {

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
        if (!simpleBankAwareAllocation(tile)) {
          tile.emitOpError("Bank-aware allocation failed.");
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
