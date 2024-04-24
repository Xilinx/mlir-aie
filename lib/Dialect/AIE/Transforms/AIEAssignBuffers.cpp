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
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Twine.h"

#define DEBUG_TYPE "aie-assign-buffers"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

//===----------------------------------------------------------------------===//
// BasicAllocation : sequential alloc from largest to smallest buffer size
//===----------------------------------------------------------------------===//
struct BasicAllocationPattern : public OpRewritePattern<TileOp> {
  using OpRewritePattern<TileOp>::OpRewritePattern;

  BasicAllocationPattern(
      MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(TileOp tile,
                                PatternRewriter &rewriter) const override {
    auto device = tile->getParentOfType<AIE::DeviceOp>();
    if (!device)
      return failure();

    const auto &targetModel = getTargetModel(tile);
    int maxDataMemorySize = 0;
    if (tile.isMemTile())
      maxDataMemorySize = targetModel.getMemTileSize();
    else
      maxDataMemorySize = targetModel.getLocalMemorySize();

    SmallVector<BufferOp, 4> buffers;
    // Collect all the buffers for this tile.
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (buffer.getTileOp() == tile)
        buffers.push_back(buffer);
    });
    // Sort by allocation size.
    std::sort(buffers.begin(), buffers.end(), [](BufferOp a, BufferOp b) {
      return a.getAllocationSize() > b.getAllocationSize();
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

    for (auto buffer : buffers) {
      if (buffer.getAddress())
        buffer->emitWarning("Overriding existing address");
      buffer.setAddress(address);
      address += buffer.getAllocationSize();
    }

    // Sort by smallest address before printing memory map.
    std::sort(buffers.begin(), buffers.end(),
              [](BufferOp a, BufferOp b) {
                assert(a.getAddress().has_value() &&
                        "buffer must have address assigned");
                assert(b.getAddress().has_value() &&
                        "buffer must have address assigned");
                return a.getAddress().value() < b.getAddress().value();
              });
    // Check if memory was exceeded on any bank and print debug info.
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
      return failure();
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SimpleBankAwareAllocation : round-robin each alloc over available banks
//===----------------------------------------------------------------------===//
struct SimpleBankAwareAllocationPattern : public OpRewritePattern<TileOp> {
  using OpRewritePattern<TileOp>::OpRewritePattern;

  SimpleBankAwareAllocationPattern(
      MLIRContext *ctx) : OpRewritePattern(ctx) {}

  typedef struct BankLimits {
    int64_t startAddr;
    int64_t endAddr;
  } BankLimits;

  LogicalResult matchAndRewrite(TileOp tile,
                                PatternRewriter &rewriter) const override {
    auto device = tile->getParentOfType<AIE::DeviceOp>();
    if (!device)
      return failure();

    std::vector<int64_t>
      nextAddrInBanks; // each entry is the next address available for use
                       // for that bank
                       // (e.g. nextAddrInBanks[tile_0][1] = next available
                       // address in bank 1 for tile_0)
    std::vector<BankLimits>
        bankLimits; // the entries contain pairs of start and end addresses
                    // for each bank

    const auto &targetModel = getTargetModel(tile);
    int maxDataMemorySize = 0;
    if (tile.isMemTile())
      maxDataMemorySize = targetModel.getMemTileSize();
    else
      maxDataMemorySize = targetModel.getLocalMemorySize();

    int numBanks = getNumBanks(tile);
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

    SmallVector<BufferOp, 4> buffers;
    SmallVector<BufferOp, 4> allBuffers;
    // Collect all the buffers for this tile.
    // If possible, the buffers with an already specified address will not
    // be overwritten (the available address range of the bank the buffers
    // are in will start AFTER the specified adress + buffer size).
    // Buffers with a specified mem_bank will be assigned first, after
    // the above.
    for (auto buffer : device.getOps<BufferOp>()) {
      if (buffer.getTileOp() == tile) {
        bool has_addr = checkAndAddBufferWithAddress(buffer, numBanks, nextAddrInBanks, bankLimits);
        bool has_bank = checkAndAddBufferWithMemBank(buffer, numBanks, nextAddrInBanks, bankLimits);
        if (!has_addr && !has_bank)
          buffers.push_back(buffer);
        allBuffers.push_back(buffer);
      }
    }

    // Sort by largest allocation size before allocating.
    std::sort(buffers.begin(), buffers.end(), [](BufferOp a, BufferOp b) {
      return a.getAllocationSize() > b.getAllocationSize();
    });

    // Set addresses for remaining buffers.
    int bankIndex = 0;
    for (auto buffer : buffers)
      bankIndex = setBufferAddress(buffer, numBanks, bankIndex, nextAddrInBanks, bankLimits);

    // Sort by smallest address before printing memory map.
    std::sort(allBuffers.begin(), allBuffers.end(),
              [](BufferOp a, BufferOp b) {
                assert(a.getAddress().has_value() &&
                        "buffer must have address assigned");
                assert(b.getAddress().has_value() &&
                        "buffer must have address assigned");
                return a.getAddress().value() < b.getAddress().value();
              });
    // Check if memory was exceeded on any bank and print debug info.
    return checkAndPrintOverflow(tile, numBanks, stacksize, allBuffers, nextAddrInBanks, bankLimits);
  }

private:
  // TODO: add to target model
  int getNumBanks(TileOp tile) const {
    if (tile.isMemTile())
      return 1;
    else
      return 4;
  }

  // Function that given a number of banks and their size, computes
  // the start and end addresses for each bank and fills in the entry
  // in the bankLimits vector.
  void fillBankLimits(int numBanks, int bankSize, std::vector<BankLimits>
                      &bankLimits) const {
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
                                 int64_t end_addr, std::vector<int64_t>
                                 &nextAddrInBanks) const {
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
  bool checkAndAddBufferWithAddress(BufferOp buffer, int numBanks,
                                    std::vector<int64_t> &nextAddrInBanks,
                                    std::vector<BankLimits> &bankLimits) const {
    if (auto addrAttr = buffer->getAttrOfType<IntegerAttr>("address")) {
      int addr = addrAttr.getInt();
      for (int i = 0; i < numBanks; i++) {
        if (bankLimits[i].startAddr <= addr &&
            addr < bankLimits[i].endAddr) {
          if (addr >= nextAddrInBanks[i]) {
            nextAddrInBanks[i] = addr + buffer.getAllocationSize();
            buffer.setMemBank(i);
          } else {
            buffer->emitWarning("Overriding existing address");
            return false;
          }
        }
      }
      return true;
    }
    return false;
  }

  // Function that checks whether the given buffer already has a set mem_bank
  // attribute. If it does, it checks whether there is enough space left for
  // it. If there is, it sets the buffer's address field and if not, the
  // function emits a warning that the mem_bank will be overwritten and returns
  // false (which will cause the buffer to be added to the list of buffers
  // without addresses, to be completed later on).
  bool checkAndAddBufferWithMemBank(BufferOp buffer, int numBanks,
                                    std::vector<int64_t> &nextAddrInBanks,
                                    std::vector<BankLimits> &bankLimits) const {
    if (auto memBankAttr = buffer->getAttrOfType<IntegerAttr>("mem_bank")) {
      int mem_bank = memBankAttr.getInt();
      int64_t startAddr = nextAddrInBanks[mem_bank];
      int64_t endAddr = startAddr + buffer.getAllocationSize();
      if (endAddr <= bankLimits[mem_bank].endAddr) {
        setAndUpdateAddressInBank(buffer, startAddr, endAddr, nextAddrInBanks);
      } else {
        buffer->emitWarning("Overriding existing mem_bank");
        return false;
      }
      return true;
    }
    return false;
  }

  // Function that given a buffer will iterate over all the memory banks
  // starting from the given index to try and find a bank with enough
  // space. If it does, it will set the buffer's address and mem_bank
  // attributes and update the nextAddrInBanks vector.
  // If it does not find one with enough space, it will allocate the
  // buffer in the last checked bank (this will be picked up during
  // overflow error checking). Finally, the function returns the index
  // of the next bank to search (which should be given to subsequent
  // calls of this function to ensure a round-robin allocation scheme
  // over the available banks).
  int setBufferAddress(BufferOp buffer, int numBanks, int startBankIndex,
                       std::vector<int64_t> &nextAddrInBanks,
                       std::vector<BankLimits> &bankLimits) const {
    int bankIndex = startBankIndex;
    for (int i = 0; i < numBanks; i++) {
      int64_t startAddr = nextAddrInBanks[bankIndex];
      int64_t endAddr = startAddr + buffer.getAllocationSize();
      if (endAddr <= bankLimits[bankIndex].endAddr || i == numBanks - 1) {
        buffer.setMemBank(bankIndex);
        setAndUpdateAddressInBank(buffer, startAddr, endAddr, nextAddrInBanks);
        bankIndex++;
        break;
      }
      bankIndex++;
    }
    bankIndex %= numBanks;
    return bankIndex;
  }

  LogicalResult checkAndPrintOverflow(TileOp tile, int numBanks, int stacksize,
                             SmallVector<BufferOp, 4> allBuffers,
                             std::vector<int64_t> &nextAddrInBanks,
                             std::vector<BankLimits> &bankLimits) const {
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
          tile.emitOpError("allocated buffers exceeded available memory\n");
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
              << "0x" << llvm::utohexstr(bankLimits[i].startAddr)
              << "-0x" << llvm::utohexstr(bankLimits[i].endAddr - 1)
              << "\n";
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
      return failure();
    }
    return success();
  }
};

struct AIEAssignBufferAddressesPass
    : AIEAssignBufferAddressesBase<AIEAssignBufferAddressesPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());
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

    auto ctx = device->getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    // Select allocation scheme
    if (clBasicAlloc) {
      patterns.insert<BasicAllocationPattern>(ctx);
    } else {
      patterns.insert<SimpleBankAwareAllocationPattern>(ctx);
    }

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferAddressesPass() {
  return std::make_unique<AIEAssignBufferAddressesPass>();
}
