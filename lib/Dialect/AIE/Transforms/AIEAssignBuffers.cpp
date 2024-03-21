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

#include "llvm/ADT/Twine.h"

#define DEBUG_TYPE "aie-assign-buffers"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIEAssignBufferAddressesPass
    : AIEAssignBufferAddressesBase<AIEAssignBufferAddressesPass> {

  typedef struct BankLimits {
    int64_t startAddr;
    int64_t endAddr;
  } BankLimits;

  std::vector<int64_t> nextAddrInBanks; // each entry is the next address available for use for that bank (e.g. nextAddrInBanks[1] = next available address in bank 1)
  std::vector<BankLimits> bankLimits;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void fillBankLimits(int numBanks, int bankSize) {
    for (int i = 0; i < numBanks; i++) {
      auto startAddr = bankSize * i;
      auto endAddr = bankSize * (i + 1);
      bankLimits.push_back({startAddr, endAddr});
    }
  }

  bool checkAndAddBufferWithAddress(BufferOp buffer, int bankSize) {
    if (auto addrAttr = buffer->getAttrOfType<IntegerAttr>("address")) {
      int addr = addrAttr.getInt();
      if ((0 <= addr) && (addr < bankSize)) {
        if (addr >= nextAddrInBanks[0])
          nextAddrInBanks[0] = addr + buffer.getAllocationSize();
      } else if (addr < bankSize * 2) {
        if (addr >= nextAddrInBanks[1])
          nextAddrInBanks[1] = addr + buffer.getAllocationSize();
      } else if (addr < bankSize * 3) {
        if (addr >= nextAddrInBanks[2])
          nextAddrInBanks[2] = addr + buffer.getAllocationSize();
      } else {
        if (addr >= nextAddrInBanks[3])
          nextAddrInBanks[3] = addr + buffer.getAllocationSize();
      }
      return true;
    }
    return false;
  }

  bool checkAndAddBufferWithMemBank(BufferOp buffer, int numBanks, int bankSize) {
    if (auto memBankAttr = buffer->getAttrOfType<IntegerAttr>("mem_bank")) {
      int mem_bank = memBankAttr.getInt();
      int64_t startAddr = nextAddrInBanks[mem_bank];
      int64_t endAddr = startAddr + buffer.getAllocationSize();
      if (endAddr <= bankLimits[mem_bank].endAddr) {
        nextAddrInBanks[mem_bank] = endAddr;
        // Fixme: alignment
        buffer.setAddress(startAddr);
      } else {
        buffer->emitWarning("Overriding existing mem_bank");
        int bankIndex = mem_bank;
        for (int i = 0; i < numBanks; i++) {
          bankIndex %= numBanks;
          startAddr = nextAddrInBanks[bankIndex];
          endAddr = startAddr + buffer.getAllocationSize();
          if (endAddr <= bankLimits[bankIndex].endAddr) {
            break;
          }
          bankIndex++;
        }
        nextAddrInBanks[bankIndex] = endAddr;
        // Fixme: alignment
        buffer.setAddress(startAddr);
      }
      buffer->removeAttr("mem_bank");
      return true;
    }
    return false;
  }

  int setBufferAddress(BufferOp buffer, int numBanks, int bankSize, int startBankIndex) {
    int bankIndex = startBankIndex;
    int64_t startAddr = 0;
    int64_t endAddr = 0;
    for (int i = 0; i < numBanks; i++) {
      bankIndex %= numBanks;
      startAddr = nextAddrInBanks[bankIndex];
      endAddr = startAddr + buffer.getAllocationSize();
      if (endAddr <= bankLimits[bankIndex].endAddr || i == numBanks - 1) {
        nextAddrInBanks[bankIndex] = endAddr;
        // Fixme: alignment
        buffer.setAddress(startAddr);
        bankIndex++;
        break;
      }
      bankIndex++;
    }
    return bankIndex;
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

    for (auto tile : device.getOps<TileOp>()) {
      const auto &targetModel = getTargetModel(tile);
      int maxDataMemorySize = 0;
      if (tile.isMemTile())
        maxDataMemorySize = targetModel.getMemTileSize();
      else
        maxDataMemorySize = targetModel.getLocalMemorySize();

      // Address range owned by the MemTile is 0x80000.
      // Address range owned by the tile is 0x8000 in
      // AIE1 and 0x10000 in AIE2, but we need room at
      // the bottom for stack.
      // TODO: add function in TargetModel that returns num banks per arch
      int numBanks = 0;
      if (tile.isMemTile())
        numBanks = 1;
      else
        numBanks = 4;
      int bankSize = maxDataMemorySize / numBanks;
      int stacksize = 0;
      for (int i = 0; i < numBanks; i++)
        nextAddrInBanks.push_back(bankSize * i);
      if (auto core = tile.getCoreOp()) {
        stacksize = core.getStackSize();
        nextAddrInBanks[0] = 0;
        nextAddrInBanks[0] += stacksize;
      }
      fillBankLimits(numBanks, bankSize);

      SmallVector<BufferOp, 4> buffers;
      SmallVector<BufferOp, 4> allBuffers;
      // Collect all the buffers for this tile.
      // The buffers with an already specified address will not be 
      // overwritten (the available address range of the bank the buffers 
      // are in will start AFTER the specified adress + buffer size).
      // Buffers with a specified mem_bank will be assigned first, after 
      // the above.
      for (auto buffer : device.getOps<BufferOp>()) {
        if (buffer.getTileOp() == tile) {
          bool has_addr = checkAndAddBufferWithAddress(buffer, bankSize);
          bool has_bank = checkAndAddBufferWithMemBank(buffer, numBanks, bankSize);
          if (!has_addr && !has_bank)
            buffers.push_back(buffer);
          allBuffers.push_back(buffer);
        }
      }

      // Sort by largest allocation size.
      std::sort(buffers.begin(), buffers.end(), [](BufferOp a, BufferOp b) {
        return a.getAllocationSize() > b.getAllocationSize();
      });

      // Set addresses for remaining buffers.
      int bankIndex = 0;
      for (auto buffer : buffers) {
        bankIndex = setBufferAddress(buffer, numBanks, bankSize, bankIndex);
      }

      // Check if memory was exceeded on any bank and print debug info.
      if (!tile.isMemTile() && !tile.isShimTile()) {
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
            note << "\t" << "\t" << name << " \t"
                << ": 0x" << llvm::utohexstr(address) << "-0x"
                << llvm::utohexstr(address + size - 1) << " \t(" << size
                << " bytes)\n";
          };
          for (int i = 0; i < numBanks; i++) {
            note << "\t" << "bank : " << i << "\t"
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
              assert(buffer.getAddress().has_value() &&
                    "buffer must have address assigned");
              auto addr = buffer.getAddress().value();
              if (bankLimits[i].startAddr <= addr && addr < bankLimits[i].endAddr) 
                printbuffer(buffer.name(), addr, buffer.getAllocationSize());
            }
          }
          return signalPassFailure();
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferAddressesPass() {
  return std::make_unique<AIEAssignBufferAddressesPass>();
}
