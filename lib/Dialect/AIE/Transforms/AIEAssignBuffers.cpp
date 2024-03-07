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

    for (auto tile : device.getOps<TileOp>()) {
      // Address range owned by the MemTile is 0x80000.
      // Address range owned by the tile is 0x8000 in
      // AIE1 and 0x10000 in AIE2, but we need room at
      // the bottom for stack.
      const auto &targetModel = getTargetModel(tile);
      int maxDataMemorySize = 0;
      if (tile.isMemTile())
        maxDataMemorySize = targetModel.getMemTileSize();
      else
        maxDataMemorySize = targetModel.getLocalMemorySize();

      int numBanks = 0;
        if (tile.isMemTile())
          numBanks = 1;
        else 
          numBanks = 4;
      auto bankSize = maxDataMemorySize / numBanks;
      int stacksize = 0;
      int address = 0;
      std::vector<int64_t> nextAddrInBanks;
      for (int i = 0; i < numBanks; i++)
        nextAddrInBanks.push_back(bankSize * i);
      if (auto core = tile.getCoreOp()) {
        stacksize = core.getStackSize();
        address += stacksize;
        nextAddrInBanks[0] += stacksize;
      }

      SmallVector<BufferOp, 4> buffers;
      // Collect all the buffers for this tile.
      // Those without an address will be assigned one in a round-robin
      // fashion over the tile's memory banks. The buffers with an already
      // specified address will not be overwritten (the available address
      // range of the bank the buffers are in will start AFTER the specified
      // adress + buffer size).
      for (auto buffer : device.getOps<BufferOp>()) {
        if (buffer.getTileOp() == tile) {
          if (auto addrAttr = buffer->getAttrOfType<IntegerAttr>("address")) {
            auto addr = addrAttr.getInt();
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
          } else {
            buffers.push_back(buffer);
          }
        }
      }

      // Sort by allocation size.
      std::sort(buffers.begin(), buffers.end(), [](BufferOp a, BufferOp b) {
        return a.getAllocationSize() > b.getAllocationSize();
      });

      int bankIndex = 0;
      for (auto buffer : buffers) {
        //address += assignAddress(buffer, bankSize, nextAddrInBanks, builder);

        Operation *Op = buffer.getOperation();
        builder.setInsertionPointToEnd(Op->getBlock());

        for (int i = 0; i < numBanks; i++) {
          bankIndex %= numBanks;
          auto bankLimit = bankSize * (bankIndex + 1);
          auto startAddr = nextAddrInBanks[bankIndex];
          auto endAddr = startAddr + buffer.getAllocationSize();
          if (endAddr <= bankLimit) {
            nextAddrInBanks[bankIndex] = endAddr;
            // Fixme: alignment
            buffer->setAttr("address", builder.getI32IntegerAttr(startAddr));
            bankIndex++;
            break;
          }
          if (i == (numBanks - 1)) {
            nextAddrInBanks[bankIndex] = endAddr;
            buffer->setAttr("address", builder.getI32IntegerAttr(startAddr));
            bankIndex++;
            address += endAddr;
          }
        }
      }

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
        return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEAssignBufferAddressesPass() {
  return std::make_unique<AIEAssignBufferAddressesPass>();
}
