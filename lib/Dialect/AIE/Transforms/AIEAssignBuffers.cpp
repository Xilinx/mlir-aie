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
      std::shared_ptr<AIETargetModel> targetModel = getTargetModel(tile);
      int maxDataMemorySize = 0;
      if (tile.isMemTile())
        maxDataMemorySize = targetModel->getMemTileSize();
      else
        maxDataMemorySize = targetModel->getLocalMemorySize();
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
      // Address range owned by the tile is 0x8000,
      // but we need room at the bottom for stack.
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
