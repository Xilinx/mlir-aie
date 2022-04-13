//===- AIEAssignBuffers.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-assign-buffers"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

static int64_t assignAddress(AIE::BufferOp op, int64_t lastAddress,
                             OpBuilder &rewriter) {
  Operation *Op = op.getOperation();
  rewriter.setInsertionPointToEnd(Op->getBlock());

  int64_t startAddr = lastAddress;
  int64_t endAddr = startAddr + op.getAllocationSize();
  Op->setAttr("address", rewriter.getI32IntegerAttr(startAddr));
  // Fixme: alignment
  return endAddr;
}

struct AIEAssignBufferAddressesPass
    : public AIEAssignBufferAddressesBase<AIEAssignBufferAddressesPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());
    // Make sure all the buffers have a name
    int counter = 0;
    for (auto buffer : m.getOps<BufferOp>()) {
      if (!buffer.hasName()) {
        std::string name = "_anonymous";
        name += std::to_string(counter++);
        buffer->setAttr(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
      }
    }

    for (auto tile : m.getOps<TileOp>()) {
      SmallVector<BufferOp, 4> buffers;
      // Collect all the buffers for this tile.
      for (auto buffer : m.getOps<BufferOp>())
        if (buffer.getTileOp() == tile)
          buffers.push_back(buffer);
      // Sort by allocation size.
      std::sort(buffers.begin(), buffers.end(), [](BufferOp a, BufferOp b) {
        return a.getAllocationSize() > b.getAllocationSize();
      });

      // Address range owned by the tile is 0x8000,
      // but we need room at the bottom for stack.
      int address = 0x1000;
      for (auto buffer : buffers)
        address = assignAddress(buffer, address, builder);
      if (address > 0x8000) {
        tile.emitOpError("allocated buffers exceeded available memory");
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEAssignBufferAddressesPass() {
  return std::make_unique<AIEAssignBufferAddressesPass>();
}