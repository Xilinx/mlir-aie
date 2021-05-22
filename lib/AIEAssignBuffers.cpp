// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "AIEDialect.h"

#define DEBUG_TYPE "aie-find-flows"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

static int64_t assignAddress(AIE::BufferOp op,
                             int64_t lastAddress,
                             OpBuilder &rewriter) {
  Operation *Op = op.getOperation();
  rewriter.setInsertionPointToEnd(Op->getBlock());

  int64_t startAddr = lastAddress;
  int64_t endAddr = startAddr + op.getAllocationSize();
  Op->setAttr("address", rewriter.getI32IntegerAttr(startAddr));
  // Fixme: alignment
  return endAddr;
}

struct AIEAssignBufferAddressesPass : public PassWrapper<AIEAssignBufferAddressesPass,
                                             OperationPass<ModuleOp>> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());
    for (auto tile : m.getOps<TileOp>()) {
      SmallVector<BufferOp, 4> buffers;
      // Collect all the buffers for this tile.
      for (auto buffer : m.getOps<BufferOp>())
        if(buffer.getTileOp() == tile)
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
      if(address > 0x8000) {
        tile.emitOpError("allocated buffers exceeded available memory");
      }
    }
  }
};

void xilinx::AIE::registerAIEAssignBufferAddressesPass() {
    PassRegistration<AIEAssignBufferAddressesPass>(
      "aie-assign-buffer-addresses",
      "Assign memory locations for buffers in each tile");
}
