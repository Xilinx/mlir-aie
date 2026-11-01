//===- AIESubstituteShimDMAAllocations.cpp -----------------------*- C++
//-*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIESUBSTITUTESHIMDMAALLOCATIONS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIESubstituteShimDMAAllocationsPass
    : xilinx::AIEX::impl::AIESubstituteShimDMAAllocationsBase<
          AIESubstituteShimDMAAllocationsPass> {

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    mlir::SymbolTable symbolTable(device);

    SmallVector<DMAConfigureTaskForOp> tasks;
    device.walk([&](DMAConfigureTaskForOp op) { tasks.push_back(op); });

    IRRewriter rewriter(&getContext());
    for (DMAConfigureTaskForOp op : tasks) {
      AIE::ShimDMAAllocationOp alloc_op =
          symbolTable.lookup<AIE::ShimDMAAllocationOp>(
              op.getAlloc().getRootReference());
      if (!alloc_op) {
        op.emitOpError("no shim DMA allocation found for symbol");
        return signalPassFailure();
      }

      AIE::TileOp tile = alloc_op.getTileOp();
      if (!tile) {
        op.emitOpError("shim DMA allocation must reference a valid TileOp");
        return signalPassFailure();
      }

      rewriter.setInsertionPoint(op);
      DMAConfigureTaskOp new_op = DMAConfigureTaskOp::create(
          rewriter, op.getLoc(), rewriter.getIndexType(), tile.getResult(),
          alloc_op.getChannelDirAttr(),
          rewriter.getI32IntegerAttr((int32_t)alloc_op.getChannelIndex()),
          rewriter.getBoolAttr(op.getIssueToken()),
          rewriter.getI32IntegerAttr(op.getRepeatCount()),
          /*repeat_count_val=*/op.getRepeatCountVal(),
          alloc_op.getPacket().value_or(nullptr));
      rewriter.replaceAllUsesWith(op.getResult(), new_op.getResult());
      rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(),
                                  new_op.getBody().begin());
      rewriter.eraseOp(op);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIESubstituteShimDMAAllocationsPass() {
  return std::make_unique<AIESubstituteShimDMAAllocationsPass>();
}
