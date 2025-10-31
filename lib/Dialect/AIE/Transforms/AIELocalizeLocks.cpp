//===- AIELocalizeLocks.cpp ---------------------------------------*- C++
//-*-===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "aie-localize-locks"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIELocalizeLocksPass : AIELocalizeLocksBase<AIELocalizeLocksPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {

    DeviceOp deviceOp = getOperation();

    for (auto coreOp : deviceOp.getOps<CoreOp>()) {
      // Collect the locks used in this core.
      const auto &targetModel = getTargetModel(coreOp);

      auto thisTile = dyn_cast<TileOp>(coreOp.getTile().getDefiningOp());
      int col = thisTile.colIndex();
      int row = thisTile.rowIndex();

      // Find the neighboring tiles
      SmallVector<TileOp, 4> accessibleTiles;
      for (auto tile : deviceOp.getOps<TileOp>())
        if (int dstRow = tile.rowIndex();
            targetModel.isLegalMemAffinity(col, row, tile.colIndex(), dstRow))
          accessibleTiles.push_back(tile);

      for (auto tile : accessibleTiles) {
        int dstCol = tile.colIndex();
        int dstRow = tile.rowIndex();

        const auto &targetModel = getTargetModel(tile);
        for (auto user : tile.getResult().getUsers())
          if (auto lock = dyn_cast<LockOp>(user)) {
            // At this point, we are iterating over all locks that are
            // accessible from within the current core coreOp. We only need to
            // localize the current lock if it is used within the core. Note
            // that this pass is to be applied after the scf-to-cf lowering, so
            // it suffices to check if the parent of a UseLockOp is coreOp.
            if (llvm::none_of(lock.getResult().getUsers(),
                              [&](Operation *user) {
                                return user->getParentOp() == coreOp;
                              }))
              continue;

            auto lockIndexOffset =
                targetModel.getLockLocalBaseIndex(col, row, dstCol, dstRow);
            if (!lockIndexOffset)
              llvm_unreachable("Found illegal lock user!");

            int localLockIndex =
                lockIndexOffset.value() + lock.getLockIDValue();

            OpBuilder builder =
                OpBuilder::atBlockBegin(&coreOp.getBody().front());

            Value coreLockIDValue = builder.create<arith::ConstantIndexOp>(
                builder.getUnknownLoc(), localLockIndex);
            lock.getResult().replaceUsesWithIf(
                coreLockIDValue, [&](OpOperand &opOperand) {
                  return opOperand.getOwner()->getParentOp() == coreOp;
                });
          }
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIELocalizeLocksPass() {
  return std::make_unique<AIELocalizeLocksPass>();
}