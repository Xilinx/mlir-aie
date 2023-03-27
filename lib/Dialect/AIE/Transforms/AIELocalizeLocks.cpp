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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-localize-locks"
using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIELocalizeLocksPass
    : public AIELocalizeLocksBase<AIELocalizeLocksPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {

    DeviceOp deviceOp = getOperation();

    for (auto coreOp : deviceOp.getOps<CoreOp>()) {
      // Collect the locks used in this core.
      const auto &target_model = xilinx::AIE::getTargetModel(coreOp);

      TileOp thisTile = dyn_cast<TileOp>(coreOp.getTile().getDefiningOp());
      int col = thisTile.colIndex();
      int row = thisTile.rowIndex();

      // Find the neighboring tiles
      SmallVector<TileOp, 4> accessibleTiles;
      for (auto tile : deviceOp.getOps<TileOp>()) {
        int dstCol = tile.colIndex();
        int dstRow = tile.rowIndex();

        if (target_model.isLegalMemAffinity(col, row, dstCol, dstRow))
          accessibleTiles.push_back(tile);
      }

      for (auto tile : accessibleTiles) {
        int dstCol = tile.colIndex();
        int dstRow = tile.rowIndex();
        int cardinalMemOffset = 0;

        const auto &target_model = xilinx::AIE::getTargetModel(tile);
        for (auto user : tile.getResult().getUsers())
          if (auto lock = dyn_cast<LockOp>(user)) {
            if (target_model.isMemSouth(col, row, dstCol, dstRow))
              cardinalMemOffset = 0;
            else if (target_model.isMemWest(col, row, dstCol, dstRow))
              cardinalMemOffset = 16;
            else if (target_model.isMemNorth(col, row, dstCol, dstRow))
              cardinalMemOffset = 32;
            else if (target_model.isMemEast(col, row, dstCol, dstRow))
              cardinalMemOffset = 48;
            else
              llvm_unreachable("Found illegal lock user!");

            int localLockIndex = cardinalMemOffset + lock.getLockIDValue();

            OpBuilder builder =
                OpBuilder::atBlockBegin(&(coreOp.getBody().front()));

            Value coreLockIDValue = builder.create<arith::ConstantIndexOp>(
                builder.getUnknownLoc(), localLockIndex);
            // builder.getIndexType(),
            // //  IntegerType::get(builder.getContext(), 32),
            // builder.getI32IntegerAttr(localLockIndex));
            lock.getResult().replaceUsesWithIf(
                coreLockIDValue, [&](OpOperand &opOperand) {
                  return opOperand.getOwner()->getParentOp() == coreOp;
                });
          }
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIELocalizeLocksPass() {
  return std::make_unique<AIELocalizeLocksPass>();
}