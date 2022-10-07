//===- LockOp.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Transform/AIE/Physical/LockOp.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Rewrite/RemoveOp.h"

#include "aie/AIEDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::phy::physical;
using namespace xilinx::phy::transform::aie;

class LockOpToAieLowering : public OpConversionPattern<LockOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename LockOp::Adaptor;

public:
  LockOpToAieLowering(mlir::MLIRContext *context,
                      AIELoweringPatternSets *lowering)
      : OpConversionPattern<LockOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(LockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tile = lowering->getTile(op);
    auto id = lowering->getId(op);

    auto lock = lowering->getLock(tile, id);
    rewriter.eraseOp(op);
    // AIE.useLock(%0, state, Release)
    rewriter.setInsertionPointAfter(lock);
    rewriter.create<xilinx::AIE::UseLockOp>(rewriter.getUnknownLoc(), lock,
                                            op.getState(),
                                            xilinx::AIE::LockAction::Release);

    return success();
  }
};

class LockAcquireOpToAieLowering : public OpConversionPattern<LockAcquireOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename LockAcquireOp::Adaptor;

public:
  LockAcquireOpToAieLowering(mlir::MLIRContext *context,
                             AIELoweringPatternSets *lowering)
      : OpConversionPattern<LockAcquireOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(LockAcquireOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto phy_lock = dyn_cast_or_null<LockOp>(op.getLock().getDefiningOp());
    if (!phy_lock)
      return failure();

    auto tile = lowering->getTile(phy_lock);
    auto id = lowering->getId(phy_lock);

    auto lock = lowering->getLock(tile, id);
    rewriter.replaceOpWithNewOp<xilinx::AIE::UseLockOp>(
        op, lock, op.getState(), xilinx::AIE::LockAction::Acquire);

    return success();
  }
};

class LockReleaseOpToAieLowering : public OpConversionPattern<LockReleaseOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename LockReleaseOp::Adaptor;

public:
  LockReleaseOpToAieLowering(mlir::MLIRContext *context,
                             AIELoweringPatternSets *lowering)
      : OpConversionPattern<LockReleaseOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(LockReleaseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto phy_lock = dyn_cast_or_null<LockOp>(op.getLock().getDefiningOp());
    if (!phy_lock)
      return failure();

    auto tile = lowering->getTile(phy_lock);
    auto id = lowering->getId(phy_lock);

    auto lock = lowering->getLock(tile, id);
    rewriter.replaceOpWithNewOp<xilinx::AIE::UseLockOp>(
        op, lock, op.getState(), xilinx::AIE::LockAction::Release);

    return success();
  }
};

void LockOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {

  patterns.add<LockOpToAieLowering>(patterns.getContext(), lowering);
  patterns.add<LockAcquireOpToAieLowering>(patterns.getContext(), lowering);
  patterns.add<LockReleaseOpToAieLowering>(patterns.getContext(), lowering);
}
