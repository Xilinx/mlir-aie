//===- AIELowerSetLock.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIETokenAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-lower-set-lock"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

struct SetLockToWrite32Pattern : OpConversionPattern<SetLockOp> {
  using OpConversionPattern<SetLockOp>::OpConversionPattern;

public:
  SetLockToWrite32Pattern(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(SetLockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    const AIE::AIETargetModel &tm = dev.getTargetModel();

    auto lockOp = op.getLockOp();
    if (!lockOp.getLockID()) {
      op->emitError("Tried to lower a SetLockOp on an unassigned lock");
      return failure();
    }

    auto col = lockOp.colIndex();
    auto row = lockOp.rowIndex();
    uint32_t lockID = lockOp.getLockIDValue();

    // The validity of this optional is already checked in the verifier
    auto localLockAddress =
        tm.getLocalLockAddress(lockID, lockOp.getTileID()).value();

    rewriter.replaceOpWithNewOp<NpuWrite32Op>(
        op, localLockAddress, op.getValue(), nullptr,
        rewriter.getI32IntegerAttr(col), rewriter.getI32IntegerAttr(row));

    return success();
  };
};

struct AIELowerSetLockPass : public AIELowerSetLockBase<AIELowerSetLockPass> {
  void runOnOperation() override {

    DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalOp<NpuWrite32Op>();
    target.addIllegalOp<SetLockOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<SetLockToWrite32Pattern>(&getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIELowerSetLockPass() {
  return std::make_unique<AIELowerSetLockPass>();
}
