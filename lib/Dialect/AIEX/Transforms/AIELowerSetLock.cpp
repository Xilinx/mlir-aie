//===- AIELowerSetLock.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIETokenAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIELOWERSETLOCK
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

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

    // The validity of this optional is already checked in the verifier.
    auto localLockAddressOpt =
        tm.getLocalLockAddress(lockID, lockOp.getTileID());
    assert(localLockAddressOpt && "verifier guarantees a valid lock address");
    auto localLockAddress = *localLockAddressOpt;

    Location loc = op.getLoc();
    Value value = adaptor.getValue();
    uint32_t maxValue = tm.getMaxLockValue();
    Value inRange = rewriter.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ule, value,
        createConstantI32(rewriter, loc, maxValue));
    if (failed(emitRuntimeCheck(
            rewriter, loc, inRange,
            "a runtime lock value must be in [0:" + Twine(maxValue) + "]")))
      return failure();
    rewriter.replaceOpWithNewOp<NpuWrite32Op>(
        op, createConstantI32(rewriter, loc, localLockAddress), value, nullptr,
        rewriter.getI32IntegerAttr(col), rewriter.getI32IntegerAttr(row));

    return success();
  };
};

struct AIELowerSetLockPass
    : public xilinx::AIEX::impl::AIELowerSetLockBase<AIELowerSetLockPass> {
  void runOnOperation() override {

    DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalOp<NpuWrite32Op, cf::AssertOp>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<SetLockOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<SetLockToWrite32Pattern>(&getContext());

    SmallVector<Operation *> roots;
    device.walk([&](SetLockOp op) { roots.push_back(op); });
    if (failed(applyPartialConversion(roots, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIELowerSetLockPass() {
  return std::make_unique<AIELowerSetLockPass>();
}
