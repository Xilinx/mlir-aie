//===- AIELowerCoreReset.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIELOWERCORERESET
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-lower-core-reset"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

struct CoreResetToMaskWrite32Pattern : OpConversionPattern<CoreResetOp> {
  using OpConversionPattern<CoreResetOp>::OpConversionPattern;

  CoreResetToMaskWrite32Pattern(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(CoreResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIE::TileOp tile = op.getTileOp();
    int col = tile.getCol();
    int row = tile.getRow();

    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    const AIE::AIETargetModel &tm = dev.getTargetModel();

    // Resolve CORE_CONTROL and its reset bit from the register database rather
    // than restating the layout here. Core tiles only; the verifier rejects
    // mem/shim, so this is always the "core" module.
    const AIE::RegisterInfo *ctrlReg =
        tm.lookupRegister("Core_Control", tile.getTileID());
    if (!ctrlReg)
      return op.emitOpError("target has no Core_Control register in its "
                            "register database");
    const AIE::BitFieldInfo *resetField = ctrlReg->getField("Reset");
    if (!resetField)
      return op.emitOpError("Core_Control has no Reset field in the register "
                            "database");
    std::optional<uint32_t> resetMask = tm.getFieldMask(*resetField);
    if (!resetMask)
      return op.emitOpError("Core_Control Reset field does not fit in a 32-bit "
                            "register");

    Location loc = op.getLoc();
    IntegerAttr colAttr = rewriter.getI32IntegerAttr(col);
    IntegerAttr rowAttr = rewriter.getI32IntegerAttr(row);

    // NpuMaskWrite32Op re-folds col/row from the attributes, so the address
    // operand is the tile-local offset, mirroring how AIELowerSetLock passes
    // the local lock address with col/row.
    //
    // Reset pulse: assert the reset bit, then clear it. Both writes mask to the
    // reset bit only, so the pulse preserves the Enable field packed in the
    // same word instead of clobbering it. This mirrors aie-rt's
    // XAie_CoreReset/XAie_CoreUnreset, which drive the reset bit with a
    // MaskWrite32. Constants are materialized in named locals so the emitted IR
    // order does not depend on unspecified C++ argument-evaluation order.
    Value assertAddr = createConstantI32(rewriter, loc, ctrlReg->offset);
    Value assertVal =
        createConstantI32(rewriter, loc, tm.encodeFieldValue(*resetField, 1));
    Value assertMask = createConstantI32(rewriter, loc, *resetMask);
    rewriter.create<NpuMaskWrite32Op>(loc, assertAddr, assertVal, assertMask,
                                      nullptr, colAttr, rowAttr);

    Value clearAddr = createConstantI32(rewriter, loc, ctrlReg->offset);
    Value clearVal = createConstantI32(rewriter, loc, 0u);
    Value clearMask = createConstantI32(rewriter, loc, *resetMask);
    rewriter.replaceOpWithNewOp<NpuMaskWrite32Op>(
        op, clearAddr, clearVal, clearMask, nullptr, colAttr, rowAttr);

    return success();
  };
};

struct AIELowerCoreResetPass
    : public xilinx::AIEX::impl::AIELowerCoreResetBase<AIELowerCoreResetPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalOp<NpuMaskWrite32Op>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<CoreResetOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<CoreResetToMaskWrite32Pattern>(&getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIELowerCoreResetPass() {
  return std::make_unique<AIELowerCoreResetPass>();
}
