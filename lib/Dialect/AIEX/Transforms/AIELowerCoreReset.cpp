//===- AIELowerCoreReset.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/RegisterField.h"

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

// CORE_CONTROL register, tile-local offset. Uniform across the AIE2 family:
// XAIE2PGBL_CORE_MODULE_CORE_CONTROL (aie2p) and
// XAIEMLGBL_CORE_MODULE_CORE_CONTROL (aie2) are both 0x00032000. The op only
// lowers on core tiles (the verifier rejects mem/shim), which exist on aie2 and
// aie2p; the npu runtime sequence path is AIE2-only.
static constexpr uint32_t kCoreCtrlAddr = 0x00032000;

// CORE_CONTROL.RESET field: bit 1, matching
// XAIE2PGBL_CORE_MODULE_CORE_CONTROL_RESET_{LSB,MASK} (aie2p reginit:
// Aie2PCoreCtrlReg.CtrlRst in xaie2pgbl_reginit.c). Bit 0 of the same word is
// ENABLE (CtrlEn); going through RegField + createMaskWriteField instead of a
// hand-coded write32 is what keeps this pulse from clobbering it.
static constexpr RegField kCoreCtrlResetField = {
    /*name=*/"CORE_CONTROL.RESET", /*regOff=*/kCoreCtrlAddr, /*mask=*/0x2};

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

    Location loc = op.getLoc();
    IntegerAttr colAttr = rewriter.getI32IntegerAttr(col);
    IntegerAttr rowAttr = rewriter.getI32IntegerAttr(row);

    // NpuMaskWrite32Op re-folds col/row from the attributes, so the address
    // operand is the tile-local CORE_CONTROL offset, mirroring how
    // AIELowerSetLock passes the local lock address with col/row.
    //
    // Reset pulse: assert the reset bit, then clear it. createMaskWriteField
    // derives the shift and mask from kCoreCtrlResetField and always emits
    // npu.maskwrite32, so the pulse preserves the ENABLE field packed in the
    // same CORE_CONTROL word instead of clobbering it. This mirrors aie-rt's
    // XAie_CoreReset/XAie_CoreUnreset, which drive the reset bit with a
    // MaskWrite32.
    FailureOr<NpuMaskWrite32Op> assertWrite = createMaskWriteField(
        rewriter, loc, op, kCoreCtrlResetField, /*value=*/1, colAttr, rowAttr);
    if (failed(assertWrite))
      return failure();

    FailureOr<NpuMaskWrite32Op> clearWrite = createMaskWriteField(
        rewriter, loc, op, kCoreCtrlResetField, /*value=*/0, colAttr, rowAttr);
    if (failed(clearWrite))
      return failure();

    rewriter.replaceOp(op, *clearWrite);
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
