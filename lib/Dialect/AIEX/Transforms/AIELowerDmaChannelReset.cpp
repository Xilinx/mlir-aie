//===- AIELowerDmaChannelReset.cpp ------------------------------*- C++ -*-===//
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
#define GEN_PASS_DEF_AIELOWERDMACHANNELRESET
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-lower-dma-channel-reset"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

// The S2MM/MM2S channel control register reset bit (bit 1). Matches
// XAIE2PGBL_MEMORY_MODULE_DMA_S2MM_0_CTRL_RESET_MASK.
static constexpr uint32_t kDmaCtrlResetMask = 0x2;

struct DmaChannelResetToWrite32Pattern
    : OpConversionPattern<DmaChannelResetOp> {
  using OpConversionPattern<DmaChannelResetOp>::OpConversionPattern;

  DmaChannelResetToWrite32Pattern(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(DmaChannelResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    const AIE::AIETargetModel &tm = dev.getTargetModel();

    int col = op.getColumn();
    int row = op.getRow();
    int channel = op.getChannel();
    AIE::DMAChannelDir dir = op.getDirection();

    // getDmaControlAddress returns the absolute address (col/row folded in);
    // NpuWrite32Op re-folds col/row, so pass the local offset + col/row, as
    // AIELowerSetLock does for the lock address.
    uint32_t ctrlAddrLocal =
        tm.getDmaControlAddress(col, row, channel, dir) & 0xFFFFF;

    Location loc = op.getLoc();
    // Reset pulse: assert the reset bit, then clear it. This is the sequence
    // validated on aie2p. NOTE: a full write32 clears other CTRL bits; a
    // maskwrite32 form that preserves them is a follow-up (see PR notes).
    rewriter.create<NpuWrite32Op>(
        loc, createConstantI32(rewriter, loc, ctrlAddrLocal),
        createConstantI32(rewriter, loc, kDmaCtrlResetMask), nullptr,
        rewriter.getI32IntegerAttr(col), rewriter.getI32IntegerAttr(row));
    rewriter.replaceOpWithNewOp<NpuWrite32Op>(
        op, createConstantI32(rewriter, loc, ctrlAddrLocal),
        createConstantI32(rewriter, loc, 0u), nullptr,
        rewriter.getI32IntegerAttr(col), rewriter.getI32IntegerAttr(row));

    return success();
  };
};

struct AIELowerDmaChannelResetPass
    : public xilinx::AIEX::impl::AIELowerDmaChannelResetBase<
          AIELowerDmaChannelResetPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();

    ConversionTarget target(getContext());
    target.addLegalOp<NpuWrite32Op>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<DmaChannelResetOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<DmaChannelResetToWrite32Pattern>(&getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIELowerDmaChannelResetPass() {
  return std::make_unique<AIELowerDmaChannelResetPass>();
}
