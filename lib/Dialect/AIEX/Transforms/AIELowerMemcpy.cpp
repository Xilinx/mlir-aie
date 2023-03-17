//===- AIELowerMemcpy.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIETokenAnalysis.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

static TileOp srcTileOp(xilinx::AIEX::MemcpyOp op) {
  return llvm::dyn_cast<xilinx::AIE::TileOp>(op.getSrcTile().getDefiningOp());
}
static TileOp dstTileOp(xilinx::AIEX::MemcpyOp op) {
  return llvm::dyn_cast<xilinx::AIE::TileOp>(op.getDstTile().getDefiningOp());
}

struct LowerAIEMemcpy : public OpConversionPattern<MemcpyOp> {
  using OpConversionPattern<MemcpyOp>::OpConversionPattern;

  LowerAIEMemcpy(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<MemcpyOp>(context, benefit) {}

  void createDMABlocksAndOps(MemOp &mem, StringRef tokenName, int acquireTknVal,
                             int releaseTknVal, Value buf, int offset, int len,
                             DMAChannelDir dmaDir, int channelIndex,
                             ConversionPatternRewriter &rewriter) const {

    Region &r = mem.getBody();
    Block &endBlock = r.back();
    Block *dmaBlock = rewriter.createBlock(&endBlock);
    Block *bdBlock = rewriter.createBlock(&endBlock);

    rewriter.setInsertionPointToStart(dmaBlock);
    rewriter.create<DMAStartOp>(rewriter.getUnknownLoc(), dmaDir, channelIndex,
                                bdBlock, &endBlock);

    // Setup bd Block
    // It should contain locking operations (lock or token) as well as DMABD op
    // for specifying DMA Block description (which buffer type (A/B), transfer
    // length/address, etc.)
    rewriter.setInsertionPointToStart(bdBlock);
    rewriter.create<UseTokenOp>(rewriter.getUnknownLoc(), tokenName,
                                acquireTknVal, LockAction::Acquire);
    rewriter.create<DMABDOp>(rewriter.getUnknownLoc(), buf, offset, len,
                             0); // A type for now
    rewriter.create<UseTokenOp>(rewriter.getUnknownLoc(), tokenName,
                                releaseTknVal, LockAction::Release);
    rewriter.create<NextBDOp>(rewriter.getUnknownLoc(), &endBlock);
  }

  LogicalResult
  matchAndRewrite(MemcpyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    Value srcBuf = op.getSrcBuf();
    Value dstBuf = op.getDstBuf();

    StringRef tokenName = op.getTokenName();
    int acquireTknVal = op.getAcquireTokenValue();
    int releaseTknVal = op.getReleaseTokenValue();
    int srcOffset = op.getSrcOffsetValue();
    int dstOffset = op.getDstOffsetValue();
    int srcLen = op.getSrcLenValue();
    int dstLen = op.getDstLenValue();

    MemOp srcMem = srcTileOp(op).getMemOp();
    MemOp dstMem = dstTileOp(op).getMemOp();

    createDMABlocksAndOps(srcMem, tokenName, acquireTknVal, releaseTknVal,
                          srcBuf, srcOffset, srcLen, DMAChannelDir::MM2S, 0,
                          rewriter);
    createDMABlocksAndOps(dstMem, tokenName, acquireTknVal, releaseTknVal,
                          dstBuf, dstOffset, dstLen, DMAChannelDir::S2MM, 0,
                          rewriter);

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIELowerMemcpyPass : public AIELowerMemcpyBase<AIELowerMemcpyPass> {
  void runOnOperation() override {

    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

    // Setup FlowOps
    // Since memcpy moves data from one memory module to another, we use
    // WireBundle::DMA for both the source and the destination In addition, we
    // only have two DMA ports per each direction (MM2S/S2MM), and in a
    // circuit-switch mode, dest port/channel sharing is not possible.
    // Therefore, we will generate error if the number of logical flows
    // (streams) targeting the same destination (S2MM) is more than 2
    DenseMap<Value, int> destChannel;
    for (auto op : device.getOps<MemcpyOp>()) {
      builder.setInsertionPoint(op);
      TileOp srcTile = dyn_cast<TileOp>(op.getSrcTile().getDefiningOp());
      TileOp dstTile = dyn_cast<TileOp>(op.getDstTile().getDefiningOp());
      // TODO: perhaps a better approach is to not assert here, but rather have
      // a subsequent pass that legally relocates the ports
      assert(destChannel[op.getDstTile()] <= 2 &&
             "Could not allocate more than two dest. channel when creating "
             "FlowOp");
      builder.create<FlowOp>(builder.getUnknownLoc(), srcTile, WireBundle::DMA,
                             0, dstTile, WireBundle::DMA,
                             destChannel[op.getDstTile()]);
      destChannel[op.getDstTile()]++;
    }

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    target.addLegalOp<DMAStartOp>();
    target.addLegalOp<DMABDOp>();
    target.addLegalOp<UseTokenOp>();
    target.addLegalOp<NextBDOp>();

    patterns.insert<LowerAIEMemcpy>(&getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIELowerMemcpyPass() {
  return std::make_unique<AIELowerMemcpyPass>();
}
