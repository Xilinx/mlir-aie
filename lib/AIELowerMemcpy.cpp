// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Translation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "AIEDialect.h"
#include "AIETokenAnalysis.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

static TileOp srcTileOp(xilinx::AIE::MemcpyOp op) { return llvm::dyn_cast<xilinx::AIE::TileOp>(op.srcTile().getDefiningOp()); }
static TileOp dstTileOp(xilinx::AIE::MemcpyOp op) { return llvm::dyn_cast<xilinx::AIE::TileOp>(op.dstTile().getDefiningOp()); }

struct LowerAIEMemcpy : public OpConversionPattern<MemcpyOp> {
  using OpConversionPattern<MemcpyOp>::OpConversionPattern;
  ModuleOp &module;

  LowerAIEMemcpy(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1) :
    OpConversionPattern<MemcpyOp>(context, benefit),
      module(m) {}

  void createDMABlocksAndOps(MemOp &mem,
    StringRef tokenName, int acquireTknVal, int releaseTknVal,
    Value buf, int offset, int len,
    DMAChan dmaChannel,
    ConversionPatternRewriter &rewriter) const {

    Region &r = mem.body();
    Block &endBlock = r.back();
    Block *dmaBlock = rewriter.createBlock(&endBlock);
    Block *bdBlock = rewriter.createBlock(&endBlock);

    rewriter.setInsertionPointToStart(dmaBlock);
    rewriter.create<DMAStartOp>(rewriter.getUnknownLoc(), dmaChannel, bdBlock, &endBlock);

    // Setup bd Block
    // It should contain locking operations (lock or token) as well as DMABD op for specifying
    // DMA Block description (which buffer type (A/B), transfer length/address, etc.)
    rewriter.setInsertionPointToStart(bdBlock);
    rewriter.create<UseTokenOp>(rewriter.getUnknownLoc(), tokenName, acquireTknVal, LockAction::Acquire);
    rewriter.create<DMABDOp>(rewriter.getUnknownLoc(), buf, offset, len, 0); // A type for now
    rewriter.create<UseTokenOp>(rewriter.getUnknownLoc(), tokenName, releaseTknVal, LockAction::Release);
    rewriter.create<BranchOp>(rewriter.getUnknownLoc(), &endBlock);
  }

  LogicalResult matchAndRewrite(MemcpyOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    Value srcBuf = op.srcBuf();
    Value dstBuf = op.dstBuf();

    StringRef tokenName = op.tokenName();
    int acquireTknVal = op.getAcquireTokenValue();
    int releaseTknVal = op.getReleaseTokenValue();
    int srcOffset = op.getSrcOffsetValue();
    int dstOffset = op.getDstOffsetValue();
    int srcLen = op.getSrcLenValue();
    int dstLen = op.getDstLenValue();

    MemOp srcMem = srcTileOp(op).getMemOp();
    MemOp dstMem = dstTileOp(op).getMemOp();

    createDMABlocksAndOps(srcMem, tokenName, acquireTknVal, releaseTknVal,
                          srcBuf, srcOffset, srcLen, DMAChan::MM2S0, rewriter);
    createDMABlocksAndOps(dstMem, tokenName, acquireTknVal, releaseTknVal,
                          dstBuf, dstOffset, dstLen, DMAChan::S2MM0, rewriter);

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIELowerMemcpyPass : public PassWrapper<AIELowerMemcpyPass,
  OperationPass<ModuleOp>> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
  }
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder(m.getBody()->getTerminator());

    // Setup FlowOps
    // Since memcpy moves data from one memory module to another, we use WireBundle::DMA
    // for both the source and the destination
    // In addition, we only have two DMA ports per each direction (MM2S/S2MM), and in a
    // circuit-switch mode, dest port/channel sharing is not possible. Therefore, we will generate error
    // if the number of logical flows (streams) targeting the same destination (S2MM) is more than 2
    DenseMap<Value, int> destChannel;
    for (auto op : m.getOps<MemcpyOp>()) {
      builder.setInsertionPoint(op);
      TileOp srcTile = dyn_cast<TileOp>(op.srcTile().getDefiningOp());
      TileOp dstTile = dyn_cast<TileOp>(op.dstTile().getDefiningOp());
      // TODO: perhaps a better approach is to not assert here, but rather have a subsequent pass
      // that legally relocates the ports
      assert(destChannel[op.dstTile()] <= 2 &&
             "Could not allocate more than two dest. channel when creating FlowOp");
      // WireBundle[1] = DMA
      builder.create<FlowOp>(builder.getUnknownLoc(), srcTile, 1, 0, dstTile, 1, destChannel[op.dstTile()]);
      destChannel[op.dstTile()]++;
    }

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;
    target.addLegalOp<DMAStartOp>();
    target.addLegalOp<DMABDOp>();
    target.addLegalOp<UseTokenOp>();
    target.addLegalOp<BranchOp>();
    target.addLegalOp<CondBranchOp>();

    patterns.insert<LowerAIEMemcpy>(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

void xilinx::AIE::registerAIELowerMemcpyPass() {
    PassRegistration<AIELowerMemcpyPass>(
      "aie-lower-memcpy",
      "Lower AIE.Memcpy operations to Flows and DMA programs");
}
