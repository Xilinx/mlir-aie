// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "AIEDialect.h"
#include "AIENetlistAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Translation.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct DMARemoval : public OpConversionPattern<DMAStartOp> {
  using OpConversionPattern<DMAStartOp>::OpConversionPattern;
  ModuleOp &module;
  SmallVector<std::pair<Operation *, Operation *>, 4> &dmaPairRemoval;

  DMARemoval(
      MLIRContext *context, ModuleOp &m,
      SmallVector<std::pair<Operation *, Operation *>, 4> &dmaPairRemoval,
      PatternBenefit benefit = 1)
      : OpConversionPattern<DMAStartOp>(context, benefit), module(m),
        dmaPairRemoval(dmaPairRemoval) {}

  LogicalResult
  matchAndRewrite(DMAStartOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    bool foundDMA = false;

    for (auto pair : dmaPairRemoval) {
      Operation *srcDmaOp = pair.first;
      Operation *dstDmaOp = pair.second;

      if ((srcDmaOp == Op) || (dstDmaOp == Op)) {
        foundDMA = true;
        break;
      }
    }

    if (!foundDMA)
      return failure();

    MemOp mem = dyn_cast<MemOp>(Op->getParentOp());
    Region &r = mem.body();

    SmallVector<Block *, 4> blocks;
    Block *entryBlock = &r.front();
    Block *endBlock = &r.back();
    Operation *oldTermOp = entryBlock->getTerminator();
    TerminatorOp oldTerm = dyn_cast<TerminatorOp>(oldTermOp);
    SmallVector<Block *, 4> succBlocks(oldTerm.dests());

    for (auto condBr : r.getOps<CondBranchOp>()) {
      Operation *dmaOp = condBr.getCondition().getDefiningOp();
      if (dmaOp != Op)
        continue;

      Operation *condBrOp = condBr.getOperation();
      Block *dmaBlock = condBrOp->getBlock();
      Block *firstBd = condBr.getTrueDest();
      blocks.push_back(dmaBlock);
      Block *curBd = firstBd;
      while (curBd != endBlock) {
        blocks.push_back(curBd);
        curBd = curBd->getSuccessors()[0];
      }

      auto it = std::find(succBlocks.begin(), succBlocks.end(), dmaBlock);
      if (it != succBlocks.end())
        succBlocks.erase(it);
    }

    rewriter.setInsertionPointToEnd(entryBlock);
    TerminatorOp newTerm =
        rewriter.create<TerminatorOp>(rewriter.getUnknownLoc(), succBlocks);

    for (auto block : blocks)
      rewriter.eraseBlock(block);

    rewriter.eraseOp(oldTermOp);
    rewriter.eraseOp(Op);

    return success();
  }
};

template <typename MyOp>
struct BufferUseRemoval : public OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  ModuleOp &module;
  SmallVector<Operation *, 4> &oldUseOps;

  BufferUseRemoval(MLIRContext *context, ModuleOp &m,
                   SmallVector<Operation *, 4> &oldUseOps,
                   PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit), module(m),
        oldUseOps(oldUseOps) {}

  LogicalResult
  matchAndRewrite(MyOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    if (std::find(oldUseOps.begin(), oldUseOps.end(), Op) == oldUseOps.end())
      return failure();

    rewriter.eraseOp(Op);
    return success();
  }
};

llvm::SmallSet<Operation *, 4>
getCommonTiles(ArrayRef<Operation *> Ops,
               DenseMap<std::pair<int, int>, Operation *> tiles) {

  llvm::SmallSet<Operation *, 4> commonTiles;

  bool IsFirst = true;
  for (auto Op : Ops) {
    if (isa<MemOp>(Op))
      continue;

    llvm::SmallSet<Operation *, 4> workingSet;
    CoreOp core = dyn_cast<CoreOp>(Op);
    int col = core.colIndex();
    int row = core.rowIndex();

    auto setPtr = &workingSet;
    if (IsFirst)
      setPtr = &commonTiles;

    if (tiles.count(std::make_pair(col, row)) == 1)
      setPtr->insert(tiles[std::make_pair(col, row)]);
    if (tiles.count(std::make_pair(col - 1, row)) == 1)
      setPtr->insert(tiles[std::make_pair(col - 1, row)]);
    if (tiles.count(std::make_pair(col, row - 1)) == 1)
      setPtr->insert(tiles[std::make_pair(col, row - 1)]);
    if (tiles.count(std::make_pair(col + 1, row)) == 1)
      setPtr->insert(tiles[std::make_pair(col + 1, row)]);
    if (tiles.count(std::make_pair(col, row + 1)) == 1)
      setPtr->insert(tiles[std::make_pair(col, row + 1)]);

    if (!IsFirst)
      set_intersect(commonTiles, workingSet);

    IsFirst = false;
  }

  return commonTiles;
}

struct AIEBufferMergePass
    : public PassWrapper<AIEBufferMergePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder(m.getBody()->getTerminator());

    DenseMap<std::pair<int, int>, Operation *> tiles;
    DenseMap<Operation *, CoreOp> cores;
    DenseMap<Operation *, MemOp> mems;
    DenseMap<std::pair<Operation *, int>, LockOp> locks;
    DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;
    DenseMap<Operation *, SwitchboxOp> switchboxes;

    NetlistAnalysis NLA(m, tiles, cores, mems, locks, buffers, switchboxes);
    NLA.runAnalysis();
    NLA.dmaAnalysis();
    NLA.lockAnalysis();

    NLA.collectBufferUsage();

    DenseMap<Operation *, SmallVector<Operation *, 4>> bufferUsers(
        NLA.getBufferUsers());
    DenseMap<Operation *, SmallVector<Operation *, 4>> dma2BufMap(
        NLA.getDMA2BufMap());
    DenseMap<std::pair<Operation *, int>, Operation *> dmas(NLA.getDMAs());
    DenseMap<Operation *, SmallVector<Operation *, 4>> dmaConnections(
        NLA.getDMAConnections());
    DenseMap<std::pair<Operation *, Operation *>, SmallVector<Operation *, 4>>
        buf2NewTilesMap;

    DenseMap<Operation *, Operation *> lockPairs(NLA.getLockPairs());
    SmallVector<std::pair<Operation *, Operation *>, 4> lockChains(
        NLA.getLockChains());
    DenseMap<Operation *, SmallVector<Operation *, 4>> bufAcqLocks(
        NLA.getBufAcqLocks());
    DenseMap<Operation *, SmallVector<Operation *, 4>> dma2ConnectsMap(
        NLA.getDma2ConnectsMap());
    SmallVector<std::pair<Operation *, Operation *>, 4> dmaConnectOps;
    SmallVector<std::pair<Operation *, Operation *>, 4> dmaPairRemoval;

    for (auto map : dmaConnections) {
      Operation *srcDma = map.first;
      SmallVector<Operation *, 4> srcBufs(dma2BufMap[srcDma]);
      SmallVector<Operation *, 4> dstDmas(map.second);
      SmallVector<Operation *, 4> srcConnectOps(dma2ConnectsMap[srcDma]);

      for (auto dstDma : dstDmas) {
        SmallVector<Operation *, 4> dstBufs(dma2BufMap[dstDma]);
        SmallVector<Operation *, 4> dstConnectOps(dma2ConnectsMap[dstDma]);

        // only deal with single consumer for now
        if (dstBufs.size() > 1)
          continue;

        bool CanShareBuf = false;
        for (auto srcBuf : srcBufs) {
          for (auto dstBuf : dstBufs) {
            SmallVector<Operation *, 4> srcUsers(bufferUsers[srcBuf]);
            SmallVector<Operation *, 4> dstUsers(bufferUsers[dstBuf]);

            llvm::SmallSet<Operation *, 4> srcCommonTiles(
                getCommonTiles(srcUsers, tiles));
            llvm::SmallSet<Operation *, 4> dstCommonTiles(
                getCommonTiles(dstUsers, tiles));

            set_intersect(srcCommonTiles, dstCommonTiles);

            if (srcCommonTiles.size() == 0)
              continue;

            for (auto commonTile : srcCommonTiles)
              buf2NewTilesMap[std::make_pair(srcBuf, dstBuf)].push_back(
                  commonTile);

            CanShareBuf = true;
          }
        }

        if (CanShareBuf) {
          for (auto srcConnectOp : srcConnectOps)
            for (auto dstConnectOp : dstConnectOps)
              dmaConnectOps.push_back(
                  std::make_pair(srcConnectOp, dstConnectOp));

          dmaPairRemoval.push_back(std::make_pair(srcDma, dstDma));
        }
      }
    }

    BlockAndValueMapping mapper;
    SmallVector<Operation *, 4> oldUseOps;

    for (auto map : buf2NewTilesMap) {
      Operation *srcBufOp = map.first.first;
      Operation *dstBufOp = map.first.second;
      Operation *newTileOp = map.second.front();

      BufferOp srcBuf = dyn_cast<BufferOp>(srcBufOp);
      BufferOp dstBuf = dyn_cast<BufferOp>(dstBufOp);
      TileOp newTile = dyn_cast<TileOp>(newTileOp);
      MemRefType t = srcBuf.getType().cast<MemRefType>();
      builder.setInsertionPoint(srcBuf);
      BufferOp newBuf =
          builder.create<BufferOp>(builder.getUnknownLoc(), t, newTile);
      mapper.map(srcBuf, newBuf);
      mapper.map(dstBuf, newBuf);

      SmallVector<Operation *, 4> users(bufferUsers[srcBufOp]);
      users.insert(users.end(), bufferUsers[dstBufOp].begin(),
                   bufferUsers[dstBufOp].end());

      int newLockID = NLA.getAvailableLockID(newTileOp);
      assert(newLockID >= 0 && "Could not get a new lock!");

      // create a new common lock at newTile
      builder.setInsertionPointAfter(newTileOp);
      LockOp newLock =
          builder.create<LockOp>(builder.getUnknownLoc(), newTile, newLockID);

      SmallVector<Operation *, 4> acqLockOps(bufAcqLocks[srcBufOp]);
      acqLockOps.insert(acqLockOps.end(), bufAcqLocks[dstBufOp].begin(),
                        bufAcqLocks[dstBufOp].end());

      for (auto Op : users) {
        Region *r = nullptr;
        bool IsCoreOp = false;

        if (CoreOp core = dyn_cast<CoreOp>(Op)) {
          r = &core.body();
          IsCoreOp = true;
        } else if (MemOp mem = dyn_cast<MemOp>(Op)) {
          r = &mem.body();
        }

        assert(r && "Expected MemOp or CoreOp!");

        r->walk([&](Operation *childOp) {
          Operation *bufOp = nullptr;
          for (Value operand : childOp->getOperands()) {
            Operation *operandOp = operand.getDefiningOp();
            if (operandOp == srcBufOp)
              bufOp = srcBufOp;
            if (operandOp == dstBufOp)
              bufOp = dstBufOp;
          }

          if (bufOp) {
            if (IsCoreOp) {
              builder.setInsertionPointAfter(childOp);
              builder.clone(*childOp, mapper);
            }
            oldUseOps.push_back(childOp);
          }

          if (std::find(acqLockOps.begin(), acqLockOps.end(), childOp) !=
              acqLockOps.end()) {
            UseLockOp oldAcqLock = dyn_cast<UseLockOp>(childOp);
            int acqLockValue = oldAcqLock.getLockValue();
            int acqTimeout = oldAcqLock.getTimeout();
            builder.setInsertionPointAfter(childOp);
            builder.create<UseLockOp>(builder.getUnknownLoc(), newLock,
                                      acqLockValue, LockAction::Acquire,
                                      acqTimeout);

            Operation *oldRelLockOp = lockPairs[childOp];
            UseLockOp oldRelLock = dyn_cast<UseLockOp>(oldRelLockOp);
            int relLockValue = oldRelLock.getLockValue();
            int relTimeout = oldRelLock.getTimeout();
            builder.setInsertionPointAfter(oldRelLockOp);
            builder.create<UseLockOp>(builder.getUnknownLoc(), newLock,
                                      relLockValue, LockAction::Release,
                                      relTimeout);

            oldUseOps.push_back(childOp);
            oldUseOps.push_back(oldRelLockOp);
          }
        });
      }
    }

    for (auto pair : dmaConnectOps) {
      Operation *srcConnectOp = pair.first;
      Operation *dstConnectOp = pair.second;
      ConnectOp srcConnect = dyn_cast<ConnectOp>(srcConnectOp);
      ConnectOp dstConnect = dyn_cast<ConnectOp>(dstConnectOp);
      ArrayRef<Operation *> dmaRoutes(
          NLA.findRoutes(srcConnectOp, dstConnectOp));
      for (auto Op : dmaRoutes) {
        ConnectOp op = dyn_cast<ConnectOp>(Op);
        oldUseOps.push_back(Op);
      }
    }

    ConversionTarget target(getContext());
    target.addLegalOp<AIE::TerminatorOp>();
    target.addLegalOp<AIE::MemOp>();

    OwningRewritePatternList patterns;
    patterns.insert<BufferUseRemoval<StoreOp>, BufferUseRemoval<LoadOp>,
                    BufferUseRemoval<UseLockOp>, BufferUseRemoval<ConnectOp>>(
        m.getContext(), m, oldUseOps);
    patterns.insert<DMARemoval>(m.getContext(), m, dmaPairRemoval);

    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};

void xilinx::AIE::registerAIEBufferMergePass() {
  PassRegistration<AIEBufferMergePass>(
      "aie-merge-buffers", "Merge distant buffers to maximize sharing "
                           "opportunities and reduce DMA overhead");
}
