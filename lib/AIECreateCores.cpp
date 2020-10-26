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

struct LowerAIEMemcpy : public OpConversionPattern<MemcpyOp> {
  using OpConversionPattern<MemcpyOp>::OpConversionPattern;
  ModuleOp &module;
  DenseMap<Operation *, MemOp> &mems;
  DenseMap<Value, Value> &buffers;

  LowerAIEMemcpy(MLIRContext *context, ModuleOp &m,
    DenseMap<Operation *, MemOp> &mems,
    DenseMap<Value, Value> &buffers,
    PatternBenefit benefit = 1) :
    OpConversionPattern<MemcpyOp>(context, benefit),
      module(m),
      mems(mems),
      buffers(buffers) {}

  void createDMABlocksAndOps(MemOp &mem,
    StringRef tokenName, int acquireTknVal, int releaseTknVal,
    Value buf, int offset, int len,
    DMAChan dmaChannel,
    ConversionPatternRewriter &rewriter) const {

    Region &r = mem.body();
    Block &entryBlock = r.front();
    Block &endBlock = r.back();
    Block *dmaBlock = rewriter.createBlock(&endBlock);
    Block *bdBlock = rewriter.createBlock(&endBlock);

    Operation *termOp = entryBlock.getTerminator();
    TerminatorOp oldTerm = dyn_cast<TerminatorOp>(termOp);

    // Retrieve current successor dma Blocks, and add new successor Block
    SmallVector<Block *, 4> succBlocks(oldTerm.dests());
    succBlocks.push_back(dmaBlock);

    rewriter.setInsertionPoint(oldTerm);
    DMAStartOp dmaStart= rewriter.create<DMAStartOp>(rewriter.getUnknownLoc(), dmaChannel);

    // Replace old terminator with new terminator that includes new dma Block as a successor
    rewriter.setInsertionPointToEnd(&entryBlock);
    TerminatorOp newTerm = rewriter.create<TerminatorOp>(rewriter.getUnknownLoc(), succBlocks);
    rewriter.eraseOp(termOp);

    // Setup dma Block
    // It should contain a conditional branch op that points to a bd Block.
    // The bd Block is the start block description of the DMA transfer
    rewriter.setInsertionPointToStart(dmaBlock);
    rewriter.create<CondBranchOp>(rewriter.getUnknownLoc(), dmaStart, bdBlock, &endBlock);

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
    Value origSrcBuf = op.srcBuf();
    Value origDstBuf = op.dstBuf();

    StringRef tokenName = op.tokenName();
    int acquireTknVal = op.getAcquireTokenValue();
    int releaseTknVal = op.getReleaseTokenValue();
    int srcOffset = op.getSrcOffsetValue();
    int dstOffset = op.getDstOffsetValue();
    int srcLen = op.getSrcLenValue();
    int dstLen = op.getDstLenValue();

    MemOp srcMem = mems[op.srcTile().getDefiningOp()];
    MemOp dstMem = mems[op.dstTile().getDefiningOp()];

    Value srcBuf = buffers[origSrcBuf];
    Value dstBuf = buffers[origDstBuf];

    createDMABlocksAndOps(srcMem, tokenName, acquireTknVal, releaseTknVal,
                          srcBuf, srcOffset, srcLen, DMAChan::MM2S0, rewriter);
    createDMABlocksAndOps(dstMem, tokenName, acquireTknVal, releaseTknVal,
                          dstBuf, dstOffset, dstLen, DMAChan::S2MM0, rewriter);

    rewriter.eraseOp(Op);
    return success();
  }
};

struct RemoveAIEFuncs : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;
  ModuleOp &module;
  DenseMap<FuncOp, std::pair<int, int>> &funcs;

  RemoveAIEFuncs(MLIRContext *context, ModuleOp &m,
    DenseMap<FuncOp, std::pair<int, int>> &funcs,
    PatternBenefit benefit = 1) :
    OpConversionPattern<FuncOp>(context, benefit), module(m), funcs(funcs) {}

  LogicalResult matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    if (funcs.find(op) == funcs.end())
      return failure();

    rewriter.eraseOp(Op);
    return success();
  }
};

struct RemoveAIECalls : public OpConversionPattern<CallOp> {
  using OpConversionPattern<CallOp>::OpConversionPattern;
  ModuleOp &module;

  RemoveAIECalls(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1) :
    OpConversionPattern<CallOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(CallOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    if (!op.getAttr("aie.x") || !op.getAttr("aie.y"))
      return failure();

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIECreateCoresPass : public PassWrapper<AIECreateCoresPass,
  OperationPass<ModuleOp>> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {  
    registry.insert<StandardOpsDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder(m.getBody()->getTerminator());

    DenseMap<std::pair<int, int>, Operation *> tiles;
    DenseMap<Operation *, CoreOp> cores;
    DenseMap<Operation *, MemOp> mems;
    DenseMap<Value, Value> buffers;
    DenseMap<FuncOp, std::pair<int, int>> funcs;

    // Collect existing TileOps
    for (auto tile : m.getOps<TileOp>()) {
      int colIndex = tile.colIndex();
      int rowIndex = tile.rowIndex();
      tiles[std::make_pair(colIndex, rowIndex)] = tile;
    }

    // Bind FuncOp to an AIE core based on attributes of the CallOp
    // A CoreOp will be created for the core, and the FuncOp body is cloned
    // to the CoreOp region
    for (auto callOp : m.getOps<CallOp>()) {
      if (!callOp.getAttr("aie.x") || !callOp.getAttr("aie.y"))
        continue;

      SmallVector<Value, 4> callOperands(callOp.getArgOperands());
      SmallVector<std::pair<MemRefType, int>, 4> coreBufTypes;

      int colIndex = callOp.getAttrOfType<IntegerAttr>("aie.x").getInt();
      int rowIndex = callOp.getAttrOfType<IntegerAttr>("aie.y").getInt();

      // get or create TileOp
      if (!tiles[std::make_pair(colIndex, rowIndex)]) {
        builder.setInsertionPointToStart(m.getBody());
        TileOp tile = builder.create<TileOp>(builder.getUnknownLoc(), colIndex, rowIndex);
        tiles[std::make_pair(colIndex, rowIndex)] = tile;
      }
      Operation *tileOp = tiles[std::make_pair(colIndex, rowIndex)];
      TileOp tile = dyn_cast<TileOp>(tileOp);
      builder.setInsertionPointAfter((Value)tile);

      // create MemOp
      if (!mems[tileOp]) {
        for (unsigned i = 0; i < callOperands.size(); i++) {
          Value operand = callOperands[i];
          MemRefType t = nullptr;
          if (operand.getType().isa<MemRefType>()) {
            t = operand.getType().cast<MemRefType>();
          } else if (operand.getType().isIntOrFloat()) {
            // promote scalar type to memref type
            int64_t shape[1] = {1};
            t = MemRefType::get(shape, operand.getType());
          }

          assert(t && "Unsupported type!");
          coreBufTypes.push_back(std::make_pair(t, i));
          BufferOp buf = builder.create<BufferOp>(builder.getUnknownLoc(), t, tile);
//          buf.setAttr("sym_name", builder.getStringAttr("test_name"));
          buffers[callOperands[i]] = buf;
        }

        MemOp mem = builder.create<MemOp>(builder.getUnknownLoc(), builder.getIndexType(), tile);
        Region &r = mem.body();
        Block *entryBlock = builder.createBlock(&r);
        Block *endBlock = builder.createBlock(&r);

        builder.setInsertionPointToStart(entryBlock);
        builder.create<TerminatorOp>(builder.getUnknownLoc(), ArrayRef<Block *>(endBlock));
        // block terminator
        builder.setInsertionPointToStart(endBlock);
        builder.create<EndOp>(builder.getUnknownLoc());
        mems[tileOp] = mem;
      }

      // create CoreOp with buffer reference
      if (CallOpInterface call = dyn_cast<CallOpInterface>(callOp.getOperation())) {
        Operation *callable = call.resolveCallable();
        if (FuncOp func = dyn_cast<FuncOp>(callable)) {
          funcs[func] = std::make_pair(colIndex, rowIndex);

          BlockAndValueMapping mapper;

          builder.setInsertionPoint(callOp);

          CoreOp core;
          Block *currentBlock;
          
          if (!cores[tileOp]) {
            core = builder.create<CoreOp>(builder.getUnknownLoc(), builder.getIndexType(), tile);
            Region &r = core.body();
            currentBlock = builder.createBlock(&r);
            builder.setInsertionPointToStart(currentBlock);
          } else {
            core = cores[tileOp];
            currentBlock = &core.body().back();
            builder.setInsertionPoint(currentBlock->getTerminator());
          }

          // Mapping between function arguments (FuncOp) and AIE buffers (CoreOp)
          // We will create one buffer for each function argument
          // If the function argument's type is a scalar, we promote it to a one-element memref,
          // and do a load to the buffer at index 0
          for (auto pair : coreBufTypes) {
            MemRefType t = pair.first;
            int operandID = pair.second;
            Value arg = func.getArgument(operandID);
            Value buf = buffers[callOperands[operandID]];
            if (arg.getType().isIntOrFloat()) {
              assert(t.getShape().size() == 1 && "Expected MemRefType of shape 1");
              assert(t.getShape()[0] == 1 && "Expected MemRefType of single element");

              Value zero = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 0);
              auto loadOp = builder.create<LoadOp>(builder.getUnknownLoc(), arg.getType(), buf, zero);
              mapper.map(arg, loadOp);
            } else {
              mapper.map(arg, buf);
            }
          }

          // Clone ops from the original function to CoreOp's body
          for (auto &childOp : func.getCallableRegion()->getOps()) {
            // skip ReturnOp since it lives only within a funcOp
            if (auto returnOp = dyn_cast<ReturnOp>(childOp))
              continue;

            builder.clone(childOp, mapper);
          }
          if (!cores[tileOp]) {
            // block terminator
            builder.create<EndOp>(builder.getUnknownLoc());
            cores[tileOp] = core;
          }
        }
      }
    }



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
    target.addLegalOp<AIE::TerminatorOp>();
    target.addLegalOp<BranchOp>();
    target.addLegalOp<CondBranchOp>();

    patterns.insert<LowerAIEMemcpy>(m.getContext(), m, mems, buffers);

    // Remove standard CallOps and FuncOps that are bound to AIE CoreOps
    patterns.insert<RemoveAIECalls>(m.getContext(), m);
    patterns.insert<RemoveAIEFuncs>(m.getContext(), m, funcs);

    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};

void xilinx::AIE::registerAIECreateCoresPass() {
    PassRegistration<AIECreateCoresPass>(
      "aie-create-cores",
      "Create CoreOp, MemOp, and FlowOp of AIE dialect");
}
