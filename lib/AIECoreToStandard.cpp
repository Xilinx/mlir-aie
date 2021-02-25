// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Translation.h"
#include "AIEDialect.h"
#include "AIENetlistAnalysis.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
//using namespace mlir::LLVM;

template <typename MyAIEOp>
struct AIEOpRemoval : public OpConversionPattern<MyAIEOp> {
  using OpConversionPattern<MyAIEOp>::OpConversionPattern;
  ModuleOp &module;

  AIEOpRemoval(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<MyAIEOp>(context, benefit),
    module(m) {}

  LogicalResult matchAndRewrite(MyAIEOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEDebugOpToStdLowering : public OpConversionPattern<DebugOp> {
  using OpConversionPattern<DebugOp>::OpConversionPattern;
  ModuleOp &module;

  AIEDebugOpToStdLowering(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<DebugOp>(context, benefit),
    module(m) {}

  LogicalResult matchAndRewrite(DebugOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    std::string funcName = "debug_i32";
    auto func = module.lookupSymbol<FuncOp>(funcName);
    assert(func && "Could not find the intrinsic function!");
    SmallVector<Value, 1> args;
    args.push_back(op.arg());
    rewriter.create<CallOp>(rewriter.getUnknownLoc(), func, args);
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEPutStreamToStdLowering : public OpConversionPattern<PutStreamOp> {
  using OpConversionPattern<PutStreamOp>::OpConversionPattern;
  ModuleOp &module;

  AIEPutStreamToStdLowering(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<PutStreamOp>(context, benefit),
    module(m) {}

  LogicalResult matchAndRewrite(PutStreamOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    std::string funcName = "llvm.aie.put.";
    if (op.isWideStream())
      funcName += "wms";
    else if (op.isFloatStream())
      funcName += "fms";
    else
      funcName += "ms";

    llvm::dbgs() << "FINDING: " << funcName << "\n";
    auto putMSFunc = module.lookupSymbol<FuncOp>(funcName);
    assert(putMSFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.channel());
    args.push_back(op.streamValue());
    rewriter.create<CallOp>(rewriter.getUnknownLoc(), putMSFunc, args);
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEGetStreamToStdLowering : public OpConversionPattern<GetStreamOp> {
  using OpConversionPattern<GetStreamOp>::OpConversionPattern;
  ModuleOp &module;

  AIEGetStreamToStdLowering(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<GetStreamOp>(context, benefit),
    module(m) {}

  LogicalResult matchAndRewrite(GetStreamOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
    std::string funcName = "llvm.aie.get.";
    if (op.isWideStream())
      funcName += "wss";
    else if (op.isFloatStream())
      funcName += "fss";
    else
      funcName += "ss";

    auto getSSFunc = module.lookupSymbol<FuncOp>(funcName);
    assert(getSSFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.channel());
    auto getSSCall = rewriter.create<CallOp>(rewriter.getUnknownLoc(), getSSFunc, args);
    rewriter.replaceOp(op, getSSCall.getResult(0));
    return success();
  }
};

struct AIEPutCascadeToStdLowering : public OpConversionPattern<PutCascadeOp> {
  using OpConversionPattern<PutCascadeOp>::OpConversionPattern;
  ModuleOp &module;

  AIEPutCascadeToStdLowering(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<PutCascadeOp>(context, benefit),
    module(m) {}

  LogicalResult matchAndRewrite(PutCascadeOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    std::string funcName = "llvm.aie.put.mcd";
    auto putMCDFunc = module.lookupSymbol<FuncOp>(funcName);
    assert(putMCDFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.cascadeValue());
    rewriter.create<CallOp>(rewriter.getUnknownLoc(), putMCDFunc, args);
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEGetCascadeToStdLowering : public OpConversionPattern<GetCascadeOp> {
  using OpConversionPattern<GetCascadeOp>::OpConversionPattern;
  ModuleOp &module;

  AIEGetCascadeToStdLowering(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<GetCascadeOp>(context, benefit),
    module(m) {}

  LogicalResult matchAndRewrite(GetCascadeOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
    std::string funcName = "llvm.aie.get.scd";
    auto getSCDFunc = module.lookupSymbol<FuncOp>(funcName);
    assert(getSCDFunc && "Could not find the intrinsic function!");
    auto getSCDCall = rewriter.create<CallOp>(rewriter.getUnknownLoc(), getSCDFunc, ValueRange({}));
    rewriter.replaceOp(op, getSCDCall.getResult(0));
    return success();
  }
};


struct AIECoreToStandardFunc : public OpConversionPattern<CoreOp> {
  using OpConversionPattern<CoreOp>::OpConversionPattern;
  ModuleOp &module;
  BlockAndValueMapping &mapper;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers;
  int tileCol = 0;
  int tileRow = 0;

  AIECoreToStandardFunc(MLIRContext *context, ModuleOp &m,
    BlockAndValueMapping &mapper,
    DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers,
    PatternBenefit benefit = 1,
    int tileCol = 1,
    int tileRow = 1
  ) : OpConversionPattern<CoreOp>(context, benefit),
    module(m), mapper(mapper), tileToBuffers(tileToBuffers),
    tileCol(tileCol), tileRow(tileRow) {}

  LogicalResult matchAndRewrite(CoreOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {

    Operation *Op = op.getOperation();
    int col = op.colIndex();
    int row = op.rowIndex();

    // Only pull code for the indicated function
    if((tileRow != row) || (tileCol != col)) {
      rewriter.eraseOp(Op);
      return success();
    }

    std::string coreName("core" + std::to_string(col) + std::to_string(row));
    auto coreFunc = rewriter.create<FuncOp>(rewriter.getUnknownLoc(), coreName,
                  FunctionType::get(rewriter.getContext(), {}, {}));

    rewriter.cloneRegionBefore(op.body(), coreFunc.getBody(), coreFunc.getBody().begin(), mapper);

    // Create a main function that just calls the core function above.
    auto mainFunc = rewriter.create<FuncOp>(rewriter.getUnknownLoc(), "_main",
                 FunctionType::get(rewriter.getContext(), {}, {}));
    rewriter.setInsertionPointToStart(mainFunc.addEntryBlock());
    SmallVector<Value, 8> args;
    rewriter.create<CallOp>(rewriter.getUnknownLoc(), coreFunc, args); // call with no args.
    rewriter.create<ReturnOp>(rewriter.getUnknownLoc(), args); // return nothing

    DenseMap<Operation *, Value> newAllocated;

    for (auto map : tileToBuffers) {
      Operation *tileOp = map.first;
      SmallVector<BufferOp, 4> buffers(map.second);
      TileOp tile = dyn_cast<TileOp>(tileOp);
      int dstCol = tile.colIndex();
      int dstRow = tile.rowIndex();

      if (!isLegalMemAffinity(col, row, dstCol, dstRow))
        continue;

      rewriter.setInsertionPoint(coreFunc);
      for (auto buffer : buffers) {
        auto symName = buffer.name().getValue();
        assert(t.getShape().size() == 1 && "Only supporting MemRefType of shape 1 for now!");
        rewriter.create<GlobalMemrefOp>(rewriter.getUnknownLoc(), symName, rewriter.getStringAttr("public"), TypeAttr::get(buffer.getType()), nullptr, false);
      }
      rewriter.setInsertionPointToStart(&coreFunc.getBody().front());
      for (auto buffer : buffers) {
        MemRefType t = buffer.getType().cast<MemRefType>();
        auto symName = buffer.name().getValue();
        assert(t.getShape().size() == 1 && "Only supporting MemRefType of shape 1 for now!");
        auto allocated = rewriter.create<GetGlobalMemrefOp>(rewriter.getUnknownLoc(), t, symName);
        newAllocated[buffer] = allocated.result();
        rewriter.replaceOp(buffer, allocated.result());
      }
    }

    coreFunc.getBody().walk([&](Operation *childOp) {
      rewriter.setInsertionPointAfter(childOp);

      if (EndOp end = dyn_cast<EndOp>(childOp)) {
        rewriter.create<ReturnOp>(rewriter.getUnknownLoc(), ValueRange({}));
        rewriter.eraseOp(childOp);
      }
      else if (UseLockOp useLock = dyn_cast<UseLockOp>(childOp)) {
        LockOp lock = dyn_cast<LockOp>(useLock.lock().getDefiningOp());
        TileOp tile = dyn_cast<TileOp>(lock.tile().getDefiningOp());
        int dstCol = tile.colIndex();
        int dstRow = tile.rowIndex();

        int cardinalMemOffset = 0;

        if (isMemSouth(col, row, dstCol, dstRow))
          cardinalMemOffset = 0;
        else if (isMemWest(col, row, dstCol, dstRow))
          cardinalMemOffset = 16;
        else if (isMemNorth(col, row, dstCol, dstRow))
          cardinalMemOffset = 32;
        else if (isMemEast(col, row, dstCol, dstRow))
          cardinalMemOffset = 48;
        else
          llvm_unreachable("Found illegal lock user!");

        int coreLockID = cardinalMemOffset + lock.getLockID();

        std::string funcName = "llvm.aie.lock.";
        if (useLock.acquire())
          funcName += "acquire.reg";
        else if (useLock.release())
          funcName += "release.reg";

        auto useLockFunc = module.lookupSymbol<FuncOp>(funcName);
        assert(useLockFunc && "Could not find the intrinsic function!");
        SmallVector<Value, 2> args;
        Value lockValue = rewriter.create<ConstantOp>(
          rewriter.getUnknownLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(useLock.getLockValue()));

        Value coreLockIDValue = rewriter.create<ConstantOp>(
          rewriter.getUnknownLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(coreLockID));

        args.push_back(coreLockIDValue);
        args.push_back(lockValue);

        rewriter.create<CallOp>(rewriter.getUnknownLoc(), useLockFunc, args);

        rewriter.eraseOp(childOp);
      }
    });

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIECoreToStandardPass : public AIECoreToStandardBase<AIECoreToStandardPass> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder(m.getBody()->getTerminator());

    // Extract all CoreOps
    // Create an LLVM func for each CoreOp
    // Clone the region body of each CoreOp to the newly created LLVM func

    DenseMap<std::pair<int, int>, Operation *> tiles;
    DenseMap<Operation *, CoreOp> cores;
    DenseMap<Operation *, MemOp> mems;
    DenseMap<std::pair<Operation *, int>, LockOp> locks;
    DenseMap<Operation *, SmallVector<BufferOp, 4>> tileToBuffers;
    DenseMap<Operation *, SwitchboxOp> switchboxes;

    NetlistAnalysis NL(m, tiles, cores, mems, locks, tileToBuffers, switchboxes);
    NL.collectTiles(tiles);
    NL.collectCores(cores);
    NL.collectBuffers(tileToBuffers);

    // Populate intrinsic functions
    // Intrinsic information: peano/llvm-project/llvm/lib/Target/AIE/AIEInstrInfo.td
    // Also take a look at the tests: peano/llvm-project/llvm/test/CodeGen/AIE
    builder.setInsertionPointToStart(m.getBody());

    SmallVector<Type, 2> callArgTypes;

    Type int32Type = IntegerType::get(builder.getContext(), 32);
    Type int128Type = IntegerType::get(builder.getContext(), 128);
    Type int384Type = IntegerType::get(builder.getContext(), 384);
    Type floatType = FloatType::getF32(builder.getContext());

    // llvm.func @debug_i32(%val: !llvm.i32) -> ()
    builder.create<FuncOp>(builder.getUnknownLoc(), "debug_i32",
        FunctionType::get(builder.getContext(), {int32Type}, {})).setPrivate();

    // llvm.func @llvm.aie.put.ms(%channel: !llvm.i1, %stream_val: !llvm.i32) -> ()
    builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.put.ms",
        FunctionType::get(builder.getContext(), {int32Type, int32Type}, {})).setPrivate();

    // llvm.func @llvm.aie.put.mws(%channel: !llvm.i1, %stream_val: !llvm.i128) -> ()
    builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.put.wms",
        FunctionType::get(builder.getContext(), {int32Type, int128Type}, {})).setPrivate();

    // llvm.func @llvm.aie.put.mfs(%channel: !llvm.i1, %stream_val: !llvm.float) -> ()
    builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.put.fms",
        FunctionType::get(builder.getContext(), {int32Type, floatType}, {})).setPrivate();

    // llvm.func @llvm.aie.get.ss(%channel: !llvm.i1) -> !llvm.i32
    builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.get.ss",
        FunctionType::get(builder.getContext(), {int32Type}, {int32Type})).setPrivate();

    // llvm.func @llvm.aie.get.wss(%channel: !llvm.i1) -> !llvm.i128
    builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.get.wss",
        FunctionType::get(builder.getContext(), {int32Type}, {int128Type})).setPrivate();

    // llvm.func @llvm.aie.get.fss(%channel: !llvm.i1) -> !llvm.float
    builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.get.fss",
        FunctionType::get(builder.getContext(), {int32Type}, {floatType})).setPrivate();

    // llvm.func @llvm.aie.put.scd(%scd_val: !llvm.i384) -> ()
    builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.put.mcd",
        FunctionType::get(builder.getContext(), {int384Type}, {})).setPrivate();

    // llvm.func @llvm.aie.get.scd() -> !llvm.i384
    builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.get.scd",
        FunctionType::get(builder.getContext(), {}, {int384Type})).setPrivate();

    // llvm.func @llvm.aie.lock.acquire.reg(%lock_id: !llvm.i32, %lock_val: !llvm.i32) ->()
    builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.lock.acquire.reg",
        FunctionType::get(builder.getContext(), {int32Type, int32Type}, {})).setPrivate();

    // llvm.func @llvm.aie.lock.release.reg(%lock_id: !llvm.i32, %lock_val: !llvm.i32) ->()
    builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.lock.release.reg",
        FunctionType::get(builder.getContext(), {int32Type, int32Type}, {})).setPrivate();


    BlockAndValueMapping mapper;
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<FuncOp, ModuleOp, mlir::ModuleTerminatorOp>();

    OwningRewritePatternList patterns;
    patterns.insert<AIEPutStreamToStdLowering,
                    AIEGetStreamToStdLowering,
                    AIEPutCascadeToStdLowering,
                    AIEGetCascadeToStdLowering,
                    AIEDebugOpToStdLowering
                    >(m.getContext(), m);

    patterns.insert<AIECoreToStandardFunc>(m.getContext(), m, mapper, tileToBuffers, 1,
       tileCol, tileRow);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
    patterns.insert<AIEOpRemoval<AIE::TileOp>,
                    AIEOpRemoval<AIE::FlowOp>,
                    AIEOpRemoval<AIE::MemOp>,
                    AIEOpRemoval<AIE::ShimMuxOp>,
                    AIEOpRemoval<AIE::SwitchboxOp>,
                    AIEOpRemoval<AIE::LockOp>,
                    AIEOpRemoval<AIE::BufferOp>
                   >(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIECoreToStandardPass() {
  return std::make_unique<AIECoreToStandardPass>();
}
