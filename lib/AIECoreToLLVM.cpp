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
using namespace mlir::LLVM;

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

struct AIEPutStreamLowering : public OpConversionPattern<PutStreamOp> {
  using OpConversionPattern<PutStreamOp>::OpConversionPattern;
  ModuleOp &module;
  LLVMTypeConverter &converter;

  AIEPutStreamLowering(MLIRContext *context, ModuleOp &m, LLVMTypeConverter &converter,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<PutStreamOp>(context, benefit),
    module(m), converter(converter) {}

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

    auto putMSFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    assert(putMSFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.streamValue());
    args.push_back(op.channel());
    auto putMSCall = rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), putMSFunc, args);
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEGetStreamLowering : public OpConversionPattern<GetStreamOp> {
  using OpConversionPattern<GetStreamOp>::OpConversionPattern;
  ModuleOp &module;
  LLVMTypeConverter &converter;

  AIEGetStreamLowering(MLIRContext *context, ModuleOp &m, LLVMTypeConverter &converter,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<GetStreamOp>(context, benefit),
    module(m), converter(converter) {}

  LogicalResult matchAndRewrite(GetStreamOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    std::string funcName = "llvm.aie.get.";
    if (op.isWideStream())
      funcName += "wss";
    else if (op.isFloatStream())
      funcName += "fss";
    else
      funcName += "ss";

    auto getSSFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    assert(getSSFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.channel());
    auto getSSCall = rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), getSSFunc, args);
    rewriter.replaceOp(op, getSSCall.getResult(0));
    return success();
  }
};

struct AIEPutCascadeLowering : public OpConversionPattern<PutCascadeOp> {
  using OpConversionPattern<PutCascadeOp>::OpConversionPattern;
  ModuleOp &module;
  LLVMTypeConverter &converter;

  AIEPutCascadeLowering(MLIRContext *context, ModuleOp &m, LLVMTypeConverter &converter,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<PutCascadeOp>(context, benefit),
    module(m), converter(converter) {}

  LogicalResult matchAndRewrite(PutCascadeOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    std::string funcName = "llvm.aie.put.mcd";
    auto putMCDFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    assert(putMCDFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.cascadeValue());
    auto putMCDCall = rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), putMCDFunc, args);
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEGetCascadeLowering : public OpConversionPattern<GetCascadeOp> {
  using OpConversionPattern<GetCascadeOp>::OpConversionPattern;
  ModuleOp &module;
  LLVMTypeConverter &converter;

  AIEGetCascadeLowering(MLIRContext *context, ModuleOp &m, LLVMTypeConverter &converter,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<GetCascadeOp>(context, benefit),
    module(m), converter(converter) {}

  LogicalResult matchAndRewrite(GetCascadeOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    std::string funcName = "llvm.aie.get.scd";
    auto getSCDFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    assert(getSCDFunc && "Could not find the intrinsic function!");
    auto getSCDCall = rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), getSCDFunc, ValueRange({}));
    rewriter.replaceOp(op, getSCDCall.getResult(0));
    return success();
  }
};

struct AIECoreToLLVMFunc : public OpConversionPattern<CoreOp> {
  using OpConversionPattern<CoreOp>::OpConversionPattern;
  ModuleOp &module;
  BlockAndValueMapping &mapper;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers;
  LLVMTypeConverter &converter;

  AIECoreToLLVMFunc(MLIRContext *context, ModuleOp &m,
    BlockAndValueMapping &mapper,
    DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers,
    LLVMTypeConverter &converter,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<CoreOp>(context, benefit),
    module(m), mapper(mapper), buffers(buffers), converter(converter) {}

  LogicalResult matchAndRewrite(CoreOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {

    Operation *Op = op.getOperation();
    int col = op.colIndex();
    int row = op.rowIndex();
    std::string coreName("core" + std::to_string(col) + std::to_string(row));
    auto llvmCoreFunc = rewriter.create<LLVMFuncOp>(rewriter.getUnknownLoc(), coreName,
                  LLVMType::getFunctionTy(LLVMType::getVoidTy(converter.getDialect()),
                  {}, /*isVarArg=*/false));

    rewriter.cloneRegionBefore(op.body(), llvmCoreFunc.body(), llvmCoreFunc.body().begin(), mapper);

    DenseMap<Operation *, Value> newAllocated;

    for (auto map : buffers) {
      Operation *tileOp = map.first;
      SmallVector<BufferOp, 4> buffers(map.second);
      TileOp tile = dyn_cast<TileOp>(tileOp);
      int dstCol = tile.colIndex();
      int dstRow = tile.rowIndex();

      if (!isLegalMemAffinity(col, row, dstCol, dstRow))
        continue;

      rewriter.setInsertionPointToStart(&llvmCoreFunc.body().front());
      for (auto buffer : buffers) {
        MemRefType t = buffer.getType().cast<MemRefType>();
        assert(t.getShape().size() == 1 && "Only supporting MemRefType of shape 1 for now!");

        auto int64Ty = LLVM::LLVMType::getInt64Ty(converter.getDialect());
        auto indexType = IndexType::get(rewriter.getContext());
        Value dim = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), int64Ty,
          IntegerAttr::get(indexType, t.getShape()[0]));
        auto ptrType = converter.convertType(t.getElementType()).cast<LLVM::LLVMType>().getPointerTo();
        Value allocated = rewriter.create<LLVM::AllocaOp>(rewriter.getUnknownLoc(),
          ptrType, dim, /*alignment=*/0);
        newAllocated[buffer] = allocated;
      }
    }

    llvmCoreFunc.body().walk([&](Operation *childOp) {
      rewriter.setInsertionPointAfter(childOp);

      if (EndOp end = dyn_cast<EndOp>(childOp)) {
        auto llvmReturn = rewriter.create<LLVM::ReturnOp>(rewriter.getUnknownLoc(), ValueRange({}));
        rewriter.eraseOp(childOp);
      } else if (UseLockOp useLock = dyn_cast<UseLockOp>(childOp)) {
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

        auto useLockFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
        assert(useLockFunc && "Could not find the intrinsic function!");
        SmallVector<Value, 2> args;
        Value lockValue = rewriter.create<LLVM::ConstantOp>(
          rewriter.getUnknownLoc(), LLVMType::getInt32Ty(converter.getDialect()),
          rewriter.getI32IntegerAttr(useLock.getLockValue()));

        Value coreLockIDValue = rewriter.create<LLVM::ConstantOp>(
          rewriter.getUnknownLoc(), LLVMType::getInt32Ty(converter.getDialect()),
          rewriter.getI32IntegerAttr(coreLockID));

        args.push_back(coreLockIDValue);
        args.push_back(lockValue);

        auto useLockCall = rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), useLockFunc, args);

        rewriter.eraseOp(childOp);
      } else if (mlir::StoreOp store = dyn_cast<mlir::StoreOp>(childOp)) {
        // TODO: support multi-dimension indexing
        Value storeIdx = store.indices()[0];
        Value storeValue = store.getValueToStore();
        Operation *storeBuf = store.getMemRef().getDefiningOp();
        if (newAllocated.count(storeBuf) != 0) {
          Value allocated = newAllocated[storeBuf];
          auto indexPtrType = converter.convertType(
            IndexType::get(rewriter.getContext())).cast<LLVM::LLVMType>().getPointerTo();
          auto gep = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(),
                                                  indexPtrType, allocated, ValueRange({storeIdx}));
          rewriter.create<LLVM::StoreOp>(rewriter.getUnknownLoc(), storeValue, gep);

          rewriter.eraseOp(childOp);
        }
      } else if (mlir::LoadOp load = dyn_cast<mlir::LoadOp>(childOp)) {
        // TODO: support multi-dimension indexing
        Value loadIdx= load.indices()[0];
        Operation *loadBuf = load.getMemRef().getDefiningOp();
        if (newAllocated.count(loadBuf) != 0) {
          Value allocated = newAllocated[loadBuf];
          auto indexPtrType = converter.convertType(
            IndexType::get(rewriter.getContext())).cast<LLVM::LLVMType>().getPointerTo();
          auto gep = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(),
                                                  indexPtrType, allocated, ValueRange({loadIdx}));
          auto llvmLoad = rewriter.create<LLVM::LoadOp>(rewriter.getUnknownLoc(), gep);

          rewriter.replaceOp(load, ValueRange{llvmLoad});
        }
      }
    });

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIECoreToLLVMPass : public PassWrapper<AIECoreToLLVMPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder(m.getBody()->getTerminator());

    LLVMTypeConverter converter(&getContext());

    // Extract all CoreOps
    // Create an LLVM func for each CoreOp
    // Clone the region body of each CoreOp to the newly created LLVM func

    DenseMap<std::pair<int, int>, Operation *> tiles;
    DenseMap<Operation *, CoreOp> cores;
    DenseMap<Operation *, MemOp> mems;
    DenseMap<std::pair<Operation *, int>, LockOp> locks;
    DenseMap<Operation *, SmallVector<BufferOp, 4>> buffers;
    DenseMap<Operation *, SwitchboxOp> switchboxes;

    NetlistAnalysis NL(m, tiles, cores, mems, locks, buffers, switchboxes);
    NL.collectTiles(tiles);
    NL.collectCores(cores);
    NL.collectBuffers(buffers);

    // Populate intrinsic functions
    // Intrinsic information: peano/llvm-project/llvm/lib/Target/AIE/AIEInstrInfo.td
    // Also take a look at the tests: peano/llvm-project/llvm/test/CodeGen/AIE
    builder.setInsertionPointToStart(m.getBody());

    SmallVector<LLVMType, 2> callArgTypes;

    // llvm.func @llvm.aie.put.ms(%stream_val: !llvm.i32, %channel: !llvm.i1) -> ()
    callArgTypes.push_back(LLVMType::getInt32Ty(converter.getDialect()));
    callArgTypes.push_back(LLVMType::getIntNTy(converter.getDialect(), 1));
    auto putMSFunc = builder.create<LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.ms",
        LLVMType::getFunctionTy(LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.mws(%stream_val: !llvm.i128, %channel: !llvm.i1) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(LLVMType::getIntNTy(converter.getDialect(), 128));
    callArgTypes.push_back(LLVMType::getIntNTy(converter.getDialect(), 1));
    auto putWMSFunc = builder.create<LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.wms",
        LLVMType::getFunctionTy(LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.mfs(%stream_val: !llvm.float, %channel: !llvm.i1) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(LLVMType::getFloatTy(converter.getDialect()));
    callArgTypes.push_back(LLVMType::getIntNTy(converter.getDialect(), 1));
    auto putFMFunc = builder.create<LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.fms",
        LLVMType::getFunctionTy(LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.ss(%channel: !llvm.i1) -> !llvm.i32
    callArgTypes.clear();
    callArgTypes.push_back(LLVMType::getIntNTy(converter.getDialect(), 1));
    auto getSSFunc = builder.create<LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.ss",
        LLVMType::getFunctionTy(LLVMType::getInt32Ty(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.wss(%channel: !llvm.i1) -> !llvm.i128
    callArgTypes.clear();
    callArgTypes.push_back(LLVMType::getIntNTy(converter.getDialect(), 1));
    auto getWSSFunc = builder.create<LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.wss",
        LLVMType::getFunctionTy(LLVMType::getIntNTy(converter.getDialect(), 128),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.fss(%channel: !llvm.i1) -> !llvm.float
    callArgTypes.clear();
    callArgTypes.push_back(LLVMType::getIntNTy(converter.getDialect(), 1));
    auto getFSSFunc = builder.create<LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.fss",
        LLVMType::getFunctionTy(LLVMType::getFloatTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.scd(%scd_val: !llvm.i384) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(LLVMType::getIntNTy(converter.getDialect(), 384));
    auto putMCDFunc = builder.create<LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.mcd",
        LLVMType::getFunctionTy(LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.scd() -> !llvm.i384
    auto getSCDFunc = builder.create<LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.scd",
        LLVMType::getFunctionTy(LLVMType::getIntNTy(converter.getDialect(), 384),
        {}, /*isVarArg=*/false));

    // llvm.func @llvm.aie.lock.acquire.reg(%lock_id: !llvm.i32, %lock_val: !llvm.i32) ->()
    callArgTypes.clear();
    callArgTypes.push_back(LLVMType::getInt32Ty(converter.getDialect()));
    callArgTypes.push_back(LLVMType::getInt32Ty(converter.getDialect()));
    auto acqLockFunc = builder.create<LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.lock.acquire.reg",
        LLVMType::getFunctionTy(LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.lock.release.reg(%lock_id: !llvm.i32, %lock_val: !llvm.i32) ->()
    callArgTypes.clear();
    callArgTypes.push_back(LLVMType::getInt32Ty(converter.getDialect()));
    callArgTypes.push_back(LLVMType::getInt32Ty(converter.getDialect()));
    auto relLockFunc = builder.create<LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.lock.release.reg",
        LLVMType::getFunctionTy(LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));


    BlockAndValueMapping mapper;

    LLVMConversionTarget target(getContext());

    OwningRewritePatternList patterns;
    patterns.insert<AIEPutStreamLowering,
                    AIEGetStreamLowering,
                    AIEPutCascadeLowering,
                    AIEGetCascadeLowering
                    >(m.getContext(), m, converter);

    populateStdToLLVMConversionPatterns(converter, patterns);
    patterns.insert<AIECoreToLLVMFunc>(m.getContext(), m, mapper, buffers, converter);

    patterns.insert<AIEOpRemoval<AIE::TileOp>,
                    AIEOpRemoval<AIE::MemOp>,
                    AIEOpRemoval<AIE::SwitchboxOp>,
                    AIEOpRemoval<AIE::LockOp>,
                    AIEOpRemoval<AIE::BufferOp>
                   >(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};

void xilinx::AIE::registerAIECoreToLLVMPass() {
    PassRegistration<AIECoreToLLVMPass>(
      "aie-llvm-lowering",
      "Lowering operations in AIE cores' regions to LLVM");
}
