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

struct AIEDebugOpLowering : public OpConversionPattern<DebugOp> {
  using OpConversionPattern<DebugOp>::OpConversionPattern;
  ModuleOp &module;

  AIEDebugOpLowering(MLIRContext *context, ModuleOp &m,
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

struct AIEPutStreamLowering : public OpConversionPattern<PutStreamOp> {
  using OpConversionPattern<PutStreamOp>::OpConversionPattern;
  ModuleOp &module;

  AIEPutStreamLowering(MLIRContext *context, ModuleOp &m,
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

    auto putMSFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    assert(putMSFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.channel());
    args.push_back(op.streamValue());
    rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), putMSFunc, args);
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEGetStreamLowering : public OpConversionPattern<GetStreamOp> {
  using OpConversionPattern<GetStreamOp>::OpConversionPattern;
  ModuleOp &module;

  AIEGetStreamLowering(MLIRContext *context, ModuleOp &m,
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

  AIEPutCascadeLowering(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<PutCascadeOp>(context, benefit),
    module(m) {}

  LogicalResult matchAndRewrite(PutCascadeOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    std::string funcName = "llvm.aie.put.mcd";
    auto putMCDFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    assert(putMCDFunc && "Could not find the intrinsic function!");
    SmallVector<Value, 2> args;
    args.push_back(op.cascadeValue());
    rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), putMCDFunc, args);
    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIEGetCascadeLowering : public OpConversionPattern<GetCascadeOp> {
  using OpConversionPattern<GetCascadeOp>::OpConversionPattern;
  ModuleOp &module;

  AIEGetCascadeLowering(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<GetCascadeOp>(context, benefit),
    module(m) {}

  LogicalResult matchAndRewrite(GetCascadeOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const override {
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
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers;
  DenseMap<Operation *, LLVM::GlobalOp> &bufferToGlobal;
  LLVMTypeConverter &converter;
  int tileCol = 0;
  int tileRow = 0;

  AIECoreToLLVMFunc(MLIRContext *context, ModuleOp &m,
    BlockAndValueMapping &mapper,
    DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers,
    DenseMap<Operation *, LLVM::GlobalOp> &bufferToGlobal,
    LLVMTypeConverter &converter,
    PatternBenefit benefit = 1,
    int tileCol = 1,
    int tileRow = 1
  ) : OpConversionPattern<CoreOp>(context, benefit),
    module(m), mapper(mapper), tileToBuffers(tileToBuffers), bufferToGlobal(bufferToGlobal), converter(converter),
    tileCol(tileCol), tileRow(tileRow) {}

  LogicalResult matchAndRewrite(CoreOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {

    // auto moduleOp = rewriter.create<ModuleOp>(op.getLoc());
    // rewriter.setInsertionPointToStart(moduleOp.getBody());

    // // Clone the existing module for this core, but remove all the coreOps.
    // rewriter.cloneRegionBefore(module.getBodyRegion(), moduleOp.getBodyRegion(), moduleOp.getBodyRegion().begin());
    // for (auto core : moduleOp.getOps<CoreOp>()) {
    //   rewriter.eraseOp(core);
    // }

    Operation *Op = op.getOperation();
    int col = op.colIndex();
    int row = op.rowIndex();

    // Only pull code for the indicated function
    if((tileRow != row) || (tileCol != col)) {
      rewriter.eraseOp(Op);
      return success();
    }

    std::string coreName("core" + std::to_string(col) + std::to_string(row));
    auto llvmCoreFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), coreName,
                  LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&converter.getContext()),
                  {}, /*isVarArg=*/false));

    rewriter.cloneRegionBefore(op.body(), llvmCoreFunc.body(), llvmCoreFunc.body().begin(), mapper);

    // Create a main function that just calls the core function above.
    auto mainFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), "_main",
                 LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&converter.getContext()),
                  {}, /*isVarArg=*/false));
    rewriter.setInsertionPointToStart(mainFunc.addEntryBlock());
    SmallVector<Value, 8> args;
    rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), llvmCoreFunc, args); // call with no args.
    rewriter.create<LLVM::ReturnOp>(rewriter.getUnknownLoc(), args); // return nothing

    DenseMap<Operation *, Value> newAllocated;

    for (auto map : tileToBuffers) {
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

        auto int64Ty = IntegerType::get(&converter.getContext(), 64);
        auto indexType = IndexType::get(rewriter.getContext());
        rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), int64Ty,
          IntegerAttr::get(indexType, t.getShape()[0]));
        LLVM::GlobalOp global = bufferToGlobal[buffer.getOperation()];
        Value allocated = rewriter.create<LLVM::AddressOfOp>(rewriter.getUnknownLoc(), global);
        newAllocated[buffer] = allocated;
      }
    }

    llvmCoreFunc.body().walk([&](Operation *childOp) {
      rewriter.setInsertionPointAfter(childOp);

      if (EndOp end = dyn_cast<EndOp>(childOp)) {
        rewriter.create<LLVM::ReturnOp>(rewriter.getUnknownLoc(), ValueRange({}));
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
          rewriter.getUnknownLoc(), IntegerType::get(&converter.getContext(), 32),
          rewriter.getI32IntegerAttr(useLock.getLockValue()));

        Value coreLockIDValue = rewriter.create<LLVM::ConstantOp>(
          rewriter.getUnknownLoc(), IntegerType::get(&converter.getContext(), 32),
          rewriter.getI32IntegerAttr(coreLockID));

        args.push_back(coreLockIDValue);
        args.push_back(lockValue);

        rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), useLockFunc, args);
        rewriter.eraseOp(childOp);
      } else if (auto store = dyn_cast<mlir::memref::StoreOp>(childOp)) {
        // TODO: support multi-dimension indexing
        Value storeIdx = store.indices()[0];
        Value storeValue = store.getValueToStore();
        Operation *storeBuf = store.getMemRef().getDefiningOp();
        if (newAllocated.count(storeBuf) != 0) {
          Value allocated = newAllocated[storeBuf];
          auto int32Ty = IntegerType::get(&converter.getContext(), 32);
          auto int64Ty = IntegerType::get(&converter.getContext(), 64);
          auto indexType = IndexType::get(rewriter.getContext());
          Value constZero = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), int64Ty,
            IntegerAttr::get(indexType, 0));
          auto gep = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(),
                                                  LLVM::LLVMPointerType::get(int32Ty),
                                                  allocated, ValueRange({constZero, storeIdx}));
          rewriter.create<LLVM::StoreOp>(rewriter.getUnknownLoc(), storeValue, gep);

          rewriter.eraseOp(childOp);
        }
      } else if (auto load = dyn_cast<mlir::memref::LoadOp>(childOp)) {
        // TODO: support multi-dimension indexing
        Value loadIdx= load.indices()[0];
        Operation *loadBuf = load.getMemRef().getDefiningOp();
        if (newAllocated.count(loadBuf) != 0) {
          Value allocated = newAllocated[loadBuf];
          auto retType = allocated.getType().cast<LLVM::LLVMPointerType>().getElementType().cast<LLVM::LLVMArrayType>().getElementType();
          auto int64Ty = IntegerType::get(&converter.getContext(), 64);
          auto indexType = IndexType::get(rewriter.getContext());
          Value constZero = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), int64Ty,
            IntegerAttr::get(indexType, 0));
          auto gep = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(),
                                                  LLVM::LLVMPointerType::get(retType),
                                                  allocated, ValueRange({constZero, loadIdx}));
          auto llvmLoad = rewriter.create<LLVM::LoadOp>(rewriter.getUnknownLoc(), gep);

          rewriter.replaceOp(load, ValueRange{llvmLoad});
        }
      }
    });

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIECoreToLLVMPass : public AIECoreToLLVMBase<AIECoreToLLVMPass> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder(m.getBody()->getTerminator());

    LowerToLLVMOptions options(
        m.getContext(),
        DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    options.useBarePtrCallConv = true;

    LLVMTypeConverter converter(&getContext());

    // Extract all CoreOps
    // Create an LLVM func for each CoreOp
    // Clone the region body of each CoreOp to the newly created LLVM func

    DenseMap<std::pair<int, int>, Operation *> tiles;
    DenseMap<Operation *, CoreOp> cores;
    DenseMap<Operation *, MemOp> mems;
    DenseMap<std::pair<Operation *, int>, LockOp> locks;
    DenseMap<Operation *, SmallVector<BufferOp, 4>> tileToBuffers;
    DenseMap<Operation *, LLVM::GlobalOp> bufferToGlobal;
    DenseMap<Operation *, SwitchboxOp> switchboxes;

    NetlistAnalysis NL(m, tiles, cores, mems, locks, tileToBuffers, switchboxes);
    NL.collectTiles(tiles);
    NL.collectCores(cores);
    NL.collectBuffers(tileToBuffers);

    // Populate intrinsic functions
    // Intrinsic information: peano/llvm-project/llvm/lib/Target/AIE/AIEInstrInfo.td
    // Also take a look at the tests: peano/llvm-project/llvm/test/CodeGen/AIE
    builder.setInsertionPointToStart(m.getBody());

    // Create a Global for each BufferOp
    for (auto buffer : m.getOps<BufferOp>()) {
      MemRefType t = buffer.getType().cast<MemRefType>();
      auto elementType = converter.convertType(t.getElementType());
      // FIXME: Support multi-dimensional types.
      auto globalType = LLVM::LLVMArrayType::get(elementType, t.getShape()[0]);
      Attribute value; // has no value.
      auto symName = buffer.name().getValue();
      auto global = builder.create<LLVM::GlobalOp>(buffer.getLoc(), globalType,
         false, LLVM::Linkage::External, symName, value);
      bufferToGlobal[buffer.getOperation()] = global;
    }

    SmallVector<Type, 2> callArgTypes;

    // llvm.func @debug_i32(%val: !llvm.i32) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "debug_i32",
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&converter.getContext()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.ms(%channel: !llvm.i1, %stream_val: !llvm.i32) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    // callArgTypes.push_back(IntegerType::get(&converter.getContext(), 1));
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.ms",
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&converter.getContext()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.mws(%channel: !llvm.i1, %stream_val: !llvm.i128) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 128));
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.wms",
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&converter.getContext()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.mfs(%channel: !llvm.i1, %stream_val: !llvm.float) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    callArgTypes.push_back(Float32Type::get(&converter.getContext()));
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.fms",
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&converter.getContext()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.ss(%channel: !llvm.i1) -> !llvm.i32
    callArgTypes.clear();
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.ss",
        LLVM::LLVMFunctionType::get(IntegerType::get(&converter.getContext(), 32),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.wss(%channel: !llvm.i1) -> !llvm.i128
    callArgTypes.clear();
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.wss",
        LLVM::LLVMFunctionType::get(IntegerType::get(&converter.getContext(), 128),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.fss(%channel: !llvm.i1) -> !llvm.float
    callArgTypes.clear();
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.fss",
        LLVM::LLVMFunctionType::get(Float32Type::get(&converter.getContext()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.scd(%scd_val: !llvm.i384) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 384));
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.mcd",
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&converter.getContext()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.scd() -> !llvm.i384
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.scd",
        LLVM::LLVMFunctionType::get(IntegerType::get(&converter.getContext(), 384),
        {}, /*isVarArg=*/false));

    // llvm.func @llvm.aie.lock.acquire.reg(%lock_id: !llvm.i32, %lock_val: !llvm.i32) ->()
    callArgTypes.clear();
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.lock.acquire.reg",
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&converter.getContext()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.lock.release.reg(%lock_id: !llvm.i32, %lock_val: !llvm.i32) ->()
    callArgTypes.clear();
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    callArgTypes.push_back(IntegerType::get(&converter.getContext(), 32));
    builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.lock.release.reg",
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(&converter.getContext()),
        callArgTypes, /*isVarArg=*/false));


    BlockAndValueMapping mapper;

    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    // target.addLegalDialect<LLVM::LLVMDialect>();
    // target.addIllegalOp<LLVM::DialectCastOp>();

    OwningRewritePatternList patterns(&getContext());
    patterns.insert<AIEPutStreamLowering,
                    AIEGetStreamLowering,
                    AIEPutCascadeLowering,
                    AIEGetCascadeLowering
//                    AIEDebugOpLowering
                    >(m.getContext(), m);

    populateStdToLLVMConversionPatterns(converter, patterns);
    patterns.insert<AIECoreToLLVMFunc>(m.getContext(), m, mapper, tileToBuffers, bufferToGlobal, converter, 1,
       tileCol, tileRow);

    // if (failed(applyPartialConversion(m, target, std::move(patterns))))
    //   signalPassFailure();

    patterns.insert<AIEOpRemoval<AIE::FlowOp>,
                    AIEOpRemoval<AIE::TileOp>,
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
xilinx::AIE::createAIECoreToLLVMPass() {
  return std::make_unique<AIECoreToLLVMPass>();
}
