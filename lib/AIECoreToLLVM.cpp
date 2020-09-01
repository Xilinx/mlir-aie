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
    auto call = rewriter.create<CallOp>(rewriter.getUnknownLoc(), func, args);
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
    auto putMSCall = rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), putMSFunc, args);
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
    auto putMCDCall = rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), putMCDFunc, args);
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
  DenseMap<Operation *, LLVM::GlobalOp> &bufferToGlobal;
  LLVMTypeConverter &converter;

  AIECoreToLLVMFunc(MLIRContext *context, ModuleOp &m,
    BlockAndValueMapping &mapper,
    DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers,
    DenseMap<Operation *, LLVM::GlobalOp> &bufferToGlobal,
    LLVMTypeConverter &converter,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<CoreOp>(context, benefit),
    module(m), mapper(mapper), buffers(buffers), bufferToGlobal(bufferToGlobal), converter(converter) {}

  LogicalResult matchAndRewrite(CoreOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {

    Operation *Op = op.getOperation();
    int col = op.colIndex();
    int row = op.rowIndex();
    std::string coreName("core" + std::to_string(col) + std::to_string(row));
    auto llvmCoreFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(), coreName,
                  LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(converter.getDialect()),
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
        LLVM::GlobalOp global = bufferToGlobal[buffer.getOperation()];
        Value allocated = rewriter.create<LLVM::AddressOfOp>(rewriter.getUnknownLoc(), global);
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
          rewriter.getUnknownLoc(), LLVM::LLVMType::getInt32Ty(converter.getDialect()),
          rewriter.getI32IntegerAttr(useLock.getLockValue()));

        Value coreLockIDValue = rewriter.create<LLVM::ConstantOp>(
          rewriter.getUnknownLoc(), LLVM::LLVMType::getInt32Ty(converter.getDialect()),
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
          auto retType = allocated.getType().cast<LLVM::LLVMType>().getPointerElementTy().getArrayElementType();
          auto int32Ty = LLVM::LLVMType::getInt32Ty(converter.getDialect());
          auto int64Ty = LLVM::LLVMType::getInt64Ty(converter.getDialect());
          auto indexType = IndexType::get(rewriter.getContext());
          Value constZero = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), int64Ty,
            IntegerAttr::get(indexType, 0));
          auto gep = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(),
                                                  int32Ty.getPointerTo(),
                                                  allocated, ValueRange({constZero, storeIdx}));
          rewriter.create<LLVM::StoreOp>(rewriter.getUnknownLoc(), storeValue, gep);

          rewriter.eraseOp(childOp);
        }
      } else if (mlir::LoadOp load = dyn_cast<mlir::LoadOp>(childOp)) {
        // TODO: support multi-dimension indexing
        Value loadIdx= load.indices()[0];
        Operation *loadBuf = load.getMemRef().getDefiningOp();
        if (newAllocated.count(loadBuf) != 0) {
          Value allocated = newAllocated[loadBuf];
          auto retType = allocated.getType().cast<LLVM::LLVMType>().getPointerElementTy().getArrayElementType();
          auto int32Ty = LLVM::LLVMType::getInt32Ty(converter.getDialect());
          auto int64Ty = LLVM::LLVMType::getInt64Ty(converter.getDialect());
          auto indexType = IndexType::get(rewriter.getContext());
          Value constZero = rewriter.create<LLVM::ConstantOp>(rewriter.getUnknownLoc(), int64Ty,
            IntegerAttr::get(indexType, 0));
          auto gep = rewriter.create<LLVM::GEPOp>(rewriter.getUnknownLoc(),
                                                  retType.getPointerTo(),
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
    DenseMap<Operation *, LLVM::GlobalOp> bufferToGlobal;
    DenseMap<Operation *, SwitchboxOp> switchboxes;

    NetlistAnalysis NL(m, tiles, cores, mems, locks, buffers, switchboxes);
    NL.collectTiles(tiles);
    NL.collectCores(cores);
    NL.collectBuffers(buffers);

    // Populate intrinsic functions
    // Intrinsic information: peano/llvm-project/llvm/lib/Target/AIE/AIEInstrInfo.td
    // Also take a look at the tests: peano/llvm-project/llvm/test/CodeGen/AIE
    builder.setInsertionPointToStart(m.getBody());

    // Create a Global for each BufferOp
    for (auto buffer : m.getOps<BufferOp>()) {
      MemRefType t = buffer.getType().cast<MemRefType>();
      auto elementType = converter.convertType(t.getElementType()).cast<LLVM::LLVMType>();
      // FIXME: Support multi-dimensional types.
      auto globalType = LLVM::LLVMType::getArrayTy(elementType, t.getShape()[0]);
      Attribute value; // has no value.
      // FIXME: might not exist
      auto symName = buffer.getAttrOfType<StringAttr>("sym_name").getValue();
      auto global = builder.create<LLVM::GlobalOp>(buffer.getLoc(), globalType,
         false, LLVM::Linkage::External, symName, value);
      bufferToGlobal[buffer.getOperation()] = global;
    }

    SmallVector<LLVM::LLVMType, 2> callArgTypes;

    // llvm.func @debug_i32(%val: !llvm.i32) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(LLVM::LLVMType::getInt32Ty(converter.getDialect()));
    auto debug_i32Func = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "debug_i32",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.ms(%channel: !llvm.i1, %stream_val: !llvm.i32) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(LLVM::LLVMType::getIntNTy(converter.getDialect(), 32));
    callArgTypes.push_back(LLVM::LLVMType::getInt32Ty(converter.getDialect()));
    // callArgTypes.push_back(LLVM::LLVMType::getIntNTy(converter.getDialect(), 1));
    auto putMSFunc = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.ms",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.mws(%channel: !llvm.i1, %stream_val: !llvm.i128) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(LLVM::LLVMType::getIntNTy(converter.getDialect(), 32));
    callArgTypes.push_back(LLVM::LLVMType::getIntNTy(converter.getDialect(), 128));
    auto putWMSFunc = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.wms",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.mfs(%channel: !llvm.i1, %stream_val: !llvm.float) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(LLVM::LLVMType::getIntNTy(converter.getDialect(), 32));
    callArgTypes.push_back(LLVM::LLVMType::getFloatTy(converter.getDialect()));
    auto putFMFunc = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.fms",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.ss(%channel: !llvm.i1) -> !llvm.i32
    callArgTypes.clear();
    callArgTypes.push_back(LLVM::LLVMType::getIntNTy(converter.getDialect(), 32));
    auto getSSFunc = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.ss",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getInt32Ty(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.wss(%channel: !llvm.i1) -> !llvm.i128
    callArgTypes.clear();
    callArgTypes.push_back(LLVM::LLVMType::getIntNTy(converter.getDialect(), 32));
    auto getWSSFunc = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.wss",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getIntNTy(converter.getDialect(), 128),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.fss(%channel: !llvm.i1) -> !llvm.float
    callArgTypes.clear();
    callArgTypes.push_back(LLVM::LLVMType::getIntNTy(converter.getDialect(), 32));
    auto getFSSFunc = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.fss",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getFloatTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.put.scd(%scd_val: !llvm.i384) -> ()
    callArgTypes.clear();
    callArgTypes.push_back(LLVM::LLVMType::getIntNTy(converter.getDialect(), 384));
    auto putMCDFunc = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.put.mcd",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.get.scd() -> !llvm.i384
    auto getSCDFunc = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.get.scd",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getIntNTy(converter.getDialect(), 384),
        {}, /*isVarArg=*/false));

    // llvm.func @llvm.aie.lock.acquire.reg(%lock_id: !llvm.i32, %lock_val: !llvm.i32) ->()
    callArgTypes.clear();
    callArgTypes.push_back(LLVM::LLVMType::getInt32Ty(converter.getDialect()));
    callArgTypes.push_back(LLVM::LLVMType::getInt32Ty(converter.getDialect()));
    auto acqLockFunc = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.lock.acquire.reg",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));

    // llvm.func @llvm.aie.lock.release.reg(%lock_id: !llvm.i32, %lock_val: !llvm.i32) ->()
    callArgTypes.clear();
    callArgTypes.push_back(LLVM::LLVMType::getInt32Ty(converter.getDialect()));
    callArgTypes.push_back(LLVM::LLVMType::getInt32Ty(converter.getDialect()));
    auto relLockFunc = builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), "llvm.aie.lock.release.reg",
        LLVM::LLVMType::getFunctionTy(LLVM::LLVMType::getVoidTy(converter.getDialect()),
        callArgTypes, /*isVarArg=*/false));


    BlockAndValueMapping mapper;

    LLVMConversionTarget target(getContext());

    OwningRewritePatternList patterns;
    patterns.insert<AIEPutStreamLowering,
                    AIEGetStreamLowering,
                    AIEPutCascadeLowering,
                    AIEGetCascadeLowering
//                    AIEDebugOpLowering
                    >(m.getContext(), m);

    populateStdToLLVMConversionPatterns(converter, patterns);
    patterns.insert<AIECoreToLLVMFunc>(m.getContext(), m, mapper, buffers, bufferToGlobal, converter);

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

struct AIECoreToStandardFunc : public OpConversionPattern<CoreOp> {
  using OpConversionPattern<CoreOp>::OpConversionPattern;
  ModuleOp &module;
  BlockAndValueMapping &mapper;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers;

  AIECoreToStandardFunc(MLIRContext *context, ModuleOp &m,
    BlockAndValueMapping &mapper,
    DenseMap<Operation *, SmallVector<BufferOp, 4>> &buffers,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<CoreOp>(context, benefit),
    module(m), mapper(mapper), buffers(buffers) {}

  LogicalResult matchAndRewrite(CoreOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {

    Operation *Op = op.getOperation();
    int col = op.colIndex();
    int row = op.rowIndex();
    std::string coreName("core" + std::to_string(col) + std::to_string(row));
    auto coreFunc = rewriter.create<FuncOp>(rewriter.getUnknownLoc(), coreName,
                  FunctionType::get({}, {}, rewriter.getContext()));

    rewriter.cloneRegionBefore(op.body(), coreFunc.getBody(), coreFunc.getBody().begin(), mapper);

    DenseMap<Operation *, Value> newAllocated;

    for (auto map : buffers) {
      Operation *tileOp = map.first;
      SmallVector<BufferOp, 4> buffers(map.second);
      TileOp tile = dyn_cast<TileOp>(tileOp);
      int dstCol = tile.colIndex();
      int dstRow = tile.rowIndex();

      if (!isLegalMemAffinity(col, row, dstCol, dstRow))
        continue;

      rewriter.setInsertionPointToStart(&coreFunc.getBody().front());
      for (auto buffer : buffers) {
        MemRefType t = buffer.getType().cast<MemRefType>();
        assert(t.getShape().size() == 1 && "Only supporting MemRefType of shape 1 for now!");
        Value allocated = rewriter.create<AllocOp>(rewriter.getUnknownLoc(), t);
        newAllocated[buffer] = allocated;
      }
    }

    coreFunc.getBody().walk([&](Operation *childOp) {
      rewriter.setInsertionPointAfter(childOp);

      if (EndOp end = dyn_cast<EndOp>(childOp)) {
        auto returnOp = rewriter.create<ReturnOp>(rewriter.getUnknownLoc(), ValueRange({}));
        rewriter.eraseOp(childOp);
      }
      //  else if (UseLockOp useLock = dyn_cast<UseLockOp>(childOp)) {
      //   LockOp lock = dyn_cast<LockOp>(useLock.lock().getDefiningOp());
      //   TileOp tile = dyn_cast<TileOp>(lock.tile().getDefiningOp());
      //   int dstCol = tile.colIndex();
      //   int dstRow = tile.rowIndex();

      //   int cardinalMemOffset = 0;

      //   if (isMemSouth(col, row, dstCol, dstRow))
      //     cardinalMemOffset = 0;
      //   else if (isMemWest(col, row, dstCol, dstRow))
      //     cardinalMemOffset = 16;
      //   else if (isMemNorth(col, row, dstCol, dstRow))
      //     cardinalMemOffset = 32;
      //   else if (isMemEast(col, row, dstCol, dstRow))
      //     cardinalMemOffset = 48;
      //   else
      //     llvm_unreachable("Found illegal lock user!");

      //   int coreLockID = cardinalMemOffset + lock.getLockID();

      //   std::string funcName = "llvm.aie.lock.";
      //   if (useLock.acquire())
      //     funcName += "acquire.reg";
      //   else if (useLock.release())
      //     funcName += "release.reg";

      //   auto useLockFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
      //   assert(useLockFunc && "Could not find the intrinsic function!");
      //   SmallVector<Value, 2> args;
      //   Value lockValue = rewriter.create<LLVM::ConstantOp>(
      //     rewriter.getUnknownLoc(), LLVM::LLVMType::getInt32Ty(converter.getDialect()),
      //     rewriter.getI32IntegerAttr(useLock.getLockValue()));

      //   Value coreLockIDValue = rewriter.create<LLVM::ConstantOp>(
      //     rewriter.getUnknownLoc(), LLVM::LLVMType::getInt32Ty(converter.getDialect()),
      //     rewriter.getI32IntegerAttr(coreLockID));

      //   args.push_back(coreLockIDValue);
      //   args.push_back(lockValue);

      //   auto useLockCall = rewriter.create<LLVM::CallOp>(rewriter.getUnknownLoc(), useLockFunc, args);

      //   rewriter.eraseOp(childOp);
      // }
    });

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIECoreToStandardPass : public PassWrapper<AIECoreToStandardPass, OperationPass<ModuleOp>> {
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

    SmallVector<LLVM::LLVMType, 2> callArgTypes;

    Type int32Type = IntegerType::get(32, builder.getContext());
    Type int128Type = IntegerType::get(128, builder.getContext());
    Type int384Type = IntegerType::get(384, builder.getContext());
    Type floatType = FloatType::getF32(builder.getContext());

    // llvm.func @debug_i32(%val: !llvm.i32) -> ()
    auto debug_i32Func = builder.create<FuncOp>(builder.getUnknownLoc(), "debug_i32",
        FunctionType::get({int32Type}, {}, builder.getContext()));

    // llvm.func @llvm.aie.put.ms(%channel: !llvm.i1, %stream_val: !llvm.i32) -> ()
    auto putMSFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.put.ms",
        FunctionType::get({int32Type, int32Type}, {}, builder.getContext()));

    // llvm.func @llvm.aie.put.mws(%channel: !llvm.i1, %stream_val: !llvm.i128) -> ()
    auto putWMSFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.put.wms",
        FunctionType::get({int32Type, int128Type}, {}, builder.getContext()));

    // llvm.func @llvm.aie.put.mfs(%channel: !llvm.i1, %stream_val: !llvm.float) -> ()
    auto putFMFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.put.fms",
        FunctionType::get({int32Type, floatType}, {}, builder.getContext()));

    // llvm.func @llvm.aie.get.ss(%channel: !llvm.i1) -> !llvm.i32
    auto getSSFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.get.ss",
        FunctionType::get({int32Type}, {int32Type}, builder.getContext()));

    // llvm.func @llvm.aie.get.wss(%channel: !llvm.i1) -> !llvm.i128
    auto getWSSFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.get.wss",
        FunctionType::get({int32Type}, {int128Type}, builder.getContext()));

    // llvm.func @llvm.aie.get.fss(%channel: !llvm.i1) -> !llvm.float
    auto getFSSFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.get.fss",
        FunctionType::get({int32Type}, {floatType}, builder.getContext()));

    // llvm.func @llvm.aie.put.scd(%scd_val: !llvm.i384) -> ()
    auto putMCDFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.put.mcd",
        FunctionType::get({int384Type}, {}, builder.getContext()));

    // llvm.func @llvm.aie.get.scd() -> !llvm.i384
    auto getSCDFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.get.scd",
        FunctionType::get({}, {int384Type}, builder.getContext()));

    // llvm.func @llvm.aie.lock.acquire.reg(%lock_id: !llvm.i32, %lock_val: !llvm.i32) ->()
    auto acqLockFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.lock.acquire.reg",
        FunctionType::get({int32Type, int32Type}, {}, builder.getContext()));

    // llvm.func @llvm.aie.lock.release.reg(%lock_id: !llvm.i32, %lock_val: !llvm.i32) ->()
    auto relLockFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "llvm.aie.lock.release.reg",
        FunctionType::get({int32Type, int32Type}, {}, builder.getContext()));


    BlockAndValueMapping mapper;

    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<FuncOp>();

    OwningRewritePatternList patterns;
    patterns.insert<
    // AIEPutStreamLowering,
    //                 AIEGetStreamLowering,
    //                 AIEPutCascadeLowering,
    //                 AIEGetCascadeLowering,
                     AIEDebugOpLowering
                    >(m.getContext(), m);

    patterns.insert<AIECoreToStandardFunc>(m.getContext(), m, mapper, buffers);

    // patterns.insert<AIEOpRemoval<AIE::TileOp>,
    //                 AIEOpRemoval<AIE::MemOp>,
    //                 AIEOpRemoval<AIE::SwitchboxOp>,
    //                 AIEOpRemoval<AIE::LockOp>,
    //                 AIEOpRemoval<AIE::BufferOp>
    //                >(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};

void xilinx::AIE::registerAIECoreToLLVMPass() {
    PassRegistration<AIECoreToLLVMPass>(
      "aie-llvm-lowering",
      "Lowering operations in AIE cores' regions to LLVM");
    PassRegistration<AIECoreToStandardPass>(
      "aie-standard-lowering",
      "Lowering operations in AIE cores' regions to Standard Dialect");
}
