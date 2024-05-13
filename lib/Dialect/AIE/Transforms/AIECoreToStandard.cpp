//===- AIECoreToStandard.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::vector;
using namespace xilinx;
using namespace xilinx::AIE;

static StringRef getArchIntrinsicString(AIEArch arch) {
  switch (arch) {
  case AIEArch::AIE1:
    return "aie";
  case AIEArch::AIE2:
    return "aie2";
  }
  llvm::report_fatal_error("unsupported arch");
}

typedef std::tuple<const char *, std::vector<Type>, std::vector<Type>>
    IntrinsicDecl;
typedef std::vector<IntrinsicDecl> IntrinsicDecls;

static auto getAIE1Intrinsics(OpBuilder &builder) {
  Type int32Type = IntegerType::get(builder.getContext(), 32);
  Type int128Type = IntegerType::get(builder.getContext(), 128);
  Type int384Type = IntegerType::get(builder.getContext(), 384);
  Type floatType = FloatType::getF32(builder.getContext());

  // Note that not all of these are valid for a particular design, or needed.
  // For right now, we will just accept the noise.
  IntrinsicDecls functions = {
      {"debug_i32", {int32Type}, {}},
      {"llvm.aie.event0", {}, {}},
      {"llvm.aie.event1", {}, {}},
      {"llvm.aie.put.ms",
       {int32Type, int32Type},
       {}}, //(%channel, %value) -> ()
      {"llvm.aie.put.wms",
       {int32Type, int128Type},
       {}}, //(%channel, %value) -> ()
      {"llvm.aie.put.fms",
       {int32Type, floatType},
       {}},                                          //(%channel, %value) -> ()
      {"llvm.aie.get.ss", {int32Type}, {int32Type}}, //(%channel, %value) -> ()
      {"llvm.aie.get.wss",
       {int32Type},
       {int128Type}},                                 //(%channel, %value) -> ()
      {"llvm.aie.get.fss", {int32Type}, {floatType}}, //(%channel, %value) -> ()
      {"llvm.aie.put.mcd", {int384Type}, {}},
      {"llvm.aie.get.scd", {}, {int384Type}},
      {"llvm.aie.lock.acquire.reg",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
      {"llvm.aie.lock.release.reg",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
  };
  return functions;
}

static auto getAIE2Intrinsics(OpBuilder &builder) {
  Type int32Type = IntegerType::get(builder.getContext(), 32);
  Type accType = VectorType::get({16}, int32Type);
  IntrinsicDecls functions = {
      {"debug_i32", {int32Type}, {}},
      {"llvm.aie2.put.ms", {int32Type, int32Type}, {}}, //(%value, %tlast) -> ()
      {"llvm.aie2.get.ss", {}, {int32Type, int32Type}}, //() -> (%value, %tlast)
      {"llvm.aie2.mcd.write.vec",
       {accType, int32Type},
       {}}, // (%value, %enable) -> ()
      {"llvm.aie2.scd.read.vec",
       {int32Type},
       {accType}}, // (%enable) -> (%value)
      {"llvm.aie2.acquire",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
      {"llvm.aie2.release",
       {int32Type, int32Type},
       {}}, //(%lock_id, %lock_val) -> ()
  };
  return functions;
}

static void declareAIEIntrinsics(AIEArch arch, OpBuilder &builder) {
  auto registerIntrinsics = [&builder](IntrinsicDecls functions) {
    for (auto &i : functions) {
      auto [name, argTypes, retTypes] = i;
      builder
          .create<func::FuncOp>(
              builder.getUnknownLoc(), name,
              FunctionType::get(builder.getContext(), argTypes, retTypes))
          .setPrivate();
    }
  };
  switch (arch) {
  case AIEArch::AIE1:
    registerIntrinsics(getAIE1Intrinsics(builder));
    return;
  case AIEArch::AIE2:
    registerIntrinsics(getAIE2Intrinsics(builder));
    return;
  }
  llvm::report_fatal_error("unsupported arch");
}

template <typename MyAIEOp>
struct AIEOpRemoval : OpConversionPattern<MyAIEOp> {
  using OpConversionPattern<MyAIEOp>::OpConversionPattern;
  using OpAdaptor = typename MyAIEOp::Adaptor;
  ModuleOp &module;

  AIEOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyAIEOp>(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(MyAIEOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEDebugOpToStdLowering : OpConversionPattern<DebugOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEDebugOpToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(DebugOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string funcName = "debug_i32";
    auto func = module.lookupSymbol<func::FuncOp>(funcName);
    if (!func)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 1> args;
    args.push_back(op.getArg());
    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), func, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEPutStreamToStdLowering : OpConversionPattern<PutStreamOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEPutStreamToStdLowering(MLIRContext *context, ModuleOp &m,
                            PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(PutStreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.put.";
    else
      funcName = "llvm.aie2.put.";

    if (op.isWideStream())
      funcName += "wms";
    else if (op.isFloatStream())
      funcName += "fms";
    else
      funcName += "ms";

    auto putMSFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!putMSFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    if (targetModel.getTargetArch() == AIEArch::AIE1) {
      args.push_back(op.getChannel());
      args.push_back(op.getStreamValue());
    } else {
      args.push_back(op.getStreamValue());
      args.push_back(rewriter.create<arith::ConstantOp>(
          op.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(0))); // tlast
    }
    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), putMSFunc, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEGetStreamToStdLowering : OpConversionPattern<GetStreamOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEGetStreamToStdLowering(MLIRContext *context, ModuleOp &m,
                            PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(GetStreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.get.";
    else
      funcName = "llvm.aie2.get.";

    if (op.isWideStream())
      funcName += "wss";
    else if (op.isFloatStream())
      funcName += "fss";
    else
      funcName += "ss";

    auto getSSFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!getSSFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      args.push_back(op.getChannel());
    auto getSSCall = rewriter.create<func::CallOp>(rewriter.getUnknownLoc(),
                                                   getSSFunc, args);
    rewriter.replaceOp(op, getSSCall.getResult(0));
    // Capture TLAST in AIEv2?
    return success();
  }
};

struct AIEPutCascadeToStdLowering : OpConversionPattern<PutCascadeOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEPutCascadeToStdLowering(MLIRContext *context, ModuleOp &m,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(PutCascadeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.put.mcd";
    else
      funcName = "llvm.aie2.mcd.write.vec";
    auto putMCDFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!putMCDFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    args.push_back(op.getCascadeValue());
    if (targetModel.getTargetArch() == AIEArch::AIE2)
      args.push_back(rewriter.create<arith::ConstantOp>(
          op.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(1))); // enable

    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), putMCDFunc, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEGetCascadeToStdLowering : OpConversionPattern<GetCascadeOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEGetCascadeToStdLowering(MLIRContext *context, ModuleOp &m,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(GetCascadeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto device = op->getParentOfType<DeviceOp>();
    const auto &targetModel = device.getTargetModel();
    std::string funcName;
    if (targetModel.getTargetArch() == AIEArch::AIE1)
      funcName = "llvm.aie.get.scd";
    else
      funcName = "llvm.aie2.scd.read.vec";
    auto getSCDFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!getSCDFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    SmallVector<Value, 2> args;
    if (targetModel.getTargetArch() == AIEArch::AIE2)
      args.push_back(rewriter.create<arith::ConstantOp>(
          op.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(1))); // enable

    auto getSCDCall = rewriter.create<func::CallOp>(rewriter.getUnknownLoc(),
                                                    getSCDFunc, args);
    rewriter.replaceOp(op, getSCDCall.getResult(0));
    return success();
  }
};

struct AIEUseLockToStdLowering : OpConversionPattern<UseLockOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEUseLockToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}
  LogicalResult
  matchAndRewrite(UseLockOp useLock, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<DeviceOp>(useLock->getParentOp())) {
      auto device = useLock->getParentOfType<DeviceOp>();
      if (!device) {
        return module.emitOpError("Device Not found!");
      }
      const auto &targetModel = device.getTargetModel();

      // Generate the intrinsic name
      std::string funcName;
      if (targetModel.getTargetArch() == AIEArch::AIE1)
        funcName = "llvm.aie.lock.";
      else
        funcName = "llvm.aie2.";
      if (useLock.acquire() || useLock.acquireGE())
        funcName += "acquire";
      else if (useLock.release())
        funcName += "release";
      if (targetModel.getTargetArch() == AIEArch::AIE1)
        funcName += ".reg";

      auto useLockFunc = module.lookupSymbol<func::FuncOp>(funcName);
      if (!useLockFunc)
        return useLock.emitOpError("Could not find the intrinsic function!");

      SmallVector<Value, 2> args;
      auto lockValue = useLock.getLockValue();

      // AIE2 acquire greater equal is encoded as a negative value.
      if (useLock.acquireGE()) {
        lockValue = -lockValue;
      }
      args.push_back(rewriter.create<arith::IndexCastOp>(
          useLock.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          useLock.getLock()));
      args.push_back(rewriter.create<arith::ConstantOp>(
          useLock.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(lockValue)));

      rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), useLockFunc,
                                    args);
    }
    rewriter.eraseOp(useLock);
    return success();
  }
};

struct AIEBufferToStandard : OpConversionPattern<BufferOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;
  int tileCol = 0;
  int tileRow = 0;
  AIEBufferToStandard(MLIRContext *context, ModuleOp &m,
                      PatternBenefit benefit = 1, int tileCol = -1,
                      int tileRow = -1)
      : OpConversionPattern(context, benefit), module(m), tileCol(tileCol),
        tileRow(tileRow) {}
  LogicalResult
  matchAndRewrite(BufferOp buffer, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPointToStart(module.getBody());
    auto t = llvm::cast<MemRefType>(buffer.getType());
    int col = llvm::cast<TileOp>(buffer.getTile().getDefiningOp()).getCol();
    int row = llvm::cast<TileOp>(buffer.getTile().getDefiningOp()).getRow();
    auto symName = buffer.name().getValue();
    mlir::ElementsAttr initValue = buffer.getInitialValueAttr();
    // Don't emit initialization for cores that don't "own" the buffer (to
    // prevent duplication in the data section of the elf/object file)
    if ((tileRow != row && tileRow != -1) || (tileCol != col && tileCol != -1))
      initValue = nullptr;
    rewriter.create<memref::GlobalOp>(
        rewriter.getUnknownLoc(), symName, rewriter.getStringAttr("public"),
        buffer.getType(), initValue, /*constant*/ false,
        /*alignment*/ nullptr);

    for (auto &use : make_early_inc_range(buffer.getResult().getUses())) {
      Operation *user = use.getOwner();
      rewriter.setInsertionPoint(user);
      auto allocated = rewriter.create<memref::GetGlobalOp>(
          rewriter.getUnknownLoc(), t, symName);
      // Assume that buffers are aligned so they can be vectorized.
      rewriter.create<memref::AssumeAlignmentOp>(rewriter.getUnknownLoc(),
                                                 allocated, 32);

      use.set(allocated.getResult());
    }

    rewriter.eraseOp(buffer);
    return success();
  }
};

struct AIECoreToStandardFunc : OpConversionPattern<CoreOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;
  IRMapping &mapper;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers;
  int tileCol = 0;
  int tileRow = 0;

  AIECoreToStandardFunc(
      MLIRContext *context, ModuleOp &m, IRMapping &mapper,
      DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers,
      PatternBenefit benefit = 1, int tileCol = 1, int tileRow = 1)
      : OpConversionPattern(context, benefit), module(m), mapper(mapper),
        tileToBuffers(tileToBuffers), tileCol(tileCol), tileRow(tileRow) {}

  LogicalResult
  matchAndRewrite(CoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    int col = op.colIndex();
    int row = op.rowIndex();

    // Only pull code for the indicated function
    if ((tileRow != row && tileRow != -1) ||
        (tileCol != col && tileCol != -1)) {
      rewriter.eraseOp(op);
      return success();
    }

    // The parent should be an AIE.device op.
    rewriter.setInsertionPointAfter(op->getParentOp());

    std::string coreName("core_" + std::to_string(col) + "_" +
                         std::to_string(row));
    auto coreFunc = rewriter.create<func::FuncOp>(
        rewriter.getUnknownLoc(), coreName,
        FunctionType::get(rewriter.getContext(), {}, {}));

    rewriter.cloneRegionBefore(op.getBody(), coreFunc.getBody(),
                               coreFunc.getBody().begin(), mapper);

    // Rewrite the AIE.end() op
    coreFunc.getBody().walk([&](Operation *childOp) {
      rewriter.setInsertionPointAfter(childOp);

      if (isa<EndOp>(childOp)) {
        rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc(),
                                        ValueRange({}));
        rewriter.eraseOp(childOp);
      }
    });

    rewriter.eraseOp(op);
    return success();
  }
};

// Move all the ops with OpTy inside device, to just before the device.
template <typename OpTy>
void outlineOps(DeviceOp device) {
  SmallVector<OpTy, 16> ops;
  for (const auto &op : device.getOps<OpTy>())
    ops.push_back(op);

  for (const auto &op : ops)
    op->moveBefore(device);
}

// Lower AIE.event to llvm.aie.event intrinsic
struct AIEEventOpToStdLowering : OpConversionPattern<EventOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEEventOpToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(EventOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::string funcName = "llvm.aie.event" + std::to_string(op.getVal());
    auto eventFunc = module.lookupSymbol<func::FuncOp>(funcName);
    if (!eventFunc)
      return op.emitOpError("Could not find the intrinsic function ")
             << funcName;
    rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), eventFunc,
                                  ValueRange({}));
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIECoreToStandardPass : AIECoreToStandardBase<AIECoreToStandardPass> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    if (m.getOps<DeviceOp>().empty()) {
      m.emitOpError("expected AIE.device operation at toplevel");
      return signalPassFailure();
    }
    DeviceOp device = *m.getOps<DeviceOp>().begin();
    const auto &targetModel = device.getTargetModel();

    // Ensure that we don't have an incorrect target triple.  This may override
    // some bogus target triple in the original mlir.
    m->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               builder.getStringAttr(
                   getArchIntrinsicString(targetModel.getTargetArch())));

    DenseMap<Operation *, SmallVector<BufferOp, 4>> tileToBuffers;

    // Populate intrinsic functions
    // Intrinsic information:
    // peano/llvm-project/llvm/lib/Target/AIE/AIEInstrInfo.td Also take a look
    // at the tests: peano/llvm-project/llvm/test/CodeGen/AIE
    builder.setInsertionPointToStart(m.getBody());
    declareAIEIntrinsics(targetModel.getTargetArch(), builder);

    IRMapping mapper;
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<VectorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalOp<func::FuncOp, ModuleOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<AIEPutStreamToStdLowering, AIEGetStreamToStdLowering,
                 AIEPutCascadeToStdLowering, AIEGetCascadeToStdLowering,
                 AIEDebugOpToStdLowering, AIEUseLockToStdLowering,
                 AIEEventOpToStdLowering>(m.getContext(), m);

    patterns.add<AIEBufferToStandard>(m.getContext(), m, /*benefit*/ 1, tileCol,
                                      tileRow);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      return signalPassFailure();

    RewritePatternSet outlinePatterns(&getContext());
    outlinePatterns.add<AIECoreToStandardFunc>(m.getContext(), m, mapper,
                                               tileToBuffers, /*benefit*/ 1,
                                               tileCol, tileRow);
    if (failed(applyPartialConversion(m, target, std::move(outlinePatterns))))
      return signalPassFailure();

    // Move all the func.func ops and memref.globals from the device to the
    // module
    outlineOps<memref::GlobalOp>(device);
    outlineOps<func::FuncOp>(device);

    RewritePatternSet removepatterns(&getContext());
    removepatterns.add<
        AIEOpRemoval<DeviceOp>, AIEOpRemoval<TileOp>, AIEOpRemoval<FlowOp>,
        AIEOpRemoval<MemOp>, AIEOpRemoval<ShimDMAOp>, AIEOpRemoval<ShimMuxOp>,
        AIEOpRemoval<SwitchboxOp>, AIEOpRemoval<LockOp>, AIEOpRemoval<BufferOp>,
        AIEOpRemoval<ExternalBufferOp>, AIEOpRemoval<ShimDMAAllocationOp>,
        AIEOpRemoval<CascadeFlowOp>, AIEOpRemoval<ConfigureCascadeOp>>(
        m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(removepatterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> AIE::createAIECoreToStandardPass() {
  return std::make_unique<AIECoreToStandardPass>();
}
