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

struct AIECreateCoreModulePass : public PassWrapper<AIECreateCoreModulePass,
  OperationPass<ModuleOp>> {

  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder(m.getBody()->getTerminator());

    DenseMap<std::pair<int, int>, CoreOp> cores;
    DenseMap<std::pair<int, int>, MemOp> mems;
    DenseMap<FuncOp, std::pair<int, int>> funcs;

    for (auto core : m.getOps<CoreOp>()) {
      int colIndex = core.colIndex();
      int rowIndex = core.rowIndex();
      cores[std::make_pair(colIndex, rowIndex)] = core;
    }

    // Bind FuncOp to an AIE core based on attributes of the CallOp
    // A CoreModuleOp will be created for the core, and the FuncOp body is cloned
    // to the CoreModuleOp region
    for (auto callOp : m.getOps<CallOp>()) {
      if (!callOp.getAttr("aie.x") || !callOp.getAttr("aie.y"))
        continue;

      SmallVector<Value, 4> callOperands(callOp.getArgOperands());
      SmallVector<std::pair<MemRefType, int>, 4> coreModuleBufTypes;

      int colIndex = callOp.getAttrOfType<IntegerAttr>("aie.x").getInt();
      int rowIndex = callOp.getAttrOfType<IntegerAttr>("aie.y").getInt();

      if (!cores[std::make_pair(colIndex, rowIndex)]) {
        builder.setInsertionPointToStart(m.getBody());
        CoreOp core = builder.create<CoreOp>(builder.getUnknownLoc(), builder.getIndexType(),
                                             colIndex, rowIndex);
        cores[std::make_pair(colIndex, rowIndex)] = core;
      }

      if (!mems[std::make_pair(colIndex, rowIndex)]) {
        builder.setInsertionPointToStart(m.getBody());
        MemOp mem = builder.create<MemOp>(builder.getUnknownLoc(), builder.getIndexType(),
                                          colIndex, rowIndex);
        mem.ensureTerminator(mem.body(), builder, builder.getUnknownLoc());
        builder.setInsertionPointToStart(&mem.body().front());
        unsigned bufferID = 0;
        for (auto operand : callOperands) {
          MemRefType t = nullptr;
          if (operand.getType().isa<MemRefType>()) {
            t = operand.getType().cast<MemRefType>();
          } else if (operand.getType().isIntOrFloat()) {
            // promote scalar type to memref type
            int64_t shape[1] = {1};
            t = MemRefType::get(shape, operand.getType());
          }

          assert(t && "Unsupported type!");

          auto allocOp = builder.create<AllocOp>(builder.getUnknownLoc(), t);
          allocOp.setAttr("id", builder.getI32IntegerAttr(bufferID));
          coreModuleBufTypes.push_back(std::make_pair(t, bufferID));
          bufferID++;
        }

        mems[std::make_pair(colIndex, rowIndex)] = mem;
      }

      if (CallOpInterface call = dyn_cast<CallOpInterface>(callOp.getOperation())) {
        Operation *callable = call.resolveCallable();
        if (FuncOp func = dyn_cast<FuncOp>(callable)) {
          funcs[func] = std::make_pair(colIndex, rowIndex);

          BlockAndValueMapping mapper;
          SmallVector<Value, 4> operands;
          operands.push_back(cores[std::make_pair(colIndex, rowIndex)]);
          operands.push_back(mems[std::make_pair(colIndex, rowIndex)]);

          builder.setInsertionPoint(callOp);
          CoreModuleOp coreModule = builder.create<CoreModuleOp>(builder.getUnknownLoc(), operands);
          coreModule.ensureTerminator(coreModule.body(), builder, builder.getUnknownLoc());
          Region &r = coreModule.body();
          Block &b = r.front();
          builder.setInsertionPointToStart(&b);

          // Mapping between function arguments (FuncOp) and AIE buffers (CoreModuleOp)
          // We will create one buffer for each function argument
          // If the function argument's type is a scalar, we promote it to a one-element memref,
          // and do a load to the buffer at index 0
          for (auto pair : coreModuleBufTypes) {
            MemRefType t = pair.first;
            int bufID = pair.second;
            BufferOp buf = builder.create<BufferOp>(builder.getUnknownLoc(), t,
                                                    mems[std::make_pair(colIndex, rowIndex)],
                                                    bufID);
            Value arg = func.getArgument(bufID);
            if (arg.getType().isIntOrFloat()) {
              assert(t.getShape().size() == 1 && "Expected MemRefType of shape 1");
              assert(t.getShape()[0] == 1 && "Expected MemRefType of single element");

              Value zero = builder.create<ConstantIndexOp>(builder.getUnknownLoc(), 0);
              auto loadOp = builder.create<LoadOp>(builder.getUnknownLoc(), arg.getType(), buf, zero);
              mapper.map(arg, loadOp);
            } else
              mapper.map(arg, buf);
          }

          // Clone ops from the original function to CoreModuleOp's body
          for (auto &childOp : func.getCallableRegion()->getOps()) {
            // skip ReturnOp since it lives only within a funcOp
            if (auto returnOp = dyn_cast<ReturnOp>(childOp))
              continue;

            builder.clone(childOp, mapper);
          }
        }
      }
    }

    ConversionTarget target(getContext());
    OwningRewritePatternList patterns;

    // Remove standard CallOps and FuncOps that are bound to AIE CoreModuleOps
    patterns.insert<RemoveAIECalls>(m.getContext(), m);
    patterns.insert<RemoveAIEFuncs>(m.getContext(), m, funcs);

    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};

void xilinx::AIE::registerAIECreateCoreModulePass() {
    PassRegistration<AIECreateCoreModulePass>(
      "aie-create-coremodule",
      "Lower FuncOp from Standard dialect to CoreModuleOp of AIE dialect");
}
