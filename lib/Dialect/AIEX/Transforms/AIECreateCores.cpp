//===- AIECreateCores.cpp ---------------------------------------*- C++ -*-===//
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
#include "mlir/Dialect/Arith/IR/Arith.h"
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

struct RemoveAIEFuncs : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  DenseMap<func::FuncOp, std::pair<int, int>> &funcs;

  RemoveAIEFuncs(MLIRContext *context,
                 DenseMap<func::FuncOp, std::pair<int, int>> &funcs,
                 PatternBenefit benefit = 1)
      : OpConversionPattern<func::FuncOp>(context, benefit), funcs(funcs) {}

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    if (funcs.find(op) == funcs.end())
      return failure();

    rewriter.eraseOp(Op);
    return success();
  }
};

struct RemoveAIECalls : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  RemoveAIECalls(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<func::CallOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    if (!op->getAttr("aie.x") || !op->getAttr("aie.y"))
      return failure();

    rewriter.eraseOp(Op);
    return success();
  }
};

struct AIECreateCoresPass : public AIECreateCoresBase<AIECreateCoresPass> {
  void runOnOperation() override {

    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

    DenseMap<std::pair<int, int>, Operation *> tiles;
    DenseMap<Operation *, CoreOp> cores;
    DenseMap<Operation *, MemOp> mems;
    DenseMap<Value, Value> buffers;
    DenseMap<func::FuncOp, std::pair<int, int>> funcs;

    // Collect existing TileOps
    for (auto tile : device.getOps<TileOp>()) {
      int colIndex = tile.colIndex();
      int rowIndex = tile.rowIndex();
      tiles[std::make_pair(colIndex, rowIndex)] = tile;
    }

    // Bind FuncOp to an AIE core based on attributes of the CallOp
    // A CoreOp will be created for the core, and the FuncOp body is cloned
    // to the CoreOp region
    for (auto callOp : device.getOps<func::CallOp>()) {
      if (!callOp->getAttr("aie.x") || !callOp->getAttr("aie.y"))
        continue;

      SmallVector<Value, 4> callOperands(callOp.getArgOperands());
      SmallVector<std::pair<MemRefType, int>, 4> coreBufTypes;

      int colIndex = callOp->getAttrOfType<IntegerAttr>("aie.x").getInt();
      int rowIndex = callOp->getAttrOfType<IntegerAttr>("aie.y").getInt();

      // get or create TileOp
      if (!tiles[std::make_pair(colIndex, rowIndex)]) {
        builder.setInsertionPointToStart(device.getBody());
        TileOp tile =
            builder.create<TileOp>(builder.getUnknownLoc(), colIndex, rowIndex);
        tiles[std::make_pair(colIndex, rowIndex)] = tile;
      }
      Operation *tileOp = tiles[std::make_pair(colIndex, rowIndex)];
      TileOp tile = dyn_cast<TileOp>(tileOp);
      builder.setInsertionPointAfter(tileOp);

      // create MemOp
      if (!mems[tileOp]) {
        for (unsigned i = 0; i < callOperands.size(); i++) {
          Value operand = callOperands[i]; // Should be produced by an AllocOp
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
          BufferOp buf =
              builder.create<BufferOp>(builder.getUnknownLoc(), t, tile);
          //          buf.setAttr("sym_name",
          //          builder.getStringAttr("test_name"));
          buffers[callOperands[i]] = buf;
          operand.replaceAllUsesWith(buf.getResult());
        }

        MemOp mem = builder.create<MemOp>(builder.getUnknownLoc(),
                                          builder.getIndexType(), tile);
        Region &r = mem.getBody();
        Block *endBlock = builder.createBlock(&r);

        // block terminator
        builder.setInsertionPointToStart(endBlock);
        builder.create<EndOp>(builder.getUnknownLoc());
        mems[tileOp] = mem;
      }

      // create CoreOp with buffer reference
      if (CallOpInterface call =
              dyn_cast<CallOpInterface>(callOp.getOperation())) {
        Operation *callable = call.resolveCallable();
        if (func::FuncOp func = dyn_cast<func::FuncOp>(callable)) {
          funcs[func] = std::make_pair(colIndex, rowIndex);

          IRMapping mapper;

          builder.setInsertionPoint(callOp);

          CoreOp core;
          Block *currentBlock;

          if (!cores[tileOp]) {
            core = builder.create<CoreOp>(builder.getUnknownLoc(),
                                          builder.getIndexType(), tile);
            Region &r = core.getBody();
            currentBlock = builder.createBlock(&r);
            builder.setInsertionPointToStart(currentBlock);
          } else {
            core = cores[tileOp];
            currentBlock = &core.getBody().back();
            builder.setInsertionPoint(currentBlock->getTerminator());
          }

          // Mapping between function arguments (FuncOp) and AIE buffers
          // (CoreOp) We will create one buffer for each function argument If
          // the function argument's type is a scalar, we promote it to a
          // one-element memref, and do a load to the buffer at index 0
          for (auto pair : coreBufTypes) {
            MemRefType t = pair.first;
            int operandID = pair.second;
            Value arg = func.getArgument(operandID);
            Value buf = buffers[callOperands[operandID]];
            if (arg.getType().isIntOrFloat()) {
              assert(t.getShape().size() == 1 &&
                     "Expected MemRefType of shape 1");
              assert(t.getShape()[0] == 1 &&
                     "Expected MemRefType of single element");

              Value zero = builder.create<arith::ConstantIndexOp>(
                  builder.getUnknownLoc(), 0);
              auto loadOp = builder.create<memref::LoadOp>(
                  builder.getUnknownLoc(), arg.getType(), buf, zero);
              mapper.map(arg, loadOp);
            } else {
              mapper.map(arg, buf);
            }
          }

          // Clone ops from the original function to CoreOp's body
          for (auto &childOp : func.getCallableRegion()->getOps()) {
            // skip ReturnOp since it lives only within a funcOp
            if (auto returnOp = dyn_cast<func::ReturnOp>(childOp))
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
    // Since memcpy moves data from one memory module to another, we use
    // WireBundle::DMA for both the source and the destination In addition, we
    // only have two DMA ports per each direction (MM2S/S2MM), and in a
    // circuit-switch mode, dest port/channel sharing is not possible.
    // Therefore, we will generate error if the number of logical flows
    // (streams) targeting the same destination (S2MM) is more than 2
    // DenseMap<Value, int> destChannel;
    // for (auto op : device.getOps<MemcpyOp>()) {
    //   builder.setInsertionPoint(op);
    //   TileOp srcTile = dyn_cast<TileOp>(op.srcTile().getDefiningOp());
    //   TileOp dstTile = dyn_cast<TileOp>(op.dstTile().getDefiningOp());
    //   // TODO: perhaps a better approach is to not assert here, but rather
    //   have a subsequent pass
    //   // that legally relocates the ports
    //   assert(destChannel[op.dstTile()] <= 2 &&
    //          "Could not allocate more than two dest. channel when creating
    //          FlowOp");
    //   // WireBundle[1] = DMA
    //   builder.create<FlowOp>(builder.getUnknownLoc(), srcTile, 1, 0, dstTile,
    //   1, destChannel[op.dstTile()]); destChannel[op.dstTile()]++;
    // }

    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    target.addLegalOp<DMAStartOp>();
    target.addLegalOp<DMABDOp>();
    target.addLegalOp<UseTokenOp>();
    target.addLegalOp<NextBDOp>();

    // Remove standard CallOps and FuncOps that are bound to AIE CoreOps
    patterns.insert<RemoveAIECalls>(device.getContext());
    patterns.insert<RemoveAIEFuncs>(device.getContext(), funcs);

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIECreateCoresPass() {
  return std::make_unique<AIECreateCoresPass>();
}
