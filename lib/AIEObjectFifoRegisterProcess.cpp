//===- AIEObjectFifoRegisterProcess.cpp --------------------------*- MLIR
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: October 18th 2021
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "aie/AIETokenAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-register-objectFifos"

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//
struct RemoveAIERegisterProcess
    : public OpConversionPattern<ObjectFifoRegisterProcessOp> {
  using OpConversionPattern<ObjectFifoRegisterProcessOp>::OpConversionPattern;
  ModuleOp &module;

  RemoveAIERegisterProcess(MLIRContext *context, ModuleOp &m,
                           PatternBenefit benefit = 1)
      : OpConversionPattern<ObjectFifoRegisterProcessOp>(context, benefit),
        module(m) {}

  LogicalResult
  matchAndRewrite(ObjectFifoRegisterProcessOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    rewriter.eraseOp(Op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Register objectFifos Pass
//===----------------------------------------------------------------------===//
struct AIEObjectFifoRegisterProcessPass
    : public AIEObjectFifoRegisterProcessBase<
          AIEObjectFifoRegisterProcessPass> {

  mlir::scf::ForOp createForLoop(OpBuilder &builder, int length) {
    arith::ConstantOp lowerBound = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getIndexAttr(0),
        builder.getIndexType());
    arith::ConstantOp upperBound = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getIndexAttr(length),
        builder.getIndexType());
    arith::ConstantOp step = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getIndexAttr(1),
        builder.getIndexType());
    mlir::scf::ForOp forLoop = builder.create<mlir::scf::ForOp>(
        builder.getUnknownLoc(), lowerBound, upperBound, step);
    return forLoop;
  }

  void createPattern(OpBuilder &builder, ModuleOp &m,
                     ObjectFifoRegisterProcessOp regOp, mlir::Type elementType,
                     IntegerAttr acqNumber, IntegerAttr relNumber, int length) {
    // create for loop
    mlir::scf::ForOp forLoop;
    if (length > 1) {
      forLoop = createForLoop(builder, length);
      Region &forRegion = forLoop.getRegion();
      builder.setInsertionPointToStart(&forRegion.back());
    }

    // acquires
    if (acqNumber.getInt() > 0) {
      auto acqType = AIEObjectFifoSubviewType::get(elementType);
      auto acqOp = builder.create<ObjectFifoAcquireOp>(
          builder.getUnknownLoc(), acqType, regOp.getPortAttr(), regOp.fifo(),
          acqNumber);

      // subview accesses
      ObjectFifoSubviewAccessOp acc;
      for (int i = 0; i < acqNumber.getInt(); i++) {
        acc = builder.create<ObjectFifoSubviewAccessOp>(
            builder.getUnknownLoc(), elementType, acqOp.subview(),
            builder.getIntegerAttr(builder.getI32Type(), i));
      }

      // apply kernel
      func::FuncOp func;
      for (auto funcOp : m.getOps<func::FuncOp>()) {
        if (funcOp.getSymName() == regOp.callee()) {
          func = funcOp;
          break;
        }
      }
      builder.create<func::CallOp>(builder.getUnknownLoc(),
                                   func /*, acc.output()*/);
    }

    // releases
    if (relNumber.getInt() > 0) {
      auto relOp = builder.create<ObjectFifoReleaseOp>(builder.getUnknownLoc(),
                                                       regOp.getPortAttr(),
                                                       regOp.fifo(), relNumber);
      builder.setInsertionPointAfter(relOp);
    }

    if (length > 1)
      builder.setInsertionPointAfter(forLoop);
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());
    //===----------------------------------------------------------------------===//
    // Generate access patterns
    //===----------------------------------------------------------------------===//
    for (auto registerOp : m.getOps<ObjectFifoRegisterProcessOp>()) {
      builder.setInsertionPointToEnd(m.getBody());
      ObjectFifoCreateOp objFifo =
          registerOp.fifo().getDefiningOp<ObjectFifoCreateOp>();
      auto elementType =
          objFifo.getType().dyn_cast<AIEObjectFifoType>().getElementType();

      // identify tile on which to generate the pattern
      TileOp tile;
      if (registerOp.port() == ObjectFifoPort::Produce) {
        tile = objFifo.getProducerTileOp();
      } else if (registerOp.port() == ObjectFifoPort::Consume) {
        tile = objFifo.getConsumerTileOp();
      }

      // retrieve core associated to above tile or create new one
      CoreOp *core = nullptr;
      for (auto coreOp : m.getOps<CoreOp>()) {
        if ((coreOp.tile().getDefiningOp<TileOp>()) == tile) {
          core = &coreOp;
          break;
        }
      }
      if (core == nullptr) {
        CoreOp coreOp = builder.create<CoreOp>(builder.getUnknownLoc(),
                                               builder.getIndexType(), tile);
        Region &r = coreOp.body();
        r.push_back(new Block);
        Block &block = r.back();
        builder.setInsertionPointToStart(&block);
        builder.create<EndOp>(builder.getUnknownLoc());
        core = &coreOp;
      }
      Region &r = core->body();
      Block &endBlock = r.back();
      builder.setInsertionPointToStart(&endBlock);

      // analyze pattern
      auto acqSize = registerOp.getAcquirePattern().size();
      auto relSize = registerOp.getReleasePattern().size();

      if (acqSize == 1 && relSize == 1) {
        IntegerAttr acqNumber =
            registerOp.getAcquirePattern().getValues<IntegerAttr>()[0];
        IntegerAttr relNumber =
            registerOp.getReleasePattern().getValues<IntegerAttr>()[0];
        createPattern(builder, m, registerOp, elementType, acqNumber, relNumber,
                      registerOp.getProcessLength());

      } else {
        auto acqPattern =
            registerOp.getAcquirePattern().getValues<IntegerAttr>();
        std::vector<IntegerAttr> acqVector;
        for (auto i = acqPattern.begin(); i != acqPattern.end(); ++i) {
          acqVector.push_back(*i);
        }

        auto relPattern =
            registerOp.getReleasePattern().getValues<IntegerAttr>();
        std::vector<IntegerAttr> relVector;
        for (auto i = relPattern.begin(); i != relPattern.end(); ++i) {
          relVector.push_back(*i);
        }

        if (acqSize == 1) {
          // duplicate acquire pattern
          IntegerAttr acqNumber =
              registerOp.getAcquirePattern().getValues<IntegerAttr>()[0];
          std::vector<IntegerAttr> values(registerOp.getProcessLength(),
                                          acqNumber);
          acqVector = values;
          acqSize = registerOp.getProcessLength();

        } else if (relSize == 1) {
          // duplicate release pattern
          IntegerAttr relNumber =
              registerOp.getReleasePattern().getValues<IntegerAttr>()[0];
          std::vector<IntegerAttr> values(registerOp.getProcessLength(),
                                          relNumber);
          relVector = values;
          relSize = registerOp.getProcessLength();
        }

        int length = 1;
        for (int i = 0; i < acqSize; i++) {
          auto currAcq = acqVector[i];
          auto currRel = relVector[i];
          if (i < acqSize - 1) {
            auto nextAcq = acqVector[i + 1];
            auto nextRel = relVector[i + 1];

            if ((currAcq.getInt() == nextAcq.getInt()) &&
                (currRel.getInt() == nextRel.getInt())) {
              length++;
              continue;
            }
          }
          createPattern(builder, m, registerOp, elementType, currAcq, currRel,
                        length);
          length = 1;
        }
      }
    }

    //===----------------------------------------------------------------------===//
    // Remove old ops
    //===----------------------------------------------------------------------===//
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    patterns.insert<RemoveAIERegisterProcess>(m.getContext(), m);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEObjectFifoRegisterProcessPass() {
  return std::make_unique<AIEObjectFifoRegisterProcessPass>();
}