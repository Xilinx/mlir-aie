//===- AIEObjectFifoRegisterProcess.cpp ------------------------*- MLIR -*-===//
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

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

#include <queue>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-register-objectFifos"

//===----------------------------------------------------------------------===//
// Register objectFifos Pass
//===----------------------------------------------------------------------===//
struct AIEObjectFifoRegisterProcessPass
    : AIEObjectFifoRegisterProcessBase<AIEObjectFifoRegisterProcessPass> {

  scf::ForOp createForLoop(OpBuilder &builder, int length) {
    auto lowerBound = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getIndexAttr(0));
    auto upperBound = builder.create<arith::ConstantOp>(
        builder.getUnknownLoc(), builder.getIndexAttr(length));
    auto step = builder.create<arith::ConstantOp>(builder.getUnknownLoc(),
                                                  builder.getIndexAttr(1));
    auto forLoop = builder.create<scf::ForOp>(builder.getUnknownLoc(),
                                              lowerBound, upperBound, step);
    return forLoop;
  }

  void createPattern(OpBuilder &builder, DeviceOp &device,
                     ObjectFifoRegisterProcessOp regOp, MemRefType elementType,
                     IntegerAttr acqNumber, IntegerAttr relNumber, int length) {
    auto ctx = device->getContext();
    // create for loop
    scf::ForOp forLoop;
    if (length > 1) {
      forLoop = createForLoop(builder, length);
      Region &forRegion = forLoop.getRegion();
      builder.setInsertionPointToStart(&forRegion.back());
    }

    // acquires
    if (acqNumber.getInt() > 0) {
      auto acqType = AIEObjectFifoSubviewType::get(elementType);
      auto acqOp = builder.create<ObjectFifoAcquireOp>(
          builder.getUnknownLoc(), acqType, regOp.getPortAttr(),
          SymbolRefAttr::get(ctx, regOp.getObjFifoName()), acqNumber);

      // subview accesses
      for (int i = 0; i < acqNumber.getInt(); i++)
        (void)builder.create<ObjectFifoSubviewAccessOp>(
            builder.getUnknownLoc(), elementType, acqOp.getSubview(),
            builder.getIntegerAttr(builder.getI32Type(), i));

      // apply kernel
      func::FuncOp func;
      for (auto funcOp : device.getOps<func::FuncOp>()) {
        if (funcOp.getSymName() == regOp.getCallee()) {
          func = funcOp;
          break;
        }
      }
      builder.create<func::CallOp>(builder.getUnknownLoc(),
                                   func /*, acc.output()*/);
    }

    // releases
    if (relNumber.getInt() > 0) {
      auto relOp = builder.create<ObjectFifoReleaseOp>(
          builder.getUnknownLoc(), regOp.getPortAttr(),
          SymbolRefAttr::get(ctx, regOp.getObjFifoName()), relNumber);
      builder.setInsertionPointAfter(relOp);
    }

    if (length > 1)
      builder.setInsertionPointAfter(forLoop);
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());
    DenseMap<ObjectFifoCreateOp, std::queue<Value>> consumersPerFifo;
    //===----------------------------------------------------------------------===//
    // Generate access patterns
    //===----------------------------------------------------------------------===//
    for (auto registerOp : device.getOps<ObjectFifoRegisterProcessOp>()) {
      builder.setInsertionPointToEnd(device.getBody());
      ObjectFifoCreateOp objFifo = registerOp.getObjectFifo();
      auto elementType =
          llvm::dyn_cast<AIEObjectFifoType>(objFifo.getElemType())
              .getElementType();

      if (consumersPerFifo.find(objFifo) == consumersPerFifo.end()) {
        std::queue<Value> consumers;
        for (auto consumerTile : objFifo.getConsumerTiles())
          consumers.push(consumerTile);
        consumersPerFifo[objFifo] = consumers;
      }

      // identify tile on which to generate the pattern
      Value tile;
      if (registerOp.getPort() == ObjectFifoPort::Produce)
        tile = objFifo.getProducerTile();
      else if (registerOp.getPort() == ObjectFifoPort::Consume) {
        assert(!consumersPerFifo[objFifo].empty() &&
               "No more available consumer tiles for process.");
        tile = consumersPerFifo[objFifo].front();
        consumersPerFifo[objFifo].pop();
      }

      Operation *op = llvm::find_singleton<Operation>(
          device.getOps<CoreOp>(),
          [&tile](CoreOp coreOp, bool) {
            return coreOp.getTile() == tile ? coreOp.getOperation() : nullptr;
          },
          /*AllowRepeats=*/false);
      // retrieve core associated to above tile or create new one
      if (!op) {
        CoreOp coreOp = builder.create<CoreOp>(builder.getUnknownLoc(),
                                               builder.getIndexType(), tile);
        Region &r = coreOp.getBody();
        r.push_back(new Block);
        Block &block = r.back();
        builder.setInsertionPointToStart(&block);
        builder.create<EndOp>(builder.getUnknownLoc());
        builder.setInsertionPointToStart(&block);
      } else {
        CoreOp coreOp = llvm::dyn_cast<CoreOp>(op);
        Region &r = coreOp.getBody();
        Block &endBlock = r.back();
        builder.setInsertionPointToStart(&endBlock);
      }

      // analyze pattern
      auto acqSize = registerOp.getAcquirePattern().size();

      if (auto relSize = registerOp.getReleasePattern().size();
          acqSize == 1 && relSize == 1) {
        IntegerAttr acqNumber =
            registerOp.getAcquirePattern().getValues<IntegerAttr>()[0];
        IntegerAttr relNumber =
            registerOp.getReleasePattern().getValues<IntegerAttr>()[0];
        createPattern(builder, device, registerOp, elementType, acqNumber,
                      relNumber, registerOp.getProcessLength());

      } else {
        auto acqPattern =
            registerOp.getAcquirePattern().getValues<IntegerAttr>();
        std::vector<IntegerAttr> acqVector;
        for (auto i = acqPattern.begin(); i != acqPattern.end(); ++i)
          acqVector.push_back(*i);

        auto relPattern =
            registerOp.getReleasePattern().getValues<IntegerAttr>();
        std::vector<IntegerAttr> relVector;
        for (auto i = relPattern.begin(); i != relPattern.end(); ++i)
          relVector.push_back(*i);

        if (acqSize == 1) {
          // duplicate acquire pattern
          IntegerAttr acqNumber =
              registerOp.getAcquirePattern().getValues<IntegerAttr>()[0];
          std::vector values(registerOp.getProcessLength(), acqNumber);
          acqVector = values;
          acqSize = registerOp.getProcessLength();

        } else if (relSize == 1) {
          // duplicate release pattern
          IntegerAttr relNumber =
              registerOp.getReleasePattern().getValues<IntegerAttr>()[0];
          std::vector values(registerOp.getProcessLength(), relNumber);
          relVector = values;
        }

        int length = 1;
        for (int i = 0; i < acqSize; i++) {
          auto currAcq = acqVector[i];
          auto currRel = relVector[i];
          if (i < acqSize - 1) {
            auto nextAcq = acqVector[i + 1];
            if (auto nextRel = relVector[i + 1];
                currAcq.getInt() == nextAcq.getInt() &&
                currRel.getInt() == nextRel.getInt()) {
              length++;
              continue;
            }
          }
          createPattern(builder, device, registerOp, elementType, currAcq,
                        currRel, length);
          length = 1;
        }
      }
    }

    //===----------------------------------------------------------------------===//
    // Remove old ops
    //===----------------------------------------------------------------------===//
    SmallVector<Operation *> opsToErase;
    device.walk([&](Operation *op) {
      if (isa<ObjectFifoRegisterProcessOp>(op))
        opsToErase.push_back(op);
    });
    for (auto op : opsToErase)
      op->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIEObjectFifoRegisterProcessPass() {
  return std::make_unique<AIEObjectFifoRegisterProcessPass>();
}
