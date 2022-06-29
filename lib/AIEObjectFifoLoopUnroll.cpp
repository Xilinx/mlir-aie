//===- AIEObjectFifoLoopUnroll.cpp --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: December 8th 2021
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "aie/AIETokenAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
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

#define DEBUG_TYPE "aie-unroll-objectFifos"

#define LOOP_VAR_DEPENDENCY -2

//===----------------------------------------------------------------------===//
// Unroll objectFifo loops pass
//===----------------------------------------------------------------------===//
struct AIEObjectFifoLoopUnrollPass
    : public AIEObjectFifoLoopUnrollBase<AIEObjectFifoLoopUnrollPass> {
  std::vector<TileOp> objectFifoTiles;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    for (auto objFifoOp : m.getOps<ObjectFifoCreateOp>()) {
      objectFifoTiles.push_back(objFifoOp.getProducerTileOp());
      objectFifoTiles.push_back(objFifoOp.getConsumerTileOp());
    }

    for (auto coreOp : m.getOps<CoreOp>()) {
      if (std::find(objectFifoTiles.begin(), objectFifoTiles.end(),
                    coreOp.getTileOp()) != objectFifoTiles.end()) {

        coreOp.walk([&](mlir::scf::ForOp forLoop) {
          // look for operations on objectFifos
          // TODO: when multiple fifos in same loop, must use the smallest
          // common multiplier as the unroll factor
          bool found = false;
          int objFifoSize = 0;
          Block *body = forLoop.getBody();

          for (auto acqOp : body->getOps<ObjectFifoAcquireOp>()) {
            if (acqOp.getOperation()->getParentOp() == forLoop) {
              found = true;
              ObjectFifoCreateOp op =
                  acqOp.fifo().getDefiningOp<ObjectFifoCreateOp>();
              objFifoSize = op.size();
            } 
          }

          for (auto relOp : body->getOps<ObjectFifoReleaseOp>()) {
            if (relOp.getOperation()->getParentOp() == forLoop) {
              found = true;
              ObjectFifoCreateOp op =
                  relOp.fifo().getDefiningOp<ObjectFifoCreateOp>();
              objFifoSize = op.size();
            }
          }

          if (found) {
            std::vector<Operation *>
                operations; // operations in original loop body, without
                            // terminator operation
            DenseMap<Operation *, int>
                opIndex; // maps operations of original loop body to their
                         // position in it
            std::vector<std::vector<int>>
                dependecies; // index in first vecotr corresponds to position in
                             // original loop body dependency vector has size
                             // equal to number of operands of that operation:
                             //    * if LOOP_VAR_DEPENDENCY : operand is
                             //    dependent on loop induction variable
                             //    * if -1 : operand is not dependent on any
                             //    operation in loop body
                             //    * if >=0: operand is dependent on operation
                             //    with that index in original loop body

            // find new loop size and step
            auto old_upper_bound = forLoop.getUpperBound()
                                       .getDefiningOp<arith::ConstantOp>()
                                       .getValue();
            int64_t old_upper_value =
                old_upper_bound.dyn_cast<IntegerAttr>().getInt();
            auto old_lower_bound = forLoop.getLowerBound()
                                       .getDefiningOp<arith::ConstantOp>()
                                       .getValue();
            int64_t old_lower_value =
                old_lower_bound.dyn_cast<IntegerAttr>().getInt();
            int64_t remainder = 0;

            builder.setInsertionPoint(forLoop);
            if ((old_upper_value - old_lower_value) % (int64_t)objFifoSize >
                0) {
              int64_t new_upper_bound =
                  ((old_upper_value - old_lower_value) / (int64_t)objFifoSize) *
                      (int64_t)objFifoSize +
                  1; // +1 because upper bound is excluded
              remainder =
                  (old_upper_value - old_lower_value) % (int64_t)objFifoSize;
              arith::ConstantOp uBound = builder.create<arith::ConstantOp>(
                  builder.getUnknownLoc(),
                  builder.getIndexAttr(new_upper_bound),
                  old_upper_bound.getType());
              forLoop.setUpperBound(uBound);
            }
            arith::ConstantOp new_step = builder.create<arith::ConstantOp>(
                builder.getUnknownLoc(),
                builder.getIndexAttr((int64_t)objFifoSize),
                old_upper_bound.getType());
            forLoop.setStep(new_step);

            // record original loop body (without terminator operation) and
            // identify dependencies
            auto withoutTerminator = --body->end();
            int index = 0;
            for (auto op = body->begin(); op != withoutTerminator; op++) {
              operations.push_back(&(*op));
              opIndex[&(*op)] = index;

              // identify dependencies
              auto numOperands = (&(*op))->getNumOperands();
              std::vector<int> dependecyIndices;
              for (int i = 0; i < numOperands; i++) {
                auto operand = (&(*op))->getOperand(i);
                int dependencyIndex = -1;

                // if operand not iterator variable
                if (operand == forLoop.getInductionVar()) {
                  dependencyIndex = LOOP_VAR_DEPENDENCY;
                } else {
                  auto definingOp = operand.getDefiningOp();
                  if (definingOp->getBlock()->getParentOp() == forLoop) {
                    dependencyIndex = opIndex[definingOp];
                  }
                }
                dependecyIndices.push_back(dependencyIndex);
              }
              dependecies.push_back(dependecyIndices);

              index++;
            }

            // duplicate loop body, insert before terminator operation
            // TODO: if fewer loop iterations than objFifo elements, remove
            // forLoop entirely?
            int numDuplications = (objFifoSize - 1) + remainder;
            int originalIndex = 0;
            std::vector<Operation *>
                duplicatedOperations; // operations in current duplication
                                      // iteration
            builder.setInsertionPoint(&(body->back()));
            for (int i = 0; i < numDuplications; i++) {
              // duplicate remaining iterations after loop
              if (i == (objFifoSize - 1))
                builder.setInsertionPointAfter(forLoop);

              originalIndex = 0;
              duplicatedOperations.clear();
              for (auto op : operations) {
                // for each operand, check whether there was a dependecy
                auto clone = op->clone();
                auto numOperands = clone->getNumOperands();
                for (int operandIndex = 0; operandIndex < numOperands;
                     operandIndex++) {
                  int originalDependencyIndex =
                      dependecies[originalIndex][operandIndex];
                  if (originalDependencyIndex >= 0) {
                    // replace the operand with the result of operation with
                    // same index in current duplication
                    clone->setOperand(
                        operandIndex,
                        duplicatedOperations[originalDependencyIndex]
                            ->getResult(0)); // TODO: what if operation has
                                             // multiple results?
                  } else if (originalDependencyIndex == LOOP_VAR_DEPENDENCY) {
                    if (i >= (objFifoSize - 1)) {
                      // special case when duplicating remaining iterations
                      // after loop
                      arith::ConstantOp increment =
                          builder.create<arith::ConstantOp>(
                              builder.getUnknownLoc(),
                              builder.getIndexAttr(i - (objFifoSize - 1)),
                              builder.getIndexType());
                      arith::AddIOp sum = builder.create<arith::AddIOp>(
                          builder.getUnknownLoc(), builder.getIndexType(),
                          forLoop.getUpperBound(), increment->getResult(0));
                      clone->setOperand(operandIndex, sum->getResult(0));
                    } else {
                      arith::ConstantOp increment =
                          builder.create<arith::ConstantOp>(
                              builder.getUnknownLoc(),
                              builder.getIndexAttr(i + 1),
                              builder.getIndexType());
                      arith::AddIOp sum = builder.create<arith::AddIOp>(
                          builder.getUnknownLoc(), builder.getIndexType(),
                          forLoop.getInductionVar(), increment->getResult(0));
                      clone->setOperand(operandIndex, sum->getResult(0));
                    }
                  }
                }

                builder.insert(clone);
                duplicatedOperations.push_back(clone);
                originalIndex++;
              }
            }
          }
        });
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEObjectFifoLoopUnrollPass() {
  return std::make_unique<AIEObjectFifoLoopUnrollPass>();
}