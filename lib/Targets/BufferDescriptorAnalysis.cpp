//===- BufferDescriptorAnalysis.cpp -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Analysis/BufferDescriptorAnalysis.h"
#include "aie/Dialect/ADF/ADFDialect.h"
#include "aie/Dialect/ADF/ADFOps.h"
#include "mlir/Analysis/SliceAnalysis.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include <iostream>

#define DEBUG_TYPE "buffer-descriptor-analysis"

using namespace mlir;

namespace xilinx {
namespace AIE {

void BufferDescriptorState::print(raw_ostream &os) const {
  auto print = [&](StringRef name, SmallVector<OpFoldResult> vec) {
    os << name << " = [";
    llvm::interleaveComma(vec, os);
    os << "]";
  };
  os << "BD(";
  print("length", length);
  os << ", base = " << base;
  print(", steps", steps);
  print(", wraps", wraps);
  if (repetition)
    os << ", repetition = " << *repetition;
  os << ")\n";
}

void BufferDescriptorState::printInt(raw_ostream &os) const {
  auto print = [&](StringRef name, SmallVector<int64_t, 4> vec) {
    os << name << " = [";
    llvm::interleaveComma(vec, os);
    os << "]";
  };
  os << "BD(";
  print("length", lengthInt);
  os << ", totalLength = " << this->getTotalLengthInt();
  os << ", base = " << base;
  print(", steps", stepsInt);
  print(", wraps", wrapsInt);
  if (repetition)
    os << ", repetition = " << *repetition;
  os << ")\n";
}

void BufferDescriptorAnalysis::visitOperandReintCast(
    memref::ReinterpretCastOp reintCastOp, BufferDescriptorState &state) {
  reintCastOp.getOperation()->emitWarning("visitOperandReintCast");

  auto srcVal = reintCastOp.getSource();
  LLVM_DEBUG(llvm::dbgs() << "reintCastOp source: " << srcVal << "\n");
  state.source = srcVal;

  auto offsets = reintCastOp.getMixedOffsets();
  auto sizes = reintCastOp.getMixedSizes();
  auto strides = reintCastOp.getMixedStrides();
  auto dstShape =
      reintCastOp.getResult().getType().cast<ShapedType>().getShape();
  int64_t rank = dstShape.size();

  assert(rank == (int64_t)sizes.size());
  assert(rank == (int64_t)strides.size());
  assert(1 == offsets.size());

  // length
  for (auto size : sizes) {
    state.length.push_back(size);
    auto cstSize = getConstantIntValue(size);
    if (cstSize) {
      state.lengthInt.push_back(*cstSize);
    }
  }

  // base
  state.base = offsets[0];

  // steps
  for (auto stride : llvm::reverse(strides)) {
    state.steps.push_back(stride);
    auto cstStride = getConstantIntValue(stride);
    if (cstStride) {
      state.stepsInt.push_back(*cstStride);
    }
  }

  // wraps
  for (auto size : llvm::reverse(sizes)) {
    state.wraps.push_back(size);
    auto cstSize = getConstantIntValue(size);
    if (cstSize) {
      state.wrapsInt.push_back(*cstSize);
    }
  }

  // slice analysis from the offset
  // TODO: need `scalar evolution` to handle more situations when getting the
  // const step
  if (auto parentOp = reintCastOp.getOperation()->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      SetVector<Operation *> bwdSlices;
      auto isAddI = [](Operation *op) { return isa<arith::AddIOp>(op); };
      auto offsetVal = offsets[0].dyn_cast<Value>();

      // TODO: the filter option need to support more operations and also has to
      // check that the operations are within the for loop
      // TODO: has to verify that the base trace back to the same operators for
      // analyzing constant step
      getBackwardSlice(offsetVal, &bwdSlices, isAddI);
      bwdSlices.insert(offsetVal.getDefiningOp());

      LLVM_DEBUG(llvm::dbgs() << "backwardSlice from current offsetVal = [");
      LLVM_DEBUG(llvm::interleaveComma(bwdSlices, llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "]\n");

      int64_t totalStepPerForLoopIteration = 0;
      for (auto op : bwdSlices) {
        if (auto addIOp = dyn_cast<arith::AddIOp>(op)) {
          LLVM_DEBUG(llvm::dbgs() << "backwardSlice op: " << addIOp << "\n");

          for (auto operand : op->getOperands()) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  operand from addIOp: " << operand << "\n");
            if (bwdSlices.contains(operand.getDefiningOp()))
              continue;

            arith::ConstantOp constantOp =
                operand.getDefiningOp<arith::ConstantOp>();

            if (BlockArgument blockArg = dyn_cast<BlockArgument>(operand)) {
              auto opOperand =
                  forOp.getOpOperandForRegionIterArg(blockArg).get();
              LLVM_DEBUG(llvm::dbgs()
                         << "  is a BlockArgument, the initial operand is: "
                         << opOperand << "\n");
              constantOp = opOperand.getDefiningOp<arith::ConstantOp>();
            }
            // TODO:: else if, for-loop induction variable

            if (constantOp) {
              totalStepPerForLoopIteration += constantOp.getValue()
                                                  .cast<IntegerAttr>()
                                                  .getValue()
                                                  .getSExtValue();
            }
          }
        }
      }
      LLVM_DEBUG(llvm::dbgs() << "totalStepPerForLoopIteration = "
                              << totalStepPerForLoopIteration << "\n");
      state.constantStep = totalStepPerForLoopIteration;
    }
  }
}

void BufferDescriptorAnalysis::visitOperandSubView(
    memref::SubViewOp subViewOp, BufferDescriptorState &state) {
  subViewOp.getOperation()->emitWarning("visitOperandSubView");

  auto srcVal = subViewOp.getSource();
  // TODO: handle the masking size from subview op
  visitOperand(srcVal, state);
}

void BufferDescriptorAnalysis::visitOperandCopy(memref::CopyOp copyOp,
                                                BufferDescriptorState &state) {
  copyOp.getOperation()->emitWarning("visitOperandCopy");

  auto srcVal = copyOp.getSource();
  if (BlockArgument blockArg = dyn_cast<BlockArgument>(srcVal)) {
    LLVM_DEBUG(llvm::dbgs() << "source of memref.copy is block argument\n");

    // get the two BD states from the two reintcast
    BufferDescriptorState initialState;
    BufferDescriptorState nextIteState;

    auto parentOp = copyOp.getOperation()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      // get the initial value for the for-loop block argument
      auto initVal = forOp.getOpOperandForRegionIterArg(blockArg).get();
      LLVM_DEBUG(llvm::dbgs() << "  initial value is: " << initVal << "\n");
      visitOperand(initVal, initialState);

      // get the yield value for the for-loop block argument
      Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
      Region &forRegion = forOp.getRegion();
      Operation *yieldOp = forRegion.getBlocks().front().getTerminator();
      for (auto pair : llvm::zip(iterArgs, yieldOp->getOperands())) {
        if (std::get<0>(pair) == blockArg) {
          auto yieldVal = std::get<1>(pair);
          LLVM_DEBUG(llvm::dbgs() << "  yield value is: " << yieldVal << "\n");
          visitOperand(yieldVal, nextIteState);
          break;
        }
      }

      // merge the two BDs and propagate to current state
      LLVM_DEBUG(initialState.print(llvm::dbgs()));
      LLVM_DEBUG(initialState.printInt(llvm::dbgs()));
      LLVM_DEBUG(nextIteState.print(llvm::dbgs()));
      LLVM_DEBUG(nextIteState.printInt(llvm::dbgs()));

      // verify sizes/ranks and the source pointer
      assert(initialState.lengthInt.size() == nextIteState.lengthInt.size());
      assert(initialState.stepsInt.size() == nextIteState.stepsInt.size());
      assert(initialState.wrapsInt.size() == nextIteState.wrapsInt.size());
      assert(initialState.length.size() == nextIteState.length.size());
      assert(initialState.steps.size() == nextIteState.steps.size());
      assert(initialState.wraps.size() == nextIteState.wraps.size());
      assert(initialState.source == nextIteState.source);

      // Need deep copy for states and verify that they are compatible
      // TODO: check repetition?
      state.base = initialState.base;
      for (auto [i, s] : llvm::enumerate(initialState.lengthInt)) {
        if (s == nextIteState.lengthInt[i]) {
          state.lengthInt.push_back(s);
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "lengthInt[" << i << "] are differenet, " << s << " vs "
                     << nextIteState.lengthInt[i] << "\n");
        }
      }
      for (auto [i, s] : llvm::enumerate(initialState.stepsInt)) {
        if (s == nextIteState.stepsInt[i]) {
          state.stepsInt.push_back(s);
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "stepsInt[" << i << "] are differenet, " << s << " vs "
                     << nextIteState.stepsInt[i] << "\n");
        }
      }
      for (auto [i, s] : llvm::enumerate(initialState.wrapsInt)) {
        if (s == nextIteState.wrapsInt[i]) {
          state.wrapsInt.push_back(s);
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "wrapsInt[" << i << "] are differenet, " << s << " vs "
                     << nextIteState.wrapsInt[i] << "\n");
        }
      }
      for (auto [i, s] : llvm::enumerate(initialState.length)) {
        if (s == nextIteState.length[i]) {
          state.length.push_back(s);
        } else {
          LLVM_DEBUG(llvm::dbgs() << "length[" << i << "] are differenet, " << s
                                  << " vs " << nextIteState.length[i] << "\n");
        }
      }
      for (auto [i, s] : llvm::enumerate(initialState.steps)) {
        if (s == nextIteState.steps[i]) {
          state.steps.push_back(s);
        } else {
          LLVM_DEBUG(llvm::dbgs() << "steps[" << i << "] are differenet, " << s
                                  << " vs " << nextIteState.steps[i] << "\n");
        }
      }
      for (auto [i, s] : llvm::enumerate(initialState.wraps)) {
        if (s == nextIteState.wraps[i]) {
          state.wraps.push_back(s);
        } else {
          LLVM_DEBUG(llvm::dbgs() << "wraps[" << i << "] are differenet, " << s
                                  << " vs " << nextIteState.wraps[i] << "\n");
        }
      }

      // TODO: verify that the `base` can trace back to the same operation
      if (nextIteState.constantStep) {
        state.constantStep = *nextIteState.constantStep;
      }
    }
  } else {
    visitOperand(srcVal, state);
  }

  // get the loop tripcount if possible
  int64_t loopTripCount = 1;
  if (auto parentOp = copyOp.getOperation()->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      forOp->emitWarning("memref.copy in the scf::ForOp");

      // auto inductionVar = forOp.getInductionVar();
      auto lowerBound = forOp.getLowerBound();
      auto upperBound = forOp.getUpperBound();
      auto step = forOp.getStep();

      if (isa<arith::ConstantOp>(lowerBound.getDefiningOp()) &&
          isa<arith::ConstantOp>(upperBound.getDefiningOp()) &&
          isa<arith::ConstantOp>(step.getDefiningOp())) {
        int64_t lowerBoundVal = lowerBound.getDefiningOp<arith::ConstantOp>()
                                    .getValue()
                                    .cast<IntegerAttr>()
                                    .getValue()
                                    .getSExtValue();
        int64_t upperBoundVal = upperBound.getDefiningOp<arith::ConstantOp>()
                                    .getValue()
                                    .cast<IntegerAttr>()
                                    .getValue()
                                    .getSExtValue();
        int64_t stepVal = step.getDefiningOp<arith::ConstantOp>()
                              .getValue()
                              .cast<IntegerAttr>()
                              .getValue()
                              .getSExtValue();

        loopTripCount =
            ((upperBoundVal - lowerBoundVal) + (stepVal - 1)) / stepVal;
      }
    }
  }

  if (loopTripCount > 1 && state.constantStep) {
    // update the BD state
    // 1. multiply the loop tripcount to the BD's length
    // 2. insert the constant step to BD's steps (constant step is analyzed in
    // visiting reintcast)
    LLVM_DEBUG(llvm::dbgs()
               << "loopTripCount = " << loopTripCount
               << ", state.constantStep = " << *state.constantStep << "\n");
    state.lengthInt.push_back(loopTripCount);
    state.stepsInt.push_back(*state.constantStep);

    // TODO: OpFoldResult version? get rid of OpFoldResult?
  } else {
    // pop the last wraps since there is no additional step from for loop
    if (!state.wraps.empty())
      state.wraps.pop_back();
    if (!state.wrapsInt.empty())
      state.wrapsInt.pop_back();
  }
}

void BufferDescriptorAnalysis::visitOperandTensorStore(
    memref::TensorStoreOp tensorStoreOp, BufferDescriptorState &state) {

  tensorStoreOp.getOperation()->emitWarning("visitOperandTensorStore");

  auto dstVal = tensorStoreOp.getMemref();
  visitOperand(dstVal, state);

  // pop the last wraps since there is no additional step from for loop
  if (!state.wraps.empty())
    state.wraps.pop_back();
  if (!state.wrapsInt.empty())
    state.wrapsInt.pop_back();
}

void BufferDescriptorAnalysis::visitOperandCast(memref::CastOp castOp,
                                                BufferDescriptorState &state) {

  castOp.getOperation()->emitWarning("visitOperandCast");

  auto srcVal = castOp.getSource();
  visitOperand(srcVal, state);
}

void BufferDescriptorAnalysis::visitOperand(Value operand,
                                            BufferDescriptorState &state) {
  if (auto op = operand.getDefiningOp<memref::ReinterpretCastOp>()) {
    visitOperandReintCast(op, state);
  } else if (auto op = operand.getDefiningOp<memref::SubViewOp>()) {
    visitOperandSubView(op, state);
  } else if (auto op = operand.getDefiningOp<memref::CopyOp>()) {
    visitOperandCopy(op, state);
  } else if (auto op = operand.getDefiningOp<memref::TensorStoreOp>()) {
    visitOperandTensorStore(op, state);
  } else if (auto op = operand.getDefiningOp<memref::CastOp>()) {
    visitOperandCast(op, state);
  } else {
    operand.getDefiningOp()->dump();
    LLVM_DEBUG(
        llvm::dbgs()
        << "encountered addptr operand produced by an unsupported operation\n");
    // llvm_unreachable("encountered addptr operand produced by an "
    //                  "unsupported operation");
  }
}

} // namespace AIE
} // namespace xilinx
