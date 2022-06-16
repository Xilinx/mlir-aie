//===- AIEAssignBuffers.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-logical-locks"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIECreateLogicalLocksPass : public AIECreateLogicalLocksBase<AIECreateLogicalLocksPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder rewriter = OpBuilder::atBlockEnd(m.getBody());

    typedef std::pair<Operation*, int> LockOpID;
    std::vector<LockOpID> unique_tiles;

    //first pass
    for (auto lock : m.getOps<LockOp>()) {
      Operation *lock_tile = lock.tile().getDefiningOp();
      
      bool in_list = false;
      for (auto &unique_tile : unique_tiles) {
        if (unique_tile.first == lock_tile) {
          in_list = true;
          break;
        }
      }
      if (!in_list) {
        unique_tiles.push_back(std::make_pair(lock_tile, 0));
      }
    }
    //second pass
    for (auto lock : m.getOps<LockOp>()) {
      Operation *lock_tile = lock.tile().getDefiningOp();
      if (lock.getLockID() == -1) {
        for (auto &unique_tile : unique_tiles) {
          if (unique_tile.first == lock_tile) {
            lock->setAttr("lockID", rewriter.getI32IntegerAttr(unique_tile.second));
            unique_tile.second++;
            break;
          }
        }
      }
    }
};

std::unique_ptr<OperationPass<ModuleOp>> xilinx::AIE::createAIECreateLogicalLocksPass() {
  return std::make_unique<AIECreateLogicalLocksPass>();
}
