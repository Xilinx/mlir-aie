//===- AIECoreForeverLoops.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites core loops that are counted but never finish in practice, such as
// IRON's `range_(sys.maxsize)`, into loops without a counter.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIECOREFOREVERLOOPS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-core-forever-loops"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

struct AIECoreForeverLoopsPass
    : xilinx::AIE::impl::AIECoreForeverLoopsBase<AIECoreForeverLoopsPass> {
  using AIECoreForeverLoopsBase::AIECoreForeverLoopsBase;

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](Operation *op) {
      if (isa<CoreOp>(op) ||
          (isa<func::FuncOp>(op) && !cast<func::FuncOp>(op).isExternal()))
        op->walk([&](scf::ForOp forOp) {
          if (isForever(forOp))
            loops.push_back(forOp);
        });
    });

    IRRewriter rewriter(&getContext());
    for (scf::ForOp forOp : loops)
      rewrite(rewriter, forOp);
  }

  bool isForever(scf::ForOp forOp) {
    if (!forOp.getInductionVar().use_empty())
      return false;
    std::optional<APInt> tripCount = forOp.getStaticTripCount();
    return tripCount && tripCount->uge(minTripCount);
  }

  // scf.for %i = ... iter_args(%a = %init) { body }
  // becomes
  // scf.while (%a = %init) { scf.condition(%true) %a } do { body }
  // where the body, including its scf.yield, moves over unchanged.
  void rewrite(IRRewriter &rewriter, scf::ForOp forOp) {
    rewriter.setInsertionPoint(forOp);
    auto whileOp = scf::WhileOp::create(
        rewriter, forOp.getLoc(), forOp.getResultTypes(), forOp.getInitArgs(),
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value isTrue = arith::ConstantIntOp::create(b, loc, 1, 1);
          scf::ConditionOp::create(b, loc, isTrue, args);
        },
        nullptr);

    // The induction variable is unused, so any index value can stand in.
    SmallVector<Value> bodyArgs{forOp.getLowerBound()};
    llvm::append_range(bodyArgs, whileOp.getAfterArguments());
    rewriter.mergeBlocks(forOp.getBody(), whileOp.getAfterBody(), bodyArgs);
    rewriter.replaceOp(forOp, whileOp.getResults());
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIECoreForeverLoopsPass() {
  return std::make_unique<AIECoreForeverLoopsPass>();
}
