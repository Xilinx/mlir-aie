//===- AIESCFToControlFlow.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower scf to cf everywhere in the module EXCEPT inside aie.runtime_sequence
// bodies.
//
// A runtime sequence is NoTerminator and is translated to a flat NPU
// instruction stream with no concept of branches. Its loops are lowered by the
// dedicated runtime-sequence path (unrolling for compile-time-constant trip
// counts; the dynamic EmitC path otherwise), not by generic scf->cf. Applying
// convert-scf-to-cf to a runtime sequence would (1) break the NoTerminator
// block invariant -- the ops after a lowered loop land in a terminator-less
// block -- and (2) emit cf.br / cf.cond_br that the flat NPU emitter silently
// drops, a miscompile. So we mark scf inside a runtime sequence as legal
// (leave it alone) and convert everything else.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/APInt.h"

#include <cstdint>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIESCFTOCONTROLFLOW
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

constexpr uint64_t kEffectivelyInfiniteLoopBound = (uint64_t{1} << 32) - 1;

static bool isConstantZero(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
  return integer && integer.getValue().isZero();
}

static bool isConstantOne(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
  return integer && integer.getValue().isOne();
}

static bool hasEffectivelyInfiniteBound(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto integer =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
  if (!integer)
    return false;
  const llvm::APInt &bound = integer.getValue();
  return !bound.isNegative() && bound.getActiveBits() >= 32 &&
         bound.getLimitedValue() >= kEffectivelyInfiniteLoopBound;
}

static bool isCounterFreeForeverLoopCandidate(scf::ForOp forOp) {
  return forOp.getInitArgs().empty() && forOp.getInductionVar().use_empty() &&
         isConstantZero(forOp.getLowerBound()) &&
         isConstantOne(forOp.getStep()) &&
         hasEffectivelyInfiniteBound(forOp.getUpperBound());
}

static void replaceWithCounterFreeLoop(scf::ForOp forOp) {
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  Block *forBody = forOp.getBody();
  auto whileOp = builder.create<scf::WhileOp>(
      loc, TypeRange{}, ValueRange{},
      [&](OpBuilder &beforeBuilder, Location loc, ValueRange) {
        Value trueValue = beforeBuilder.create<arith::ConstantIntOp>(loc, 1, 1);
        beforeBuilder.create<scf::ConditionOp>(loc, trueValue, ValueRange{});
      },
      [&](OpBuilder &afterBuilder, Location loc, ValueRange) {
        afterBuilder.create<scf::YieldOp>(loc, ValueRange{});
      });
  Block *after = whileOp.getAfterBody();
  after->getOperations().splice(after->getTerminator()->getIterator(),
                                forBody->getOperations(), forBody->begin(),
                                forBody->getTerminator()->getIterator());
  forBody->eraseArgument(0);
  forOp.erase();
}

struct AIESCFToControlFlowPass
    : xilinx::AIEX::impl::AIESCFToControlFlowBase<AIESCFToControlFlowPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<scf::ForOp> foreverLoops;
    module.walk([&](scf::ForOp forOp) {
      if (forOp->getParentOfType<AIE::CoreOp>() &&
          isCounterFreeForeverLoopCandidate(forOp))
        foreverLoops.push_back(forOp);
    });
    for (scf::ForOp forOp : foreverLoops)
      replaceWithCounterFreeLoop(forOp);

    ConversionTarget target(getContext());
    // Everything is legal by default; only scf outside a runtime sequence is
    // illegal. (Matching upstream convert-scf-to-cf, which marks every other
    // op legal so the structural block-splitting can relocate them freely.)
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addLegalDialect<cf::ControlFlowDialect>();
    // scf is illegal (must be converted) UNLESS it lives inside a runtime
    // sequence, whose control flow has its own lowering path.
    target.addDynamicallyLegalDialect<scf::SCFDialect>([](Operation *op) {
      return op->getParentOfType<AIE::RuntimeSequenceOp>() != nullptr;
    });

    RewritePatternSet patterns(&getContext());
    populateSCFToControlFlowConversionPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIESCFToControlFlowPass() {
  return std::make_unique<AIESCFToControlFlowPass>();
}
