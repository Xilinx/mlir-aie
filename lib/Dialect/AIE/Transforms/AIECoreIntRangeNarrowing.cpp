//===- AIECoreIntRangeNarrowing.cpp -----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The upstream arith-int-range-narrowing pass cannot nest under aie.core
// (cores are not IsolatedFromAbove), and running it over the whole device
// would also rewrite runtime sequences. So this pass applies the same patterns
// to each core and function body in turn.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIECOREINTRANGENARROWING
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-core-int-range-narrowing"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// Drops the solver state of erased ops, so an op allocated at the same address
// later does not inherit a stale range. Mirrors the listener the upstream pass
// uses.
class DataFlowListener : public RewriterBase::Listener {
public:
  DataFlowListener(DataFlowSolver &s) : s(s) {}

protected:
  void notifyOperationErased(Operation *op) override {
    s.eraseState(s.getProgramPointAfter(op));
    for (Value res : op->getResults())
      s.eraseState(res);
  }

  DataFlowSolver &s;
};

struct AIECoreIntRangeNarrowingPass
    : xilinx::AIE::impl::AIECoreIntRangeNarrowingBase<
          AIECoreIntRangeNarrowingPass> {
  using AIECoreIntRangeNarrowingBase::AIECoreIntRangeNarrowingBase;

  void runOnOperation() override {
    SmallVector<unsigned> bitwidths(bitwidthsSupported.begin(),
                                    bitwidthsSupported.end());
    if (bitwidths.empty())
      bitwidths.push_back(32);

    SmallVector<Operation *> bodies;
    getOperation().walk([&](Operation *op) {
      if (isa<CoreOp>(op))
        bodies.push_back(op);
      else if (auto func = dyn_cast<func::FuncOp>(op);
               func && !func.isExternal())
        bodies.push_back(op);
    });

    for (Operation *body : bodies)
      if (failed(narrow(body, bitwidths)))
        return signalPassFailure();
  }

  LogicalResult narrow(Operation *body, ArrayRef<unsigned> bitwidths) {
    MLIRContext *ctx = body->getContext();
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<dataflow::IntegerRangeAnalysis>();
    // A solver run on the core never visits the values defined above it (its
    // buffers, locks and tiles). Left uninitialized, they make any control
    // flow that depends on them look dead, such as a loop bounded by a buffer
    // load. Start them pessimistic, like entry block arguments.
    SetVector<Value> captured;
    getUsedValuesDefinedAbove(body->getRegions(), captured);
    for (Value v : captured) {
      (void)solver
          .getOrCreateState<dataflow::Lattice<dataflow::ConstantValue>>(v)
          ->join(dataflow::ConstantValue::getUnknownConstant());
      (void)solver.getOrCreateState<dataflow::IntegerValueRangeLattice>(v)
          ->join(IntegerValueRange::getMaxRange(v));
    }
    if (failed(solver.initializeAndRun(body)))
      return failure();

    DataFlowListener listener(solver);
    RewritePatternSet patterns(ctx);
    arith::populateIntRangeNarrowingPatterns(patterns, solver, bitwidths);
    arith::populateControlFlowValuesNarrowingPatterns(patterns, solver,
                                                      bitwidths);
    // A core is not IsolatedFromAbove, so hand the driver its ops rather than
    // its region. Collected in postorder, they are visited bottom-up, as in
    // the upstream pass: the cmpi pattern needs the ranges attached to its
    // original operands.
    SmallVector<Operation *> ops;
    body->getRegion(0).walk([&](Operation *op) { ops.push_back(op); });
    return applyOpPatternsGreedily(ops, std::move(patterns),
                                   GreedyRewriteConfig()
                                       .setScope(&body->getRegion(0))
                                       .setListener(&listener));
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIECoreIntRangeNarrowingPass() {
  return std::make_unique<AIECoreIntRangeNarrowingPass>();
}
