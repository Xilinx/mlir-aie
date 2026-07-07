//===- AIERuntimeBdLiveness.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIERuntimeBdLiveness.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

// One step of the trace: classify all uses of `handle` into its completion
// sync(s) and at most one carry (the value the handle becomes further along
// control flow). Sets `ambiguous` if the handle has more than one carry, or a
// use the analysis cannot reduce to a single continuation. The set of syncs and
// the carry do not depend on `getUses()` iteration order (only the relative
// order within `syncs` follows use order, which no consumer relies on).
struct TraceStep {
  llvm::SmallVector<Operation *, 2> syncs; // dma_await/free reached here, in
                                           // use order
  Value carry = nullptr;        // value to continue tracing from, if any
  bool carryIsBackEdge = false; // carry crosses a loop back-edge
  bool ambiguous = false;       // multiple carries, or an unknown use
};

static TraceStep classifyUses(Value handle) {
  TraceStep step;
  auto markCarry = [&](Value v, bool backEdge) {
    if (step.carry) { // more than one continuation: cannot linearize
      step.ambiguous = true;
      return;
    }
    step.carry = v;
    step.carryIsBackEdge = backEdge;
  };
  for (OpOperand &use : handle.getUses()) {
    Operation *user = use.getOwner();
    if (isa<DMAStartTaskOp>(user))
      continue; // submission, not a sync or carry
    if (isa<DMAAwaitTaskOp, DMAFreeTaskOp>(user)) {
      // Every sync on this handle completes the same task (the await-then-free
      // idiom, or an scf.if result freed on both paths). Record them all; the
      // first is the kill point and the rest are redundant releases. A genuine
      // double free is caught later by the allocator, not treated as an
      // ambiguous live-range here.
      step.syncs.push_back(user);
      continue;
    }
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      Operation *parent = yield->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        // Yielded operand i maps to iter_arg i (region arg i+1; arg 0 is the
        // IV) of the *next* iteration: a back-edge crossing.
        markCarry(forOp.getRegionIterArgs()[use.getOperandNumber()],
                  /*backEdge=*/true);
      else if (auto ifOp = dyn_cast<scf::IfOp>(parent))
        // Yielded operand i maps to if-result i (a value join).
        markCarry(ifOp.getResult(use.getOperandNumber()), /*backEdge=*/false);
      else
        step.ambiguous = true; // yield of some other region op
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      // Handle passed as an iter_arg init operand: entry into iteration 0.
      unsigned operandIdx = use.getOperandNumber();
      if (operandIdx >= forOp.getNumControlOperands())
        markCarry(forOp.getRegionIterArgs()[operandIdx -
                                            forOp.getNumControlOperands()],
                  /*backEdge=*/false);
      else
        step.ambiguous = true; // handle used as a loop bound/step (nonsensical)
      continue;
    }
    // Any other user (e.g. used as an operand to an unknown op) is not
    // understood: refuse to guess.
    step.ambiguous = true;
  }
  return step;
}

} // namespace

namespace xilinx::AIEX {

TaskLiveRange resolveTaskLiveRange(DMAConfigureTaskOp configure) {
  TaskLiveRange range;
  range.enclosingLoop = configure->getParentOfType<scf::ForOp>().getOperation();

  // Forward-trace the handle to its completion sync. A visited set makes a
  // def-use cycle (a handle carried unchanged across a loop back-edge) a
  // terminating "ambiguous" result rather than an infinite loop.
  llvm::SmallPtrSet<Value, 8> visited;
  Value handle = configure.getResult();
  unsigned backEdges = 0;
  while (handle) {
    if (!visited.insert(handle).second) {
      range.ambiguous = true; // cycle
      break;
    }
    TraceStep step = classifyUses(handle);
    if (step.ambiguous) {
      range.ambiguous = true;
      break;
    }
    if (!step.syncs.empty()) {
      // A sync and a carry on the same value cannot both be honored statically.
      if (step.carry) {
        range.ambiguous = true;
        break;
      }
      range.syncs = std::move(step.syncs);
      range.backEdgesCrossed = backEdges;
      range.leaked = false;
      return range;
    }
    if (!step.carry) {
      range.leaked = true; // no sync reachable
      return range;
    }
    if (step.carryIsBackEdge)
      ++backEdges;
    handle = step.carry;
  }

  // Broke out of the loop via ambiguity (range.ambiguous is set); also mark
  // leaked so callers that only check leaked still reject it.
  range.leaked = true;
  return range;
}

llvm::DenseMap<Operation *, llvm::SmallVector<DMAConfigureTaskOp>>
mapSyncsToConfigures(AIE::RuntimeSequenceOp seq) {
  // Invert the forward handle-trace: for each configure, every sync op on the
  // terminal handle completes it. Collecting all syncs (not just the first
  // kill) captures the await-then-free idiom, and following scf.if joins makes
  // one join free resolve to every arm's configure -- the two multi-op forms
  // the allocator's recycle path must handle.
  llvm::DenseMap<Operation *, llvm::SmallVector<DMAConfigureTaskOp>> syncs;
  seq.walk([&](DMAConfigureTaskOp configure) {
    TaskLiveRange range = resolveTaskLiveRange(configure);
    if (range.ambiguous || range.leaked)
      return; // rejected by the allocator up front; nothing to complete
    for (Operation *sync : range.syncs)
      syncs[sync].push_back(configure);
  });
  return syncs;
}

} // namespace xilinx::AIEX
