//===- AIERuntimeBdLiveness.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "AIERuntimeBdLiveness.h"

#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIETESTRUNTIMEBDLIVENESS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

/// Follow a task handle SSA value forward to its completion-sync
/// (`dma_await_task`/`dma_free_task`), counting `scf.for` back-edges crossed.
///
/// The handle starts as the result of an `aiex.dma_configure_task` (or, in the
/// ping-pong case, as the iter_arg init operand carried into an `scf.for`). At
/// each step we inspect its uses:
///   - `dma_await_task` / `dma_free_task`  -> sync found; stop.
///   - `dma_start_task`                    -> submission, not a sync; ignore.
///   - `scf.yield` of an `scf.for`         -> carried to the matching iter_arg
///        of the next iteration (one back-edge crossing); continue from there.
///   - `scf.for` iter_arg init operand     -> carried into the loop body's
///        iter_arg; continue from there (NOT a back-edge crossing -- this is
///        entry into iteration 0).
///
/// A handle yielded to a loop *result* (escaping the loop) rather than re-fed
/// as an iter_arg is NOT followed: the in-loop incarnations are already dropped
/// at that point, so any later free of the result frees only the final
/// straggler. Such a configure is therefore reported as leaked (held to region
/// end), which is the correct basis for the per-iteration coexistence
/// accounting upstream.
///
/// Returns true and fills `outKill`/`outBackEdges` when a sync is found;
/// returns false (handle leaked to region end) otherwise.
static bool traceHandleToKill(Value handle, Operation *&outKill,
                              unsigned &outBackEdges) {
  unsigned backEdges = 0;
  while (handle) {
    Operation *kill = nullptr;
    Value next = nullptr;
    bool nextIsBackEdge = false;
    for (OpOperand &use : handle.getUses()) {
      Operation *user = use.getOwner();
      if (isa<DMAAwaitTaskOp, DMAFreeTaskOp>(user)) {
        kill = user;
        break;
      }
      if (isa<DMAStartTaskOp>(user))
        continue;
      if (auto yield = dyn_cast<scf::YieldOp>(user)) {
        // Yielded operand i maps to scf.for iter_arg i (region arg i+1; arg 0
        // is the IV) of the *next* iteration: a back-edge crossing.
        if (auto forOp = dyn_cast<scf::ForOp>(yield->getParentOp())) {
          next = forOp.getRegionIterArgs()[use.getOperandNumber()];
          nextIsBackEdge = true;
        }
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        // Handle passed as an iter_arg init operand: entry into iteration 0.
        unsigned operandIdx = use.getOperandNumber();
        if (operandIdx >= forOp.getNumControlOperands()) {
          unsigned iterIdx = operandIdx - forOp.getNumControlOperands();
          next = forOp.getRegionIterArgs()[iterIdx];
          nextIsBackEdge = false;
        }
        continue;
      }
    }
    if (kill) {
      outKill = kill;
      outBackEdges = backEdges;
      return true;
    }
    if (!next)
      return false;
    if (nextIsBackEdge)
      ++backEdges;
    handle = next;
  }
  return false;
}

} // namespace

namespace xilinx::AIEX {

TaskLiveRange resolveTaskLiveRange(DMAConfigureTaskOp configure) {
  TaskLiveRange range;
  range.configure = configure;
  range.enclosingLoop =
      configure->getParentOfType<scf::ForOp>()
          ? configure->getParentOfType<scf::ForOp>().getOperation()
          : nullptr;
  Operation *kill = nullptr;
  unsigned backEdges = 0;
  if (traceHandleToKill(configure.getResult(), kill, backEdges)) {
    range.kill = kill;
    range.backEdgesCrossed = backEdges;
    range.leaked = false;
  } else {
    range.kill = nullptr;
    range.leaked = true;
  }
  return range;
}

} // namespace xilinx::AIEX

namespace {
struct AIETestRuntimeBdLivenessPass
    : xilinx::AIEX::impl::AIETestRuntimeBdLivenessBase<
          AIETestRuntimeBdLivenessPass> {
  void runOnOperation() override {
    getOperation().walk([&](DMAConfigureTaskOp configure) {
      TaskLiveRange range = resolveTaskLiveRange(configure);
      // A task configured inside a loop whose handle is never synced
      // (await/free) accumulates one held BD per iteration: a bug for a
      // constant trip count and impossible (unbounded) for a runtime one.
      if (range.leaked && range.enclosingLoop) {
        configure.emitOpError(
            "buffer descriptor configured in a loop is never completed "
            "(no aiex.dma_await_task / aiex.dma_free_task reachable); its BD "
            "would not be reusable across iterations");
        return;
      }
      std::string info;
      if (range.leaked) {
        info = "leaked"; // held to region end (no await/free reachable)
      } else {
        info = "backedges=" + std::to_string(range.backEdgesCrossed) +
               " kill=" + range.kill->getName().getStringRef().str();
      }
      configure.emitRemark("bd-liveness: ")
          << info << (range.enclosingLoop ? " in-loop" : "");
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIETestRuntimeBdLivenessPass() {
  return std::make_unique<AIETestRuntimeBdLivenessPass>();
}
