//===- AIERuntimeBdLiveness.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIERuntimeBdLiveness.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

// One step of the trace: classify all uses of `handle` into at most one
// completion sync and at most one carry (the value the handle becomes further
// along control flow). Order-independent: the result does not depend on
// `getUses()` iteration order. Sets `ambiguous` if the handle has more than one
// sync/carry, or a use the analysis cannot reduce to a single continuation.
struct TraceStep {
  Operation *sync = nullptr;    // dma_await/free reached here, if any
  Value carry = nullptr;        // value to continue tracing from, if any
  bool carryIsBackEdge = false; // carry crosses a loop back-edge
  bool carryIsIfJoin = false;   // carry is an scf.if result (value join)
  bool ambiguous = false;       // multiple syncs/carries, or an unknown use
};

static TraceStep classifyUses(Value handle) {
  TraceStep step;
  auto markCarry = [&](Value v, bool backEdge, bool ifJoin) {
    if (step.carry) { // more than one continuation: cannot linearize
      step.ambiguous = true;
      return;
    }
    step.carry = v;
    step.carryIsBackEdge = backEdge;
    step.carryIsIfJoin = ifJoin;
  };
  for (OpOperand &use : handle.getUses()) {
    Operation *user = use.getOwner();
    if (isa<DMAStartTaskOp>(user))
      continue; // submission, not a sync or carry
    if (isa<DMAAwaitTaskOp, DMAFreeTaskOp>(user)) {
      // Multiple syncs on one handle (e.g. the common await-then-free idiom, or
      // an scf.if result freed on both paths) all complete the same task; the
      // first is the kill point. A genuine double free is caught later by the
      // allocator, not treated as an ambiguous live-range here.
      if (!step.sync)
        step.sync = user;
      continue;
    }
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      Operation *parent = yield->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        // Yielded operand i maps to iter_arg i (region arg i+1; arg 0 is the
        // IV) of the *next* iteration: a back-edge crossing.
        markCarry(forOp.getRegionIterArgs()[use.getOperandNumber()],
                  /*backEdge=*/true, /*ifJoin=*/false);
      else if (auto ifOp = dyn_cast<scf::IfOp>(parent))
        // Yielded operand i maps to if-result i (a value join).
        markCarry(ifOp.getResult(use.getOperandNumber()), /*backEdge=*/false,
                  /*ifJoin=*/true);
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
                  /*backEdge=*/false, /*ifJoin=*/false);
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
  range.configure = configure;
  range.enclosingLoop =
      configure->getParentOfType<scf::ForOp>()
          ? configure->getParentOfType<scf::ForOp>().getOperation()
          : nullptr;

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
    if (step.sync) {
      // A sync and a carry on the same value cannot both be honored statically.
      if (step.carry) {
        range.ambiguous = true;
        break;
      }
      range.kill = step.sync;
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
    if (step.carryIsIfJoin)
      range.crossedIfJoin = true;
    handle = step.carry;
  }

  // Fell out via ambiguity: report as leaked-with-ambiguous so callers reject.
  range.leaked = true;
  return range;
}

unsigned chainLength(DMAConfigureTaskOp op) {
  unsigned n = 0;
  op.walk([&](AIE::DMABDOp) { ++n; });
  return n ? n : 1;
}

} // namespace xilinx::AIEX

namespace {

/// Computes peak simultaneous BD liveness per tile across a runtime sequence by
/// a structural sweep. A configure adds its tile's held BD count to the live
/// set; the matching await/free removes it. scf.for bodies are swept with their
/// loop-carried (iter_arg-seeded) tasks already live, so a ping-pong's previous
/// iteration counts against the current one. scf.if arms are swept
/// independently and the arm with the larger contribution is taken (mutually
/// exclusive arms do not interfere). The per-tile maximum of the running live
/// count is the window size the BD-ID allocator must satisfy against that
/// tile's pool.
class PeakLivenessSweep {
public:
  /// Map from tile (col,row packed) to peak simultaneous BD count.
  using PeakMap = llvm::MapVector<std::pair<int, int>, unsigned>;

  PeakLivenessSweep(PeakMap &peaks) : peaks(peaks) {}

  /// Sweep a runtime sequence body region.
  void sweepSequence(mlir::Region &body) {
    Live live;
    sweepRegion(body, live);
  }

private:
  /// Per-tile currently-live BD count.
  using Live = llvm::MapVector<std::pair<int, int>, unsigned>;

  static std::pair<int, int> tileKey(DMAConfigureTaskOp op) {
    AIE::TileOp t = op.getTileOp();
    return {t.getCol(), t.getRow()};
  }

  void add(Live &live, std::pair<int, int> tile, unsigned n) {
    unsigned &cur = live[tile];
    cur += n;
    unsigned &peak = peaks[tile];
    peak = std::max(peak, cur);
  }

  void remove(Live &live, std::pair<int, int> tile, unsigned n) {
    unsigned &cur = live[tile];
    cur = (cur >= n) ? cur - n : 0;
  }

  /// Sweep ops in a single region (one block; runtime sequences are
  /// single-block, scf bodies are single-block).
  void sweepRegion(mlir::Region &region, Live &live) {
    for (mlir::Block &block : region)
      for (mlir::Operation &op : block)
        sweepOp(&op, live);
  }

  void sweepOp(mlir::Operation *op, Live &live) {
    if (auto cfg = dyn_cast<DMAConfigureTaskOp>(op)) {
      add(live, tileKey(cfg), chainLength(cfg));
      return;
    }
    if (auto await = dyn_cast<DMAAwaitTaskOp>(op)) {
      if (auto cfg = dyn_cast_or_null<DMAConfigureTaskOp>(
              await.getTask().getDefiningOp()))
        remove(live, tileKey(cfg), chainLength(cfg));
      return;
    }
    if (auto freeOp = dyn_cast<DMAFreeTaskOp>(op)) {
      if (auto cfg = dyn_cast_or_null<DMAConfigureTaskOp>(
              freeOp.getTask().getDefiningOp()))
        remove(live, tileKey(cfg), chainLength(cfg));
      return;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Any surviving loop is runtime-bound (constant-trip loops were unrolled
      // before this analysis) with same-iteration tasks -- a handle carried
      // across the back-edge is rejected by the allocator. Sweep the body with
      // the current live set so tasks held across the loop from before it still
      // count against the tasks inside it.
      sweepRegion(forOp.getRegion(), live);
      return;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Arms are mutually exclusive: sweep each on a copy of the live set, keep
      // the per-tile maximum. (peaks is updated as a side effect within each.)
      Live thenLive = live;
      sweepRegion(ifOp.getThenRegion(), thenLive);
      if (!ifOp.getElseRegion().empty()) {
        Live elseLive = live;
        sweepRegion(ifOp.getElseRegion(), elseLive);
        for (auto &kv : elseLive)
          thenLive[kv.first] = std::max(thenLive[kv.first], kv.second);
      }
      live = thenLive;
      return;
    }
    // Other ops with regions (none expected in a sequence today): recurse
    // conservatively so nested configures are still counted.
    for (mlir::Region &r : op->getRegions())
      sweepRegion(r, live);
  }

  PeakMap &peaks;
};

} // namespace

namespace xilinx::AIEX {

llvm::MapVector<std::pair<int, int>, unsigned>
computePeakBdLiveness(AIE::RuntimeSequenceOp seq) {
  PeakLivenessSweep::PeakMap peaks;
  PeakLivenessSweep sweep(peaks);
  sweep.sweepSequence(seq.getBody());
  return peaks;
}

} // namespace xilinx::AIEX
