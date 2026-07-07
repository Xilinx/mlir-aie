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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIETESTRUNTIMEBDLIVENESS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

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
      if (step.sync)
        step.ambiguous = true; // freed/awaited more than once
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

LoopRotationGroup resolveLoopRotationGroup(DMAConfigureTaskOp body) {
  LoopRotationGroup group;

  // A rotation body is a configure inside a loop whose handle is freed a
  // positive number of back-edges later. Anything else is not a rotation body
  // (straight-line, same-iteration free, leak, or ambiguous handle).
  TaskLiveRange range = resolveTaskLiveRange(body);
  if (range.ambiguous || range.leaked || range.backEdgesCrossed == 0 ||
      !range.enclosingLoop)
    return group; // NotARotation

  auto loop = dyn_cast<scf::ForOp>(range.enclosingLoop);
  if (!loop)
    return group; // NotARotation

  group.windowWidth = range.backEdgesCrossed + 1; // W = D + 1
  group.loop = loop;
  group.chainLength = chainLength(body);

  // The prologue tasks seed the loop's iter_args: a depth-D ping-pong threads
  // its handles through a D-deep shift register of iter_args, each initialized
  // (before iteration 0) by a prologue configure. Collect every iter_arg whose
  // init operand is a configure; those configures plus the body share the
  // window. (An init operand that is not a configure -- e.g. an outer iter_arg
  // or a non-task value -- means the handle cannot be traced to a prologue, so
  // the rotation is unresolvable.)
  llvm::SmallVector<DMAConfigureTaskOp, 4> prologues;
  for (Value init : loop.getInitArgs()) {
    auto cfg = dyn_cast_or_null<DMAConfigureTaskOp>(init.getDefiningOp());
    if (!cfg) {
      group.status = LoopRotationGroup::Unresolvable;
      return group;
    }
    prologues.push_back(cfg);
  }

  // A depth-D rotation needs exactly D prologues (one per back-edge crossed).
  if (prologues.size() != range.backEdgesCrossed) {
    group.status = LoopRotationGroup::Unresolvable;
    return group;
  }

  // Every member's chain must have the same length to rotate position-by-
  // position through a shared per-descriptor window.
  for (DMAConfigureTaskOp p : prologues) {
    if (chainLength(p) != group.chainLength) {
      group.status = LoopRotationGroup::ChainLengthMismatch;
      return group;
    }
  }

  group.members.assign(prologues.begin(), prologues.end());
  group.members.push_back(body);
  group.status = LoopRotationGroup::Ok;
  return group;
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
      // Loop-carried tasks remain live across the back-edge: sweep the body
      // with the current live set (the init tasks are already counted from
      // their configures before the loop). A ping-pong's previous-iteration
      // task is thus simultaneously live with the current one.
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
      if (range.ambiguous) {
        info =
            "ambiguous"; // handle not reducible to one sync (cycle/multi-use)
      } else if (range.leaked) {
        info = "leaked"; // held to region end (no await/free reachable)
      } else {
        info = "backedges=" + std::to_string(range.backEdgesCrossed) +
               " kill=" + range.kill->getName().getStringRef().str();
      }
      if (range.crossedIfJoin)
        info += " if-join";
      configure.emitRemark("bd-liveness: ")
          << info << (range.enclosingLoop ? " in-loop" : "");

      // For a rotation body, also report the resolved window: its width W and
      // chain length C (the allocator reserves C*W ids for the group).
      LoopRotationGroup group = resolveLoopRotationGroup(configure);
      if (group.status == LoopRotationGroup::Ok)
        configure.emitRemark("bd-rotation: width=")
            << group.windowWidth << " chain=" << group.chainLength
            << " members=" << group.members.size();
    });

    // Peak simultaneous liveness per tile (the window size the allocator must
    // fit in the tile's BD pool).
    getOperation().walk([&](AIE::RuntimeSequenceOp seq) {
      auto peaks = computePeakBdLiveness(seq);
      for (auto &kv : peaks)
        seq.emitRemark("bd-peak: tile(")
            << kv.first.first << "," << kv.first.second
            << ") peak=" << kv.second;
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIETestRuntimeBdLivenessPass() {
  return std::make_unique<AIETestRuntimeBdLivenessPass>();
}
