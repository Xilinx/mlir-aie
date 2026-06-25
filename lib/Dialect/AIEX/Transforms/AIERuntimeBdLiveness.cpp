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
        Operation *parent = yield->getParentOp();
        // scf.for: yielded operand i maps to iter_arg i (region arg i+1; arg 0
        // is the IV) of the *next* iteration: a back-edge crossing.
        if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
          next = forOp.getRegionIterArgs()[use.getOperandNumber()];
          nextIsBackEdge = true;
        } else if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
          // scf.if: yielded operand i maps to if-result i (a value join, not a
          // loop iteration). Continue from the result.
          next = ifOp.getResult(use.getOperandNumber());
          nextIsBackEdge = false;
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

  unsigned bdCount(DMAConfigureTaskOp op) {
    // One BD per dma_bd in the task's chain.
    unsigned n = 0;
    op.walk([&](AIE::DMABDOp) { ++n; });
    return n ? n : 1;
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
      add(live, tileKey(cfg), bdCount(cfg));
      return;
    }
    if (auto await = dyn_cast<DMAAwaitTaskOp>(op)) {
      if (auto cfg = dyn_cast_or_null<DMAConfigureTaskOp>(
              await.getTask().getDefiningOp()))
        remove(live, tileKey(cfg), bdCount(cfg));
      return;
    }
    if (auto freeOp = dyn_cast<DMAFreeTaskOp>(op)) {
      if (auto cfg = dyn_cast_or_null<DMAConfigureTaskOp>(
              freeOp.getTask().getDefiningOp()))
        remove(live, tileKey(cfg), bdCount(cfg));
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
      if (range.leaked) {
        info = "leaked"; // held to region end (no await/free reachable)
      } else {
        info = "backedges=" + std::to_string(range.backEdgesCrossed) +
               " kill=" + range.kill->getName().getStringRef().str();
      }
      configure.emitRemark("bd-liveness: ")
          << info << (range.enclosingLoop ? " in-loop" : "");
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
