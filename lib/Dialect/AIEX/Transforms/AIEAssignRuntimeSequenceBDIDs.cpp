//===- AIEAssignRuntimeSequenceBDIDs.cpp ------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Assigns buffer-descriptor (BD) IDs to the DMA tasks configured in a runtime
// sequence. Control-flow-aware: tasks inside scf.for / scf.if are allocated
// against a real per-tile pool, with arm-exclusivity (scf.if arms are swept
// independently and merged) and same-iteration reuse (scf.for bodies recurse).
//
// A task whose BD rotates through several physical IDs across loop iterations
// (the rolled ping-pong "free the previous iteration" pattern, where the handle
// crosses a loop back-edge via iter_args) is allocated a rotating *window* of
// `D + 1` ids (D = back-edges crossed), recorded on each member chain's
// `aie.dma_bd` as a `bd_id_window` (with `bd_id` the window base). The
// downstream lowering selects the per-iteration id from that window. Genuinely
// unallocatable forms (ambiguous handles, unbounded in-loop leaks, peak
// liveness exceeding the tile pool) are rejected with a clear diagnostic. The
// liveness analysis that classifies all of these lives in
// AIERuntimeBdLiveness.{h,cpp}.
//
//===----------------------------------------------------------------------===//

#include "AIERuntimeBdLiveness.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <map>
#include <set>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEASSIGNRUNTIMESEQUENCEBDIDS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIEAssignRuntimeSequenceBDIDsPass
    : xilinx::AIEX::impl::AIEAssignRuntimeSequenceBDIDsBase<
          AIEAssignRuntimeSequenceBDIDsPass> {

  std::map<AIE::TileOp, BdIdGenerator> gens;

  // A reserved rotating BD-id window: the C*W physical ids a depth-(W-1)
  // ping-pong's chain cycles through, plus the tile that owns them. Reserved
  // before the loop body is swept and released once after, so the window
  // survives every iteration but does not leak past the loop.
  struct ReservedWindow {
    AIE::TileOp tile;
    SmallVector<uint32_t>
        ids; // C*W ids, grouped C-major: [bd0 window][bd1 ...]
  };

  // Rotation planning, computed once per sequence before the sweep:
  //   rotationMembers: every configure that participates in a rotation window
  //     (body + prologues) -> its group, so allocate/recycle take the window
  //     path instead of the single-id path.
  //   windowsByLoop: the windows to reserve before / release after each loop.
  //   inertFrees: free/await ops whose ids are managed by a window release, so
  //     they must not return ids to the pool when swept.
  llvm::DenseMap<Operation *, AIEX::LoopRotationGroup> rotationMembers;
  llvm::DenseMap<Operation *, SmallVector<ReservedWindow>> windowsByLoop;
  llvm::DenseSet<Operation *> inertFrees;

  // Monotonic id handed to each rotation group as it is reserved, stamped onto
  // its BDs as bd_id_window_group so the unroll pass can key its round-robin
  // counter per group. Distinct groups can share identical window contents, so
  // the contents alone cannot distinguish them.
  int32_t nextWindowGroupId = 0;

  BdIdGenerator &getGeneratorForTile(AIE::TileOp tile) {
    auto it = gens.find(tile);
    if (it == gens.end()) {
      const AIETargetModel &targetModel =
          tile->getParentOfType<AIE::DeviceOp>().getTargetModel();
      it = gens.insert({tile, BdIdGenerator(tile.getCol(), tile.getRow(),
                                            targetModel)})
               .first;
    }
    return it->second;
  }

  // Trace a free/await task value back to the configure op(s) whose BDs it
  // completes, following scf.if results (to each arm's yielded source) and
  // scf.for results / iter_args (to the loop's init and yielded sources). A
  // direct configure result resolves to itself. Returns false on a value the
  // tracer does not understand; `targets` is the deduplicated set of
  // configures.
  bool resolveFreeTargets(Value task,
                          SmallVectorImpl<DMAConfigureTaskOp> &targets) {
    llvm::SmallPtrSet<Value, 8> visited;
    SmallVector<Value> worklist{task};
    llvm::SmallPtrSet<Operation *, 8> seen;
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!v || !visited.insert(v).second)
        continue;
      if (auto cfg = dyn_cast_or_null<DMAConfigureTaskOp>(v.getDefiningOp())) {
        if (seen.insert(cfg).second)
          targets.push_back(cfg);
        continue;
      }
      if (auto res = dyn_cast<OpResult>(v)) {
        Operation *def = res.getOwner();
        if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
          unsigned i = res.getResultNumber();
          worklist.push_back(ifOp.thenYield().getOperand(i));
          if (!ifOp.getElseRegion().empty())
            worklist.push_back(ifOp.elseYield().getOperand(i));
          continue;
        }
        if (auto forOp = dyn_cast<scf::ForOp>(def)) {
          unsigned i = res.getResultNumber();
          worklist.push_back(forOp.getInitArgs()[i]);
          worklist.push_back(forOp.getBody()->getTerminator()->getOperand(i));
          continue;
        }
        return false; // result of an op we do not model
      }
      if (auto arg = dyn_cast<BlockArgument>(v)) {
        if (auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
          // iter_arg k is region arg k+1 (arg 0 is the IV).
          unsigned k = arg.getArgNumber();
          if (k == 0)
            return false; // the induction variable, not a task
          worklist.push_back(forOp.getInitArgs()[k - 1]);
          worklist.push_back(
              forOp.getBody()->getTerminator()->getOperand(k - 1));
          continue;
        }
        return false; // block arg of an op we do not model
      }
      return false;
    }
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Validation: reject control-flow forms whose static BD-ID write-back is not
  // supported yet (they need a runtime-selected BD id).
  //===--------------------------------------------------------------------===//

  // Reject every form the static allocator cannot handle, BEFORE any IDs are
  // assigned, so allocation only ever runs on supported IR. This is the
  // complete gatekeeper: anything not rejected here must allocate without
  // crashing.
  LogicalResult validate(AIE::RuntimeSequenceOp seq) {
    // 1. Per-configure liveness classification. Rolled ping-pong (handle held
    // across loop back-edges) and scf.if value-joins are supported via rotation
    // windows / free-target resolution; only genuinely unallocatable forms are
    // rejected here.
    WalkResult wr = seq.walk([&](DMAConfigureTaskOp configure) -> WalkResult {
      TaskLiveRange range = resolveTaskLiveRange(configure);
      if (range.ambiguous) {
        configure.emitOpError(
            "task handle cannot be statically resolved to a single completion "
            "(it has multiple carries/syncs, or is carried unchanged across a "
            "loop back-edge); static BD-ID allocation is not supported for "
            "this "
            "form");
        return WalkResult::interrupt();
      }
      if (range.leaked && range.enclosingLoop) {
        configure.emitOpError(
            "buffer descriptor configured in a loop is never completed (no "
            "aiex.dma_await_task / aiex.dma_free_task reachable); its BD would "
            "not be reusable across iterations");
        return WalkResult::interrupt();
      }
      // A handle held across a loop back-edge anchors a rotation window when
      // the configure is itself inside the loop (the body task). A prologue
      // task is defined outside the loop and seeds an iter_arg, so it also
      // reports backEdgesCrossed > 0 but is validated via its body, not here.
      // Surface the rotation-specific rejections (mismatched chain lengths, a
      // prologue that does not trace to a configure) on the body.
      if (range.backEdgesCrossed > 0 && range.enclosingLoop) {
        LoopRotationGroup group = resolveLoopRotationGroup(configure);
        if (group.status == LoopRotationGroup::ChainLengthMismatch) {
          configure.emitOpError(
              "rotating buffer-descriptor chain length differs from the "
              "prologue tasks it rotates with; a rolled ping-pong must rotate "
              "chains of equal length");
          return WalkResult::interrupt();
        }
        if (group.status != LoopRotationGroup::Ok) {
          configure.emitOpError(
              "buffer descriptor is held across a loop back-edge but its "
              "rotation cannot be resolved (a carried task does not originate "
              "from a dma_configure_task)");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return failure();

    // 2. Every free/await must resolve to at least one configure op. A free of
    // an scf.if/for result is valid when it traces back (through yields /
    // iter_args) to the per-arm or rotation configures it completes; a free of
    // a value that traces to no configure (e.g. an scf.for result seeded by a
    // non-task constant) would crash the recycle path and is rejected here.
    wr = seq.walk([&](Operation *op) -> WalkResult {
      Value task;
      if (auto f = dyn_cast<DMAFreeTaskOp>(op))
        task = f.getTask();
      else if (auto a = dyn_cast<DMAAwaitTaskOp>(op))
        task = a.getTask();
      else
        return WalkResult::advance();
      SmallVector<DMAConfigureTaskOp> targets;
      if (!resolveFreeTargets(task, targets) || targets.empty()) {
        // Preserve the helpful "lower this first" notes for unlowered task ops.
        (void)emitUnresolvedTask(op, task);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return failure();

    // 3. Peak simultaneous BD liveness must fit the tile's BD pool.
    {
      auto peaks = computePeakBdLiveness(seq);
      const AIETargetModel &tm =
          seq->getParentOfType<AIE::DeviceOp>().getTargetModel();
      for (auto &kv : peaks) {
        int col = kv.first.first, row = kv.first.second;
        uint32_t pool = tm.getNumBDs(col, row);
        if (kv.second > pool) {
          seq.emitOpError("peak simultaneous buffer-descriptor liveness ")
              << kv.second << " on tile(" << col << "," << row
              << ") exceeds the tile's " << pool << " buffer descriptors";
          return failure();
        }
      }
    }

    // 4. A configure must not be buried inside a region op the allocator does
    // not sweep (e.g. scf.while); its BDs would be silently left unassigned.
    wr = seq.walk([&](DMAConfigureTaskOp configure) -> WalkResult {
      Operation *p = configure->getParentOp();
      while (p && p != seq.getOperation()) {
        if (!isa<scf::ForOp, scf::IfOp>(p)) {
          configure.emitOpError("is nested in an unsupported control-flow op '")
              << p->getName()
              << "'; BD-ID allocation only handles scf.for and scf.if";
          return WalkResult::interrupt();
        }
        p = p->getParentOp();
      }
      return WalkResult::advance();
    });
    return failure(wr.wasInterrupted());
  }

  //===--------------------------------------------------------------------===//
  // Rotation planning
  //===--------------------------------------------------------------------===//

  // Identify every rolled-ping-pong rotation group in the sequence and record,
  // for each, its member configures (rotationMembers), the window to reserve at
  // its loop (windowsByLoop), and the free/await ops the window release will
  // account for (inertFrees). Runs after validate(), so groups are well-formed.
  void planRotations(AIE::RuntimeSequenceOp seq) {
    rotationMembers.clear();
    windowsByLoop.clear();
    inertFrees.clear();
    seq.walk([&](DMAConfigureTaskOp configure) {
      LoopRotationGroup group = resolveLoopRotationGroup(configure);
      if (group.status != LoopRotationGroup::Ok)
        return; // not a rotation body (prologues / plain tasks resolve here
                // too)
      for (DMAConfigureTaskOp member : group.members)
        rotationMembers[member.getOperation()] = group;
      // Reserve a placeholder window for this loop; ids are filled in when the
      // loop is reached during the sweep (so sequential loops reuse the pool).
      windowsByLoop[group.loop.getOperation()].push_back(ReservedWindow{});

      // The frees/awaits that complete this group's handles are managed by the
      // window release: a free targeting a member configure, the loop's
      // iter_args, or the loop's results.
      auto markInert = [&](Operation *user) {
        if (isa<DMAFreeTaskOp, DMAAwaitTaskOp>(user))
          inertFrees.insert(user);
      };
      for (DMAConfigureTaskOp member : group.members)
        for (Operation *user : member.getResult().getUsers())
          markInert(user);
      for (Value v : group.loop.getRegionIterArgs())
        for (Operation *user : v.getUsers())
          markInert(user);
      for (Value v : group.loop.getResults())
        for (Operation *user : v.getUsers())
          markInert(user);
    });
  }

  // Reserve C*W ids for each window registered on `loop`, write bd_id +
  // bd_id_window onto every member chain, and return the reserved ids so the
  // caller can release them after the loop body is swept.
  LogicalResult reserveWindowsForLoop(scf::ForOp loop,
                                      SmallVectorImpl<ReservedWindow> &out) {
    // Collect the distinct groups whose loop is this one (windowsByLoop holds
    // one placeholder per group; recover the groups from rotationMembers).
    llvm::SmallPtrSet<Operation *, 4> doneAnchors;
    for (auto &kv : rotationMembers) {
      const LoopRotationGroup &group = kv.second;
      if (group.loop != loop)
        continue;
      DMAConfigureTaskOp body = group.members.back();
      if (!doneAnchors.insert(body.getOperation()).second)
        continue; // already reserved this group

      AIE::TileOp tile = body.getTileOp();
      BdIdGenerator &gen = getGeneratorForTile(tile);
      unsigned C = group.chainLength, W = group.windowWidth;

      // A rotating chain cycles through several physical ids, so a single
      // user-pinned bd_id cannot express it; reject rather than silently
      // overwrite the user's choice.
      for (DMAConfigureTaskOp member : group.members) {
        WalkResult pinned = member.walk([&](AIE::DMABDOp bd) {
          if (bd.getBdId().has_value()) {
            member.emitOpError(
                "participates in a rolled ping-pong rotation, so its buffer "
                "descriptor id is allocated as a rotating window and cannot be "
                "manually assigned");
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (pinned.wasInterrupted())
          return failure();
      }

      ReservedWindow rw;
      rw.tile = tile;
      // Reserve C*W ids: C per-descriptor windows of W ids each.
      for (unsigned i = 0; i < C * W; ++i) {
        std::optional<int32_t> id = gen.nextBdId(/*channelIndex=*/0);
        if (!id) {
          body.emitOpError("Allocator exhausted available buffer descriptor "
                           "IDs reserving a rotating window of ")
              << (C * W) << " ids.";
          return failure();
        }
        rw.ids.push_back(*id);
      }
      // Write per-descriptor windows onto every member's chain. Descriptor c of
      // the chain rotates through ids[c*W .. c*W+W); bd_id holds the base.
      // Every BD in this group is stamped with the same group id so the unroll
      // pass round-robins each group independently even when windows collide.
      int32_t groupId = nextWindowGroupId++;
      for (DMAConfigureTaskOp member : group.members) {
        unsigned c = 0;
        member.walk([&](AIE::DMABDOp bd) {
          SmallVector<int32_t> window(rw.ids.begin() + c * W,
                                      rw.ids.begin() + c * W + W);
          bd.setBdId(window[0]);
          if (W > 1) {
            bd->setAttr("bd_id_window",
                        DenseI32ArrayAttr::get(bd.getContext(), window));
            bd->setAttr("bd_id_window_group",
                        IntegerAttr::get(IntegerType::get(bd.getContext(), 32),
                                         groupId));
          }
          ++c;
        });
      }
      out.push_back(std::move(rw));
    }
    return success();
  }

  void releaseWindows(ArrayRef<ReservedWindow> windows) {
    for (const ReservedWindow &rw : windows) {
      BdIdGenerator &gen = getGeneratorForTile(rw.tile);
      for (uint32_t id : rw.ids)
        if (gen.bdIdAlreadyAssigned(id))
          gen.freeBdId(id);
    }
  }

  //===--------------------------------------------------------------------===//
  // Allocation
  //===--------------------------------------------------------------------===//

  LogicalResult allocateConfigure(DMAConfigureTaskOp op) {
    // Rotation members had their bd_id / bd_id_window written when their loop's
    // window was reserved; nothing more to allocate here.
    if (rotationMembers.count(op.getOperation()))
      return success();
    AIE::TileOp tile = op.getTileOp();
    BdIdGenerator &gen = getGeneratorForTile(tile);

    // First, honor all the user-specified BD IDs.
    WalkResult result = op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
      if (bd_op.getBdId().has_value()) {
        if (gen.bdIdAlreadyAssigned(bd_op.getBdId().value())) {
          op.emitOpError("Specified buffer descriptor ID ")
              << bd_op.getBdId().value()
              << " is already in use. Emit an aiex.dma_free_task operation to "
                 "reuse BDs.";
          return WalkResult::interrupt();
        }
        gen.assignBdId(bd_op.getBdId().value());
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();

    // Now allocate BD IDs for all unspecified BDs.
    result = op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
      if (bd_op.getBdId().has_value())
        return WalkResult::advance();
      // channelIndex only affects allocation on MemTiles, where the AIE2 model
      // partitions BDs by channel parity (isBdChannelAccessible). Runtime
      // sequences configure BDs on shim (and compute) tiles only, which are
      // channel-agnostic (always accessible), so passing 0 is correct here.
      std::optional<int32_t> next_id = gen.nextBdId(/*channelIndex=*/0);
      if (!next_id) {
        op.emitOpError() << "Allocator exhausted available buffer descriptor "
                            "IDs.";
        return WalkResult::interrupt();
      }
      bd_op.setBdId(*next_id);
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();

    return success();
  }

  LogicalResult recycle(DMAConfigureTaskOp task_op) {
    BdIdGenerator &gen = getGeneratorForTile(task_op.getTileOp());
    WalkResult result = task_op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd) {
      if (!bd.getBdId().has_value()) {
        bd.emitOpError("Free called on BD chain with unassigned IDs.");
        return WalkResult::interrupt();
      }
      // Ignore double-frees.
      if (gen.bdIdAlreadyAssigned(bd.getBdId().value()))
        gen.freeBdId(bd.getBdId().value());
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  // Emit the "does not reference a valid configure_task" diagnostic, pointing
  // at an unlowered task op (bd_chain / configure_task_for) when that is the
  // cause.
  LogicalResult emitUnresolvedTask(Operation *op, Value task) {
    auto err =
        op->emitOpError("does not reference a valid configure_task operation.");
    if (Operation *def = task.getDefiningOp()) {
      if (isa<DMAStartBdChainOp>(def))
        err.attachNote(def->getLoc()) << "Lower this operation first using the "
                                         "--aie-materialize-bd-chains pass.";
      if (isa<DMAConfigureTaskForOp>(def))
        err.attachNote(def->getLoc())
            << "Lower this operation first using the "
               "--aie-substitute-shim-dma-allocations pass.";
    }
    return err;
  }

  // Recycle every configure a free/await value resolves to (a single configure,
  // or the per-arm/rotation configures behind an scf.if/for result).
  LogicalResult recycleTargets(Value task, Operation *op) {
    SmallVector<DMAConfigureTaskOp> targets;
    if (!resolveFreeTargets(task, targets) || targets.empty())
      return emitUnresolvedTask(op, task);
    for (DMAConfigureTaskOp t : targets)
      if (failed(recycle(t)))
        return failure();
    return success();
  }

  LogicalResult onFree(DMAFreeTaskOp op) {
    // A free whose ids are owned by a rotation window is accounted for by the
    // window release after the loop; just erase it (free lowers to nothing).
    if (inertFrees.count(op.getOperation())) {
      op.erase();
      return success();
    }
    if (failed(recycleTargets(op.getTask(), op)))
      return failure();
    op.erase();
    return success();
  }

  LogicalResult onAwait(DMAAwaitTaskOp op) {
    if (inertFrees.count(op.getOperation()))
      return success();
    // Awaiting a task guarantees completion, so its BD IDs can be reused after.
    return recycleTargets(op.getTask(), op);
  }

  // Snapshot/restore of the per-tile allocation state, for sweeping scf.if arms
  // independently (mutually exclusive arms do not interfere).
  using Snapshot = std::map<AIE::TileOp, std::set<uint32_t>>;
  Snapshot snapshot() {
    Snapshot s;
    for (auto &kv : gens)
      s[kv.first] = kv.second.alreadyAssigned;
    return s;
  }
  void restore(const Snapshot &s) {
    for (auto &kv : gens) {
      auto it = s.find(kv.first);
      kv.second.alreadyAssigned =
          it == s.end() ? std::set<uint32_t>() : it->second;
    }
  }
  void unionInto(const Snapshot &other) {
    for (auto &kv : other) {
      BdIdGenerator &gen = getGeneratorForTile(kv.first);
      gen.alreadyAssigned.insert(kv.second.begin(), kv.second.end());
    }
  }

  LogicalResult sweepRegion(Region &region) {
    for (Block &block : region) {
      // Collect frees to erase after iterating (onFree erases the op).
      SmallVector<Operation *> ops;
      for (Operation &op : block)
        ops.push_back(&op);
      for (Operation *op : ops)
        if (failed(sweepOp(op)))
          return failure();
    }
    return success();
  }

  LogicalResult sweepOp(Operation *op) {
    if (auto cfg = dyn_cast<DMAConfigureTaskOp>(op))
      return allocateConfigure(cfg);
    if (auto freeOp = dyn_cast<DMAFreeTaskOp>(op))
      return onFree(freeOp);
    if (auto await = dyn_cast<DMAAwaitTaskOp>(op))
      return onAwait(await);
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Reserve any rotating BD windows anchored at this loop (writing their
      // bd_id / bd_id_window onto every member chain) so they are held across
      // every iteration, then sweep the body. Same-iteration tasks allocate and
      // free within one iteration; rotation members are skipped by
      // allocateConfigure and their frees are inert. Releasing the windows
      // after the body restores the pool to its pre-loop state.
      SmallVector<ReservedWindow> reserved;
      if (windowsByLoop.count(op))
        if (failed(reserveWindowsForLoop(forOp, reserved)))
          return failure();
      if (failed(sweepRegion(forOp.getRegion())))
        return failure();
      releaseWindows(reserved);
      return success();
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Arms are mutually exclusive: sweep each from the same starting state
      // and union the resulting allocations (a BD still held when an arm
      // finishes stays reserved regardless of which arm runs).
      Snapshot before = snapshot();
      if (failed(sweepRegion(ifOp.getThenRegion())))
        return failure();
      Snapshot afterThen = snapshot();
      restore(before);
      if (!ifOp.getElseRegion().empty()) {
        if (failed(sweepRegion(ifOp.getElseRegion())))
          return failure();
      }
      unionInto(afterThen);
      return success();
    }
    // Other ops (incl. those with regions not expected here) are inert for BD
    // allocation.
    return success();
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    WalkResult wr = device.walk([&](AIE::RuntimeSequenceOp seq) -> WalkResult {
      if (failed(validate(seq)))
        return WalkResult::interrupt();
      gens.clear();
      planRotations(seq);
      if (failed(sweepRegion(seq.getBody())))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (wr.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEAssignRuntimeSequenceBDIDsPass() {
  return std::make_unique<AIEAssignRuntimeSequenceBDIDsPass>();
}
