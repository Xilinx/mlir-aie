//===- DmaQueueModel.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Per-channel DMA task queue, shared by the dma_start_task and
// npu.dma_memcpy_nd paths so the retirement rule cannot drift between them.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEX_UTILS_DMAQUEUEMODEL_H
#define AIE_DIALECT_AIEX_UTILS_DMAQUEUEMODEL_H

#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <set>

namespace xilinx::AIEX {

/// Tracks, per DMA channel, the pushes not yet known to have completed along
/// one execution path. The sequence analysis below joins control-flow paths.
class DmaQueueModel {
public:
  /// {col, row, direction, channel}. Queues are per channel and per direction,
  /// and shim tiles are one per column, so the same channel number in two
  /// columns is two independent queues.
  using ChannelKey = std::array<int, 4>;

  static ChannelKey keyOf(AIE::TileOp tile, AIE::DMAChannelDir direction,
                          int64_t channel) {
    return {tile.getCol(), tile.getRow(), static_cast<int>(direction),
            static_cast<int>(channel)};
  }

  /// One channel's queue: its pushes not yet known to have completed, oldest
  /// first, each true when it issues a completion token.
  struct ChannelState {
    llvm::SmallVector<bool, 8> pushes;
    bool unknownTokens = false;
    bool operator==(const ChannelState &other) const {
      return pushes == other.pushes && unknownTokens == other.unknownTokens;
    }
  };

  /// True when a push would land on a full queue. `depth` of 0 means the
  /// target has no queued-task model and nothing can overflow.
  static bool wouldOverflow(const ChannelState &q, uint32_t depth) {
    return depth != 0 && q.pushes.size() >= depth;
  }

  /// Record a push. Every push occupies a slot, not just issue_token ones, and
  /// one push occupies exactly one whatever its repeat_count.
  static void push(ChannelState &q, bool issuesToken) {
    q.pushes.push_back(issuesToken && !q.unknownTokens);
  }

  /// Retire what a task-completion-token await proves has drained. The sync
  /// pops whichever token the channel produces first -- the oldest outstanding
  /// token-issuing push, not necessarily the task the await names -- and
  /// in-order execution means everything queued ahead of it has drained too.
  /// With no token-issuing push queued, nothing can be retired.
  static void awaitToken(ChannelState &q) {
    if (q.unknownTokens)
      return;
    auto *tok = llvm::find(q.pushes, true);
    if (tok != q.pushes.end())
      q.pushes.erase(q.pushes.begin(), std::next(tok));
  }

  /// Record that a queue-space poll has guaranteed room. Keeps the newest
  /// depth-1 entries rather than clearing. Polling does not consume completion
  /// tokens: if it drops a token-bearing entry, later syncs cannot safely be
  /// credited to newer pushes.
  static void noteSpaceGuaranteed(ChannelState &q, uint32_t depth) {
    if (depth > 0 && q.pushes.size() > depth - 1) {
      auto end = q.pushes.end() - (depth - 1);
      q.unknownTokens |=
          llvm::is_contained(llvm::make_range(q.pushes.begin(), end), true);
      q.pushes.erase(q.pushes.begin(), end);
      if (q.unknownTokens)
        llvm::fill(q.pushes, false);
    }
  }

  bool wouldOverflow(const ChannelKey &key, uint32_t depth) const {
    auto it = queued.find(key);
    return it != queued.end() && wouldOverflow(it->second, depth);
  }

  void push(const ChannelKey &key, bool issuesToken) {
    push(queued[key], issuesToken);
  }

  void awaitToken(const ChannelKey &key) {
    if (auto it = queued.find(key); it != queued.end())
      awaitToken(it->second);
  }

  void noteSpaceGuaranteed(const ChannelKey &key, uint32_t depth) {
    noteSpaceGuaranteed(queued[key], depth);
  }

  /// True the first time a channel is reported. An over-subscribed channel
  /// usually stays that way for every later push, and repeating the same
  /// diagnostic per push buries it.
  bool shouldReport(const ChannelKey &key) {
    return reported.insert(key).second;
  }

  /// Outstanding pushes on `key`, for diagnostics.
  size_t outstanding(const ChannelKey &key) const {
    auto it = queued.find(key);
    return it == queued.end() ? 0 : it->second.pushes.size();
  }

  /// Every channel's queue. `reported` is deliberately not part of it: it is
  /// diagnostic bookkeeping, not queue state, and folding it in would stop
  /// two otherwise identical states from comparing equal.
  using State = std::map<ChannelKey, ChannelState>;
  const State &state() const { return queued; }
  void setState(State s) { queued = std::move(s); }

private:
  State queued;
  std::set<ChannelKey> reported;
};

/// The channel an aiex.npu.sync retires on, or nullopt when it cannot be
/// pinned down. Its six operands are exactly a ChannelKey, with direction
/// numbered as AIE::DMAChannelDir because DmaWaitToSyncPattern builds that
/// operand by casting the enum. Unresolvable retires nothing, which
/// over-reports; guessing the other way would hide a real hang.
inline std::optional<DmaQueueModel::ChannelKey> syncChannelKey(NpuSyncOp sync) {
  std::optional<uint32_t> col = getConstantIntOperand(sync.getColumn());
  std::optional<uint32_t> row = getConstantIntOperand(sync.getRow());
  std::optional<uint32_t> dir = getConstantIntOperand(sync.getDirection());
  std::optional<uint32_t> chan = getConstantIntOperand(sync.getChannel());
  std::optional<uint32_t> colNum = getConstantIntOperand(sync.getColumnNum());
  std::optional<uint32_t> rowNum = getConstantIntOperand(sync.getRowNum());
  if (!col || !row || !dir || !chan || !colNum || !rowNum)
    return std::nullopt;
  // Nothing in tree settles whether a sync spanning a range of tiles waits for
  // one token per tile or one for the range -- the cert lowering ignores both
  // fields -- and crediting nothing is the only reading sound either way.
  if (*colNum != 1 || *rowNum != 1)
    return std::nullopt;
  return DmaQueueModel::ChannelKey{
      static_cast<int>(*col), static_cast<int>(*row), static_cast<int>(*dir),
      static_cast<int>(*chan)};
}

/// Retire what an aiex.npu.sync proves has drained. This is the lowered form of
/// npu.dma_wait and the same hardware event, so it retires by the same rule;
/// sequences written or lowered down to raw syncs would otherwise look like
/// they never drain a channel at all.
inline void awaitSync(DmaQueueModel &queue, NpuSyncOp sync) {
  if (std::optional<DmaQueueModel::ChannelKey> key = syncChannelKey(sync))
    queue.awaitToken(*key);
}

/// Shared wording for the two lowering paths, so the explanation does not
/// depend on whether the transfer came from dma_start_task or
/// npu.dma_memcpy_nd.
inline mlir::InFlightDiagnostic
emitQueueOverflowWarning(mlir::Operation *op, int col, int row,
                         AIE::DMAChannelDir dir, uint32_t channel,
                         uint32_t depth, size_t outstanding) {
  // Not op->emitWarning(): its "see current operation" note verifies and
  // numbers the whole device for each warning, quadratic in the sequence.
  return mlir::emitWarning(op->getLoc())
         << "pushes a DMA task onto tile (" << col << "," << row << ") "
         << AIE::stringifyDMAChannelDir(dir) << " channel " << channel
         << ", whose task queue is only " << depth << " deep, with "
         << outstanding
         << " push(es) not yet known to have completed. A push onto a full "
            "queue is dropped by hardware and its transfer never runs, so "
            "whatever waits on it blocks forever. Whether it happens depends "
            "on how fast the consumer drains, which is why this is a warning "
            "and not an error. Drain the channel with an await on this tile, "
            "direction and channel before pushing again";
}

/// Wording for a dma_free_task that releases a buffer descriptor id the
/// runtime pool could hand straight back out before hardware has actually
/// finished the freed transfer -- the rolled-loop counterpart to
/// aie-assign-runtime-sequence-bd-ids's unsafe-BD-reuse warning.
inline mlir::InFlightDiagnostic
emitUnsafeBdReuseWarning(mlir::Operation *freeOp, int col, int row,
                         AIE::DMAChannelDir dir, uint32_t channel) {
  // Not freeOp->emitWarning(), as in emitQueueOverflowWarning.
  return mlir::emitWarning(freeOp->getLoc())
         << "frees a task on tile (" << col << "," << row << ") "
         << AIE::stringifyDMAChannelDir(dir) << " channel " << channel
         << " with no completion guarantee, so its buffer descriptor id "
            "goes back to the runtime pool while the DMA may still be "
            "running; the pool may hand that id straight to the next task "
            "drawn on this tile, reprogramming a BD still in flight. Await "
            "this task, or a later one queued on the same tile, direction "
            "and channel, before freeing it -- an await consumes the oldest "
            "outstanding token, regardless of the task it names";
}

/// Address and mask for a poll that waits for a free task-queue slot.
/// Unavailable when the target reports no pollable
/// occupancy register, or when `depth` is not a power of two -- a maskpoll
/// tests masked equality, not "<", so "occupancy < depth" has to be the single
/// "depth bit is clear" test (for depth 4, bit 22 of Task_Queue_Size).
inline std::optional<std::pair<uint32_t, uint32_t>>
queueSpacePoll(const AIE::AIETargetModel &tm, int col, int row,
               uint32_t channel, AIE::DMAChannelDir dir, uint32_t depth) {
  uint32_t fieldMask = tm.getDmaTaskQueueSizeMask();
  if (!fieldMask || depth == 0 || (depth & (depth - 1)) != 0)
    return std::nullopt;
  std::optional<uint32_t> statusAddr =
      tm.getDmaStatusAddress(col, row, channel, dir);
  if (!statusAddr)
    return std::nullopt;
  uint32_t depthBit = depth << llvm::countr_zero(fieldMask);
  if ((depthBit & fieldMask) != depthBit)
    return std::nullopt;
  return std::make_pair(*statusAddr, depthBit);
}

inline mlir::LogicalResult
insertQueueSpaceWait(mlir::Operation *before, const AIE::AIETargetModel &tm,
                     int col, int row, uint32_t channel, AIE::DMAChannelDir dir,
                     uint32_t depth) {
  auto poll = queueSpacePoll(tm, col, row, channel, dir, depth);
  if (!poll)
    return mlir::failure();
  mlir::OpBuilder b(before);
  mlir::Location loc = before->getLoc();
  auto i32 = b.getI32Type();
  auto cst = [&](uint32_t v) {
    return mlir::arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(v))
        .getResult();
  };
  NpuMaskPollOp::create(b, loc, cst(poll->first), cst(0), cst(poll->second),
                        /*buffer=*/nullptr, /*column=*/nullptr,
                        /*row=*/nullptr);
  return mlir::success();
}

inline constexpr llvm::StringLiteral queueDiagnosedAttr =
    "aiex.dma_queue_overflow_diagnosed";

/// Handle a push on `key` that would land on a full queue: poll where one can
/// be emitted, warn where it cannot (see insertQueueSpaceWait above for when
/// that is). Warning rather than failing keeps a default from rejecting
/// designs that build today on a target the user can do nothing about; the
/// note is what keeps it from declining quietly.
inline void guardQueueOverflow(DmaQueueModel &queue, mlir::Operation *push,
                               const AIE::AIETargetModel &tm,
                               const DmaQueueModel::ChannelKey &key,
                               uint32_t depth, bool enforce) {
  int col = key[0], row = key[1];
  auto dir = static_cast<AIE::DMAChannelDir>(key[2]);
  auto chan = static_cast<uint32_t>(key[3]);

  if (enforce && mlir::succeeded(insertQueueSpaceWait(push, tm, col, row, chan,
                                                      dir, depth))) {
    queue.noteSpaceGuaranteed(key, depth);
    return;
  }
  if (!queue.shouldReport(key))
    return;
  // Keep diagnostics per sequence/channel across task and combined lowering.
  // This marker suppresses only warnings, never queue accounting or guards.
  if (auto seq = push->getParentOfType<AIE::RuntimeSequenceOp>()) {
    auto channel = mlir::DenseI32ArrayAttr::get(push->getContext(), key);
    llvm::SmallVector<mlir::Attribute> diagnosed;
    if (auto previous =
            seq->getAttrOfType<mlir::ArrayAttr>(queueDiagnosedAttr)) {
      if (llvm::is_contained(previous, channel))
        return;
      llvm::append_range(diagnosed, previous);
    }
    diagnosed.push_back(channel);
    seq->setAttr(queueDiagnosedAttr,
                 mlir::ArrayAttr::get(push->getContext(), diagnosed));
  }
  mlir::InFlightDiagnostic diag = emitQueueOverflowWarning(
      push, col, row, dir, chan, depth, queue.outstanding(key));
  if (enforce)
    diag.attachNote() << "the compiler would have waited for a free slot here, "
                         "but this target reports no pollable task-queue "
                         "occupancy register for this channel";
}

/// What one op in a loop body does to a channel queue. The two lowering paths
/// spell pushes and awaits with different ops, so the fixed point below takes
/// the classification from its caller and keeps the FIFO rule to itself.
///
/// Free is a read-only check, not a hardware event: a dma_free_task returns a
/// buffer descriptor id to the compiler's pool without touching the channel,
/// so it neither pushes nor retires anything here. It asks whether `task`'s
/// own push -- the one a later pop could still be waiting behind -- has
/// already been proven complete.
struct QueueEffect {
  enum class Kind { Ignore, Configure, Push, Await, Free };
  Kind kind = Kind::Ignore;
  DmaQueueModel::ChannelKey key{};
  bool issuesToken = false;
  mlir::Value task;

  static QueueEffect configure(DmaQueueModel::ChannelKey key,
                               mlir::Value task) {
    return {Kind::Configure, key, false, task};
  }
  static QueueEffect push(DmaQueueModel::ChannelKey key, bool issuesToken,
                          mlir::Value task = {}) {
    return {Kind::Push, key, issuesToken, task};
  }
  static QueueEffect await(DmaQueueModel::ChannelKey key) {
    return {Kind::Await, key, false, {}};
  }
  static QueueEffect free(DmaQueueModel::ChannelKey key, mlir::Value task) {
    return {Kind::Free, key, false, task};
  }
};

/// Completion is distinct from queue occupancy: Task_Queue_Size excludes the
/// running task. Task identities are path-local and renamed at joins, so a
/// configure executed again in a loop is distinct from the carried old task.
struct DmaCompletionState {
  struct Push {
    bool issuesToken;
    unsigned owner;
    // Stands in for the newest push of an unknown region, whose owners are
    // all in forgottenOwners: none of them is proven complete while it is.
    bool unknownRegion = false;
    bool operator==(const Push &other) const {
      return issuesToken == other.issuesToken && owner == other.owner &&
             unknownRegion == other.unknownRegion;
    }
  };
  llvm::SmallVector<unsigned> tasks;
  llvm::SmallVector<Push, 8> pushes;
  std::set<unsigned> forgottenOwners;
  bool unknownTokens = false;
  bool mayHaveDroppedPushes = false;

  bool operator==(const DmaCompletionState &other) const {
    return tasks == other.tasks && pushes == other.pushes &&
           forgottenOwners == other.forgottenOwners &&
           unknownTokens == other.unknownTokens &&
           mayHaveDroppedPushes == other.mayHaveDroppedPushes;
  }

  // Canonical identities keep loop fixed points finite. Instances no longer
  // named by any SSA task need only retain their token and FIFO position.
  void normalize() {
    llvm::DenseMap<unsigned, unsigned> names;
    unsigned next = 1;
    for (unsigned &task : tasks)
      if (task) {
        auto [it, inserted] = names.try_emplace(task, next);
        next += inserted;
        task = it->second;
      }
    for (Push &push : pushes)
      push.owner = names.lookup(push.owner);
    std::set<unsigned> renamed;
    for (unsigned owner : forgottenOwners)
      if (unsigned name = names.lookup(owner))
        renamed.insert(name);
    forgottenOwners = std::move(renamed);
  }

  void awaitToken() {
    if (unknownTokens)
      return;
    auto tok =
        llvm::find_if(pushes, [](const Push &p) { return p.issuesToken; });
    if (tok != pushes.end()) {
      pushes.erase(pushes.begin(), std::next(tok));
      forgetOwnersIfDrained();
    }
  }

  // Forgotten owners were pushed no later than any remaining unknown-region
  // stand-in; they are complete once every such stand-in has drained too.
  void forgetOwnersIfDrained() {
    if (llvm::none_of(pushes, [](const Push &p) { return p.unknownRegion; }))
      forgottenOwners.clear();
  }

  void noteIdle() {
    // Completion does not consume tokens; later awaits still see these pushes.
    for (Push &push : pushes) {
      push.owner = 0;
      push.unknownRegion = false;
    }
    forgottenOwners.clear();
  }

  // A real space poll leaves at most depth-1 queued tasks AND one running.
  // Abstract truncation instead retains every lost owner's uncertainty.
  void bound(unsigned limit, bool spaceGuaranteed) {
    // Attempted pushes no longer bound FIFO progress after a possible drop.
    spaceGuaranteed &= !mayHaveDroppedPushes;
    bool drained = spaceGuaranteed && pushes.size() >= limit;
    if (pushes.size() > limit) {
      auto end = pushes.end() - limit;
      for (const Push &push : llvm::make_range(pushes.begin(), end)) {
        unknownTokens |= push.issuesToken;
        if (!spaceGuaranteed && push.owner)
          forgottenOwners.insert(push.owner);
      }
      pushes.erase(pushes.begin(), end);
    }
    if (drained)
      forgetOwnersIfDrained();
  }

  bool isProvenComplete(unsigned owner) const {
    return owner && !forgottenOwners.count(owner) &&
           !llvm::any_of(pushes,
                         [&](const Push &p) { return p.owner == owner; });
  }
};

/// Guard every push in a runtime sequence that can land on a full queue, and
/// (when `warnUnsafeBdReuse`) warn at every Free effect whose own push is not
/// yet proven complete along some reachable path.
///
/// Retain all reachable states at branches and loop exits, including zero
/// iterations. Nested loops are analyzed recursively, not walked once. Queue
/// states are finite because overflowing pushes are capped at the target depth;
/// targets without a queued-task model must be ignored rather than accumulated.
/// Analyze each independent channel separately to avoid a Cartesian product
/// of states across channels.
template <typename EffectFn>
inline void guardSequenceQueueDepth(mlir::Region &body, DmaQueueModel &queue,
                                    const AIE::AIETargetModel &tm, bool enforce,
                                    bool warnUnsafeBdReuse, EffectFn effectOf) {
  auto depthOf = [&](const DmaQueueModel::ChannelKey &k) {
    return tm.getDmaTaskQueueDepth(k[0], k[1], k[3],
                                   static_cast<AIE::DMAChannelDir>(k[2]));
  };

  struct PathState {
    DmaQueueModel::ChannelState queue;
    DmaCompletionState completion;
    bool operator==(const PathState &other) const {
      return queue == other.queue && completion == other.completion;
    }
  };
  using States = llvm::SmallVector<PathState, 8>;
  llvm::DenseMap<mlir::Value, unsigned> taskIndices;
  auto track = [&](mlir::Value value) {
    if (value && value.getType().isIndex())
      taskIndices.try_emplace(value, taskIndices.size());
  };
  std::set<DmaQueueModel::ChannelKey> keys;
  for (const auto &[key, channel] : queue.state())
    if (depthOf(key) != 0)
      keys.insert(key);
  llvm::DenseMap<mlir::Operation *, QueueEffect> effects;
  body.walk([&](mlir::Operation *op) {
    if (warnUnsafeBdReuse) {
      for (mlir::Value value : op->getOperands())
        track(value);
      for (mlir::Value value : op->getResults())
        track(value);
      for (mlir::Region &region : op->getRegions())
        for (mlir::Block &block : region)
          for (mlir::Value arg : block.getArguments())
            track(arg);
    }
    QueueEffect e = effectOf(op);
    if (!warnUnsafeBdReuse) {
      e.task = {};
      if (e.kind == QueueEffect::Kind::Configure ||
          e.kind == QueueEffect::Kind::Free)
        return;
    }
    if (e.kind != QueueEffect::Kind::Ignore && depthOf(e.key) != 0) {
      keys.insert(e.key);
      effects[op] = e;
    }
  });

  // Each channel's analysis visits only its own pushes, awaits and polls and
  // the ops enclosing them, in block order, rather than the whole sequence.
  std::map<uint32_t, DmaQueueModel::ChannelKey> keyByStatus;
  for (const auto &key : keys)
    if (auto status = tm.getDmaStatusAddress(
            key[0], key[1], key[3], static_cast<AIE::DMAChannelDir>(key[2])))
      keyByStatus[*status] = key;
  using Visits =
      llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Operation *>>;
  std::map<DmaQueueModel::ChannelKey, Visits> visits;
  auto visit = [&](mlir::Operation *op, const DmaQueueModel::ChannelKey &key) {
    Visits &blocks = visits[key];
    for (mlir::Operation *at = op; at != body.getParentOp();
         at = at->getParentOp()) {
      auto &ops = blocks[at->getBlock()];
      if (!ops.empty() && ops.back() == at)
        break;
      ops.push_back(at);
    }
  };
  body.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (warnUnsafeBdReuse && mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp>(op))
      for (const auto &key : keys)
        visit(op, key);
    if (auto it = effects.find(op); it != effects.end()) {
      visit(op, it->second.key);
    } else if (auto poll = mlir::dyn_cast<NpuMaskPollOp>(op)) {
      if (auto address = poll.getAbsoluteAddress())
        if (auto it = keyByStatus.find(*address); it != keyByStatus.end())
          visit(op, it->second);
    }
  });

  DmaQueueModel::State entry = queue.state();
  DmaQueueModel::State merged = entry;
  for (const auto &key : keys) {
    uint32_t depth = depthOf(key);
    const Visits &channelVisits = visits[key];
    auto channelEffectOf = [&](mlir::Operation *op) {
      auto it = effects.find(op);
      return it != effects.end() && it->second.key == key ? it->second
                                                          : QueueEffect{};
    };
    llvm::SetVector<mlir::Operation *> overflowing;
    llvm::SetVector<mlir::Operation *> unsafeFrees;
    std::map<mlir::Operation *, DmaQueueModel::ChannelState> witnesses;
    bool trackCompletion = warnUnsafeBdReuse;
    auto warnUnsafeFree = [&](mlir::Operation *op) {
      if (warnUnsafeBdReuse && unsafeFrees.insert(op)) {
        auto diag = emitUnsafeBdReuseWarning(
            op, key[0], key[1], static_cast<AIE::DMAChannelDir>(key[2]),
            key[3]);
        if (!trackCompletion)
          diag.attachNote() << "task ownership analysis budget exhausted; "
                               "completion is conservatively unproven";
      }
    };
    size_t completionWork = 0;
    auto equivalent = [&](const PathState &a, const PathState &b) {
      return a.queue == b.queue &&
             (!trackCompletion || a.completion == b.completion);
    };
    auto contains = [&](const States &states, const PathState &state) {
      return llvm::any_of(states, [&](const PathState &other) {
        return equivalent(state, other);
      });
    };
    auto compact = [&](States &states) {
      States unique;
      for (const auto &state : states)
        if (!contains(unique, state))
          unique.push_back(state);
      states = std::move(unique);
    };
    // Bound optional ownership analysis without weakening queue-depth guards.
    // Beyond the budget, all frees are uncertain and only occupancy is tracked.
    auto append = [&](States &to, const States &from) {
      if (warnUnsafeBdReuse && !trackCompletion)
        compact(to);
      for (const auto &state : from)
        if (!contains(to, state)) {
          if (trackCompletion && to.size() >= 256) {
            trackCompletion = false;
            compact(to);
            if (contains(to, state))
              continue;
          }
          to.push_back(state);
        }
    };
    auto dedupe = [&](States &states) {
      if (trackCompletion)
        for (auto &state : states)
          state.completion.normalize();
      if (states.size() < 2)
        return;
      States unique;
      append(unique, states);
      states = std::move(unique);
    };
    auto transfer = [&](States &states, mlir::ValueRange from,
                        mlir::ValueRange to) {
      if (!trackCompletion)
        return;
      for (auto &state : states) {
        auto previous = state.completion.tasks;
        for (auto [source, dest] : llvm::zip_equal(from, to)) {
          auto src = taskIndices.find(source), dst = taskIndices.find(dest);
          if (src != taskIndices.end() && dst != taskIndices.end())
            state.completion.tasks[dst->second] = previous[src->second];
        }
      }
      dedupe(states);
    };
    auto forgetScope = [&](States &states, mlir::Region &region,
                           bool keepArguments = false) {
      if (!trackCompletion)
        return;
      for (auto [value, index] : taskIndices) {
        mlir::Region *parent = value.getParentRegion();
        if (parent != &region && !region.isAncestor(parent))
          continue;
        if (keepArguments)
          if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value))
            if (arg.getOwner() == &region.front())
              continue;
        for (auto &state : states)
          state.completion.tasks[index] = 0;
      }
      dedupe(states);
    };
    bool canPoll =
        enforce &&
        queueSpacePoll(tm, key[0], key[1], key[3],
                       static_cast<AIE::DMAChannelDir>(key[2]), depth)
            .has_value();

    // Earlier passes may already have guarded a start. Credit only polls whose
    // constant address and mask prove space or idleness on a tracked channel.
    enum class PollEffect { Ignore, Space, Idle };
    auto pollEffectOf = [&](NpuMaskPollOp poll) {
      auto address = poll.getAbsoluteAddress();
      auto mask = getConstantIntOperand(poll.getMask());
      auto value = getConstantIntOperand(poll.getValue());
      if (!address || !mask || !value)
        return PollEffect::Ignore;
      auto status = tm.getDmaStatusAddress(
          key[0], key[1], key[3], static_cast<AIE::DMAChannelDir>(key[2]));
      if (status != address)
        return PollEffect::Ignore;
      uint32_t idleMask = tm.getDmaChannelIdleMask();
      if (idleMask && (*mask & idleMask) == idleMask &&
          (*value & idleMask) == 0)
        return PollEffect::Idle;
      uint32_t fieldMask = tm.getDmaTaskQueueSizeMask();
      if (!fieldMask || (depth & (depth - 1)) != 0)
        return PollEffect::Ignore;
      uint32_t depthBit = depth << llvm::countr_zero(fieldMask);
      return depthBit && (depthBit & fieldMask) == depthBit &&
                     (*mask & depthBit) == depthBit && (*value & depthBit) == 0
                 ? PollEffect::Space
                 : PollEffect::Ignore;
    };

    // Unknown region semantics (including unstructured CFGs) must not credit
    // conditional waits. Guard their pushes and conservatively forget tokens.
    auto unknownRegion = [&](mlir::Region &region, States states) {
      region.walk([&](mlir::Operation *op) {
        QueueEffect e = channelEffectOf(op);
        if (e.kind == QueueEffect::Kind::Free)
          warnUnsafeFree(op);
        if (e.kind != QueueEffect::Kind::Push)
          return;
        for (auto &state : states) {
          state.queue.pushes.assign(depth, false);
          state.queue.unknownTokens = true;
          if (trackCompletion) {
            state.completion.unknownTokens = true;
            state.completion.mayHaveDroppedPushes |= !canPoll;
            for (unsigned owner : state.completion.tasks)
              if (owner)
                state.completion.forgottenOwners.insert(owner);
            // Later polls must count this push before crediting its owners.
            state.completion.pushes.push_back(
                {/*issuesToken=*/false, /*owner=*/0, /*unknownRegion=*/true});
            state.completion.bound(depth + 1, /*spaceGuaranteed=*/false);
          }
          if (overflowing.insert(op))
            witnesses[op] = state.queue;
        }
      });
      dedupe(states);
      return states;
    };

    std::function<States(mlir::Region &, States)> analyze;
    analyze = [&](mlir::Region &region, States states) -> States {
      if (!region.hasOneBlock())
        return unknownRegion(region, std::move(states));
      auto blockVisits = channelVisits.find(&region.front());
      if (blockVisits == channelVisits.end())
        return states;
      for (mlir::Operation *visited : blockVisits->second) {
        if (trackCompletion) {
          completionWork += states.size();
          if (completionWork > 16384) {
            trackCompletion = false;
            compact(states);
          }
        }
        mlir::Operation &op = *visited;
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(&op)) {
          auto lb = mlir::getConstantIntValue(loop.getLowerBound());
          auto ub = mlir::getConstantIntValue(loop.getUpperBound());
          if (lb && ub && *lb >= *ub &&
              (!loop->hasAttr("unsignedCmp") || (*lb >= 0 && *ub >= 0))) {
            transfer(states, loop.getInitArgs(), loop.getResults());
            continue;
          }
          // Every loop header state is a possible exit for an unknown trip
          // count. Include the entry: even an await-only loop might execute
          // zero times.
          transfer(states, loop.getInitArgs(), loop.getRegionIterArgs());
          States reachable = states;
          for (size_t i = 0; i < reachable.size();) {
            bool wasTrackingCompletion = trackCompletion;
            States next = analyze(loop.getRegion(), States{reachable[i]});
            transfer(next, loop.getYieldedValues(), loop.getRegionIterArgs());
            forgetScope(next, loop.getRegion(), /*keepArguments=*/true);
            append(reachable, next);
            if (wasTrackingCompletion && !trackCompletion) {
              // Revisit earlier paths with uncertain ownership as well.
              compact(reachable);
              i = 0;
            } else {
              ++i;
            }
          }
          transfer(reachable, loop.getRegionIterArgs(), loop.getResults());
          forgetScope(reachable, loop.getRegion());
          states = std::move(reachable);
          continue;
        }
        if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(&op)) {
          States alternatives = analyze(branch.getThenRegion(), states);
          States otherwise = analyze(branch.getElseRegion(), states);
          if (branch.getNumResults()) {
            transfer(alternatives, branch.thenYield().getResults(),
                     branch.getResults());
            transfer(otherwise, branch.elseYield().getResults(),
                     branch.getResults());
          }
          forgetScope(alternatives, branch.getThenRegion());
          forgetScope(otherwise, branch.getElseRegion());
          append(alternatives, otherwise);
          states = std::move(alternatives);
          continue;
        }
        for (mlir::Region &nested : op.getRegions())
          states = unknownRegion(nested, std::move(states));

        if (auto poll = mlir::dyn_cast<NpuMaskPollOp>(&op))
          if (auto effect = pollEffectOf(poll); effect != PollEffect::Ignore) {
            for (auto &state : states) {
              DmaQueueModel::noteSpaceGuaranteed(state.queue, depth);
              if (trackCompletion) {
                if (effect == PollEffect::Idle)
                  state.completion.noteIdle();
                else
                  state.completion.bound(depth, /*spaceGuaranteed=*/true);
              }
            }
            dedupe(states);
          }

        QueueEffect e = channelEffectOf(&op);
        if (e.kind == QueueEffect::Kind::Ignore)
          continue;
        if (e.kind == QueueEffect::Kind::Free) {
          if (warnUnsafeBdReuse)
            for (const auto &state : states)
              if (!trackCompletion ||
                  !state.completion.isProvenComplete(
                      state.completion.tasks[taskIndices.lookup(e.task)]))
                warnUnsafeFree(&op);
          continue;
        }
        for (auto &state : states) {
          if (e.kind == QueueEffect::Kind::Configure) {
            if (trackCompletion)
              state.completion.tasks[taskIndices.lookup(e.task)] =
                  taskIndices.size() + 1;
            continue;
          }
          if (e.kind == QueueEffect::Kind::Await) {
            DmaQueueModel::awaitToken(state.queue);
            if (trackCompletion)
              state.completion.awaitToken();
            continue;
          }
          if (DmaQueueModel::wouldOverflow(state.queue, depth)) {
            if (overflowing.insert(&op))
              witnesses[&op] = state.queue;
            DmaQueueModel::noteSpaceGuaranteed(state.queue, depth);
            if (trackCompletion && canPoll)
              state.completion.bound(depth, /*spaceGuaranteed=*/true);
            else if (trackCompletion)
              state.completion.mayHaveDroppedPushes = true;
          }
          DmaQueueModel::push(state.queue, e.issuesToken);
          if (trackCompletion) {
            unsigned owner =
                e.task ? state.completion.tasks[taskIndices.lookup(e.task)] : 0;
            state.completion.pushes.push_back({e.issuesToken, owner});
            state.completion.bound(depth + 1, /*spaceGuaranteed=*/false);
          }
        }
        dedupe(states);
      }
      return states;
    };

    DmaQueueModel::ChannelState channelEntry;
    if (auto it = entry.find(key); it != entry.end())
      channelEntry = it->second;
    PathState initial{channelEntry, {}};
    initial.completion.tasks.resize(taskIndices.size());
    States exits = analyze(body, States{initial});
    for (mlir::Operation *push : overflowing) {
      queue.setState({{key, witnesses[push]}});
      guardQueueOverflow(queue, push, tm, key, depth, enforce);
    }

    // Preserve exact straight-line state for callers. If control-flow paths
    // disagree, retain the largest occupancy without crediting ambiguous
    // tokens.
    DmaQueueModel::ChannelState channelExit = exits.front().queue;
    for (const auto &state : exits)
      if (!(channelExit == state.queue)) {
        channelExit.pushes.assign(
            std::max(channelExit.pushes.size(), state.queue.pushes.size()),
            false);
        channelExit.unknownTokens = true;
      }
    merged[key] = channelExit;
  }
  queue.setState(std::move(merged));
}

/// Check that `sequence` ends, on every path, with no task-completion token
/// outstanding on a channel in `awaited`, the channels some sequence of the
/// device awaits. A token left at the end outlives the command: the next
/// command's await on that channel takes it and returns before its own
/// transfer has finished (measured on Strix, NPU firmware 1.1.2.64), and every
/// completion the compiler derives from an await assumes the channel started
/// with none. A token on a channel nothing awaits is never taken. An await
/// consumes the oldest token, so only the count matters. A loop with an
/// unknown trip count must leave the count where it found it, or what remains
/// grows with the trip count. Regions the check cannot follow are not
/// diagnosed.
template <typename EffectFn>
inline mlir::LogicalResult
checkTokensConsumed(AIE::RuntimeSequenceOp sequence,
                    const std::set<DmaQueueModel::ChannelKey> &awaited,
                    EffectFn effectOf) {
  mlir::Region &body = sequence.getBody();
  // Each channel's analysis visits only its own pushes and syncs and the ops
  // enclosing them, in block order.
  llvm::DenseMap<mlir::Operation *, QueueEffect> effects;
  using Visits =
      llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Operation *>>;
  std::map<DmaQueueModel::ChannelKey, Visits> visits;
  body.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    QueueEffect e = effectOf(op);
    if ((e.kind != QueueEffect::Kind::Push &&
         e.kind != QueueEffect::Kind::Await) ||
        !awaited.count(e.key))
      return;
    effects[op] = e;
    Visits &blocks = visits[e.key];
    for (mlir::Operation *at = op; at != body.getParentOp();
         at = at->getParentOp()) {
      auto &ops = blocks[at->getBlock()];
      if (!ops.empty() && ops.back() == at)
        break;
      ops.push_back(at);
    }
  });

  auto describe = [](mlir::InFlightDiagnostic &diag,
                     const DmaQueueModel::ChannelKey &key) {
    diag << " on tile (" << key[0] << "," << key[1] << ") "
         << AIE::stringifyDMAChannelDir(static_cast<AIE::DMAChannelDir>(key[2]))
         << " channel " << key[3];
  };
  // Small constant trip counts are followed pass by pass, as unrolling would.
  constexpr int64_t maxFollowedTrips = 64;
  bool consumed = true;
  for (const auto &channel : visits) {
    const DmaQueueModel::ChannelKey &key = channel.first;
    const Visits &blocks = channel.second;
    using Counts = std::set<int64_t>;
    bool unknown = false;
    std::function<Counts(mlir::Region &, Counts)> analyze;
    analyze = [&](mlir::Region &region, Counts counts) -> Counts {
      if (region.empty())
        return counts;
      if (!region.hasOneBlock()) {
        unknown = true;
        return counts;
      }
      auto visited = blocks.find(&region.front());
      if (visited == blocks.end())
        return counts;
      for (mlir::Operation *at : visited->second) {
        mlir::Operation &op = *at;
        if (unknown)
          return counts;
        if (auto it = effects.find(&op); it != effects.end()) {
          Counts next;
          for (int64_t count : counts)
            next.insert(it->second.kind == QueueEffect::Kind::Push
                            ? count + it->second.issuesToken
                            : std::max<int64_t>(count - 1, 0));
          counts = std::move(next);
          continue;
        }
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(&op)) {
          auto lb = mlir::getConstantIntValue(loop.getLowerBound());
          auto ub = mlir::getConstantIntValue(loop.getUpperBound());
          auto step = mlir::getConstantIntValue(loop.getStep());
          if (lb && ub && step && *step > 0 && !loop->hasAttr("unsignedCmp")) {
            int64_t trips = *ub > *lb ? llvm::divideCeil(*ub - *lb, *step) : 0;
            if (trips <= maxFollowedTrips) {
              for (int64_t i = 0; i < trips && !unknown; i++)
                counts = analyze(loop.getRegion(), counts);
              continue;
            }
          }
          for (int64_t count : counts) {
            Counts after = analyze(loop.getRegion(), Counts{count});
            if (unknown)
              return counts;
            if (after == Counts{count})
              continue;
            auto diag = loop.emitOpError("changes how many task-completion "
                                         "tokens are outstanding");
            describe(diag, key);
            diag << " in each pass, so how many it leaves depends on its trip "
                    "count; await as many tokens in each pass as it issues";
            consumed = false;
            unknown = true;
            return counts;
          }
          continue;
        }
        if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(&op)) {
          Counts taken = analyze(branch.getThenRegion(), counts);
          Counts otherwise = analyze(branch.getElseRegion(), counts);
          taken.insert(otherwise.begin(), otherwise.end());
          counts = std::move(taken);
          continue;
        }
        unknown = true;
        return counts;
      }
      return counts;
    };
    Counts exits = analyze(body, Counts{0});
    if (unknown || *exits.rbegin() == 0)
      continue;
    int64_t left = *exits.rbegin();
    // Not sequence.emitOpError(): that would print the whole sequence.
    auto diag = mlir::emitError(sequence.getLoc())
                << "ends with " << left << " task-completion token"
                << (left > 1 ? "s" : "") << " no await consumes";
    describe(diag, key);
    diag << "; the next command's await on that channel would return for "
            "this sequence's transfer instead of its own. Await every "
            "token-issuing transfer before the sequence ends, or start it "
            "without issue_token";
    consumed = false;
  }
  return mlir::success(consumed);
}

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_DMAQUEUEMODEL_H
