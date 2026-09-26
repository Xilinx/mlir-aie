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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

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

  /// True when a push on `key` would land on a full queue. `depth` of 0 means
  /// the target has no queued-task model and nothing can overflow.
  bool wouldOverflow(const ChannelKey &key, uint32_t depth) const {
    if (depth == 0)
      return false;
    auto it = queued.find(key);
    return it != queued.end() && it->second.pushes.size() >= depth;
  }

  /// Record a push. Every push occupies a slot, not just issue_token ones, and
  /// one push occupies exactly one whatever its repeat_count. (A task start
  /// whose count overflows the push's field is several pushes; the BD-ID pass
  /// splits it before counting.)
  void push(const ChannelKey &key, bool issuesToken) {
    auto &q = queued[key];
    q.pushes.push_back(issuesToken && !q.unknownTokens);
  }

  /// Retire what a task-completion-token await proves has drained. The sync
  /// pops whichever token the channel produces first -- the oldest outstanding
  /// token-issuing push, not necessarily the task the await names -- and
  /// in-order execution means everything queued ahead of it has drained too.
  /// With no token-issuing push queued, nothing can be retired.
  void awaitToken(const ChannelKey &key) {
    auto it = queued.find(key);
    if (it == queued.end())
      return;
    auto &q = it->second;
    if (q.unknownTokens)
      return;
    auto *tok = llvm::find(q.pushes, true);
    if (tok != q.pushes.end())
      q.pushes.erase(q.pushes.begin(), std::next(tok));
  }

  /// Record that a queue-space poll has guaranteed room on `key`. Keeps the
  /// newest depth-1 entries rather than clearing. Polling does not consume
  /// completion tokens: if it drops a token-bearing entry, later syncs cannot
  /// safely be credited to newer pushes.
  void noteSpaceGuaranteed(const ChannelKey &key, uint32_t depth) {
    auto &q = queued[key];
    if (depth > 0 && q.pushes.size() > depth - 1) {
      auto end = q.pushes.end() - (depth - 1);
      q.unknownTokens |=
          llvm::is_contained(llvm::make_range(q.pushes.begin(), end), true);
      q.pushes.erase(q.pushes.begin(), end);
      if (q.unknownTokens)
        llvm::fill(q.pushes, false);
    }
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

  /// The whole per-channel queue, which is what the rolled-loop fixed point
  /// below iterates on. `reported` is deliberately not part of it: it is
  /// diagnostic bookkeeping, not queue state, and folding it in would stop
  /// two otherwise identical states from comparing equal.
  struct ChannelState {
    llvm::SmallVector<bool, 8> pushes;
    bool unknownTokens = false;
    bool operator==(const ChannelState &other) const {
      return pushes == other.pushes && unknownTokens == other.unknownTokens;
    }
  };
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
  return op->emitWarning()
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

/// Emit a poll that blocks until the channel's task queue has a free slot,
/// immediately before `before`. Fails when the target reports no pollable
/// occupancy register, or when `depth` is not a power of two -- a maskpoll
/// tests masked equality, not "<", so "occupancy < depth" has to be the single
/// "depth bit is clear" test (for depth 4, bit 22 of Task_Queue_Size).
inline mlir::LogicalResult
insertQueueSpaceWait(mlir::Operation *before, const AIE::AIETargetModel &tm,
                     int col, int row, uint32_t channel, AIE::DMAChannelDir dir,
                     uint32_t depth) {
  uint32_t fieldMask = tm.getDmaTaskQueueSizeMask();
  if (!fieldMask || depth == 0 || (depth & (depth - 1)) != 0)
    return mlir::failure();
  std::optional<uint32_t> statusAddr =
      tm.getDmaStatusAddress(col, row, channel, dir);
  if (!statusAddr)
    return mlir::failure();
  uint32_t depthBit = depth << llvm::countr_zero(fieldMask);
  if ((depthBit & fieldMask) != depthBit)
    return mlir::failure();

  mlir::OpBuilder b(before);
  mlir::Location loc = before->getLoc();
  auto i32 = b.getI32Type();
  auto cst = [&](uint32_t v) {
    return mlir::arith::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(v))
        .getResult();
  };
  NpuMaskPollOp::create(b, loc, cst(*statusAddr), cst(0), cst(depthBit),
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
struct QueueEffect {
  enum class Kind { Ignore, Push, Await };
  Kind kind = Kind::Ignore;
  DmaQueueModel::ChannelKey key{};
  bool issuesToken = false;

  static QueueEffect push(DmaQueueModel::ChannelKey key, bool issuesToken) {
    return {Kind::Push, key, issuesToken};
  }
  static QueueEffect await(DmaQueueModel::ChannelKey key) {
    return {Kind::Await, key, false};
  }
};

/// Guard every push in a runtime sequence that can land on a full queue.
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
                                    EffectFn effectOf) {
  auto depthOf = [&](const DmaQueueModel::ChannelKey &k) {
    return tm.getDmaTaskQueueDepth(k[0], k[1], k[3],
                                   static_cast<AIE::DMAChannelDir>(k[2]));
  };

  using States = llvm::SmallVector<DmaQueueModel::State, 8>;
  std::set<DmaQueueModel::ChannelKey> keys;
  for (const auto &[key, channel] : queue.state())
    if (depthOf(key) != 0)
      keys.insert(key);
  body.walk([&](mlir::Operation *op) {
    QueueEffect e = effectOf(op);
    if (e.kind != QueueEffect::Kind::Ignore && depthOf(e.key) != 0)
      keys.insert(e.key);
  });
  DmaQueueModel::State entry = queue.state();
  DmaQueueModel::State merged = entry;
  for (const auto &key : keys) {
    uint32_t depth = depthOf(key);
    auto channelEffectOf = [&](mlir::Operation *op) {
      QueueEffect e = effectOf(op);
      return e.key == key ? e : QueueEffect{};
    };
    llvm::SetVector<mlir::Operation *> overflowing;
    std::map<mlir::Operation *, DmaQueueModel::State> witnesses;
    auto append = [](States &to, const States &from) {
      for (const auto &state : from)
        if (!llvm::is_contained(to, state))
          to.push_back(state);
    };

    // Earlier passes may already have guarded a start. Credit only polls whose
    // constant address and depth bit prove space on a channel tracked here.
    auto pollsChannel = [&](NpuMaskPollOp poll) {
      auto address = poll.getAbsoluteAddress();
      auto mask = getConstantIntOperand(poll.getMask());
      auto value = getConstantIntOperand(poll.getValue());
      uint32_t fieldMask = tm.getDmaTaskQueueSizeMask();
      if (!address || !mask || !value || !fieldMask ||
          (depth & (depth - 1)) != 0)
        return false;
      uint32_t depthBit = depth << llvm::countr_zero(fieldMask);
      auto status = tm.getDmaStatusAddress(
          key[0], key[1], key[3], static_cast<AIE::DMAChannelDir>(key[2]));
      return depthBit && status == address &&
             (depthBit & fieldMask) == depthBit &&
             (*mask & depthBit) == depthBit && (*value & depthBit) == 0;
    };

    // Unknown region semantics (including unstructured CFGs) must not credit
    // conditional waits. Guard their pushes and conservatively forget tokens.
    auto unknownRegion = [&](mlir::Region &region, States states) {
      region.walk([&](mlir::Operation *op) {
        QueueEffect e = channelEffectOf(op);
        if (e.kind != QueueEffect::Kind::Push)
          return;
        for (auto &state : states) {
          auto &channel = state[e.key];
          channel.pushes.assign(depth, false);
          channel.unknownTokens = true;
          if (overflowing.insert(op))
            witnesses[op] = state;
        }
      });
      States unique;
      append(unique, states);
      return unique;
    };

    std::function<States(mlir::Region &, States)> analyze;
    analyze = [&](mlir::Region &region, States states) -> States {
      if (!region.hasOneBlock())
        return unknownRegion(region, std::move(states));
      for (mlir::Operation &op : region.front()) {
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(&op)) {
          auto lb = mlir::getConstantIntValue(loop.getLowerBound());
          auto ub = mlir::getConstantIntValue(loop.getUpperBound());
          if (lb && ub && *lb >= *ub &&
              (!loop->hasAttr("unsignedCmp") || (*lb >= 0 && *ub >= 0)))
            continue;
          // Every loop header state is a possible exit for an unknown trip
          // count. Include the entry: even an await-only loop might execute
          // zero times.
          States reachable = states;
          for (size_t i = 0; i < reachable.size(); ++i) {
            States next = analyze(loop.getRegion(), States{reachable[i]});
            append(reachable, next);
          }
          states = std::move(reachable);
          continue;
        }
        if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(&op)) {
          States alternatives = analyze(branch.getThenRegion(), states);
          append(alternatives, analyze(branch.getElseRegion(), states));
          states = std::move(alternatives);
          continue;
        }
        for (mlir::Region &nested : op.getRegions())
          states = unknownRegion(nested, std::move(states));

        if (auto poll = mlir::dyn_cast<NpuMaskPollOp>(&op))
          if (pollsChannel(poll)) {
            States polled;
            for (const auto &state : states) {
              DmaQueueModel path;
              path.setState(state);
              path.noteSpaceGuaranteed(key, depth);
              append(polled, States{path.state()});
            }
            states = std::move(polled);
          }

        QueueEffect e = channelEffectOf(&op);
        if (e.kind == QueueEffect::Kind::Ignore)
          continue;
        States next;
        for (const auto &state : states) {
          DmaQueueModel path;
          path.setState(state);
          if (e.kind == QueueEffect::Kind::Await) {
            path.awaitToken(e.key);
          } else {
            if (path.wouldOverflow(e.key, depth)) {
              if (overflowing.insert(&op))
                witnesses[&op] = state;
              path.noteSpaceGuaranteed(e.key, depth);
            }
            path.push(e.key, e.issuesToken);
          }
          append(next, States{path.state()});
        }
        states = std::move(next);
      }
      return states;
    };

    DmaQueueModel::State channelEntry;
    if (auto it = entry.find(key); it != entry.end())
      channelEntry.insert(*it);
    States exits = analyze(body, States{channelEntry});
    for (mlir::Operation *push : overflowing) {
      queue.setState(witnesses[push]);
      guardQueueOverflow(queue, push, tm, key, depth, enforce);
    }

    // Preserve exact straight-line state for callers. If control-flow paths
    // disagree, retain the largest occupancy without crediting ambiguous
    // tokens.
    DmaQueueModel::State channelExit = exits.front();
    for (const auto &state : exits)
      for (const auto &[key, channel] : state) {
        auto &dest = channelExit[key];
        if (!(dest == channel)) {
          dest.pushes.assign(
              std::max(dest.pushes.size(), channel.pushes.size()), false);
          dest.unknownTokens = true;
        }
      }
    if (auto it = channelExit.find(key); it != channelExit.end())
      merged[key] = it->second;
  }
  queue.setState(std::move(merged));
}

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_DMAQUEUEMODEL_H
