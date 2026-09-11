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
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstdint>
#include <map>
#include <set>

namespace xilinx::AIEX {

/// Tracks, per DMA channel, the pushes not yet known to have completed. Valid
/// only over straight-line IR walked in program order: both users run after the
/// runtime sequence has been unrolled, so a single pass is exact.
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
    return it != queued.end() && it->second.size() >= depth;
  }

  /// Record a push. Every push occupies a slot, not just issue_token ones, and
  /// a repeat_count of N still occupies exactly one.
  void push(const ChannelKey &key, bool issuesToken) {
    queued[key].push_back(issuesToken);
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
    auto *tok = llvm::find(q, true);
    if (tok != q.end())
      q.erase(q.begin(), std::next(tok));
  }

  /// Record that a queue-space poll has guaranteed room on `key`. Keeps the
  /// newest depth-1 entries rather than clearing, so their issue_token flags
  /// survive for awaitToken().
  void noteSpaceGuaranteed(const ChannelKey &key, uint32_t depth) {
    auto &q = queued[key];
    if (depth > 0 && q.size() > depth - 1)
      q.erase(q.begin(), q.end() - (depth - 1));
  }

  /// True the first time a channel is reported. An over-subscribed channel
  /// usually stays that way for every later push, and repeating the same
  /// diagnostic per push buries it.
  bool shouldReport(const ChannelKey &key) { return reported.insert(key).second; }

  /// Outstanding pushes on `key`, for diagnostics.
  size_t outstanding(const ChannelKey &key) const {
    auto it = queued.find(key);
    return it == queued.end() ? 0 : it->second.size();
  }

private:
  std::map<ChannelKey, llvm::SmallVector<bool, 8>> queued;
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
/// depend on whether the transfer came from dma_start_task or npu.dma_memcpy_nd.
inline void emitQueueOverflowWarning(mlir::Operation *op, int col, int row,
                                     AIE::DMAChannelDir dir, uint32_t channel,
                                     uint32_t depth, size_t outstanding) {
  op->emitWarning()
      << "pushes a DMA task onto tile (" << col << "," << row << ") "
      << AIE::stringifyDMAChannelDir(dir) << " channel " << channel
      << ", whose task queue is only " << depth << " deep, with " << outstanding
      << " push(es) not yet known to have completed. A push onto a full queue "
         "is dropped by hardware and its transfer never runs, so whatever "
         "waits on it blocks forever. Whether it happens depends on how fast "
         "the consumer drains, which is why this is a warning and not an "
         "error. Drain the channel with an await, or enable enforce-queue-depth "
         "to make the compiler wait for a free slot";
}

/// Emit a poll that blocks until the channel's task queue has a free slot,
/// immediately before `before`. Fails when the target reports no pollable
/// occupancy register, or when `depth` is not a power of two -- a maskpoll
/// tests masked equality, not "<", so "occupancy < depth" has to be the single
/// "depth bit is clear" test (for depth 4, bit 22 of Task_Queue_Size).
inline mlir::LogicalResult
insertQueueSpaceWait(mlir::Operation *before, const AIE::AIETargetModel &tm,
                     int col, int row, uint32_t channel,
                     AIE::DMAChannelDir dir, uint32_t depth) {
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

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_DMAQUEUEMODEL_H
