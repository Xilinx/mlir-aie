//===- AIEDeadlockModel.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEDEADLOCKMODEL_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEDEADLOCKMODEL_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEStreamDependencyAnalysis.h"

#include "llvm/ADT/SmallVector.h"

#include <map>
#include <optional>
#include <set>
#include <string>

namespace xilinx::AIE {

/// The deadlock model of docs/DeadlockModel.md for one dispatch of one runtime
/// sequence, for designs whose counts the IR fixes. Within the subset it
/// decides (no agent races another for a lock or a receiver), one run in any
/// order is every run, so `run` executes one.
class DeadlockModel {
public:
  struct LockEvent {
    unsigned lock;
    LockAction action;
    int64_t value;
    mlir::Operation *op;
  };

  struct BD {
    uint64_t words;
    std::optional<LockEvent> acquire, release;
    std::optional<int> packet;
    mlir::Operation *op;
  };

  /// BDs a channel runs in order, `passes` times, or forever from `loopTo`.
  struct Chain {
    llvm::SmallVector<BD> bds;
    uint64_t passes = 1;
    std::optional<size_t> loopTo;
    bool token = false;
    mlir::Operation *op = nullptr;
  };

  struct HostOp {
    enum class Kind { Push, Sync, Free, Set };
    Kind kind;
    TileDMAChannel channel{};
    /// Push: the chain; Free: the push it frees.
    unsigned chain = 0;
    unsigned lock = 0;
    int64_t value = 0;
    mlir::Operation *op = nullptr;
  };

  struct Core {
    TileID tile;
    llvm::SmallVector<LockEvent> prefix, cycle;
    mlir::Operation *op;
  };

  using Note = std::pair<mlir::Operation *, std::string>;

  struct Outcome {
    bool deadlock = false;
    /// At a deadlock, each agent blocked and what it waits for.
    llvm::SmallVector<Note> blocked;
    /// At the end of a finished run, what is still in flight.
    llvm::SmallVector<Note> inFlight;
    /// Completion tokens left at the end, by channel.
    std::map<TileDMAChannel, uint64_t> leftoverTokens;
  };

  /// The channels the runtime sequence waits on.
  const std::set<TileDMAChannel> &waitedChannels() const { return waited; }
  std::string describe(const TileDMAChannel &channel) const;

  /// Reads `device` for one dispatch of `sequence`; with no sequence, the
  /// design runs on its own until nothing can move.
  DeadlockModel(DeviceOp device, RuntimeSequenceOp sequence);

  /// Why the design is outside what the model decides; empty when it is not.
  llvm::ArrayRef<Note> outside() const { return outsideNotes; }

  /// One run, with `buffering` words held in front of each receiver and a
  /// shim MM2S channel reading up to `shimBuffering` words ahead of the
  /// stream before it reports its task complete.
  Outcome run(uint64_t buffering, uint64_t shimBuffering) const;

private:
  std::set<TileDMAChannel> waited;

  llvm::SmallVector<std::string> lockNames;
  llvm::SmallVector<int64_t> lockInit;
  llvm::SmallVector<Core> cores;
  std::map<TileDMAChannel, Chain> statics;
  llvm::SmallVector<Chain> pushed;
  llvm::SmallVector<HostOp> host;
  /// A sending channel and the packet id it stamps (-1 for none), to the
  /// receivers its stream reaches.
  std::map<std::pair<TileDMAChannel, int>, llvm::SmallVector<TileDMAChannel>>
      sends;
  std::map<TileDMAChannel, bool> keepsHeader;
  llvm::SmallVector<Note> outsideNotes;
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEDEADLOCKMODEL_H
