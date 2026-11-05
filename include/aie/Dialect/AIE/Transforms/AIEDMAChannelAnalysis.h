//===- AIEDMAChannelAnalysis.h ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H
#define AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace xilinx::AIE {

/// Which DMA channels of each tile are already spoken for, so that a channel is
/// handed out at most once across everything that programs one.
class DMAChannelAnalysis {
  /// Fully resolved tile aliases name the same hardware resources. Unresolved
  /// tiles retain their SSA identity until placement establishes co-location.
  mlir::DenseMap<std::pair<int, int>, mlir::Value> tilesByCoordinate;
  mlir::Value getTileKey(mlir::Value tile);

  /// Keep the reserving operation so diagnostics retain its MLIR location.
  mlir::DenseMap<std::tuple<mlir::Value, DMAChannelDir, int>, mlir::Operation *>
      usedChannels;
  /// DMA channels an explicit flow routes. First-free assignment skips them;
  /// a pinned request may still claim one.
  mlir::DenseMap<std::tuple<mlir::Value, DMAChannelDir, int>, mlir::Operation *>
      streamedChannels;
  /// The first claimant of a core stream port, and whether only packets use
  /// it; circuit occupancy is exclusive.
  struct StreamClaim {
    mlir::Operation *owner;
    bool packet;
  };
  mlir::DenseMap<std::tuple<mlir::Value, DMAChannelDir, int>, StreamClaim>
      usedStreams;
  /// The DMA program starts that claimed a channel, in program order.
  llvm::SmallVector<DMAStartOp> claimingStarts;

public:
  DMAChannelAnalysis(DeviceOp &device);

  /// The starts of DMA programs that claim a channel by index rather than
  /// through a route endpoint, one per channel they claim, in program order.
  llvm::ArrayRef<DMAStartOp> getClaimingStarts() const {
    return claimingStarts;
  }

  /// Exclusive upper bound on channel indices eligible for this transfer.
  static int getDMAChannelLimit(TileLike tile, DMAChannelDir dir,
                                bool requiresAdjacentTileAccessChannels);

  /// Next free channel of `tile` in `dir`, or -1 when the tile has none left.
  /// A channel an explicit flow routes is not free.
  /// A channel reaching an adjacent MemTile's memory must come from the
  /// target's restricted range at every compatible physical position.
  int getDMAChannelIndex(TileLike tile, DMAChannelDir dir,
                         bool requiresAdjacentTileAccessChannels,
                         mlir::Operation *owner = nullptr);

  /// Whether first-free assignment could hand out `channel`: nothing reserves
  /// it and no explicit flow routes it.
  bool isChannelFree(TileLike tile, DMAChannelDir dir, int channel);

  /// Claim `channel` for (`tile`, `dir`) so first-free assignment cannot take
  /// it. Returns the channel, or -1 when it is out of range or already
  /// claimed; the caller reports, since it knows which endpoint asked.
  int reservePinnedChannel(TileLike tile, DMAChannelDir dir, int channel,
                           mlir::Operation *owner = nullptr);

  /// Operation reserving this channel, or else the flow routing it; null if
  /// neither.
  mlir::Operation *getDMAChannelOwner(TileLike tile, DMAChannelDir dir,
                                      int channel);

  /// A DMA end of one tile still to get a channel: its direction, the BD ids
  /// it takes, and the exclusive upper bound on the channels it may take.
  struct BDPlanEnd {
    BDPlanEnd(DMAChannelDir dir, int64_t bds, int limit)
        : dir(dir), bds(bds), limit(limit) {}
    DMAChannelDir dir;
    int64_t bds;
    int limit;
  };

  /// A channel of the tile at (`col`, `row`) for each of `ends`, among those
  /// `free` leaves, with no two ends of a direction on one channel and, for
  /// every set of channels sharing BD ids, the ends on them plus the ids
  /// `spent` on each channel within those ids; nothing when no assignment
  /// fits.
  static std::optional<llvm::SmallVector<int>>
  planBDChannels(const AIETargetModel &target, int col, int row,
                 llvm::ArrayRef<BDPlanEnd> ends,
                 llvm::function_ref<bool(DMAChannelDir, int)> free,
                 llvm::function_ref<int64_t(int)> spent);

  /// Blocks of the BD chain `start` begins, following next_bd.
  static llvm::SmallVector<mlir::Block *> chainOf(DMAStartOp start);
  static int64_t countBDs(DMAStartOp start);

  /// The BD ids tiles' own programs hold before objectFIFOs take theirs: the
  /// chains DMA programs start by index, on their channels, and each id a
  /// runtime task pins, once per tile, on the lowest channel reaching it
  /// (aie-reserve-runtime-bd-ids keeps static BDs off those).
  class TakenBDs {
  public:
    TakenBDs() = default;
    explicit TakenBDs(DeviceOp device);
    /// By channel, the ids held at (`col`, `row`) by `tiles` and by every
    /// tile with those coordinates.
    llvm::DenseMap<int, int64_t>
    at(const AIETargetModel &target, int col, int row,
       llvm::ArrayRef<mlir::Operation *> tiles = {}) const;
    /// Coordinates of the tiles holding ids that have them.
    llvm::SmallVector<std::pair<int, int>> positions() const;

  private:
    struct Pin {
      unsigned order;
      int channel;
    };
    llvm::MapVector<mlir::Operation *, llvm::DenseMap<int, int64_t>> fixed;
    llvm::MapVector<mlir::Operation *, llvm::DenseMap<int32_t, Pin>> pinned;
  };

  /// Claim a raw stream port for `user`, reporting on it when the port is
  /// taken; packet users may share it with other packets.
  mlir::LogicalResult checkAIEStreamIndex(TileLike tile, DMAChannel chan,
                                          mlir::Operation *user,
                                          bool packet = false);
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H
