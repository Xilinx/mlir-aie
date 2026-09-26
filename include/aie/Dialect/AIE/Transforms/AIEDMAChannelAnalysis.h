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
  mlir::DenseSet<std::tuple<mlir::Value, DMAChannelDir, int>> usedStreams;

public:
  DMAChannelAnalysis(DeviceOp &device);

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

  /// Claim a raw stream port, reporting on `tile` when it is already taken.
  void checkAIEStreamIndex(TileLike tile, DMAChannel chan);
};

} // namespace xilinx::AIE

#endif // AIE_DIALECT_AIE_TRANSFORMS_AIEDMACHANNELANALYSIS_H
