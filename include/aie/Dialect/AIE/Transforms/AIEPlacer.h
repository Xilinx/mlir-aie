//===- AIEPlacer.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PLACER_H
#define AIE_PLACER_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

namespace xilinx::AIE {

/// Placement algorithm type for pass option
enum class PlacerType { SequentialPlacer };

// maps logical tile operations to physical coordinates
using PlacementResult = llvm::DenseMap<mlir::Operation *, TileID>;

// For each LogicalTileOp involved in a cascade flow, the set of (peer,
// thisIsCascadeDst) tuples. `thisIsCascadeDst==true` means this tile is the
// destination of a cascade_flow whose source is `peer`; the lowered cascade
// requires `peer` to be one row higher (north) or one column to the left
// (west) of this tile. `thisIsCascadeDst==false` is the reverse.
using CascadeAdjacency =
    llvm::DenseMap<mlir::Operation *,
                   llvm::SmallVector<std::pair<LogicalTileOp, bool>, 2>>;

// Track available tiles and resource usage
struct TileAvailability {
  std::vector<TileID> compTiles;
  std::vector<TileID> nonCompTiles; // Memory and shim tiles

  llvm::DenseMap<TileID, int> inputChannelsUsed;
  llvm::DenseMap<TileID, int> outputChannelsUsed;

  void removeTile(TileID tile, AIETileType type);
};

// Abstract placer interface
class Placer {
public:
  Placer() = default;
  virtual ~Placer() = default;

  virtual void initialize(const AIETargetModel &targetModel) = 0;

  virtual mlir::LogicalResult place(DeviceOp device) = 0;

  virtual llvm::StringRef getName() const = 0;

  std::optional<TileID> getPlacement(mlir::Operation *logicalTile) const {
    auto it = result.find(logicalTile);
    if (it != result.end())
      return it->second;
    return std::nullopt;
  }

protected:
  PlacementResult result;
};

// Sequential placement algorithm
//
// Places logical tiles to physical tiles using a simple strategy:
// - Compute tiles: Sequential column-major placement (fill column before next)
// - Memory/shim tiles: DMA Channel capacity placement near common column
//
// Core-to-core connections are NOT validated because SequentialPlacer
// doesn't account for shared memory optimization.
//
// Shim/Mem tiles with identical placement constraints and sufficient
// DMA capacity are merged to the same physical tile.
class SequentialPlacer : public Placer {
public:
  SequentialPlacer(std::optional<int> coresPerCol = std::nullopt)
      : coresPerCol(coresPerCol) {}

  void initialize(const AIETargetModel &targetModel) override;

  mlir::LogicalResult place(DeviceOp device) override;

  llvm::StringRef getName() const override { return "sequential_placer"; }

private:
  std::optional<int> coresPerCol;
  int deviceCoresPerCol = 0; // Actual cores per column in device
  TileAvailability availability;
  const AIETargetModel *targetModel = nullptr;

  void limitCoresPerColumn(int maxCoresPerCol, int numColumns);

  void buildObjectFifoGroups(
      llvm::SmallVector<ObjectFifoCreateOp> &objectFifos,
      llvm::SmallVector<ObjectFifoLinkOp> &objectFifoLinks,
      llvm::DenseMap<int, llvm::SmallVector<ObjectFifoCreateOp>> &groupToFifos,
      llvm::DenseMap<int, llvm::SmallVector<LogicalTileOp>>
          &groupToLogicalTiles);

  std::optional<TileID> findTileWithCapacity(int targetCol,
                                             std::vector<TileID> &tiles,
                                             int requiredInputChannels,
                                             int requiredOutputChannels,
                                             AIETileType requestedType);

  void updateChannelUsage(TileID tile, bool isOutput, int numChannels);

  bool hasAvailableChannels(TileID tile, int inputChannels, int outputChannels);

  mlir::LogicalResult validateAndUpdateChannelUsage(
      LogicalTileOp logicalTile, TileID tile,
      const llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements,
      bool isConstrained);

  llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
  buildChannelRequirements(
      llvm::SmallVector<ObjectFifoCreateOp> &objectFifos,
      llvm::SmallVector<ObjectFifoLinkOp> &objectFifoLinks);

  void buildCascadeAdjacency(llvm::ArrayRef<CascadeFlowOp> cascadeFlows,
                             CascadeAdjacency &adjacency);

  // Returns true iff placing `logicalTile` at `candidate` would satisfy every
  // cascade adjacency constraint relative to peers that are already placed in
  // `result`. Unplaced peers impose no constraint at this point — they will
  // be checked when they themselves get placed.
  bool satisfiesCascadeAdjacency(LogicalTileOp logicalTile, TileID candidate,
                                 const CascadeAdjacency &adjacency) const;
};

} // namespace xilinx::AIE

#endif // AIE_PLACER_H
