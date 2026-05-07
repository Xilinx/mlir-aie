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

// A set of LogicalTileOps connected by some connectivity op (objectfifo,
// flow, packet_flow, ...). `coreTiles` is used to compute the common column
// for placing the group's `nonCoreTiles`. Both lists may contain duplicates
// when a tile is referenced by multiple connectivity ops in the same group;
// duplicates intentionally weight the common-column average.
struct ConnectivityGroup {
  llvm::SmallVector<LogicalTileOp, 4> coreTiles;
  llvm::SmallVector<LogicalTileOp, 4> nonCoreTiles;
};

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

  void buildObjectFifoGroups(llvm::ArrayRef<ObjectFifoCreateOp> objectFifos,
                             llvm::ArrayRef<ObjectFifoLinkOp> objectFifoLinks,
                             llvm::SmallVectorImpl<ConnectivityGroup> &groups);

  void buildFlowGroups(llvm::ArrayRef<FlowOp> flows,
                       llvm::ArrayRef<PacketFlowOp> pktFlows,
                       llvm::SmallVectorImpl<ConnectivityGroup> &groups);

  mlir::LogicalResult placeNonCoreTilesInGroup(
      const ConnectivityGroup &group,
      const llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements);

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

  // Cross-tile L1 buffer access: consumer LTO's core must pass
  // targetModel->isLegalMemAffinity(coreCol=consumerCol, coreRow=consumerRow,
  //                                 memCol=ownerCol,     memRow=ownerRow).
  // Same shape as CascadeAdjacency (#3042); TODO: factor a generic
  // Adjacency<Predicate> once both have landed.
  struct BufferAdjacency {
    llvm::SmallVector<std::pair<LogicalTileOp, TileLike>, 4> edges;
  // `tileToEdges` indexes into `edges` only for `LogicalTileOp` endpoints
  // (the ones the placer visits); `TileOp` peers contribute coords via
  // `TileLike::tryGetCol`/`tryGetRow`.
  struct CascadeAdjacency {
    llvm::SmallVector<std::pair<TileLike, TileLike>, 4> edges;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
        tileToEdges;
  };

  BufferAdjacency
  buildBufferAdjacency(llvm::ArrayRef<LogicalTileOp> logicalTiles);

  bool satisfiesBufferAdjacency(LogicalTileOp logicalTile, TileID candidate,
                                const BufferAdjacency &adjacency) const;

  void attachBufferPeerNotes(mlir::InFlightDiagnostic &diag,
                             LogicalTileOp logicalTile,
                             const BufferAdjacency &adjacency) const;
  CascadeAdjacency
  buildCascadeAdjacency(llvm::ArrayRef<CascadeFlowOp> cascadeFlows);

  // Unplaced peers without pin coords defer; symmetric predicate makes one
  // check per endpoint sufficient.
  bool satisfiesCascadeAdjacency(LogicalTileOp logicalTile, TileID candidate,
                                 const CascadeAdjacency &adjacency) const;

  void attachCascadePeerNotes(mlir::InFlightDiagnostic &diag,
                              LogicalTileOp logicalTile,
                              const CascadeAdjacency &adjacency) const;
  void addChannelRequirementsFromFlows(
      llvm::ArrayRef<FlowOp> flows, llvm::ArrayRef<PacketFlowOp> pktFlows,
      llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements);
};

} // namespace xilinx::AIE

#endif // AIE_PLACER_H
