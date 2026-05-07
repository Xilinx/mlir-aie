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

  // Per-LTO peer edges indexed by either endpoint. Used by both shared-L1
  // buffer adjacency (memory affinity) and cascade adjacency (cardinal
  // direction). `tileToEdges` indexes into `edges` only for `LogicalTileOp`
  // endpoints (the ones the placer visits); `TileOp` peers contribute coords
  // via `TileLike::tryGetCol`/`tryGetRow`.
  struct Adjacency {
    llvm::SmallVector<std::pair<TileLike, TileLike>, 4> edges;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
        tileToEdges;

    // Append a peer edge and index it on every `LogicalTileOp` endpoint.
    // Centralizes the invariant that `tileToEdges[op]` lists every edge
    // mentioning `op`, but only for LTO endpoints (`TileOp` peers carry
    // their own coords).
    void addEdge(TileLike first, TileLike second) {
      unsigned idx = edges.size();
      edges.push_back({first, second});
      if (mlir::isa<LogicalTileOp>(first.getOperation()))
        tileToEdges[first.getOperation()].push_back(idx);
      if (mlir::isa<LogicalTileOp>(second.getOperation()))
        tileToEdges[second.getOperation()].push_back(idx);
    }
  };

  // Generic walker: for each `CoreTile` LTO, walk its core body and use
  // `extractOwner` to map each operand value to the owning `TileLike` (or
  // null if the operand is irrelevant). For every distinct cross-tile owner,
  // emit edge `(consumer, owner)`. Used by buffer and lock adjacency, both
  // of which derive from "core body op references something attached to
  // another tile".
  Adjacency buildCoreBodyAdjacency(
      llvm::ArrayRef<LogicalTileOp> logicalTiles,
      llvm::function_ref<TileLike(mlir::Value)> extractOwner);

  // Cross-tile L1 buffer access: consumer LTO's core must pass
  // targetModel->isLegalMemAffinity(coreCol=consumerCol, coreRow=consumerRow,
  //                                 memCol=ownerCol,     memRow=ownerRow).
  // The first endpoint of each edge is the consumer LTO; the second is the
  // owner tile.
  Adjacency buildBufferAdjacency(llvm::ArrayRef<LogicalTileOp> logicalTiles);

  // Cross-tile L1 lock access: same `isLegalMemAffinity` rule as buffers,
  // since AIE2 locks live in the L1 region of their owning tile and are only
  // visible to memory-affinity neighbors. The `UsesAreAccessible` trait on
  // `LockOp::verify` enforces this for placed `TileOp` users but skips
  // `LogicalTileOp` users, which is exactly the gap the placer fills.
  Adjacency buildLockAdjacency(llvm::ArrayRef<LogicalTileOp> logicalTiles);

  // The first endpoint of each edge is the cascade source; the second is the
  // destination.
  Adjacency buildCascadeAdjacency(llvm::ArrayRef<CascadeFlowOp> cascadeFlows);

  // Generic adjacency predicate. `pred` returns true iff `(firstPos,
  // secondPos)` satisfies the constraint, where `first` and `second` are
  // `adjacency.edges[i].first` and `.second`. Unplaced peers without pin
  // coords defer; symmetric/asymmetric predicates only need to be checked at
  // one endpoint each.
  bool satisfiesAdjacency(
      LogicalTileOp logicalTile, TileID candidate, const Adjacency &adjacency,
      llvm::function_ref<bool(TileID firstPos, TileID secondPos)> pred) const;

  // Generic peer-note attachment. `labelPeer(thisIsFirst)` returns the
  // descriptive name of the peer endpoint -- when `logicalTile` is the first
  // member of the edge, the peer is the second member, and vice versa. The
  // attached note reads `"<label> peer placed at (col, row)"`.
  void attachPeerNotes(
      mlir::InFlightDiagnostic &diag, LogicalTileOp logicalTile,
      const Adjacency &adjacency,
      llvm::function_ref<llvm::StringRef(bool thisIsFirst)> labelPeer) const;

  void addChannelRequirementsFromFlows(
      llvm::ArrayRef<FlowOp> flows, llvm::ArrayRef<PacketFlowOp> pktFlows,
      llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements);
};

} // namespace xilinx::AIE

#endif // AIE_PLACER_H
