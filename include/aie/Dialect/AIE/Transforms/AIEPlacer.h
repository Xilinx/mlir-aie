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
// By default (mergeLogicalTiles == true), Shim/Mem logical tiles with
// identical placement constraints and sufficient combined DMA capacity
// are merged to the same physical tile via TileOp::getOrCreate during
// the conversion pattern. CoreTile placement is column-major sequential
// and never merges. Callers that pre-aggregate non-core logical tiles
// can pass mergeLogicalTiles == false to make the placer pin each
// non-core LTO to its own physical tile.
class SequentialPlacer : public Placer {
public:
  SequentialPlacer(std::optional<int> coresPerCol = std::nullopt,
                   bool mergeLogicalTiles = true)
      : coresPerCol(coresPerCol), mergeLogicalTiles(mergeLogicalTiles) {}

  void initialize(const AIETargetModel &targetModel) override;

  mlir::LogicalResult place(DeviceOp device) override;

  llvm::StringRef getName() const override { return "sequential_placer"; }

private:
  std::optional<int> coresPerCol;
  bool mergeLogicalTiles;
  // Physical tiles already assigned to a non-core aie.logical_tile. Used
  // only when mergeLogicalTiles == false to forbid mapping a second
  // non-core aie.logical_tile onto a tile that already hosts one.
  llvm::DenseSet<TileID> assignedNonCoreTiles;
  int deviceCoresPerCol = 0; // Actual cores per column in device
  TileAvailability availability;
  const AIETargetModel *targetModel = nullptr;

  void limitCoresPerColumn(int maxCoresPerCol, int numColumns);

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

  // Per-LTO peer edges indexed by either endpoint. `tileToEdges` only
  // indexes `LogicalTileOp` endpoints; `TileOp` peers carry their own coords.
  struct Adjacency {
    llvm::SmallVector<std::pair<TileLike, TileLike>, 4> edges;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
        tileToEdges;

    void addEdge(TileLike first, TileLike second) {
      unsigned idx = edges.size();
      edges.push_back({first, second});
      if (mlir::isa<LogicalTileOp>(first.getOperation()))
        tileToEdges[first.getOperation()].push_back(idx);
      if (mlir::isa<LogicalTileOp>(second.getOperation()))
        tileToEdges[second.getOperation()].push_back(idx);
    }

    // Convenience for IR walkers: skip if either Value isn't a TileLike.
    void addEdgeFromValues(mlir::Value a, mlir::Value b) {
      auto aT = mlir::dyn_cast_or_null<TileLike>(a.getDefiningOp());
      auto bT = mlir::dyn_cast_or_null<TileLike>(b.getDefiningOp());
      if (aT && bT)
        addEdge(aT, bT);
    }
  };

  // Edge: (consumer LTO, owner tile). Predicate: `isLegalMemAffinity`.
  Adjacency buildBufferAdjacency(llvm::ArrayRef<LogicalTileOp> logicalTiles);

  // Edge: (cascade source, dest).
  Adjacency buildCascadeAdjacency(llvm::ArrayRef<CascadeFlowOp> cascadeFlows);

  // Edge: (producer, consumer_i) per fifo. Linked fifos connect transitively
  // through the link tile (it's the consumer of every source fifo and the
  // producer of every destination fifo), so per-fifo emission suffices.
  Adjacency
  buildObjectFifoAdjacency(llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  // Edge: (src, dst) per `aie.flow`; cross-product per `aie.packet_flow`.
  Adjacency buildFlowAdjacency(llvm::ArrayRef<FlowOp> flows,
                               llvm::ArrayRef<PacketFlowOp> pktFlows);

  // Place a non-core (mem/shim) LTO near the centroid column of its placed
  // core peers, reached transitively through `connectivityAdjacencies`.
  mlir::LogicalResult placeNonCoreTileByCentroid(
      LogicalTileOp logicalTile,
      llvm::ArrayRef<const Adjacency *> connectivityAdjacencies,
      const llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements);

  // Pairwise legality check. `pred(firstPos, secondPos)` is evaluated for
  // every edge mentioning `logicalTile`; unplaced peers defer.
  bool satisfiesAdjacency(
      LogicalTileOp logicalTile, TileID candidate, const Adjacency &adjacency,
      llvm::function_ref<bool(TileID firstPos, TileID secondPos)> pred) const;

  // Diagnostic peer notes. `labelPeer(thisIsFirst)` names the peer endpoint
  // role; the attached note reads "<label> peer placed at (col, row)".
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
