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

  // O(1) membership shadow of `compTiles` used by SequentialPlacer's
  // satisfiesComputePeer forward-look, which checks whether each of a
  // candidate's physical compute neighbors is still free. Without it the
  // forward-look would do an O(|compTiles|) std::find per direction per
  // candidate per LTO, which scales as O(N * M^2) on large arrays. Must be
  // kept in sync with `compTiles` by any code that mutates the vector
  // (initialize(), limitCoresPerColumn(), removeTile()).
  llvm::DenseSet<TileID> compTilesSet;

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
// Greedy, single-pass placer that maps each `aie.logical_tile` (LTO) to a
// physical (col, row). The algorithm runs in four phases:
//
//   Phase 1 -- Collect:
//     Walk the device and bucket the ops the placer needs:
//     LogicalTileOps, ObjectFifoCreateOps, ObjectFifoLinkOps,
//     CascadeFlowOps, FlowOps, PacketFlowOps.
//
//   Phase 2 -- Constraint construction:
//     Build per-LTO constraints from the collected IR:
//       * buffer adjacency  (shared-L1 producer/consumer pairs)
//       * cascade adjacency (cascade source -> destination pairs)
//       * compute-peer adjacency (compute->compute ObjectFifos)
//       * channelRequirements (per-LTO {input, output} DMA channel demand)
//       * needNeighborIn / needNeighborOut (the minimum number of compute
//         peers per LTO that MUST land on a shared-L1 neighbor to fit the
//         per-tile DMA budget; this is what drives placement priority and
//         soft-reservation of neighbor slots).
//
//   Phase 3 -- Compute-tile placement:
//     Sort LogicalTileOps by (constraint level ascending; pinned before
//     partial before unpinned), then within an equal level by
//     placementPriority descending (max of self demand vs. highest peer
//     demand) so high-fanin Workers and their compute peers cluster to the
//     front of the worklist. Iterate the worklist:
//       * fully pinned LTO -> validate adjacency + channel capacity at the
//         pinned coord, accept or fail.
//       * unpinned or partially-pinned compute tile -> walk candidates
//         (column-major; for high-demand LTOs additionally re-sorted to
//         prefer interior rows) in two passes: first only candidates that
//         aren't soft-reserved by an unrelated demand-bearing LTO's peers,
//         then a fall-back pass that accepts reserved candidates. Each
//         candidate is filtered by satisfiesAdjacency (buffer, cascade)
//         and satisfiesComputePeer (DMA-budget feasibility). On success,
//         remove the chosen tile from availability and (if the LTO has
//         neighbor demand) soft-reserve its mem-affinity neighbor slots
//         for that LTO's compute peers.
//       * unpinned ShimPLTile -> hard error (no DMAs).
//
//   Phase 4 -- Non-core tile placement:
//     For each remaining non-core (mem/shim) LTO, BFS its connectivity
//     component to find the centroid column of its placed-core peers,
//     then pick the tile of the requested type nearest the centroid
//     column with enough spare DMA capacity. By default (mergeLogicalTiles
//     == true) multiple non-core LTOs may share a physical tile when
//     their combined DMA channel usage still fits; with mergeLogicalTiles
//     == false each non-core LTO claims its own tile.
//
// Greedy placement: once a tile is chosen for an LTO the decision is
// final. The placer does not backtrack. If a later LTO's constraints
// become unsatisfiable given prior placements, the placer emits an error
// and asks the user to manually pin tiles to break the deadlock.
//
// Core-to-core ObjectFifos are special-cased into the compute-peer
// constraint subsystem; they aren't counted against the channel budget
// when the producer and consumer end up on shared-L1 neighbor tiles
// (the lowering picks the shared-memory path in that case).
class SequentialPlacer : public Placer {
public:
  SequentialPlacer(std::optional<int> coresPerCol = std::nullopt,
                   bool mergeLogicalTiles = true)
      : coresPerCol(coresPerCol), mergeLogicalTiles(mergeLogicalTiles) {}

  void initialize(const AIETargetModel &targetModel) override;

  mlir::LogicalResult place(DeviceOp device) override;

  llvm::StringRef getName() const override { return "sequential_placer"; }

  // Per-LTO peer edges indexed by either endpoint. `tileToEdges` only
  // indexes `LogicalTileOp` endpoints; `TileOp` peers carry their own
  // coords. Public so file-local helpers in AIEPlacer.cpp (notably the
  // forEachPeer template) can take it by reference.
  struct Adjacency {
    llvm::SmallVector<std::pair<TileLike, TileLike>, 4> edges;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
        tileToEdges;

    void addEdge(TileLike first, TileLike second) {
      if (!first || !second)
        return;
      unsigned idx = edges.size();
      edges.push_back({first, second});
      if (mlir::isa<LogicalTileOp>(first.getOperation()))
        tileToEdges[first.getOperation()].push_back(idx);
      if (mlir::isa<LogicalTileOp>(second.getOperation()))
        tileToEdges[second.getOperation()].push_back(idx);
    }

    // Convenience for IR walkers: skip if either Value isn't a TileLike.
    void addEdgeFromValues(mlir::Value a, mlir::Value b) {
      if (!a || !b)
        return;
      auto aT = mlir::dyn_cast_or_null<TileLike>(a.getDefiningOp());
      auto bT = mlir::dyn_cast_or_null<TileLike>(b.getDefiningOp());
      if (aT && bT)
        addEdge(aT, bT);
    }
  };

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

  // Per-place() bookkeeping bundled into one struct holding the per-LTO
  // neighbor-demand maps and the soft-reservation table. Lives on the
  // stack of place() for the duration of placement.
  struct PlacementContext {
    const AIETargetModel &targetModel;
    const Adjacency &computePeerAdjacency;
    const llvm::DenseMap<mlir::Operation *, int> &needNeighborIn;
    const llvm::DenseMap<mlir::Operation *, int> &needNeighborOut;
    // Mutable: each placement of an LTO with neighbor demand reserves
    // its mem-affinity neighbor slots for that LTO's compute peers.
    llvm::DenseMap<TileID, llvm::SmallVector<mlir::Operation *, 2>>
        reservedFor;

    int neighborDemand(mlir::Operation *op) const {
      return needNeighborIn.lookup(op) + needNeighborOut.lookup(op);
    }
    // Priority = max(self demand, highest peer demand). A peer of a
    // high-fanin Worker rises to the front of the sort so its neighbor
    // slots aren't consumed by unrelated LTOs.
    int placementPriority(mlir::Operation *op) const;
    // Soft-reserve `at`'s mem-affinity compute-tile neighbors for
    // `placedOp`'s peers, if `placedOp` has non-zero neighbor demand.
    void reserveNeighborSlots(mlir::Operation *placedOp, TileID at);
    // True if `candidate` is reserved by any LTO that is NOT a compute
    // peer of `lto` (i.e. would compete for the slot).
    bool isReservedForOther(mlir::Operation *lto, TileID candidate) const;
  };

  // Per-call inputs for `findUnconstrainedCoreCandidate`. Bundles the
  // adjacency + demand state that Phase 3 builds for each candidate
  // filter.
  struct UnpinnedPlacementInputs {
    const Adjacency &bufferAdjacency;
    llvm::function_ref<bool(TileID, TileID)> bufferPred;
    const Adjacency &cascadeAdjacency;
    llvm::function_ref<bool(TileID, TileID)> cascadePred;
    const Adjacency &computePeerAdjacency;
    const llvm::DenseMap<mlir::Operation *, int> &needNeighborIn;
    const llvm::DenseMap<mlir::Operation *, int> &needNeighborOut;
    llvm::function_ref<bool(mlir::Operation *, TileID)> isReservedForOther;
  };

  // Outputs of `findUnconstrainedCoreCandidate`. `placement` is set on
  // success; on failure the three boolean flags describe which filter
  // rejected the most candidates so place() can construct an actionable
  // diagnostic.
  struct UnpinnedSearchResult {
    std::optional<TileID> placement;
    bool sawConstraintMatch = false;
    bool allConstraintMatchesFailedAdjacency = true;
    bool computePeerWasCause = false;
  };

  // Two-pass candidate search for an unpinned (or partially-pinned)
  // CoreTile LTO. First pass prefers candidates that aren't
  // soft-reserved by an unrelated demand-bearing LTO's peers; second
  // pass falls back to reserved candidates. Each candidate is filtered
  // by buffer adjacency, cascade adjacency, and the compute-peer DMA
  // budget. Does not mutate placer state.
  UnpinnedSearchResult findUnconstrainedCoreCandidate(
      LogicalTileOp logicalTile, std::optional<int> col,
      std::optional<int> row, const UnpinnedPlacementInputs &inputs);

  // Edge: (consumer LTO, owner tile). Predicate: `isLegalMemAffinity`.
  Adjacency buildBufferAdjacency(llvm::ArrayRef<LogicalTileOp> logicalTiles);

  // Edge: (cascade source, dest).
  Adjacency buildCascadeAdjacency(llvm::ArrayRef<CascadeFlowOp> cascadeFlows);

  // Edge: (producer LTO, consumer LTO) for every ObjectFifo whose producer
  // AND consumer are CoreTile LTOs. Used to constrain placement so that
  // high-fanin compute Workers (those whose total fifo count exceeds the
  // compute-tile DMA budget) land on tiles with their peer producers as
  // physical neighbors — the peer fifo then uses shared neighbor memory
  // instead of a DMA channel.
  Adjacency
  buildComputePeerAdjacency(llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  // Edge: (producer, consumer_i) per fifo. Linked fifos connect transitively
  // through the link tile (it's the consumer of every source fifo and the
  // producer of every destination fifo), so per-fifo emission suffices.
  Adjacency
  buildObjectFifoAdjacency(llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  // Edge: (src, dst) per `aie.flow`; cross-product per `aie.packet_flow`.
  Adjacency buildFlowAdjacency(llvm::ArrayRef<FlowOp> flows,
                               llvm::ArrayRef<PacketFlowOp> pktFlows);

  // Phase 4 driver: for every still-unplaced non-core LTO, place it near
  // the centroid column of its core peers. Internally builds the per-fifo
  // and per-flow connectivity adjacencies needed to find the centroid,
  // then iterates non-core LTOs in descending channel-demand order so the
  // heaviest consumers get first pick of physical tiles.
  mlir::LogicalResult placeNonCoreLogicalTiles(
      llvm::ArrayRef<LogicalTileOp> logicalTiles,
      llvm::ArrayRef<ObjectFifoCreateOp> objectFifos,
      llvm::ArrayRef<FlowOp> flows, llvm::ArrayRef<PacketFlowOp> pktFlows,
      const llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements);

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

  // Compute-peer DMA budget check. Returns true if placing `logicalTile` at
  // `candidate` keeps the compute-peer neighbor demand satisfiable for BOTH
  // the LTO being placed AND every already-placed compute peer of the LTO.
  // A compute LTO with peer demand N (N compute-peer fifos must use shared
  // neighbor memory to fit DMA budget) has slack (totalComputePeers − N): it
  // can afford at most that many non-neighbor compute peers. The check also
  // forward-looks: unplaced peers still need a free physical compute-tile
  // neighbor of `candidate` to land on.
  bool satisfiesComputePeer(
      LogicalTileOp logicalTile, TileID candidate,
      const Adjacency &computePeerAdjacency,
      const llvm::DenseMap<mlir::Operation *, int> &needNeighborIn,
      const llvm::DenseMap<mlir::Operation *, int> &needNeighborOut) const;

  // Compute-peer in/out edge counts for `op`, looked up from a prebuilt
  // `computePeerAdjacency`. Returns {0, 0} if `op` has no entry. Producer
  // is edges[idx].first, consumer is .second (per buildComputePeerAdjacency).
  static std::pair<int, int> totalComputePeers(mlir::Operation *op,
                                               const Adjacency &adjacency);

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
