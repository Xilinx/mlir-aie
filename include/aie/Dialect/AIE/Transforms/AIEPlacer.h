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

#include <deque>
#include <random>

namespace xilinx::AIE {

/// Placement algorithm type for pass option
enum class PlacerType { SequentialPlacer, SAPlacer };

/// Get DMA channel capacity (maxIn, maxOut) for a tile position.
inline std::pair<int, int> getDMACapacity(const AIETargetModel &tm,
                                          TileID tile) {
  if (tile.row == 0)
    return {
        tm.getNumDestShimMuxConnections(tile.col, tile.row, WireBundle::DMA),
        tm.getNumSourceShimMuxConnections(tile.col, tile.row, WireBundle::DMA)};
  return {
      tm.getNumDestSwitchboxConnections(tile.col, tile.row, WireBundle::DMA),
      tm.getNumSourceSwitchboxConnections(tile.col, tile.row, WireBundle::DMA)};
}

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

  virtual void initialize(const AIETargetModel &targetModel);

  virtual mlir::LogicalResult place(DeviceOp device) = 0;

  virtual llvm::StringRef getName() const = 0;

  std::optional<TileID> getPlacement(mlir::Operation *logicalTile) const {
    auto it = result.find(logicalTile);
    if (it != result.end())
      return it->second;
    return std::nullopt;
  }

  // Buffer allocation redirects: (ObjectFifoCreateOp, delegate TileID)
  // Generated when a core tile's buffers exceed local memory and must
  // spill to a neighbor tile's memory module.
  using AllocateInfo = std::pair<mlir::Operation *, TileID>;
  llvm::SmallVector<AllocateInfo> &getAllocates() { return allocates; }

  // Per-LTO peer edges indexed by either endpoint. `tileToEdges` only
  // indexes `LogicalTileOp` endpoints; `TileOp` peers carry their own
  // coords.
  struct Adjacency {
    llvm::SmallVector<std::pair<TileLike, TileLike>, 4> edges;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<unsigned, 2>>
        tileToEdges;

    void addEdge(TileLike first, TileLike second);

    // Convenience for IR walkers: skip if either Value isn't a TileLike.
    void addEdgeFromValues(mlir::Value a, mlir::Value b);

    // True if `op` has at least one edge in this adjacency.
    bool hasEdges(mlir::Operation *op) const {
      return tileToEdges.count(op) != 0;
    }
  };

  // IR operations collected from a DeviceOp for placement.
  struct CollectedOps {
    llvm::SmallVector<LogicalTileOp> logicalTiles;
    llvm::SmallVector<ObjectFifoCreateOp> objectFifos;
    llvm::SmallVector<ObjectFifoLinkOp> objectFifoLinks;
    llvm::SmallVector<CascadeFlowOp> cascadeFlows;
    llvm::SmallVector<FlowOp> flows;
    llvm::SmallVector<PacketFlowOp> pktFlows;
  };

  // Collect all placement-relevant operations from the device.
  static CollectedOps collectOperations(DeviceOp device);

  // Check if a logical tile's col/row constraints are satisfied at a position.
  static bool satisfiesConstraints(mlir::Operation *tile, TileID pos);

  // Adjacency builders: pure IR walkers with no placer-specific state.

  // Edge: (cascade source, dest).
  static Adjacency
  buildCascadeAdjacency(llvm::ArrayRef<CascadeFlowOp> cascadeFlows);

  // Edge: (producer LTO, consumer LTO) for every ObjectFifo whose producer
  // AND consumer are CoreTile LTOs.
  static Adjacency
  buildComputePeerAdjacency(llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  // Edge: (producer, consumer_i) per fifo.
  static Adjacency
  buildObjectFifoAdjacency(llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  // Edge: (src, dst) per `aie.flow`; cross-product per `aie.packet_flow`.
  static Adjacency buildFlowAdjacency(llvm::ArrayRef<FlowOp> flows,
                                      llvm::ArrayRef<PacketFlowOp> pktFlows);

protected:
  PlacementResult result;
  llvm::SmallVector<AllocateInfo> allocates;
  const AIETargetModel *targetModel = nullptr;
  TileAvailability availability;
};

// Invoke `fn(peer, thisIsFirst)` for each edge in `adjacency` that
// mentions `op`. `peer` is the OTHER endpoint of the edge; `thisIsFirst`
// is true when `op` is `edge.first`.
template <typename F>
inline void forEachPeer(mlir::Operation *op, const Placer::Adjacency &adjacency,
                        F &&fn) {
  auto it = adjacency.tileToEdges.find(op);
  if (it == adjacency.tileToEdges.end())
    return;
  for (unsigned idx : it->second) {
    auto [first, second] = adjacency.edges[idx];
    bool thisIsFirst = first.getOperation() == op;
    TileLike peer = thisIsFirst ? second : first;
    fn(peer, thisIsFirst);
  }
}

// Invoke `fn(TileID)` on every CoreTile neighbor of `at` that is a legal
// shared-L1 mem-affinity neighbor per the target model.
template <typename F>
inline void forEachMemAffinityNeighbor(const AIETargetModel &targetModel,
                                       TileID at, F &&fn) {
  for (auto [dc, dr] :
       {std::pair{0, -1}, std::pair{0, 1}, std::pair{-1, 0}, std::pair{1, 0}}) {
    int nc = at.col + dc;
    int nr = at.row + dr;
    if (nc < 0 || nc >= targetModel.columns())
      continue;
    if (nr < 0 || nr >= targetModel.rows())
      continue;
    if (targetModel.getTileType(nc, nr) != AIETileType::CoreTile)
      continue;
    if (!targetModel.isLegalMemAffinity(at.col, at.row, nc, nr))
      continue;
    fn(TileID{nc, nr});
  }
}

// Sequential placement algorithm
//
// Greedy, single-pass placer that maps each `aie.logical_tile` (LTO) to a
// physical (col, row). Four phases:
//
//   1. Collect LogicalTileOp / ObjectFifo* / Cascade* / Flow* / PacketFlow*
//      from the device.
//   2. Build per-LTO constraints: buffer/cascade/compute-peer adjacencies,
//      channel requirements, and needNeighborIn/Out (the minimum number of
//      compute peers per LTO that MUST land on a shared-L1 neighbor to fit
//      the per-tile DMA budget).
//   3. Place compute tiles. Sort by constraint level (pinned -> partial ->
//      unpinned) then by descending placementPriority. For each LTO:
//      validate buffer/cascade/compute-peer constraints and channel budget,
//      then claim the tile. Unpinned candidates are tried in two passes:
//      first the slots not soft-reserved for another demand-bearing LTO's
//      peers, then a fallback pass that accepts reserved slots. Each
//      placement of a demand-bearing LTO soft-reserves its mem-affinity
//      neighbor slots for that LTO's compute peers.
//   4. Place each remaining non-core (mem/shim) LTO near the column
//      centroid of its placed-core peers. With mergeLogicalTiles == true
//      (default) several non-core LTOs may share one physical tile when
//      DMA capacity allows; with mergeLogicalTiles == false each non-core
//      LTO claims its own physical tile.
//
// Greedy: once a tile is chosen the decision is final, with no backtracking.
// If a later LTO's constraints become unsatisfiable given prior placements,
// the placer emits an error pointing at the LTOs the user can manually pin
// to break the deadlock.
//
// Compute-to-compute ObjectFifos consume no DMA channel when their endpoints
// land on shared-L1 neighbor tiles, so they enter the placer as a separate
// compute-peer adjacency rather than as channel demand.
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

  // DMA channel direction selector.
  enum class DmaDir { In, Out };

  // Whether an LTO's placement was pinned by the user (via col/row
  // attributes) or chosen by the placer.
  enum class PlacementOrigin { Pinned, Selected };

  void limitCoresPerColumn(int maxCoresPerCol, int numColumns);

  std::optional<TileID> findTileWithCapacity(int targetCol,
                                             llvm::ArrayRef<TileID> tiles,
                                             int requiredInputChannels,
                                             int requiredOutputChannels,
                                             AIETileType requestedType);

  void updateChannelUsage(TileID tile, DmaDir direction, int numChannels);

  bool hasAvailableChannels(TileID tile, int inputChannels, int outputChannels);

  mlir::LogicalResult validateAndUpdateChannelUsage(
      LogicalTileOp logicalTile, TileID tile,
      const llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
          &channelRequirements,
      PlacementOrigin origin);

  llvm::DenseMap<mlir::Operation *, std::pair<int, int>>
  buildChannelRequirements(
      llvm::SmallVector<ObjectFifoCreateOp> &objectFifos,
      llvm::SmallVector<ObjectFifoLinkOp> &objectFifoLinks);

  // Per-place() neighbor-demand maps and soft-reservation table.
  struct PlacementContext {
    const AIETargetModel &targetModel;
    const Adjacency &computePeerAdjacency;
    const llvm::DenseMap<mlir::Operation *, int> &needNeighborIn;
    const llvm::DenseMap<mlir::Operation *, int> &needNeighborOut;
    // Mutable: each placement of an LTO with neighbor demand reserves
    // its mem-affinity neighbor slots for that LTO's compute peers.
    llvm::DenseMap<TileID, llvm::SmallVector<mlir::Operation *, 2>>
        reservedFor{};

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

  // A pairwise-legality adjacency constraint (buffer, cascade, ...)
  // bundled with everything the placer needs to check it and report a
  // useful diagnostic when it fails: the edge set, the predicate, the
  // peer-role label for notes, the short name for the error message,
  // and the educational hint text appended after the peer notes.
  struct AdjacencyKind {
    const Adjacency &adjacency;
    llvm::function_ref<bool(TileID firstPos, TileID secondPos)> pred;
    llvm::function_ref<llvm::StringRef(bool thisIsFirst)> peerLabel;
    llvm::StringRef name;           // e.g. "shared-L1 buffer", "cascade"
    llvm::StringRef constraintHint; // appended as a final attachNote
  };

  // Check `logicalTile` at `tile` against `kind`'s adjacency. On
  // failure, emit a fully-formed diagnostic (error + peer notes +
  // constraint hint) and return failure.
  mlir::LogicalResult enforceAdjacency(LogicalTileOp logicalTile, TileID tile,
                                       const AdjacencyKind &kind);

  // Record that `logicalTile` is placed at `tile`. For CoreTile LTOs
  // this also removes the tile from availability and asks `ctx` to
  // soft-reserve the tile's mem-affinity neighbor slots for the LTO's
  // compute peers. `debugLabel` ("pinned" / "unconstrained") names the
  // placement path in the LLVM_DEBUG trace.
  void recordPlacement(LogicalTileOp logicalTile, TileID tile,
                       PlacementContext &ctx, llvm::StringRef debugLabel);

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
  UnpinnedSearchResult
  findUnconstrainedCoreCandidate(LogicalTileOp logicalTile,
                                 std::optional<int> col, std::optional<int> row,
                                 const UnpinnedPlacementInputs &inputs);

  // Edge: (consumer LTO, owner tile). Predicate: `isLegalMemAffinity`.
  // SequentialPlacer-only: walks CoreOps and BufferOps.
  Adjacency buildBufferAdjacency(llvm::ArrayRef<LogicalTileOp> logicalTiles);

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

  // Per-LTO peer-Value lists, one list per flow that mentions the LTO.
  // Built once and shared so each centroid lookup is O(#flows on the LTO).
  struct FlowMembership {
    llvm::DenseMap<mlir::Value,
                   llvm::SmallVector<llvm::SmallVector<mlir::Value>>>
        ltoFlows;
  };

  FlowMembership
  buildFlowMembership(llvm::ArrayRef<FlowOp> flows,
                      llvm::ArrayRef<PacketFlowOp> pktFlows,
                      llvm::ArrayRef<ObjectFifoCreateOp> objectFifos);

  // Pick the column that minimizes total routing cost across the LTO's
  // flows. See AIEPlacer.cpp for the per-flow cost formulas and tiebreak.
  int computeCentroidColumn(LogicalTileOp logicalTile,
                            const FlowMembership &flowIndex);

  mlir::LogicalResult placeNonCoreTileByCentroid(
      LogicalTileOp logicalTile, const FlowMembership &flowIndex,
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

// Configuration for the SA placer's cost model and schedule.
//
// The SA placer minimizes a total cost with four components:
//
//   totalCost = HPWL + resourcePenalty + cascadePenalty + memoryPressure
//
// 1. HPWL (half-perimeter wire length): sum of bounding-box perimeters
//    across all nets.
//
// 2. Resource penalty (hard, blocks legality): penalizes MemTile buffer
//    overflow, core tile overflow, DMA channel overuse, and BD count
//    overuse.
//
// 3. Cascade penalty (hard, blocks legality): penalizes cascade put/get
//    pairs that aren't adjacent.
//
// 4. Memory pressure (soft, guides optimization): quadratic penalty when
//    tile utilization exceeds a threshold.
struct SAConfig {
  // HPWL (cost 1)
  int multicastThreshold = 4;  // consumer count above which penalty applies
  int multicastMultiplier = 2; // HPWL multiplier for multicast nets

  // Resource penalty (cost 2)
  int memPenaltyPerKB = 30;      // per KB of unresolved memory overflow
  int dmaPenaltyPerChannel = 20; // per DMA channel or BD over limit

  // Cascade penalty (cost 3)
  int cascadeWeightPerDist = 30; // per Manhattan distance to valid position

  // Memory pressure (cost 4)
  int pressurePerKB = 3;           // quadratic pressure above threshold
  double pressureThreshold = 0.75; // utilization fraction triggering pressure

  // Adaptive cooling
  double coolFast = 0.995;        // accept ratio > 96%: cool faster
  double coolSteady = 0.998;      // accept ratio > 80%: steady progress
  double coolBalanced = 0.999;    // accept ratio > 15%: balanced exploration
  double coolStuck = 0.9998;      // accept ratio <= 15%: barely cool
  double greedyThreshold = 0.005; // switch to greedy (T=0) below this

  // Schedule sizing
  int minMovesPerIter = 100;
  int maxMovesPerIter = 2000;
  int minMaxIters = 10000;
  int greedyMultiplier = 50;       // greedyIters = multiplier * numMovable
  double tempScaleEstimate = 10.0; // initTemp cap: scale * estimated T
  double tempScaleCost = 2.0;      // initTemp cap: scale * totalCost

  // Temperature estimation
  int tempSearchIters = 100;         // binary search iterations
  double tempSearchCeiling = 5.0;    // highT = ceiling * maxDelta
  double tempSearchEpsilon = 1e-10;  // zero guard in binary search
  double tempSearchTolerance = 1e-6; // convergence tolerance

  // SA loop intervals
  int debugInterval = 5000;          // iterations between debug log lines
  int fullRecomputeInterval = 10000; // iterations between full cost recompute

  // Move generation
  int maxMoveAttempts = 20; // retries per move before giving up

  // Acceptance tracking
  int windowSize = 100; // sliding window for acceptance ratio
};

// SA temperature schedule with windowed acceptance tracking.
class SASchedule {
public:
  SASchedule() = default;
  SASchedule(double startTemp, int movesPerIter, int maxIters, int greedyIters,
             const SAConfig &cfg = {});

  double getTemperature() const { return temperature; }
  int getIteration() const { return currIteration; }
  int getMovesPerIter() const { return movesPerIter; }
  bool isGreedy() const { return inGreedyStage; }
  bool limitReached() const;
  double getAcceptanceRatio() const;
  void record(bool accepted);
  void cool();

private:
  double temperature = 0.0;
  int currIteration = 0;
  int currGreedyIteration = 0;
  int movesPerIter = 100;
  int maxIters = 5000;
  int greedyIters = 100;
  int windowSize = 100;
  SAConfig config;
  bool inGreedyStage = false;
  std::deque<int> history;
  int acceptCount = 0;
  int rejectCount = 0;

  void trimWindow();
};

// Per-net bounding box state for incremental HPWL updates.
// countAt* tracks how many endpoints sit at each extreme so we can avoid
// full rescans when an endpoint moves away from a non-unique extreme.
struct NetBoundingBox {
  int minCol = 0, maxCol = 0, minRow = 0, maxRow = 0;
  int countAtMinCol = 0, countAtMaxCol = 0;
  int countAtMinRow = 0, countAtMaxRow = 0;
};

struct NetInfo {
  llvm::SmallVector<mlir::Operation *> endpoints; // producer + all consumers
  NetBoundingBox bb;
  bool isMulticast = false; // >4 consumers triggers 2x HPWL penalty
};

// Simulated annealing placement algorithm
//
// Uses HPWL (half-perimeter wire length) cost with incremental bounding-box
// updates. Nets are derived from ObjectFifo producer/consumer connectivity.
// All tile types (compute, mem, shim) participate in SA moves. After SA
// converges, a merge pass collapses mem/shim logical tiles that landed in the
// same column onto a shared physical tile when DMA capacity permits.
class SAPlacer : public Placer {
public:
  explicit SAPlacer(unsigned seed = 0, SAConfig config = {})
      : rngSeed(seed), config(config) {}

  mlir::LogicalResult place(DeviceOp device) override;
  llvm::StringRef getName() const override { return "sa_placer"; }

private:
  // Configuration
  unsigned rngSeed;
  SAConfig config;
  std::mt19937 rng;
  SASchedule schedule;

  // Collected IR state
  CollectedOps collected;
  llvm::DenseMap<mlir::Operation *, AIETileType> tileTypes;
  llvm::DenseMap<mlir::Operation *, int64_t> staticBufferSizes;
  llvm::DenseMap<mlir::Operation *, int64_t> stackSizes;

  // Net model
  struct FifoBufferInfo {
    mlir::Operation *fifoOp = nullptr;
    mlir::Operation *producer = nullptr;
    llvm::SmallVector<mlir::Operation *> consumers;
    int64_t producerSizeBytes = 0;
    int64_t consumerSizeBytes = 0;
    int producerDepth = 0;
    llvm::SmallVector<int> consumerDepths;
    int producerDMADepth = 0;
    llvm::SmallVector<int> consumerDMADepths;
    bool forcesDMA = false;
    bool linkSharedProd = false;

    int consumerDepthAt(size_t ci) const {
      return (ci < consumerDepths.size()) ? consumerDepths[ci] : producerDepth;
    }
  };
  llvm::SmallVector<NetInfo> nets;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<size_t>> tileToNetIndices;
  llvm::SmallVector<FifoBufferInfo> fifoBuffers;

  // Placement state
  llvm::DenseMap<mlir::Operation *, TileID> currentPlacement;
  llvm::DenseMap<TileID, mlir::Operation *> physToLogical;
  llvm::SmallVector<mlir::Operation *> movableTiles;
  llvm::DenseSet<mlir::Operation *> constrainedTiles;

  // Resource tracking
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<size_t>>
      tileToFifoIndices;
  llvm::DenseMap<TileID, int64_t> currentMemUsage;
  llvm::DenseMap<TileID, std::pair<int, int>> currentDMAUsage;
  llvm::DenseMap<size_t, TileID> sharedMemDestination;
  int cachedResourcePenalty = 0;

  // Cascade
  Adjacency cascadeAdjacency;
  static constexpr std::pair<int, int> kCascadeOffsets[] = {{1, 0}, {0, -1}};

  // SA loop state
  int totalCost = 0;
  int cascadePen = 0;
  int bestCost = INT_MAX;        // best legal (resPenalty==0 && cascade==0)
  int bestOverallCost = INT_MAX; // best regardless of legality (fallback)
  PlacementResult bestPlacement; // corresponds to bestCost
  PlacementResult bestOverallPlacement; // corresponds to bestOverallCost
  std::uniform_real_distribution<double> acceptDist{0.0, 1.0};
  int deviceSlots = 0;
  int zeroDeltaMoves = 0, posDeltaMoves = 0, negDeltaMoves = 0;
  int acceptedUphill = 0, rejectedMoves = 0;
  std::chrono::steady_clock::time_point startTime;

  // Phase methods
  mlir::LogicalResult collectAndBuildModel(DeviceOp device);
  mlir::LogicalResult generateInitialPlacement();
  void initializeSAState();
  void runSAMainLoop();
  mlir::LogicalResult finalizePlacement(DeviceOp device);
  mlir::LogicalResult mergeMemShimTiles(DeviceOp device);

  // Cost and resource methods
  void initResourceTracking();
  void addFifoContribution(size_t fifoIdx, int sign);
  int updateResourcePenalty(
      const llvm::SmallVector<std::pair<mlir::Operation *, TileID>>
          &oldPlacements);
  int getResourcePenalty() const { return cachedResourcePenalty; }
  int computePenalty() const;
  int computeMemSpilloverPenalty() const;
  int computeCoreOverflowPenalty() const;
  int computeDMAChannelPenalty() const;
  int computeBDCountPenalty() const;
  int computeMemoryPressure() const;
  int computeAdjacencyPenalty(const Adjacency &adj,
                              llvm::ArrayRef<std::pair<int, int>> validOffsets,
                              int weight) const;
  void generateAllocates();

  // Move methods
  void tryMove(llvm::SmallVector<std::pair<mlir::Operation *, TileID>> &moves);
  bool
  generateMove(llvm::SmallVector<std::pair<mlir::Operation *, TileID>> &moves);
  int evaluateMove(
      mlir::Operation *tile, TileID newPos,
      llvm::SmallVector<std::pair<size_t, NetBoundingBox>> &backups);
  void
  revertMove(llvm::SmallVector<std::pair<size_t, NetBoundingBox>> &backups);

  // Utility methods
  void buildNetModel(llvm::SmallVector<ObjectFifoCreateOp> &objectFifos,
                     llvm::SmallVector<ObjectFifoLinkOp> &objectFifoLinks);
  void buildFifoBufferInfo(DeviceOp device,
                           llvm::ArrayRef<ObjectFifoCreateOp> objectFifos,
                           llvm::ArrayRef<ObjectFifoLinkOp> objectFifoLinks);
  int computeNetHPWL(const NetInfo &net) const;
  int computeTotalHPWL() const;
  void initBoundingBoxes();
  double estimateInitialTemperature(int numSamples);
  bool isLegalPosition(mlir::Operation *tile, TileID pos) const;
  int64_t computeTileMemoryWeight(mlir::Operation *tile) const;
  void printPlacementStats(int64_t elapsedMs) const;
};

} // namespace xilinx::AIE

#endif // AIE_PLACER_H
