//===- AIEPlacer.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-placer"

static std::optional<TileID> resolvePeerPosition(TileLike peer,
                                                 const PlacementResult &placed);

namespace {
// Invoke `fn(peer, thisIsFirst)` for each edge in `adjacency` that
// mentions `op`. `peer` is the OTHER endpoint of the edge; `thisIsFirst`
// is true when `op` is `edge.first` (the producer in compute-peer
// adjacency, the consumer in buffer adjacency, the source in cascade
// adjacency -- callers interpret per their own adjacency convention).
template <typename F>
void forEachPeer(mlir::Operation *op,
                 const SequentialPlacer::Adjacency &adjacency, F &&fn) {
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
// shared-L1 mem-affinity neighbor per the target model. Bounds-checks
// the (col, row) arithmetic and applies both the tile-type and
// isLegalMemAffinity filters in one place so the placer's hot paths
// (reserveNeighborSlots, satisfiesComputePeer forward-look) stay
// declarative.
template <typename F>
void forEachMemAffinityNeighbor(const AIETargetModel &targetModel, TileID at,
                                F &&fn) {
  for (auto [dc, dr] : {std::pair{0, -1}, std::pair{0, 1}, std::pair{-1, 0},
                        std::pair{1, 0}}) {
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
} // namespace

void SequentialPlacer::Adjacency::addEdge(TileLike first, TileLike second) {
  if (!first || !second)
    return;
  unsigned idx = edges.size();
  edges.push_back({first, second});
  if (mlir::isa<LogicalTileOp>(first.getOperation()))
    tileToEdges[first.getOperation()].push_back(idx);
  if (mlir::isa<LogicalTileOp>(second.getOperation()))
    tileToEdges[second.getOperation()].push_back(idx);
}

void SequentialPlacer::Adjacency::addEdgeFromValues(Value a, Value b) {
  if (!a || !b)
    return;
  auto aT = dyn_cast_or_null<TileLike>(a.getDefiningOp());
  auto bT = dyn_cast_or_null<TileLike>(b.getDefiningOp());
  if (aT && bT)
    addEdge(aT, bT);
}

void SequentialPlacer::initialize(const AIETargetModel &targetModel) {
  this->targetModel = &targetModel;
  assignedNonCoreTiles.clear();

  // Collect all available physical tiles from device
  for (int col = 0; col < targetModel.columns(); col++) {
    for (int row = 0; row < targetModel.rows(); row++) {
      TileID id = {col, row};
      AIETileType type = targetModel.getTileType(col, row);

      switch (type) {
      case AIETileType::CoreTile:
        availability.compTiles.push_back(id);
        break;
      case AIETileType::MemTile:
      case AIETileType::ShimNOCTile:
        availability.nonCompTiles.push_back(id);
        break;
      default:
        break;
      }
    }
  }

  // Sort tiles for sequential placement
  // Compute tiles: column-major (fill column vertically before next column)
  auto compTileCmp = [](TileID a, TileID b) {
    if (a.col != b.col)
      return a.col < b.col;
    return a.row < b.row;
  };

  // Non-compute tiles: row-major
  auto rowMajorCmp = [](TileID a, TileID b) {
    if (a.row != b.row)
      return a.row < b.row;
    return a.col < b.col;
  };

  std::sort(availability.compTiles.begin(), availability.compTiles.end(),
            compTileCmp);
  std::sort(availability.nonCompTiles.begin(), availability.nonCompTiles.end(),
            rowMajorCmp);
  availability.compTilesSet.clear();
  availability.compTilesSet.insert(availability.compTiles.begin(),
                                   availability.compTiles.end());

  // Limit cores per column if specified
  if (coresPerCol.has_value()) {
    // Calculate max cores per column in the device
    llvm::DenseMap<int, int> coresInColumn;
    for (const auto &tile : availability.compTiles) {
      coresInColumn[tile.col]++;
    }
    int maxDeviceCoresPerCol = 0;
    for (const auto &[col, count] : coresInColumn) {
      maxDeviceCoresPerCol = std::max(maxDeviceCoresPerCol, count);
    }
    deviceCoresPerCol = maxDeviceCoresPerCol;

    limitCoresPerColumn(*coresPerCol, targetModel.columns());
  }
}

void SequentialPlacer::limitCoresPerColumn(int maxCoresPerCol, int numColumns) {
  // Group compute tiles by column
  llvm::DenseMap<int, std::vector<TileID>> tilesByColumn;

  for (const auto &tile : availability.compTiles) {
    tilesByColumn[tile.col].push_back(tile);
  }

  // Build new limited list, taking only first maxCoresPerCol from each column
  std::vector<TileID> limitedTiles;

  for (int col = 0; col < numColumns; col++) {
    auto it = tilesByColumn.find(col);
    if (it == tilesByColumn.end())
      continue; // No tiles in this column

    const auto &tilesInCol = it->second;
    size_t numToTake =
        std::min(tilesInCol.size(), static_cast<size_t>(maxCoresPerCol));

    // Take first N tiles from this column (already sorted within column)
    limitedTiles.insert(limitedTiles.end(), tilesInCol.begin(),
                        tilesInCol.begin() + numToTake);
  }

  // Replace availability.compTiles with limited list and rebuild the
  // O(1) membership shadow used by the forward-look hot path.
  availability.compTiles = limitedTiles;
  availability.compTilesSet.clear();
  availability.compTilesSet.insert(availability.compTiles.begin(),
                                   availability.compTiles.end());
}

LogicalResult SequentialPlacer::place(DeviceOp device) {
  // Phase 0: Validate options
  if (!targetModel)
    return device.emitError() << "SequentialPlacer::place called before "
                                 "initialize(); targetModel is null";
  if (coresPerCol.has_value() && *coresPerCol > deviceCoresPerCol) {
    return device.emitError() << "requested cores-per-col (" << *coresPerCol
                              << ") exceeds device capacity ("
                              << deviceCoresPerCol << " cores per column)";
  }

  // Phase 1: Collect operations needed for placement
  SmallVector<LogicalTileOp> logicalTiles;
  SmallVector<ObjectFifoCreateOp> objectFifos;
  SmallVector<ObjectFifoLinkOp> objectFifoLinks;
  SmallVector<CascadeFlowOp> cascadeFlows;
  SmallVector<FlowOp> flows;
  SmallVector<PacketFlowOp> pktFlows;

  device.walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<LogicalTileOp>([&](auto lt) { logicalTiles.push_back(lt); })
        .Case<ObjectFifoCreateOp>([&](auto of) { objectFifos.push_back(of); })
        .Case<ObjectFifoLinkOp>(
            [&](auto link) { objectFifoLinks.push_back(link); })
        .Case<CascadeFlowOp>([&](auto cf) { cascadeFlows.push_back(cf); })
        .Case<FlowOp>([&](auto f) { flows.push_back(f); })
        .Case<PacketFlowOp>([&](auto pf) { pktFlows.push_back(pf); });
  });

  // Phase 2a: Build placement constraints
  auto bufferAdjacency = buildBufferAdjacency(logicalTiles);
  auto computePeerAdjacency = buildComputePeerAdjacency(objectFifos);

  // Phase 2b: Build channel requirements from ObjectFifo and Flow connectivity,
  // and cascade adjacency constraints from CascadeFlow connectivity.
  auto channelRequirements =
      buildChannelRequirements(objectFifos, objectFifoLinks);
  addChannelRequirementsFromFlows(flows, pktFlows, channelRequirements);

  auto cascadeAdjacency = buildCascadeAdjacency(cascadeFlows);

  // Phase 2c: For every compute LTO, compute the minimum number of its
  // compute-peer producers/consumers that must land on physical neighbors so
  // the LTO fits its DMA channel budget. A compute→compute ObjectFifo can use
  // shared neighbor memory (no DMA) when the two endpoints sit on physical-
  // neighbor compute tiles; otherwise it consumes one DMA channel on each
  // side. Non-compute-peer fifos (from/to memtile or shim) always need a DMA
  // channel and already appear in `channelRequirements`, so the headroom left
  // for compute-peer fifos is (DMA-budget − channelRequirements). If a
  // Worker has more compute-peer fifos than that headroom, the surplus MUST
  // come from neighbors.
  llvm::DenseMap<Operation *, int> needNeighborIn, needNeighborOut;
  for (auto lt : logicalTiles) {
    if (lt.getTileType() != AIETileType::CoreTile)
      continue;
    auto [peerIn, peerOut] =
        totalComputePeers(lt.getOperation(), computePeerAdjacency);
    int inBudget = lt.getNumDestConnections(WireBundle::DMA);
    int outBudget = lt.getNumSourceConnections(WireBundle::DMA);
    auto [nonPeerIn, nonPeerOut] =
        channelRequirements.lookup(lt.getOperation());
    needNeighborIn[lt.getOperation()] =
        std::max(0, peerIn - (inBudget - nonPeerIn));
    needNeighborOut[lt.getOperation()] =
        std::max(0, peerOut - (outBudget - nonPeerOut));
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << ": LTO " << lt.getLoc() << " peerIn=" << peerIn
               << " peerOut=" << peerOut << " nonPeerIn=" << nonPeerIn
               << " nonPeerOut=" << nonPeerOut << " => needNeighborIn="
               << needNeighborIn[lt.getOperation()]
               << " needNeighborOut=" << needNeighborOut[lt.getOperation()]
               << "\n");
  }
  // The per-place() context bundles demand bookkeeping and the soft-
  // reservation table. Soft-reservation: when a demand-bearing LTO is
  // placed, its mem-affinity neighbor slots are reserved for that LTO's
  // compute peers. Other LTOs fall into a reserved slot only if no
  // unreserved slot fits, preventing unrelated low-demand LTOs from
  // cornering a high-fanin Worker's neighbors and making the design
  // unsolvable.
  PlacementContext ctx{*targetModel, computePeerAdjacency, needNeighborIn,
                       needNeighborOut, /*reservedFor=*/{}};

  // Per-kind predicates / labelers shared by all phase-3 call sites below.
  // Buffer: consumer LTO (edge.first) must satisfy isLegalMemAffinity to the
  // owner tile (edge.second).
  auto bufferPred = [this](TileID consumerPos, TileID ownerPos) {
    return targetModel->isLegalMemAffinity(consumerPos.col, consumerPos.row,
                                           ownerPos.col, ownerPos.row);
  };
  auto bufferLabel = [](bool thisIsConsumer) -> StringRef {
    return thisIsConsumer ? "shared-L1 buffer owner"
                          : "shared-L1 buffer consumer";
  };
  // Cascade: same predicate as AIELowerCascadeFlowsPass -- src (edge.first)
  // must be one row North or one column West of dst (edge.second); rows
  // increase upward, so isSouth(src,dst) means dst is south of src.
  auto cascadePred = [this](TileID srcPos, TileID dstPos) {
    return targetModel->isSouth(srcPos.col, srcPos.row, dstPos.col,
                                dstPos.row) ||
           targetModel->isEast(srcPos.col, srcPos.row, dstPos.col, dstPos.row);
  };
  auto cascadeLabel = [](bool thisIsSrc) -> StringRef {
    return thisIsSrc ? "cascade destination" : "cascade source";
  };

  // Bundle each pairwise-legality adjacency with its predicate, peer
  // label, short error name, and educational hint so the pinned-tile
  // violation paths can share one enforceAdjacency call per kind.
  AdjacencyKind bufferKind{
      bufferAdjacency, bufferPred, bufferLabel, "shared-L1 buffer",
      "shared-L1 buffer adjacency requires this LTO to be on a tile "
      "whose L1 is shared with the buffer owner's tile (N/S neighbor, "
      "or W neighbor per the device's checkerboard rule)"};
  AdjacencyKind cascadeKind{
      cascadeAdjacency, cascadePred, cascadeLabel, "cascade",
      "cascade adjacency requires the destination tile to be one row "
      "South or one column East of the source tile"};

  // Phase 3: Place constrained tiles then compute tiles.
  //
  // Process LogicalTileOps in order of how constrained they are: fully pinned
  // (both col and row) first, partially constrained (col xor row) next, fully
  // unpinned last. This guarantees every pinned coordinate is recorded — and
  // for CoreTiles, removed from availability — before any unpinned tile gets
  // to pick from `availability.compTiles`, so unpinned tiles cannot land on a
  // coordinate that a later-iterated pinned tile claims.
  //
  // Within the unpinned bucket, place compute LTOs with the highest
  // neighbor-demand first so a high-fanin Worker grabs an interior tile (with
  // two compute neighbors) before its producers consume the surrounding
  // slots. Producers and consumers that the heavy Worker needs as physical
  // neighbors are then steered to those slots by `computePeerAdjacency`.
  SmallVector<LogicalTileOp> orderedTiles(logicalTiles.begin(),
                                          logicalTiles.end());
  std::stable_sort(orderedTiles.begin(), orderedTiles.end(),
                   [&](LogicalTileOp a, LogicalTileOp b) {
                     auto rank = [](LogicalTileOp lt) {
                       bool hasCol = lt.tryGetCol().has_value();
                       bool hasRow = lt.tryGetRow().has_value();
                       if (hasCol && hasRow)
                         return 0; // fully pinned
                       if (hasCol || hasRow)
                         return 1; // partially constrained
                       return 2;   // fully unpinned
                     };
                     int ra = rank(a), rb = rank(b);
                     if (ra != rb)
                       return ra < rb;
                     // Same constraint level: high priority first. Priority
                     // = max(self demand, highest peer demand) so a Worker's
                     // compute-peer producers/consumers also rise to the
                     // front and claim the Worker's neighbor tiles before
                     // unrelated LTOs consume them.
                     int pa = ctx.placementPriority(a.getOperation());
                     int pb = ctx.placementPriority(b.getOperation());
                     if (pa != pb)
                       return pa > pb;
                     // Among equal priority, place the high-demand LTO
                     // itself first so its peers can be steered to its
                     // neighbors immediately afterward.
                     return ctx.neighborDemand(a.getOperation()) >
                            ctx.neighborDemand(b.getOperation());
                   });

  for (auto logicalTile : orderedTiles) {
    // Place fully constrained tiles at their specified coordinates
    auto col = logicalTile.tryGetCol();
    auto row = logicalTile.tryGetRow();
    if (col && row) {
      TileID tile{*col, *row};
      if (failed(enforceAdjacency(logicalTile, tile, bufferKind)))
        return failure();
      if (failed(enforceAdjacency(logicalTile, tile, cascadeKind)))
        return failure();
      if (failed(validateAndUpdateChannelUsage(
              logicalTile, tile, channelRequirements,
              PlacementOrigin::Pinned)))
        return failure();

      // Only CoreTile placements remove from availability + reserve
      // neighbor slots; mem/shim may still host additional logical
      // tiles as long as channel/DMA capacity permits.
      recordPlacement(logicalTile, tile, ctx, "pinned");
      continue;
    }

    // Place compute tiles with partial constraint support
    if (logicalTile.getTileType() == AIETileType::CoreTile) {
      auto isReservedForOtherBound = [&](Operation *lto, TileID candidate) {
        return ctx.isReservedForOther(lto, candidate);
      };
      UnpinnedPlacementInputs inputs{
          bufferAdjacency,    bufferPred,         cascadeAdjacency,
          cascadePred,        computePeerAdjacency, needNeighborIn,
          needNeighborOut,    isReservedForOtherBound};
      auto search =
          findUnconstrainedCoreCandidate(logicalTile, col, row, inputs);
      std::optional<TileID> placement = search.placement;
      bool sawConstraintMatch = search.sawConstraintMatch;
      bool allConstraintMatchesFailedAdjacency =
          search.allConstraintMatchesFailedAdjacency;
      bool computePeerWasCause = search.computePeerWasCause;

      if (!placement) {
        bool adjacencyWasCause =
            sawConstraintMatch && allConstraintMatchesFailedAdjacency &&
            bufferAdjacency.hasEdges(logicalTile.getOperation());
        bool hasCascade =
            cascadeAdjacency.hasEdges(logicalTile.getOperation());
        InFlightDiagnostic diag = logicalTile.emitError();
        if (col || row) {
          diag << "no compute tile available matching constraint ("
               << (col ? std::to_string(*col) : "?") << ", "
               << (row ? std::to_string(*row) : "?") << ")"
               << (adjacencyWasCause ? " and shared-L1 buffer adjacency" : "")
               << (hasCascade ? " and cascade adjacency" : "")
               << (computePeerWasCause ? " and compute-peer DMA budget" : "");
        } else {
          diag << "no available compute tiles for placement"
               << (adjacencyWasCause
                       ? " (shared-L1 buffer adjacency unsatisfiable)"
                       : "")
               << (hasCascade ? " (cascade adjacency unsatisfiable)" : "")
               << (computePeerWasCause
                       ? " (compute-peer DMA budget unsatisfiable)"
                       : "");
        }
        if (adjacencyWasCause)
          attachPeerNotes(diag, logicalTile, bufferAdjacency, bufferLabel);
        if (hasCascade)
          attachPeerNotes(diag, logicalTile, cascadeAdjacency, cascadeLabel);
        if (computePeerWasCause) {
          attachPeerNotes(diag, logicalTile, computePeerAdjacency,
                          [](bool thisIsProducer) -> StringRef {
                            return thisIsProducer ? "compute-peer consumer"
                                                  : "compute-peer producer";
                          });
          diag.attachNote()
              << "to fix, pin this LTO to a position that is shared-L1 with "
                 "the placed compute-peers above, or pin one of those peers "
                 "to a different coordinate so this LTO's constraints become "
                 "satisfiable";
        }
        return failure();
      }

      if (failed(validateAndUpdateChannelUsage(
              logicalTile, *placement, channelRequirements,
              PlacementOrigin::Selected)))
        return failure();

      recordPlacement(logicalTile, *placement, ctx, "unconstrained");
    }

    if (logicalTile.getTileType() == AIETileType::ShimPLTile) {
      return logicalTile.emitError(
          "DMA channel-based SequentialPlacer does not support unplaced "
          "ShimPLTiles (no DMAs).");
    }
  }

  // Phase 4: place every still-unplaced non-core (mem/shim) LTO at the
  // centroid column of its placed core peers.
  return placeNonCoreLogicalTiles(logicalTiles, objectFifos, flows, pktFlows,
                                  channelRequirements);
}

LogicalResult SequentialPlacer::placeNonCoreLogicalTiles(
    ArrayRef<LogicalTileOp> logicalTiles, ArrayRef<ObjectFifoCreateOp> objectFifos,
    ArrayRef<FlowOp> flows, ArrayRef<PacketFlowOp> pktFlows,
    const llvm::DenseMap<Operation *, std::pair<int, int>>
        &channelRequirements) {
  // Connectivity adjacencies (per-fifo and per-flow) drive the centroid
  // BFS that picks the column near a non-core LTO's compute peers.
  auto objectFifoAdjacency = buildObjectFifoAdjacency(objectFifos);
  auto flowAdjacency = buildFlowAdjacency(flows, pktFlows);
  SmallVector<const Adjacency *, 2> connectivityAdjacencies = {
      &objectFifoAdjacency, &flowAdjacency};

  // Sort the unplaced non-core LTOs by descending channel demand so the
  // heaviest consumers (e.g. memtile joins that need many input channels)
  // get first pick of physical tiles before lighter pass-through fifos
  // crowd into the same column and leave no room.
  SmallVector<LogicalTileOp> nonCoreOrdered;
  for (auto logicalTile : logicalTiles) {
    if (result.count(logicalTile.getOperation()))
      continue;
    AIETileType tileType = logicalTile.getTileType();
    if (tileType == AIETileType::CoreTile ||
        tileType == AIETileType::ShimPLTile)
      continue;
    nonCoreOrdered.push_back(logicalTile);
  }
  std::stable_sort(nonCoreOrdered.begin(), nonCoreOrdered.end(),
                   [&](LogicalTileOp a, LogicalTileOp b) {
                     auto demand = [&](LogicalTileOp lt) {
                       auto chans =
                           channelRequirements.lookup(lt.getOperation());
                       return chans.first + chans.second;
                     };
                     return demand(a) > demand(b);
                   });

  for (auto logicalTile : nonCoreOrdered) {
    if (failed(placeNonCoreTileByCentroid(logicalTile, connectivityAdjacencies,
                                          channelRequirements)))
      return failure();
  }
  return success();
}

int SequentialPlacer::PlacementContext::placementPriority(Operation *op) const {
  int self = neighborDemand(op);
  int best = 0;
  forEachPeer(op, computePeerAdjacency,
              [&](TileLike peer, bool /*thisIsFirst*/) {
                best = std::max(best, neighborDemand(peer.getOperation()));
              });
  return std::max(self, best);
}

void SequentialPlacer::PlacementContext::reserveNeighborSlots(
    Operation *placedOp, TileID at) {
  if (neighborDemand(placedOp) <= 0)
    return;
  forEachMemAffinityNeighbor(targetModel, at, [&](TileID nb) {
    reservedFor[nb].push_back(placedOp);
  });
}

bool SequentialPlacer::PlacementContext::isReservedForOther(
    Operation *lto, TileID candidate) const {
  auto it = reservedFor.find(candidate);
  if (it == reservedFor.end())
    return false;
  for (Operation *owner : it->second) {
    if (owner == lto)
      continue; // The LTO's own slot reservation never counts as "other".
    bool isPeer = false;
    forEachPeer(lto, computePeerAdjacency,
                [&](TileLike peer, bool /*thisIsFirst*/) {
                  if (peer.getOperation() == owner)
                    isPeer = true;
                });
    if (!isPeer)
      return true;
  }
  return false;
}

SequentialPlacer::UnpinnedSearchResult
SequentialPlacer::findUnconstrainedCoreCandidate(
    LogicalTileOp logicalTile, std::optional<int> col, std::optional<int> row,
    const UnpinnedPlacementInputs &inputs) {
  UnpinnedSearchResult result;
  auto neighborDemand = [&](Operation *op) {
    return inputs.needNeighborIn.lookup(op) +
           inputs.needNeighborOut.lookup(op);
  };

  // compTiles is sorted column-major top-to-bottom. For high-fanin LTOs
  // (neighborDemand > 0) the placer needs an INTERIOR tile (with two
  // compute neighbors) so its compute-peer producers/consumers can sit
  // on either side. Reorder the candidate iteration for THIS LTO only
  // to prefer high-neighbor-count rows first; LTOs with no neighbor
  // demand fall through to the default column-major order so existing
  // single-worker tests keep landing on (col, 2).
  auto computeNeighborCount = [&](TileID t) {
    int n = 0;
    if (t.row > 0 &&
        targetModel->getTileType(t.col, t.row - 1) == AIETileType::CoreTile)
      ++n;
    if (t.row + 1 < targetModel->rows() &&
        targetModel->getTileType(t.col, t.row + 1) == AIETileType::CoreTile)
      ++n;
    return n;
  };
  SmallVector<TileID> orderedCandidates(availability.compTiles.begin(),
                                        availability.compTiles.end());
  if (neighborDemand(logicalTile.getOperation()) > 0) {
    std::stable_sort(orderedCandidates.begin(), orderedCandidates.end(),
                     [&](TileID a, TileID b) {
                       if (a.col != b.col)
                         return a.col < b.col;
                       int na = computeNeighborCount(a);
                       int nb = computeNeighborCount(b);
                       if (na != nb)
                         return na > nb;
                       return a.row < b.row;
                     });
  }

  // Two passes: first prefer candidates that aren't reserved for an
  // unrelated demand-bearing LTO's peers; fall back to reserved
  // candidates only if nothing else fits.
  for (bool allowReserved : {false, true}) {
    for (const TileID &candidate : orderedCandidates) {
      if (col && candidate.col != *col)
        continue;
      if (row && candidate.row != *row)
        continue;
      if (!allowReserved &&
          inputs.isReservedForOther(logicalTile.getOperation(), candidate))
        continue;
      result.sawConstraintMatch = true;
      if (!satisfiesAdjacency(logicalTile, candidate, inputs.bufferAdjacency,
                              inputs.bufferPred))
        continue;
      result.allConstraintMatchesFailedAdjacency = false;
      if (!satisfiesAdjacency(logicalTile, candidate, inputs.cascadeAdjacency,
                              inputs.cascadePred))
        continue;
      if (!satisfiesComputePeer(logicalTile, candidate,
                                inputs.computePeerAdjacency,
                                inputs.needNeighborIn,
                                inputs.needNeighborOut)) {
        result.computePeerWasCause = true;
        continue;
      }
      result.placement = candidate;
      return result;
    }
  }
  return result;
}

LogicalResult SequentialPlacer::validateAndUpdateChannelUsage(
    LogicalTileOp logicalTile, TileID tile,
    const llvm::DenseMap<Operation *, std::pair<int, int>> &channelRequirements,
    PlacementOrigin origin) {

  // Get channel requirements
  auto [inChannels, outChannels] =
      channelRequirements.lookup(logicalTile.getOperation());

  // Validate capacity
  if (!hasAvailableChannels(tile, inChannels, outChannels)) {
    // Get max channels
    int maxIn = logicalTile.getNumDestConnections(WireBundle::DMA);
    int maxOut = logicalTile.getNumSourceConnections(WireBundle::DMA);
    int availIn = maxIn - availability.inputChannelsUsed[tile];
    int availOut = maxOut - availability.outputChannelsUsed[tile];

    auto diag = logicalTile.emitError();
    diag << "tile (" << tile.col << ", " << tile.row << ") requires "
         << inChannels << " input/" << outChannels
         << " output DMA channels, but only " << availIn << " input/"
         << availOut << " output available";
    if (origin == PlacementOrigin::Selected)
      diag.attachNote() << "placer selected this tile; to fix, pin this LTO "
                           "to a tile with more spare DMA capacity, or reduce "
                           "the LTO's DMA fanin (e.g. via memtile staging)";
    return failure();
  }

  // Update channel usage
  if (inChannels > 0)
    updateChannelUsage(tile, DmaDir::In, inChannels);
  if (outChannels > 0)
    updateChannelUsage(tile, DmaDir::Out, outChannels);

  return success();
}

llvm::DenseMap<Operation *, std::pair<int, int>>
SequentialPlacer::buildChannelRequirements(
    SmallVector<ObjectFifoCreateOp> &objectFifos,
    SmallVector<ObjectFifoLinkOp> &objectFifoLinks) {
  llvm::DenseMap<Operation *, std::pair<int, int>> channelRequirements;

  // Build map of ObjectFifo name -> CreateOp for link processing
  llvm::StringMap<ObjectFifoCreateOp> fifoNameToOp;
  for (auto ofOp : objectFifos) {
    fifoNameToOp[ofOp.getSymName()] = ofOp;
  }

  // For each fifo involved in a link, track which side IS the link tile so we
  // don't double-count it: the subsequent link loop credits the link tile with
  // (#sources input + #dests output) and we'd otherwise count the same channels
  // again in the per-fifo loop below. A linked source fifo terminates at the
  // link tile on its CONSUMER side; a linked dest fifo originates from the
  // link tile on its PRODUCER side.
  llvm::DenseSet<llvm::StringRef> linkedAsSource;
  llvm::DenseSet<llvm::StringRef> linkedAsDest;
  for (auto linkOp : objectFifoLinks) {
    for (auto srcFifoAttr : linkOp.getFifoIns())
      if (auto sym = dyn_cast<FlatSymbolRefAttr>(srcFifoAttr))
        linkedAsSource.insert(sym.getValue());
    for (auto dstFifoAttr : linkOp.getFifoOuts())
      if (auto sym = dyn_cast<FlatSymbolRefAttr>(dstFifoAttr))
        linkedAsDest.insert(sym.getValue());
  }

  // Count channels for every ObjectFifo, omitting only the link-tile side of
  // linked fifos. The off-link-tile endpoint still needs DMA: a shim that
  // produces a linked source fifo consumes an MM2S channel; a core that
  // consumes a linked dest fifo consumes an S2MM channel.
  for (auto ofOp : objectFifos) {
    bool skipConsumerSide = linkedAsSource.count(ofOp.getSymName());
    bool skipProducerSide = linkedAsDest.count(ofOp.getSymName());

    Value producerTile = ofOp.getProducerTile();
    auto *producerOp = producerTile.getDefiningOp();
    auto producerLogicalTile = dyn_cast_or_null<LogicalTileOp>(producerOp);

    // Check if ANY consumer is a different tile type (needs DMA channel)
    bool producerNeedsDMA = false;

    for (Value consumerTile : ofOp.getConsumerTiles()) {
      auto *consumerOp = consumerTile.getDefiningOp();
      auto consumerLogicalTile = dyn_cast_or_null<LogicalTileOp>(consumerOp);

      // Skip core-to-core connections (SequentialPlacer doesn't account for
      // these)
      if (producerLogicalTile && consumerLogicalTile &&
          producerLogicalTile.getTileType() == AIETileType::CoreTile &&
          consumerLogicalTile.getTileType() == AIETileType::CoreTile)
        continue;

      // This consumer needs a DMA channel (unless it's the link tile, which
      // the link loop credits separately).
      if (consumerOp && !skipConsumerSide)
        channelRequirements[consumerOp].first++; // input++

      producerNeedsDMA = true;
    }

    // Producer needs ONE output channel if any consumer needs DMA (unless the
    // producer IS the link tile of an outgoing linked dest fifo).
    if (producerNeedsDMA && producerOp && !skipProducerSide)
      channelRequirements[producerOp].second++; // output++
  }

  // For linked ObjectFifos, count channels based on the link structure
  // Link tiles need channels for all sources and destinations
  for (auto linkOp : objectFifoLinks) {

    // Find the tile that fifos are linked on
    Operation *linkTileOp = nullptr;

    // Get link tile from first source fifo's consumer
    for (auto srcFifoAttr : linkOp.getFifoIns()) {
      auto sym = dyn_cast<FlatSymbolRefAttr>(srcFifoAttr);
      if (!sym)
        continue;
      auto srcFifoName = sym.getValue();
      auto it = fifoNameToOp.find(srcFifoName);
      if (it == fifoNameToOp.end())
        continue;

      auto srcFifo = it->second;
      for (Value consumerTile : srcFifo.getConsumerTiles()) {
        if (auto *consumerOp = consumerTile.getDefiningOp()) {
          linkTileOp = consumerOp;
          break;
        }
      }
      if (linkTileOp)
        break;
    }

    if (!linkTileOp)
      continue;

    // Link tile needs:
    // - Input channels = number of source ObjectFifos
    // - Output channels = number of dest ObjectFifos
    int numInputChannels = linkOp.getFifoIns().size();
    int numOutputChannels = linkOp.getFifoOuts().size();

    channelRequirements[linkTileOp].first += numInputChannels;
    channelRequirements[linkTileOp].second += numOutputChannels;
  }

  return channelRequirements;
}

// Walk view-like aliasing memref ops back to the underlying BufferOp.
static BufferOp traceToBuffer(Value val) {
  for (Operation *op = val.getDefiningOp(); op;) {
    if (auto buf = dyn_cast<BufferOp>(op))
      return buf;
    if (auto view = dyn_cast<ViewLikeOpInterface>(op)) {
      val = view.getViewSource();
      op = val.getDefiningOp();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

static std::optional<TileID>
resolvePeerPosition(TileLike peer, const PlacementResult &placed) {
  if (!peer)
    return std::nullopt;
  auto it = placed.find(peer.getOperation());
  if (it != placed.end())
    return it->second;
  auto col = peer.tryGetCol();
  auto row = peer.tryGetRow();
  if (col && row)
    return TileID{*col, *row};
  return std::nullopt;
}

SequentialPlacer::Adjacency
SequentialPlacer::buildBufferAdjacency(ArrayRef<LogicalTileOp> logicalTiles) {
  Adjacency adjacency;
  for (auto consumer : logicalTiles) {
    if (consumer.getTileType() != AIETileType::CoreTile)
      continue;
    CoreOp core = nullptr;
    for (Operation *user : consumer.getResult().getUsers())
      if (auto c = dyn_cast<CoreOp>(user)) {
        core = c;
        break;
      }
    if (!core)
      continue;

    llvm::SmallDenseSet<Operation *, 4> seenOwners;
    core.getBody().walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        BufferOp buf = traceToBuffer(operand);
        if (!buf)
          continue;
        auto owner = dyn_cast_or_null<TileLike>(buf.getTile().getDefiningOp());
        if (!owner)
          continue;
        if (owner.getOperation() == consumer.getOperation())
          continue;
        if (!seenOwners.insert(owner.getOperation()).second)
          continue;
        adjacency.addEdge(TileLike(consumer), owner);
      }
    });
  }
  return adjacency;
}

SequentialPlacer::Adjacency
SequentialPlacer::buildCascadeAdjacency(ArrayRef<CascadeFlowOp> cascadeFlows) {
  Adjacency adjacency;
  for (auto cf : cascadeFlows) {
    TileLike src = cf.getSourceTileLike();
    TileLike dst = cf.getDestTileLike();
    if (!src || !dst)
      continue;
    adjacency.addEdge(src, dst);
  }
  return adjacency;
}

SequentialPlacer::Adjacency SequentialPlacer::buildComputePeerAdjacency(
    ArrayRef<ObjectFifoCreateOp> objectFifos) {
  Adjacency adjacency;
  for (auto ofOp : objectFifos) {
    auto producer =
        dyn_cast_or_null<LogicalTileOp>(ofOp.getProducerTile().getDefiningOp());
    if (!producer || producer.getTileType() != AIETileType::CoreTile)
      continue;
    for (Value consumerVal : ofOp.getConsumerTiles()) {
      auto consumer =
          dyn_cast_or_null<LogicalTileOp>(consumerVal.getDefiningOp());
      if (!consumer || consumer.getTileType() != AIETileType::CoreTile)
        continue;
      // Self-loops live entirely in the tile's own L1 and consume no DMA
      // channel, so they should not push the LTO toward a neighbor-heavy
      // placement.
      if (producer.getOperation() == consumer.getOperation())
        continue;
      adjacency.addEdge(TileLike(producer), TileLike(consumer));
    }
  }
  return adjacency;
}

bool SequentialPlacer::satisfiesAdjacency(
    LogicalTileOp logicalTile, TileID candidate, const Adjacency &adjacency,
    llvm::function_ref<bool(TileID firstPos, TileID secondPos)> pred) const {
  bool ok = true;
  forEachPeer(logicalTile.getOperation(), adjacency,
              [&](TileLike peer, bool thisIsFirst) {
                if (!ok)
                  return;
                auto peerPos = resolvePeerPosition(peer, result);
                if (!peerPos)
                  return;
                TileID firstPos = thisIsFirst ? candidate : *peerPos;
                TileID secondPos = thisIsFirst ? *peerPos : candidate;
                if (!pred(firstPos, secondPos))
                  ok = false;
              });
  return ok;
}

LogicalResult SequentialPlacer::enforceAdjacency(LogicalTileOp logicalTile,
                                                 TileID tile,
                                                 const AdjacencyKind &kind) {
  if (satisfiesAdjacency(logicalTile, tile, kind.adjacency, kind.pred))
    return success();
  auto diag = logicalTile.emitError()
              << "tile (" << tile.col << ", " << tile.row << ") violates "
              << kind.name << " adjacency";
  attachPeerNotes(diag, logicalTile, kind.adjacency, kind.peerLabel);
  diag.attachNote() << kind.constraintHint;
  return failure();
}

void SequentialPlacer::recordPlacement(LogicalTileOp logicalTile, TileID tile,
                                       PlacementContext &ctx,
                                       StringRef debugLabel) {
  result[logicalTile] = tile;
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": placed " << debugLabel << " LTO "
                          << logicalTile.getLoc() << " at (" << tile.col << ", "
                          << tile.row << ")\n");
  if (logicalTile.getTileType() == AIETileType::CoreTile) {
    availability.removeTile(tile, AIETileType::CoreTile);
    ctx.reserveNeighborSlots(logicalTile.getOperation(), tile);
  }
}

std::pair<int, int>
SequentialPlacer::totalComputePeers(Operation *op, const Adjacency &adjacency) {
  int in = 0, out = 0;
  forEachPeer(op, adjacency, [&](TileLike /*peer*/, bool thisIsFirst) {
    // In compute-peer adjacency edge.first = producer, edge.second =
    // consumer. So `op` being `first` means it's the producer (counts
    // as OUT); `op` being `second` means it's the consumer (counts
    // as IN).
    if (thisIsFirst)
      ++out;
    else
      ++in;
  });
  return {in, out};
}

bool SequentialPlacer::satisfiesComputePeer(
    LogicalTileOp logicalTile, TileID candidate,
    const Adjacency &computePeerAdjacency,
    const llvm::DenseMap<Operation *, int> &needNeighborIn,
    const llvm::DenseMap<Operation *, int> &needNeighborOut) const {
  // No compute-peer edges => the compute-peer DMA budget constraint is
  // vacuous for this LTO. (Phase 2c can still populate a non-zero
  // needNeighborIn/Out for a tile whose non-peer channel count exceeds
  // its DMA budget, but no neighbor placement can fix that and the
  // channel validator emits the right error later.)
  if (!computePeerAdjacency.hasEdges(logicalTile.getOperation()))
    return true;

  // Self side: count placed peers that would land non-neighbor of candidate,
  // and reject if that exceeds our slack. In compute-peer adjacency,
  // edge.first = producer, edge.second = consumer; so `logicalTile` being
  // the second endpoint (thisIsFirst == false) means it's a consumer
  // (peer is producing INTO it), which counts toward the IN direction.
  int selfNonNeighborIn = 0, selfNonNeighborOut = 0;
  int selfNeighborIn = 0, selfNeighborOut = 0;
  forEachPeer(
      logicalTile.getOperation(), computePeerAdjacency,
      [&](TileLike peer, bool thisIsFirst) {
        auto peerPos = resolvePeerPosition(peer, result);
        if (!peerPos)
          return;
        bool isNeighbor = targetModel->isLegalMemAffinity(
            candidate.col, candidate.row, peerPos->col, peerPos->row);
        bool thisIsConsumer = !thisIsFirst;
        if (isNeighbor) {
          if (thisIsConsumer)
            ++selfNeighborIn;
          else
            ++selfNeighborOut;
        } else {
          if (thisIsConsumer)
            ++selfNonNeighborIn;
          else
            ++selfNonNeighborOut;
        }
      });
  auto [selfTotalIn, selfTotalOut] =
      totalComputePeers(logicalTile.getOperation(), computePeerAdjacency);
  int selfNeedIn = needNeighborIn.lookup(logicalTile.getOperation());
  int selfNeedOut = needNeighborOut.lookup(logicalTile.getOperation());
  int selfSlackIn = selfTotalIn - selfNeedIn;
  int selfSlackOut = selfTotalOut - selfNeedOut;
  if (selfNonNeighborIn > selfSlackIn)
    return false;
  if (selfNonNeighborOut > selfSlackOut)
    return false;

  // Forward-look: even if no placed peer violates slack at this candidate,
  // future peers still to be placed must land on a physical compute neighbor
  // of candidate. Count how many of candidate's physical compute-tile
  // neighbors are still in availability.compTiles (free slots that a future
  // peer could occupy). Iterate all 4 cardinal directions and ask
  // isLegalMemAffinity to decide which are actually shared-L1 neighbors --
  // AIE2 cores share L1 with N, S, and the checkerboard W neighbor (not E).
  // The remaining need after subtracting already-neighbor placements must fit
  // in those free slots. Free slots can be used for an IN-peer OR an OUT-peer
  // (not both), so we approximate by checking the sum.
  if (selfNeedIn > 0 || selfNeedOut > 0) {
    int freeNeighborSlots = 0;
    forEachMemAffinityNeighbor(*targetModel, candidate, [&](TileID nb) {
      if (availability.compTilesSet.contains(nb))
        ++freeNeighborSlots;
    });
    int remainingInNeed = std::max(0, selfNeedIn - selfNeighborIn);
    int remainingOutNeed = std::max(0, selfNeedOut - selfNeighborOut);
    if (remainingInNeed + remainingOutNeed > freeNeighborSlots)
      return false;
  }

  // Symmetric side: for each already-placed compute peer p, placing
  // logicalTile at candidate adds one non-neighbor entry to p's count if
  // candidate isn't a neighbor of p. Re-derive p's count using the current
  // `result` plus this candidate, then check p's slack.
  bool symmetricOk = true;
  forEachPeer(
      logicalTile.getOperation(), computePeerAdjacency,
      [&](TileLike peer, bool thisIsFirst) {
        if (!symmetricOk)
          return;
        auto peerLTO = dyn_cast_or_null<LogicalTileOp>(peer.getOperation());
        if (!peerLTO)
          return;
        auto peerPos = resolvePeerPosition(peer, result);
        if (!peerPos)
          return;
        // From peer's POV, logicalTile is one of peer's compute peers.
        // If logicalTile is the consumer here (thisIsFirst == false),
        // peer is the producer and logicalTile sits on peer's OUT side.
        bool logicalIsPeerOut = !thisIsFirst;
        int peerNeedDir = logicalIsPeerOut
                              ? needNeighborOut.lookup(peerLTO.getOperation())
                              : needNeighborIn.lookup(peerLTO.getOperation());
        auto [peerTotalIn, peerTotalOut] =
            totalComputePeers(peerLTO.getOperation(), computePeerAdjacency);
        int peerTotalDir = logicalIsPeerOut ? peerTotalOut : peerTotalIn;
        int peerSlackDir = peerTotalDir - peerNeedDir;

        // Count peer's currently placed non-neighbor peers in this direction.
        int peerNonNeighborDir = 0;
        forEachPeer(
            peerLTO.getOperation(), computePeerAdjacency,
            [&](TileLike peerPeer, bool peerLTOIsFirst) {
              // peerLTO is the consumer in this edge when it's the
              // second endpoint (compute-peer convention is
              // producer=first, consumer=second). logicalIsPeerOut ==
              // true selects edges where peerLTO is the producer (i.e.
              // peerLTOIsFirst), so we want the OPPOSITE direction.
              bool peerIsConsumerHere = !peerLTOIsFirst;
              if (logicalIsPeerOut == peerIsConsumerHere)
                return;
              if (peerPeer.getOperation() == logicalTile.getOperation()) {
                // The edge to logicalTile we are currently deciding:
                // count candidate against it.
                if (!targetModel->isLegalMemAffinity(
                        peerPos->col, peerPos->row, candidate.col,
                        candidate.row))
                  ++peerNonNeighborDir;
                return;
              }
              auto pp = resolvePeerPosition(peerPeer, result);
              if (!pp)
                return;
              if (!targetModel->isLegalMemAffinity(peerPos->col, peerPos->row,
                                                   pp->col, pp->row))
                ++peerNonNeighborDir;
            });
        if (peerNonNeighborDir > peerSlackDir)
          symmetricOk = false;
      });
  return symmetricOk;
}

void SequentialPlacer::attachPeerNotes(
    InFlightDiagnostic &diag, LogicalTileOp logicalTile,
    const Adjacency &adjacency,
    llvm::function_ref<StringRef(bool thisIsFirst)> labelPeer) const {
  int unplacedCount = 0;
  forEachPeer(logicalTile.getOperation(), adjacency,
              [&](TileLike peer, bool thisIsFirst) {
                auto peerPos = resolvePeerPosition(peer, result);
                if (!peerPos) {
                  ++unplacedCount;
                  return;
                }
                diag.attachNote(peer.getLoc())
                    << labelPeer(thisIsFirst) << " peer placed at ("
                    << peerPos->col << ", " << peerPos->row << ")";
              });
  if (unplacedCount > 0)
    diag.attachNote() << unplacedCount
                      << " additional peer(s) on this LTO are not yet placed "
                         "(placed after this LTO; their positions may "
                         "subsequently constrain or unblock this placement)";
}

SequentialPlacer::Adjacency SequentialPlacer::buildObjectFifoAdjacency(
    ArrayRef<ObjectFifoCreateOp> objectFifos) {
  Adjacency adjacency;
  for (auto ofOp : objectFifos)
    for (Value consumer : ofOp.getConsumerTiles())
      adjacency.addEdgeFromValues(ofOp.getProducerTile(), consumer);
  return adjacency;
}

SequentialPlacer::Adjacency
SequentialPlacer::buildFlowAdjacency(ArrayRef<FlowOp> flows,
                                     ArrayRef<PacketFlowOp> pktFlows) {
  Adjacency adjacency;
  for (auto flow : flows)
    adjacency.addEdgeFromValues(flow.getSource(), flow.getDest());

  for (auto pktFlow : pktFlows) {
    SmallVector<Value> srcs, dsts;
    pktFlow.walk([&](Operation *op) {
      if (auto src = dyn_cast<PacketSourceOp>(op))
        srcs.push_back(src.getTile());
      else if (auto dst = dyn_cast<PacketDestOp>(op))
        dsts.push_back(dst.getTile());
    });
    for (auto s : srcs)
      for (auto d : dsts)
        adjacency.addEdgeFromValues(s, d);
  }
  return adjacency;
}

// Already-resolved TileOps have no placer-visible budget, so only flows whose
// endpoints are LogicalTileOps contribute to channelRequirements. A DMA
// channel is a hardware resource: a broadcast (one source channel feeding
// multiple destinations) consumes one MM2S channel on the producer regardless
// of the number of `aie.flow` ops it lowers to, and a merge (multiple sources
// landing on one destination channel) consumes one S2MM channel on the
// consumer. Dedup by (tile, channel) so the producer-of-broadcast and
// consumer-of-merge sides are each counted once.
void SequentialPlacer::addChannelRequirementsFromFlows(
    ArrayRef<FlowOp> flows, ArrayRef<PacketFlowOp> pktFlows,
    llvm::DenseMap<Operation *, std::pair<int, int>> &channelRequirements) {

  llvm::DenseSet<std::tuple<Operation *, int>> seenSrc, seenDst;

  auto incIfDMA = [&](Operation *tileOp, WireBundle bundle, int channel,
                      bool isOutput) {
    if (!tileOp || !isa<LogicalTileOp>(tileOp))
      return;
    if (bundle != WireBundle::DMA)
      return;
    auto &seen = isOutput ? seenSrc : seenDst;
    if (!seen.insert({tileOp, channel}).second)
      return;
    if (isOutput)
      channelRequirements[tileOp].second++;
    else
      channelRequirements[tileOp].first++;
  };

  for (auto flow : flows) {
    incIfDMA(flow.getSource().getDefiningOp(), flow.getSourceBundle(),
             flow.getSourceChannel(), /*isOutput=*/true);
    incIfDMA(flow.getDest().getDefiningOp(), flow.getDestBundle(),
             flow.getDestChannel(), /*isOutput=*/false);
  }

  for (auto pktFlow : pktFlows) {
    pktFlow.walk([&](Operation *op) {
      if (auto src = dyn_cast<PacketSourceOp>(op)) {
        incIfDMA(src.getTile().getDefiningOp(), src.getBundle(),
                 src.getChannel(), /*isOutput=*/true);
      } else if (auto dst = dyn_cast<PacketDestOp>(op)) {
        incIfDMA(dst.getTile().getDefiningOp(), dst.getBundle(),
                 dst.getChannel(), /*isOutput=*/false);
      }
    });
  }
}

int SequentialPlacer::computeCentroidColumn(
    LogicalTileOp logicalTile,
    ArrayRef<const Adjacency *> connectivityAdjacencies) {
  // BFS the connected component, summing placed-core columns. Walks
  // through unplaced LTO peers so the centroid sees cores reachable
  // transitively; TileOp peers terminate the walk. Cores are leaves
  // because they don't relay between non-core peers via core-to-core
  // fifos.
  llvm::DenseSet<Operation *> visited;
  SmallVector<Operation *, 8> queue{logicalTile.getOperation()};
  visited.insert(logicalTile.getOperation());

  int sumCols = 0;
  int placedCoreCount = 0;
  while (!queue.empty()) {
    Operation *current = queue.pop_back_val();
    for (const Adjacency *adj : connectivityAdjacencies) {
      forEachPeer(current, *adj, [&](TileLike peer, bool /*thisIsFirst*/) {
        Operation *peerOp = peer.getOperation();
        if (!peerOp)
          return;
        if (!visited.insert(peerOp).second)
          return;
        auto peerLT = dyn_cast_or_null<LogicalTileOp>(peerOp);
        if (!peerLT)
          return; // TileOp peer: don't BFS through it.
        if (peerLT.getTileType() == AIETileType::CoreTile) {
          if (auto resIt = result.find(peerOp); resIt != result.end()) {
            sumCols += resIt->second.col;
            ++placedCoreCount;
          }
          return;
        }
        queue.push_back(peerOp);
      });
    }
  }
  if (placedCoreCount == 0)
    return 0;
  // Round-to-nearest column: +half before truncating.
  return (sumCols + placedCoreCount / 2) / placedCoreCount;
}

LogicalResult SequentialPlacer::placeNonCoreTileByCentroid(
    LogicalTileOp logicalTile,
    ArrayRef<const Adjacency *> connectivityAdjacencies,
    const llvm::DenseMap<Operation *, std::pair<int, int>>
        &channelRequirements) {
  int centroidCol =
      computeCentroidColumn(logicalTile, connectivityAdjacencies);
  auto colConstraint = logicalTile.tryGetCol();
  int targetCol = colConstraint ? *colConstraint : centroidCol;

  auto [numInputChannels, numOutputChannels] =
      channelRequirements.lookup(logicalTile.getOperation());

  auto maybeTile = findTileWithCapacity(targetCol, availability.nonCompTiles,
                                        numInputChannels, numOutputChannels,
                                        logicalTile.getTileType());
  if (!maybeTile) {
    StringRef tileTypeName =
        stringifyAIETileType(logicalTile.getTileType());
    auto diag = logicalTile.emitError();
    diag << "no " << tileTypeName << " has sufficient DMA capacity for "
         << numInputChannels << " input/" << numOutputChannels
         << " output channels ";
    if (colConstraint)
      diag << "at column " << *colConstraint;
    else
      diag << "near centroid column " << centroidCol;
    diag.attachNote() << "to fix, pin this " << tileTypeName
                      << " to a column with available DMA budget, or rebalance "
                         "compute peers so the centroid lands on a less-busy "
                         "column";
    return failure();
  }

  result[logicalTile] = *maybeTile;
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": placed non-core LTO "
                          << logicalTile.getLoc() << " at (" << maybeTile->col
                          << ", " << maybeTile->row
                          << ") (centroid col=" << centroidCol
                          << ", target col=" << targetCol << ")\n");
  if (!mergeLogicalTiles)
    assignedNonCoreTiles.insert(*maybeTile);
  if (numInputChannels > 0)
    updateChannelUsage(*maybeTile, DmaDir::In, numInputChannels);
  if (numOutputChannels > 0)
    updateChannelUsage(*maybeTile, DmaDir::Out, numOutputChannels);
  return success();
}

std::optional<TileID> SequentialPlacer::findTileWithCapacity(
    int targetCol, llvm::ArrayRef<TileID> tiles, int requiredInputChannels,
    int requiredOutputChannels, AIETileType requestedType) {
  // Choose a physical tile by lexicographic minimum of
  //   (|col - targetCol|, current load, col, row).
  // Distance-from-centroid comes first to preserve routing locality: a tile
  // in the centroid column with any spare capacity beats an idle tile in an
  // adjacent column. The load tiebreaker spreads logical tiles when more
  // than one physical tile sits at the same distance (e.g. two centroid-
  // adjacent columns), avoiding the first-match pile-up that filled a single
  // column's non-core tile until DMA-full before trying anywhere else.
  // (col, row) is the final tiebreaker for determinism.
  std::optional<TileID> best;
  std::tuple<int, int, int, int> bestKey;

  for (auto &tile : tiles) {
    if (targetModel->getTileType(tile.col, tile.row) != requestedType)
      continue;
    if (!hasAvailableChannels(tile, requiredInputChannels,
                              requiredOutputChannels))
      continue;
    // When merge-logical-tiles is disabled, a tile that already hosts a
    // non-core aie.logical_tile is off-limits even if it has spare DMA
    // capacity.
    if (!mergeLogicalTiles && assignedNonCoreTiles.contains(tile))
      continue;
    int dist = std::abs(tile.col - targetCol);
    int load = availability.inputChannelsUsed[tile] +
               availability.outputChannelsUsed[tile];
    std::tuple<int, int, int, int> key{dist, load, tile.col, tile.row};
    if (!best || key < bestKey) {
      best = tile;
      bestKey = key;
    }
  }

  return best;
}

void SequentialPlacer::updateChannelUsage(TileID tile, DmaDir direction,
                                          int numChannels) {
  if (direction == DmaDir::Out)
    availability.outputChannelsUsed[tile] += numChannels;
  else
    availability.inputChannelsUsed[tile] += numChannels;

  if (!hasAvailableChannels(tile, 0, 0)) {
    AIETileType type = targetModel->getTileType(tile.col, tile.row);
    availability.removeTile(tile, type);
  }
}

bool SequentialPlacer::hasAvailableChannels(TileID tile, int inputChannels,
                                            int outputChannels) {
  // Get max channels based on tile type and row
  int maxIn, maxOut;
  if (tile.row == 0) {
    // Shim tiles use ShimMux connections
    maxIn = targetModel->getNumDestShimMuxConnections(tile.col, tile.row,
                                                      WireBundle::DMA);
    maxOut = targetModel->getNumSourceShimMuxConnections(tile.col, tile.row,
                                                         WireBundle::DMA);
  } else {
    // Other tiles use Switchbox connections
    maxIn = targetModel->getNumDestSwitchboxConnections(tile.col, tile.row,
                                                        WireBundle::DMA);
    maxOut = targetModel->getNumSourceSwitchboxConnections(tile.col, tile.row,
                                                           WireBundle::DMA);
  }

  int currentIn = availability.inputChannelsUsed[tile];
  int currentOut = availability.outputChannelsUsed[tile];

  return (currentIn + inputChannels <= maxIn) &&
         (currentOut + outputChannels <= maxOut);
}

void TileAvailability::removeTile(TileID tile, AIETileType type) {
  auto removeFromVector = [&](std::vector<TileID> &vec) {
    vec.erase(std::remove(vec.begin(), vec.end(), tile), vec.end());
  };

  switch (type) {
  case AIETileType::CoreTile:
    removeFromVector(compTiles);
    compTilesSet.erase(tile);
    break;
  case AIETileType::MemTile:
  case AIETileType::ShimNOCTile:
  case AIETileType::ShimPLTile:
    removeFromVector(nonCompTiles);
    break;
  default:
    break;
  }
}
