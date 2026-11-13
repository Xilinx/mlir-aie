//===- AIEPlacer.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIE/Transforms/AIEChannelTurns.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEStructuralRank.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Utils/DmaDecomposition.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <limits>
#include <numeric>

using namespace mlir;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-placer"

static std::optional<TileID> resolvePeerPosition(TileLike peer,
                                                 const PlacementResult &placed);

void Placer::Adjacency::addEdge(TileLike first, TileLike second) {
  if (!first || !second)
    return;
  unsigned idx = edges.size();
  edges.push_back({first, second});
  if (mlir::isa<LogicalTileOp>(first.getOperation()))
    tileToEdges[first.getOperation()].push_back(idx);
  if (mlir::isa<LogicalTileOp>(second.getOperation()))
    tileToEdges[second.getOperation()].push_back(idx);
}

void Placer::Adjacency::addEdgeFromValues(Value a, Value b) {
  if (!a || !b)
    return;
  auto aT = dyn_cast_or_null<TileLike>(a.getDefiningOp());
  auto bT = dyn_cast_or_null<TileLike>(b.getDefiningOp());
  if (aT && bT)
    addEdge(aT, bT);
}

void Placer::initialize(DeviceOp device, bool controlOverlay,
                        bool adaptiveOverlay, bool packetSwitched) {
  const AIETargetModel &tm = device.getTargetModel();
  targetModel = &tm;
  this->controlOverlay = controlOverlay;
  this->adaptiveOverlay = adaptiveOverlay;
  this->packetSwitched = packetSwitched;
  availability.physicalCores.clear();
  for (CoreOp core : device.getOps<CoreOp>())
    if (auto tile = dyn_cast_or_null<TileOp>(core.getTile().getDefiningOp()))
      availability.physicalCores.insert(tile.getTileID());
  availability.compTiles.clear();
  availability.overBudgetCompTiles.clear();
  availability.nonCompTiles.clear();
  for (int col = 0; col < tm.columns(); col++) {
    for (int row = 0; row < tm.rows(); row++) {
      TileID id = {col, row};
      switch (tm.getTileType(col, row)) {
      case AIETileType::CoreTile:
        if (!availability.physicalCores.contains(id))
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
}

Placer::CollectedOps Placer::collectOperations(DeviceOp device) {
  CollectedOps ops;
  device.walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<LogicalTileOp>([&](auto lt) { ops.logicalTiles.push_back(lt); })
        .Case<ObjectFifoCreateOp>(
            [&](auto of) { ops.objectFifos.push_back(of); })
        .Case<ObjectFifoLinkOp>(
            [&](auto link) { ops.objectFifoLinks.push_back(link); })
        .Case<ObjectFifoAllocateOp>(
            [&](auto alloc) { ops.objectFifoAllocates.push_back(alloc); })
        .Case<CascadeFlowOp>([&](auto cf) { ops.cascadeFlows.push_back(cf); })
        .Case<FlowOp>([&](auto f) { ops.flows.push_back(f); })
        .Case<PacketFlowOp>([&](auto pf) { ops.pktFlows.push_back(pf); })
        .Case<RouteOp>([&](auto route) { ops.routes.push_back(route); });
  });
  // Every later tie between logical tiles falls back on this order, so it is
  // the design's structure rather than the order the tiles were written in:
  // shim tiles, which have the fewest places to go, then MemTiles, then cores.
  DenseMap<Operation *, unsigned> rank = StructuralGraph::of(device).rank();
  llvm::stable_sort(ops.logicalTiles, [&](LogicalTileOp a, LogicalTileOp b) {
    if (a.getTileType() != b.getTileType())
      return a.getTileType() > b.getTileType();
    return rank.lookup(a) < rank.lookup(b);
  });
  return ops;
}

bool Placer::satisfiesConstraints(Operation *tile, TileID pos) {
  if (auto lt = dyn_cast<LogicalTileOp>(tile)) {
    if (auto c = lt.tryGetCol(); c && *c != pos.col)
      return false;
    if (auto r = lt.tryGetRow(); r && *r != pos.row)
      return false;
  }
  return true;
}

void SequentialPlacer::initialize(DeviceOp device, bool controlOverlay,
                                  bool adaptiveOverlay, bool packetSwitched) {
  Placer::initialize(device, controlOverlay, adaptiveOverlay, packetSwitched);

  // Compute tiles iterate column-major (fill a column top-to-bottom before
  // moving right); non-compute tiles iterate row-major.
  auto compTileCmp = [](TileID a, TileID b) {
    if (a.col != b.col)
      return a.col < b.col;
    return a.row < b.row;
  };
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

  if (coresPerCol.has_value()) {
    deviceCoresPerCol = 0;
    for (int col = 0; col < targetModel->columns(); col++) {
      int cores = 0;
      for (int row = 0; row < targetModel->rows(); row++)
        cores += targetModel->getTileType(col, row) == AIETileType::CoreTile;
      deviceCoresPerCol = std::max(deviceCoresPerCol, cores);
    }
    DenseSet<TileID> held(availability.physicalCores.begin(),
                          availability.physicalCores.end());
    for (CoreOp core : device.getOps<CoreOp>())
      if (auto tile =
              dyn_cast_or_null<LogicalTileOp>(core.getTile().getDefiningOp()))
        if (auto col = tile.tryGetCol(), row = tile.tryGetRow(); col && row)
          held.insert({*col, *row});
    limitCoresPerColumn(*coresPerCol, targetModel->columns(), held);
  }
}

void SequentialPlacer::limitCoresPerColumn(int maxCoresPerCol, int numColumns,
                                           const DenseSet<TileID> &held) {
  llvm::DenseMap<int, std::vector<TileID>> tilesByColumn;
  for (const auto &tile : availability.compTiles)
    if (!held.contains(tile))
      tilesByColumn[tile.col].push_back(tile);

  // Keep the first maxCoresPerCol tiles from each column, less the column's
  // held cores. The per-column vectors are already row-sorted because
  // compTiles is column-major.
  DenseMap<int, int> heldInColumn;
  for (TileID tile : held)
    ++heldInColumn[tile.col];
  std::vector<TileID> limitedTiles;
  for (int col = 0; col < numColumns; col++) {
    auto it = tilesByColumn.find(col);
    if (it == tilesByColumn.end())
      continue;
    const auto &tilesInCol = it->second;
    size_t numToTake = std::min<size_t>(
        tilesInCol.size(),
        std::max(0, maxCoresPerCol - heldInColumn.lookup(col)));
    limitedTiles.insert(limitedTiles.end(), tilesInCol.begin(),
                        tilesInCol.begin() + numToTake);
    availability.overBudgetCompTiles.insert(
        availability.overBudgetCompTiles.end(), tilesInCol.begin() + numToTake,
        tilesInCol.end());
  }

  placeableCores = limitedTiles.size();
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
  auto collected = collectOperations(device);
  auto &logicalTiles = collected.logicalTiles;
  auto &objectFifos = collected.objectFifos;
  auto &objectFifoLinks = collected.objectFifoLinks;
  auto &cascadeFlows = collected.cascadeFlows;
  auto &flows = collected.flows;
  auto &pktFlows = collected.pktFlows;
  auto &routes = collected.routes;

  // Phase 2a: Build placement constraints
  auto bufferAdjacency =
      buildBufferAdjacency(logicalTiles, collected.objectFifoAllocates);
  auto computePeerAdjacency = buildComputePeerAdjacency(objectFifos);
  auto sharedMemAdjacency = buildSharedMemAdjacency(objectFifos);

  // Phase 2b: Build channel requirements from ObjectFifo and Flow connectivity,
  // and cascade adjacency constraints from CascadeFlow connectivity.
  auto channelRequirements =
      buildChannelRequirements(objectFifos, objectFifoLinks);
  if (pinnedRule.failed())
    return failure();
  namedChannels.clear();
  llvm::DenseMap<Operation *, Operation *> demandKeys;
  DMAChannelAnalysis channelAnalysis(device);
  claimsOf.clear();
  for (const DMAChannelAnalysis::ChannelClaim &claim :
       channelAnalysis.getChannelClaims())
    claimsOf[claim.tile.getDefiningOp()].push_back(claim);
  for (auto &[tile, inOut] : flowChannelDemand(
           flows, pktFlows, routes, channelAnalysis.getChannelClaims(),
           &namedChannels, &demandKeys)) {
    channelRequirements[tile].first += inOut.first;
    channelRequirements[tile].second += inOut.second;
  }
  countTurns(device, objectFifos, channelRequirements);
  buildStreamDemand(objectFifos);
  runtimeBdIds = RuntimeBdIds(device, *targetModel);
  shareableShimEnds = collectShareableShimEnds(device, objectFifos, demandKeys);
  shimSpans.emplace(
      device, [&](StringAttr symbol) { return shimEndOf(device, symbol); },
      provenShimGates(device), controlOverlay);
  shimTilesAt.clear();
  overlayTilesAt.clear();
  collectOverlayUses(device);
  // Seed each physical tile op's position with the ends, flows and claims it
  // already holds.
  for (TileOp placed : device.getOps<TileOp>()) {
    TileID at = placed.getTileID();
    auto [in, out] = channelRequirements.lookup(placed);
    if (in)
      updateChannelUsage(at, DmaDir::In, in);
    if (out)
      updateChannelUsage(at, DmaDir::Out, out);
    overlayTilesAt[at].push_back(placed);
    for (const PinnedChannels::End &end : pinnedEnds.lookup(placed))
      pinsAt.place(at, end, 1);
    for (const DMAChannelAnalysis::ChannelClaim &claim :
         claimsOf.lookup(placed))
      pinsAt.claim(at, claim, 1);
    recordShimEnds(at, placed);
  }
  // Other devices of the overlay's target count where they sit.
  for (Operation *tile : overlayTileOps())
    if (auto placed = dyn_cast<TileOp>(tile);
        placed && placed->getParentOfType<DeviceOp>() != device)
      overlayTilesAt[placed.getTileID()].push_back(placed);

  auto cascadeAdjacency = buildCascadeAdjacency(cascadeFlows);

  // A forced-shared-mem peer is already in `computePeerAdjacency`; this
  // tallies which of a tile's peers are forced, to floor its neighbor
  // demand below regardless of DMA channel headroom.
  llvm::DenseMap<Operation *, int> forcedSharedMemIn, forcedSharedMemOut;
  for (auto ofOp : objectFifos) {
    if (!ofOp.forcesSharedMemory() || ofOp.getConsumerTiles().size() != 1)
      continue;
    auto producer =
        dyn_cast_or_null<LogicalTileOp>(ofOp.getProducerTile().getDefiningOp());
    auto consumer = dyn_cast_or_null<LogicalTileOp>(
        ofOp.getConsumerTiles()[0].getDefiningOp());
    if (!producer || producer.getTileType() != AIETileType::CoreTile)
      continue;
    if (!consumer || consumer.getTileType() != AIETileType::CoreTile)
      continue;
    if (producer.getOperation() == consumer.getOperation())
      continue;
    ++forcedSharedMemOut[producer.getOperation()];
    ++forcedSharedMemIn[consumer.getOperation()];
  }

  // Phase 2c: For every compute LTO, compute the minimum number of its
  // compute-peer producers/consumers that must land on physical neighbors so
  // the LTO fits its DMA channel budget. A compute→compute ObjectFifo can use
  // shared neighbor memory (no DMA) when the two endpoints sit on physical-
  // neighbor compute tiles; otherwise it consumes one DMA channel on each
  // side. Non-compute-peer fifos (from/to memtile or shim) always need a DMA
  // channel and already appear in `channelRequirements`, so the headroom left
  // for compute-peer fifos is (DMA-budget − channelRequirements). If a
  // Worker has more compute-peer fifos than that headroom, the surplus MUST
  // come from neighbors -- and a peer forcing shared memory always must,
  // budget or not.
  llvm::DenseMap<Operation *, int> needNeighborIn, needNeighborOut;
  for (auto lt : logicalTiles) {
    if (lt.getTileType() != AIETileType::CoreTile)
      continue;
    auto [peerIn, peerOut] =
        totalComputePeers(lt.getOperation(), computePeerAdjacency);
    auto [inBudget, outBudget] =
        getDMACapacity(*targetModel, lt.getCanonicalTileID());
    auto [nonPeerIn, nonPeerOut] =
        channelRequirements.lookup(lt.getOperation());
    needNeighborIn[lt.getOperation()] =
        std::max({0, peerIn - (inBudget - nonPeerIn),
                  forcedSharedMemIn.lookup(lt.getOperation())});
    needNeighborOut[lt.getOperation()] =
        std::max({0, peerOut - (outBudget - nonPeerOut),
                  forcedSharedMemOut.lookup(lt.getOperation())});
    LLVM_DEBUG(llvm::dbgs()
               << DEBUG_TYPE << ": LTO " << lt.getLoc() << " peerIn=" << peerIn
               << " peerOut=" << peerOut << " nonPeerIn=" << nonPeerIn
               << " nonPeerOut=" << nonPeerOut
               << " => needNeighborIn=" << needNeighborIn[lt.getOperation()]
               << " needNeighborOut=" << needNeighborOut[lt.getOperation()]
               << "\n");
  }
  PlacementContext ctx{*targetModel, computePeerAdjacency, needNeighborIn,
                       needNeighborOut};

  // Phase 2d: group cores by the non-core tile they share the most flows
  // with, so each group can be placed in one column and its hub's centroid
  // lands there too.
  FlowMembership flowIndex =
      buildFlowMembership(flows, pktFlows, routes, objectFifos);
  CoreGroups coreGroups = buildCoreGroups(logicalTiles, flowIndex);

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
    return targetModel->isLegalCascade(srcPos.col, srcPos.row, dstPos.col,
                                       dstPos.row);
  };
  auto cascadeLabel = [](bool thisIsSrc) -> StringRef {
    return thisIsSrc ? "cascade destination" : "cascade source";
  };
  // Shared memory: unlike buffer ownership, either end satisfying
  // isLegalMemAffinity to the other is enough -- a shared_mem objectFifo
  // has no owner/consumer asymmetry, it just needs both ends on tiles that
  // share a memory module. Same bidirectional check as SAPlacer's own
  // shared-memory penalty.
  auto sharedMemPred = [this](TileID aPos, TileID bPos) {
    return targetModel->isLegalMemAffinity(aPos.col, aPos.row, bPos.col,
                                           bPos.row) ||
           targetModel->isLegalMemAffinity(bPos.col, bPos.row, aPos.col,
                                           aPos.row);
  };
  auto sharedMemLabel = [](bool thisIsProducer) -> StringRef {
    return thisIsProducer ? "shared-memory consumer" : "shared-memory producer";
  };

  AdjacencyKind bufferKind{
      bufferAdjacency, bufferPred, bufferLabel, "shared-L1 buffer",
      "shared-L1 buffer adjacency requires this LTO to be on a tile "
      "whose L1 is shared with the buffer owner's tile (N/S neighbor, "
      "or W neighbor per the device's checkerboard rule)"};
  AdjacencyKind cascadeKind{
      cascadeAdjacency, cascadePred, cascadeLabel, "cascade",
      "cascade adjacency requires the destination tile to be one row "
      "South or one column East of the source tile"};
  AdjacencyKind sharedMemKind{
      sharedMemAdjacency, sharedMemPred, sharedMemLabel,
      "shared-memory transport",
      "a `shared_mem` objectFifo requires its producer and consumer to "
      "land on tiles that share a memory module (N/S neighbor, or W "
      "neighbor per the device's checkerboard rule)"};

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
  //
  // Before sorting, each core group is made contiguous at its first member,
  // so the order in which the program creates cores does not decide which
  // of them share a column.
  SmallVector<LogicalTileOp> orderedTiles;
  llvm::DenseSet<unsigned> emittedGroups;
  for (auto lt : logicalTiles) {
    auto group = coreGroups.groupOf.find(lt.getOperation());
    if (group == coreGroups.groupOf.end())
      orderedTiles.push_back(lt);
    else if (emittedGroups.insert(group->second).second)
      llvm::append_range(orderedTiles, coreGroups.members[group->second]);
  }
  llvm::stable_sort(orderedTiles, [&](LogicalTileOp a, LogicalTileOp b) {
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

  // An unpinned tile sharing L1 with another unpinned tile is placed right
  // after it, while that tile's neighbors are still free.
  auto unpinned = [](LogicalTileOp lt) {
    return !lt.tryGetCol() && !lt.tryGetRow();
  };
  SmallVector<LogicalTileOp> sequenced;
  llvm::DenseSet<Operation *> queued;
  for (auto start : orderedTiles) {
    if (!queued.insert(start.getOperation()).second)
      continue;
    size_t next = sequenced.size();
    sequenced.push_back(start);
    for (; next < sequenced.size(); ++next) {
      LogicalTileOp lt = sequenced[next];
      if (!unpinned(lt))
        continue;
      forEachPeer(lt.getOperation(), bufferAdjacency,
                  [&](TileLike peer, bool /*thisIsFirst*/) {
                    auto peerLt = dyn_cast<LogicalTileOp>(peer.getOperation());
                    if (peerLt && unpinned(peerLt) &&
                        queued.insert(peerLt.getOperation()).second)
                      sequenced.push_back(peerLt);
                  });
    }
  }
  orderedTiles = std::move(sequenced);

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
      if (failed(enforceAdjacency(logicalTile, tile, sharedMemKind)))
        return failure();
      if (failed(validateAndUpdateChannelUsage(
              logicalTile, tile, channelRequirements, PlacementOrigin::Pinned)))
        return failure();

      recordPlacement(logicalTile, tile, ctx, "pinned");
      continue;
    }

    // Place compute tiles with partial constraint support
    if (logicalTile.getTileType() == AIETileType::CoreTile) {
      auto isReservedForOtherBound = [&](Operation *lto, TileID candidate) {
        return ctx.isReservedForOther(lto, candidate);
      };
      SmallVector<int> preferredCols =
          preferredGroupColumns(logicalTile, coreGroups);
      UnpinnedPlacementInputs inputs{
          bufferAdjacency,      bufferPred,
          cascadeAdjacency,     cascadePred,
          sharedMemAdjacency,   sharedMemPred,
          computePeerAdjacency, needNeighborIn,
          needNeighborOut,      isReservedForOtherBound,
          preferredCols};
      // A pin outranks the budget, so a pinned core searches the tiles it
      // left out as well.
      bool searchedPastBudget = col || row;
      auto search = findUnconstrainedCoreCandidate(
          logicalTile, col, row, inputs, availability.compTiles,
          searchedPastBudget
              ? ArrayRef<TileID>(availability.overBudgetCompTiles)
              : ArrayRef<TileID>());
      std::optional<TileID> placement = search.placement;
      bool sawConstraintMatch = search.sawConstraintMatch;
      bool allConstraintMatchesFailedAdjacency =
          search.allConstraintMatchesFailedAdjacency;
      bool computePeerWasCause = search.computePeerWasCause;

      if (!placement) {
        bool adjacencyWasCause =
            sawConstraintMatch && allConstraintMatchesFailedAdjacency &&
            bufferAdjacency.hasEdges(logicalTile.getOperation());
        bool hasCascade = cascadeAdjacency.hasEdges(logicalTile.getOperation());
        bool hasSharedMem =
            sharedMemAdjacency.hasEdges(logicalTile.getOperation());
        InFlightDiagnostic diag = logicalTile.emitError();
        if (col || row) {
          diag << "no compute tile available matching constraint ("
               << (col ? std::to_string(*col) : "?") << ", "
               << (row ? std::to_string(*row) : "?") << ")"
               << (adjacencyWasCause ? " and shared-L1 buffer adjacency" : "")
               << (hasCascade ? " and cascade adjacency" : "")
               << (hasSharedMem ? " and shared-memory transport" : "")
               << (computePeerWasCause ? " and compute-peer DMA budget" : "");
        } else {
          diag << "no available compute tiles for placement"
               << (adjacencyWasCause
                       ? " (shared-L1 buffer adjacency unsatisfiable)"
                       : "")
               << (hasCascade ? " (cascade adjacency unsatisfiable)" : "")
               << (hasSharedMem ? " (shared-memory transport unsatisfiable)"
                                : "")
               << (computePeerWasCause
                       ? " (compute-peer DMA budget unsatisfiable)"
                       : "");
        }
        // A cores-per-col budget leaves no trace in the message above: the
        // tiles it removed are simply absent from `availability.compTiles`.
        if (coresPerCol.has_value() && deviceCoresPerCol > 0 &&
            !searchedPastBudget) {
          int columns = targetModel->columns();
          diag.attachNote()
              << "cores-per-col=" << *coresPerCol << " leaves "
              << placeableCores << " of this device's "
              << columns * deviceCoresPerCol
              << " compute tiles placeable and may be limiting placement; "
                 "consider raising it or reducing the design's core count";
        }
        if (search.pinClash)
          notePinClash(diag, logicalTile, *search.pinClash);
        if (adjacencyWasCause)
          attachPeerNotes(diag, logicalTile, bufferAdjacency, bufferLabel);
        if (hasCascade)
          attachPeerNotes(diag, logicalTile, cascadeAdjacency, cascadeLabel);
        if (hasSharedMem)
          attachPeerNotes(diag, logicalTile, sharedMemAdjacency,
                          sharedMemLabel);
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
        if (adjacencyWasCause || hasCascade || hasSharedMem ||
            computePeerWasCause)
          diag.attachNote()
              << "the sequential placer places tiles one at a time and never "
                 "moves a placed peer; if these constraints can be met "
                 "together, --placer=sa_placer, which searches placements "
                 "jointly, may place them";
        return failure();
      }

      if (failed(validateAndUpdateChannelUsage(logicalTile, *placement,
                                               channelRequirements,
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
  return placeNonCoreLogicalTiles(logicalTiles, flowIndex, channelRequirements);
}

SequentialPlacer::CoreGroups
SequentialPlacer::buildCoreGroups(ArrayRef<LogicalTileOp> logicalTiles,
                                  const FlowMembership &flowIndex) const {
  // Flows from each core to each non-core tile, in first-seen order.
  SmallVector<std::pair<LogicalTileOp, llvm::MapVector<Operation *, int>>>
      coreFlows;
  llvm::DenseMap<Operation *, int> coresOnHub;
  for (auto lt : logicalTiles) {
    if (lt.getTileType() != AIETileType::CoreTile)
      continue;
    llvm::MapVector<Operation *, int> counts;
    auto it = flowIndex.ltoFlows.find(lt.getResult());
    if (it != flowIndex.ltoFlows.end())
      for (auto &peers : it->second)
        for (Value p : peers) {
          auto tile = dyn_cast_or_null<TileLike>(p.getDefiningOp());
          if (tile && !tile.isCoreTile())
            ++counts[p.getDefiningOp()];
        }
    for (auto &entry : counts)
      ++coresOnHub[entry.first];
    coreFlows.emplace_back(lt, std::move(counts));
  }

  // A core's hub is the tile it has the most flows with; on a tie, the one
  // fewer cores touch (the more specific one), then the first seen.
  CoreGroups groups;
  llvm::DenseMap<Operation *, unsigned> groupOfHub;
  for (auto &[lt, counts] : coreFlows) {
    Operation *hub = nullptr;
    int best = 0;
    for (auto &[peer, n] : counts)
      if (!hub || n > best ||
          (n == best && coresOnHub[peer] < coresOnHub[hub])) {
        hub = peer;
        best = n;
      }
    unsigned idx = groups.members.size();
    if (hub) {
      auto [it, inserted] = groupOfHub.try_emplace(hub, idx);
      idx = it->second;
      if (!inserted) {
        groups.members[idx].push_back(lt);
        groups.groupOf[lt.getOperation()] = idx;
        continue;
      }
    }
    groups.members.push_back({lt});
    groups.groupOf[lt.getOperation()] = idx;
  }
  return groups;
}

SmallVector<int>
SequentialPlacer::preferredGroupColumns(LogicalTileOp logicalTile,
                                        const CoreGroups &groups) const {
  auto group = groups.groupOf.find(logicalTile.getOperation());
  if (group == groups.groupOf.end())
    return {};
  SmallVector<std::pair<int, int>> placedCols; // (column, members there)
  int unplaced = 0;
  for (auto member : groups.members[group->second]) {
    auto placed = result.find(member.getOperation());
    if (placed == result.end()) {
      ++unplaced;
      continue;
    }
    auto *entry = llvm::find_if(
        placedCols, [&](auto &e) { return e.first == placed->second.col; });
    if (entry == placedCols.end())
      placedCols.push_back({placed->second.col, 1});
    else
      ++entry->second;
  }
  llvm::stable_sort(placedCols, [](auto &a, auto &b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });

  SmallVector<int> cols;
  for (auto &[col, n] : placedCols)
    cols.push_back(col);
  // compTiles is column-major, so this visits columns in order.
  llvm::MapVector<int, int> freeCores;
  for (TileID t : availability.compTiles)
    ++freeCores[t.col];
  for (auto &[col, n] : freeCores)
    if (n >= unplaced && !llvm::is_contained(cols, col))
      cols.push_back(col);
  return cols;
}

LogicalResult SequentialPlacer::placeNonCoreLogicalTiles(
    ArrayRef<LogicalTileOp> logicalTiles, const FlowMembership &flowIndex,
    const llvm::DenseMap<Operation *, std::pair<int, int>>
        &channelRequirements) {
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
  llvm::stable_sort(nonCoreOrdered, [&](LogicalTileOp a, LogicalTileOp b) {
    auto demand = [&](LogicalTileOp lt) {
      auto chans = channelRequirements.lookup(lt.getOperation());
      return chans.first + chans.second;
    };
    return demand(a) > demand(b);
  });

  for (auto logicalTile : nonCoreOrdered) {
    if (failed(placeNonCoreTileByCentroid(logicalTile, flowIndex,
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
  forEachMemAffinityNeighbor(
      targetModel, at, [&](TileID nb) { reservedFor[nb].push_back(placedOp); });
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
    const UnpinnedPlacementInputs &inputs, ArrayRef<TileID> candidates,
    ArrayRef<TileID> pastBudget) {
  UnpinnedSearchResult result;
  auto neighborDemand = [&](Operation *op) {
    return inputs.needNeighborIn.lookup(op) + inputs.needNeighborOut.lookup(op);
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
  SmallVector<TileID> orderedCandidates(candidates);
  if (neighborDemand(logicalTile.getOperation()) > 0) {
    llvm::stable_sort(orderedCandidates, [&](TileID a, TileID b) {
      if (a.col != b.col)
        return a.col < b.col;
      int na = computeNeighborCount(a);
      int nb = computeNeighborCount(b);
      if (na != nb)
        return na > nb;
      return a.row < b.row;
    });
  } else if (!inputs.preferredCols.empty()) {
    // Keep the LTO with its core group: preferred columns first, in
    // preference order; the rest keep the column-major order after them.
    auto rankOf = [&](TileID t) {
      return std::distance(inputs.preferredCols.begin(),
                           llvm::find(inputs.preferredCols, t.col));
    };
    llvm::stable_sort(orderedCandidates, [&](TileID a, TileID b) {
      return rankOf(a) < rankOf(b);
    });
  }

  // Tiles past the budget come after it, so a pinned core exceeds the budget
  // only when no tile within it passes a pass's checks; the strict passes see
  // them before the last pass takes a tile a floating peer needs.
  llvm::append_range(orderedCandidates, pastBudget);

  // A candidate that leaves no free core tile for one of this LTO's
  // unplaced shared-L1 peers is tried only in the final pass. Each peer
  // claims its own tile; claiming greedily is exact when the peers share one
  // candidate set, as one owner's consumers do.
  auto leavesPeerRoom = [&](TileID candidate) {
    llvm::DenseSet<TileID> claimed;
    bool room = true;
    forEachPeer(logicalTile.getOperation(), inputs.bufferAdjacency,
                [&](TileLike peer, bool thisIsFirst) {
                  if (!room || resolvePeerPosition(peer, this->result))
                    return;
                  // thisIsFirst: this tile is the consumer, peer the owner.
                  auto free =
                      llvm::find_if(availability.compTiles, [&](TileID t) {
                        return t != candidate && !claimed.contains(t) &&
                               (thisIsFirst ? inputs.bufferPred(candidate, t)
                                            : inputs.bufferPred(t, candidate));
                      });
                  room = free != availability.compTiles.end();
                  if (room)
                    claimed.insert(*free);
                });
    return room;
  };

  auto fitsColumnStreams = [&](TileID candidate) {
    StreamDemand demand = streamDemand.lookup(logicalTile);
    StreamDemand used = columnStreamsUsed.lookup(candidate.col);
    auto incoming = columnIncomingFifos.find(candidate.col);
    if (incoming != columnIncomingFifos.end())
      for (Operation *fifo : incomingFifos.lookup(logicalTile))
        demand.southIn -= incoming->second.contains(fifo);
    // Measure the bottleneck at the bottom compute row, not at a candidate
    // in the middle of the column. Overflow can still route via other columns.
    int bottomRow = candidate.row;
    while (bottomRow > 0 &&
           targetModel->isCoreTile(candidate.col, bottomRow - 1))
      --bottomRow;
    return (!demand.southIn ||
            used.southIn + demand.southIn <=
                static_cast<int>(targetModel->getNumSourceSwitchboxConnections(
                    candidate.col, bottomRow, WireBundle::South))) &&
           (!demand.southOut ||
            used.southOut + demand.southOut <=
                static_cast<int>(targetModel->getNumDestSwitchboxConnections(
                    candidate.col, bottomRow, WireBundle::South)));
  };

  // First prefer columns with stream capacity, then allow overloaded columns
  // while preserving peer room and reservations. Relax those only as a last
  // resort.
  for (int pass = 0; pass < 3; ++pass) {
    for (const TileID &candidate : orderedCandidates) {
      if (col && candidate.col != *col)
        continue;
      if (row && candidate.row != *row)
        continue;
      if (pass < 2 &&
          (inputs.isReservedForOther(logicalTile.getOperation(), candidate) ||
           !leavesPeerRoom(candidate)))
        continue;
      if (pass == 0 && !fitsColumnStreams(candidate))
        continue;
      result.sawConstraintMatch = true;
      if (!satisfiesAdjacency(logicalTile, candidate, inputs.bufferAdjacency,
                              inputs.bufferPred))
        continue;
      result.allConstraintMatchesFailedAdjacency = false;
      if (!satisfiesAdjacency(logicalTile, candidate, inputs.cascadeAdjacency,
                              inputs.cascadePred))
        continue;
      if (!satisfiesAdjacency(logicalTile, candidate, inputs.sharedMemAdjacency,
                              inputs.sharedMemPred))
        continue;
      if (!satisfiesComputePeer(
              logicalTile, candidate, inputs.computePeerAdjacency,
              inputs.needNeighborIn, inputs.needNeighborOut)) {
        result.computePeerWasCause = true;
        continue;
      }
      if (!pinsAt.heldAt(candidate,
                         pinnedEnds.lookup(logicalTile.getOperation()))) {
        if (!result.pinClash)
          result.pinClash = candidate;
        continue;
      }
      result.placement = candidate;
      return result;
    }
  }
  return result;
}

void SequentialPlacer::notePinClash(InFlightDiagnostic &diag,
                                    LogicalTileOp logicalTile, TileID at) {
  for (PinnedChannels::End end :
       pinnedEnds.lookup(logicalTile.getOperation())) {
    std::optional<PinnedChannels::End> other = pinsAt.clashWith(at, end);
    std::optional<DMAChannelAnalysis::ChannelClaim> claim =
        other ? std::nullopt : pinsAt.claimWith(at, end);
    if (!other && !claim)
      continue;
    auto &note = diag.attachNote();
    note << "on (" << at.col << ", " << at.row
         << "), the first compute tile it fits otherwise, @"
         << end.fifo.getSymName() << " pins " << (end.sends ? "MM2S" : "S2MM")
         << " channel " << end.channel;
    if (other) {
      std::optional<std::string> why = pinnedRule.whyApart(end, *other);
      note << ", which @" << other->fifo.getSymName() << " pins there too, but "
           << (why ? *why : "they clash");
    } else if (claim) {
      note << ", which another op holds";
      diag.attachNote(claim->owner->getLoc()) << "held here";
    }
    return;
  }
}

LogicalResult SequentialPlacer::validateAndUpdateChannelUsage(
    LogicalTileOp logicalTile, TileID tile,
    const llvm::DenseMap<Operation *, std::pair<int, int>> &channelRequirements,
    PlacementOrigin origin) {
  auto [inChannels, outChannels] =
      channelRequirements.lookup(logicalTile.getOperation());
  SmallVector<PinnedChannels::End> pins =
      pinnedEnds.lookup(logicalTile.getOperation());
  std::optional<std::pair<int, int>> held = pinsAt.heldAt(tile, pins);
  if (!held) {
    for (PinnedChannels::End end : pins) {
      if (std::optional<PinnedChannels::End> other =
              pinsAt.clashWith(tile, end)) {
        std::optional<std::string> why = pinnedRule.whyApart(end, *other);
        return end.fifo.emitError()
               << "pins " << (end.sends ? "MM2S" : "S2MM") << " channel "
               << end.channel << " of tile (" << tile.col << ", " << tile.row
               << "), which @" << other->fifo.getSymName()
               << " pins there too, but " << (why ? *why : "they clash");
      }
      if (std::optional<DMAChannelAnalysis::ChannelClaim> claim =
              pinsAt.claimWith(tile, end)) {
        auto diag = end.fifo.emitError()
                    << "pins " << (end.sends ? "MM2S" : "S2MM") << " channel "
                    << end.channel << " of tile (" << tile.col << ", "
                    << tile.row << "), which another op holds";
        diag.attachNote(claim->owner->getLoc()) << "held here";
        return diag;
      }
    }
    llvm_unreachable(
        "heldAt reports only a clash clashWith or claimWith finds");
  }
  inChannels -= held->first;
  outChannels -= held->second;

  if (!hasAvailableChannels(tile, inChannels, outChannels, logicalTile) &&
      !fitsSharingShim(tile, logicalTile, {inChannels, outChannels})) {
    // The same capacity as hasAvailableChannels, so the message matches it.
    auto [maxIn, maxOut] = getDMACapacity(*targetModel, tile);
    int availIn = maxIn - availability.inputChannelsUsed[tile];
    int availOut = maxOut - availability.outputChannelsUsed[tile] -
                   overlayWith(tile, logicalTile).channels;

    auto diag = logicalTile.emitError();
    diag << "tile (" << tile.col << ", " << tile.row << ") requires "
         << inChannels << " input/" << outChannels
         << " output DMA channels, but only " << availIn << " input/"
         << availOut << " output available";
    if (OverlayFit overlay = overlayWith(tile, logicalTile);
        overlay.channels || !overlay.fits)
      diag.attachNote()
          << "the control overlay "
          << (overlay.fits ? "takes one of its output channels"
                           : "has no output channel here it may take or "
                             "share")
          << "; aie-generate-column-control-overlay shares one only with "
             "ends never in flight with its control packets";
    if (origin == PlacementOrigin::Selected)
      diag.attachNote() << "placer selected this tile; to fix, pin this LTO "
                           "to a tile with more spare DMA capacity, or reduce "
                           "the LTO's DMA fanin (e.g. via memtile staging)";
    return failure();
  }

  // Its pins and named channels first: they decide where the overlay sits.
  recordStreamAndBdUsage(logicalTile, tile);
  if (inChannels > 0)
    updateChannelUsage(tile, DmaDir::In, inChannels);
  if (outChannels > 0)
    updateChannelUsage(tile, DmaDir::Out, outChannels);
  recordShimEnds(tile, logicalTile);

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

  // A link's tile is the one producing its dest fifos (each fifo has one
  // producer). The link loop below credits it with (#sources input + #dests
  // output), so the per-fifo loop skips the side of each linked fifo on a tile
  // of a link reading it, and only those: a linked source fifo can have other
  // consumers, such as a shim that also drains it, and those still need their
  // own channels.
  llvm::DenseMap<Operation *, Operation *> linkTileOf;
  llvm::StringMap<SmallVector<Operation *>> linkTilesOfSource;
  llvm::DenseSet<llvm::StringRef> linkedAsDest;
  for (auto linkOp : objectFifoLinks) {
    Operation *linkTileOp = nullptr;
    for (auto dstFifoAttr : linkOp.getFifoOuts())
      if (auto sym = dyn_cast<FlatSymbolRefAttr>(dstFifoAttr)) {
        linkedAsDest.insert(sym.getValue());
        auto it = fifoNameToOp.find(sym.getValue());
        if (!linkTileOp && it != fifoNameToOp.end())
          linkTileOp = it->second.getProducerTile().getDefiningOp();
      }
    linkTileOf[linkOp.getOperation()] = linkTileOp;
    if (!linkTileOp)
      continue;
    for (auto srcFifoAttr : linkOp.getFifoIns())
      if (auto sym = dyn_cast<FlatSymbolRefAttr>(srcFifoAttr))
        linkTilesOfSource[sym.getValue()].push_back(linkTileOp);
  }

  // A tile op's ends that pin one channel pay for it once; those of tile ops
  // placed together meet in pinsAt.
  pinnedRule = PinnedChannels(*targetModel, packetSwitched);
  pinsAt = PinnedChannelUse(pinnedRule);
  pinnedEnds.clear();
  llvm::DenseSet<unsigned> counted;
  auto drawsChannel = [&](ObjectFifoCreateOp fifo,
                          std::optional<unsigned> consumer, Operation *tile) {
    std::optional<PinnedChannels::End> end =
        PinnedChannels::endOf(fifo, consumer);
    if (!end)
      return true;
    pinnedEnds[tile].push_back(*end);
    return counted.insert(pinnedRule.share(*end)).second;
  };
  // An end takes a channel of its own when the fifo moves by DMA: a shim's
  // unless it is a PLIO, any other one the split gives a DMA end.
  auto dmaEnd = [](ObjectFifoCreateOp fifo, std::optional<unsigned> consumer,
                   Operation *tile) {
    auto like = dyn_cast_or_null<TileLike>(tile);
    return like && like.isShimTile() ? !fifo.getPlio()
                                     : fifo.plannedDmaEnd(consumer).has_value();
  };

  // Count channels for every ObjectFifo, omitting only the link-tile side of
  // linked fifos. The off-link-tile endpoint still needs DMA: a shim that
  // produces a linked source fifo consumes an MM2S channel; a core that
  // consumes a linked dest fifo consumes an S2MM channel.
  for (auto ofOp : objectFifos) {
    SmallVector<Operation *> linkConsumers =
        linkTilesOfSource.lookup(ofOp.getSymName());
    bool skipProducerSide = linkedAsDest.count(ofOp.getSymName());

    Value producerTile = ofOp.getProducerTile();
    auto *producerOp = producerTile.getDefiningOp();
    auto producerLogicalTile = dyn_cast_or_null<LogicalTileOp>(producerOp);

    // Check if ANY consumer is a different tile type (needs DMA channel)
    bool producerNeedsDMA = false;

    for (auto [consumerIndex, consumerTile] :
         llvm::enumerate(ofOp.getConsumerTiles())) {
      auto *consumerOp = consumerTile.getDefiningOp();
      auto consumerLogicalTile = dyn_cast_or_null<LogicalTileOp>(consumerOp);

      // Skip a core-to-core fifo neighbor memory may carry instead of DMA: a
      // compute peer (buildComputePeerAdjacency). One that moves by DMA
      // wherever it sits (dmaReason) is counted here.
      if (producerLogicalTile && consumerLogicalTile &&
          producerLogicalTile.getTileType() == AIETileType::CoreTile &&
          consumerLogicalTile.getTileType() == AIETileType::CoreTile &&
          !ofOp.dmaReason())
        continue;

      // Linked consumers and shared channels are counted separately.
      if (consumerOp && !llvm::is_contained(linkConsumers, consumerOp) &&
          dmaEnd(ofOp, consumerIndex, consumerOp) &&
          drawsChannel(ofOp, consumerIndex, consumerOp))
        channelRequirements[consumerOp].first++; // input++

      producerNeedsDMA = true;
    }

    // Producer needs ONE output channel if any consumer needs DMA (unless the
    // producer IS the link tile of an outgoing linked dest fifo, or it shares
    // an already-counted shared channel).
    if (producerNeedsDMA && producerOp && !skipProducerSide &&
        dmaEnd(ofOp, std::nullopt, producerOp) &&
        drawsChannel(ofOp, std::nullopt, producerOp))
      channelRequirements[producerOp].second++; // output++
  }

  // For linked ObjectFifos, count channels based on the link structure
  // Link tiles need channels for all sources and destinations
  for (auto linkOp : objectFifoLinks) {

    Operation *linkTileOp = linkTileOf.lookup(linkOp.getOperation());
    if (!linkTileOp)
      continue;

    // Link tile needs one input channel per source ObjectFifo and one output
    // channel per dest ObjectFifo, except that a time link's turns share one,
    // and so do ends pinning one channel; their pins hold it there.
    int numInputChannels = 0, numOutputChannels = 0;
    for (auto [sources, input] : {std::make_pair(linkOp.getFifoIns(), true),
                                  std::make_pair(linkOp.getFifoOuts(), false)})
      for (Attribute attr : sources) {
        auto sym = dyn_cast<FlatSymbolRefAttr>(attr);
        ObjectFifoCreateOp fifo =
            sym ? fifoNameToOp.lookup(sym.getValue()) : ObjectFifoCreateOp();
        if (!fifo) {
          ++(input ? numInputChannels : numOutputChannels);
          continue;
        }
        if (!input) {
          numOutputChannels += drawsChannel(fifo, std::nullopt, linkTileOp);
          continue;
        }
        for (unsigned index = 0, e = fifo.getConsumerTiles().size(); index < e;
             ++index)
          if (fifo.getConsumerLink(index) == linkOp)
            numInputChannels += drawsChannel(fifo, index, linkTileOp);
      }
    if (linkOp.isMerge())
      numInputChannels = 1;
    if (linkOp.isDispatch())
      numOutputChannels = 1;

    channelRequirements[linkTileOp].first += numInputChannels;
    channelRequirements[linkTileOp].second += numOutputChannels;
  }

  return channelRequirements;
}

void SequentialPlacer::countTurns(
    DeviceOp device, ArrayRef<ObjectFifoCreateOp> objectFifos,
    llvm::DenseMap<Operation *, std::pair<int, int>> &requirements) {
  for (CoreOp core : device.getOps<CoreOp>()) {
    auto tile = dyn_cast_or_null<LogicalTileOp>(core.getTile().getDefiningOp());
    if (!tile || !requirements.count(tile))
      continue;
    auto &[in, out] = requirements[tile];
    for (bool coreDrains : {true, false}) {
      int &need = coreDrains ? in : out;
      auto [inBudget, outBudget] =
          getDMACapacity(*targetModel, tile.getCanonicalTileID());
      int budget = coreDrains ? inBudget : outBudget;
      if (need <= budget)
        continue;
      // Peers with coordinates are one mem tile however many ops name it.
      llvm::MapVector<std::pair<Operation *, std::optional<TileID>>,
                      SmallVector<ObjectFifoCreateOp>>
          byTile;
      for (ObjectFifoCreateOp fifo : objectFifos) {
        if (!fifo.canTakeTurns())
          continue;
        auto consumers = fifo.getConsumerTiles();
        Value mine = coreDrains ? consumers[0] : fifo.getProducerTile();
        Value other = coreDrains ? fifo.getProducerTile() : consumers[0];
        auto peer = dyn_cast_or_null<TileLike>(other.getDefiningOp());
        if (mine != tile.getResult() || !peer || !peer.isMemTile())
          continue;
        std::optional<int> col = peer.tryGetCol(), row = peer.tryGetRow();
        if (col && row)
          byTile[{nullptr, TileID{*col, *row}}].push_back(fifo);
        else
          byTile[{peer, std::nullopt}].push_back(fifo);
      }
      for (auto &[peer, fifos] : byTile) {
        if (fifos.size() < 2 || need <= budget)
          continue;
        SmallVector<StringAttr> names;
        SmallVector<int64_t> depths;
        for (ObjectFifoCreateOp fifo : fifos) {
          names.push_back(fifo.getSymNameAttr());
          depths.push_back(fifo.dmaDepth(coreDrains ? std::optional<unsigned>(0)
                                                    : std::nullopt));
        }
        CoreTurns turns;
        std::string reason;
        if (succeeded(turns.read(core, names, coreDrains, reason)) &&
            turns.keepsUp(depths))
          need -= static_cast<int>(fifos.size()) - 1;
      }
    }
  }
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

Placer::Adjacency SequentialPlacer::buildBufferAdjacency(
    ArrayRef<LogicalTileOp> logicalTiles,
    ArrayRef<ObjectFifoAllocateOp> allocates) {
  Adjacency adjacency;
  // One edge per (user, owner) pair: a duplicate would book a second free
  // tile for the same peer in leavesPeerRoom.
  llvm::DenseSet<std::pair<Operation *, Operation *>> seen;
  auto addOnce = [&](TileLike user, TileLike owner) {
    if (seen.insert({user.getOperation(), owner.getOperation()}).second)
      adjacency.addEdge(user, owner);
  };
  // An allocated fifo's objects live in the delegate's memory, which each of
  // its ends then reaches as it would a buffer the delegate owns.
  for (auto alloc : allocates) {
    auto delegate =
        dyn_cast_or_null<TileLike>(alloc.getDelegateTile().getDefiningOp());
    ObjectFifoCreateOp fifo = alloc.getObjectFifo();
    if (!delegate || !fifo || delegate.getTileType() != AIETileType::CoreTile)
      continue;
    SmallVector<Value> ends{fifo.getProducerTile()};
    llvm::append_range(ends, fifo.getConsumerTiles());
    for (Value end : ends) {
      auto tile = dyn_cast_or_null<TileLike>(end.getDefiningOp());
      if (tile && tile.getTileType() == AIETileType::CoreTile &&
          tile.getOperation() != delegate.getOperation())
        addOnce(tile, delegate);
    }
  }
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

    core.getBody().walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        BufferOp buf = traceToBuffer(operand);
        if (!buf)
          continue;
        auto owner = dyn_cast_or_null<TileLike>(buf.getTile().getDefiningOp());
        if (owner && owner.getOperation() != consumer.getOperation())
          addOnce(TileLike(consumer), owner);
      }
    });
  }
  return adjacency;
}

Placer::Adjacency
Placer::buildCascadeAdjacency(ArrayRef<CascadeFlowOp> cascadeFlows) {
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

Placer::Adjacency
Placer::buildComputePeerAdjacency(ArrayRef<ObjectFifoCreateOp> objectFifos) {
  Adjacency adjacency;
  for (auto ofOp : objectFifos) {
    auto producer =
        dyn_cast_or_null<LogicalTileOp>(ofOp.getProducerTile().getDefiningOp());
    if (!producer || producer.getTileType() != AIETileType::CoreTile)
      continue;
    // Only a fifo neighbor memory may carry is a compute peer.
    if (ofOp.dmaReason())
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

Placer::Adjacency
Placer::buildSharedMemAdjacency(ArrayRef<ObjectFifoCreateOp> objectFifos) {
  Adjacency adjacency;
  for (auto ofOp : objectFifos)
    if (ofOp.forcesSharedMemory() && ofOp.getConsumerTiles().size() == 1)
      adjacency.addEdgeFromValues(ofOp.getProducerTile(),
                                  ofOp.getConsumerTiles()[0]);
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
    recordColumnStreamUsage(logicalTile, tile.col);
    availability.removeTile(tile, AIETileType::CoreTile);
    ctx.reserveNeighborSlots(logicalTile.getOperation(), tile);
  }
}

std::pair<int, int>
SequentialPlacer::totalComputePeers(Operation *op, const Adjacency &adjacency) {
  int in = 0, out = 0;
  // Compute-peer convention: edge.first is the producer (OUT for `op`),
  // edge.second is the consumer (IN for `op`).
  forEachPeer(op, adjacency, [&](TileLike /*peer*/, bool thisIsFirst) {
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

  // Self side: count placed peers that would land non-neighbor of
  // candidate, and reject if the count exceeds the slack budget.
  int selfNonNeighborIn = 0, selfNonNeighborOut = 0;
  int selfNeighborIn = 0, selfNeighborOut = 0;
  forEachPeer(logicalTile.getOperation(), computePeerAdjacency,
              [&](TileLike peer, bool thisIsFirst) {
                auto peerPos = resolvePeerPosition(peer, result);
                if (!peerPos)
                  return;
                bool isNeighbor =
                    isSharedComputeMemory(*targetModel, candidate, *peerPos);
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

  // Forward-look: even if no placed peer violates slack at this
  // candidate, future peers still to be placed must land on a physical
  // compute neighbor of candidate. Count candidate's free shared-L1
  // compute neighbors (N, S, and the checkerboard W neighbor on AIE2)
  // and reject if the remaining demand exceeds that count. A free slot
  // can host an IN-peer OR an OUT-peer (not both); the sum-check is a
  // conservative approximation.
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
              // peerLTO is the consumer of this edge when it sits on
              // edge.second (compute-peer convention is producer=first,
              // consumer=second). logicalIsPeerOut selects peer's OUT
              // edges (peer = producer), which is the opposite role.
              bool peerIsConsumerHere = !peerLTOIsFirst;
              if (logicalIsPeerOut == peerIsConsumerHere)
                return;
              if (peerPeer.getOperation() == logicalTile.getOperation()) {
                // For the edge currently being decided, score candidate
                // against peerPos.
                if (!isSharedComputeMemory(*targetModel, *peerPos, candidate))
                  ++peerNonNeighborDir;
                return;
              }
              auto pp = resolvePeerPosition(peerPeer, result);
              if (!pp)
                return;
              if (!isSharedComputeMemory(*targetModel, *peerPos, *pp))
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

Placer::Adjacency
Placer::buildObjectFifoAdjacency(ArrayRef<ObjectFifoCreateOp> objectFifos) {
  Adjacency adjacency;
  for (auto ofOp : objectFifos)
    for (Value consumer : ofOp.getConsumerTiles())
      adjacency.addEdgeFromValues(ofOp.getProducerTile(), consumer);
  return adjacency;
}

Placer::ShareableShimEnds Placer::collectShareableShimEnds(
    DeviceOp device, ArrayRef<ObjectFifoCreateOp> objectFifos,
    const llvm::DenseMap<Operation *, Operation *> &routeKeys) {
  llvm::DenseSet<std::pair<Operation *, Value>> registered;
  for (auto reg : device.getOps<ObjectFifoRegisterExternalBuffersOp>())
    registered.insert({reg.getObjectFifo(), reg.getTile()});
  ShareableShimEnds ends;
  auto shareable = [&](ObjectFifoCreateOp fifo, Value tile,
                       ObjectFifoEndPortAttr port) -> Operation * {
    auto lt = dyn_cast_or_null<TileLike>(tile.getDefiningOp());
    if (!lt || lt.getTileType() != AIETileType::ShimNOCTile || fifo.getPlio() ||
        (port && port.channelIndex()) || registered.contains({fifo, tile}))
      return nullptr;
    return lt.getOperation();
  };
  for (auto fifo : objectFifos) {
    if (Operation *lt =
            shareable(fifo, fifo.getProducerTile(), fifo.getProdPortAttr()))
      ends[lt][1].ends.push_back(fifo.getSymNameAttr());
    for (auto [i, consumer] : llvm::enumerate(fifo.getConsumerTiles()))
      if (Operation *lt = shareable(fifo, consumer, fifo.getConsumerPort(i)))
        ends[lt][0].ends.push_back(fifo.getSymNameAttr());
  }
  for (auto endpoint : device.getOps<RouteEndpointOp>()) {
    auto tile = dyn_cast<TileLike>(endpoint.getTile().getDefiningOp());
    if (!tile || !tile.isShimTile() || endpoint.getBundle() != WireBundle::DMA)
      continue;
    ShimSide &side =
        ends[routeKeys.lookup_or(tile.getOperation(), tile.getOperation())]
            [endpoint.getRouteDirection() == DMAChannelDir::MM2S ? 1 : 0];
    if (std::optional<int> channel = endpoint.getChannelIndex())
      side.pinned.push_back(*channel);
    else if (isa<LogicalTileOp>(tile.getOperation()))
      side.ends.push_back(endpoint.getSymNameAttr());
  }
  for (auto alloc : device.getOps<ShimDMAAllocationOp>()) {
    MLIRContext *context = alloc->getContext();
    auto tile = dyn_cast_or_null<TileLike>(alloc.getTile().getDefiningOp());
    auto allowed = AIEDialect::ShareableWithAttrHelper(context).getAttr(alloc);
    if (!tile || !tile.isShimTile() || !allowed ||
        AIEDialect::IsCtrlPktOverlayAttrHelper(context).isAttrPresent(alloc))
      continue;
    SmallVector<StringAttr> listed;
    for (Attribute attr : allowed)
      if (auto symbol = dyn_cast<FlatSymbolRefAttr>(attr))
        listed.push_back(symbol.getAttr());
    ends[routeKeys.lookup_or(tile.getOperation(), tile.getOperation())]
        [alloc.getChannelDir() == DMAChannelDir::MM2S ? 1 : 0]
            .shared.emplace_back(alloc.getChannelIndex(), std::move(listed));
  }
  return ends;
}

int Placer::shimChannelsSaved(const ShimTransferSpans &spans,
                              ArrayRef<const ShimSide *> sides,
                              ArrayRef<StringAttr> extra,
                              ArrayRef<int> pinnedHere) {
  SmallVector<StringAttr> ends;
  SmallVector<int> pinned(pinnedHere);
  for (const ShimSide *side : sides) {
    llvm::append_range(ends, side->ends);
    llvm::append_range(pinned, side->pinned);
  }
  llvm::append_range(ends, extra);
  SmallVector<ShimTransferSpans::SharedChannel> shared;
  for (const ShimSide *side : sides)
    for (auto &[channel, listed] : side->shared)
      shared.emplace_back(listed, llvm::is_contained(pinned, channel));
  ShimTransferSpans::Plan plan = spans.plan(ends, ends, shared);
  return static_cast<int>(ends.size() - plan.groups.size());
}

std::optional<Placer::PinnedChannels::End>
Placer::PinnedChannels::endOf(ObjectFifoCreateOp fifo,
                              std::optional<unsigned> consumer) {
  ObjectFifoEndPortAttr port =
      consumer ? fifo.getConsumerPort(*consumer) : fifo.getProdPortAttr();
  std::optional<int> pin = port ? port.channelIndex() : std::nullopt;
  if (!pin || port.isCore() || *pin < 0)
    return std::nullopt;
  Value tile =
      consumer ? fifo.getConsumerTiles()[*consumer] : fifo.getProducerTile();
  auto offCore = [](Value end) {
    return cast<TileLike>(end.getDefiningOp()).getTileType() !=
           AIETileType::CoreTile;
  };
  bool movesByDma = fifo.dmaReason() || offCore(fifo.getProducerTile()) ||
                    llvm::any_of(fifo.getConsumerTiles(), offCore);
  return End{fifo, consumer, tile.getDefiningOp(), !consumer, *pin, movesByDma};
}

std::optional<std::string> Placer::PinnedChannels::whyApart(End a,
                                                            End b) const {
  if (a.sends) {
    for (ObjectFifoCreateOp fifo : {a.fifo, b.fifo}) {
      auto tile = cast<TileLike>(fifo.getProducerTile().getDefiningOp());
      if (tile.isShimTile())
        continue;
      if (!tile.isCoreTile() || targetModel->getDmaTaskQueueDepth() == 0)
        return std::string("only a shim, or a core that pushes its tasks, "
                           "sends two fifos on one channel");
    }
    if (!isPacketSwitched(a.fifo.packetHeader(), packetSwitched) ||
        !isPacketSwitched(b.fifo.packetHeader(), packetSwitched))
      return std::string("a shared channel carries both at once, which only "
                         "packet-switched routes tell apart");
    // A core's sharers each get push_queue (sharesSendChannel).
    for (ObjectFifoCreateOp fifo : {a.fifo, b.fifo})
      if (std::optional<StringRef> why =
              cast<TileLike>(fifo.getProducerTile().getDefiningOp())
                      .isShimTile()
                  ? std::nullopt
                  : fifo.pushQueueRefusal())
        return "@" + fifo.getSymName().str() + "'s " + why->str();
  } else {
    for (End end : {a, b})
      if (std::optional<StringRef> why =
              ObjectFifoCreateOp::sharedReceiveRefusal(
                  cast<TileLike>(end.tile).isShimTile(),
                  end.fifo.getOutOfOrder()))
        return why->str();
  }
  std::optional<uint16_t> id =
      a.fifo.packetHeader() ? a.fifo.packetHeader().getPktId() : std::nullopt;
  std::optional<uint16_t> otherId =
      b.fifo.packetHeader() ? b.fifo.packetHeader().getPktId() : std::nullopt;
  if (id && otherId && *id == *otherId)
    return "both pin packet id " + std::to_string(*id);
  return std::nullopt;
}

unsigned Placer::PinnedChannels::share(End end) {
  auto key =
      std::make_tuple(end.tile, end.channel, static_cast<int>(end.sends));
  auto &[before, number] = taken
                               .try_emplace(key, SmallVector<End, 2>{},
                                            static_cast<unsigned>(taken.size()))
                               .first->second;
  for (End other : before) {
    if (!end.movesByDma || !other.movesByDma)
      continue;
    if (std::optional<std::string> why = whyApart(end, other)) {
      end.fifo.emitError() << "shares " << (end.sends ? "MM2S" : "S2MM")
                           << " channel " << end.channel << " with @"
                           << other.fifo.getSymName() << ", but " << *why;
      diagnosed = true;
      break;
    }
  }
  before.push_back(end);
  return number;
}

bool Placer::PinnedChannels::sharesClaim(
    End end, const DMAChannelAnalysis::ChannelClaim &claim) const {
  auto alloc = dyn_cast_or_null<ShimDMAAllocationOp>(claim.owner);
  if (!alloc)
    return false;
  auto allowed =
      AIEDialect::ShareableWithAttrHelper(alloc->getContext()).getAttr(alloc);
  if (!allowed || llvm::none_of(allowed, [&](Attribute attr) {
        auto symbol = dyn_cast<FlatSymbolRefAttr>(attr);
        return symbol && symbol.getAttr() == end.fifo.getSymNameAttr();
      }))
    return false;
  return end.sends ? isPacketSwitched(end.fifo.packetHeader(), packetSwitched)
                   : end.fifo.getOutOfOrder();
}

void Placer::PinnedChannelUse::claim(
    TileID at, const DMAChannelAnalysis::ChannelClaim &claim, int sign) {
  auto key =
      std::make_tuple(at.col, at.row, claim.channel,
                      static_cast<int>(claim.dir == DMAChannelDir::MM2S));
  auto &claims = claimed[key];
  if (sign > 0) {
    claims.push_back(claim);
    return;
  }
  auto *found =
      llvm::find_if(claims, [&](const DMAChannelAnalysis::ChannelClaim &c) {
        return c.owner == claim.owner;
      });
  assert(found != claims.end() && "removing a claim never placed");
  claims.erase(found);
  if (claims.empty())
    claimed.erase(key);
}

std::optional<DMAChannelAnalysis::ChannelClaim>
Placer::PinnedChannelUse::claimWith(TileID at,
                                    const PinnedChannels::End &end) const {
  auto it = claimed.find(std::make_tuple(at.col, at.row, end.channel,
                                         static_cast<int>(end.sends)));
  if (it == claimed.end())
    return std::nullopt;
  for (const DMAChannelAnalysis::ChannelClaim &claim : it->second)
    if (!rule->sharesClaim(end, claim))
      return claim;
  return std::nullopt;
}

int Placer::PinnedChannelUse::place(TileID at, const PinnedChannels::End &end,
                                    int sign) {
  auto key =
      std::make_tuple(at.col, at.row, end.channel, static_cast<int>(end.sends));
  if (sign > 0) {
    auto &ends = placed[key];
    ends.push_back(end);
    return ends.size() == 1 ? 1 : 0;
  }
  auto it = placed.find(key);
  assert(it != placed.end() && "removing an end never placed");
  auto *found = llvm::find_if(it->second, [&](const PinnedChannels::End &e) {
    return e.fifo == end.fifo && e.consumer == end.consumer;
  });
  assert(found != it->second.end() && "removing an end never placed");
  it->second.erase(found);
  if (!it->second.empty())
    return 0;
  placed.erase(it);
  return -1;
}

int Placer::PinnedChannelUse::clashes(TileID at) const {
  auto claimsClash = [](ArrayRef<DMAChannelAnalysis::ChannelClaim> claims) {
    return llvm::any_of(claims,
                        [&](const DMAChannelAnalysis::ChannelClaim &claim) {
                          return claim.tile != claims.front().tile;
                        });
  };
  int count = 0;
  for (auto it = placed.lower_bound(
           std::make_tuple(at.col, at.row, std::numeric_limits<int>::min(), 0));
       it != placed.end() && std::get<0>(it->first) == at.col &&
       std::get<1>(it->first) == at.row;
       ++it) {
    const auto &ends = it->second;
    bool clash = false;
    for (size_t i = 0; i < ends.size() && !clash; ++i)
      for (size_t j = i + 1; j < ends.size() && !clash; ++j)
        clash = (ends[i].tile != ends[j].tile || !ends[i].movesByDma ||
                 !ends[j].movesByDma) &&
                rule->whyApart(ends[i], ends[j]).has_value();
    if (auto held = claimed.find(it->first); held != claimed.end())
      clash = clash || claimsClash(held->second) ||
              llvm::any_of(ends, [&](const PinnedChannels::End &end) {
                return llvm::any_of(
                    held->second,
                    [&](const DMAChannelAnalysis::ChannelClaim &claim) {
                      return !rule->sharesClaim(end, claim);
                    });
              });
    count += clash;
  }
  for (auto it = claimed.lower_bound(
           std::make_tuple(at.col, at.row, std::numeric_limits<int>::min(), 0));
       it != claimed.end() && std::get<0>(it->first) == at.col &&
       std::get<1>(it->first) == at.row;
       ++it)
    count += !placed.count(it->first) && claimsClash(it->second);
  return count;
}

std::optional<Placer::PinnedChannels::End>
Placer::PinnedChannelUse::clashWith(TileID at,
                                    const PinnedChannels::End &end) const {
  auto it = placed.find(std::make_tuple(at.col, at.row, end.channel,
                                        static_cast<int>(end.sends)));
  if (it == placed.end())
    return std::nullopt;
  for (const PinnedChannels::End &other : it->second)
    if (other.tile != end.tile && rule->whyApart(end, other))
      return other;
  return std::nullopt;
}

llvm::SmallVector<int> Placer::PinnedChannelUse::channelsAt(TileID at,
                                                            bool sends) const {
  llvm::SmallVector<int> channels;
  for (auto it = placed.lower_bound(
           std::make_tuple(at.col, at.row, std::numeric_limits<int>::min(), 0));
       it != placed.end() && std::get<0>(it->first) == at.col &&
       std::get<1>(it->first) == at.row;
       ++it)
    if (std::get<3>(it->first) == static_cast<int>(sends))
      channels.push_back(std::get<2>(it->first));
  return channels;
}

llvm::DenseMap<int, llvm::SmallVector<Placer::PinnedChannels::End, 2>>
Placer::PinnedChannelUse::sendingAt(TileID at) const {
  llvm::DenseMap<int, llvm::SmallVector<PinnedChannels::End, 2>> sending;
  for (auto it = placed.lower_bound(
           std::make_tuple(at.col, at.row, std::numeric_limits<int>::min(), 0));
       it != placed.end() && std::get<0>(it->first) == at.col &&
       std::get<1>(it->first) == at.row;
       ++it)
    if (std::get<3>(it->first))
      llvm::append_range(sending[std::get<2>(it->first)], it->second);
  return sending;
}

std::optional<std::pair<Operation *, std::pair<int, int>>>
Placer::namedClash(Operation *tile, ArrayRef<Operation *> others) const {
  auto mine = namedChannels.find(tile);
  if (mine == namedChannels.end())
    return std::nullopt;
  for (Operation *other : others) {
    auto theirs = namedChannels.find(other);
    if (other == tile || theirs == namedChannels.end())
      continue;
    for (std::pair<int, int> named : mine->second)
      if (llvm::is_contained(theirs->second, named))
        return std::make_pair(other, named);
  }
  return std::nullopt;
}

LogicalResult Placer::emitNamedClash(
    Operation *tile, TileID at,
    std::pair<Operation *, std::pair<int, int>> clash) const {
  auto [other, named] = clash;
  auto [output, channel] = named;
  StringRef dir = output ? "MM2S" : "S2MM";
  InFlightDiagnostic diag =
      tile->emitError()
      << "names " << dir << " channel " << channel << ", as a tile op on ("
      << at.col << ", " << at.row << ") does, and each "
      << stringifyAIETileType(targetModel->getTileType(at.col, at.row))
      << " with room for it holds a tile op naming one of its channels; tile "
         "ops whose DMA programs or flows name one channel cannot share a "
         "physical tile, so give one of them another channel";
  diag.attachNote(other->getLoc())
      << "the tile op naming " << dir << " channel " << channel;
  return diag;
}

void Placer::collectOverlayUses(DeviceOp device) {
  overlayUses = shimMM2SUses(device, packetSwitched);
  auto module = device->getParentOfType<ModuleOp>();
  if (!controlOverlay || !module)
    return;
  for (DeviceOp other : module.getOps<DeviceOp>()) {
    if (other == device || other.getDevice() != device.getDevice() ||
        !getsColumnControlOverlay(other, true))
      continue;
    ShimMM2SUses uses = shimMM2SUses(other, packetSwitched);
    for (auto fifo : other.getOps<ObjectFifoCreateOp>()) {
      std::optional<PinnedChannels::End> end =
          PinnedChannels::endOf(fifo, std::nullopt);
      if (!end)
        continue;
      auto tile = dyn_cast_or_null<TileOp>(end->tile);
      if (!tile || !tile.isShimNOCorPLTile())
        continue;
      if (isPacketSwitched(fifo.packetHeader(), packetSwitched))
        uses.packetEnds[{tile, end->channel}].push_back(fifo.getSymNameAttr());
      else
        uses.circuits[tile].insert(end->channel);
    }
    for (auto &[tile, channels] : uses.circuits)
      if (isa<TileOp>(tile))
        overlayUses.circuits[tile].insert(channels.begin(), channels.end());
    for (auto &[key, ends] : uses.packetEnds)
      if (isa<TileOp>(key.first))
        llvm::append_range(overlayUses.packetEnds[key], ends);
    for (auto &[key, flows] : uses.packetFlows)
      if (isa<TileOp>(key.first))
        overlayUses.packetFlows[key] += flows;
  }
}

SmallVector<Operation *> Placer::overlayTileOps() const {
  llvm::SetVector<Operation *> tiles;
  for (auto &[tile, channels] : overlayUses.circuits)
    tiles.insert(tile);
  for (auto &[key, ends] : overlayUses.packetEnds)
    tiles.insert(key.first);
  for (auto &[key, flows] : overlayUses.packetFlows)
    tiles.insert(key.first);
  return tiles.takeVector();
}

Placer::OverlayFit Placer::overlayAt(TileID at, const PinnedChannelUse &pins,
                                     ArrayRef<PinnedChannels::End> extra,
                                     ArrayRef<Operation *> tilesHere,
                                     const ShimTransferSpans *spans) const {
  if (!overlayChannels(at))
    return {};
  llvm::DenseSet<int> circuits;
  llvm::DenseMap<int, SmallVector<StringAttr>> packetEnds;
  llvm::DenseMap<int, unsigned> packetFlows;
  for (Operation *tile : tilesHere)
    if (auto it = overlayUses.circuits.find(tile);
        it != overlayUses.circuits.end())
      circuits.insert(it->second.begin(), it->second.end());
  for (auto &[key, ends] : overlayUses.packetEnds)
    if (llvm::is_contained(tilesHere, key.first))
      llvm::append_range(packetEnds[key.second], ends);
  for (auto &[key, flows] : overlayUses.packetFlows)
    if (llvm::is_contained(tilesHere, key.first))
      packetFlows[key.second] += flows;
  auto onChannel = pins.sendingAt(at);
  for (const PinnedChannels::End &end : extra)
    if (end.sends)
      onChannel[end.channel].push_back(end);
  int count = adaptiveOverlay ? targetModel->getNumSourceShimMuxConnections(
                                    at.col, at.row, WireBundle::DMA)
                              : 1;
  std::optional<int> chosen;
  size_t fewest = std::numeric_limits<size_t>::max();
  for (int channel = 0; channel < count; ++channel) {
    auto ends = onChannel.lookup(channel);
    if (circuits.contains(channel) ||
        llvm::any_of(ends, [&](PinnedChannels::End end) {
          return !isPacketSwitched(end.fifo.packetHeader(), packetSwitched);
        }))
      continue;
    size_t demand = ends.size() + packetEnds.lookup(channel).size() +
                    packetFlows.lookup(channel);
    if (demand < fewest) {
      fewest = demand;
      chosen = channel;
    }
  }
  if (!chosen)
    return {/*fits=*/false, /*channels=*/1, /*own=*/false};
  SmallVector<StringAttr> sharers = packetEnds.lookup(*chosen);
  for (PinnedChannels::End end : onChannel.lookup(*chosen))
    sharers.push_back(end.fifo.getSymNameAttr());
  if (sharers.empty() && !packetFlows.lookup(*chosen))
    return {/*fits=*/true, /*channels=*/1, /*own=*/true};
  bool apart = llvm::all_of(sharers, [&](StringAttr end) {
    return spans && spans->apart(spans->overlayEnd(at.col), end);
  });
  return {apart, /*channels=*/0, /*own=*/false};
}

std::optional<std::pair<int, int>>
Placer::PinnedChannelUse::heldAt(TileID at,
                                 ArrayRef<PinnedChannels::End> ends) const {
  std::pair<int, int> held{0, 0};
  llvm::SmallDenseSet<std::pair<int, int>, 4> counted;
  for (const PinnedChannels::End &end : ends) {
    if (clashWith(at, end) || claimWith(at, end))
      return std::nullopt;
    auto key = std::make_tuple(at.col, at.row, end.channel,
                               static_cast<int>(end.sends));
    if ((placed.count(key) || claimed.count(key)) &&
        counted.insert({end.channel, end.sends}).second)
      ++(end.sends ? held.second : held.first);
  }
  return held;
}

DenseMap<StringAttr, SmallVector<ShimTransferSpans::Gate>>
Placer::provenShimGates(DeviceOp device) {
  auto module = device->getParentOfType<ModuleOp>();
  if (!module)
    return {};
  unsigned index = 0;
  for (DeviceOp other : module.getOps<DeviceOp>()) {
    if (other == device)
      break;
    ++index;
  }
  // A context of its own: the passes below cannot run nested in this one's.
  std::string text;
  llvm::raw_string_ostream os(text);
  module->print(os, OpPrintingFlags().printGenericOpForm());
  MLIRContext context(device.getContext()->getDialectRegistry(),
                      MLIRContext::Threading::DISABLED);
  ScopedDiagnosticHandler quiet(&context,
                                [](Diagnostic &) { return success(); });
  OwningOpRef<ModuleOp> copy = parseSourceString<ModuleOp>(text, &context);
  if (!copy)
    return {};
  auto devices = llvm::to_vector(copy->getOps<DeviceOp>());
  if (index >= devices.size())
    return {};
  DeviceOp clone = devices[index];

  // Split needs placed tiles. With every fifo between two tiles on DMA,
  // which tiles they are does not change what verify proves.
  const AIETargetModel &target = clone.getTargetModel();
  DenseSet<TileID> usedCores;
  for (auto tile : clone.getOps<TileOp>())
    if (tile.isCoreTile())
      usedCores.insert({tile.getCol(), tile.getRow()});
  DenseMap<AIETileType, unsigned> turn;
  OpBuilder builder(clone.getBody(), clone.getBody()->begin());
  for (auto lt : llvm::to_vector(clone.getOps<LogicalTileOp>())) {
    SmallVector<TileID> fits;
    for (int col = 0; col < target.columns(); ++col)
      for (int row = 0; row < target.rows(); ++row)
        if (target.getTileType(col, row) == lt.getTileType() &&
            satisfiesConstraints(lt, {col, row}) &&
            (lt.getTileType() != AIETileType::CoreTile ||
             !usedCores.contains({col, row})))
          fits.push_back({col, row});
    if (fits.empty())
      return {};
    TileID at = lt.getTileType() == AIETileType::CoreTile
                    ? fits.front()
                    : fits[turn[lt.getTileType()]++ % fits.size()];
    if (lt.getTileType() == AIETileType::CoreTile)
      usedCores.insert(at);
    TileOp tile = TileOp::getOrCreate(builder, clone, at.col, at.row);
    lt.getResult().replaceAllUsesWith(tile.getResult());
    lt.erase();
  }
  for (auto fifo : clone.getOps<ObjectFifoCreateOp>()) {
    if (fifo.forcesDMA() ||
        llvm::all_of(fifo.getConsumerTiles(), [&](Value consumer) {
          return consumer == fifo.getProducerTile();
        }))
      continue;
    ObjectFifoTransportAttr transport = fifo.getTransportAttr();
    fifo.setTransportAttr(ObjectFifoTransportAttr::get(
        &context, ObjectFifoTransportMode::DMA,
        transport ? transport.getPacket() : PacketInfoAttr(),
        transport && transport.getOutOfOrder()));
  }
  for (auto allocate : llvm::to_vector(clone.getOps<ObjectFifoAllocateOp>()))
    allocate.erase();

  PassManager pm(&context, DeviceOp::getOperationName());
  pm.addPass(createAIEObjectFifoSplitPass());
  pm.addPass(createAIEObjectFifoVerifyPass());
  if (failed(pm.run(clone)))
    return {};
  DenseMap<StringAttr, SmallVector<ShimTransferSpans::Gate>> gates;
  MLIRContext *ctx = device.getContext();
  for (auto &[fifo, proven] :
       ShimTransferSpans::provenGates(clone, [](RouteEndpointOp endpoint) {
         return endpoint.getFifoNameAttr();
       }))
    for (const ShimTransferSpans::Gate &gate : proven)
      gates[StringAttr::get(ctx, fifo.getValue())].push_back(
          {StringAttr::get(ctx, gate.input.getValue()), gate.inputBytes,
           gate.outputBytes, gate.granuleBytes});
  return gates;
}

std::optional<ShimTransferSpans::End> Placer::shimEndOf(DeviceOp device,
                                                        StringAttr symbol) {
  if (auto endpoint = dyn_cast_or_null<RouteEndpointOp>(
          SymbolTable::lookupNearestSymbolFrom(device, symbol)))
    return ShimTransferSpans::End{symbol, endpoint.getRouteDirection() ==
                                              DMAChannelDir::MM2S};
  auto fifo = dyn_cast_or_null<ObjectFifoCreateOp>(
      SymbolTable::lookupNearestSymbolFrom(device, symbol));
  if (!fifo)
    return std::nullopt;
  auto producer =
      dyn_cast_or_null<TileLike>(fifo.getProducerTile().getDefiningOp());
  return ShimTransferSpans::End{symbol, producer && producer.isShimTile()};
}

Placer::Adjacency Placer::buildFlowAdjacency(ArrayRef<FlowOp> flows,
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

// Tiles with both coordinates are one tile however many ops name them, as in
// DMAChannelAnalysis::getTileKey: their demand is keyed by the first op seen
// there, placed or not, so a physical tile's channels count at its position.
// A DMA channel is a hardware resource: a broadcast (one source channel feeding
// multiple destinations) consumes one MM2S channel on the producer regardless
// of the number of `aie.flow` ops it lowers to, and a merge (multiple sources
// landing on one destination channel) consumes one S2MM channel on the
// consumer. Dedup by (tile, channel) so the producer-of-broadcast and
// consumer-of-merge sides are each counted once. An `aie.route` end whose
// channel allocation has yet to pick holds a channel of its own.
llvm::MapVector<Operation *, std::pair<int, int>> Placer::flowChannelDemand(
    ArrayRef<FlowOp> flows, ArrayRef<PacketFlowOp> pktFlows,
    ArrayRef<RouteOp> routes, ArrayRef<DMAChannelAnalysis::ChannelClaim> claims,
    llvm::DenseMap<Operation *, SmallVector<std::pair<int, int>>> *named,
    llvm::DenseMap<Operation *, Operation *> *keys) {
  llvm::MapVector<Operation *, std::pair<int, int>> demand;
  llvm::DenseSet<std::tuple<Operation *, int>> seenSrc, seenDst;
  llvm::DenseMap<std::pair<int, int>, Operation *> byCoordinate;

  auto incIfDMA = [&](Operation *tileOp, WireBundle bundle,
                      std::optional<int> channel, bool isOutput) {
    auto tile = dyn_cast_or_null<TileLike>(tileOp);
    if (!tile || bundle != WireBundle::DMA)
      return;
    Operation *key = tileOp;
    if (auto col = tile.tryGetCol(), row = tile.tryGetRow(); col && row)
      key = byCoordinate.try_emplace({*col, *row}, tileOp).first->second;
    if (keys)
      (*keys)[tileOp] = key;
    tileOp = key;
    auto &seen = isOutput ? seenSrc : seenDst;
    if (channel && !seen.insert({tileOp, *channel}).second)
      return;
    if (channel && named)
      (*named)[tileOp].push_back({isOutput, *channel});
    if (isOutput)
      demand[tileOp].second++;
    else
      demand[tileOp].first++;
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

  for (const DMAChannelAnalysis::ChannelClaim &claim : claims)
    incIfDMA(claim.tile.getDefiningOp(), WireBundle::DMA, claim.channel,
             claim.dir == DMAChannelDir::MM2S);

  // Packet routes may share an endpoint, which holds one channel however many
  // routes name it, and a fan-in's sources share one destination channel.
  llvm::DenseSet<Operation *> countedEndpoints;
  for (auto route : routes) {
    auto count = [&](StringRef name, bool isOutput) {
      auto endpoint = SymbolTable::lookupNearestSymbolFrom<RouteEndpoint>(
          route, StringAttr::get(route.getContext(), name));
      if (endpoint && countedEndpoints.insert(endpoint).second)
        incIfDMA(endpoint.getTile().getDefiningOp(), endpoint.getRouteBundle(),
                 endpoint.getRouteChannel(), isOutput);
    };
    for (StringRef source : route.getSourceNames())
      count(source, /*isOutput=*/true);
    for (StringRef dest : route.getDestinationNames())
      count(dest, /*isOutput=*/false);
  }
  return demand;
}

SequentialPlacer::FlowMembership SequentialPlacer::buildFlowMembership(
    ArrayRef<FlowOp> flows, ArrayRef<PacketFlowOp> pktFlows,
    ArrayRef<RouteOp> routes, ArrayRef<ObjectFifoCreateOp> objectFifos) {
  // packet_flow connectivity is sources x destinations -- destinations
  // are never each other's peers. Same asymmetry for route and objectfifo
  // producer/consumers.
  FlowMembership idx;
  auto isLto = [](Value v) {
    return v && mlir::isa_and_nonnull<LogicalTileOp>(v.getDefiningOp());
  };
  auto addEntry = [&](Value lto, ArrayRef<Value> peers) {
    if (!isLto(lto) || peers.empty())
      return;
    idx.ltoFlows[lto].push_back(SmallVector<Value>(peers.begin(), peers.end()));
  };

  for (auto flow : flows) {
    Value s = flow.getSource(), d = flow.getDest();
    addEntry(s, {d});
    addEntry(d, {s});
  }
  for (auto pf : pktFlows) {
    SmallVector<Value> srcs, dsts;
    pf.walk([&](Operation *op) {
      if (auto src = dyn_cast<PacketSourceOp>(op))
        srcs.push_back(src.getTile());
      else if (auto dst = dyn_cast<PacketDestOp>(op))
        dsts.push_back(dst.getTile());
    });
    for (Value s : srcs)
      addEntry(s, dsts);
    for (Value d : dsts)
      addEntry(d, srcs);
  }
  for (auto route : routes) {
    auto tileOf = [&](StringRef name) -> Value {
      auto endpoint = SymbolTable::lookupNearestSymbolFrom<RouteEndpoint>(
          route, StringAttr::get(route.getContext(), name));
      return endpoint ? endpoint.getTile() : Value();
    };
    SmallVector<Value> srcs, dsts;
    for (StringRef source : route.getSourceNames())
      if (Value tile = tileOf(source))
        srcs.push_back(tile);
    for (StringRef dest : route.getDestinationNames())
      if (Value tile = tileOf(dest))
        dsts.push_back(tile);
    for (Value s : srcs)
      addEntry(s, dsts);
    for (Value d : dsts)
      addEntry(d, srcs);
  }
  for (auto of : objectFifos) {
    Value prod = of.getProducerTile();
    auto cons = of.getConsumerTiles();
    SmallVector<Value> consVec(cons.begin(), cons.end());
    addEntry(prod, consVec);
    for (Value c : consVec)
      addEntry(c, {prod});
  }
  return idx;
}

std::optional<int>
SequentialPlacer::computeCentroidColumn(LogicalTileOp logicalTile,
                                        const FlowMembership &flowIndex) {
  // Per-flow routing cost as a function of the LTO's column S:
  //   single-dest flow to col D : cost(S) = |S - D|
  //   N-dest flow spanning [lo, hi]:
  //                cost(S) = 0       if lo <= S <= hi
  //                        = lo - S  if S < lo
  //                        = S - hi  if S > hi
  // Inside a broadcast's span the cost is flat -- broadcasts constrain
  // the LTO to stay within their reach but don't pull it to any column.
  // Tiebreak by closest-to-mean-of-dests so a pure broadcast lands mid
  // and two single-dest flows average to their midpoint.
  Value ltoVal = logicalTile.getResult();
  auto ltoIt = flowIndex.ltoFlows.find(ltoVal);
  if (ltoIt == flowIndex.ltoFlows.end())
    return std::nullopt;

  // Column of a flow endpoint, or nullopt if not yet fixed. A memtile that
  // already knows its column answers here; looking past it to its cores
  // would pick up the other flows it carries.
  auto resolvePeerCol = [&](Value v) -> std::optional<int> {
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return std::nullopt;
    // A physical tile's column is fixed by construction, whatever its type.
    if (auto tileOp = dyn_cast<TileOp>(defOp))
      return tileOp.getCol();
    auto lto = dyn_cast<LogicalTileOp>(defOp);
    if (!lto)
      return std::nullopt;
    if (auto c = lto.tryGetCol())
      return c;
    auto it = result.find(defOp);
    if (it != result.end())
      return it->second.col;
    return std::nullopt;
  };

  // One level of indirection for shim->memtile->core, used only when the
  // memtile's own column is still unknown. Multi-level memtile chains aren't
  // currently produced by mlir-aie.
  auto resolveLtoCoreCols = [&](Value v) {
    SmallVector<int> out;
    auto it = flowIndex.ltoFlows.find(v);
    if (it == flowIndex.ltoFlows.end())
      return out;
    for (auto &peerList : it->second)
      for (Value p : peerList)
        if (auto col = resolvePeerCol(p))
          out.push_back(*col);
    return out;
  };

  SmallVector<SmallVector<int>> perFlowDests;
  for (auto &peers : ltoIt->second) {
    SmallVector<int> dests;
    for (Value p : peers) {
      if (auto col = resolvePeerCol(p)) {
        dests.push_back(*col);
        continue;
      }
      if (p == ltoVal)
        continue;
      for (int c : resolveLtoCoreCols(p))
        dests.push_back(c);
    }
    if (!dests.empty())
      perFlowDests.push_back(std::move(dests));
  }

  if (perFlowDests.empty())
    return std::nullopt;

  double meanDestCol = 0.0;
  int destCount = 0;
  for (auto &dests : perFlowDests)
    for (int d : dests) {
      meanDestCol += d;
      ++destCount;
    }
  meanDestCol /= destCount;

  int numCols = targetModel->columns();
  int bestCol = 0;
  int bestCost = std::numeric_limits<int>::max();
  double bestTiebreak = std::numeric_limits<double>::infinity();
  for (int S = 0; S < numCols; ++S) {
    int cost = 0;
    for (auto &dests : perFlowDests) {
      if (dests.size() == 1) {
        cost += std::abs(S - dests[0]);
      } else {
        int lo = *llvm::min_element(dests);
        int hi = *llvm::max_element(dests);
        if (S < lo)
          cost += lo - S;
        else if (S > hi)
          cost += S - hi;
      }
    }
    double tiebreak = std::abs(S - meanDestCol);
    if (cost < bestCost || (cost == bestCost && tiebreak < bestTiebreak)) {
      bestCost = cost;
      bestTiebreak = tiebreak;
      bestCol = S;
    }
  }
  return bestCol;
}

LogicalResult SequentialPlacer::placeNonCoreTileByCentroid(
    LogicalTileOp logicalTile, const FlowMembership &flowIndex,
    const llvm::DenseMap<Operation *, std::pair<int, int>>
        &channelRequirements) {
  std::optional<int> centroidCol =
      computeCentroidColumn(logicalTile, flowIndex);
  auto colConstraint = logicalTile.tryGetCol();

  // With spread-unanchored-tiles off, an LTO with no CoreTile peer targets
  // column 0, which ranks every candidate at the same distance and so pins the
  // whole design to one column until it fills up.
  std::optional<int> targetCol = colConstraint ? colConstraint : centroidCol;
  if (!targetCol && !spreadUnanchoredTiles)
    targetCol = 0;

  auto [numInputChannels, numOutputChannels] =
      channelRequirements.lookup(logicalTile.getOperation());

  auto maybeTile = findTileWithCapacity(targetCol, availability.nonCompTiles,
                                        numInputChannels, numOutputChannels,
                                        logicalTile.getTileType(), logicalTile,
                                        /*withinBudgets=*/true);
  if (!maybeTile)
    maybeTile = findTileWithCapacity(targetCol, availability.nonCompTiles,
                                     numInputChannels, numOutputChannels,
                                     logicalTile.getTileType(), logicalTile,
                                     /*withinBudgets=*/false);
  // Exhausted shim tiles are no longer in availability, so look at them all.
  if (!maybeTile && mergeLogicalTiles &&
      logicalTile.getTileType() == AIETileType::ShimNOCTile) {
    std::optional<std::pair<int, int>> bestKey;
    for (int col = 0, numCols = targetModel->columns(); col < numCols; ++col) {
      TileID tile{col, 0};
      std::optional<std::pair<int, int>> held =
          pinsAt.heldAt(tile, pinnedEnds.lookup(logicalTile));
      if ((colConstraint && col != *colConstraint) || !held ||
          namedClash(logicalTile, overlayTilesAt.lookup(tile)) ||
          !fitsSharingShim(tile, logicalTile,
                           {numInputChannels - held->first,
                            numOutputChannels - held->second}))
        continue;
      std::pair<int, int> key{targetCol ? std::abs(col - *targetCol) : 0, col};
      if (!bestKey || key < *bestKey) {
        bestKey = key;
        maybeTile = tile;
      }
    }
  }
  if (!maybeTile) {
    if (std::optional<TileID> named = findTileWithCapacity(
            targetCol, availability.nonCompTiles, numInputChannels,
            numOutputChannels, logicalTile.getTileType(), logicalTile,
            /*withinBudgets=*/false, /*apartByName=*/false))
      if (auto clash = namedClash(logicalTile, overlayTilesAt.lookup(*named)))
        return emitNamedClash(logicalTile, *named, *clash);
    AIETileType requestedType = logicalTile.getTileType();
    StringRef tileTypeName = stringifyAIETileType(requestedType);
    auto diag = logicalTile.emitError();
    CapacityDiagnosis diagnosis = diagnoseCapacityExhaustion(
        requestedType, numInputChannels, numOutputChannels, logicalTile);
    auto noteClash = [&]() {
      if (!diagnosis.clash)
        return;
      auto &[at, end, other] = *diagnosis.clash;
      std::string why = pinnedRule.whyApart(end, other).value_or("they clash");
      diag.attachNote() << "@" << end.fifo.getSymName() << " pins "
                        << (end.sends ? "MM2S" : "S2MM") << " channel "
                        << end.channel << ", which @" << other.fifo.getSymName()
                        << " pins on (" << at.col << ", " << at.row
                        << ") too, but " << why << "; " << diagnosis.clashTiles
                        << " " << tileTypeName << "(s) hold such a pin";
    };
    if (!diagnosis.capacityExistsSomewhere && diagnosis.clash) {
      diag << "no " << tileTypeName
           << " with DMA channels to spare can hold this tile's pinned ends "
              "beside the ends pinned there";
      noteClash();
    } else if (!diagnosis.capacityExistsSomewhere) {
      diag << "no " << tileTypeName << " on the device has " << numInputChannels
           << " input/" << numOutputChannels
           << " output DMA channel(s) free: all " << diagnosis.tilesOfType
           << " " << tileTypeName << "(s) are at " << diagnosis.inUsed << "/"
           << diagnosis.inMax << " input, " << diagnosis.outUsed << "/"
           << diagnosis.outMax << " output channels used";
      diag.attachNote() << "this is the device's total " << tileTypeName
                        << " DMA budget, not a placement choice -- no column "
                           "has spare capacity to pin to";
      if (diagnosis.overlayOut)
        diag.attachNote()
            << diagnosis.overlayOut
            << " of the output channels used are the control overlay's, "
               "which aie-generate-column-control-overlay shares only with "
               "ends never in flight with its control packets";
      if (diagnosis.overlayUnfit)
        diag.attachNote() << "on " << diagnosis.overlayUnfit << " "
                          << tileTypeName
                          << "(s), with this tile's pinned ends, the control "
                             "overlay has no output channel it may take or "
                             "share";
      int shortIn = diagnosis.closestShortIn;
      int shortOut = diagnosis.closestShortOut;
      if (shortIn == 0 && shortOut == 0)
        return failure();
      auto &remedy = diag.attachNote();
      remedy << "the closest " << tileTypeName << " is ";
      if (shortIn > 0 && shortOut > 0)
        remedy << "short " << shortIn << " input/" << shortOut
               << " output channel(s); ";
      else if (shortIn > 0)
        remedy << "short " << shortIn << " input channel(s); ";
      else if (shortOut > 0)
        remedy << "short " << shortOut << " output channel(s); ";
      remedy << "fan traffic through a MemTile via aie.objectfifo.link, or "
                "reduce the number of DMA-attached ObjectFIFOs feeding this "
                "tile";
    } else {
      diag << "no " << tileTypeName << " has sufficient DMA capacity for "
           << numInputChannels << " input/" << numOutputChannels
           << " output channels ";
      if (colConstraint)
        diag << "at column " << *colConstraint;
      else if (centroidCol)
        diag << "near centroid column " << *centroidCol;
      else
        diag << "on any column";
      noteClash();
      if (!mergeLogicalTiles)
        diag.attachNote() << "every " << tileTypeName
                          << " with spare DMA capacity already hosts a "
                             "different logical tile (merge-logical-tiles is "
                             "off); enable merging, or pin logical tiles "
                             "one-per-column if fewer of them than physical "
                          << tileTypeName << "s remain unclaimed";
    }
    return failure();
  }

  result[logicalTile] = *maybeTile;
  LLVM_DEBUG({
    auto showCol = [](std::optional<int> c) {
      return c ? std::to_string(*c) : std::string("none");
    };
    llvm::dbgs() << DEBUG_TYPE << ": placed non-core LTO "
                 << logicalTile.getLoc() << " at (" << maybeTile->col << ", "
                 << maybeTile->row << ") (centroid col=" << showCol(centroidCol)
                 << ", target col=" << showCol(targetCol) << ")\n";
  });
  if (!mergeLogicalTiles)
    assignedNonCoreTiles.insert(*maybeTile);
  std::optional<std::pair<int, int>> held =
      pinsAt.heldAt(*maybeTile, pinnedEnds.lookup(logicalTile));
  if (!held)
    return logicalTile.emitError()
           << "pins a channel that (" << maybeTile->col << ", "
           << maybeTile->row << ") cannot hold beside its other pinned ends";
  auto [heldIn, heldOut] = *held;
  recordStreamAndBdUsage(logicalTile, *maybeTile);
  if (numInputChannels > heldIn)
    updateChannelUsage(*maybeTile, DmaDir::In, numInputChannels - heldIn);
  if (numOutputChannels > heldOut)
    updateChannelUsage(*maybeTile, DmaDir::Out, numOutputChannels - heldOut);
  recordShimEnds(*maybeTile, logicalTile);
  return success();
}

bool SequentialPlacer::fitsSharingShim(TileID tile, Operation *logicalTile,
                                       std::pair<int, int> required) {
  if (targetModel->getTileType(tile.col, tile.row) !=
          AIETileType::ShimNOCTile ||
      !shimSpans)
    return false;
  const ShimTransferSpans &spans = *shimSpans;
  auto [maxIn, maxOut] = getDMACapacity(*targetModel, tile);
  OverlayFit overlay = overlayWith(tile, logicalTile);
  if (!overlay.fits)
    return false;
  int used[2] = {availability.inputChannelsUsed[tile],
                 availability.outputChannelsUsed[tile] + overlay.channels};
  int need[2] = {required.first, required.second};
  int max[2] = {maxIn, maxOut};
  SmallVector<Operation *> tilesHere(shimTilesAt.lookup(tile));
  if (shareableShimEnds.contains(logicalTile))
    tilesHere.push_back(logicalTile);
  for (int dir : {0, 1}) {
    if (used[dir] + need[dir] <= max[dir])
      continue;
    SmallVector<const ShimSide *> sides;
    for (Operation *op : tilesHere)
      sides.push_back(&shareableShimEnds.find(op)->second[dir]);
    SmallVector<StringAttr> extra;
    if (dir == 1 && overlay.own)
      extra.push_back(spans.overlayEnd(tile.col));
    SmallVector<int> pinnedHere = pinsAt.channelsAt(tile, dir == 1);
    for (const PinnedChannels::End &end : pinnedEnds.lookup(logicalTile))
      if (end.sends == (dir == 1))
        pinnedHere.push_back(end.channel);
    if (used[dir] + need[dir] -
            shimChannelsSaved(spans, sides, extra, pinnedHere) >
        max[dir])
      return false;
  }
  return true;
}

void SequentialPlacer::recordShimEnds(TileID tile, Operation *logicalTile) {
  if (shareableShimEnds.contains(logicalTile))
    shimTilesAt[tile].push_back(logicalTile);
}

std::optional<TileID> SequentialPlacer::findTileWithCapacity(
    std::optional<int> targetCol, llvm::ArrayRef<TileID> tiles,
    int requiredInputChannels, int requiredOutputChannels,
    AIETileType requestedType, Operation *logicalTile, bool withinBudgets,
    bool apartByName) {
  // Choose a physical tile by lexicographic minimum of
  //   (|col - targetCol| + broadcast pressure, current load, col, row).
  // A memtile's extra broadcast destinations penalize sharing its bandwidth,
  // even when an idle tile is slightly farther from the centroid. Without
  // broadcasts, and for shims or explicit column hints, distance still leads.
  // (col, row) is the final tiebreaker for determinism.
  //
  // Without a targetCol there is no routing anchor to measure against, so the
  // distance term is dropped and load leads. Load is then counted only in the
  // directions this LTO actually needs, matching hasAvailableChannels, which
  // already filters per direction: summing both would push a chain's read and
  // write shims onto different columns and need 2N columns for N chains.
  std::optional<TileID> best;
  std::tuple<int, int, int, int> bestKey;
  SmallVector<PinnedChannels::End> pins = pinnedEnds.lookup(logicalTile);

  for (auto &tile : tiles) {
    if (targetModel->getTileType(tile.col, tile.row) != requestedType)
      continue;
    std::optional<std::pair<int, int>> held = pinsAt.heldAt(tile, pins);
    if (!held || !hasAvailableChannels(
                     tile, requiredInputChannels - held->first,
                     requiredOutputChannels - held->second, logicalTile))
      continue;
    // When merge-logical-tiles is disabled, a tile that already hosts a
    // non-core aie.logical_tile is off-limits even if it has spare DMA
    // capacity.
    if (!mergeLogicalTiles && assignedNonCoreTiles.contains(tile))
      continue;
    if (withinBudgets && !fitsStreamAndBdBudgets(logicalTile, tile))
      continue;
    if (apartByName &&
        namedClash(logicalTile, overlayTilesAt.lookup(tile)).has_value())
      continue;
    int dist = targetCol ? std::abs(tile.col - *targetCol) : 0;
    if (requestedType == AIETileType::MemTile &&
        !cast<LogicalTileOp>(logicalTile).tryGetCol())
      dist += broadcastFanoutUsed.lookup(tile);
    int load;
    if (targetCol ||
        (requiredInputChannels == 0 && requiredOutputChannels == 0)) {
      load = availability.inputChannelsUsed[tile] +
             availability.outputChannelsUsed[tile];
    } else {
      load = 0;
      if (requiredInputChannels > 0)
        load += availability.inputChannelsUsed[tile];
      if (requiredOutputChannels > 0)
        load += availability.outputChannelsUsed[tile];
    }
    std::tuple<int, int, int, int> key{dist, load, tile.col, tile.row};
    if (!best || key < bestKey) {
      best = tile;
      bestKey = key;
    }
  }

  return best;
}

SequentialPlacer::CapacityDiagnosis
SequentialPlacer::diagnoseCapacityExhaustion(AIETileType requestedType,
                                             int requiredInputChannels,
                                             int requiredOutputChannels,
                                             Operation *joining) {
  // Walk the device directly rather than `availability.nonCompTiles`: a
  // fully-exhausted tile is meant to stay enumerable here even if a future
  // fix makes `updateChannelUsage`'s prune (see its `removeTile` call)
  // actually fire, which today it structurally cannot.
  CapacityDiagnosis diagnosis;
  for (int col = 0, numCols = targetModel->columns(); col < numCols; ++col) {
    for (int row = 0, numRows = targetModel->rows(); row < numRows; ++row) {
      if (targetModel->getTileType(col, row) != requestedType)
        continue;
      TileID tile{col, row};
      ++diagnosis.tilesOfType;
      auto [maxIn, maxOut] = getDMACapacity(*targetModel, tile);
      diagnosis.inMax += maxIn;
      diagnosis.outMax += maxOut;
      int usedIn = availability.inputChannelsUsed.lookup(tile);
      OverlayFit overlay = overlayWith(tile, joining);
      diagnosis.overlayOut += overlay.channels;
      diagnosis.overlayUnfit += !overlay.fits;
      int usedOut =
          availability.outputChannelsUsed.lookup(tile) + overlay.channels;
      diagnosis.inUsed += usedIn;
      diagnosis.outUsed += usedOut;
      int shortIn = std::max(0, requiredInputChannels - (maxIn - usedIn));
      int shortOut = std::max(0, requiredOutputChannels - (maxOut - usedOut));
      if (diagnosis.tilesOfType == 1 ||
          shortIn + shortOut <
              diagnosis.closestShortIn + diagnosis.closestShortOut) {
        diagnosis.closestShortIn = shortIn;
        diagnosis.closestShortOut = shortOut;
      }
      SmallVector<PinnedChannels::End> pins = pinnedEnds.lookup(joining);
      std::optional<std::pair<int, int>> held = pinsAt.heldAt(tile, pins);
      if (!held) {
        ++diagnosis.clashTiles;
        for (const PinnedChannels::End &end : pins)
          if (std::optional<PinnedChannels::End> other =
                  pinsAt.clashWith(tile, end);
              other && !diagnosis.clash)
            diagnosis.clash = std::make_tuple(tile, end, *other);
        continue;
      }
      if (hasAvailableChannels(tile, requiredInputChannels - held->first,
                               requiredOutputChannels - held->second, joining))
        diagnosis.capacityExistsSomewhere = true;
    }
  }
  return diagnosis;
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
                                            int outputChannels,
                                            Operation *joining) {
  auto [maxIn, maxOut] = getDMACapacity(*targetModel, tile);
  OverlayFit overlay = overlayWith(tile, joining);

  int currentIn = availability.inputChannelsUsed[tile];
  int currentOut = availability.outputChannelsUsed[tile] + overlay.channels;

  return overlay.fits && (currentIn + inputChannels <= maxIn) &&
         (currentOut + outputChannels <= maxOut);
}

Placer::OverlayFit SequentialPlacer::overlayWith(TileID tile,
                                                 Operation *joining) {
  if (!overlayChannels(tile))
    return {};
  SmallVector<Operation *> tilesHere(overlayTilesAt.lookup(tile));
  SmallVector<PinnedChannels::End> pins;
  if (joining) {
    tilesHere.push_back(joining);
    pins = pinnedEnds.lookup(joining);
  }
  return overlayAt(tile, pinsAt, pins, tilesHere,
                   shimSpans ? &*shimSpans : nullptr);
}

void SequentialPlacer::recordColumnStreamUsage(Operation *tile, int col) {
  auto &incoming = columnIncomingFifos[col];
  for (Operation *fifo : incomingFifos.lookup(tile))
    incoming.insert(fifo);
  auto &used = columnStreamsUsed[col];
  used += streamDemand.lookup(tile);
  used.southIn = incoming.size();
}

void SequentialPlacer::buildStreamDemand(
    ArrayRef<ObjectFifoCreateOp> objectFifos) {
  streamDemand.clear();
  columnStreamsUsed.clear();
  incomingFifos.clear();
  columnIncomingFifos.clear();
  broadcastFanout.clear();
  broadcastFanoutUsed.clear();
  // Rows rise from shims through memtiles to cores, so a stream to a higher
  // rank leaves through the North port and arrives through the South port.
  auto rank = [](Operation *op) -> std::optional<int> {
    auto tile = dyn_cast_or_null<TileLike>(op);
    if (!tile)
      return std::nullopt;
    if (tile.isShimTile())
      return 0;
    return tile.isMemTile() ? 1 : 2;
  };
  for (auto of : objectFifos) {
    Operation *producer = of.getProducerTile().getDefiningOp();
    std::optional<int> producerRank = rank(producer);
    if (!producerRank)
      continue;
    if (*producerRank == 1)
      broadcastFanout[producer] +=
          std::max(0, static_cast<int>(of.getConsumerTiles().size()) - 1);
    bool north = false, south = false;
    for (Value consumerTile : of.getConsumerTiles()) {
      Operation *consumer = consumerTile.getDefiningOp();
      std::optional<int> consumerRank = rank(consumer);
      if (!consumerRank || *consumerRank == *producerRank)
        continue;
      if (*consumerRank > *producerRank) {
        north = true;
        streamDemand[consumer].southIn++;
        if (*consumerRank == 2)
          incomingFifos[consumer].push_back(of.getOperation());
      } else {
        south = true;
        streamDemand[consumer].northIn++;
      }
    }
    if (north)
      streamDemand[producer].northOut++;
    if (south)
      streamDemand[producer].southOut++;
  }
  for (auto &[op, demand] : streamDemand)
    if (auto tile = dyn_cast<TileOp>(op))
      if (tile.isCoreTile())
        recordColumnStreamUsage(op, tile.getCol());
  for (auto &[op, fanout] : broadcastFanout)
    if (auto tile = dyn_cast<TileOp>(op))
      broadcastFanoutUsed[{tile.getCol(), tile.getRow()}] += fanout;
}

Placer::RuntimeBdIds::RuntimeBdIds(DeviceOp device,
                                   const AIETargetModel &targetModel)
    : targetModel(&targetModel) {
  auto shimOf = [](Operation *tile) -> Operation * {
    auto like = dyn_cast_or_null<TileLike>(tile);
    return like && like.isShimNOCTile() ? tile : nullptr;
  };
  for (auto program : device.getOps<ShimDMAOp>())
    if (Operation *shim = shimOf(program.getTile().getDefiningOp())) {
      StaticBds &bds = statics[shim];
      program.walk([&](DMABDOp bd) {
        if (std::optional<int32_t> id = bd.getBdId())
          bds.pinned.push_back(*id);
        else
          ++bds.unpinned;
      });
    }
  // How a descriptor splits depends on the tile type, not on its column.
  std::optional<int> shimCol;
  for (int col = 0; col < targetModel.columns() && !shimCol; ++col)
    if (targetModel.getTileType(col, 0) == AIETileType::ShimNOCTile)
      shimCol = col;

  SymbolTable symbols(device);
  device.walk([&](RuntimeSequenceOp seq) {
    auto &events = sequences.emplace_back();
    DenseMap<Value, unsigned> taskOf;
    auto configure = [&](Operation *shim, Value result, Region &body) {
      Task task{shim, {}, AIEX::DMAStartTaskOp::headBdIdAlignment(result)};
      body.walk([&](DMABDOp bd) {
        std::optional<uint32_t> pinned;
        if (std::optional<int32_t> id = bd.getBdId())
          pinned = static_cast<uint32_t>(*id);
        auto plan = AIEX::decomposeTaskBdPattern(bd, targetModel, *shimCol, 0);
        size_t pieces =
            succeeded(plan) && !plan->pieces.empty() ? plan->pieces.size() : 1;
        // A pinned id stays with the first piece.
        task.bdIds.push_back(pinned);
        task.bdIds.append(pieces - 1, std::nullopt);
      });
      taskOf[result] = tasks.size();
      events.push_back({tasks.size(), true});
      tasks.push_back(std::move(task));
    };
    seq.walk([&](Operation *op) {
      if (!shimCol)
        return;
      if (auto cfg = dyn_cast<AIEX::DMAConfigureTaskLike>(op)) {
        if (Operation *shim = shimOf(AIEX::getRuntimeDmaTile(op, symbols)))
          configure(shim, op->getResult(0), cfg.getBody());
        return;
      }
      if (auto copy = dyn_cast<AIEX::NpuDmaMemcpyNdOp>(op)) {
        if (Operation *shim = shimOf(AIEX::getRuntimeDmaTile(op, symbols))) {
          events.push_back({tasks.size(), true});
          tasks.push_back(
              {shim, {static_cast<uint32_t>(copy.getId())}, 1, true});
        }
        return;
      }
      Value released;
      if (auto await = dyn_cast<AIEX::DMAAwaitTaskOp>(op))
        released = await.getTask();
      else if (auto free = dyn_cast<AIEX::DMAFreeTaskOp>(op))
        released = free.getTask();
      if (auto it = taskOf.find(released); it != taskOf.end()) {
        events.push_back({it->second, false});
        taskOf.erase(it);
      }
    });
  });

  for (auto &[shim, bds] : statics) {
    Peak &peak = peaks[shim];
    peak.live = bds.pinned.size() + bds.unpinned;
    peak.constrained |= !bds.pinned.empty();
  }
  DenseMap<Operation *, int> mostHeld;
  for (const auto &events : sequences) {
    DenseMap<Operation *, int> held;
    for (auto [index, configured] : events) {
      const Task &task = tasks[index];
      int bds = static_cast<int>(task.bdIds.size());
      int &now = held[task.owner];
      now += configured ? bds : -bds;
      int &most = mostHeld[task.owner];
      most = std::max(most, now);
    }
  }
  for (const Task &task : tasks)
    peaks[task.owner].constrained |=
        task.headAlignment > 1 ||
        llvm::any_of(task.bdIds, [](auto id) { return id.has_value(); });
  for (auto &[shim, most] : mostHeld)
    peaks[shim].live += most;
  for (auto &[shim, peak] : peaks)
    if (auto tile = dyn_cast<TileOp>(shim))
      placed[{tile.getCol(), tile.getRow()}] = shim;
}

int Placer::RuntimeBdIds::shortfall(ArrayRef<Operation *> onTile,
                                    TileID tile) const {
  std::vector<Operation *> key;
  int live = 0;
  bool constrained = false;
  auto add = [&](Operation *shim) {
    auto peak = peaks.find(shim);
    if (peak == peaks.end() || llvm::is_contained(key, shim))
      return;
    key.push_back(shim);
    live += peak->second.live;
    constrained |= peak->second.constrained;
  };
  for (Operation *shim : onTile)
    add(shim);
  if (Operation *shim = placed.lookup(tile))
    add(shim);
  // Lowest-free ids never fragment, so peaks that fit together always do.
  if (!constrained &&
      live <= static_cast<int>(targetModel->getNumBDs(tile.col, tile.row)))
    return 0;
  llvm::sort(key);
  auto [cached, added] = shortfallOf.try_emplace(key, 0);
  if (!added)
    return cached->second;
  int &missing = cached->second;

  // aie-assign-bd-ids: runtime-pinned ids are reserved, then static BDs take
  // theirs.
  BdIdGenerator assigned(tile.col, tile.row, *targetModel);
  for (const Task &task : tasks)
    if (llvm::binary_search(key, task.owner))
      for (std::optional<uint32_t> pinned : task.bdIds)
        if (pinned && !assigned.bdIdAlreadyAssigned(*pinned))
          assigned.assignBdId(*pinned);
  SmallVector<uint32_t> staticIds;
  for (Operation *shim : key) {
    StaticBds bds = statics.lookup(shim);
    for (uint32_t id : bds.pinned) {
      if (assigned.bdIdAlreadyAssigned(id)) {
        ++missing;
        continue;
      }
      assigned.assignBdId(id);
      staticIds.push_back(id);
    }
  }
  for (Operation *shim : key)
    for (int i = 0, e = statics.lookup(shim).unpinned; i < e; ++i) {
      if (std::optional<uint32_t> id = assigned.nextBdId(0))
        staticIds.push_back(*id);
      else
        ++missing;
    }

  for (const auto &events : sequences) {
    BdIdGenerator ids(tile.col, tile.row, *targetModel);
    for (uint32_t id : staticIds)
      ids.assignBdId(id);
    DenseMap<unsigned, SmallVector<uint32_t>> held;
    DenseSet<uint32_t> copied;
    for (auto [index, configured] : events) {
      const Task &task = tasks[index];
      if (!llvm::binary_search(key, task.owner))
        continue;
      if (task.copy) {
        // Copies reprogram their id at once, so they share it.
        uint32_t id = *task.bdIds.front();
        if (!copied.insert(id).second)
          continue;
        if (ids.bdIdAlreadyAssigned(id))
          ++missing;
        else
          ids.assignBdId(id);
        continue;
      }
      if (!configured) {
        for (uint32_t id : held.lookup(index))
          ids.freeBdId(id);
        held.erase(index);
        continue;
      }
      SmallVector<uint32_t> &taken = held[index];
      for (std::optional<uint32_t> pinned : task.bdIds) {
        if (!pinned)
          continue;
        if (ids.bdIdAlreadyAssigned(*pinned)) {
          ++missing;
          continue;
        }
        ids.assignBdId(*pinned);
        taken.push_back(*pinned);
      }
      for (auto [i, pinned] : llvm::enumerate(task.bdIds)) {
        if (pinned)
          continue;
        if (std::optional<uint32_t> id =
                ids.nextBdId(0, i == 0 ? task.headAlignment : 1))
          taken.push_back(*id);
        else
          ++missing;
      }
    }
  }
  return missing;
}

bool SequentialPlacer::fitsStreamAndBdBudgets(Operation *logicalTile,
                                              TileID tile) const {
  // Only the ports this LTO adds to: a tile a fallback already overfilled
  // stays open to LTOs that use none of its streams.
  StreamDemand demand = streamDemand.lookup(logicalTile);
  StreamDemand streams = availability.streamsUsed.lookup(tile);
  streams += demand;
  const AIETargetModel &tm = *targetModel;
  if ((demand.northIn &&
       streams.northIn > static_cast<int>(tm.getNumSourceSwitchboxConnections(
                             tile.col, tile.row, WireBundle::North))) ||
      (demand.southIn &&
       streams.southIn > static_cast<int>(tm.getNumSourceSwitchboxConnections(
                             tile.col, tile.row, WireBundle::South))) ||
      (demand.northOut &&
       streams.northOut > static_cast<int>(tm.getNumDestSwitchboxConnections(
                              tile.col, tile.row, WireBundle::North))) ||
      (demand.southOut &&
       streams.southOut > static_cast<int>(tm.getNumDestSwitchboxConnections(
                              tile.col, tile.row, WireBundle::South))))
    return false;

  if (!runtimeBdIds.uses(logicalTile))
    return true;
  // Only the ids this LTO adds to: a tile a fallback already overfilled stays
  // open to LTOs whose tasks take none.
  SmallVector<Operation *> onTile = availability.runtimeBdTiles.lookup(tile);
  int before = runtimeBdIds.shortfall(onTile, tile);
  onTile.push_back(logicalTile);
  return runtimeBdIds.shortfall(onTile, tile) == before;
}

void SequentialPlacer::recordStreamAndBdUsage(Operation *logicalTile,
                                              TileID tile) {
  broadcastFanoutUsed[tile] += broadcastFanout.lookup(logicalTile);
  overlayTilesAt[tile].push_back(logicalTile);
  for (const PinnedChannels::End &end : pinnedEnds.lookup(logicalTile))
    pinsAt.place(tile, end, 1);
  for (const DMAChannelAnalysis::ChannelClaim &claim :
       claimsOf.lookup(logicalTile))
    pinsAt.claim(tile, claim, 1);
  if (auto it = streamDemand.find(logicalTile); it != streamDemand.end())
    availability.streamsUsed[tile] += it->second;
  if (runtimeBdIds.uses(logicalTile))
    availability.runtimeBdTiles[tile].push_back(logicalTile);
}

void TileAvailability::removeTile(TileID tile, AIETileType type) {
  auto removeFromVector = [&](std::vector<TileID> &vec) {
    vec.erase(std::remove(vec.begin(), vec.end(), tile), vec.end());
  };

  switch (type) {
  case AIETileType::CoreTile:
    removeFromVector(compTiles);
    removeFromVector(overBudgetCompTiles);
    compTilesSet.erase(tile);
    break;
  case AIETileType::MemTile:
  case AIETileType::ShimNOCTile:
  case AIETileType::ShimPLTile:
    removeFromVector(nonCompTiles);
    break;
  }
}
