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
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-placer"

void SequentialPlacer::initialize(const AIETargetModel &targetModel) {
  this->targetModel = &targetModel;

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

  // Limit cores per column if specified
  if (coresPerCol.has_value()) {
    // Calculate max cores per column in the device
    std::map<int, int> coresInColumn;
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
  std::map<int, std::vector<TileID>> tilesByColumn;

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

  // Replace availability.compTiles with limited list
  availability.compTiles = limitedTiles;
}

LogicalResult SequentialPlacer::place(DeviceOp device) {
  // Phase 0: Validate options
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
    if (auto lt = dyn_cast<LogicalTileOp>(op))
      logicalTiles.push_back(lt);
    if (auto of = dyn_cast<ObjectFifoCreateOp>(op))
      objectFifos.push_back(of);
    if (auto link = dyn_cast<ObjectFifoLinkOp>(op))
      objectFifoLinks.push_back(link);
    if (auto cf = dyn_cast<CascadeFlowOp>(op))
      cascadeFlows.push_back(cf);
    if (auto f = dyn_cast<FlowOp>(op))
      flows.push_back(f);
    if (auto pf = dyn_cast<PacketFlowOp>(op))
      pktFlows.push_back(pf);
  });

  // Phase 2a: Build placement constraints
  auto bufferAdjacency = buildBufferAdjacency(logicalTiles);

  // Phase 2b: Build channel requirements from ObjectFifo and Flow connectivity,
  // and cascade adjacency constraints from CascadeFlow connectivity.
  auto channelRequirements =
      buildChannelRequirements(objectFifos, objectFifoLinks);
  addChannelRequirementsFromFlows(flows, pktFlows, channelRequirements);

  auto cascadeAdjacency = buildCascadeAdjacency(cascadeFlows);

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

  // Phase 3: Place constrained tiles then compute tiles
  size_t nextCompIdx = 0;
  for (auto logicalTile : logicalTiles) {
    // Place fully constrained tiles at their specified coordinates
    auto col = logicalTile.tryGetCol();
    auto row = logicalTile.tryGetRow();
    if (col && row) {
      TileID tile{*col, *row};
      if (!satisfiesAdjacency(logicalTile, tile, bufferAdjacency, bufferPred)) {
        auto diag = logicalTile.emitError()
                    << "tile (" << tile.col << ", " << tile.row
                    << ") violates shared-L1 buffer adjacency";
        attachPeerNotes(diag, logicalTile, bufferAdjacency, bufferLabel);
        return failure();
      }
      if (!satisfiesAdjacency(logicalTile, tile, cascadeAdjacency,
                              cascadePred)) {
        auto diag = logicalTile.emitError()
                    << "tile (" << tile.col << ", " << tile.row
                    << ") violates cascade adjacency";
        attachPeerNotes(diag, logicalTile, cascadeAdjacency, cascadeLabel);
        return failure();
      }
      if (failed(validateAndUpdateChannelUsage(logicalTile, tile,
                                               channelRequirements, true)))
        return failure();

      result[logicalTile] = tile;
      // Only remove fully constrained compute tiles from availability.
      // Mem/Shim may still host additional logical tiles as long as
      // channel/DMA capacity permits.
      if (logicalTile.getTileType() == AIETileType::CoreTile)
        availability.removeTile(tile, logicalTile.getTileType());
      continue;
    }

    // Place compute tiles with partial constraint support
    if (logicalTile.getTileType() == AIETileType::CoreTile) {
      std::optional<TileID> placement = std::nullopt;
      bool sawConstraintMatch = false;
      bool allConstraintMatchesFailedAdjacency = true;

      for (size_t i = nextCompIdx; i < availability.compTiles.size(); ++i) {
        TileID candidate = availability.compTiles[i];

        // Check partial constraints
        if (col && candidate.col != *col)
          continue;
        if (row && candidate.row != *row)
          continue;
        sawConstraintMatch = true;
        if (!satisfiesAdjacency(logicalTile, candidate, bufferAdjacency,
                                bufferPred))
          continue;
        allConstraintMatchesFailedAdjacency = false;
        if (!satisfiesAdjacency(logicalTile, candidate, cascadeAdjacency,
                                cascadePred))
          continue;

        // Found valid tile - swap to nextCompIdx position and use
        std::swap(availability.compTiles[i],
                  availability.compTiles[nextCompIdx]);
        placement = availability.compTiles[nextCompIdx++];
        break;
      }

      if (!placement) {
        bool adjacencyWasCause =
            sawConstraintMatch && allConstraintMatchesFailedAdjacency &&
            bufferAdjacency.tileToEdges.count(logicalTile.getOperation());
        bool hasCascade =
            cascadeAdjacency.tileToEdges.count(logicalTile.getOperation());
        InFlightDiagnostic diag = logicalTile.emitError();
        if (col || row) {
          diag << "no compute tile available matching constraint ("
               << (col ? std::to_string(*col) : "?") << ", "
               << (row ? std::to_string(*row) : "?") << ")"
               << (adjacencyWasCause ? " and shared-L1 buffer adjacency" : "")
               << (hasCascade ? " and cascade adjacency" : "");
        } else {
          diag << "no available compute tiles for placement"
               << (adjacencyWasCause
                       ? " (shared-L1 buffer adjacency unsatisfiable)"
                       : "")
               << (hasCascade ? " (cascade adjacency unsatisfiable)" : "");
        }
        if (adjacencyWasCause)
          attachPeerNotes(diag, logicalTile, bufferAdjacency, bufferLabel);
        if (hasCascade)
          attachPeerNotes(diag, logicalTile, cascadeAdjacency, cascadeLabel);
        return failure();
      }

      if (failed(validateAndUpdateChannelUsage(logicalTile, *placement,
                                               channelRequirements, false)))
        return failure();

      result[logicalTile] = *placement;
    }

    if (logicalTile.getTileType() == AIETileType::ShimPLTile) {
      return logicalTile.emitError(
          "DMA channel-based SequentialPlacer does not support unplaced "
          "ShimPLTiles (no DMAs).");
    }
  }

  // Phase 4: Place each remaining non-core (mem/shim) LTO at the centroid
  // column of its placed core peers, reached transitively through any
  // connectivity adjacency.
  auto objectFifoAdjacency = buildObjectFifoAdjacency(objectFifos);
  auto flowAdjacency = buildFlowAdjacency(flows, pktFlows);
  SmallVector<const Adjacency *, 2> connectivityAdjacencies = {
      &objectFifoAdjacency, &flowAdjacency};

  for (auto logicalTile : logicalTiles) {
    if (result.count(logicalTile.getOperation()))
      continue;
    AIETileType tileType = logicalTile.getTileType();
    if (tileType == AIETileType::CoreTile ||
        tileType == AIETileType::ShimPLTile)
      continue;
    if (failed(placeNonCoreTileByCentroid(logicalTile, connectivityAdjacencies,
                                          channelRequirements)))
      return failure();
  }

  return success();
}

LogicalResult SequentialPlacer::validateAndUpdateChannelUsage(
    LogicalTileOp logicalTile, TileID tile,
    const llvm::DenseMap<Operation *, std::pair<int, int>> &channelRequirements,
    bool isConstrained) {

  // Get channel requirements
  auto it = channelRequirements.find(logicalTile.getOperation());
  int inChannels = 0, outChannels = 0;
  if (it != channelRequirements.end()) {
    inChannels = it->second.first;
    outChannels = it->second.second;
  }

  // Validate capacity
  if (!hasAvailableChannels(tile, inChannels, outChannels)) {
    // Get max channels
    int maxIn = logicalTile.getNumDestConnections(WireBundle::DMA);
    int maxOut = logicalTile.getNumSourceConnections(WireBundle::DMA);
    int availIn = maxIn - availability.inputChannelsUsed[tile];
    int availOut = maxOut - availability.outputChannelsUsed[tile];

    auto diag = logicalTile.emitError();
    if (isConstrained)
      diag << "tile (" << tile.col << ", " << tile.row << ") requires ";
    else
      diag << "tile requires ";
    diag << inChannels << " input/" << outChannels
         << " output DMA channels, but only " << availIn << " input/"
         << availOut << " output available";
    return failure();
  }

  // Update channel usage
  if (inChannels > 0)
    updateChannelUsage(tile, false, inChannels);
  if (outChannels > 0)
    updateChannelUsage(tile, true, outChannels);

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

  // Build set of ObjectFifos that are involved in links
  // These will be handled specially for channel counting
  llvm::DenseSet<llvm::StringRef> linkedFifoNames;

  for (auto linkOp : objectFifoLinks) {
    for (auto srcFifoAttr : linkOp.getFifoIns()) {
      auto srcFifoName = cast<FlatSymbolRefAttr>(srcFifoAttr).getValue();
      linkedFifoNames.insert(srcFifoName);
    }

    for (auto dstFifoAttr : linkOp.getFifoOuts()) {
      auto dstFifoName = cast<FlatSymbolRefAttr>(dstFifoAttr).getValue();
      linkedFifoNames.insert(dstFifoName);
    }
  }

  // Count channels for non-linked ObjectFifos normally
  for (auto ofOp : objectFifos) {
    // Skip linked fifos - they'll be handled separately
    if (linkedFifoNames.count(ofOp.getSymName()))
      continue;

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

      // This consumer needs a DMA channel
      if (consumerOp)
        channelRequirements[consumerOp].first++; // input++

      producerNeedsDMA = true;
    }

    // Producer needs ONE output channel if any consumer needs DMA
    if (producerNeedsDMA && producerOp)
      channelRequirements[producerOp].second++; // output++
  }

  // For linked ObjectFifos, count channels based on the link structure
  // Link tiles need channels for all sources and destinations
  for (auto linkOp : objectFifoLinks) {

    // Find the tile that fifos are linked on
    Operation *linkTileOp = nullptr;

    // Get link tile from first source fifo's consumer
    for (auto srcFifoAttr : linkOp.getFifoIns()) {
      auto srcFifoName = cast<FlatSymbolRefAttr>(srcFifoAttr).getValue();
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

bool SequentialPlacer::satisfiesAdjacency(
    LogicalTileOp logicalTile, TileID candidate, const Adjacency &adjacency,
    llvm::function_ref<bool(TileID firstPos, TileID secondPos)> pred) const {
  auto it = adjacency.tileToEdges.find(logicalTile.getOperation());
  if (it == adjacency.tileToEdges.end())
    return true;

  for (unsigned idx : it->second) {
    auto [edgeFirst, edgeSecond] = adjacency.edges[idx];
    bool thisIsFirst = edgeFirst.getOperation() == logicalTile.getOperation();
    TileLike peer = thisIsFirst ? edgeSecond : edgeFirst;
    auto peerPos = resolvePeerPosition(peer, result);
    if (!peerPos)
      continue;
    TileID firstPos = thisIsFirst ? candidate : *peerPos;
    TileID secondPos = thisIsFirst ? *peerPos : candidate;
    if (!pred(firstPos, secondPos))
      return false;
  }
  return true;
}

void SequentialPlacer::attachPeerNotes(
    InFlightDiagnostic &diag, LogicalTileOp logicalTile,
    const Adjacency &adjacency,
    llvm::function_ref<StringRef(bool thisIsFirst)> labelPeer) const {
  auto it = adjacency.tileToEdges.find(logicalTile.getOperation());
  if (it == adjacency.tileToEdges.end())
    return;
  for (unsigned idx : it->second) {
    auto [edgeFirst, edgeSecond] = adjacency.edges[idx];
    bool thisIsFirst = edgeFirst.getOperation() == logicalTile.getOperation();
    TileLike peer = thisIsFirst ? edgeSecond : edgeFirst;
    auto peerPos = resolvePeerPosition(peer, result);
    if (!peerPos)
      continue;
    diag.attachNote(peer.getLoc())
        << labelPeer(thisIsFirst) << " peer placed at (" << peerPos->col << ", "
        << peerPos->row << ")";
  }
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

LogicalResult SequentialPlacer::placeNonCoreTileByCentroid(
    LogicalTileOp logicalTile,
    ArrayRef<const Adjacency *> connectivityAdjacencies,
    const llvm::DenseMap<Operation *, std::pair<int, int>>
        &channelRequirements) {

  // BFS the connected component, summing placed-core columns. Walks
  // through unplaced LTO peers so the centroid sees cores reachable
  // transitively; TileOp peers terminate the walk.
  llvm::DenseSet<Operation *> visited;
  SmallVector<Operation *, 8> queue{logicalTile.getOperation()};
  visited.insert(logicalTile.getOperation());

  int sumCols = 0;
  int placedCoreCount = 0;
  while (!queue.empty()) {
    Operation *current = queue.pop_back_val();
    for (const Adjacency *adj : connectivityAdjacencies) {
      auto it = adj->tileToEdges.find(current);
      if (it == adj->tileToEdges.end())
        continue;
      for (unsigned idx : it->second) {
        auto [first, second] = adj->edges[idx];
        Operation *peerOp = (first.getOperation() == current)
                                ? second.getOperation()
                                : first.getOperation();
        if (!visited.insert(peerOp).second)
          continue;
        auto peerLT = dyn_cast<LogicalTileOp>(peerOp);
        if (!peerLT)
          continue; // TileOp peer: don't BFS through it.
        if (peerLT.getTileType() == AIETileType::CoreTile) {
          if (auto resIt = result.find(peerOp); resIt != result.end()) {
            sumCols += resIt->second.col;
            ++placedCoreCount;
          }
        }
        queue.push_back(peerOp);
      }
    }
  }

  auto colConstraint = logicalTile.tryGetCol();
  int centroidCol = placedCoreCount > 0
                        ? (sumCols + placedCoreCount / 2) / placedCoreCount
                        : 0;
  int targetCol = colConstraint ? *colConstraint : centroidCol;

  auto chanIt = channelRequirements.find(logicalTile.getOperation());
  int numInputChannels =
      chanIt != channelRequirements.end() ? chanIt->second.first : 0;
  int numOutputChannels =
      chanIt != channelRequirements.end() ? chanIt->second.second : 0;

  auto maybeTile = findTileWithCapacity(targetCol, availability.nonCompTiles,
                                        numInputChannels, numOutputChannels,
                                        logicalTile.getTileType());
  if (!maybeTile)
    return logicalTile.emitError()
           << "no " << stringifyAIETileType(logicalTile.getTileType())
           << " with sufficient DMA capacity";

  result[logicalTile] = *maybeTile;
  if (numInputChannels > 0)
    updateChannelUsage(*maybeTile, false, numInputChannels);
  if (numOutputChannels > 0)
    updateChannelUsage(*maybeTile, true, numOutputChannels);
  return success();
}

std::optional<TileID> SequentialPlacer::findTileWithCapacity(
    int targetCol, std::vector<TileID> &tiles, int requiredInputChannels,
    int requiredOutputChannels, AIETileType requestedType) {
  int maxCol = targetModel->columns();

  // Search columns rightward
  for (int offset = 0; offset < maxCol; ++offset) {
    int searchCol = targetCol + offset;
    if (searchCol >= maxCol)
      continue;

    for (auto &tile : tiles) {
      AIETileType tileType = targetModel->getTileType(tile.col, tile.row);
      if (tileType != requestedType)
        continue;

      if (tile.col == searchCol) {
        // Check if tile has capacity for both input and output channels
        if (hasAvailableChannels(tile, requiredInputChannels,
                                 requiredOutputChannels)) {
          return tile;
        }
      }
    }
  }

  return std::nullopt;
}

void SequentialPlacer::updateChannelUsage(TileID tile, bool isOutput,
                                          int numChannels) {
  if (isOutput) {
    availability.outputChannelsUsed[tile] += numChannels;
  } else {
    availability.inputChannelsUsed[tile] += numChannels;
  }

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
