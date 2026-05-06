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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
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
  SmallVector<FlowOp> flows;
  SmallVector<PacketFlowOp> pktFlows;

  device.walk([&](Operation *op) {
    if (auto lt = dyn_cast<LogicalTileOp>(op))
      logicalTiles.push_back(lt);
    if (auto of = dyn_cast<ObjectFifoCreateOp>(op))
      objectFifos.push_back(of);
    if (auto link = dyn_cast<ObjectFifoLinkOp>(op))
      objectFifoLinks.push_back(link);
    if (auto f = dyn_cast<FlowOp>(op))
      flows.push_back(f);
    if (auto pf = dyn_cast<PacketFlowOp>(op))
      pktFlows.push_back(pf);
  });

  // Phase 2a: Build placement constraints
  auto bufferAdjacency = buildBufferAdjacency(logicalTiles);

  // Phase 2b: Build channel requirements from ObjectFifo and Flow connectivity
  auto channelRequirements =
      buildChannelRequirements(objectFifos, objectFifoLinks);
  addChannelRequirementsFromFlows(flows, pktFlows, channelRequirements);

  // Phase 3: Place constrained tiles then compute tiles
  size_t nextCompIdx = 0;
  for (auto logicalTile : logicalTiles) {
    // Place fully constrained tiles at their specified coordinates
    auto col = logicalTile.tryGetCol();
    auto row = logicalTile.tryGetRow();
    if (col && row) {
      TileID tile{*col, *row};
      if (!satisfiesBufferAdjacency(logicalTile, tile, bufferAdjacency)) {
        auto diag = logicalTile.emitError()
                    << "tile (" << tile.col << ", " << tile.row
                    << ") violates shared-L1 buffer adjacency";
        attachBufferPeerNotes(diag, logicalTile, bufferAdjacency);
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
        if (!satisfiesBufferAdjacency(logicalTile, candidate, bufferAdjacency))
          continue;
        allConstraintMatchesFailedAdjacency = false;

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
        InFlightDiagnostic diag = logicalTile.emitError();
        if (col || row) {
          diag << "no compute tile available matching constraint ("
               << (col ? std::to_string(*col) : "?") << ", "
               << (row ? std::to_string(*row) : "?") << ")"
               << (adjacencyWasCause ? " and shared-L1 buffer adjacency" : "");
        } else {
          diag << "no available compute tiles for placement"
               << (adjacencyWasCause
                       ? " (shared-L1 buffer adjacency unsatisfiable)"
                       : "");
        }
        if (adjacencyWasCause)
          attachBufferPeerNotes(diag, logicalTile, bufferAdjacency);
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

  // Phase 4: Place mem/shim tiles by connectivity groups (ObjectFifo + Flow).
  // Vector iteration order is stable, so placement is reproducible when
  // multiple groups compete for the same physical tiles.
  SmallVector<ConnectivityGroup> groups;
  buildObjectFifoGroups(objectFifos, objectFifoLinks, groups);
  buildFlowGroups(flows, pktFlows, groups);

  for (auto &group : groups) {
    if (failed(placeNonCoreTilesInGroup(group, channelRequirements)))
      return failure();
  }

  // Phase 5: Fallback for remaining unplaced non-core tiles
  for (auto logicalTile : logicalTiles) {
    // Skip already placed tiles
    if (result.count(logicalTile.getOperation()))
      continue;

    // Skip core tiles (handled above) and ShimPLTile (unsupported)
    AIETileType tileType = logicalTile.getTileType();
    if (tileType == AIETileType::CoreTile ||
        tileType == AIETileType::ShimPLTile)
      continue;

    // Use column constraint if specified, otherwise start from column 0
    auto colConstraint = logicalTile.tryGetCol();
    int targetCol = colConstraint ? *colConstraint : 0;

    // Find first available tile of matching type (no DMA requirements)
    auto maybeTile = findTileWithCapacity(targetCol, availability.nonCompTiles,
                                          0, 0, tileType);

    if (!maybeTile)
      return logicalTile.emitError()
             << "no " << stringifyAIETileType(tileType) << " available";

    result[logicalTile] = *maybeTile;
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

// TODO: identical to the helper introduced by the cascade-adjacency PR
// (#3042). Hoist to a shared internal header once both have landed.
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

SequentialPlacer::BufferAdjacency
SequentialPlacer::buildBufferAdjacency(ArrayRef<LogicalTileOp> logicalTiles) {
  BufferAdjacency adjacency;
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
        unsigned idx = adjacency.edges.size();
        adjacency.edges.push_back({consumer, owner});
        adjacency.tileToEdges[consumer.getOperation()].push_back(idx);
        if (isa<LogicalTileOp>(owner.getOperation()))
          adjacency.tileToEdges[owner.getOperation()].push_back(idx);
      }
    });
  }
  return adjacency;
}

bool SequentialPlacer::satisfiesBufferAdjacency(
    LogicalTileOp logicalTile, TileID candidate,
    const BufferAdjacency &adjacency) const {
  auto it = adjacency.tileToEdges.find(logicalTile.getOperation());
  if (it == adjacency.tileToEdges.end())
    return true;

  for (unsigned idx : it->second) {
    auto [consumer, owner] = adjacency.edges[idx];
    bool thisIsConsumer = consumer.getOperation() == logicalTile.getOperation();
    TileLike peer = thisIsConsumer ? owner : TileLike(consumer);
    auto peerPos = resolvePeerPosition(peer, result);
    if (!peerPos)
      continue;
    TileID consumerPos = thisIsConsumer ? candidate : *peerPos;
    TileID ownerPos = thisIsConsumer ? *peerPos : candidate;
    if (!targetModel->isLegalMemAffinity(consumerPos.col, consumerPos.row,
                                         ownerPos.col, ownerPos.row))
      return false;
  }
  return true;
}

void SequentialPlacer::attachBufferPeerNotes(
    InFlightDiagnostic &diag, LogicalTileOp logicalTile,
    const BufferAdjacency &adjacency) const {
  auto it = adjacency.tileToEdges.find(logicalTile.getOperation());
  if (it == adjacency.tileToEdges.end())
    return;
  for (unsigned idx : it->second) {
    auto [consumer, owner] = adjacency.edges[idx];
    bool thisIsConsumer = consumer.getOperation() == logicalTile.getOperation();
    TileLike peer = thisIsConsumer ? owner : TileLike(consumer);
    auto peerPos = resolvePeerPosition(peer, result);
    if (!peerPos)
      continue;
    diag.attachNote(peer.getLoc())
        << "shared-L1 buffer " << (thisIsConsumer ? "owner" : "consumer")
        << " peer placed at (" << peerPos->col << ", " << peerPos->row << ")";
  }
}

void SequentialPlacer::buildObjectFifoGroups(
    ArrayRef<ObjectFifoCreateOp> objectFifos,
    ArrayRef<ObjectFifoLinkOp> objectFifoLinks,
    SmallVectorImpl<ConnectivityGroup> &groups) {

  // Linked fifos share a group ID. Unlinked fifos each get their own.
  llvm::StringMap<int> fifoToGroup;
  int nextGroupId = 0;
  for (auto linkOp : objectFifoLinks) {
    int groupId = nextGroupId++;
    for (auto fifoAttr : linkOp.getFifoIns())
      fifoToGroup[cast<FlatSymbolRefAttr>(fifoAttr).getValue()] = groupId;
    for (auto fifoAttr : linkOp.getFifoOuts())
      fifoToGroup[cast<FlatSymbolRefAttr>(fifoAttr).getValue()] = groupId;
  }

  // Map group ID -> index into `groups`. Insertion order = first-encountered
  // fifo order, which matches IR walk order.
  llvm::DenseMap<int, size_t> groupIdToIndex;

  auto getOrCreateGroup = [&](int groupId) -> ConnectivityGroup & {
    auto [it, inserted] = groupIdToIndex.try_emplace(groupId, groups.size());
    if (inserted)
      groups.emplace_back();
    return groups[it->second];
  };

  auto recordEndpoint = [&](Value tileVal, ConnectivityGroup &group) {
    auto lt = dyn_cast_or_null<LogicalTileOp>(tileVal.getDefiningOp());
    if (!lt)
      return;
    if (lt.getTileType() == AIETileType::CoreTile)
      group.coreTiles.push_back(lt);
    else
      group.nonCoreTiles.push_back(lt);
  };

  int unlinkedGroupId = nextGroupId;
  for (auto ofOp : objectFifos) {
    auto it = fifoToGroup.find(ofOp.getSymName());
    int groupId = (it != fifoToGroup.end()) ? it->second : unlinkedGroupId++;
    ConnectivityGroup &group = getOrCreateGroup(groupId);
    recordEndpoint(ofOp.getProducerTile(), group);
    for (Value consumerTile : ofOp.getConsumerTiles())
      recordEndpoint(consumerTile, group);
  }
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

// MapVector / SetVector preserve insertion order so that group identity and
// per-group tile order are stable across runs even when DMA capacity is tight.
void SequentialPlacer::buildFlowGroups(
    ArrayRef<FlowOp> flows, ArrayRef<PacketFlowOp> pktFlows,
    SmallVectorImpl<ConnectivityGroup> &groups) {

  llvm::MapVector<LogicalTileOp, llvm::SmallSetVector<LogicalTileOp, 4>> adj;

  auto addEdge = [&](Value srcVal, Value dstVal) {
    auto srcLT = dyn_cast_or_null<LogicalTileOp>(srcVal.getDefiningOp());
    auto dstLT = dyn_cast_or_null<LogicalTileOp>(dstVal.getDefiningOp());
    if (!srcLT || !dstLT)
      return;
    adj[srcLT].insert(dstLT);
    adj[dstLT].insert(srcLT);
  };

  for (auto flow : flows)
    addEdge(flow.getSource(), flow.getDest());

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
        addEdge(s, d);
  }

  llvm::DenseSet<LogicalTileOp> visited;
  for (auto &entry : adj) {
    LogicalTileOp seed = entry.first;
    if (visited.count(seed))
      continue;
    ConnectivityGroup &group = groups.emplace_back();
    SmallVector<LogicalTileOp> stack{seed};
    visited.insert(seed);
    while (!stack.empty()) {
      LogicalTileOp cur = stack.pop_back_val();
      if (cur.getTileType() == AIETileType::CoreTile)
        group.coreTiles.push_back(cur);
      else
        group.nonCoreTiles.push_back(cur);
      for (auto nbr : adj[cur]) {
        if (visited.insert(nbr).second)
          stack.push_back(nbr);
      }
    }
  }
}

LogicalResult SequentialPlacer::placeNonCoreTilesInGroup(
    const ConnectivityGroup &group,
    const llvm::DenseMap<Operation *, std::pair<int, int>>
        &channelRequirements) {

  // Common column = rounded average of placed core endpoints' columns.
  int sumCols = 0;
  int totalCoreEndpoints = 0;
  for (auto coreTile : group.coreTiles) {
    auto it = result.find(coreTile.getOperation());
    if (it == result.end())
      continue;
    sumCols += it->second.col;
    ++totalCoreEndpoints;
  }
  int groupCommonCol =
      totalCoreEndpoints > 0
          ? (sumCols + totalCoreEndpoints / 2) / totalCoreEndpoints
          : 0;

  for (auto logicalTile : group.nonCoreTiles) {
    if (result.count(logicalTile.getOperation()))
      continue;

    auto it = channelRequirements.find(logicalTile.getOperation());
    int numInputChannels =
        it != channelRequirements.end() ? it->second.first : 0;
    int numOutputChannels =
        it != channelRequirements.end() ? it->second.second : 0;

    auto colConstraint = logicalTile.tryGetCol();
    int targetCol = colConstraint ? *colConstraint : groupCommonCol;

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
  }
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
