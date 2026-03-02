//===- AIEPlacer.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-placer"

void SequentialPlacer::initialize(DeviceOp dev, const AIETargetModel &tm) {
  device = dev;
  targetModel = &tm;

  // Collect all available physical tiles from device
  for (int col = 0; col < tm.columns(); col++) {
    for (int row = 0; row < tm.rows(); row++) {
      TileID id = {col, row};
      AIETileType type = tm.getTileType(col, row);

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
}

LogicalResult SequentialPlacer::place(DeviceOp device,
                                      PlacementResult &result) {
  // Collect operations needed for placement
  SmallVector<LogicalTileOp> logicalTiles;
  SmallVector<ObjectFifoCreateOp> objectFifos;
  SmallVector<ObjectFifoLinkOp> objectFifoLinks;

  device.walk([&](Operation *op) {
    if (auto lt = dyn_cast<LogicalTileOp>(op))
      logicalTiles.push_back(lt);
    if (auto of = dyn_cast<ObjectFifoCreateOp>(op))
      objectFifos.push_back(of);
    if (auto link = dyn_cast<ObjectFifoLinkOp>(op))
      objectFifoLinks.push_back(link);
  });

  // Build channel requirements map by analyzing ObjectFifo connectivity
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

  // Place constrained tiles then compute tiles
  size_t nextCompIdx = 0;
  for (auto logicalTile : logicalTiles) {
    // Place fully constrained tiles at their specified coordinates
    auto col = logicalTile.tryGetCol();
    auto row = logicalTile.tryGetRow();
    if (col && row) {
      TileID tile{*col, *row};
      if (failed(validateAndUpdateChannelUsage(logicalTile, tile,
                                               channelRequirements, true)))
        return failure();

      result[logicalTile] = tile;
      availability.removeTile(tile, logicalTile.getTileType());
      continue;
    }

    // Place unconstrained compute tiles sequentially
    // NOTE: Partial constraints (col-only or row-only) are currently ignored
    if (logicalTile.getTileType() == AIETileType::CoreTile) {
      if (nextCompIdx >= availability.compTiles.size())
        return logicalTile.emitError(
            "no available compute tiles for placement");

      TileID tile = availability.compTiles[nextCompIdx++];
      if (failed(validateAndUpdateChannelUsage(logicalTile, tile,
                                               channelRequirements, false)))
        return failure();

      result[logicalTile] = tile;
    }
  }

  // Place mem/shim tiles by ObjectFifo groups to enable efficient merging
  llvm::DenseMap<int, SmallVector<ObjectFifoCreateOp>> groupToFifos;
  llvm::DenseMap<int, SmallVector<LogicalTileOp>> groupToLogicalTiles;
  buildObjectFifoGroups(objectFifos, objectFifoLinks, groupToFifos,
                        groupToLogicalTiles);

  // Process each group's logical tiles together
  llvm::DenseSet<int> processedGroups;

  // Process each group
  for (auto &[groupId, logicalTiles] : groupToLogicalTiles) {
    if (processedGroups.count(groupId))
      continue;
    processedGroups.insert(groupId);

    // Compute common column from ALL fifos in this group
    int groupCommonCol = 0;
    int totalCoreEndpoints = 0;
    int sumCols = 0;

    auto fifosIt = groupToFifos.find(groupId);
    if (fifosIt != groupToFifos.end()) {
      for (auto ofOp : fifosIt->second) {
        // Get core tile endpoints for this fifo
        Value producerTile = ofOp.getProducerTile();
        if (auto *producerOp = producerTile.getDefiningOp()) {
          if (auto prodLogical = dyn_cast<LogicalTileOp>(producerOp)) {
            if (prodLogical.getTileType() == AIETileType::CoreTile) {
              if (result.count(prodLogical.getOperation())) {
                sumCols += result[prodLogical.getOperation()].col;
                totalCoreEndpoints++;
              }
            }
          }
        }

        for (Value consumerTile : ofOp.getConsumerTiles()) {
          if (auto *consumerOp = consumerTile.getDefiningOp()) {
            if (auto consLogical = dyn_cast<LogicalTileOp>(consumerOp)) {
              if (consLogical.getTileType() == AIETileType::CoreTile) {
                if (result.count(consLogical.getOperation())) {
                  sumCols += result[consLogical.getOperation()].col;
                  totalCoreEndpoints++;
                }
              }
            }
          }
        }
      }
    }

    if (totalCoreEndpoints > 0) {
      groupCommonCol = (sumCols + totalCoreEndpoints / 2) / totalCoreEndpoints;
    }

    // Place each logical tile from this group
    for (auto logicalTile : logicalTiles) {
      // Skip if already placed (e.g., constrained)
      if (result.count(logicalTile.getOperation()))
        continue;

      // Get channel requirements for this tile
      auto it = channelRequirements.find(logicalTile.getOperation());
      int numInputChannels = 0, numOutputChannels = 0;
      if (it != channelRequirements.end()) {
        numInputChannels = it->second.first;
        numOutputChannels = it->second.second;
      }

      // Find tile with capacity near this group's common column
      auto maybeTile = findTileWithCapacity(
          groupCommonCol, availability.nonCompTiles, numInputChannels,
          numOutputChannels, logicalTile.getTileType());

      if (!maybeTile)
        return logicalTile.emitError("no tile with sufficient DMA capacity");

      result[logicalTile] = *maybeTile;

      // Update channel usage
      if (numInputChannels > 0)
        updateChannelUsage(*maybeTile, false, numInputChannels);
      if (numOutputChannels > 0)
        updateChannelUsage(*maybeTile, true, numOutputChannels);
    }
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

void SequentialPlacer::buildObjectFifoGroups(
    SmallVector<ObjectFifoCreateOp> &objectFifos,
    SmallVector<ObjectFifoLinkOp> &objectFifoLinks,
    llvm::DenseMap<int, SmallVector<ObjectFifoCreateOp>> &groupToFifos,
    llvm::DenseMap<int, SmallVector<LogicalTileOp>> &groupToLogicalTiles) {

  // Build map: ObjectFifo name -> group ID (for linked fifos)
  llvm::StringMap<int> fifoToGroup;
  int nextGroupId = 0;

  // Group ObjectFifos that are linked together
  for (auto linkOp : objectFifoLinks) {
    int groupId = nextGroupId++;

    // All source fifos belong to same group
    for (auto srcFifoAttr : linkOp.getFifoIns()) {
      auto srcFifoName = cast<FlatSymbolRefAttr>(srcFifoAttr).getValue();
      fifoToGroup[srcFifoName] = groupId;
    }

    // All dest fifos belong to same group
    for (auto dstFifoAttr : linkOp.getFifoOuts()) {
      auto dstFifoName = cast<FlatSymbolRefAttr>(dstFifoAttr).getValue();
      fifoToGroup[dstFifoName] = groupId;
    }
  }

  // Build maps: group ID -> logical tiles and fifos
  // Linked fifos share a group, unlinked fifos get individual entries
  int unlinkedGroupId = nextGroupId;

  for (auto ofOp : objectFifos) {
    // Determine which group this fifo belongs to
    int groupId;
    auto groupIt = fifoToGroup.find(ofOp.getSymName());
    if (groupIt != fifoToGroup.end()) {
      // This fifo is part of a link
      groupId = groupIt->second;
    } else {
      // Unlinked fifo gets its own group
      groupId = unlinkedGroupId++;
    }

    // Add to groupToFifos
    groupToFifos[groupId].push_back(ofOp);

    // Collect non-core tiles from this fifo
    // Check producer tile
    Value producerTile = ofOp.getProducerTile();
    if (auto *producerOp = producerTile.getDefiningOp()) {
      if (auto prodLogical = dyn_cast<LogicalTileOp>(producerOp)) {
        if (prodLogical.getTileType() != AIETileType::CoreTile) {
          groupToLogicalTiles[groupId].push_back(prodLogical);
        }
      }
    }

    // Check consumer tiles
    for (Value consumerTile : ofOp.getConsumerTiles()) {
      if (auto *consumerOp = consumerTile.getDefiningOp()) {
        if (auto consLogical = dyn_cast<LogicalTileOp>(consumerOp)) {
          if (consLogical.getTileType() != AIETileType::CoreTile) {
            groupToLogicalTiles[groupId].push_back(consLogical);
          }
        }
      }
    }
  }
}

LogicalResult PlacementAnalysis::runAnalysis(DeviceOp &device) {
  placer->initialize(device, device.getTargetModel());
  return placer->place(device, result);
}

std::optional<TileID>
PlacementAnalysis::getPlacement(Operation *logicalTile) const {
  auto it = result.find(logicalTile);
  if (it != result.end())
    return it->second;
  return std::nullopt;
}

// Find tile with available DMA capacity near target column
// This function checks capacity for BOTH input and output channels
// simultaneously. For unidirectional tiles, pass 0 for the unused direction:
//   - Input-only:  findTileWithCapacity(..., numInCh, 0, type)
//   - Output-only: findTileWithCapacity(..., 0, numOutCh, type)
//   - Both: findTileWithCapacity(..., numInCh, numOutCh, type)
// The requestedType parameter filters tiles to only consider matching types
// (e.g., only MemTiles for MemTile logical tiles, only ShimNOCTiles for shims).
std::optional<TileID> SequentialPlacer::findTileWithCapacity(
    int targetCol, std::vector<TileID> &tiles, int requiredInputChannels,
    int requiredOutputChannels, AIETileType requestedType) {
  // Search starting from target column, expanding outward
  int maxCol = targetModel->columns();

  for (int offset = 0; offset < maxCol; ++offset) {
    int searchCol = targetCol + offset;
    if (searchCol >= maxCol)
      continue;

    for (auto &tile : tiles) {
      // Filter by tile type - only consider tiles of the requested type
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

  // Check if tile is now exhausted and should be removed
  if (!hasAvailableChannels(tile, 0, 0)) {
    // Determine tile type to remove from appropriate list
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
