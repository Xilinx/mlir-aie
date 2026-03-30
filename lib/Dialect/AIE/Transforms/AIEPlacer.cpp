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

  device.walk([&](Operation *op) {
    if (auto lt = dyn_cast<LogicalTileOp>(op))
      logicalTiles.push_back(lt);
    if (auto of = dyn_cast<ObjectFifoCreateOp>(op))
      objectFifos.push_back(of);
    if (auto link = dyn_cast<ObjectFifoLinkOp>(op))
      objectFifoLinks.push_back(link);
  });

  // Phase 2: Build channel requirements from ObjectFifo connectivity
  auto channelRequirements =
      buildChannelRequirements(objectFifos, objectFifoLinks);

  // Phase 3: Place constrained tiles then compute tiles
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

      for (size_t i = nextCompIdx; i < availability.compTiles.size(); ++i) {
        TileID candidate = availability.compTiles[i];

        // Check partial constraints
        if (col && candidate.col != *col)
          continue;
        if (row && candidate.row != *row)
          continue;

        // Found valid tile - swap to nextCompIdx position and use
        std::swap(availability.compTiles[i],
                  availability.compTiles[nextCompIdx]);
        placement = availability.compTiles[nextCompIdx++];
        break;
      }

      if (!placement) {
        if (col || row) {
          return logicalTile.emitError()
                 << "no compute tile available matching constraint ("
                 << (col ? std::to_string(*col) : "?") << ", "
                 << (row ? std::to_string(*row) : "?") << ")";
        }
        return logicalTile.emitError(
            "no available compute tiles for placement");
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

  // Phase 4: Place mem/shim tiles by ObjectFifo groups
  llvm::DenseMap<int, SmallVector<ObjectFifoCreateOp>> groupToFifos;
  llvm::DenseMap<int, SmallVector<LogicalTileOp>> groupToLogicalTiles;
  buildObjectFifoGroups(objectFifos, objectFifoLinks, groupToFifos,
                        groupToLogicalTiles);

  // Process each group's logical tiles together
  llvm::DenseSet<int> processedGroups;

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

    // Compute common column as rounded average of all core tile endpoints.
    // This places mem/shim tiles near the center of their connected cores
    // to minimize routing distance.
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

      // Use column constraint if specified, otherwise use group's common column
      auto colConstraint = logicalTile.tryGetCol();
      int targetCol = colConstraint ? *colConstraint : groupCommonCol;

      // Find tile with capacity near target column
      auto maybeTile = findTileWithCapacity(
          targetCol, availability.nonCompTiles, numInputChannels,
          numOutputChannels, logicalTile.getTileType());

      if (!maybeTile)
        return logicalTile.emitError()
               << "no " << stringifyAIETileType(logicalTile.getTileType())
               << " with sufficient DMA capacity";

      result[logicalTile] = *maybeTile;

      // Update channel usage
      if (numInputChannels > 0)
        updateChannelUsage(*maybeTile, false, numInputChannels);
      if (numOutputChannels > 0)
        updateChannelUsage(*maybeTile, true, numOutputChannels);
    }
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
  int unlinkedGroupId = nextGroupId;

  for (auto ofOp : objectFifos) {
    int groupId;
    auto groupIt = fifoToGroup.find(ofOp.getSymName());

    // Linked fifos share a group, unlinked fifos get individual entries
    if (groupIt != fifoToGroup.end()) {
      groupId = groupIt->second;
    } else {
      groupId = unlinkedGroupId++;
    }

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
