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

LogicalResult SequentialPlacer::place(ArrayRef<Operation *> logicalTiles,
                                      ArrayRef<Operation *> objectFifos,
                                      ArrayRef<Operation *> cores,
                                      PlacementResult &result) {
  // Build channel requirements map by analyzing ObjectFifo connectivity
  llvm::DenseMap<Operation *, std::pair<int, int>> channelRequirements;

  for (auto *op : objectFifos) {
    auto ofOp = dyn_cast<ObjectFifoCreateOp>(op);
    if (!ofOp)
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

  SmallVector<LogicalTileOp> computeLogicalTiles;
  SmallVector<LogicalTileOp> memLogicalTiles;
  SmallVector<LogicalTileOp> shimLogicalTiles;

  for (auto *op : logicalTiles) {
    auto logicalTile = dyn_cast<LogicalTileOp>(op);
    if (!logicalTile)
      continue;

    switch (logicalTile.getTileType()) {
    case AIETileType::CoreTile:
      computeLogicalTiles.push_back(logicalTile);
      break;
    case AIETileType::MemTile:
      memLogicalTiles.push_back(logicalTile);
      break;
    case AIETileType::ShimNOCTile:
      shimLogicalTiles.push_back(logicalTile);
      break;
    default:
      return logicalTile.emitError(
          "unsupported tile type for SequentialPlacer");
    }
  }

  // Place ALL constrained tiles at requested tile
  for (auto *op : logicalTiles) {
    auto logicalTile = dyn_cast<LogicalTileOp>(op);
    if (!logicalTile)
      continue;

    auto col = logicalTile.tryGetCol();
    auto row = logicalTile.tryGetRow();
    if (!col || !row)
      continue; // Not fully constrained

    TileID tile{*col, *row};
    if (failed(validateAndUpdateChannelUsage(logicalTile, tile,
                                             channelRequirements, true)))
      return failure();

    result[logicalTile] = tile;
    availability.removeTile(tile, logicalTile.getTileType());
  }

  // Place unconstrained compute tiles sequentially
  size_t nextCompIdx = 0;
  for (auto logicalTile : computeLogicalTiles) {
    if (result.count(logicalTile.getOperation()))
      continue; // Already placed by constraint

    // Unconstrained: sequential placement
    // NOTE: Partial constraints (col-only or row-only) are currently ignored
    if (nextCompIdx >= availability.compTiles.size())
      return logicalTile.emitError("no available compute tiles for placement");

    TileID tile = availability.compTiles[nextCompIdx++];
    if (failed(validateAndUpdateChannelUsage(logicalTile, tile,
                                             channelRequirements, false)))
      return failure();

    result[logicalTile] = tile;
  }

  // Place mem/shim tiles considering channel capacity
  SmallVector<LogicalTileOp> nonComputeTiles;
  nonComputeTiles.append(memLogicalTiles.begin(), memLogicalTiles.end());
  nonComputeTiles.append(shimLogicalTiles.begin(), shimLogicalTiles.end());

  int commonCol = getCommonColumn(result);
  for (auto logicalTile : nonComputeTiles) {
    if (result.count(logicalTile.getOperation()))
      continue; // Already placed by constraint

    // Get channel requirements
    auto it = channelRequirements.find(logicalTile.getOperation());
    int numInputChannels = 0, numOutputChannels = 0;
    if (it != channelRequirements.end()) {
      numInputChannels = it->second.first;
      numOutputChannels = it->second.second;
    }

    // Find tile with capacity near common column
    // Pass tile type to ensure we only search for matching tile types
    auto maybeTile = findTileWithCapacity(commonCol, availability.nonCompTiles,
                                          numInputChannels, numOutputChannels,
                                          logicalTile.getTileType());
    if (!maybeTile)
      return logicalTile.emitError("no tile with sufficient DMA capacity");

    result[logicalTile] = *maybeTile;

    // Update channel usage
    if (numInputChannels > 0)
      updateChannelUsage(*maybeTile, false, numInputChannels);
    if (numOutputChannels > 0)
      updateChannelUsage(*maybeTile, true, numOutputChannels);
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

int SequentialPlacer::getCommonColumn(const PlacementResult &result) {
  SmallVector<int> computeCols;

  for (const auto &[op, tile] : result) {
    if (auto logicalTile = dyn_cast<LogicalTileOp>(op)) {
      if (logicalTile.getTileType() == AIETileType::CoreTile) {
        computeCols.push_back(tile.col);
      }
    }
  }

  if (computeCols.empty())
    return 0;

  int sum = std::accumulate(computeCols.begin(), computeCols.end(), 0);
  return static_cast<int>(
      std::round(static_cast<double>(sum) / computeCols.size()));
}

LogicalResult PlacementAnalysis::runAnalysis(DeviceOp &device) {
  SmallVector<Operation *> logicalTiles;
  SmallVector<Operation *> objectFifos;
  SmallVector<Operation *> cores;

  // Collect operations
  device.walk([&](Operation *op) {
    if (isa<LogicalTileOp>(op))
      logicalTiles.push_back(op);
    if (isa<ObjectFifoCreateOp>(op))
      objectFifos.push_back(op);
    if (isa<CoreOp>(op))
      cores.push_back(op);
  });

  // Initialize placer
  const auto &targetModel = device.getTargetModel();
  placer->initialize(device, targetModel);

  // Run placement
  return placer->place(logicalTiles, objectFifos, cores, result);
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
//   - Input-only:  findTileWithCapacity(..., numInputChannels, 0, type)
//   - Output-only: findTileWithCapacity(..., 0, numOutputChannels, type)
//   - Both: findTileWithCapacity(..., numInputChannels, numOutputChannels, type)
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
