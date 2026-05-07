//===- AIESABasedPlacer.cpp -------------------------------------*- C++ -*-===//
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
#include <chrono>
#include <cmath>
#include <numeric>

using namespace mlir;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-sa-placer"

namespace {

// Check if a tile satisfies partial placement constraints at a given position.
static bool satisfiesConstraints(Operation *tile, TileID pos) {
  if (auto lt = dyn_cast<LogicalTileOp>(tile)) {
    if (auto c = lt.tryGetCol(); c && *c != pos.col)
      return false;
    if (auto r = lt.tryGetRow(); r && *r != pos.row)
      return false;
  }
  return true;
}


} // namespace

//===----------------------------------------------------------------------===//
// Net model
//===----------------------------------------------------------------------===//

void SABasedPlacer::buildNetModel(
    SmallVector<ObjectFifoCreateOp> &objectFifos,
    SmallVector<ObjectFifoLinkOp> &objectFifoLinks) {
  nets.clear();
  tileToNetIndices.clear();

  for (auto ofOp : objectFifos) {
    NetInfo net;
    Value producerTile = ofOp.getProducerTile();
    if (auto *prodOp = producerTile.getDefiningOp())
      if (isa<LogicalTileOp>(prodOp))
        net.endpoints.push_back(prodOp);

    for (Value consumerTile : ofOp.getConsumerTiles()) {
      if (auto *consOp = consumerTile.getDefiningOp())
        if (isa<LogicalTileOp>(consOp))
          net.endpoints.push_back(consOp);
    }

    // Skip degenerate nets (0-1 endpoints)
    if (net.endpoints.size() < 2)
      continue;

    // >4 consumers = multicast penalty
    net.isMulticast = (ofOp.getConsumerTiles().size() > 4);

    size_t idx = nets.size();
    nets.push_back(net);

    // Build reverse index
    for (auto *ep : net.endpoints)
      tileToNetIndices[ep].push_back(idx);
  }
}

int SABasedPlacer::computeNetHPWL(const NetInfo &net) const {
  int spanCol = net.bb.maxCol - net.bb.minCol;
  int spanRow = net.bb.maxRow - net.bb.minRow;
  int hpwl = spanCol + spanRow;
  if (net.isMulticast)
    hpwl *= 2;
  // Area penalty: penalizes spread-out nets that consume more routing tracks.
  // A row-aligned broadcast (area=0) is much cheaper to route than a
  // scattered one (area=cols*rows).
  int area = spanCol * spanRow;
  return hpwl + area;
}

int SABasedPlacer::computeTotalHPWL() const {
  int total = 0;
  for (const auto &net : nets)
    total += computeNetHPWL(net);
  return total;
}

void SABasedPlacer::initBoundingBoxes() {
  for (auto &net : nets) {
    if (net.endpoints.empty())
      continue;

    TileID first = currentPlacement[net.endpoints[0]];
    net.bb = {first.col, first.col, first.row, first.row, 1, 1, 1, 1};

    for (size_t i = 1; i < net.endpoints.size(); i++) {
      TileID pos = currentPlacement[net.endpoints[i]];

      if (pos.col < net.bb.minCol) {
        net.bb.minCol = pos.col;
        net.bb.countAtMinCol = 1;
      } else if (pos.col == net.bb.minCol) {
        net.bb.countAtMinCol++;
      }

      if (pos.col > net.bb.maxCol) {
        net.bb.maxCol = pos.col;
        net.bb.countAtMaxCol = 1;
      } else if (pos.col == net.bb.maxCol) {
        net.bb.countAtMaxCol++;
      }

      if (pos.row < net.bb.minRow) {
        net.bb.minRow = pos.row;
        net.bb.countAtMinRow = 1;
      } else if (pos.row == net.bb.minRow) {
        net.bb.countAtMinRow++;
      }

      if (pos.row > net.bb.maxRow) {
        net.bb.maxRow = pos.row;
        net.bb.countAtMaxRow = 1;
      } else if (pos.row == net.bb.maxRow) {
        net.bb.countAtMaxRow++;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Incremental move evaluation
//===----------------------------------------------------------------------===//

int SABasedPlacer::evaluateMove(
    Operation *tile, TileID newPos,
    SmallVector<std::pair<size_t, NetBoundingBox>> &backups) {
  backups.clear();
  int deltaCost = 0;

  auto it = tileToNetIndices.find(tile);
  if (it == tileToNetIndices.end())
    return 0;

  for (size_t netIdx : it->second) {
    NetInfo &net = nets[netIdx];
    backups.push_back({netIdx, net.bb});
    int oldCost = computeNetHPWL(net);

    // Recompute BB from scratch for this net
    int newMinCol = INT_MAX, newMaxCol = INT_MIN;
    int newMinRow = INT_MAX, newMaxRow = INT_MIN;
    int cMinCol = 0, cMaxCol = 0, cMinRow = 0, cMaxRow = 0;

    for (auto *ep : net.endpoints) {
      TileID pos = (ep == tile) ? newPos : currentPlacement[ep];

      if (pos.col < newMinCol) {
        newMinCol = pos.col;
        cMinCol = 1;
      } else if (pos.col == newMinCol) {
        cMinCol++;
      }
      if (pos.col > newMaxCol) {
        newMaxCol = pos.col;
        cMaxCol = 1;
      } else if (pos.col == newMaxCol) {
        cMaxCol++;
      }
      if (pos.row < newMinRow) {
        newMinRow = pos.row;
        cMinRow = 1;
      } else if (pos.row == newMinRow) {
        cMinRow++;
      }
      if (pos.row > newMaxRow) {
        newMaxRow = pos.row;
        cMaxRow = 1;
      } else if (pos.row == newMaxRow) {
        cMaxRow++;
      }
    }

    net.bb = {newMinCol, newMaxCol, newMinRow, newMaxRow,
              cMinCol,   cMaxCol,   cMinRow,   cMaxRow};

    deltaCost += computeNetHPWL(net) - oldCost;
  }

  return deltaCost;
}

void SABasedPlacer::revertMove(
    SmallVector<std::pair<size_t, NetBoundingBox>> &backups) {
  for (auto &[netIdx, savedBB] : backups)
    nets[netIdx].bb = savedBB;
}

void SABasedPlacer::applyMove(Operation *tile, TileID newPos) {
  TileID oldPos = currentPlacement[tile];

  // Update reverse map (only for compute -- mem/shim can share)
  auto it = tileTypes.find(tile);
  if (it != tileTypes.end() && it->second == AIETileType::CoreTile) {
    physToLogical.erase(oldPos);
    physToLogical[newPos] = tile;
  }

  currentPlacement[tile] = newPos;
}

//===----------------------------------------------------------------------===//
// Move generation
//===----------------------------------------------------------------------===//

bool SABasedPlacer::isLegalPosition(Operation *tile, TileID pos) const {
  auto it = tileTypes.find(tile);
  if (it == tileTypes.end())
    return false;

  AIETileType needed = it->second;
  AIETileType actual = targetModel->getTileType(pos.col, pos.row);
  if (needed != actual)
    return false;

  // Compute tiles: must be unoccupied
  if (needed == AIETileType::CoreTile) {
    auto occIt = physToLogical.find(pos);
    if (occIt != physToLogical.end() && occIt->second != tile)
      return false;
  }

  return satisfiesConstraints(tile, pos);
}

//===----------------------------------------------------------------------===//
// Memory capacity tracking
//===----------------------------------------------------------------------===//

// Overhead per core tile for stack, etc.
static constexpr int64_t kCoreOverhead = 4096;

// Add a single fifo's contributions to memory and DMA usage maps.
void SABasedPlacer::addFifoContribution(size_t fifoIdx, int sign) {
  const auto &fb = fifoBuffers[fifoIdx];
  Operation *prod = fb.producer;
  if (!prod)
    return;
  auto prodPlaceIt = currentPlacement.find(prod);
  if (prodPlaceIt == currentPlacement.end())
    return;
  TileID prodPos = prodPlaceIt->second;
  auto prodTypeIt = tileTypes.find(prod);
  bool prodIsCore =
      prodTypeIt != tileTypes.end() &&
      prodTypeIt->second == AIETileType::CoreTile;

  bool prodCharged = false;
  bool prodNeedsDMA = false;

  for (size_t ci = 0; ci < fb.consumers.size(); ci++) {
    auto *cons = fb.consumers[ci];
    if (!cons)
      continue;
    auto consPlaceIt = currentPlacement.find(cons);
    if (consPlaceIt == currentPlacement.end())
      continue;
    TileID consPos = consPlaceIt->second;
    auto consTypeIt = tileTypes.find(cons);
    bool consIsCore =
        consTypeIt != tileTypes.end() &&
        consTypeIt->second == AIETileType::CoreTile;

    int consDepth = (ci < fb.consumerDepths.size()) ? fb.consumerDepths[ci]
                                                    : fb.producerDepth;

    // Memory contribution
    // For MemTile producers: cap per-fifo charge at tile capacity.
    // When total buffers (sizeBytes * depth) exceed a MemTile's capacity,
    // the stateful transform auto-spills excess buffers to a neighbor.
    // We charge only what fits locally; the overflow is assumed to spill.
    auto capForMemTile = [&](int64_t totalBytes, TileID pos) -> int64_t {
      if (targetModel->getTileType(pos.col, pos.row) == AIETileType::MemTile) {
        int64_t capacity = targetModel->getMemTileSize();
        return std::min(totalBytes, capacity);
      }
      return totalBytes;
    };

    if (prodPos == consPos) {
      // Same tile: use max of producer/consumer size × depth
      int depth = std::max(fb.producerDepth, consDepth);
      int64_t maxSize = std::max(fb.producerSizeBytes, fb.consumerSizeBytes);
      int64_t total = maxSize * depth;
      currentMemUsage[prodPos] += sign * capForMemTile(total, prodPos);
    } else if (prodIsCore && consIsCore) {
      if (!prodCharged) {
        currentMemUsage[prodPos] += sign * fb.producerSizeBytes * fb.producerDepth;
        prodCharged = true;
      }
      currentMemUsage[consPos] += sign * fb.consumerSizeBytes * consDepth;
    } else if (prodIsCore && !consIsCore) {
      if (!prodCharged) {
        currentMemUsage[prodPos] += sign * fb.producerSizeBytes * fb.producerDepth;
        prodCharged = true;
      }
    } else if (!prodIsCore && consIsCore) {
      // MemTile/Shim producer to CoreTile consumer
      if (!prodCharged) {
        int64_t total = fb.producerSizeBytes * fb.producerDepth;
        currentMemUsage[prodPos] += sign * capForMemTile(total, prodPos);
        prodCharged = true;
      }
      // Consumer uses consumer element size (may be smaller)
      currentMemUsage[consPos] += sign * fb.consumerSizeBytes * consDepth;
    } else if (!prodIsCore && !consIsCore) {
      // MemTile to MemTile or Shim: charge producer with cap
      if (!prodCharged) {
        int64_t total = fb.producerSizeBytes * fb.producerDepth;
        currentMemUsage[prodPos] += sign * capForMemTile(total, prodPos);
        prodCharged = true;
      }
    }

    // DMA contribution
    if (prodPos != consPos) {
      bool sharedMem = prodIsCore && consIsCore &&
                       prodPos.col == consPos.col &&
                       std::abs(prodPos.row - consPos.row) == 1;
      if (!sharedMem) {
        currentDMAUsage[consPos].first += sign;
        prodNeedsDMA = true;
      }
    }
  }

  if (prodNeedsDMA)
    currentDMAUsage[prodPos].second += sign;
}

void SABasedPlacer::initResourceTracking() {
  currentMemUsage.clear();
  currentDMAUsage.clear();

  // Build reverse index: tile → fifo indices
  tileToFifoIndices.clear();
  for (size_t i = 0; i < fifoBuffers.size(); i++) {
    const auto &fb = fifoBuffers[i];
    if (fb.producer)
      tileToFifoIndices[fb.producer].push_back(i);
    for (auto *cons : fb.consumers)
      if (cons)
        tileToFifoIndices[cons].push_back(i);
  }

  // Add all fifo contributions
  for (size_t i = 0; i < fifoBuffers.size(); i++)
    addFifoContribution(i, +1);

  // Add static buffers and core overhead
  for (auto &[op, bufSize] : staticBufferSizes) {
    auto posIt = currentPlacement.find(op);
    if (posIt != currentPlacement.end())
      currentMemUsage[posIt->second] += bufSize;
  }
  for (auto &[op, pos] : currentPlacement) {
    auto typeIt = tileTypes.find(op);
    if (typeIt != tileTypes.end() &&
        typeIt->second == AIETileType::CoreTile)
      currentMemUsage[pos] += kCoreOverhead;
  }

  cachedResourcePenalty = computePenaltyFromMaps();
}

int SABasedPlacer::computePenaltyFromMaps() const {
  constexpr int kPenaltyFactor = 10;
  int64_t coreCapacity = targetModel->getLocalMemorySize();
  int penalty = 0;

  // Memory penalty with neighbor spillover
  int64_t memTileCapacity = targetModel->getMemTileSize();
  for (auto &[tilePos, used] : currentMemUsage) {
    auto tileType = targetModel->getTileType(tilePos.col, tilePos.row);
    // Skip shim tiles (no data memory to overflow)
    if (tileType == AIETileType::ShimNOCTile ||
        tileType == AIETileType::ShimPLTile)
      continue;
    int64_t capacity = (tileType == AIETileType::MemTile)
                           ? memTileCapacity
                           : coreCapacity;
    if (used <= capacity)
      continue;
    int64_t overflow = used - capacity;
    TileID neighbors[] = {
        {tilePos.col - 1, tilePos.row},
        {tilePos.col, tilePos.row + 1},
        {tilePos.col, tilePos.row - 1},
    };
    for (auto &nbr : neighbors) {
      if (overflow <= 0)
        break;
      if (nbr.col < 0 || nbr.col >= targetModel->columns() ||
          nbr.row < 0 || nbr.row >= targetModel->rows())
        continue;
      if (targetModel->getTileType(nbr.col, nbr.row) != AIETileType::CoreTile)
        continue;
      auto it = currentMemUsage.find(nbr);
      int64_t nbrUsed = (it != currentMemUsage.end()) ? it->second : 0;
      int64_t nbrSpare = coreCapacity - nbrUsed;
      if (nbrSpare > 0)
        overflow -= nbrSpare;
    }
    if (overflow > 0)
      penalty += (static_cast<int>(overflow / coreCapacity) + 1) * kPenaltyFactor;
  }

  // DMA channel penalty
  for (auto &[tilePos, usage] : currentDMAUsage) {
    auto [maxIn, maxOut] = getDMACapacity(*targetModel, tilePos);
    if (usage.first > maxIn)
      penalty += (usage.first - maxIn) * kPenaltyFactor;
    if (usage.second > maxOut)
      penalty += (usage.second - maxOut) * kPenaltyFactor;
  }

  return penalty;
}

int SABasedPlacer::updateResourcePenalty(
    const SmallVector<std::pair<Operation *, TileID>> &oldPlacements) {
  // Collect all fifo indices affected by the moved tiles
  DenseSet<size_t> affectedFifos;
  for (auto &[op, _] : oldPlacements) {
    auto it = tileToFifoIndices.find(op);
    if (it != tileToFifoIndices.end())
      for (size_t idx : it->second)
        affectedFifos.insert(idx);
  }

  // Update static buffer and overhead contributions for moved tiles
  for (auto &[op, oldPos] : oldPlacements) {
    // Subtract old static buffer contribution
    auto bufIt = staticBufferSizes.find(op);
    if (bufIt != staticBufferSizes.end())
      currentMemUsage[oldPos] -= bufIt->second;
    // Subtract old core overhead
    auto typeIt = tileTypes.find(op);
    if (typeIt != tileTypes.end() &&
        typeIt->second == AIETileType::CoreTile)
      currentMemUsage[oldPos] -= kCoreOverhead;
  }

  // Subtract old fifo contributions (positions already updated in
  // currentPlacement, but we need old positions for subtraction).
  // Temporarily swap back old positions for affected fifos.
  DenseMap<Operation *, TileID> savedPositions;
  for (auto &[op, oldPos] : oldPlacements) {
    savedPositions[op] = currentPlacement[op];
    currentPlacement[op] = oldPos;
  }
  for (size_t fi : affectedFifos)
    addFifoContribution(fi, -1);

  // Restore new positions and add new contributions
  for (auto &[op, newPos] : savedPositions)
    currentPlacement[op] = newPos;
  for (size_t fi : affectedFifos)
    addFifoContribution(fi, +1);

  // Add new static buffer and overhead contributions
  for (auto &[op, _] : oldPlacements) {
    TileID newPos = currentPlacement[op];
    auto bufIt = staticBufferSizes.find(op);
    if (bufIt != staticBufferSizes.end())
      currentMemUsage[newPos] += bufIt->second;
    auto typeIt = tileTypes.find(op);
    if (typeIt != tileTypes.end() &&
        typeIt->second == AIETileType::CoreTile)
      currentMemUsage[newPos] += kCoreOverhead;
  }

  cachedResourcePenalty = computePenaltyFromMaps();
  return cachedResourcePenalty;
}

//===----------------------------------------------------------------------===//
// Cascade group helpers
//===----------------------------------------------------------------------===//

bool SABasedPlacer::getCascadeGroupPositions(
    const CascadeGroup &group, TileID anchorPos,
    SmallVector<TileID> &positions) const {
  positions.clear();
  positions.push_back(anchorPos);

  for (size_t i = 1; i < group.tiles.size(); i++) {
    TileID pos;
    if (group.orientation == CascadeOrientation::Horizontal) {
      // dst is east of src: col+i, same row
      pos = {anchorPos.col + static_cast<int>(i), anchorPos.row};
    } else {
      // dst is south of src: same col, row-i
      pos = {anchorPos.col, anchorPos.row - static_cast<int>(i)};
    }

    // Check bounds
    if (pos.col < 0 || pos.col >= targetModel->columns() || pos.row < 0 ||
        pos.row >= targetModel->rows())
      return false;

    // Must be a CoreTile position
    if (targetModel->getTileType(pos.col, pos.row) != AIETileType::CoreTile)
      return false;

    positions.push_back(pos);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Move generation
//===----------------------------------------------------------------------===//

bool SABasedPlacer::generateShiftMove(Operation *&tile, TileID &newPos) {
  if (movableTiles.empty())
    return false;

  for (int attempt = 0; attempt < 20; attempt++) {
    // Pick random movable tile
    std::uniform_int_distribution<size_t> tileDist(0,
                                                    movableTiles.size() - 1);
    tile = movableTiles[tileDist(rng)];

    // Skip cascade group members — they use group swap moves
    if (cascadeGroupOf.count(tile))
      continue;

    auto typeIt = tileTypes.find(tile);
    if (typeIt == tileTypes.end())
      continue;
    AIETileType type = typeIt->second;

    // Pick random physical tile of matching type
    const auto &candidates = (type == AIETileType::CoreTile)
                                 ? availability.compTiles
                                 : availability.nonCompTiles;
    if (candidates.empty())
      continue;

    std::uniform_int_distribution<size_t> posDist(0, candidates.size() - 1);
    TileID candidate = candidates[posDist(rng)];

    // Filter by tile type for non-compute (nonCompTiles has mixed types)
    if (type != AIETileType::CoreTile &&
        targetModel->getTileType(candidate.col, candidate.row) != type)
      continue;

    // Skip if same position
    if (candidate == currentPlacement[tile])
      continue;

    if (!isLegalPosition(tile, candidate))
      continue;

    newPos = candidate;
    return true;
  }
  return false;
}

bool SABasedPlacer::generateSwapMove(Operation *&tile1, Operation *&tile2) {
  if (movableTiles.size() < 2)
    return false;

  for (int attempt = 0; attempt < 20; attempt++) {
    std::uniform_int_distribution<size_t> dist(0, movableTiles.size() - 1);
    size_t idx1 = dist(rng);
    size_t idx2 = dist(rng);
    if (idx1 == idx2)
      continue;

    tile1 = movableTiles[idx1];
    tile2 = movableTiles[idx2];

    // Skip cascade group members — they use group swap instead
    if (cascadeGroupOf.count(tile1) || cascadeGroupOf.count(tile2))
      continue;

    // Must be same tile type
    if (tileTypes[tile1] != tileTypes[tile2])
      continue;

    TileID pos1 = currentPlacement[tile1];
    TileID pos2 = currentPlacement[tile2];

    // For swaps, skip occupancy check (both tiles move simultaneously).
    auto t1It = tileTypes.find(tile1);
    auto t2It = tileTypes.find(tile2);
    if (t1It == tileTypes.end() || t2It == tileTypes.end())
      continue;
    if (targetModel->getTileType(pos2.col, pos2.row) != t1It->second)
      continue;
    if (targetModel->getTileType(pos1.col, pos1.row) != t2It->second)
      continue;
    if (!satisfiesConstraints(tile1, pos2) ||
        !satisfiesConstraints(tile2, pos1))
      continue;

    return true;
  }
  return false;
}

// Generate a cascade group swap: move the group to new positions and
// swap with the tiles currently there. Orientation may flip.
// Returns the list of all (tile, newPos) pairs for the swap.
bool SABasedPlacer::generateGroupSwapMove(
    int groupIdx,
    SmallVector<std::pair<Operation *, TileID>> &moves) {
  moves.clear();
  CascadeGroup &group = cascadeGroups[groupIdx];

  for (int attempt = 0; attempt < 40; attempt++) {
    // Pick random anchor from all core tile positions
    const auto &candidates = availability.compTiles;
    if (candidates.empty())
      return false;
    std::uniform_int_distribution<size_t> posDist(0, candidates.size() - 1);
    TileID anchorPos = candidates[posDist(rng)];

    // Optionally flip orientation (20% probability)
    CascadeOrientation origOrientation = group.orientation;
    std::uniform_real_distribution<double> flipDist(0.0, 1.0);
    if (flipDist(rng) < 0.2) {
      group.orientation =
          (group.orientation == CascadeOrientation::Horizontal)
              ? CascadeOrientation::Vertical
              : CascadeOrientation::Horizontal;
    }

    // Compute target positions for the group
    SmallVector<TileID> newPositions;
    if (!getCascadeGroupPositions(group, anchorPos, newPositions)) {
      group.orientation = origOrientation;
      continue;
    }

    // Get current group positions
    SmallVector<TileID> oldPositions;
    for (auto *t : group.tiles)
      oldPositions.push_back(currentPlacement[t]);

    // Skip if identical
    if (oldPositions == newPositions) {
      group.orientation = origOrientation;
      continue;
    }

    // Check group members' partial constraints at new positions
    bool valid = true;
    for (size_t i = 0; i < group.tiles.size(); i++) {
      if (!satisfiesConstraints(group.tiles[i], newPositions[i])) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      group.orientation = origOrientation;
      continue;
    }

    // Collect tiles being displaced: tiles at target positions that aren't
    // part of this group. Also collect old positions that are being vacated
    // and aren't also target positions (for displaced tiles to go to).
    SmallVector<Operation *> displacedOps;
    SmallVector<TileID> vacatedPositions;

    // Build set of new positions and old positions for quick lookup
    llvm::DenseSet<TileID> newPosSet, oldPosSet;
    for (auto &p : newPositions) newPosSet.insert(p);
    for (auto &p : oldPositions) oldPosSet.insert(p);

    // Find tiles at target positions (excluding our own group members)
    for (auto &pos : newPositions) {
      auto it = physToLogical.find(pos);
      if (it == physToLogical.end())
        continue;
      Operation *occ = it->second;
      // Skip own group members
      bool ownMember = false;
      for (auto *t : group.tiles)
        if (occ == t) { ownMember = true; break; }
      if (ownMember)
        continue;
      // Don't displace other cascade group members (would break their group)
      if (cascadeGroupOf.count(occ)) {
        valid = false;
        break;
      }
      displacedOps.push_back(occ);
    }
    if (!valid) {
      group.orientation = origOrientation;
      continue;
    }

    // Positions being vacated by the group (that aren't also targets)
    for (auto &p : oldPositions) {
      if (!newPosSet.count(p))
        vacatedPositions.push_back(p);
    }

    // Check we have enough vacated positions for displaced tiles
    if (displacedOps.size() > vacatedPositions.size()) {
      group.orientation = origOrientation;
      continue;
    }

    // Check displaced tiles' partial constraints at vacated positions
    for (size_t i = 0; i < displacedOps.size(); i++) {
      if (!satisfiesConstraints(displacedOps[i], vacatedPositions[i])) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      group.orientation = origOrientation;
      continue;
    }

    // Build move list
    for (size_t i = 0; i < group.tiles.size(); i++)
      moves.push_back({group.tiles[i], newPositions[i]});
    for (size_t i = 0; i < displacedOps.size(); i++)
      moves.push_back({displacedOps[i], vacatedPositions[i]});

    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Initial temperature estimation
//===----------------------------------------------------------------------===//

double SABasedPlacer::estimateInitialTemperature(int numSamples) {
  // Collect delta costs from random moves
  std::vector<double> deltaCosts;
  deltaCosts.reserve(numSamples);

  SmallVector<std::pair<size_t, NetBoundingBox>> backups;

  for (int i = 0; i < numSamples; i++) {
    Operation *tile = nullptr;
    TileID newPos;
    if (!generateShiftMove(tile, newPos))
      continue;

    int delta = evaluateMove(tile, newPos, backups);
    deltaCosts.push_back(delta);
    revertMove(backups);
  }

  if (deltaCosts.empty())
    return 1.0;

  // Equilibrium binary search: find T where E[delta_cost] ≈ 0
  double maxDelta = 0.0;
  for (double d : deltaCosts)
    maxDelta = std::max(maxDelta, std::abs(d));

  if (maxDelta == 0.0)
    return 1.0;

  auto getExpectedValue = [&](double T) -> double {
    double ev = 0.0;
    for (double d : deltaCosts) {
      if (d <= 0)
        ev += d;
      else
        ev += d * std::exp(-d / T);
    }
    return ev;
  };

  double lowT = 0.0;
  double highT = 5.0 * maxDelta;
  double T = 1.0;

  for (int iter = 0; iter < 100; iter++) {
    double ev = getExpectedValue(T);
    if (std::abs(highT - lowT) / std::max(T, 1e-10) < 1e-6)
      break;
    if (ev > 0) {
      highT = T;
    } else {
      lowT = T;
    }
    T = (lowT + highT) / 2.0;
  }

  return T;
}

//===----------------------------------------------------------------------===//
// Post-SA mem/shim merge
//===----------------------------------------------------------------------===//

LogicalResult SABasedPlacer::mergeMemShimTiles(DeviceOp device) {
  // After SA, each logical tile has its own physical position.
  // Mem/shim tiles in the same column and of the same type can share a
  // physical tile if DMA capacity permits. Merge them greedily.

  // Group non-compute logical tiles by (col, tileType)
  using Key = std::pair<int, AIETileType>;
  llvm::DenseMap<Key, SmallVector<Operation *>> groups;

  for (auto &[op, pos] : currentPlacement) {
    auto it = tileTypes.find(op);
    if (it == tileTypes.end())
      continue;
    if (it->second == AIETileType::CoreTile)
      continue;
    groups[{pos.col, it->second}].push_back(op);
  }

  for (auto &[key, ops] : groups) {
    if (ops.size() <= 1)
      continue;

    TileID mergeTarget = currentPlacement[ops[0]];

    // Track accumulated channel usage on the merge target
    int usedIn = 0, usedOut = 0;
    auto dmaIt = currentDMAUsage.find(mergeTarget);
    if (dmaIt != currentDMAUsage.end()) {
      usedIn = dmaIt->second.first;
      usedOut = dmaIt->second.second;
    }

    auto [maxIn, maxOut] = getDMACapacity(*targetModel, mergeTarget);

    for (size_t i = 1; i < ops.size(); i++) {
      // For merge, we need to estimate how many additional channels
      // this logical tile would add. Use a simple heuristic: count
      // ObjectFifos where this tile is producer or consumer.
      int needIn = 0, needOut = 0;
      for (const auto &fb : fifoBuffers) {
        if (fb.producer == ops[i])
          needOut++;
        for (auto *cons : fb.consumers) {
          if (cons == ops[i])
            needIn++;
        }
      }

      if (usedIn + needIn <= maxIn && usedOut + needOut <= maxOut) {
        // Merge: point this logical tile to the same physical tile
        currentPlacement[ops[i]] = mergeTarget;
        usedIn += needIn;
        usedOut += needOut;
      }
      // If can't merge, it keeps its own position (SA placed it somewhere
      // valid already).
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Main SA loop
//===----------------------------------------------------------------------===//

LogicalResult SABasedPlacer::place(DeviceOp device) {
  auto startTime = std::chrono::steady_clock::now();

  // Seed RNG
  if (rngSeed == 0) {
    rng.seed(std::random_device{}());
  } else {
    rng.seed(rngSeed);
  }

  // Phase 1: Collect IR operations
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

  if (logicalTiles.empty())
    return success();

  // Cache tile types
  for (auto lt : logicalTiles)
    tileTypes[lt.getOperation()] = lt.getTileType();

  // Collect static buffer sizes per logical tile
  device.walk([&](BufferOp bufOp) {
    auto *tileOp = bufOp.getTile().getDefiningOp();
    if (!tileOp || !isa<LogicalTileOp>(tileOp))
      return;
    staticBufferSizes[tileOp] += bufOp.getAllocationSize();
  });

  // Phase 2: Build net model and buffer requirements
  buildNetModel(objectFifos, objectFifoLinks);

  // Collect buffer requirements from ObjectFifos
  {
    mlir::DataLayout dataLayout(device->getParentOfType<ModuleOp>());

    for (auto ofOp : objectFifos) {
      FifoBufferInfo fb;
      fb.fifoOp = ofOp.getOperation();

      // Get producer
      auto *prodOp = ofOp.getProducerTile().getDefiningOp();
      fb.producer = (prodOp && isa<LogicalTileOp>(prodOp)) ? prodOp : nullptr;

      // Get consumers
      for (Value ct : ofOp.getConsumerTiles()) {
        auto *consOp = ct.getDefiningOp();
        if (consOp && isa<LogicalTileOp>(consOp))
          fb.consumers.push_back(consOp);
      }

      // Compute buffer element size (producer side)
      auto fifoType = llvm::cast<AIEObjectFifoType>(ofOp.getElemType());
      auto elemType = llvm::cast<MemRefType>(fifoType.getElementType());
      int64_t elementBits =
          dataLayout.getTypeSizeInBits(elemType.getElementType());
      fb.producerSizeBytes = elemType.getNumElements() * elementBits / 8;

      // Consumer element size (may differ with consumerElemType)
      fb.consumerSizeBytes = fb.producerSizeBytes;
      if (auto consType = ofOp.getConsumerElemType()) {
        auto consOFType = llvm::cast<AIEObjectFifoType>(consType.value());
        auto consMemref = llvm::cast<MemRefType>(consOFType.getElementType());
        int64_t consBits =
            dataLayout.getTypeSizeInBits(consMemref.getElementType());
        fb.consumerSizeBytes = consMemref.getNumElements() * consBits / 8;
      }

      // Get per-endpoint depths: [producer, consumer0, consumer1, ...]
      if (auto arrayAttr = dyn_cast<ArrayAttr>(ofOp.getElemNumber())) {
        auto values = arrayAttr.getValue();
        fb.producerDepth =
            values.empty()
                ? 1
                : static_cast<int>(cast<IntegerAttr>(values[0]).getInt());
        for (size_t i = 0; i < fb.consumers.size(); i++) {
          int idx = i + 1; // consumer depths start at index 1
          int d = (idx < static_cast<int>(values.size()))
                      ? static_cast<int>(
                            cast<IntegerAttr>(values[idx]).getInt())
                      : fb.producerDepth;
          fb.consumerDepths.push_back(d);
        }
      } else {
        int d = static_cast<int>(
            cast<IntegerAttr>(ofOp.getElemNumber()).getInt());
        fb.producerDepth = d;
        for (size_t i = 0; i < fb.consumers.size(); i++)
          fb.consumerDepths.push_back(d);
      }

      fifoBuffers.push_back(fb);
    }

    LLVM_DEBUG(llvm::dbgs() << "[SA] Collected " << fifoBuffers.size()
                            << " ObjectFifo buffer requirements\n");
  }

  // Collect cascade flow ops and build cascade groups
  {
    // Map each tile to its cascade connections
    llvm::DenseMap<Operation *, Operation *> cascadeSrcToDst;
    llvm::DenseMap<Operation *, Operation *> cascadeDstToSrc;

    device.walk([&](CascadeFlowOp cascadeOp) {
      Value srcTile = cascadeOp.getSourceTile();
      Value dstTile = cascadeOp.getDestTile();
      auto *srcOp = srcTile.getDefiningOp();
      auto *dstOp = dstTile.getDefiningOp();
      if (srcOp && dstOp && isa<LogicalTileOp>(srcOp) &&
          isa<LogicalTileOp>(dstOp)) {
        cascadeSrcToDst[srcOp] = dstOp;
        cascadeDstToSrc[dstOp] = srcOp;
      }
    });

    // Build cascade groups by following chains: find tiles that are
    // cascade sources but not cascade destinations (chain heads)
    llvm::DenseSet<Operation *> visited;
    for (auto &[src, dst] : cascadeSrcToDst) {
      if (cascadeDstToSrc.count(src))
        continue; // Not a chain head
      if (visited.count(src))
        continue;

      CascadeGroup group;
      Operation *current = src;
      while (current) {
        group.tiles.push_back(current);
        visited.insert(current);
        auto it = cascadeSrcToDst.find(current);
        current = (it != cascadeSrcToDst.end()) ? it->second : nullptr;
      }

      if (group.tiles.size() >= 2) {
        int groupIdx = cascadeGroups.size();
        for (auto *t : group.tiles)
          cascadeGroupOf[t] = groupIdx;
        cascadeGroups.push_back(std::move(group));
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "[SA] Found " << cascadeGroups.size()
                            << " cascade groups\n");
  }

  // Phase 3: Classify tiles and generate initial placement
  // Separate available tiles by type for random assignment
  std::vector<TileID> availComp = availability.compTiles;
  std::vector<TileID> availMem, availShim;
  for (auto &t : availability.nonCompTiles) {
    AIETileType type = targetModel->getTileType(t.col, t.row);
    if (type == AIETileType::MemTile)
      availMem.push_back(t);
    else if (type == AIETileType::ShimNOCTile)
      availShim.push_back(t);
  }

  // Shuffle for random initial placement
  std::shuffle(availComp.begin(), availComp.end(), rng);
  std::shuffle(availMem.begin(), availMem.end(), rng);
  std::shuffle(availShim.begin(), availShim.end(), rng);

  size_t compIdx = 0, memIdx = 0, shimIdx = 0;

  for (auto lt : logicalTiles) {
    Operation *op = lt.getOperation();
    auto col = lt.tryGetCol();
    auto row = lt.tryGetRow();
    AIETileType type = lt.getTileType();

    if (col && row) {
      // Fully constrained
      TileID pos{*col, *row};
      currentPlacement[op] = pos;
      physToLogical[pos] = op;
      constrainedTiles.insert(op);

      // Remove from available pool
      if (type == AIETileType::CoreTile)
        availComp.erase(std::remove(availComp.begin(), availComp.end(), pos),
                        availComp.end());
      else if (type == AIETileType::MemTile)
        availMem.erase(std::remove(availMem.begin(), availMem.end(), pos),
                       availMem.end());
      else if (type == AIETileType::ShimNOCTile)
        availShim.erase(std::remove(availShim.begin(), availShim.end(), pos),
                        availShim.end());
    }
  }

  // Place cascade groups first as rigid units
  llvm::DenseSet<Operation *> cascadePlaced;
  for (auto &group : cascadeGroups) {
    // Skip groups where all tiles are already constrained
    bool allConstrained = true;
    for (auto *t : group.tiles) {
      if (!constrainedTiles.count(t)) {
        allConstrained = false;
        break;
      }
    }
    if (allConstrained)
      continue;

    // Try random positions for the anchor (src) tile
    bool placed = false;
    for (int attempt = 0; attempt < 100; attempt++) {
      // Pick random anchor from available compute tiles
      if (compIdx >= availComp.size())
        break;
      std::uniform_int_distribution<size_t> anchorDist(compIdx,
                                                        availComp.size() - 1);
      TileID anchorPos = availComp[anchorDist(rng)];

      // Try both orientations
      for (auto orient : {CascadeOrientation::Horizontal,
                          CascadeOrientation::Vertical}) {
        group.orientation = orient;
        SmallVector<TileID> positions;
        if (!getCascadeGroupPositions(group, anchorPos, positions))
          continue;

        // Check all positions are available and legal
        bool allOk = true;
        for (size_t i = 0; i < group.tiles.size(); i++) {
          // Check partial constraints
          if (auto lt = dyn_cast<LogicalTileOp>(group.tiles[i])) {
            auto colC = lt.tryGetCol();
            auto rowC = lt.tryGetRow();
            if (colC && *colC != positions[i].col) {
              allOk = false;
              break;
            }
            if (rowC && *rowC != positions[i].row) {
              allOk = false;
              break;
            }
          }
          // Check not occupied by a non-group tile
          auto occIt = physToLogical.find(positions[i]);
          if (occIt != physToLogical.end()) {
            allOk = false;
            break;
          }
        }
        if (!allOk)
          continue;

        // Place the group
        for (size_t i = 0; i < group.tiles.size(); i++) {
          Operation *op = group.tiles[i];
          currentPlacement[op] = positions[i];
          physToLogical[positions[i]] = op;
          cascadePlaced.insert(op);
          // Remove from available pool
          availComp.erase(
              std::remove(availComp.begin(), availComp.end(), positions[i]),
              availComp.end());
        }
        placed = true;
        break;
      }
      if (placed)
        break;
    }
    if (!placed) {
      return group.tiles[0]->emitError(
          "SA placer: could not find valid initial placement for cascade group");
    }
  }

  // Place unconstrained tiles (non-cascade). All tile types participate in SA.
  // Core tiles get exclusive slots. Mem/shim tiles can share physical tiles
  // (DMA channel and memory constraints enforced via resource penalty).
  for (auto lt : logicalTiles) {
    Operation *op = lt.getOperation();
    if (constrainedTiles.count(op) || cascadePlaced.count(op))
      continue;

    AIETileType type = lt.getTileType();
    TileID pos;
    bool found = false;

    if (type == AIETileType::CoreTile) {
      for (size_t i = compIdx; i < availComp.size(); i++) {
        if (isLegalPosition(op, availComp[i])) {
          pos = availComp[i];
          std::swap(availComp[i], availComp[compIdx]);
          compIdx++;
          found = true;
          break;
        }
      }
      if (!found)
        return lt.emitError("no available physical tile for placement");
      currentPlacement[op] = pos;
      physToLogical[pos] = op;
    } else if (type == AIETileType::MemTile) {
      if (!availMem.empty()) {
        pos = availMem[memIdx % availMem.size()];
        memIdx++;
        found = isLegalPosition(op, pos);
      }
      if (!found)
        return lt.emitError("no available physical tile for placement");
      currentPlacement[op] = pos;
    } else if (type == AIETileType::ShimNOCTile) {
      if (!availShim.empty()) {
        pos = availShim[shimIdx % availShim.size()];
        shimIdx++;
        found = isLegalPosition(op, pos);
      }
      if (!found)
        return lt.emitError("no available physical tile for placement");
      currentPlacement[op] = pos;
    } else {
      return lt.emitError("SA placer does not support tile type: ")
             << stringifyAIETileType(type);
    }

    movableTiles.push_back(op);
  }

  // Add cascade group anchor tiles (src tiles) to movableTiles
  // (individual group members are NOT in movableTiles -- only the
  // group as a whole moves via group move operators)
  for (auto &group : cascadeGroups) {
    bool anyConstrained = false;
    for (auto *t : group.tiles) {
      if (constrainedTiles.count(t)) {
        anyConstrained = true;
        break;
      }
    }
    if (!anyConstrained)
      movableTiles.push_back(group.tiles[0]);
  }

  // Phase 4: Initialize bounding boxes
  initBoundingBoxes();

  // Initialize incremental resource tracking
  initResourceTracking();

  int totalCost = computeTotalHPWL() + getResourcePenalty();

  LLVM_DEBUG(llvm::dbgs() << "[SA] Initial cost: " << totalCost
                          << " (HPWL=" << computeTotalHPWL()
                          << ", resPenalty=" << getResourcePenalty()
                          << "), movable tiles: " << movableTiles.size()
                          << ", nets: " << nets.size()
                          << ", fifos: " << fifoBuffers.size() << "\n");

  // If nothing to optimize, skip SA
  if (movableTiles.empty() || nets.empty()) {
    if (failed(mergeMemShimTiles(device)))
      return failure();
  } else {

  // Phase 5: SA main loop
  // SA schedule parameters scaled by design and device characteristics.
  int numMovable = movableTiles.size();
  int numNets = nets.size();
  int deviceSlots = availability.compTiles.size() + numMovable; // total core positions
  int numCascade = cascadeGroups.size();

  // Moves per iteration: proportional to N * sqrt(nets) to balance
  // exploration breadth with net complexity. Capped to avoid excessive
  // runtime on large designs.
  int movesPerIter = std::max(
      static_cast<int>(numMovable * std::sqrt(std::max(numNets, 1))), 100);
  movesPerIter = std::min(movesPerIter, 2000);

  // Max iterations: scale with design complexity (movable tiles × nets).
  // Larger designs need more iterations to converge.
  // Target: small designs ~1500 iters, large designs ~10000 iters.
  // Scale iterations with design complexity. Placement runs once during
  // compilation, so a minute of wall time is acceptable for large designs.
  int maxIters = std::max(10000,
      static_cast<int>(1000 * std::sqrt(
          static_cast<double>(numMovable) * std::max(numNets, 1))));

  // Greedy iterations: proportional to movable tiles
  int greedyIters = 50 * numMovable;

  double coolingRate = 0.999;

  // Estimate initial temperature from sample move deltas.
  // Target: ~80% acceptance at initial T.
  int numSamples = std::max(10 * numMovable, 50);
  double estimatedT = estimateInitialTemperature(numSamples);
  // Cap to avoid runaway temperatures on designs with large penalties.
  double initTemp = std::min(10.0 * estimatedT,
                             static_cast<double>(totalCost) * 2.0);
  initTemp = std::max(initTemp, 1.0); // floor at 1.0

  LLVM_DEBUG(llvm::dbgs() << "[SA] Schedule: movesPerIter=" << movesPerIter
                          << " maxIters=" << maxIters
                          << " cooling=" << coolingRate
                          << " initTemp=" << initTemp << "\n");

  // Cooling: T should reach ~1 at 70% of iterations, leaving 30% for greedy.
  // initTemp * cooling^(0.7*maxIters) = 1
  // cooling = (1/initTemp)^(1/(0.7*maxIters))
  if (initTemp > 1.0) {
    coolingRate = std::pow(1.0 / initTemp, 1.0 / (0.7 * maxIters));
  }

  schedule = SASchedule(initTemp, movesPerIter, maxIters, greedyIters);
  schedule.setCoolingFactor(coolingRate);

  // Track best placement -- only memory-legal placements are "best"
  auto bestPlacement = currentPlacement;
  int bestCost = (getResourcePenalty() == 0) ? totalCost : INT_MAX;

  std::uniform_real_distribution<double> acceptDist(0.0, 1.0);
  std::uniform_real_distribution<double> moveDist(0.0, 1.0);

  int swapAttempts = 0, swapSuccess = 0;
  int cascadeAttempts = 0, cascadeSuccess = 0;
  int shiftAttempts = 0, shiftSuccess = 0;

  // Helper lambda: evaluate and accept/reject a multi-tile move
  auto tryMultiTileMove =
      [&](SmallVector<std::pair<Operation *, TileID>> &moves) {
        // Save old positions for revert and incremental update
        SmallVector<std::pair<Operation *, TileID>> oldPlacements;
        for (auto &[t, _] : moves)
          oldPlacements.push_back({t, currentPlacement[t]});
        int oldResPenalty = cachedResourcePenalty;

        // Remove affected tiles from physToLogical
        for (auto &[t, _] : moves)
          physToLogical.erase(currentPlacement[t]);

        // Evaluate HPWL delta and apply moves
        SmallVector<SmallVector<std::pair<size_t, NetBoundingBox>>>
            allBackups(moves.size());
        int hpwlDelta = 0;
        for (size_t i = 0; i < moves.size(); i++) {
          hpwlDelta += evaluateMove(moves[i].first, moves[i].second,
                                    allBackups[i]);
          currentPlacement[moves[i].first] = moves[i].second;
        }
        for (auto &[t, pos] : moves)
          physToLogical[pos] = t;

        // Incremental resource penalty update
        int newResPenalty = updateResourcePenalty(oldPlacements);
        int delta = hpwlDelta + (newResPenalty - oldResPenalty);

        bool accept = delta <= 0 ||
                      (schedule.getTemperature() > 0 &&
                       acceptDist(rng) <
                           std::exp(-delta / schedule.getTemperature()));

        if (accept) {
          totalCost += delta;
          schedule.recordAccept();
          if (totalCost < bestCost && newResPenalty == 0) {
            bestCost = totalCost;
            bestPlacement = currentPlacement;
          }
        } else {
          // Revert: undo move and resource tracking
          for (auto &[t, pos] : moves)
            physToLogical.erase(pos);
          for (int i = moves.size() - 1; i >= 0; i--) {
            revertMove(allBackups[i]);
            currentPlacement[moves[i].first] = oldPlacements[i].second;
          }
          for (auto &[op, oldPos] : oldPlacements)
            physToLogical[oldPos] = op;
          // Revert resource tracking
          updateResourcePenalty(moves); // moves contains the "new" positions
                                        // which are now "old" to revert from
          schedule.recordReject();
        }
      };

  SmallVector<std::pair<size_t, NetBoundingBox>> backups;

  while (!schedule.limitReached()) {
    for (int m = 0; m < schedule.getMovesPerIter(); m++) {

      // Three move types with design-scaled probabilities:
      // 1. Cascade group swap — proportional to cascade tile fraction
      // 2. Tile swap — primary move type
      // 3. Tile shift — proportional to available empty slots
      double r = moveDist(rng);
      double cascadeProb = (numMovable > 0 && numCascade > 0)
                               ? static_cast<double>(numCascade * 2) / numMovable
                               : 0.0;
      cascadeProb = std::min(cascadeProb, 0.4); // cap at 40%
      double occupancy = (deviceSlots > 0)
                             ? static_cast<double>(numMovable) / deviceSlots
                             : 1.0;
      double shiftProb = (1.0 - occupancy) * (1.0 - cascadeProb);
      // swap gets the remainder

      if (r < cascadeProb && !cascadeGroups.empty()) {
        // Cascade group swap
        cascadeAttempts++;
        std::uniform_int_distribution<size_t> cgDist(
            0, cascadeGroups.size() - 1);
        SmallVector<std::pair<Operation *, TileID>> groupMoves;
        if (!generateGroupSwapMove(cgDist(rng), groupMoves))
          continue;
        cascadeSuccess++;
        tryMultiTileMove(groupMoves);

      } else if (r < 1.0 - shiftProb) {
        // Regular tile swap
        swapAttempts++;
        Operation *tile1 = nullptr, *tile2 = nullptr;
        if (!generateSwapMove(tile1, tile2))
          continue;
        swapSuccess++;

        SmallVector<std::pair<Operation *, TileID>> moves;
        moves.push_back({tile1, currentPlacement[tile2]});
        moves.push_back({tile2, currentPlacement[tile1]});
        tryMultiTileMove(moves);

      } else {
        // Tile shift (only works if empty slots exist)
        shiftAttempts++;
        Operation *tile = nullptr;
        TileID newPos;
        if (!generateShiftMove(tile, newPos))
          continue;
        if (cascadeGroupOf.count(tile))
          continue;
        shiftSuccess++;

        SmallVector<std::pair<Operation *, TileID>> moves;
        moves.push_back({tile, newPos});
        tryMultiTileMove(moves);
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[SA] Iter " << schedule.getIteration()
               << " T=" << schedule.getTemperature() << " cost=" << totalCost
               << " best=" << bestCost
               << " accept=" << schedule.getAcceptanceRatio()
               << (schedule.isGreedy() ? " (greedy)" : "") << "\n");

    if (schedule.getIteration() % 500 == 0)
      llvm::errs() << "[SA] Iter " << schedule.getIteration()
                   << " T=" << schedule.getTemperature() << " cost=" << totalCost
                   << " best=" << bestCost
                   << " swap=" << swapSuccess << "/" << swapAttempts
                   << " cas=" << cascadeSuccess << "/" << cascadeAttempts
                   << " shift=" << shiftSuccess << "/" << shiftAttempts
                   << "\n";

    schedule.cool();
  }

  // Restore best placement
  currentPlacement = bestPlacement;

  // Rebuild physToLogical from best placement
  physToLogical.clear();
  for (auto &[op, pos] : currentPlacement)
    physToLogical[pos] = op;

  // Phase 6: Merge mem/shim tiles that SA placed at the same position
  if (failed(mergeMemShimTiles(device)))
    return failure();

    int finalMemPenalty = getResourcePenalty();
    auto endTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);

    llvm::errs() << "[SA] placement: " << movableTiles.size() << " movable, "
                 << nets.size() << " nets, HPWL=" << bestCost
                 << ", resPenalty=" << finalMemPenalty << ", "
                 << schedule.getIteration() << " iters, " << elapsed.count()
                 << " ms\n";
  } // end of SA loop + post-SA phases (else branch)

  // Phase 7: Generate objectfifo.allocate for overflowing core tiles
  {
    // Use cached memory usage from SA
    auto &memUsage = currentMemUsage;
    int64_t coreCapacity = targetModel->getLocalMemorySize();

    LLVM_DEBUG({
      for (auto &[pos, used] : memUsage) {
        if (pos.row >= 2 && used > coreCapacity)
          llvm::dbgs() << "[SA] Memory overflow at (" << pos.col << ","
                       << pos.row << "): " << used << " / " << coreCapacity
                       << "\n";
      }
    });

    // Find core tiles that exceed capacity
    for (auto &[tilePos, used] : memUsage) {
      if (tilePos.row < 2 || used <= coreCapacity)
        continue;

      // This core tile overflows. Find ObjectFifos allocated on it
      // and try to redirect the largest ones to a neighbor.
      // Collect (fifoIdx, bytes) pairs for fifos on this tile
      SmallVector<std::pair<size_t, int64_t>> fifoOnTile;
      for (size_t fi = 0; fi < fifoBuffers.size(); fi++) {
        const auto &fb = fifoBuffers[fi];
        // Check if this fifo contributes memory to this tile
        // Producer side
        if (fb.producer && currentPlacement.count(fb.producer) &&
            currentPlacement[fb.producer] == tilePos) {
          auto typeIt = tileTypes.find(fb.producer);
          if (typeIt != tileTypes.end() &&
              typeIt->second == AIETileType::CoreTile) {
            fifoOnTile.push_back(
                {fi, fb.producerSizeBytes * fb.producerDepth});
          }
        }
        // Consumer side
        for (size_t ci = 0; ci < fb.consumers.size(); ci++) {
          auto *cons = fb.consumers[ci];
          if (!cons || !currentPlacement.count(cons))
            continue;
          if (currentPlacement[cons] != tilePos)
            continue;
          auto typeIt = tileTypes.find(cons);
          if (typeIt == tileTypes.end() ||
              typeIt->second != AIETileType::CoreTile)
            continue;
          int d = (ci < fb.consumerDepths.size()) ? fb.consumerDepths[ci]
                                                  : fb.producerDepth;
          fifoOnTile.push_back({fi, fb.producerSizeBytes * d});
        }
      }

      // Sort by size descending — redirect largest first
      llvm::sort(fifoOnTile, [](auto &a, auto &b) {
        return a.second > b.second;
      });

      LLVM_DEBUG(llvm::dbgs() << "[SA] Tile (" << tilePos.col << ","
                             << tilePos.row << ") has " << fifoOnTile.size()
                             << " candidate fifos for redirect\n");

      int64_t overflow = used - coreCapacity;

      // Try to redirect fifos to neighbors
      for (auto &[fi, bytes] : fifoOnTile) {
        if (overflow <= 0)
          break;

        const auto &fb = fifoBuffers[fi];
        // objectfifo.allocate only works for 1-to-1 ObjectFifos
        if (fb.consumers.size() != 1)
          continue;

        // Find a neighbor with shared memory and spare capacity
        TileID neighbors[] = {
            {tilePos.col - 1, tilePos.row}, // west
            {tilePos.col, tilePos.row + 1}, // north
            {tilePos.col, tilePos.row - 1}, // south
            {tilePos.col + 1, tilePos.row}, // east (self is already tried)
        };

        for (auto &nbr : neighbors) {
          if (nbr.col < 0 || nbr.col >= targetModel->columns() ||
              nbr.row < 0 || nbr.row >= targetModel->rows())
            continue;
          if (targetModel->getTileType(nbr.col, nbr.row) !=
              AIETileType::CoreTile)
            continue;

          // Check shared memory affinity: delegate must be accessible
          // from both producer and consumer
          TileID prodPos = fb.producer && currentPlacement.count(fb.producer)
                              ? currentPlacement[fb.producer]
                              : TileID{-1, -1};
          TileID consPos = !fb.consumers.empty() && fb.consumers[0] &&
                                   currentPlacement.count(fb.consumers[0])
                               ? currentPlacement[fb.consumers[0]]
                               : TileID{-1, -1};

          bool prodAccess =
              (prodPos.col < 0) ||
              targetModel->isLegalMemAffinity(prodPos.col, prodPos.row,
                                             nbr.col, nbr.row);
          bool consAccess =
              (consPos.col < 0) ||
              targetModel->isLegalMemAffinity(consPos.col, consPos.row,
                                             nbr.col, nbr.row);

          if (!prodAccess || !consAccess)
            continue;

          // Check neighbor has spare capacity
          int64_t nbrUsed = memUsage.count(nbr) ? memUsage[nbr] : 0;
          if (nbrUsed + bytes > coreCapacity)
            continue;

          // Redirect this fifo's buffers to the neighbor
          allocates.push_back({fb.fifoOp, nbr});
          memUsage[nbr] += bytes;
          overflow -= bytes;
          LLVM_DEBUG(llvm::dbgs()
                     << "[SA] Redirecting fifo to neighbor (" << nbr.col
                     << "," << nbr.row << ") to relieve (" << tilePos.col
                     << "," << tilePos.row << ")\n");
          break;
        }
      }

      if (overflow > 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[SA] Warning: core (" << tilePos.col << ","
                   << tilePos.row << ") still overflows by " << overflow
                   << " bytes after allocate generation\n");
      }
    }
  }

  // Write result
  result = currentPlacement;
  return success();
}
