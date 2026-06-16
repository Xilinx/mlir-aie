//===- AIESAPlacer.cpp ------------------------------------------*- C++ -*-===//
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

/// Return neighboring tiles whose memory module `tilePos` can access.
/// The direction matches the allocate op requirement: the source tile
/// can store buffers in the neighbor's memory.
static SmallVector<TileID>
getMemAffinityNeighbors(const AIETargetModel &targetModel, TileID tilePos) {
  SmallVector<TileID> neighbors;
  for (TileID candidate : {TileID{tilePos.col - 1, tilePos.row},
                           TileID{tilePos.col + 1, tilePos.row},
                           TileID{tilePos.col, tilePos.row + 1},
                           TileID{tilePos.col, tilePos.row - 1}}) {
    if (candidate.col < 0 || candidate.col >= targetModel.columns() ||
        candidate.row < 0 || candidate.row >= targetModel.rows())
      continue;
    if (!targetModel.isLegalMemAffinity(tilePos.col, tilePos.row, candidate.col,
                                        candidate.row))
      continue;
    neighbors.push_back(candidate);
  }
  return neighbors;
}

//===----------------------------------------------------------------------===//
// SASchedule
//===----------------------------------------------------------------------===//

SASchedule::SASchedule(double startTemp, int movesPerIter, int maxIters,
                       int greedyIters, const SAConfig &cfg)
    : temperature(startTemp), movesPerIter(movesPerIter), maxIters(maxIters),
      greedyIters(greedyIters),
      windowSize(std::max(movesPerIter, cfg.windowSize)), config(cfg) {}

bool SASchedule::limitReached() const {
  return currIteration >= maxIters ||
         (inGreedyStage && currGreedyIteration >= greedyIters);
}

double SASchedule::getAcceptanceRatio() const {
  int total = acceptCount + rejectCount;
  return (total == 0) ? 1.0 : static_cast<double>(acceptCount) / total;
}

void SASchedule::record(bool accepted) {
  history.push_back(accepted ? 1 : 0);
  ++(accepted ? acceptCount : rejectCount);
  trimWindow();
}

void SASchedule::cool() {
  if (!inGreedyStage) {
    double ratio = getAcceptanceRatio();
    double adaptiveFactor;
    if (ratio > 0.96)
      adaptiveFactor = config.coolFast;
    else if (ratio > 0.8)
      adaptiveFactor = config.coolSteady;
    else if (ratio > 0.15)
      adaptiveFactor = config.coolBalanced;
    else
      adaptiveFactor = config.coolStuck;

    temperature *= adaptiveFactor;

    if (ratio < config.greedyThreshold || temperature < 1e-8) {
      inGreedyStage = true;
      temperature = 0.0;
    }
  } else {
    currGreedyIteration++;
  }
  currIteration++;
}

void SASchedule::trimWindow() {
  while (static_cast<int>(history.size()) > windowSize) {
    if (history.front() == 1)
      acceptCount--;
    else
      rejectCount--;
    history.pop_front();
  }
}

//===----------------------------------------------------------------------===//
// Net model
//===----------------------------------------------------------------------===//

void SAPlacer::buildNetModel(SmallVector<ObjectFifoCreateOp> &objectFifos,
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

    // Cost increase for large multicast
    net.isMulticast = (ofOp.getConsumerTiles().size() >
                       static_cast<unsigned>(config.multicastThreshold));

    size_t idx = nets.size();
    nets.push_back(net);

    // Build reverse index
    for (auto *ep : net.endpoints)
      tileToNetIndices[ep].push_back(idx);
  }
}

int SAPlacer::computeNetHPWL(const NetInfo &net) const {
  int spanCol = net.bb.maxCol - net.bb.minCol;
  int spanRow = net.bb.maxRow - net.bb.minRow;
  int hpwl = spanCol + spanRow;
  if (net.isMulticast)
    hpwl *= config.multicastMultiplier;
  // Area penalty: penalizes spread-out nets that consume more routing tracks.
  // A row-aligned broadcast (area=0) is much cheaper to route than a
  // scattered one (area=cols*rows).
  int area = spanCol * spanRow;
  return hpwl + area;
}

int SAPlacer::computeTotalHPWL() const {
  int total = 0;
  for (const auto &net : nets)
    total += computeNetHPWL(net);
  return total;
}

void SAPlacer::initBoundingBoxes() {
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

int SAPlacer::evaluateMove(
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

void SAPlacer::revertMove(
    SmallVector<std::pair<size_t, NetBoundingBox>> &backups) {
  for (auto &[netIdx, savedBB] : backups)
    nets[netIdx].bb = savedBB;
}

//===----------------------------------------------------------------------===//
// Move generation
//===----------------------------------------------------------------------===//

bool SAPlacer::isLegalPosition(Operation *tile, TileID pos) const {
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

// Add a single fifo's contributions to memory and DMA usage maps.
void SAPlacer::addFifoContribution(size_t fifoIdx, int sign) {
  const auto &fb = fifoBuffers[fifoIdx];
  Operation *prod = fb.producer;
  if (!prod)
    return;
  auto prodPlaceIt = currentPlacement.find(prod);
  if (prodPlaceIt == currentPlacement.end())
    return;
  TileID prodPos = prodPlaceIt->second;
  auto prodTypeIt = tileTypes.find(prod);
  bool prodIsCore = prodTypeIt != tileTypes.end() &&
                    prodTypeIt->second == AIETileType::CoreTile;
  bool prodIsShim = prodTypeIt != tileTypes.end() &&
                    (prodTypeIt->second == AIETileType::ShimNOCTile ||
                     prodTypeIt->second == AIETileType::ShimPLTile);

  // Charge memory based on connection topology:
  //   intratile     → max(sizes) * max(depths) on shared tile
  //   shared-mem    → max(sizes) * max(depths) on chosen adjacent tile
  //   core↔core DMA → prod: prodSize*dmaDepth, cons: consSize*dmaDepth
  //   core→mem/shim → prod: dmaDepth, cons: declaredDepth (skip shim)
  //   mem/shim→core → prod: declaredDepth (skip shim/link), cons: dmaDepth
  //   mem↔mem       → both declaredDepth (skip shim/link)
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
    bool consIsCore = consTypeIt != tileTypes.end() &&
                      consTypeIt->second == AIETileType::CoreTile;
    bool consIsShim = consTypeIt != tileTypes.end() &&
                      (consTypeIt->second == AIETileType::ShimNOCTile ||
                       consTypeIt->second == AIETileType::ShimPLTile);

    // Declared depth (for shared-mem) and DMA depth (for DMA connections).
    // The appropriate depth is selected in each branch below based on
    // whether the connection uses DMA or shared memory.
    int consDeclDepth = fb.consumerDepthAt(ci);
    int consDMADepth = (ci < fb.consumerDMADepths.size())
                           ? fb.consumerDMADepths[ci]
                           : fb.producerDMADepth;

    // Check shared memory between core tiles (skip if fifo forces DMA)
    bool isSharedMem =
        !fb.forcesDMA && prodIsCore && consIsCore && prodPos != consPos &&
        (targetModel->isLegalMemAffinity(prodPos.col, prodPos.row, consPos.col,
                                         consPos.row) ||
         targetModel->isLegalMemAffinity(consPos.col, consPos.row, prodPos.col,
                                         prodPos.row));

    if (prodPos == consPos) {
      // Intratile: shared-mem, use declared depths
      int depth = std::max(fb.producerDepth, consDeclDepth);
      int64_t maxSize = std::max(fb.producerSizeBytes, fb.consumerSizeBytes);
      int64_t bytes = maxSize * depth;
      currentMemUsage[prodPos] += sign * bytes;
    } else if (isSharedMem) {
      // Shared memory: use declared depths
      bool rightShared = targetModel->isLegalMemAffinity(
          prodPos.col, prodPos.row, consPos.col, consPos.row);
      bool leftShared = targetModel->isLegalMemAffinity(
          consPos.col, consPos.row, prodPos.col, prodPos.row);
      TileID bufTile;
      if (rightShared && leftShared) {
        if (sign > 0) {
          int64_t prodUsed =
              currentMemUsage.count(prodPos) ? currentMemUsage[prodPos] : 0;
          int64_t consUsed =
              currentMemUsage.count(consPos) ? currentMemUsage[consPos] : 0;
          bufTile = (prodUsed <= consUsed) ? prodPos : consPos;
          // Only track shared-mem destination for single-consumer fifos.
          // allocate ops require all endpoints share access to the delegate
          // tile, which is only guaranteed for single-consumer fifos.
          if (fb.consumers.size() == 1)
            sharedMemDestination[fifoIdx] = bufTile;
        } else {
          auto it = sharedMemDestination.find(fifoIdx);
          bufTile = (it != sharedMemDestination.end()) ? it->second : prodPos;
        }
      } else if (rightShared) {
        bufTile = consPos;
      } else {
        bufTile = prodPos;
      }
      int depth = std::max(fb.producerDepth, consDeclDepth);
      int64_t maxSize = std::max(fb.producerSizeBytes, fb.consumerSizeBytes);
      currentMemUsage[bufTile] += sign * maxSize * depth;
    } else if (prodIsCore && consIsCore) {
      // Non-adjacent core-to-core: DMA, use DMA depths (maxAcquire+1)
      if (!prodCharged) {
        int64_t bytes = fb.producerSizeBytes * fb.producerDMADepth;
        currentMemUsage[prodPos] += sign * bytes;
        prodCharged = true;
      }
      int64_t consBytes = fb.consumerSizeBytes * consDMADepth;
      currentMemUsage[consPos] += sign * consBytes;
    } else if (prodIsCore && !consIsCore) {
      // CoreTile producer → MemTile/Shim consumer
      // Producer (core): DMA depth. Consumer (MemTile): declared depth.
      if (!prodCharged) {
        int64_t bytes = fb.producerSizeBytes * fb.producerDMADepth;
        currentMemUsage[prodPos] += sign * bytes;
        prodCharged = true;
      }
      if (!consIsShim) {
        // MemTile consumer: use declared depth
        int64_t consBytes = fb.consumerSizeBytes * consDeclDepth;
        currentMemUsage[consPos] += sign * consBytes;
      }
    } else if (!prodIsCore && consIsCore) {
      // MemTile/Shim producer → CoreTile consumer
      // Producer (MemTile): declared depth. Consumer (core): DMA depth.
      if (!prodCharged && !prodIsShim && !fb.linkSharedProd) {
        currentMemUsage[prodPos] +=
            sign * fb.producerSizeBytes * fb.producerDepth;
        prodCharged = true;
      }
      int64_t consBytes = fb.consumerSizeBytes * consDMADepth;
      currentMemUsage[consPos] += sign * consBytes;
    } else if (!prodIsCore && !consIsCore) {
      // MemTile/Shim → MemTile/Shim: both use declared depths
      if (!prodCharged && !prodIsShim && !fb.linkSharedProd) {
        currentMemUsage[prodPos] +=
            sign * fb.producerSizeBytes * fb.producerDepth;
        prodCharged = true;
      }
      if (!consIsShim) {
        int64_t consBytes = fb.consumerSizeBytes * consDeclDepth;
        currentMemUsage[consPos] += sign * consBytes;
      }
    }

    // DMA contribution — shared-memory connections use no DMA
    if (prodPos != consPos && !isSharedMem) {
      currentDMAUsage[consPos].first += sign;
      prodNeedsDMA = true;
    }
  }

  if (prodNeedsDMA)
    currentDMAUsage[prodPos].second += sign;
}

void SAPlacer::initResourceTracking() {
  currentMemUsage.clear();
  currentDMAUsage.clear();
  sharedMemDestination.clear();

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

  // Add static buffers
  for (auto &[op, bufSize] : staticBufferSizes) {
    auto posIt = currentPlacement.find(op);
    if (posIt != currentPlacement.end())
      currentMemUsage[posIt->second] += bufSize;
  }
  // Add core stack overhead (reserved by assign-buffer-addresses pass)
  for (auto &[op, pos] : currentPlacement) {
    auto typeIt = tileTypes.find(op);
    if (typeIt == tileTypes.end() || typeIt->second != AIETileType::CoreTile)
      continue;
    auto it = stackSizes.find(op);
    if (it != stackSizes.end())
      currentMemUsage[pos] += it->second;
  }
  cachedResourcePenalty = computePenalty();
}

int SAPlacer::computeMemoryPressure() const {
  int64_t coreCapacity = targetModel->getLocalMemorySize();
  int64_t memTileCapacity = targetModel->getMemTileSize();
  int pressure = 0;

  for (auto &[tilePos, used] : currentMemUsage) {
    auto tileType = targetModel->getTileType(tilePos.col, tilePos.row);
    if (tileType == AIETileType::ShimNOCTile ||
        tileType == AIETileType::ShimPLTile)
      continue;
    int64_t capacity =
        (tileType == AIETileType::MemTile) ? memTileCapacity : coreCapacity;
    double utilization = static_cast<double>(used) / capacity;
    if (utilization > config.pressureThreshold) {
      double excess = (utilization - config.pressureThreshold) /
                      (1.0 - config.pressureThreshold);
      pressure += static_cast<int>(config.pressurePerKB * excess * excess);
    }
  }
  return pressure;
}

int SAPlacer::computePenalty() const {
  return computeMemSpilloverPenalty() + computeCoreOverflowPenalty() +
         computeDMAChannelPenalty() + computeBDCountPenalty();
}

// Simulates per-buffer MemTile allocation with neighbor spillover.
// Matches the stateful transform's allocation strategy: buffers are
// collected globally across all MemTile columns, sorted largest-first,
// and each buffer is placed on its home column when possible, spilling
// to the neighbor with the most remaining capacity when the home is full.
int SAPlacer::computeMemSpilloverPenalty() const {
  int64_t memTileCapacity = targetModel->getMemTileSize();
  int numCols = targetModel->columns();
  int penalty = 0;

  // Collect all MemTile buffers globally, each tagged with its home column.
  // Pair: (bufferSize, homeColumn).
  llvm::SmallVector<std::pair<int64_t, int>> allBufs;
  llvm::SmallVector<int64_t> memTileBaseline(numCols, 0);

  for (size_t fi = 0; fi < fifoBuffers.size(); fi++) {
    const auto &fb = fifoBuffers[fi];
    if (!fb.producer)
      continue;
    auto prodIt = currentPlacement.find(fb.producer);
    if (prodIt == currentPlacement.end())
      continue;
    TileID prodPos = prodIt->second;
    auto prodTypeIt = tileTypes.find(fb.producer);
    if (prodTypeIt == tileTypes.end())
      continue;
    if (prodTypeIt->second != AIETileType::MemTile)
      continue;
    if (fb.linkSharedProd)
      continue;

    int col = prodPos.col;
    for (int d = 0; d < fb.producerDepth; d++)
      allBufs.push_back({fb.producerSizeBytes, col});

    for (size_t ci = 0; ci < fb.consumers.size(); ci++) {
      auto *cons = fb.consumers[ci];
      if (!cons)
        continue;
      auto consIt = currentPlacement.find(cons);
      if (consIt == currentPlacement.end())
        continue;
      auto consTypeIt = tileTypes.find(cons);
      if (consTypeIt == tileTypes.end() ||
          consTypeIt->second != AIETileType::MemTile)
        continue;
      int consCol = consIt->second.col;
      int consDepth = fb.consumerDepthAt(ci);
      for (int d = 0; d < consDepth; d++)
        allBufs.push_back({fb.consumerSizeBytes, consCol});
    }
  }

  // Static buffers as baseline (non-spillable).
  for (auto &[op, bufSize] : staticBufferSizes) {
    auto posIt = currentPlacement.find(op);
    if (posIt == currentPlacement.end())
      continue;
    auto typeIt = tileTypes.find(op);
    if (typeIt == tileTypes.end() || typeIt->second != AIETileType::MemTile)
      continue;
    memTileBaseline[posIt->second.col] += bufSize;
  }

  // remaining[col] = capacity available after baseline.
  llvm::SmallVector<int64_t> remaining(numCols, 0);
  for (int col = 0; col < numCols; col++) {
    if (targetModel->getTileType(col, 1) != AIETileType::MemTile)
      continue;
    remaining[col] =
        std::max(memTileCapacity - memTileBaseline[col], int64_t{0});
  }

  // Sort globally by size descending (matching stateful transform).
  llvm::sort(allBufs,
             [](const auto &a, const auto &b) { return a.first > b.first; });

  // Place each buffer: try home first, then neighbor with most capacity.
  for (auto &[bufSize, homeCol] : allBufs) {
    if (remaining[homeCol] >= bufSize) {
      remaining[homeCol] -= bufSize;
    } else {
      // Pick neighbor with the most remaining capacity.
      int bestNbr = -1;
      int64_t bestCap = -1;
      for (int nbr : {homeCol - 1, homeCol + 1}) {
        if (nbr < 0 || nbr >= numCols)
          continue;
        if (targetModel->getTileType(nbr, 1) != AIETileType::MemTile)
          continue;
        if (remaining[nbr] >= bufSize && remaining[nbr] > bestCap) {
          bestCap = remaining[nbr];
          bestNbr = nbr;
        }
      }
      if (bestNbr >= 0) {
        remaining[bestNbr] -= bufSize;
      } else {
        // Ceiling division: bytes → KB.
        penalty += ((bufSize + 1023) / 1024) * config.memPenaltyPerKB;
      }
    }
  }

  return penalty;
}

// Penalizes core tiles whose buffer usage exceeds local memory, accounting
// for intratile fifos that can be relocated to a neighbor via allocate.
int SAPlacer::computeCoreOverflowPenalty() const {
  int64_t coreCapacity = targetModel->getLocalMemorySize();
  int penalty = 0;

  for (auto &[tilePos, used] : currentMemUsage) {
    if (targetModel->getTileType(tilePos.col, tilePos.row) !=
        AIETileType::CoreTile)
      continue;
    if (used <= coreCapacity)
      continue;
    int64_t overflow = used - coreCapacity;

    // Build accessible neighbor spare capacities.
    llvm::SmallVector<int64_t, 4> nbrSpares;
    for (TileID nbr : getMemAffinityNeighbors(*targetModel, tilePos)) {
      auto it = currentMemUsage.find(nbr);
      int64_t nbrUsed = (it != currentMemUsage.end()) ? it->second : 0;
      int64_t spare = std::max(coreCapacity - nbrUsed, int64_t{0});
      if (spare > 0)
        nbrSpares.push_back(spare);
    }
    llvm::sort(nbrSpares, std::greater<int64_t>());

    // Collect relocatable intratile fifo sizes on this tile.
    int64_t unresolved = overflow;
    auto opIt = physToLogical.find(tilePos);
    if (opIt != physToLogical.end()) {
      auto fifoIt = tileToFifoIndices.find(opIt->second);
      if (fifoIt != tileToFifoIndices.end()) {
        llvm::SmallVector<int64_t, 8> fifoSizes;
        llvm::DenseSet<size_t> seen;
        for (size_t fi : fifoIt->second) {
          if (!seen.insert(fi).second)
            continue;
          const auto &fb = fifoBuffers[fi];
          if (fb.consumers.size() != 1 || !fb.producer || !fb.consumers[0])
            continue;
          auto prodIt = currentPlacement.find(fb.producer);
          auto consIt = currentPlacement.find(fb.consumers[0]);
          if (prodIt == currentPlacement.end() ||
              consIt == currentPlacement.end())
            continue;
          if (prodIt->second != tilePos || consIt->second != tilePos)
            continue;
          int consDepth = fb.consumerDepths.empty() ? fb.producerDepth
                                                    : fb.consumerDepths[0];
          int64_t sz = std::max(fb.producerSizeBytes, fb.consumerSizeBytes) *
                       std::max(fb.producerDepth, consDepth);
          fifoSizes.push_back(sz);
        }
        llvm::sort(fifoSizes, std::greater<int64_t>());

        // Greedy packing: assign each fifo to the first neighbor that fits.
        int64_t totalAbsorbed = 0;
        for (int64_t sz : fifoSizes) {
          for (int64_t &spare : nbrSpares) {
            if (spare >= sz) {
              spare -= sz;
              totalAbsorbed += sz;
              break;
            }
          }
        }
        unresolved = overflow - std::min(totalAbsorbed, overflow);
      }
    }

    if (unresolved > 0)
      penalty += ((unresolved + 1023) / 1024) * config.memPenaltyPerKB;
  }

  return penalty;
}

int SAPlacer::computeDMAChannelPenalty() const {
  int penalty = 0;
  for (auto &[tilePos, usage] : currentDMAUsage) {
    auto [maxIn, maxOut] = getDMACapacity(*targetModel, tilePos);
    if (usage.first > maxIn)
      penalty += (usage.first - maxIn) * config.dmaPenaltyPerChannel;
    if (usage.second > maxOut)
      penalty += (usage.second - maxOut) * config.dmaPenaltyPerChannel;
  }
  return penalty;
}

// Penalizes MemTiles that exceed their BD slot limit.
int SAPlacer::computeBDCountPenalty() const {
  int memTileBDMax = targetModel->getNumBDs(AIETileType::MemTile);
  int penalty = 0;

  llvm::DenseMap<TileID, int> memTileBDs;
  for (const auto &fb : fifoBuffers) {
    if (fb.producer) {
      auto posIt = currentPlacement.find(fb.producer);
      auto typeIt = tileTypes.find(fb.producer);
      if (posIt != currentPlacement.end() && typeIt != tileTypes.end() &&
          typeIt->second == AIETileType::MemTile && !fb.linkSharedProd)
        memTileBDs[posIt->second] += fb.producerDepth;
    }
    for (size_t ci = 0; ci < fb.consumers.size(); ci++) {
      auto *cons = fb.consumers[ci];
      if (!cons)
        continue;
      auto posIt = currentPlacement.find(cons);
      auto typeIt = tileTypes.find(cons);
      if (posIt == currentPlacement.end() || typeIt == tileTypes.end() ||
          typeIt->second != AIETileType::MemTile)
        continue;
      int depth = fb.consumerDepthAt(ci);
      memTileBDs[posIt->second] += depth;
    }
  }
  for (auto &[tilePos, totalBDs] : memTileBDs) {
    if (totalBDs > memTileBDMax)
      penalty += (totalBDs - memTileBDMax) * config.dmaPenaltyPerChannel;
  }

  return penalty;
}

int SAPlacer::computeAdjacencyPenalty(
    const Adjacency &adj, ArrayRef<std::pair<int, int>> validOffsets,
    int weight) const {
  int penalty = 0;
  for (auto [first, second] : adj.edges) {
    auto firstIt = currentPlacement.find(first.getOperation());
    auto secondIt = currentPlacement.find(second.getOperation());
    if (firstIt == currentPlacement.end() || secondIt == currentPlacement.end())
      continue;

    TileID firstPos = firstIt->second;
    TileID secondPos = secondIt->second;

    // Find minimum Manhattan distance to any valid offset.
    int minDist = INT_MAX;
    for (auto [dc, dr] : validOffsets) {
      int dist = std::abs(secondPos.col - (firstPos.col + dc)) +
                 std::abs(secondPos.row - (firstPos.row + dr));
      minDist = std::min(minDist, dist);
    }
    if (minDist > 0)
      penalty += weight * minDist;
  }
  return penalty;
}

void SAPlacer::generateAllocates() {
  int64_t coreCapacity = targetModel->getLocalMemorySize();
  DenseSet<size_t> emittedFifoIndices;

  // Case A: Intratile fifo relocation for overflowing core tiles.
  // Snapshot overflowing tiles first to avoid iterator invalidation
  // (delegate updates modify currentMemUsage during packing).
  SmallVector<TileID> overflowTiles;
  for (auto &[tilePos, used] : currentMemUsage) {
    if (targetModel->getTileType(tilePos.col, tilePos.row) ==
            AIETileType::CoreTile &&
        used > coreCapacity)
      overflowTiles.push_back(tilePos);
  }
  for (auto &tilePos : overflowTiles) {

    // Build accessible neighbor spare capacities: (tilePos, spare)
    SmallVector<std::pair<TileID, int64_t>, 4> nbrSpares;
    for (TileID nbr : getMemAffinityNeighbors(*targetModel, tilePos)) {
      auto it = currentMemUsage.find(nbr);
      int64_t nbrUsed = (it != currentMemUsage.end()) ? it->second : 0;
      int64_t spare = std::max(coreCapacity - nbrUsed, int64_t{0});
      if (spare > 0)
        nbrSpares.push_back({nbr, spare});
    }
    llvm::sort(nbrSpares, [](const auto &a, const auto &b) {
      return a.second > b.second;
    });

    // Collect relocatable intratile fifos on this tile
    auto opIt = physToLogical.find(tilePos);
    if (opIt == physToLogical.end())
      continue;
    auto fifoIt = tileToFifoIndices.find(opIt->second);
    if (fifoIt == tileToFifoIndices.end())
      continue;

    // (fifoIndex, totalSize)
    SmallVector<std::pair<size_t, int64_t>, 8> relocFifos;
    DenseSet<size_t> seen;
    for (size_t fi : fifoIt->second) {
      if (!seen.insert(fi).second)
        continue;
      const auto &fb = fifoBuffers[fi];
      if (fb.consumers.size() != 1 || !fb.producer || !fb.consumers[0])
        continue;
      auto prodIt = currentPlacement.find(fb.producer);
      auto consIt = currentPlacement.find(fb.consumers[0]);
      if (prodIt == currentPlacement.end() || consIt == currentPlacement.end())
        continue;
      if (prodIt->second != tilePos || consIt->second != tilePos)
        continue;
      int consDepth =
          fb.consumerDepths.empty() ? fb.producerDepth : fb.consumerDepths[0];
      int64_t sz = std::max(fb.producerSizeBytes, fb.consumerSizeBytes) *
                   std::max(fb.producerDepth, consDepth);
      relocFifos.push_back({fi, sz});
    }
    llvm::sort(relocFifos, [](const auto &a, const auto &b) {
      return a.second > b.second;
    });

    // Greedy: assign each whole fifo to the first neighbor that fits.
    // Verify against actual currentMemUsage to avoid overflowing delegate.
    for (auto &[fifoIdx, fifoSize] : relocFifos) {
      if (!emittedFifoIndices.insert(fifoIdx).second)
        continue;
      for (auto &[nbrPos, nbrSpare] : nbrSpares) {
        if (nbrSpare >= fifoSize) {
          // Double-check: delegate won't overflow after adding this fifo
          auto memIt = currentMemUsage.find(nbrPos);
          int64_t delegateUsed =
              (memIt != currentMemUsage.end()) ? memIt->second : 0;
          if (delegateUsed + fifoSize > coreCapacity)
            continue;
          nbrSpare -= fifoSize;
          currentMemUsage[nbrPos] += fifoSize;
          allocates.push_back({fifoBuffers[fifoIdx].fifoOp, nbrPos});
          break;
        }
      }
    }
  }

  // Case B: Shared-mem destination communication.
  // sharedMemDestination is populated by addFifoContribution during
  // initResourceTracking, which runs on the final placement before this.
  for (auto &[fifoIdx, destTile] : sharedMemDestination) {
    if (emittedFifoIndices.contains(fifoIdx))
      continue;
    const auto &fb = fifoBuffers[fifoIdx];
    if (!fb.producer || fb.consumers.empty() || !fb.consumers[0])
      continue;
    auto prodIt = currentPlacement.find(fb.producer);
    auto consIt = currentPlacement.find(fb.consumers[0]);
    if (prodIt == currentPlacement.end() || consIt == currentPlacement.end())
      continue;
    if (prodIt->second == consIt->second)
      continue;
    if (destTile != prodIt->second)
      allocates.push_back({fb.fifoOp, destTile});
  }
}

int SAPlacer::updateResourcePenalty(
    const SmallVector<std::pair<Operation *, TileID>> &oldPlacements) {
  // Collect all fifo indices affected by the moved tiles
  DenseSet<size_t> affectedFifos;
  for (auto &[op, _] : oldPlacements) {
    auto it = tileToFifoIndices.find(op);
    if (it != tileToFifoIndices.end())
      for (size_t idx : it->second)
        affectedFifos.insert(idx);
  }

  // Update static buffer and stack contributions for moved tiles
  for (auto &[op, oldPos] : oldPlacements) {
    auto bufIt = staticBufferSizes.find(op);
    if (bufIt != staticBufferSizes.end())
      currentMemUsage[oldPos] -= bufIt->second;
    auto stackIt = stackSizes.find(op);
    if (stackIt != stackSizes.end())
      currentMemUsage[oldPos] -= stackIt->second;
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

  // Add new static buffer and stack contributions
  for (auto &[op, _] : oldPlacements) {
    TileID newPos = currentPlacement[op];
    auto bufIt = staticBufferSizes.find(op);
    if (bufIt != staticBufferSizes.end())
      currentMemUsage[newPos] += bufIt->second;
    auto stackIt = stackSizes.find(op);
    if (stackIt != stackSizes.end())
      currentMemUsage[newPos] += stackIt->second;
  }

  cachedResourcePenalty = computePenalty();
  return cachedResourcePenalty;
}

//===----------------------------------------------------------------------===//
// Move generation
//===----------------------------------------------------------------------===//

bool SAPlacer::generateMove(
    SmallVector<std::pair<Operation *, TileID>> &moves) {
  if (movableTiles.empty())
    return false;

  for (int attempt = 0; attempt < config.maxMoveAttempts; attempt++) {
    // Pick a random movable tile.
    std::uniform_int_distribution<size_t> tileDist(0, movableTiles.size() - 1);
    Operation *tile = movableTiles[tileDist(rng)];

    auto typeIt = tileTypes.find(tile);
    if (typeIt == tileTypes.end())
      continue;
    AIETileType type = typeIt->second;

    // Pick a random target position of matching type.
    const auto &candidates = (type == AIETileType::CoreTile)
                                 ? availability.compTiles
                                 : availability.nonCompTiles;
    if (candidates.empty())
      continue;

    std::uniform_int_distribution<size_t> posDist(0, candidates.size() - 1);
    TileID targetPos = candidates[posDist(rng)];

    // Filter by tile type for non-compute (nonCompTiles has mixed types).
    if (type != AIETileType::CoreTile &&
        targetModel->getTileType(targetPos.col, targetPos.row) != type)
      continue;

    TileID currentPos = currentPlacement[tile];
    if (targetPos == currentPos)
      continue;

    if (!satisfiesConstraints(tile, targetPos))
      continue;

    // Check if target is occupied by another movable tile.
    Operation *occupant = nullptr;
    for (auto *t : movableTiles) {
      if (t != tile && currentPlacement[t] == targetPos) {
        occupant = t;
        break;
      }
    }

    if (occupant) {
      // Swap: both tiles must accept each other's position.
      if (tileTypes[occupant] != type)
        continue;
      if (!satisfiesConstraints(occupant, currentPos))
        continue;
      moves.clear();
      moves.push_back({tile, targetPos});
      moves.push_back({occupant, currentPos});
    } else {
      // Shift: target is empty.
      if (!isLegalPosition(tile, targetPos))
        continue;
      moves.clear();
      moves.push_back({tile, targetPos});
    }
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Initial temperature estimation
//===----------------------------------------------------------------------===//

double SAPlacer::estimateInitialTemperature(int numSamples) {
  // Collect delta costs from random moves
  std::vector<double> deltaCosts;
  deltaCosts.reserve(numSamples);

  SmallVector<std::pair<size_t, NetBoundingBox>> backups;

  for (int i = 0; i < numSamples; i++) {
    SmallVector<std::pair<Operation *, TileID>> moves;
    if (!generateMove(moves))
      continue;

    // Use the first tile's HPWL delta as cost sample.
    int delta = evaluateMove(moves[0].first, moves[0].second, backups);
    deltaCosts.push_back(delta);
    revertMove(backups);
  }

  if (deltaCosts.empty())
    return 1.0;

  // Equilibrium binary search: find T where E[delta_cost] ≈ 0,
  // improving and worsening moves roughly cancel out and
  // SA accepts most moves freely - good starting point
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
  double highT = config.tempSearchCeiling * maxDelta;
  double T = 1.0;

  for (int iter = 0; iter < config.tempSearchIters; iter++) {
    double ev = getExpectedValue(T);
    if (std::abs(highT - lowT) / std::max(T, config.tempSearchEpsilon) <
        config.tempSearchTolerance)
      break;
    if (ev > 0) {
      // Expected value positive → worsening moves dominate → T too high
      // (accepting too many bad moves). Lower the ceiling.
      highT = T;
    } else {
      // Expected value negative → improving moves dominate → T too low.
      // Raise the floor.
      lowT = T;
    }
    T = (lowT + highT) / 2.0;
  }

  return T;
}

//===----------------------------------------------------------------------===//
// Memory weight estimation (for initial placement sorting)
//===----------------------------------------------------------------------===//

int64_t SAPlacer::computeTileMemoryWeight(Operation *tile) const {
  int64_t weight = 0;

  // Static buffers (aie.buffer ops)
  auto bufIt = staticBufferSizes.find(tile);
  if (bufIt != staticBufferSizes.end())
    weight += bufIt->second;

  // Core stack overhead
  auto stackIt = stackSizes.find(tile);
  if (stackIt != stackSizes.end())
    weight += stackIt->second;

  // Fifo endpoint contributions
  for (size_t i = 0; i < fifoBuffers.size(); i++) {
    const auto &fb = fifoBuffers[i];
    bool isProd = (fb.producer == tile);
    int consIdx = -1;
    for (size_t ci = 0; ci < fb.consumers.size(); ci++) {
      if (fb.consumers[ci] == tile) {
        consIdx = static_cast<int>(ci);
        break;
      }
    }
    if (!isProd && consIdx < 0)
      continue;

    if (isProd && consIdx >= 0) {
      // Intratile: max(sizes) * max(depths)
      int depth = std::max(fb.producerDepth, fb.consumerDepths[consIdx]);
      int64_t sz = std::max(fb.producerSizeBytes, fb.consumerSizeBytes);
      weight += sz * depth;
    } else if (isProd) {
      weight += fb.producerSizeBytes * fb.producerDepth;
    } else {
      weight += fb.consumerSizeBytes * fb.consumerDepths[consIdx];
    }
  }
  return weight;
}

//===----------------------------------------------------------------------===//
// Post-SA mem/shim merge
//===----------------------------------------------------------------------===//

LogicalResult SAPlacer::mergeMemShimTiles(DeviceOp device) {
  // After SA, each logical tile has its own physical position.
  // Mem/shim tiles in the same column and of the same type share a
  // physical tile. The SA's penalty model already enforces DMA capacity
  // and memory constraints, so merge unconditionally.

  // Group non-compute logical tiles by (col, tileType)
  llvm::DenseMap<std::pair<int, AIETileType>, SmallVector<Operation *>> groups;

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
    for (size_t i = 1; i < ops.size(); i++)
      currentPlacement[ops[i]] = mergeTarget;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SA placement
//===----------------------------------------------------------------------===//

LogicalResult SAPlacer::place(DeviceOp device) {
  startTime = std::chrono::steady_clock::now();

  // Seed RNG
  if (rngSeed == 0) {
    rng.seed(std::random_device{}());
  } else {
    rng.seed(rngSeed);
  }

  // Phase 1-2: Collect IR ops, build net model, fifo info, cascade groups.
  if (failed(collectAndBuildModel(device)))
    return failure();
  // Phase 3: Classify tiles by type, generate initial placement.
  if (failed(generateInitialPlacement()))
    return failure();
  // Phase 4: Initialize bounding boxes, resource tracking, initial cost.
  initializeSAState();
  // Phase 5: SA main loop (skip if nothing to optimize).
  runSAMainLoop();
  // Phase 6: Restore best, generate allocates, merge mem/shim, stats.
  return finalizePlacement(device);
}

LogicalResult SAPlacer::collectAndBuildModel(DeviceOp device) {
  collected = collectOperations(device);
  auto &logicalTiles = collected.logicalTiles;
  auto &objectFifos = collected.objectFifos;
  auto &objectFifoLinks = collected.objectFifoLinks;

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

  // Collect core stack sizes (reserved by assign-buffer-addresses pass)
  device.walk([&](CoreOp coreOp) {
    auto *tileOp = coreOp.getTile().getDefiningOp();
    if (!tileOp || !isa<LogicalTileOp>(tileOp))
      return;
    stackSizes[tileOp] = coreOp.getStackSize();
  });

  // Build net model, fifo buffer info, and cascade groups
  buildNetModel(objectFifos, objectFifoLinks);
  buildFifoBufferInfo(device, objectFifos, objectFifoLinks);
  cascadeAdjacency = buildCascadeAdjacency(collected.cascadeFlows);
  LLVM_DEBUG(llvm::dbgs() << "[SA] Cascade adjacency: "
                          << cascadeAdjacency.edges.size() << " edges\n");

  return success();
}

void SAPlacer::buildFifoBufferInfo(DeviceOp device,
                                   ArrayRef<ObjectFifoCreateOp> objectFifos,
                                   ArrayRef<ObjectFifoLinkOp> objectFifoLinks) {
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
        size_t idx = i + 1; // consumer depths start at index 1
        int d = (idx < values.size())
                    ? static_cast<int>(cast<IntegerAttr>(values[idx]).getInt())
                    : fb.producerDepth;
        fb.consumerDepths.push_back(d);
      }
    } else {
      int d =
          static_cast<int>(cast<IntegerAttr>(ofOp.getElemNumber()).getInt());
      fb.producerDepth = d;
      for (size_t i = 0; i < fb.consumers.size(); i++)
        fb.consumerDepths.push_back(d);
    }

    // Determine if this fifo forces DMA even when tiles are adjacent.
    // Must match AIEObjectFifoStatefulTransform::requiresDMAs().
    bool isLinked = false;
    for (auto link : objectFifoLinks) {
      for (auto in : link.getInputObjectFifos())
        if (in.name() == ofOp.getSymName())
          isLinked = true;
      for (auto out : link.getOutputObjectFifos())
        if (out.name() == ofOp.getSymName())
          isLinked = true;
    }
    fb.forcesDMA = ofOp.getVia_DMA() || ofOp.getRepeatCount().has_value() ||
                   ofOp.getConsumerElemType().has_value() ||
                   !ofOp.getDimensionsToStream().empty() || isLinked;
    // For linked output fifos, producer buffers share memory with
    // the link input fifo's consumer buffer on the MemTile.
    // Mark to skip producer-side MemTile memory charge.
    if (isLinked) {
      for (auto link : objectFifoLinks) {
        for (auto out : link.getOutputObjectFifos()) {
          if (out.name() == ofOp.getSymName())
            fb.linkSharedProd = true;
        }
      }
    }
    fifoBuffers.push_back(fb);
  }

  LLVM_DEBUG(llvm::dbgs() << "[SA] seed=" << rngSeed << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[SA] Collected " << fifoBuffers.size()
                          << " ObjectFifo buffer requirements\n");

  // Compute DMA depths per endpoint: maxAcquire + 1 from core body
  // acquire ops, matching findObjectFifoSize() in the stateful transform.
  // DMA depths are used when the connection requires DMA; declared
  // depths are used for shared-mem connections. The SA selects the
  // appropriate depth in addFifoContribution based on placement.
  DenseMap<std::pair<Operation *, Operation *>, int> endpointMaxAcquire;
  device.walk([&](CoreOp coreOp) {
    auto *tileOp = coreOp.getTile().getDefiningOp();
    if (!tileOp || !isa<LogicalTileOp>(tileOp))
      return;
    coreOp.walk([&](ObjectFifoAcquireOp acqOp) {
      auto fifoOp = acqOp.getObjectFifo();
      if (!fifoOp)
        return;
      auto endpoint = std::pair{static_cast<Operation *>(fifoOp), tileOp};
      int acqNum = acqOp.acqNumber();
      auto it = endpointMaxAcquire.find(endpoint);
      if (it == endpointMaxAcquire.end() || acqNum > it->second)
        endpointMaxAcquire[endpoint] = acqNum;
    });
  });

  for (auto &fb : fifoBuffers) {
    // Producer DMA depth
    fb.producerDMADepth = fb.producerDepth;
    if (fb.producer) {
      auto it = endpointMaxAcquire.find(std::pair{fb.fifoOp, fb.producer});
      if (it != endpointMaxAcquire.end()) {
        int maxAcq = it->second;
        fb.producerDMADepth =
            (maxAcq == 1 && fb.producerDepth == 1) ? 1 : maxAcq + 1;
      }
    }
    // Consumer DMA depths
    for (size_t ci = 0; ci < fb.consumers.size(); ci++) {
      int dmaDepth =
          fb.consumerDepths.empty() ? fb.producerDepth : fb.consumerDepths[ci];
      if (fb.consumers[ci]) {
        auto it =
            endpointMaxAcquire.find(std::pair{fb.fifoOp, fb.consumers[ci]});
        if (it != endpointMaxAcquire.end()) {
          int maxAcq = it->second;
          int declared = fb.consumerDepthAt(ci);
          dmaDepth = (maxAcq == 1 && declared == 1) ? 1 : maxAcq + 1;
        }
      }
      fb.consumerDMADepths.push_back(dmaDepth);
    }
  }
}

LogicalResult SAPlacer::generateInitialPlacement() {
  auto &logicalTiles = collected.logicalTiles;

  // Classify available tiles by type
  std::vector<TileID> availComp = availability.compTiles;
  std::vector<TileID> availMem, availShim;
  for (auto &t : availability.nonCompTiles) {
    AIETileType type = targetModel->getTileType(t.col, t.row);
    if (type == AIETileType::MemTile)
      availMem.push_back(t);
    else if (type == AIETileType::ShimNOCTile)
      availShim.push_back(t);
  }

  // Order compute positions in checkerboard pattern: phase-0 positions
  // ((col+row) even) first, then phase-1. No two phase-0 positions are
  // 4-connected neighbors, so the heaviest tiles placed first are guaranteed
  // non-adjacent. Within each phase, row-major order.
  llvm::sort(availComp, [](TileID a, TileID b) {
    int phaseA = (a.col + a.row) % 2;
    int phaseB = (b.col + b.row) % 2;
    if (phaseA != phaseB)
      return phaseA < phaseB;
    if (a.row != b.row)
      return a.row < b.row;
    return a.col < b.col;
  });
  // Order MemTile positions: even columns first (0,2,4,6,1,3,5,7).
  // Heavy MemTile tiles placed first get non-adjacent columns, avoiding
  // per-buffer spill collisions.
  llvm::sort(availMem, [](TileID a, TileID b) {
    int phaseA = a.col % 2;
    int phaseB = b.col % 2;
    if (phaseA != phaseB)
      return phaseA < phaseB;
    return a.col < b.col;
  });
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
        llvm::erase(availComp, pos);
      else if (type == AIETileType::MemTile)
        llvm::erase(availMem, pos);
      else if (type == AIETileType::ShimNOCTile)
        llvm::erase(availShim, pos);
    }
  }

  // Place unconstrained core tiles sorted by memory weight (heaviest first)
  // into checkerboard-spread positions. Cascade tiles are placed independently;
  // the cascade penalty guides them to adjacent positions during SA.
  auto byMemWeightDesc = [this](Operation *a, Operation *b) {
    int64_t wa = computeTileMemoryWeight(a);
    int64_t wb = computeTileMemoryWeight(b);
    if (wa != wb)
      return wa > wb;
    return a->isBeforeInBlock(b);
  };
  {
    SmallVector<Operation *> sortedCoreTiles;
    for (auto lt : logicalTiles) {
      Operation *op = lt.getOperation();
      if (constrainedTiles.contains(op))
        continue;
      if (lt.getTileType() == AIETileType::CoreTile)
        sortedCoreTiles.push_back(op);
    }
    llvm::sort(sortedCoreTiles, byMemWeightDesc);

    for (auto *op : sortedCoreTiles) {
      TileID pos;
      bool found = false;
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
        return cast<LogicalTileOp>(op).emitError(
            "no available physical tile for placement");
      currentPlacement[op] = pos;
      physToLogical[pos] = op;
      movableTiles.push_back(op);
    }
  }

  // Place unconstrained MemTile tiles sorted by memory weight (heaviest
  // first into spread positions), then shim tiles.
  {
    SmallVector<Operation *> sortedMemTiles, shimTiles;
    for (auto lt : logicalTiles) {
      Operation *op = lt.getOperation();
      if (constrainedTiles.contains(op))
        continue;
      AIETileType type = lt.getTileType();
      if (type == AIETileType::MemTile)
        sortedMemTiles.push_back(op);
      else if (type == AIETileType::ShimNOCTile)
        shimTiles.push_back(op);
    }
    llvm::sort(sortedMemTiles, byMemWeightDesc);

    for (auto *op : sortedMemTiles) {
      if (availMem.empty())
        return cast<LogicalTileOp>(op).emitError(
            "no available physical tile for placement");
      TileID pos = availMem[memIdx % availMem.size()];
      memIdx++;
      currentPlacement[op] = pos;
      movableTiles.push_back(op);
    }

    for (auto *op : shimTiles) {
      if (availShim.empty())
        return cast<LogicalTileOp>(op).emitError(
            "no available physical tile for placement");
      TileID pos = availShim[shimIdx % availShim.size()];
      shimIdx++;
      if (!isLegalPosition(op, pos))
        return cast<LogicalTileOp>(op).emitError(
            "no available physical tile for placement");
      currentPlacement[op] = pos;
      movableTiles.push_back(op);
    }
  }

  return success();
}

void SAPlacer::initializeSAState() {
  initBoundingBoxes();
  initResourceTracking();

  cascadePen = computeAdjacencyPenalty(cascadeAdjacency, kCascadeOffsets,
                                       config.cascadeWeightPerDist);
  int memPressure = computeMemoryPressure();
  totalCost =
      computeTotalHPWL() + getResourcePenalty() + cascadePen + memPressure;

  deviceSlots = availability.compTiles.size() + movableTiles.size();

  LLVM_DEBUG(llvm::dbgs() << "[SA] Initial cost: " << totalCost
                          << " (HPWL=" << computeTotalHPWL()
                          << ", resPenalty=" << getResourcePenalty()
                          << "), movable tiles: " << movableTiles.size()
                          << ", nets: " << nets.size()
                          << ", fifos: " << fifoBuffers.size() << "\n");
}

void SAPlacer::runSAMainLoop() {
  // Skip SA when there is nothing to optimize.
  bool hasCostTerms =
      !nets.empty() || !fifoBuffers.empty() || !cascadeAdjacency.edges.empty();
  if (movableTiles.empty() || !hasCostTerms)
    return;

  // SA schedule parameters scaled by design and device.
  int numMovable = movableTiles.size();
  int numNets = nets.size();

  int movesPerIter =
      std::clamp(static_cast<int>(numMovable * std::sqrt(std::max(numNets, 1))),
                 config.minMovesPerIter, config.maxMovesPerIter);

  int maxIters = std::max(
      config.minMaxIters,
      static_cast<int>(1000 * std::sqrt(static_cast<double>(numMovable) *
                                        std::max(numNets, 1))));

  int greedyIters = config.greedyMultiplier * numMovable;

  int numSamples = std::max(10 * numMovable, 50);
  double estimatedT = estimateInitialTemperature(numSamples);
  double initTemp =
      std::min(config.tempScaleEstimate * estimatedT,
               static_cast<double>(totalCost) * config.tempScaleCost);
  initTemp = std::max(initTemp, 1.0);

  LLVM_DEBUG(llvm::dbgs() << "[SA] Schedule: movesPerIter=" << movesPerIter
                          << " maxIters=" << maxIters
                          << " initTemp=" << initTemp << "\n");

  schedule = SASchedule(initTemp, movesPerIter, maxIters, greedyIters, config);

  bestPlacement = currentPlacement;
  bestCost =
      (getResourcePenalty() == 0 && cascadePen == 0) ? totalCost : INT_MAX;
  bestOverallPlacement = currentPlacement;
  bestOverallCost = totalCost;
  zeroDeltaMoves = posDeltaMoves = negDeltaMoves = 0;
  acceptedUphill = rejectedMoves = 0;

  while (!schedule.limitReached()) {
    for (int m = 0; m < schedule.getMovesPerIter(); m++) {

      SmallVector<std::pair<Operation *, TileID>> moves;
      if (!generateMove(moves))
        continue;
      tryMove(moves);
    }

    if (schedule.getIteration() % config.debugInterval == 0)
      LLVM_DEBUG(llvm::dbgs()
                 << "[SA] Iter " << schedule.getIteration()
                 << " T=" << llvm::format("%.3e", schedule.getTemperature())
                 << " cost=" << totalCost << " best=" << bestCost << " accept="
                 << llvm::format("%.3f", schedule.getAcceptanceRatio())
                 << " d0=" << zeroDeltaMoves << " d+=" << posDeltaMoves
                 << " d-=" << negDeltaMoves << " uphill=" << acceptedUphill
                 << " rej=" << rejectedMoves
                 << (schedule.isGreedy() ? " (greedy)" : "") << "\n");

    // Periodic full recompute to reset incremental cost drift.
    if (schedule.getIteration() % config.fullRecomputeInterval == 0 &&
        schedule.getIteration() > 0) {
      initResourceTracking();
      cascadePen = computeAdjacencyPenalty(cascadeAdjacency, kCascadeOffsets,
                                           config.cascadeWeightPerDist);
      totalCost = computeTotalHPWL() + getResourcePenalty() + cascadePen +
                  computeMemoryPressure();
    }

    schedule.cool();
  }
}

//===----------------------------------------------------------------------===//
// Move acceptance
//===----------------------------------------------------------------------===//

void SAPlacer::tryMove(SmallVector<std::pair<Operation *, TileID>> &moves) {
  // Snapshot state for revert on rejection.
  SmallVector<std::pair<Operation *, TileID>> oldPlacements;
  for (auto &[t, _] : moves)
    oldPlacements.push_back({t, currentPlacement[t]});
  int oldResPenalty = cachedResourcePenalty;
  auto savedMemUsage = currentMemUsage;
  auto savedDMAUsage = currentDMAUsage;
  auto savedSharedMemDest = sharedMemDestination;

  // Apply moves: update placement maps and compute HPWL delta.
  for (auto &[t, _] : moves)
    physToLogical.erase(currentPlacement[t]);
  SmallVector<SmallVector<std::pair<size_t, NetBoundingBox>>> allBackups(
      moves.size());
  int hpwlDelta = 0;
  for (size_t i = 0; i < moves.size(); i++) {
    hpwlDelta += evaluateMove(moves[i].first, moves[i].second, allBackups[i]);
    currentPlacement[moves[i].first] = moves[i].second;
  }
  for (auto &[t, pos] : moves)
    physToLogical[pos] = t;

  // Compute total cost delta across all cost terms.
  int oldCascade = cascadePen;
  int oldPressure = computeMemoryPressure();
  int newResPenalty = updateResourcePenalty(oldPlacements);
  int newCascade = computeAdjacencyPenalty(cascadeAdjacency, kCascadeOffsets,
                                           config.cascadeWeightPerDist);
  int newPressure = computeMemoryPressure();
  int delta = hpwlDelta + (newResPenalty - oldResPenalty) +
              (newCascade - oldCascade) + (newPressure - oldPressure);

  if (delta == 0)
    zeroDeltaMoves++;
  else if (delta > 0)
    posDeltaMoves++;
  else
    negDeltaMoves++;

  // Accept improving moves unconditionally; accept worsening moves
  // with probability exp(-delta/T).
  bool accept =
      delta <= 0 ||
      (schedule.getTemperature() > 0 &&
       acceptDist(rng) < std::exp(-delta / schedule.getTemperature()));

  if (accept) {
    if (delta > 0)
      acceptedUphill++;
    totalCost += delta;
    cascadePen = newCascade;
    schedule.record(true);
    if (totalCost < bestOverallCost) {
      bestOverallCost = totalCost;
      bestOverallPlacement = currentPlacement;
    }
    // Only update bestPlacement when fully legal (no resource or cascade
    // violations).
    if (totalCost < bestCost && newResPenalty == 0 && newCascade == 0) {
      bestCost = totalCost;
      bestPlacement = currentPlacement;
    }
  } else {
    // Revert: undo moves in reverse order and restore saved state.
    for (auto &[t, pos] : moves)
      physToLogical.erase(pos);
    for (int i = moves.size() - 1; i >= 0; i--) {
      revertMove(allBackups[i]);
      currentPlacement[moves[i].first] = oldPlacements[i].second;
    }
    for (auto &[op, oldPos] : oldPlacements)
      physToLogical[oldPos] = op;
    currentMemUsage = savedMemUsage;
    currentDMAUsage = savedDMAUsage;
    sharedMemDestination = savedSharedMemDest;
    cachedResourcePenalty = oldResPenalty;
    rejectedMoves++;
    schedule.record(false);
  }
}

//===----------------------------------------------------------------------===//
// Debug output
//===----------------------------------------------------------------------===//

void SAPlacer::printPlacementStats(int64_t elapsedMs) const {
  int finalHPWL = computeTotalHPWL();
  int finalHardPenalty = getResourcePenalty();
  int finalPressure = computeMemoryPressure();
  int finalCascade = computeAdjacencyPenalty(cascadeAdjacency, kCascadeOffsets,
                                             config.cascadeWeightPerDist);
  bool isLegal = (finalHardPenalty == 0 && finalCascade == 0);

  llvm::dbgs() << "[SA] seed=" << rngSeed << " Final HPWL=" << finalHPWL
               << ", resPenalty=" << finalHardPenalty
               << ", pressure=" << finalPressure << ", cascade=" << finalCascade
               << ", legal=" << (isLegal ? "yes" : "NO") << "\n";
  llvm::dbgs() << "[SA] " << schedule.getIteration() << " iters, "
               << movableTiles.size() << " tiles, " << nets.size() << " nets, "
               << allocates.size() << " allocates, " << elapsedMs << " ms\n";
  llvm::dbgs() << "[SA] Moves: d-=" << negDeltaMoves << " d0=" << zeroDeltaMoves
               << " d+=" << posDeltaMoves << " uphill=" << acceptedUphill
               << " rej=" << rejectedMoves << "\n";

  // Per-tile resource utilization (tiles above 75% capacity).
  int64_t coreCap = targetModel->getLocalMemorySize();
  int64_t memCap = targetModel->getMemTileSize();
  for (auto &[pos, used] : currentMemUsage) {
    auto type = targetModel->getTileType(pos.col, pos.row);
    int64_t cap = (type == AIETileType::MemTile) ? memCap : coreCap;
    if (used > cap * 3 / 4)
      llvm::dbgs() << "[SA-RES] mem(" << pos.col << "," << pos.row
                   << "): " << used << "/" << cap
                   << (used > cap ? " OVERFLOW" : "") << "\n";
  }
  for (auto &[pos, usage] : currentDMAUsage) {
    auto [maxIn, maxOut] = getDMACapacity(*targetModel, pos);
    if (usage.first >= maxIn || usage.second >= maxOut)
      llvm::dbgs() << "[SA-RES] dma(" << pos.col << "," << pos.row
                   << "): in=" << usage.first << "/" << maxIn
                   << " out=" << usage.second << "/" << maxOut
                   << (usage.first > maxIn || usage.second > maxOut
                           ? " OVERFLOW"
                           : " AT-LIMIT")
                   << "\n";
  }

  // Detailed breakdown for overflowing core tiles: neighbor spare capacity.
  for (auto &[pos, used] : currentMemUsage) {
    auto type = targetModel->getTileType(pos.col, pos.row);
    if (type != AIETileType::CoreTile || used <= coreCap)
      continue;
    int64_t overflow = used - coreCap;
    for (TileID n : getMemAffinityNeighbors(*targetModel, pos)) {
      auto it = currentMemUsage.find(n);
      int64_t nbrUsed = it != currentMemUsage.end() ? it->second : 0;
      llvm::dbgs() << "  nbr(" << n.col << "," << n.row << "): used=" << nbrUsed
                   << " spare=" << std::max(coreCap - nbrUsed, int64_t{0})
                   << "\n";
    }
    llvm::dbgs() << "[SA-PENALTY] tile(" << pos.col << "," << pos.row
                 << "): used=" << used << " cap=" << coreCap
                 << " overflow=" << overflow << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Post-SA finalization
//===----------------------------------------------------------------------===//

LogicalResult SAPlacer::finalizePlacement(DeviceOp device) {
  // Restore best placement found during SA. Skip if SA loop didn't run
  // (bestOverallCost stays at INT_MAX when runSAMainLoop returns early).
  if (bestOverallCost < INT_MAX) {
    if (bestCost < INT_MAX)
      currentPlacement = bestPlacement;
    else
      currentPlacement = bestOverallPlacement;
    physToLogical.clear();
    for (auto &[op, pos] : currentPlacement)
      physToLogical[pos] = op;
    initResourceTracking();
  }

  // Generate allocates (state is fresh from initResourceTracking)
  generateAllocates();

  // Merge mem/shim tiles that SA placed at the same position
  if (failed(mergeMemShimTiles(device)))
    return failure();

  auto endTime = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);

  LLVM_DEBUG(printPlacementStats(elapsed.count()));

  result = currentPlacement;
  return success();
}
