//===- AIESAPlacer.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEBufferAllocation.h"
#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

using namespace mlir;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-sa-placer"

// The std:: distributions and std::shuffle are implementation-defined, so the
// same seed yields different placements under libstdc++ and MSVC. These
// helpers consume raw std::mt19937 output, which the standard fixes.

/// Draw an index in [0, upperExclusive).
static size_t drawBoundedIndex(std::mt19937 &rng, size_t upperExclusive) {
  return static_cast<size_t>(rng()) % upperExclusive;
}

/// Draw a double in [0, 1).
static double drawUnitReal(std::mt19937 &rng) {
  constexpr double denominator = static_cast<double>(std::mt19937::max()) + 1.0;
  return static_cast<double>(rng()) / denominator;
}

/// Fisher-Yates shuffle.
static void deterministicShuffle(MutableArrayRef<TileID> values,
                                 std::mt19937 &rng) {
  for (size_t i = values.size(); i > 1; --i)
    std::swap(values[i - 1], values[drawBoundedIndex(rng, i)]);
}

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

/// Bytes a buffer takes on a tile once assign-buffer-addresses aligns it.
static int64_t alignedBufferBytes(const AIETargetModel &targetModel,
                                  AIETileType tileType, int64_t bytes) {
  TileMemoryLimits limits = tileMemoryLimits(targetModel, tileType);
  return llvm::alignTo(bytes,
                       requiredBufferAlignBits(limits.tileAlignBitWidth,
                                               limits.maxVecAlignBits, bytes) /
                           8);
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

// A route is a net over the tiles of its ends, and each DMA end is one channel
// on its tile; a pool is bytes on its tile, and descriptors when the tile is a
// mem tile. None of it needs the DMA-or-shared-memory guess a fifo needs: a
// route already moves data by DMA.
void SAPlacer::buildRouteModel(
    DeviceOp device, ArrayRef<FlowOp> flows, ArrayRef<PacketFlowOp> pktFlows,
    ArrayRef<RouteOp> routes,
    ArrayRef<DMAChannelAnalysis::ChannelClaim> claims) {
  routeEnds.clear();
  for (auto &[tile, inOut] :
       flowChannelDemand(flows, pktFlows, routes, claims)) {
    for (int i = 0; i < inOut.first; ++i)
      routeEnds.push_back({tile, /*output=*/false});
    for (int i = 0; i < inOut.second; ++i)
      routeEnds.push_back({tile, /*output=*/true});
  }
  for (RouteOp route : routes) {
    NetInfo net;
    auto addEnd = [&](StringRef name, bool output) {
      auto endpoint = SymbolTable::lookupNearestSymbolFrom<RouteEndpoint>(
          route, StringAttr::get(route.getContext(), name));
      if (!endpoint)
        return;
      Operation *tileOp = endpoint.getTile().getDefiningOp();
      if (!isa_and_nonnull<LogicalTileOp>(tileOp))
        return;
      net.endpoints.push_back(tileOp);
    };
    for (StringRef source : route.getSourceNames())
      addEnd(source, /*output=*/true);
    for (StringRef dest : route.getDestinationNames())
      addEnd(dest, /*output=*/false);
    if (net.endpoints.size() < 2)
      continue;
    unsigned threshold = static_cast<unsigned>(config.multicastThreshold);
    net.isMulticast = route.getDestinations().size() > threshold ||
                      route.getSources().size() > threshold;
    size_t idx = nets.size();
    nets.push_back(net);
    for (auto *ep : net.endpoints)
      tileToNetIndices[ep].push_back(idx);
  }

  for (auto pool : device.getOps<ObjectFifoPoolOp>()) {
    Operation *tileOp = pool.getTile().getDefiningOp();
    if (!isa_and_nonnull<LogicalTileOp>(tileOp))
      continue;
    staticBufferSizes[tileOp] += pool.getObjectSizeInBytes() * pool.getDepth();
  }
  // Ends already split take the BD ids they count; other DMA ends a channel.
  for (auto endpoint : device.getOps<RouteEndpoint>()) {
    Operation *tileOp = endpoint.getTile().getDefiningOp();
    if (endpoint.getRouteBundle() != WireBundle::DMA ||
        !isa_and_nonnull<LogicalTileOp>(tileOp) ||
        tileTypes.lookup(tileOp) != AIETileType::MemTile)
      continue;
    PlannedDmaEnd end{0, nullptr, -1, 0, false, 0, 0};
    if (auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation())) {
      ObjectFifoPoolOp pool = dma.getPoolOp();
      end = {dma.getNumBDs(),
             pool,
             -1,
             pool.getDepth(),
             pool.canWalkWhole(),
             pool.getObjectSizeInBytes(),
             pool.getContiguous() ? pool.getObjectsPerBuffer() : 0};
    }
    memTileEnds[tileOp].push_back({endpoint.getRouteDirection(), end});
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

std::optional<TileID> SAPlacer::positionOf(Operation *tile) const {
  if (auto it = currentPlacement.find(tile);
      tile && it != currentPlacement.end())
    return it->second;
  if (auto tileOp = dyn_cast_or_null<TileOp>(tile))
    return tileOp.getTileID();
  return std::nullopt;
}

// Add a single fifo's contributions to memory and DMA usage maps.
void SAPlacer::addRouteEndContribution(size_t endIdx, int sign) {
  const RouteEndInfo &end = routeEnds[endIdx];
  std::optional<TileID> pos = positionOf(end.tile);
  if (!pos)
    return;
  if (end.output)
    currentDMAUsage[*pos].second += sign;
  else
    currentDMAUsage[*pos].first += sign;
}

void SAPlacer::addFifoContribution(size_t fifoIdx, int sign) {
  const auto &fb = fifoBuffers[fifoIdx];
  std::optional<TileID> prodAt = positionOf(fb.producerTile);
  if (!prodAt)
    return;
  TileID prodPos = *prodAt;
  const FifoEndTypes &types = fifoEndTypes[fifoIdx];
  bool prodIsShim = types.producer == AIETileType::ShimNOCTile ||
                    types.producer == AIETileType::ShimPLTile;

  // A fifo that already has an aie.objectfifo.allocate keeps its shared-memory
  // objects on that delegate tile.
  std::optional<TileID> delegatePos = positionOf(fb.delegate);

  // A fifo shared memory carries keeps one pool of sharedDepth() objects;
  // otherwise each end off a shim keeps its own, except a link output's
  // producer, whose objects are the link input's consumer pool.
  bool prodCharged = false;
  bool prodNeedsDMA = false;

  for (size_t ci = 0; ci < fb.consumerTiles.size(); ci++) {
    std::optional<TileID> consAt = positionOf(fb.consumerTiles[ci]);
    if (!consAt)
      continue;
    TileID consPos = *consAt;
    bool consIsShim = types.consumers[ci] == AIETileType::ShimNOCTile ||
                      types.consumers[ci] == AIETileType::ShimPLTile;

    using SharedMemory = AIETargetModel::SharedMemory;
    SharedMemory shared =
        fb.forcesDMA
            ? SharedMemory::None
            : ObjectFifoCreateOp::sharedMemoryOf(*targetModel, prodPos, consPos,
                                                 delegatePos, fb.linked);
    bool isSharedMem = shared != SharedMemory::None;
    int64_t sharedBytes =
        std::max(fb.producerSizeBytes, fb.consumerSizeBytes) * fb.sharedDepth;

    if (prodPos == consPos && isSharedMem) {
      currentMemUsage[delegatePos.value_or(prodPos)] += sign * sharedBytes;
    } else if (isSharedMem) {
      TileID bufTile;
      if (delegatePos) {
        bufTile = *delegatePos;
      } else if (shared == SharedMemory::Either) {
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
      } else if (shared == SharedMemory::Second) {
        bufTile = consPos;
      } else {
        bufTile = prodPos;
      }
      currentMemUsage[bufTile] += sign * sharedBytes;
    } else {
      if (!prodCharged && !prodIsShim && !fb.linkSharedProd) {
        currentMemUsage[prodPos] +=
            sign * fb.producerSizeBytes * fb.producerDepth;
        prodCharged = true;
      }
      if (!consIsShim)
        currentMemUsage[consPos] +=
            sign * fb.consumerSizeBytes * fb.consumerDepths[ci];
    }

    // DMA contribution: a fifo shared memory does not carry runs through the
    // DMAs, back to its own tile as well. An end takes a channel of its own
    // unless it is a stream port, a PLIO, or a merge's or dispatch's
    // participant whose shared end another took.
    if (!isSharedMem) {
      bool plio = cast<ObjectFifoCreateOp>(fb.fifoOp).getPlio();
      if (consIsShim ? !plio : fb.consumerEnds[ci].has_value())
        currentDMAUsage[consPos].first += sign;
      prodNeedsDMA = prodIsShim ? !plio : fb.producerEnd.has_value();
    }
  }

  if (prodNeedsDMA)
    currentDMAUsage[prodPos].second += sign;
}

void SAPlacer::initResourceTracking() {
  currentMemUsage.clear();
  currentDMAUsage.clear();
  sharedMemDestination.clear();

  auto typeOf = [&](Operation *tile) -> std::optional<AIETileType> {
    if (auto it = tileTypes.find(tile); it != tileTypes.end())
      return it->second;
    if (auto placed = dyn_cast_or_null<TileOp>(tile))
      return placed.getTileType();
    return std::nullopt;
  };
  fifoEndTypes.clear();
  for (const auto &fb : fifoBuffers) {
    FifoEndTypes &types = fifoEndTypes.emplace_back();
    types.producer = typeOf(fb.producerTile);
    for (Operation *cons : fb.consumerTiles)
      types.consumers.push_back(typeOf(cons));
  }

  // Build reverse index: tile → fifo indices
  tileToFifoIndices.clear();
  for (size_t i = 0; i < fifoBuffers.size(); i++) {
    const auto &fb = fifoBuffers[i];
    if (fb.producer)
      tileToFifoIndices[fb.producer].push_back(i);
    for (auto *cons : fb.consumers)
      if (cons)
        tileToFifoIndices[cons].push_back(i);
    if (fb.delegate)
      tileToFifoIndices[fb.delegate].push_back(i);
  }

  // Static buffers and stacks go first: a shared-memory fifo picks the emptier
  // of its two tiles as it is added, and must see what they already hold.
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

  for (size_t i = 0; i < fifoBuffers.size(); i++)
    addFifoContribution(i, +1);

  tileToRouteEnds.clear();
  for (size_t i = 0; i < routeEnds.size(); i++) {
    tileToRouteEnds[routeEnds[i].tile].push_back(i);
    addRouteEndContribution(i, +1);
  }

  spillTiles.clear();
  for (auto &[op, _] : memTileEnds)
    spillTiles.insert(op);
  for (auto &[op, _] : memTileFixed)
    spillTiles.insert(op);
  spillTiles.insert(logicalMemTiles.begin(), logicalMemTiles.end());
  delegateTermTiles.clear();
  for (const auto &fb : fifoBuffers) {
    if (!fb.delegate)
      continue;
    delegateTermTiles.insert(fb.delegate);
    delegateTermTiles.insert(fb.producer);
    delegateTermTiles.insert(fb.consumers.begin(), fb.consumers.end());
  }
  pairedTiles.clear();
  for (const Adjacency *adj : {&cascadeAdjacency, &sharedMemAdjacency})
    for (auto [first, second] : adj->edges) {
      pairedTiles.insert(first.getOperation());
      pairedTiles.insert(second.getOperation());
    }

  memSpilloverPenalty = computeMemSpilloverPenalty();
  bdCountPenalty = computeBDCountPenalty();
  delegatePenalty = computeDelegatePenalty();
  dmaPenalty = computeDMAChannelPenalty();
  cachedResourcePenalty = memSpilloverPenalty + computeCoreOverflowPenalty() +
                          dmaPenalty + bdCountPenalty + delegatePenalty;
  cachedPressure = computeMemoryPressure();
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

// Both ends of a fifo with an aie.objectfifo.allocate must reach the delegate
// tile's memory module, or the objectfifo split pass rejects the allocate.
int SAPlacer::computeDelegatePenalty() const {
  int penalty = 0;
  for (const auto &fb : fifoBuffers) {
    std::optional<TileID> delegatePos = positionOf(fb.delegate);
    if (!delegatePos)
      continue;
    SmallVector<Operation *, 2> ends = {fb.producer};
    llvm::append_range(ends, fb.consumers);
    for (Operation *end : ends) {
      std::optional<TileID> endPos = positionOf(end);
      if (!endPos)
        continue;
      auto shared = targetModel->getSharedMemory(*delegatePos, *endPos);
      if (shared == AIETargetModel::SharedMemory::First ||
          shared == AIETargetModel::SharedMemory::Either)
        continue;
      int dist = std::abs(delegatePos->col - endPos->col) +
                 std::abs(delegatePos->row - endPos->row);
      penalty += config.delegateWeightPerDist * std::max(dist - 1, 1);
    }
  }
  return penalty;
}

// Charges by the KB what MemTileSpillSearch, in the allocator's attempt order,
// cannot place.
int SAPlacer::computeMemSpilloverPenalty() {
  SmallVector<MemTileSpillSearch::Tile> tiles;
  DenseMap<TileID, unsigned> index;
  for (int col = 0; col < targetModel->columns(); ++col)
    for (int row = 0; row < targetModel->rows(); ++row)
      if (targetModel->getTileType(col, row) == AIETileType::MemTile) {
        index[{col, row}] = tiles.size();
        tiles.push_back({targetModel->getMemTileSize(), {}});
      }
  for (auto &[op, fixed] : memTileFixed)
    if (auto at = currentPlacement.find(op); at != currentPlacement.end())
      llvm::append_range(tiles[index.lookup(at->second)].fixed, fixed);

  // Per pool, in the allocator's size-first order: as split, and as walked.
  SmallVector<std::pair<SpillPool, TileID>> split, walkedPools;
  bool anyWalked = false;
  DenseSet<TileID> positions;
  for (auto &[tile, _] : memTileEnds) {
    auto at = currentPlacement.find(tile);
    if (at == currentPlacement.end() || !positions.insert(at->second).second)
      continue;
    SmallVector<Operation *> here = memTilesAt(at->second);
    DenseSet<std::pair<Operation *, int>> walked;
    (void)bdShortfall(at->second, here, &walked);
    anyWalked |= !walked.empty();
    llvm::MapVector<std::pair<Operation *, int>,
                    std::pair<const PlannedDmaEnd *, std::pair<int, int>>>
        pools;
    for (Operation *op : here)
      for (const MemTileEnd &end : memTileEnds.find(op)->second) {
        const PlannedDmaEnd &plan = end.end;
        auto split = dyn_cast_or_null<ObjectFifoPoolOp>(plan.pool);
        if (!plan.pool || (split && split.getBuffers()))
          continue;
        auto &[first, demand] = pools[{plan.pool, plan.id}];
        first = first ? first : &plan;
        ++(end.dir == DMAChannelDir::S2MM ? demand.first : demand.second);
      }
    unsigned home = index.lookup(at->second);
    for (auto &[key, entry] : pools) {
      auto [plan, demand] = entry;
      int busier = std::max(demand.first, demand.second);
      int64_t whole = plan->contiguousObjects;
      split.push_back({{plan->objectBytes * std::max<int64_t>(whole, 1),
                        whole ? 1 : plan->depth, home, busier},
                       at->second});
      if (!whole && walked.contains(key))
        whole = plan->depth;
      walkedPools.push_back({{plan->objectBytes * std::max<int64_t>(whole, 1),
                              whole ? 1 : plan->depth, home, busier},
                             at->second});
    }
  }
  for (auto *pools : {&split, &walkedPools}) {
    llvm::stable_sort(*pools, [](const auto &a, const auto &b) {
      return std::make_pair(a.second.col, a.second.row) <
             std::make_pair(b.second.col, b.second.row);
    });
    sortLargestBufferFirst(*pools,
                           [](const auto &pool) { return pool.first.bytes; });
  }

  auto attempt = [&](ArrayRef<std::pair<SpillPool, TileID>> pools,
                     ArrayRef<unsigned> order) {
    SmallVector<MemTileSpillSearch::Buffer> buffers;
    for (unsigned i : order) {
      auto [pool, at] = pools[i];
      SmallVector<unsigned, 2> neighbours;
      for (TileID neighbour : memTileSpillNeighbours(*targetModel, at))
        if (auto it = index.find(neighbour); it != index.end())
          neighbours.push_back(it->second);
      buffers.append(pool.count, {pool.bytes, pool.home, true, neighbours});
    }
    MemTileSpillSearch search(tiles,
                              targetModel->getMemTileLoadStoreBusWidth() / 8);
    return search.place(buffers) ? int64_t{0}
                                 : std::max<int64_t>(search.unplacedBytes, 1);
  };
  auto inOrder = [](size_t size) {
    return llvm::to_vector(llvm::seq<unsigned>(0, size));
  };
  int64_t unplaced = attempt(walkedPools, inOrder(walkedPools.size()));
  if (unplaced && !anyWalked) {
    SmallVector<SpillPool> records;
    for (auto &[pool, _] : split)
      records.push_back(pool);
    if (!attempt(split, demandFirstOrder(records)))
      unplaced = 0;
  }
  return ((unplaced + 1023) / 1024) * config.memPenaltyPerKB;
}

// In position order: room one tile is lent is gone for the next, so the
// order decides the outcome and must not follow the map's hashing.
SmallVector<TileID> SAPlacer::overflowingCoreTiles() const {
  SmallVector<TileID> tiles;
  for (auto &[tilePos, used] : currentMemUsage)
    if (targetModel->getTileType(tilePos.col, tilePos.row) ==
            AIETileType::CoreTile &&
        used > targetModel->getLocalMemorySize())
      tiles.push_back(tilePos);
  llvm::sort(tiles);
  return tiles;
}

// Penalizes core tiles whose buffer usage exceeds local memory, accounting
// for intratile fifos that can be relocated to a neighbor via allocate.
int SAPlacer::computeCoreOverflowPenalty() const {
  int64_t coreCapacity = targetModel->getLocalMemorySize();
  int penalty = 0;
  // Room a neighbour gives one overflowing tile is gone for the next, as
  // generateAllocates hands it out.
  llvm::DenseMap<TileID, int64_t> lent;

  for (TileID tilePos : overflowingCoreTiles()) {
    int64_t overflow = currentMemUsage.lookup(tilePos) - coreCapacity;

    // Build accessible neighbor spare capacities.
    llvm::SmallVector<std::pair<int64_t, TileID>, 4> nbrSpares;
    for (TileID nbr : getMemAffinityNeighbors(*targetModel, tilePos)) {
      auto it = currentMemUsage.find(nbr);
      int64_t nbrUsed = (it != currentMemUsage.end()) ? it->second : 0;
      int64_t spare =
          std::max(coreCapacity - nbrUsed - lent.lookup(nbr), int64_t{0});
      if (spare > 0)
        nbrSpares.push_back({spare, nbr});
    }
    llvm::sort(nbrSpares,
               [](const auto &a, const auto &b) { return a.first > b.first; });

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
          if (fb.consumers.size() != 1 || !fb.producer || !fb.consumers[0] ||
              fb.delegate)
            continue;
          auto prodIt = currentPlacement.find(fb.producer);
          auto consIt = currentPlacement.find(fb.consumers[0]);
          if (prodIt == currentPlacement.end() ||
              consIt == currentPlacement.end())
            continue;
          if (prodIt->second != tilePos || consIt->second != tilePos)
            continue;
          int64_t sz = std::max(fb.producerSizeBytes, fb.consumerSizeBytes) *
                       fb.sharedDepth;
          fifoSizes.push_back(sz);
        }
        llvm::sort(fifoSizes, std::greater<int64_t>());

        // Greedy packing: assign each fifo to the first neighbor that fits.
        int64_t totalAbsorbed = 0;
        for (int64_t sz : fifoSizes) {
          for (auto &[spare, nbr] : nbrSpares) {
            if (spare >= sz) {
              spare -= sz;
              lent[nbr] += sz;
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

int SAPlacer::computeTileDMAPenalty(TileID tilePos) const {
  auto usage = currentDMAUsage.lookup(tilePos);
  auto [maxIn, maxOut] = getDMACapacity(*targetModel, tilePos);
  int used[2] = {usage.first, usage.second};
  int max[2] = {maxIn, maxOut};
  int penalty = 0;
  for (int dir : {0, 1}) {
    if (used[dir] <= max[dir])
      continue;
    if (shimSpans && targetModel->getTileType(tilePos.col, tilePos.row) ==
                         AIETileType::ShimNOCTile) {
      SmallVector<StringAttr> ends;
      for (auto &[lt, own] : shareableShimEnds) {
        if (positionOf(lt) == tilePos)
          llvm::append_range(ends, own[dir]);
      }
      used[dir] += static_cast<int>(shimSpans->groups(ends).size()) -
                   static_cast<int>(ends.size());
    }
    penalty += std::max(0, used[dir] - max[dir]) * config.dmaPenaltyPerChannel;
  }
  if (targetModel->getTileType(tilePos.col, tilePos.row) ==
      AIETileType::ShimNOCTile) {
    SmallVector<Operation *> onTile;
    for (auto &[lt, peak] : runtimeBdIds.users())
      if (auto at = currentPlacement.find(lt);
          at != currentPlacement.end() && at->second == tilePos)
        onTile.push_back(lt);
    penalty +=
        runtimeBdIds.shortfall(onTile, tilePos) * config.dmaPenaltyPerChannel;
  }
  return penalty;
}

int SAPlacer::computeDMAChannelPenalty() {
  dmaPenaltyAt.clear();
  int penalty = 0;
  for (auto &[tilePos, _] : currentDMAUsage) {
    int at = computeTileDMAPenalty(tilePos);
    dmaPenaltyAt[tilePos] = at;
    penalty += at;
  }
  return penalty;
}

SmallVector<Operation *> SAPlacer::memTilesAt(TileID position) const {
  SmallVector<Operation *> tiles;
  for (auto &[tile, ends] : memTileEnds)
    if (auto at = currentPlacement.find(tile);
        at != currentPlacement.end() && at->second == position)
      tiles.push_back(tile);
  return tiles;
}

int64_t SAPlacer::bdShortfall(TileID position, ArrayRef<Operation *> tiles,
                              DenseSet<std::pair<Operation *, int>> *walked) {
  auto [layout, added] = bdLayoutOf.try_emplace(position, position);
  if (added) {
    auto access = [&](TileID at) {
      std::vector<bool> bits;
      int channels = targetModel->getNumSourceSwitchboxConnections(
          at.col, at.row, WireBundle::DMA);
      for (int channel = 0; channel < channels; ++channel)
        for (uint32_t bd = 0, e = targetModel->getNumBDs(at.col, at.row);
             bd < e; ++bd)
          bits.push_back(
              targetModel->isBdChannelAccessible(at.col, at.row, bd, channel));
      return bits;
    };
    for (auto &[other, rep] : bdLayoutOf)
      if (other == rep && !(other == position) &&
          access(other) == access(position)) {
        layout->second = other;
        break;
      }
  }
  // Bounded, since tiles meet at positions in more ways than a design has.
  if (bdShortfallOf.size() > (1u << 16))
    bdShortfallOf.clear();
  // Every tile at the position holds what its programs take; a tile with
  // fixed coordinates is only ever here, so the list keys the position too.
  SmallVector<Operation *> holders(tiles);
  for (Operation *tile : logicalMemTiles)
    if (auto at = currentPlacement.find(tile);
        at != currentPlacement.end() && at->second == position &&
        !llvm::is_contained(holders, tile))
      holders.push_back(tile);
  for (Operation *tile : physicalTiles.lookup(position))
    if (!llvm::is_contained(holders, tile))
      holders.push_back(tile);
  std::pair<TileID, std::vector<Operation *>> key{
      layout->second, std::vector<Operation *>(holders.begin(), holders.end())};
  if (auto cached = bdShortfallOf.find(key); cached != bdShortfallOf.end()) {
    if (walked)
      *walked = DenseSet<std::pair<Operation *, int>>(
          cached->second.second.begin(), cached->second.second.end());
    return cached->second.first;
  }
  SmallVector<const MemTileEnd *> ends;
  for (Operation *tile : tiles)
    if (auto *it = memTileEnds.find(tile); it != memTileEnds.end())
      for (const MemTileEnd &end : it->second)
        ends.push_back(&end);
  auto [col, row] = position;
  int channels[2] = {
      static_cast<int>(targetModel->getNumDestSwitchboxConnections(
          col, row, WireBundle::DMA)),
      static_cast<int>(targetModel->getNumSourceSwitchboxConnections(
          col, row, WireBundle::DMA))};
  auto free = [&](DMAChannelDir dir, int channel) {
    return llvm::all_of(holders, [&](Operation *tile) {
      return channelAnalysis->isChannelFree(cast<TileLike>(tile), dir, channel);
    });
  };
  DenseMap<int, int64_t> taken = takenBDs.at(*targetModel, col, row, holders);
  auto spent = [&](int channel) { return taken.lookup(channel); };
  auto unspent = [](int) -> int64_t { return 0; };
  DenseSet<std::pair<Operation *, int>> whole;
  int64_t shortfall = 0;
  while (true) {
    SmallVector<DMAChannelAnalysis::BDPlanEnd> planned, unweighted;
    int64_t total = 0;
    for (const MemTileEnd *end : ends) {
      int64_t bds =
          whole.contains({end->end.pool, end->end.id}) ? 1 : end->end.bds;
      int limit = channels[end->dir == DMAChannelDir::MM2S];
      planned.push_back({end->dir, bds, limit});
      unweighted.push_back({end->dir, 0, limit});
      total += bds;
    }
    if (DMAChannelAnalysis::planBDChannels(*targetModel, col, row, planned,
                                           free, spent))
      break;
    const MemTileEnd *deepest = nullptr;
    for (const MemTileEnd *end : ends)
      if (end->end.walkable && !whole.contains({end->end.pool, end->end.id}) &&
          (!deepest || end->end.depth > deepest->end.depth))
        deepest = end;
    if (deepest) {
      whole.insert({deepest->end.pool, deepest->end.id});
      continue;
    }
    // Too few channels is the DMA term's to charge.
    int64_t ids = targetModel->getNumBDs(col, row);
    for (auto [channel, held] : taken)
      total += held;
    shortfall = DMAChannelAnalysis::planBDChannels(*targetModel, col, row,
                                                   unweighted, free, unspent)
                    ? std::max<int64_t>(1, total - ids)
                    : std::max<int64_t>(0, total - ids);
    break;
  }
  bdShortfallOf[key] = {shortfall, {whole.begin(), whole.end()}};
  if (walked)
    *walked = std::move(whole);
  return shortfall;
}

// Penalizes mem tile positions whose ends' BD ids no channel plan fits.
int SAPlacer::computeBDCountPenalty() {
  bdShortfallAt.clear();
  int64_t penalty = 0;
  for (auto &[tile, ends] : memTileEnds) {
    auto at = currentPlacement.find(tile);
    if (at == currentPlacement.end() || bdShortfallAt.count(at->second))
      continue;
    int64_t shortfall = bdShortfall(at->second, memTilesAt(at->second));
    bdShortfallAt[at->second] = shortfall;
    penalty += shortfall * config.dmaPenaltyPerChannel;
  }
  return static_cast<int>(penalty);
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

int SAPlacer::computePairingPenalty() const {
  int penalty =
      computeAdjacencyPenalty(cascadeAdjacency, targetModel->getCascadeSteps(),
                              config.cascadeWeightPerDist);
  for (auto [first, second] : sharedMemAdjacency.edges) {
    auto firstIt = currentPlacement.find(first.getOperation());
    auto secondIt = currentPlacement.find(second.getOperation());
    if (firstIt == currentPlacement.end() || secondIt == currentPlacement.end())
      continue;
    TileID a = firstIt->second, b = secondIt->second;
    if (a == b || targetModel->isLegalMemAffinity(a.col, a.row, b.col, b.row) ||
        targetModel->isLegalMemAffinity(b.col, b.row, a.col, a.row))
      continue;
    int distance = std::abs(a.col - b.col) + std::abs(a.row - b.row);
    penalty += config.sharedMemWeightPerDist * std::max(1, distance - 1);
  }
  return penalty;
}

void SAPlacer::generateAllocates() {
  int64_t coreCapacity = targetModel->getLocalMemorySize();
  DenseSet<size_t> emittedFifoIndices;

  // Case A: Intratile fifo relocation for overflowing core tiles.
  // Snapshot overflowing tiles first to avoid iterator invalidation
  // (delegate updates modify currentMemUsage during packing).
  for (TileID tilePos : overflowingCoreTiles()) {

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
      if (fb.consumers.size() != 1 || !fb.producer || !fb.consumers[0] ||
          fb.delegate || fb.forcesDMA)
        continue;
      auto prodIt = currentPlacement.find(fb.producer);
      auto consIt = currentPlacement.find(fb.consumers[0]);
      if (prodIt == currentPlacement.end() || consIt == currentPlacement.end())
        continue;
      if (prodIt->second != tilePos || consIt->second != tilePos)
        continue;
      int64_t sz =
          std::max(fb.producerSizeBytes, fb.consumerSizeBytes) * fb.sharedDepth;
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
  SmallVector<std::pair<Operation *, TileID>, 3> savedPositions;
  for (auto &[op, oldPos] : oldPlacements) {
    TileID &pos = currentPlacement[op];
    savedPositions.push_back({op, pos});
    pos = oldPos;
  }
  for (size_t fi : affectedFifos)
    addFifoContribution(fi, -1);
  for (auto &[op, _] : oldPlacements)
    for (size_t ei : tileToRouteEnds.lookup(op))
      addRouteEndContribution(ei, -1);

  // Restore new positions and add new contributions
  for (auto &[op, newPos] : savedPositions)
    currentPlacement[op] = newPos;
  for (size_t fi : affectedFifos)
    addFifoContribution(fi, +1);
  for (auto &[op, _] : oldPlacements)
    for (size_t ei : tileToRouteEnds.lookup(op))
      addRouteEndContribution(ei, +1);

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

  auto reads = [&](const DenseSet<Operation *> &tiles) {
    return llvm::any_of(oldPlacements, [&](const auto &moved) {
      return tiles.contains(moved.first);
    });
  };
  if (reads(spillTiles))
    memSpilloverPenalty = computeMemSpilloverPenalty();
  bdUndo.clear();
  SmallVector<TileID> bdTouched;
  for (auto &[op, oldPos] : oldPlacements)
    if (memTileEnds.count(op) || llvm::is_contained(logicalMemTiles, op))
      for (TileID pos : {oldPos, currentPlacement[op]})
        if (!llvm::is_contained(bdTouched, pos))
          bdTouched.push_back(pos);
  for (TileID pos : bdTouched) {
    int64_t &at = bdShortfallAt[pos];
    bdUndo.push_back({pos, at});
    bdCountPenalty -= static_cast<int>(at * config.dmaPenaltyPerChannel);
    at = bdShortfall(pos, memTilesAt(pos));
    bdCountPenalty += static_cast<int>(at * config.dmaPenaltyPerChannel);
  }
  if (reads(delegateTermTiles))
    delegatePenalty = computeDelegatePenalty();

  // DMA usage changed only where the affected fifos' ends and the moved
  // tiles sit, before or after the move, and so did which shim ends share.
  llvm::SmallDenseSet<TileID, 16> dmaTouched;
  for (auto &[op, oldPos] : oldPlacements) {
    dmaTouched.insert(oldPos);
    dmaTouched.insert(currentPlacement[op]);
  }
  for (size_t fi : affectedFifos) {
    const auto &fb = fifoBuffers[fi];
    SmallVector<Operation *, 4> ends = {fb.producer};
    llvm::append_range(ends, fb.consumers);
    for (Operation *end : ends)
      if (auto it = currentPlacement.find(end); it != currentPlacement.end())
        dmaTouched.insert(it->second);
  }
  dmaUndo.clear();
  for (TileID tile : dmaTouched) {
    int &at = dmaPenaltyAt[tile];
    dmaUndo.push_back({tile, at});
    int now = computeTileDMAPenalty(tile);
    dmaPenalty += now - at;
    at = now;
  }

  cachedResourcePenalty = memSpilloverPenalty + computeCoreOverflowPenalty() +
                          dmaPenalty + bdCountPenalty + delegatePenalty;
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
    Operation *tile = movableTiles[drawBoundedIndex(rng, movableTiles.size())];

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

    TileID targetPos = candidates[drawBoundedIndex(rng, candidates.size())];

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
  for (const auto &fb : fifoBuffers) {
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
      weight +=
          std::max(fb.producerSizeBytes, fb.consumerSizeBytes) * fb.sharedDepth;
    } else if (isProd) {
      weight += fb.producerSizeBytes * fb.producerDepth;
    } else {
      weight += fb.consumerSizeBytes * fb.consumerDepths[consIdx];
    }
  }
  return weight;
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
    int64_t bytes = bufOp.getAllocationSize();
    if (tileTypes.lookup(tileOp) == AIETileType::MemTile)
      memTileFixed[tileOp].push_back(
          {bytes,
           bufOp.getAligned() ? targetModel->getMemTileLoadStoreBusWidth() / 8
                              : 1,
           bufOp.getAddress()});
    staticBufferSizes[tileOp] +=
        bufOp.getAligned()
            ? alignedBufferBytes(*targetModel, tileTypes.lookup(tileOp), bytes)
            : bytes;
  });

  // Collect core stack and data regions (reserved by assign-buffer-addresses)
  device.walk([&](CoreOp coreOp) {
    auto *tileOp = coreOp.getTile().getDefiningOp();
    if (!tileOp || !isa<LogicalTileOp>(tileOp))
      return;
    // The data region a core sets aside for its own sections is as much its
    // memory as the stack.
    stackSizes[tileOp] =
        alignedBufferBytes(*targetModel, AIETileType::CoreTile,
                           coreOp.getEffectiveStackSize()) +
        alignedBufferBytes(
            *targetModel, AIETileType::CoreTile,
            coreOp.getDataSize().value_or(coreOp.getMeasuredDataSize().value_or(
                config.coreSectionsBytes)));
  });

  // Build net model, fifo buffer info, and cascade groups
  buildNetModel(objectFifos, objectFifoLinks);
  buildFifoBufferInfo(device, objectFifos, objectFifoLinks);
  buildRouteModel(device, collected.flows, collected.pktFlows, collected.routes,
                  channelAnalysis.emplace(device).getChannelClaims());
  shareableShimEnds = collectShareableShimEnds(device, objectFifos);
  shimSpans.emplace(
      device, [&](StringAttr symbol) { return shimEndOf(device, symbol); },
      provenShimGates(device));
  runtimeBdIds = RuntimeBdIds(device, objectFifos, *targetModel);
  takenBDs = DMAChannelAnalysis::TakenBDs(device);
  logicalMemTiles.clear();
  for (auto tile : device.getOps<LogicalTileOp>())
    if (tile.getTileType() == AIETileType::MemTile)
      logicalMemTiles.push_back(tile);
  physicalTiles.clear();
  for (TileOp tile : device.getOps<TileOp>())
    physicalTiles[{tile.getCol(), tile.getRow()}].push_back(tile);
  cascadeAdjacency = buildCascadeAdjacency(collected.cascadeFlows);
  LLVM_DEBUG(llvm::dbgs() << "[SA] Cascade adjacency: "
                          << cascadeAdjacency.edges.size() << " edges\n");
  for (ObjectFifoCreateOp fifo : objectFifos)
    if (fifo.forcesSharedMemory() && fifo.getConsumerTiles().size() == 1)
      sharedMemAdjacency.addEdgeFromValues(fifo.getProducerTile(),
                                           fifo.getConsumerTiles()[0]);

  return success();
}

void SAPlacer::buildFifoBufferInfo(DeviceOp device,
                                   ArrayRef<ObjectFifoCreateOp> objectFifos,
                                   ArrayRef<ObjectFifoLinkOp> objectFifoLinks) {
  mlir::DataLayout dataLayout(device->getParentOfType<ModuleOp>());

  memTileEnds.clear();
  llvm::StringMap<Operation *> delegates;
  for (auto alloc : device.getOps<ObjectFifoAllocateOp>())
    delegates[alloc.getObjFifoName()] = alloc.getDelegateTile().getDefiningOp();

  for (auto ofOp : objectFifos) {
    FifoBufferInfo fb;
    fb.fifoOp = ofOp.getOperation();
    fb.delegate = delegates.lookup(ofOp.getSymName());

    // Get producer
    auto *prodOp = ofOp.getProducerTile().getDefiningOp();
    fb.producer = (isa_and_nonnull<LogicalTileOp>(prodOp)) ? prodOp : nullptr;
    fb.producerTile = prodOp;

    // Null for a placed consumer, so indices stay the fifo's own.
    for (Value ct : ofOp.getConsumerTiles()) {
      auto *consOp = ct.getDefiningOp();
      fb.consumers.push_back(isa_and_nonnull<LogicalTileOp>(consOp) ? consOp
                                                                    : nullptr);
      fb.consumerTiles.push_back(consOp);
    }

    // Compute buffer element size (producer side)
    auto fifoType = llvm::cast<AIEObjectFifoType>(ofOp.getElemType());
    auto elemType = llvm::cast<MemRefType>(fifoType.getElementType());
    int64_t producerBytes =
        elemType.getNumElements() *
        static_cast<int64_t>(dataLayout.getTypeSize(elemType.getElementType()));
    auto prodLike = dyn_cast_or_null<TileLike>(prodOp);
    AIETileType prodType =
        prodLike ? prodLike.getTileType() : AIETileType::CoreTile;
    fb.producerSizeBytes =
        alignedBufferBytes(*targetModel, prodType, producerBytes);

    // Consumer element size (may differ with consumerElemType)
    int64_t consumerBytes = producerBytes;
    if (auto consType = ofOp.getConsumerElemType()) {
      auto consOFType = llvm::cast<AIEObjectFifoType>(consType.value());
      auto consMemref = llvm::cast<MemRefType>(consOFType.getElementType());
      consumerBytes = consMemref.getNumElements() *
                      static_cast<int64_t>(
                          dataLayout.getTypeSize(consMemref.getElementType()));
    }
    // One size serves every consumer, so align it for a core tile if any
    // consumer is one.
    bool coreConsumer = llvm::any_of(fb.consumerTiles, [&](Operation *cons) {
      auto tile = dyn_cast_or_null<TileLike>(cons);
      return tile && tile.getTileType() == AIETileType::CoreTile;
    });
    fb.consumerSizeBytes = alignedBufferBytes(
        *targetModel,
        coreConsumer ? AIETileType::CoreTile : AIETileType::MemTile,
        consumerBytes);

    fb.sharedDepth = ofOp.sharedDepth();
    fb.producerDepth = ofOp.dmaDepth(std::nullopt);
    for (unsigned ci = 0, e = fb.consumers.size(); ci < e; ++ci)
      fb.consumerDepths.push_back(ofOp.dmaDepth(ci));

    fb.producerEnd = ofOp.plannedDmaEnd(std::nullopt);
    for (unsigned ci = 0, e = fb.consumers.size(); ci < e; ++ci)
      fb.consumerEnds.push_back(ofOp.plannedDmaEnd(ci));
    auto onMemTile = [&](Operation *tile) {
      return tile && tileTypes.lookup(tile) == AIETileType::MemTile;
    };
    if (fb.producerEnd && onMemTile(fb.producer))
      memTileEnds[fb.producer].push_back(
          {DMAChannelDir::MM2S, *fb.producerEnd});
    for (unsigned ci = 0, e = fb.consumers.size(); ci < e; ++ci)
      if (const std::optional<PlannedDmaEnd> &end = fb.consumerEnds[ci];
          end && onMemTile(fb.consumers[ci]))
        memTileEnds[fb.consumers[ci]].push_back({DMAChannelDir::S2MM, *end});

    std::optional<ObjectFifoLinkOp> link = ofOp.getLink();
    fb.forcesDMA = ofOp.dmaReason().has_value();
    fb.linked = link.has_value();
    // A link output's producer objects are the link input's consumer pool.
    fb.linkSharedProd =
        link && llvm::is_contained(link->getOutputObjectFifos(), ofOp);
    fifoBuffers.push_back(fb);
  }

  LLVM_DEBUG(llvm::dbgs() << "[SA] seed=" << rngSeed << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[SA] Collected " << fifoBuffers.size()
                          << " ObjectFifo buffer requirements\n");
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
  deterministicShuffle(availShim, rng);

  size_t compIdx = 0, memIdx = 0, shimIdx = 0;

  // A core position holds one tile. Mem and shim positions hold several, as
  // moves and allocation allow, so a pinned one only sends its position to
  // the back of the pool, after the free ones.
  SmallVector<TileID> pinnedMem, pinnedShim;
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

      if (type == AIETileType::CoreTile) {
        llvm::erase(availComp, pos);
      } else if (llvm::is_contained(availMem, pos)) {
        llvm::erase(availMem, pos);
        pinnedMem.push_back(pos);
      } else if (llvm::is_contained(availShim, pos)) {
        llvm::erase(availShim, pos);
        pinnedShim.push_back(pos);
      }
    }
  }
  llvm::append_range(availMem, pinnedMem);
  llvm::append_range(availShim, pinnedShim);

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

  cascadePen = computePairingPenalty();
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

  // A large effort saturates rather than overflowing the int conversion.
  auto scaled = [&](int budget) {
    double v = std::min(budget * config.effort,
                        static_cast<double>(std::numeric_limits<int>::max()));
    return std::max(1, static_cast<int>(v));
  };
  movesPerIter = scaled(movesPerIter);
  // Same floor as movesPerIter above: a small positive effort must still
  // scale the greedy stage down, but never to zero -- skipping it entirely
  // is what leaves violations unresolved (see finalizePlacement's legality
  // check).
  greedyIters = scaled(greedyIters);

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
      cascadePen = computePairingPenalty();
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
  int oldMemSpillover = memSpilloverPenalty, oldBDCount = bdCountPenalty,
      oldDelegate = delegatePenalty, oldDMA = dmaPenalty;
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
  int oldPressure = cachedPressure;
  int newResPenalty = updateResourcePenalty(oldPlacements);
  int newCascade = llvm::any_of(moves,
                                [&](const auto &move) {
                                  return pairedTiles.contains(move.first);
                                })
                       ? computePairingPenalty()
                       : cascadePen;
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
       drawUnitReal(rng) < std::exp(-delta / schedule.getTemperature()));

  if (accept) {
    if (delta > 0)
      acceptedUphill++;
    totalCost += delta;
    cascadePen = newCascade;
    cachedPressure = newPressure;
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
    memSpilloverPenalty = oldMemSpillover;
    bdCountPenalty = oldBDCount;
    delegatePenalty = oldDelegate;
    dmaPenalty = oldDMA;
    for (auto &[tile, at] : dmaUndo)
      dmaPenaltyAt[tile] = at;
    for (auto &[tile, at] : llvm::reverse(bdUndo))
      bdShortfallAt[tile] = at;
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
  int finalCascade = computePairingPenalty();
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
  finalCost = totalCost;
  if (bestOverallCost < INT_MAX) {
    if (bestCost < INT_MAX) {
      currentPlacement = bestPlacement;
      finalCost = bestCost;
    } else {
      currentPlacement = bestOverallPlacement;
      finalCost = bestOverallCost;
    }
    physToLogical.clear();
    for (auto &[op, pos] : currentPlacement)
      physToLogical[pos] = op;
    initResourceTracking();
  }

  // Generate allocates (state is fresh from initResourceTracking)
  generateAllocates();

  auto endTime = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);

  LLVM_DEBUG(printPlacementStats(elapsed.count()));

  // SA optimizes for legality but is not guaranteed to reach it (e.g. a very
  // low --sa-effort can starve the greedy stage that normally drives hard
  // violations to zero). Cascade and shared memory pairs and delegate reach
  // are exact, and a placement breaking them would only fail later and less
  // clearly, so they fail here. Memory, DMA channel and BD use are this pass's
  // estimates of what later passes allocate, and those passes check them
  // exactly, so an overrun here is a warning.
  int finalPairing = computePairingPenalty();
  int finalDelegate = computeDelegatePenalty();
  if (finalPairing != 0 || finalDelegate != 0) {
    device.emitError("SA placer failed to find a legal placement (cascade and "
                     "shared memory penalty=")
        << finalPairing << ", delegate penalty=" << finalDelegate
        << "); try increasing --sa-effort";
    return failure();
  }
  if (int estimated = getResourcePenalty() - finalDelegate; estimated != 0)
    device.emitWarning("SA placer's resource estimate is over capacity "
                       "(penalty=")
        << estimated
        << "); later passes check exactly, and increasing --sa-effort may help";

  result = currentPlacement;
  return success();
}
