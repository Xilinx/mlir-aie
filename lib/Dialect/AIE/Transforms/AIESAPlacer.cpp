//===- AIESAPlacer.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEBufferAllocation.h"
#include "aie/Dialect/AIE/Transforms/AIEDMAChannelAnalysis.h"
#include "aie/Dialect/AIE/Transforms/AIEPlacer.h"
#include "aie/Dialect/AIE/Transforms/AIEPoolLocks.h"
#include "llvm/ADT/BitVector.h"
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

/// Return the neighbouring tiles an aie.objectfifo.allocate can move a fifo
/// with both ends on `tilePos` to, by the rule split checks it with.
static SmallVector<TileID>
getMemAffinityNeighbors(const AIETargetModel &targetModel, TileID tilePos) {
  SmallVector<TileID> neighbors;
  for (std::optional<TileID> candidate :
       {targetModel.getMemWest(tilePos), targetModel.getMemEast(tilePos),
        targetModel.getMemNorth(tilePos), targetModel.getMemSouth(tilePos)})
    if (candidate && *candidate != tilePos &&
        ObjectFifoCreateOp::delegateReaches(targetModel, *candidate, tilePos,
                                            tilePos))
      neighbors.push_back(*candidate);
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
  namedChannels.clear();
  for (auto &[tile, inOut] :
       flowChannelDemand(flows, pktFlows, routes, claims, &namedChannels)) {
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
    if (!isa_and_nonnull<TileLike>(tileOp))
      continue;
    staticBufferSizes[tileOp] += pool.getObjectSizeInBytes() * pool.getDepth();
  }
  // Ends already split count the BD ids they take; other DMA ends only hold
  // a channel.
  for (auto endpoint : device.getOps<RouteEndpoint>()) {
    auto tile = dyn_cast_or_null<TileLike>(endpoint.getTile().getDefiningOp());
    if (endpoint.getRouteBundle() != WireBundle::DMA || !tile ||
        tile.getTileType() != AIETileType::MemTile)
      continue;
    std::optional<int> channel = endpoint.getRouteChannel();
    PlannedDmaEnd end{0, nullptr, -1, 0, false, 0, 0, channel};
    if (auto dma = dyn_cast<ObjectFifoDmaEndpointOp>(endpoint.getOperation())) {
      ObjectFifoPoolOp pool = dma.getPoolOp();
      end = {dma.getNumBDs(),
             pool,
             -1,
             pool.getDepth(),
             pool.canWalkWhole(),
             pool.getObjectSizeInBytes(),
             pool.getContiguous() ? pool.getObjectsPerBuffer() : 0,
             channel};
    }
    memTileEnds[tile].push_back({endpoint.getRouteDirection(), end});
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

  return satisfiesConstraints(tile, pos) &&
         !namedClash(tile, namedAt(tile, pos));
}

SmallVector<Operation *> SAPlacer::namedAt(Operation *tile, TileID pos,
                                           Operation *leaving) const {
  SmallVector<Operation *> here;
  if (!namedChannels.contains(tile))
    return here;
  for (auto &[other, named] : namedChannels)
    if (other != tile && other != leaving && positionOf(other) == pos)
      here.push_back(other);
  return here;
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
  bool prodIsShim = isShimEndType(types.producer);

  // A fifo that already has an aie.objectfifo.allocate keeps its shared-memory
  // objects on that delegate tile.
  std::optional<TileID> delegatePos = positionOf(fb.delegate);

  // A fifo shared memory carries keeps one pool of sharedDepth() objects;
  // otherwise each end off a shim keeps its own, except a link output's
  // producer, whose objects are the link input's consumer pool.
  bool prodCharged = false;
  bool prodNeedsDMA = false;
  // The producer's pool, or a shared one, holds the initial objects; a
  // consumer's starts empty.
  auto heldLocks = [&](int64_t depth) {
    return PoolLocks::needsLocks(fb.unsynchronized, fb.initObjects, depth,
                                 fb.producerCore, fb.rearmed);
  };
  auto addLocks = [&](TileID pos, int segments, int64_t depth, bool needed) {
    if (!needed || !segments)
      return;
    SmallVector<int> counts =
        LockPlan::groupCounts(*targetModel, depth, segments);
    int locks = std::accumulate(counts.begin(), counts.end(), 0);
    currentLockDemand[pos] += sign * locks;
    lockUndo.push_back({pos, sign * locks});
  };
  using LinkPool = FifoBufferInfo::LinkPool;
  auto chargeLinkPool = [&](TileID pos, const PlannedDmaEnd &end) {
    int count = end.contiguousObjects ? 1 : end.depth;
    int64_t bytes =
        end.objectBytes * std::max<int64_t>(end.contiguousObjects, 1);
    currentMemUsage[pos] +=
        sign * count *
        alignedBufferBytes(*targetModel,
                           targetModel->getTileType(pos.col, pos.row), bytes);
    chargeCoreFifo(pos, bytes, sign * count);
  };
  // A link's ends keep no objects of their own, shared memory or not: the
  // split builds the link's one pool on its tile.
  bool prodLinked = fb.producerLinkPool != LinkPool::None;
  if (const std::optional<PlannedDmaEnd> &plan = fb.producerLinkPlan) {
    chargeLinkPool(prodPos, *plan);
    addLocks(prodPos, fb.producerLinkSegments, plan->depth, true);
  }

  for (size_t ci = 0; ci < fb.consumerTiles.size(); ci++) {
    std::optional<TileID> consAt = positionOf(fb.consumerTiles[ci]);
    if (!consAt)
      continue;
    TileID consPos = *consAt;
    bool consIsShim = isShimEndType(types.consumers[ci]);

    using SharedMemory = AIETargetModel::SharedMemory;
    SharedMemory shared = fb.forcesDMA
                              ? SharedMemory::None
                              : ObjectFifoCreateOp::sharedMemoryOf(
                                    *targetModel, prodPos, consPos, delegatePos,
                                    positionOf(fb.linkTile));
    bool isSharedMem = shared != SharedMemory::None;
    int64_t sharedBytes =
        std::max(fb.producerSizeBytes, fb.consumerSizeBytes) * fb.sharedDepth;

    int64_t sharedRaw = std::max(fb.producerRawBytes, fb.consumerRawBytes);
    bool consLinked = fb.consumerLinkPools[ci] != LinkPool::None;
    if (const std::optional<PlannedDmaEnd> &plan = fb.consumerLinkPlans[ci]) {
      chargeLinkPool(consPos, *plan);
      addLocks(consPos, fb.consumerLinkSegments[ci], plan->depth, true);
    }
    bool sharedPool = isSharedMem && !prodLinked && !consLinked;
    if (prodPos == consPos && sharedPool) {
      currentMemUsage[delegatePos.value_or(prodPos)] += sign * sharedBytes;
      chargeCoreFifo(delegatePos.value_or(prodPos), sharedRaw,
                     sign * fb.sharedDepth);
      if (ci == 0)
        addLocks(delegatePos.value_or(prodPos), 1, fb.sharedDepth,
                 heldLocks(fb.sharedDepth));
    } else if (sharedPool) {
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
      chargeCoreFifo(bufTile, sharedRaw, sign * fb.sharedDepth);
      if (ci == 0)
        addLocks(bufTile, 1, fb.sharedDepth, heldLocks(fb.sharedDepth));
    } else if (!isSharedMem) {
      if (!prodCharged && !prodLinked && !prodIsShim && !fb.linkSharedProd) {
        currentMemUsage[prodPos] +=
            sign * fb.producerSizeBytes * fb.producerDepth;
        chargeCoreFifo(prodPos, fb.producerRawBytes, sign * fb.producerDepth);
        if (fb.producerEnd)
          addLocks(prodPos, 1, fb.producerDepth, heldLocks(fb.producerDepth));
        prodCharged = true;
      }
      if (!consLinked && !consIsShim) {
        currentMemUsage[consPos] +=
            sign * fb.consumerSizeBytes * fb.consumerDepths[ci];
        chargeCoreFifo(consPos, fb.consumerRawBytes,
                       sign * fb.consumerDepths[ci]);
        if (fb.consumerEnds[ci])
          addLocks(consPos, 1, fb.consumerDepths[ci], !fb.unsynchronized);
      }
    }

    // DMA contribution: a fifo that shared memory carries does not run
    // through the DMAs, back to its own tile as well. An end takes a channel
    // of its own unless it is a stream port, a PLIO, or a merge's or
    // dispatch's participant whose shared end another took.
    if (!isSharedMem) {
      bool plio = cast<ObjectFifoCreateOp>(fb.fifoOp).getPlio();
      if (consIsShim ? !plio : fb.consumerEnds[ci].has_value())
        drawChannel(consPos, false, fb.consumerPins[ci], sign);
      prodNeedsDMA = prodIsShim ? !plio : fb.producerEnd.has_value();
    }
  }

  if (prodNeedsDMA)
    drawChannel(prodPos, true, fb.producerPin, sign);
}

void SAPlacer::drawChannel(TileID position, bool output,
                           const std::optional<PinnedChannels::End> &pin,
                           int sign) {
  int taken = pin ? pinUse.place(position, *pin, sign) : sign;
  auto &[in, out] = currentDMAUsage[position];
  (output ? out : in) += taken;
}

void SAPlacer::initResourceTracking() {
  currentMemUsage.clear();
  currentLockDemand.clear();
  SmallVector<std::pair<unsigned, int64_t>> depths;
  auto at = [this](Value tile) { return positionOf(tile.getDefiningOp()); };
  for (auto [index, variable] : llvm::enumerate(variableLinks))
    depths.push_back({index, variable.plan.depthAt(*targetModel, at)});
  replanLinks(depths);
  currentDMAUsage.clear();
  pinUse = PinnedChannelUse(pinnedRule);
  sharedMemDestination.clear();
  coreFifoBuffers.clear();

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
  for (auto &[op, bufSize] : staticBufferSizes)
    if (std::optional<TileID> pos = positionOf(op))
      currentMemUsage[*pos] += bufSize;
  // Add core stack overhead (reserved by assign-buffer-addresses pass)
  for (auto &[op, bytes] : stackSizes)
    if (std::optional<TileID> pos = positionOf(op))
      currentMemUsage[*pos] += bytes;

  for (size_t i = 0; i < fifoBuffers.size(); i++)
    addFifoContribution(i, +1);
  coreFifoLog.clear();
  coreUnplacedAt.clear();
  for (auto &[op, _] : coreContents)
    if (std::optional<TileID> pos = positionOf(op))
      coreDirty.insert(*pos);
  refreshCoreUnplaced();

  tileToRouteEnds.clear();
  for (size_t i = 0; i < routeEnds.size(); i++) {
    tileToRouteEnds[routeEnds[i].tile].push_back(i);
    addRouteEndContribution(i, +1);
  }
  for (auto &[tile, claims] : claimsOf)
    if (std::optional<TileID> pos = positionOf(tile))
      for (const DMAChannelAnalysis::ChannelClaim &claim : claims)
        pinUse.claim(*pos, claim, 1);

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
  coreOverflowTiles = overflowingCoreTiles();
  coreOverflowPenalty = computeCoreOverflowPenalty();
  cachedResourcePenalty = memSpilloverPenalty + coreOverflowPenalty +
                          computeLockPenalty() + dmaPenalty + bdCountPenalty +
                          delegatePenalty;
  cachedPressure = computeMemoryPressure();
}

int SAPlacer::computeMemoryPressure() const {
  int64_t coreCapacity = targetModel->getLocalMemorySize();
  int64_t memTileCapacity = targetModel->getMemTileSize();
  int pressure = 0;

  for (auto &[tilePos, used] : currentMemUsage) {
    auto tileType = targetModel->getTileType(tilePos.col, tilePos.row);
    if (targetModel->isShimNOCorPLTile(tilePos.col, tilePos.row))
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

// The objectfifo split keeps an allocated fifo's objects where
// ObjectFifoCreateOp::allocatedHome says, which both ends must reach.
// Otherwise it moves a link's participant that does not own the link's pool by
// DMA, and rejects any other allocate. A fifo that shares a link's pool must
// fit it, or ObjectFifoAllocateOp's verifier rejects the placement.
int SAPlacer::computeDelegatePenalty(SmallVectorImpl<TileID> *charged) const {
  int penalty = 0;
  for (const auto &fb : fifoBuffers) {
    std::optional<TileID> delegatePos = positionOf(fb.delegate);
    if (!delegatePos)
      continue;
    std::optional<TileID> prodPos = positionOf(fb.producerTile);
    std::optional<TileID> consPos = positionOf(fb.consumerTiles.front());
    if (!prodPos || !consPos)
      continue;
    std::optional<TileID> linkPos = positionOf(fb.linkTile);
    bool home = ObjectFifoCreateOp::allocatedHome(
        *targetModel, *prodPos, *consPos, *delegatePos, linkPos);
    if (home ? fb.fitsLinkPool : fb.allocateFallsBackToDMA)
      continue;
    if (home) {
      penalty += config.delegateWeightPerDist;
      if (charged)
        charged->append({*delegatePos, *prodPos, *consPos});
      continue;
    }
    for (std::optional<TileID> held : {delegatePos, linkPos}) {
      if (!held)
        continue;
      for (TileID end : {*prodPos, *consPos}) {
        if (ObjectFifoCreateOp::delegateReaches(*targetModel, *held, end, end))
          continue;
        int dist =
            std::abs(held->col - end.col) + std::abs(held->row - end.row);
        penalty += config.delegateWeightPerDist * std::max(dist - 1, 1);
        if (charged)
          charged->append({*held, end});
      }
    }
  }
  return penalty;
}

// Charges by the KB what MemTileSpillSearch, in the allocator's attempt order,
// cannot place.
int SAPlacer::computeMemSpilloverPenalty(SmallVectorImpl<TileID> *charged) {
  std::vector<int64_t> key;
  auto keyPosition = [&](Operation *op) {
    std::optional<TileID> at = positionOf(op);
    key.push_back(at ? at->col : -1);
    key.push_back(at ? at->row : -1);
  };
  for (auto &[op, _] : memTileEnds)
    keyPosition(op);
  for (auto &[op, _] : memTileFixed)
    keyPosition(op);
  for (Operation *tile : logicalMemTiles)
    keyPosition(tile);
  for (const VariableLink &variable : variableLinks)
    key.push_back(variable.depth);
  if (const int *known = charged ? nullptr : spillPenaltyOf.find(key))
    return *known;

  SmallVector<MemTileSpillSearch::Tile> tiles;
  DenseMap<TileID, unsigned> index;
  for (int col = 0; col < targetModel->columns(); ++col)
    for (int row = 0; row < targetModel->rows(); ++row)
      if (targetModel->getTileType(col, row) == AIETileType::MemTile) {
        index[{col, row}] = tiles.size();
        tiles.push_back({targetModel->getMemTileSize(), {}});
      }
  for (auto &[op, fixed] : memTileFixed)
    if (std::optional<TileID> at = positionOf(op))
      llvm::append_range(tiles[index.lookup(*at)].fixed, fixed);

  // Each placed position's pools, with the sets of them it may walk whole.
  struct Position {
    TileID at;
    llvm::MapVector<std::pair<Operation *, int>,
                    std::pair<const PlannedDmaEnd *, std::pair<int, int>>>
        pools;
    WalkSets sets;
    SmallVector<int64_t> bytes;
  };
  SmallVector<Position> positions;
  DenseSet<TileID> seen;
  for (auto &[tile, _] : memTileEnds) {
    std::optional<TileID> at = positionOf(tile);
    if (!at || !seen.insert(*at).second)
      continue;
    Position &position = positions.emplace_back();
    position.at = *at;
    SmallVector<Operation *> here = memTilesAt(*at);
    (void)bdShortfall(*at, here, &position.sets, &position.bytes);
    for (Operation *op : here)
      for (const MemTileEnd &end : memTileEnds.find(op)->second) {
        const PlannedDmaEnd &plan = end.end;
        auto split = dyn_cast_or_null<ObjectFifoPoolOp>(plan.pool);
        if (!plan.pool || (split && split.getBuffers()))
          continue;
        auto &[first, demand] = position.pools[{plan.pool, plan.id}];
        first = first ? first : &plan;
        ++(end.dir == DMAChannelDir::S2MM ? demand.first : demand.second);
      }
  }
  llvm::sort(positions, [](const Position &a, const Position &b) {
    return std::make_pair(a.at.col, a.at.row) <
           std::make_pair(b.at.col, b.at.row);
  });
  bool anyWalked =
      !chargedPools.empty() || llvm::any_of(positions, [](const Position &p) {
        return !p.sets.empty();
      });

  // Per pool of the positions `members`, in the allocator's size-first order,
  // with each position walking the set `walk` gives it (none when null).
  using Walk = const SmallVector<std::pair<Operation *, int>> *;
  auto shapes = [&](ArrayRef<unsigned> members, ArrayRef<Walk> walk) {
    SmallVector<std::pair<SpillPool, TileID>> shaped;
    for (unsigned member : members) {
      const Position &position = positions[member];
      unsigned home = index.lookup(position.at);
      for (auto &[key, entry] : position.pools) {
        auto [plan, demand] = entry;
        int busier = std::max(demand.first, demand.second);
        int64_t whole = plan->contiguousObjects;
        if (!whole &&
            (chargedPools.contains(plan->pool) ||
             (walk[member] && llvm::is_contained(*walk[member], key))))
          whole = plan->depth;
        shaped.push_back({{plan->objectBytes * std::max<int64_t>(whole, 1),
                           whole ? 1 : plan->depth, home, busier},
                          position.at});
      }
    }
    llvm::stable_sort(shaped, [](const auto &a, const auto &b) {
      return std::make_pair(a.second.col, a.second.row) <
             std::make_pair(b.second.col, b.second.row);
    });
    sortLargestBufferFirst(shaped,
                           [](const auto &pool) { return pool.first.bytes; });
    return shaped;
  };

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
  // A component's outcome depends only on its tiles' contents and its walk
  // sets, so it is cached. bdShortfall's sets also depend on every tile at a
  // position whose programs hold channels or BD ids there.
  DenseMap<TileID, std::vector<Operation *>> opsAt;
  for (auto &[op, _] : memTileEnds)
    if (std::optional<TileID> at = positionOf(op))
      opsAt[*at].push_back(op);
  for (auto &[op, _] : memTileFixed)
    if (std::optional<TileID> at = positionOf(op))
      opsAt[*at].push_back(op);
  for (Operation *tile : logicalMemTiles)
    if (std::optional<TileID> at = positionOf(tile))
      opsAt[*at].push_back(tile);
  for (auto &[at, tiles] : physicalTiles)
    llvm::append_range(opsAt[at], tiles);
  for (auto &[_, ops] : opsAt) {
    llvm::sort(ops);
    ops.erase(llvm::unique(ops), ops.end());
  }
  SmallVector<TileID> homes =
      llvm::map_to_vector(positions, [](const Position &p) { return p.at; });
  SmallVector<unsigned> componentOf =
      memTileSpillComponents(*targetModel, homes);
  unsigned components =
      componentOf.empty() ? 0 : *llvm::max_element(componentOf) + 1;
  SmallVector<Walk> walk(positions.size(), nullptr);
  int64_t unplaced = 0;
  bool anyOverfull = false;
  for (unsigned component = 0; component < components; ++component) {
    SmallVector<unsigned> members;
    for (auto [i, c] : llvm::enumerate(componentOf))
      if (c == component)
        members.push_back(i);
    std::vector<uintptr_t> tilesKey;
    std::set<std::pair<int, int>> reach;
    for (unsigned member : members) {
      reach.insert({positions[member].at.col, positions[member].at.row});
      for (TileID n :
           memTileSpillNeighbours(*targetModel, positions[member].at))
        reach.insert({n.col, n.row});
    }
    for (auto [col, row] : reach) {
      tilesKey.push_back(col);
      tilesKey.push_back(row);
      std::vector<Operation *> ops = opsAt.lookup({col, row});
      tilesKey.push_back(ops.size());
      for (Operation *op : ops)
        tilesKey.push_back(reinterpret_cast<uintptr_t>(op));
    }
    for (const VariableLink &variable : variableLinks)
      tilesKey.push_back(static_cast<uintptr_t>(variable.depth));
    SmallVector<SmallVector<int64_t>> bytes;
    for (unsigned member : members)
      bytes.push_back(positions[member].sets.empty() ? SmallVector<int64_t>{0}
                                                     : positions[member].bytes);
    int64_t last = 0;
    auto places = [&](ArrayRef<unsigned> choice) {
      std::vector<uintptr_t> key = tilesKey;
      for (auto [member, set] : llvm::zip_equal(members, choice)) {
        key.push_back(set);
        walk[member] = positions[member].sets.empty()
                           ? nullptr
                           : &positions[member].sets[set];
      }
      if (const int64_t *known = unplacedOf.find(key))
        return (last = *known) == 0;
      auto shaped = shapes(members, walk);
      last = attempt(shaped, inOrder(shaped.size()));
      unplacedOf.insert(key, last);
      return last == 0;
    };
    // Walking changes no buffer's bytes in total, so when the component's
    // buffers outweigh its tiles' room, no choice of walk sets places them.
    int64_t needed = 0, room = 0;
    for (auto &[pool, _] : shapes(members, SmallVector<Walk>(walk.size())))
      needed += pool.bytes * pool.count;
    for (auto [col, row] : reach) {
      room += targetModel->getMemTileSize();
      for (const BufferAllocation &buffer :
           tiles[index.lookup({col, row})].fixed)
        room -= buffer.size;
    }
    DMAChannelAnalysis::WalkChoice choice;
    choice.sets.assign(members.size(), 0);
    anyOverfull |= needed > room;
    if (needed <= room)
      choice = DMAChannelAnalysis::searchWalkProduct(bytes, places, 1);
    if (!choice.placed)
      (void)places(choice.sets);
    for (auto [member, set] : llvm::zip_equal(members, choice.sets))
      walk[member] = positions[member].sets.empty()
                         ? nullptr
                         : &positions[member].sets[set];
    unplaced += last;
    if (charged && last)
      for (unsigned member : members)
        charged->push_back(positions[member].at);
  }
  // An overfull component fails the whole-device attempt too.
  if (unplaced && !anyWalked && !anyOverfull) {
    SmallVector<unsigned> all =
        llvm::to_vector(llvm::seq<unsigned>(0, positions.size()));
    auto split = shapes(all, SmallVector<Walk>(positions.size(), nullptr));
    SmallVector<SpillPool> records;
    for (auto &[pool, _] : split)
      records.push_back(pool);
    if (!attempt(split, demandFirstOrder(records)))
      unplaced = 0;
  }
  int penalty = ((unplaced + 1023) / 1024) * config.memPenaltyPerKB;
  if (!charged)
    spillPenaltyOf.insert(key, penalty);
  return penalty;
}

// In position order: room that one tile is lent is gone for the next, so the
// order decides the outcome and must not follow the map's hashing.
SmallVector<TileID> SAPlacer::overflowingCoreTiles() const {
  SmallVector<TileID> tiles;
  for (auto &[tilePos, unplaced] : coreUnplacedAt)
    if (unplaced > 0)
      tiles.push_back(tilePos);
  llvm::sort(tiles);
  return tiles;
}

void SAPlacer::chargeCoreFifo(TileID pos, int64_t bytes, int count) {
  if (count == 0 ||
      targetModel->getTileType(pos.col, pos.row) != AIETileType::CoreTile)
    return;
  std::map<int64_t, int> &held = coreFifoBuffers[pos];
  if ((held[bytes] += count) == 0)
    held.erase(bytes);
  coreFifoLog.push_back({pos, bytes, count});
  std::map<int64_t, int> &change = coreFifoChange[pos];
  if ((change[bytes] += count) == 0)
    change.erase(bytes);
}

int64_t SAPlacer::coreUnplaced(TileID pos, const std::map<int64_t, int> &added,
                               const std::map<int64_t, int> &removed) const {
  SmallVector<const CoreTileContents *, 2> here;
  int numBanks = targetModel->getNumBanks(pos.col, pos.row);
  // The tile ops and objectFIFO buffers decide the layout, so they key it.
  std::vector<int64_t> tileKey = {numBanks};
  auto addOp = [&](Operation *op) {
    if (auto it = coreContents.find(op); it != coreContents.end()) {
      here.push_back(&it->second);
      tileKey.push_back(reinterpret_cast<intptr_t>(op));
    }
  };
  if (auto physical = physicalTiles.find(pos); physical != physicalTiles.end())
    for (Operation *op : physical->second)
      addOp(op);
  if (Operation *op = physToLogical.lookup(pos))
    if (auto at = currentPlacement.find(op);
        at != currentPlacement.end() && at->second == pos)
      addOp(op);
  tileKey.push_back(-1);
  static const std::map<int64_t, int> none;
  auto held = coreFifoBuffers.find(pos);
  const std::map<int64_t, int> &base =
      held == coreFifoBuffers.end() ? none : held->second;
  std::map<int64_t, int> adjusted;
  if (!added.empty() || !removed.empty()) {
    adjusted = base;
    for (auto [bytes, count] : added)
      adjusted[bytes] += count;
    for (auto [bytes, count] : removed)
      adjusted[bytes] -= count;
  }
  const std::map<int64_t, int> &fifo =
      added.empty() && removed.empty() ? base : adjusted;
  for (auto [bytes, count] : fifo)
    if (count > 0)
      tileKey.insert(tileKey.end(), {bytes, count});
  if (auto cached = coreUnplacedByTile.find(tileKey);
      cached != coreUnplacedByTile.end())
    return cached->second;

  TileMemoryLimits limits =
      tileMemoryLimits(*targetModel, AIETileType::CoreTile);
  TileLayoutProblem problem;
  problem.numBanks = numBanks;
  problem.tileAlignBitWidth = limits.tileAlignBitWidth;
  problem.maxVecAlignBits = limits.maxVecAlignBits;
  problem.maxDataMemorySize = limits.maxDataMemorySize;
  problem.budget = config.placementBudget;
  for (auto [bytes, count] : fifo) {
    TileLayoutBuffer buffer;
    buffer.size = bytes;
    buffer.alignBytes = requiredBufferAlignBits(limits.tileAlignBitWidth,
                                                limits.maxVecAlignBits, bytes) /
                        8;
    for (int i = 0; i < count; ++i)
      problem.buffers.push_back(buffer);
  }
  for (const CoreTileContents *contents : here) {
    llvm::append_range(problem.buffers, contents->buffers);
    if (contents->hasCore) {
      problem.hasCore = true;
      problem.stackRun = contents->stackRun;
      problem.stackBank = contents->stackBank;
      problem.stackAlignBytes = contents->stackAlignBytes;
    }
  }

  // Different tiles holding the same buffers share a layout.
  std::vector<int64_t> key = {problem.numBanks,
                              problem.hasCore,
                              problem.stackRun.start,
                              problem.stackRun.size,
                              problem.stackBank.value_or(-1),
                              problem.stackAlignBytes};
  for (const TileLayoutBuffer &buffer : problem.buffers) {
    key.insert(key.end(),
               {buffer.size, buffer.alignBytes, buffer.sectionAlignBytes,
                buffer.aligned, buffer.address.value_or(-1),
                buffer.memBank.value_or(-1), buffer.initialized,
                static_cast<int64_t>(buffer.spaceBanks.size())});
    llvm::append_range(key, buffer.spaceBanks);
  }
  // Both bounded, since a run meets more layouts than a design has tiles.
  if (coreUnplacedByTile.size() > (1u << 18))
    coreUnplacedByTile.clear();
  if (auto cached = coreUnplacedOf.find(key); cached != coreUnplacedOf.end()) {
    coreUnplacedByTile[std::move(tileKey)] = cached->second;
    return cached->second;
  }
  if (coreUnplacedOf.size() > (1u << 16))
    coreUnplacedOf.clear();
  // More bytes than the tile holds fail any layout, so the excess is
  // charged without searching for the bytes the search would leave.
  int64_t bytes = problem.stackRun.size;
  for (const TileLayoutBuffer &buffer : problem.buffers)
    bytes += buffer.size;
  if (bytes > problem.maxDataMemorySize) {
    int64_t excess = bytes - problem.maxDataMemorySize;
    coreUnplacedOf[std::move(key)] = excess;
    coreUnplacedByTile[std::move(tileKey)] = excess;
    return excess;
  }
  TileLayoutStats stats;
  TileLayout layout = layoutTile(problem, stats);
  int64_t unplaced = 0;
  if (layout.status == TileLayout::Status::NoRoom) {
    unplaced = layout.unplacedBytes;
  } else if (layout.status != TileLayout::Status::Placed) {
    // A pin or stack the tile cannot honour fails addressing whatever is
    // placed with it.
    for (const TileLayoutBuffer &buffer : problem.buffers)
      unplaced += buffer.size;
    unplaced = std::max<int64_t>(unplaced, 1);
  }
  coreUnplacedOf[std::move(key)] = unplaced;
  coreUnplacedByTile[std::move(tileKey)] = unplaced;
  return unplaced;
}

SmallVector<TileID> SAPlacer::refreshCoreUnplaced() {
  // A fifo a move takes off a tile and puts back leaves it as it was.
  for (auto &[pos, change] : coreFifoChange)
    if (!change.empty())
      coreDirty.insert(pos);
  coreFifoChange.clear();
  SmallVector<TileID> changed;
  for (TileID pos : coreDirty) {
    if (targetModel->getTileType(pos.col, pos.row) != AIETileType::CoreTile)
      continue;
    coreUndo.push_back({pos, coreUnplacedAt.lookup(pos)});
    coreUnplacedAt[pos] = coreUnplaced(pos);
    changed.push_back(pos);
  }
  coreDirty.clear();
  return changed;
}

DenseSet<TileID> SAPlacer::coreOverflowReads() const {
  DenseSet<TileID> reads;
  for (TileID pos : coreOverflowTiles) {
    reads.insert(pos);
    for (TileID nbr : getMemAffinityNeighbors(*targetModel, pos))
      reads.insert(nbr);
  }
  return reads;
}

int64_t
SAPlacer::planCoreFit(TileID pos,
                      DenseMap<TileID, std::map<int64_t, int>> &lent,
                      SmallVectorImpl<std::pair<size_t, TileID>> *moves) const {
  // The intratile fifos an allocate can move to a neighbour, largest first,
  // then with the larger objects, then the one the core acquires first.
  SmallVector<std::pair<size_t, int64_t>, 8> relocatable;
  if (Operation *op = physToLogical.lookup(pos))
    if (auto fifoIt = tileToFifoIndices.find(op);
        fifoIt != tileToFifoIndices.end()) {
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
        if (prodIt == currentPlacement.end() ||
            consIt == currentPlacement.end() || prodIt->second != pos ||
            consIt->second != pos)
          continue;
        relocatable.push_back(
            {fi, std::max(fb.producerSizeBytes, fb.consumerSizeBytes) *
                     fb.sharedDepth});
      }
    }
  llvm::stable_sort(relocatable, [&](const auto &a, const auto &b) {
    const FifoBufferInfo &x = fifoBuffers[a.first];
    const FifoBufferInfo &y = fifoBuffers[b.first];
    return std::make_tuple(b.second,
                           std::max(y.producerRawBytes, y.consumerRawBytes),
                           firstAcquire.lookup_or(x.fifoOp, UINT_MAX),
                           collected.rank.lookup(x.fifoOp)) <
           std::make_tuple(a.second,
                           std::max(x.producerRawBytes, x.consumerRawBytes),
                           firstAcquire.lookup_or(y.fifoOp, UINT_MAX),
                           collected.rank.lookup(y.fifoOp));
  });

  // Neighbours in position order, so the plan reads only the tiles' contents.
  SmallVector<TileID, 4> neighbours(getMemAffinityNeighbors(*targetModel, pos));
  llvm::sort(neighbours);

  std::map<int64_t, int> removed;
  for (auto [fi, size] : relocatable) {
    const auto &fb = fifoBuffers[fi];
    int64_t bytes = std::max(fb.producerRawBytes, fb.consumerRawBytes);
    for (TileID nbr : neighbours) {
      std::map<int64_t, int> added = lent.lookup(nbr);
      added[bytes] += fb.sharedDepth;
      if (coreUnplaced(nbr, added) != 0)
        continue;
      lent[nbr] = std::move(added);
      removed[bytes] += fb.sharedDepth;
      if (moves)
        moves->push_back({fi, nbr});
      break;
    }
  }
  return coreUnplaced(pos, lent.lookup(pos), removed);
}

int SAPlacer::computeCoreOverflowPenalty(
    SmallVectorImpl<TileID> *charged) const {
  int penalty = 0;
  // Room a neighbour gives one overflowing tile is gone for the next, as
  // generateAllocates hands it out.
  DenseMap<TileID, std::map<int64_t, int>> lent;
  for (TileID tilePos : overflowingCoreTiles()) {
    int64_t unresolved = planCoreFit(tilePos, lent, nullptr);
    if (unresolved <= 0)
      continue;
    penalty += ((unresolved + 1023) / 1024) * config.memPenaltyPerKB;
    if (charged)
      charged->push_back(tilePos);
  }
  return penalty;
}

SmallVector<std::pair<int, SmallVector<std::pair<TileID, bool>, 2>>>
SAPlacer::lockGroups() const {
  SmallVector<std::pair<int, SmallVector<std::pair<TileID, bool>, 2>>> groups;
  auto add = [&](const SmallVector<std::pair<TileID, bool>, 2> &users,
                 int segments, int64_t depth) {
    for (int count : LockPlan::groupCounts(*targetModel, depth, segments))
      groups.push_back({count, users});
  };
  auto isCore = [](std::optional<AIETileType> type) {
    return type == AIETileType::CoreTile;
  };
  // A pool on an end's tile serves that end's core, if it uses the fifo, and
  // the tile's DMA.
  auto endUsers = [&](TileID at, bool core, std::optional<AIETileType> type) {
    SmallVector<std::pair<TileID, bool>, 2> users;
    if (core)
      users.push_back({at, false});
    users.push_back({at, isCore(type)});
    return users;
  };
  using LinkPool = FifoBufferInfo::LinkPool;
  for (size_t index = 0; index < fifoBuffers.size(); ++index) {
    const FifoBufferInfo &fb = fifoBuffers[index];
    std::optional<TileID> prodPos = positionOf(fb.producerTile);
    if (!prodPos)
      continue;
    auto heldLocks = [&](int64_t depth) {
      return PoolLocks::needsLocks(fb.unsynchronized, fb.initObjects, depth,
                                   fb.producerCore, fb.rearmed);
    };
    const FifoEndTypes &types = fifoEndTypes[index];
    std::optional<TileID> delegatePos = positionOf(fb.delegate);
    bool prodLinked = fb.producerLinkPool != LinkPool::None;
    if (const std::optional<PlannedDmaEnd> &plan = fb.producerLinkPlan)
      add(endUsers(*prodPos, fb.producerCore, types.producer),
          fb.producerLinkSegments, plan->depth);
    bool prodPool = false;
    for (size_t ci = 0; ci < fb.consumerTiles.size(); ++ci) {
      std::optional<TileID> consPos = positionOf(fb.consumerTiles[ci]);
      if (!consPos)
        continue;
      bool consLinked = fb.consumerLinkPools[ci] != LinkPool::None;
      if (const std::optional<PlannedDmaEnd> &plan = fb.consumerLinkPlans[ci])
        add(endUsers(*consPos, fb.consumerCore[ci], types.consumers[ci]),
            fb.consumerLinkSegments[ci], plan->depth);
      using SharedMemory = AIETargetModel::SharedMemory;
      if (!fb.forcesDMA && ObjectFifoCreateOp::sharedMemoryOf(
                               *targetModel, *prodPos, *consPos, delegatePos,
                               positionOf(fb.linkTile)) != SharedMemory::None) {
        if (prodLinked || consLinked || ci != 0 || !heldLocks(fb.sharedDepth))
          continue;
        SmallVector<std::pair<TileID, bool>, 2> users;
        if (fb.producerCore)
          users.push_back({*prodPos, false});
        if (fb.consumerCore[ci])
          users.push_back({*consPos, false});
        add(users, 1, fb.sharedDepth);
        continue;
      }
      if (!prodPool && !prodLinked && !isShimEndType(types.producer) &&
          !fb.linkSharedProd && fb.producerEnd) {
        prodPool = true;
        if (heldLocks(fb.producerDepth))
          add(endUsers(*prodPos, fb.producerCore, types.producer), 1,
              fb.producerDepth);
      }
      if (!consLinked && !isShimEndType(types.consumers[ci]) &&
          fb.consumerEnds[ci] && !fb.unsynchronized)
        add(endUsers(*consPos, fb.consumerCore[ci], types.consumers[ci]), 1,
            fb.consumerDepths[ci]);
    }
  }
  for (const InputLockGroup &group : inputLockGroups) {
    SmallVector<std::pair<TileID, bool>, 2> users;
    bool placed = true;
    for (auto [tile, localOnly] : group.users) {
      std::optional<TileID> at = positionOf(tile);
      placed &= at.has_value();
      if (at)
        users.push_back({*at, localOnly});
    }
    if (placed)
      groups.push_back({group.locks, users});
  }
  return groups;
}

int SAPlacer::computeLockPenalty(bool *stopped,
                                 SmallVectorImpl<TileID> *charged) const {
  if (stopped)
    *stopped = false;
  // Each group taking its locks at home always fits there.
  auto fitsAtHome = [&](const DenseMap<TileID, int> &atHome) {
    return llvm::all_of(atHome, [&](const auto &entry) {
      return entry.second <= static_cast<int>(targetModel->getNumLocks(
                                 entry.first.col, entry.first.row));
    });
  };
  if (fixedLocks.empty() && placedLocks.empty() && inputLockGroups.empty()
          ? fitsAtHome(currentLockDemand)
          : [&] {
              DenseMap<TileID, int> atHome = currentLockDemand;
              for (auto &[tile, locks] : fixedLocks)
                if (std::optional<TileID> at = positionOf(tile))
                  atHome[*at] += locks;
              for (auto &[at, locks] : placedLocks)
                atHome[at] += locks;
              for (const InputLockGroup &group : inputLockGroups)
                if (std::optional<TileID> at = positionOf(group.home))
                  atHome[*at] += group.locks;
              return fitsAtHome(atHome);
            }())
    return 0;

  auto groups = lockGroups();
  SmallVector<TileID> positions;
  for (auto &[op, at] : currentPlacement)
    positions.push_back(at);
  llvm::append_range(positions, placedTiles);
  llvm::sort(positions, [](TileID a, TileID b) {
    return std::make_pair(a.col, a.row) < std::make_pair(b.col, b.row);
  });
  positions.erase(llvm::unique(positions), positions.end());

  // The verdict reads the groups, the locks logical tiles hold and the
  // occupied positions a group may take: those its first user reaches, or
  // every one for a group without users. Placed tiles' locks never move.
  // Recording stops after a long run of misses, where moves seldom revisit.
  bool recording = !charged && lockVerdictMissRun < kLockVerdictMissRun;
  std::vector<int64_t> key;
  auto known = lockVerdicts.end();
  if (recording) {
    bool unanchored = false;
    int rows = targetModel->rows();
    BitVector reached(targetModel->columns() * rows);
    BitVector anchored(reached.size());
    SmallVector<TileID> anchors;
    for (auto &[locks, users] : groups) {
      llvm::append_values(key, locks, static_cast<int64_t>(users.size()));
      for (auto [at, localOnly] : users)
        llvm::append_values(key, at.col, at.row,
                            static_cast<int64_t>(localOnly));
      if (users.empty())
        unanchored = true;
      else if (auto [first, localOnly] = users.front(); localOnly)
        reached.set(first.col * rows + first.row);
      else if (!anchored.test(first.col * rows + first.row)) {
        anchored.set(first.col * rows + first.row);
        anchors.push_back(first);
      }
    }
    for (TileID anchor : anchors)
      for (TileID at : lockReach.find(anchor)->second)
        reached.set(at.col * rows + at.row);
    key.push_back(-1);
    for (auto &[tile, locks] : fixedLocks) {
      std::optional<TileID> at = positionOf(tile);
      llvm::append_values(key, at ? at->col : -1, at ? at->row : -1, locks);
    }
    llvm::append_values(key, -1, static_cast<int64_t>(positions.empty()));
    for (TileID at : positions)
      if (unanchored || reached.test(at.col * rows + at.row))
        llvm::append_values(key, at.col, at.row);
    known = lockVerdicts.find(key);
    if (known != lockVerdicts.end()) {
      lockVerdictMissRun = 0;
#ifndef EXPENSIVE_CHECKS
      if (stopped)
        *stopped = known->second.second;
      return known->second.first;
#endif
    }
  }

  LockPlan plan(*targetModel);
  DenseMap<TileID, unsigned> index;
  auto tileAt = [&](TileID at) {
    auto [entry, fresh] = index.try_emplace(at, 0);
    if (fresh)
      entry->second = plan.addTile({{at}, at});
    return entry->second;
  };
  int penalty = 0;
  for (auto &[tile, locks] : fixedLocks)
    if (std::optional<TileID> at = positionOf(tile))
      if (!plan.reserve(tileAt(*at), locks)) {
        penalty += locks * config.dmaPenaltyPerChannel;
        if (charged)
          charged->push_back(*at);
      }
  for (auto &[at, locks] : placedLocks)
    if (!plan.reserve(tileAt(at), locks)) {
      penalty += locks * config.dmaPenaltyPerChannel;
      if (charged)
        charged->push_back(at);
    }
  SmallVector<unsigned> ranked;
  for (TileID at : positions)
    ranked.push_back(tileAt(at));
  if (ranked.empty())
    return penalty;
  // A group may take the occupied positions all its users reach, so those
  // come from the first user's reach, not from every position.
  SmallVector<LockPlan::Group> planned;
  for (auto &[locks, users] : groups) {
    SmallVector<LockPlan::User> records;
    for (auto [at, localOnly] : users)
      records.push_back({tileAt(at), localOnly});
    auto reachesAt = [&](std::pair<TileID, bool> user, TileID at) {
      return user.first == at ||
             (!user.second &&
              llvm::is_contained(lockReach.find(user.first)->second, at));
    };
    SmallVector<unsigned> eligible;
    if (users.empty()) {
      eligible = ranked;
    } else {
      auto [first, localOnly] = users.front();
      for (TileID at : localOnly ? ArrayRef<TileID>(first)
                                 : ArrayRef(lockReach.find(first)->second))
        if (index.contains(at) &&
            llvm::all_of(users, [&](auto user) { return reachesAt(user, at); }))
          eligible.push_back(index.lookup(at));
    }
#ifdef EXPENSIVE_CHECKS
    for (unsigned tile : ranked)
      assert(llvm::is_contained(eligible, tile) ==
                 llvm::all_of(records,
                              [&](const LockPlan::User &user) {
                                return plan.reaches(user, tile);
                              }) &&
             "a lock group's tiles differ from those every user reaches");
#endif
    planned.push_back({locks, std::move(eligible), std::move(records)});
  }
  // Allocation places in its own order first, which the SA cannot know; the
  // canonical order alone fits only where allocation fits too.
  LockPlan::Assignment assignment =
      plan.assign(planned, &lockMemo, /*callerOrder=*/false);
  if (stopped)
    *stopped = assignment.stopped;
  for (auto [group, tile, grouped] :
       llvm::zip(planned, assignment.tiles, groups)) {
    if (tile)
      continue;
    penalty += group.count * config.dmaPenaltyPerChannel;
    if (charged)
      for (auto [at, localOnly] : grouped.second)
        charged->push_back(at);
  }
#ifdef EXPENSIVE_CHECKS
  assert((known == lockVerdicts.end() ||
          known->second == std::pair(penalty, assignment.stopped)) &&
         "a recorded lock verdict differs from a fresh one");
#endif
  if (recording && known == lockVerdicts.end()) {
    if (++lockVerdictMissRun == kLockVerdictMissRun) {
      decltype(lockVerdicts)().swap(lockVerdicts);
      lockVerdictBytes = 0;
      return penalty;
    }
    // Bounded by bytes, since moves reach more lock states than they revisit.
    key.shrink_to_fit();
    size_t bytes = key.capacity() * sizeof(int64_t);
    if (lockVerdictBytes + bytes > kLockVerdictBytes) {
      lockVerdicts.clear();
      lockVerdictBytes = 0;
    }
    lockVerdictBytes += bytes;
    lockVerdicts.emplace(std::move(key),
                         std::pair(penalty, assignment.stopped));
  }
  return penalty;
}

void SAPlacer::replanLinks(ArrayRef<std::pair<unsigned, int64_t>> depths) {
  auto at = [this](Value tile) { return positionOf(tile.getDefiningOp()); };
  SmallVector<unsigned> changed;
  for (auto [index, depth] : depths) {
    VariableLink &variable = variableLinks[index];
    if (depth == variable.depth)
      continue;
    variable.depth = depth;
    changed.push_back(index);
  }
  if (changed.empty())
    return;
  // A link whose tiles stayed keeps its depth, so these are every link's.
  std::vector<int64_t> all;
  for (const VariableLink &variable : variableLinks)
    all.push_back(variable.depth);
  for (unsigned index : changed) {
    VariableLink &variable = variableLinks[index];
    auto [replan, added] = variable.replans.try_emplace(all);
    if (added) {
      // The pool's plan is the first end's that names it, in fifo order, as
      // buildFifoBufferInfo took it.
      for (size_t fifoIndex : variable.fifos) {
        FifoBufferInfo &fb = fifoBuffers[fifoIndex];
        auto fifo = cast<ObjectFifoCreateOp>(fb.fifoOp);
        auto &ends = replan->second.ends.emplace_back();
        for (unsigned ci = 0, e = fb.consumerEnds.size(); ci < e; ++ci)
          ends.push_back(fifo.plannedDmaEnd(ci, at));
        ends.push_back(fifo.plannedDmaEnd(std::nullopt, at));
        for (const std::optional<PlannedDmaEnd> &end : ends)
          if (!replan->second.held && end &&
              end->pool == variable.link.getOperation())
            replan->second.held = end;
      }
    }
    for (auto [fifoIndex, ends] :
         llvm::zip_equal(variable.fifos, replan->second.ends)) {
      FifoBufferInfo &fb = fifoBuffers[fifoIndex];
      auto install = [&](std::optional<PlannedDmaEnd> &end,
                         const std::optional<PlannedDmaEnd> &plan,
                         std::optional<unsigned> entry, Operation *tile) {
        end = plan;
        if (entry)
          memTileEnds.find(tile)->second[*entry].end = *end;
      };
      for (unsigned ci = 0, e = fb.consumerEnds.size(); ci < e; ++ci)
        install(fb.consumerEnds[ci], ends[ci], fb.consumerMemTileEntries[ci],
                fb.consumerTiles[ci]);
      install(fb.producerEnd, ends.back(), fb.producerMemTileEntry,
              fb.producerTile);
    }
    for (auto [fifoIndex, consumer] : variable.holds) {
      FifoBufferInfo &fb = fifoBuffers[fifoIndex];
      (consumer ? fb.consumerLinkPlans[*consumer] : fb.producerLinkPlan) =
          replan->second.held;
    }
  }
}

int SAPlacer::computeTileDMAPenalty(TileID tilePos, int *bdIds) const {
  auto usage = currentDMAUsage.lookup(tilePos);
  auto [maxIn, maxOut] = getDMACapacity(*targetModel, tilePos);
  SmallVector<Operation *> tilesHere;
  if (overlayChannels(tilePos))
    for (Operation *tile : overlayTiles)
      if (positionOf(tile) == tilePos)
        tilesHere.push_back(tile);
  OverlayFit overlay = overlayAt(tilePos, pinUse, {}, tilesHere,
                                 shimSpans ? &*shimSpans : nullptr);
  int used[2] = {usage.first, usage.second + overlay.channels};
  int max[2] = {maxIn, maxOut};
  // Two ends that cannot share a channel they both pin clash here wherever
  // the rest goes, and so does an overlay with nowhere to sit.
  int penalty =
      (pinUse.clashes(tilePos) + !overlay.fits) * config.dmaPenaltyPerChannel;
  for (int dir : {0, 1}) {
    if (used[dir] <= max[dir])
      continue;
    if (shimSpans && targetModel->getTileType(tilePos.col, tilePos.row) ==
                         AIETileType::ShimNOCTile) {
      SmallVector<const ShimSide *> sides;
      for (auto &[lt, own] : shareableShimEnds) {
        if (positionOf(lt) == tilePos)
          sides.push_back(&own[dir]);
      }
      SmallVector<StringAttr> extra;
      if (dir == 1 && overlay.own)
        extra.push_back(shimSpans->overlayEnd(tilePos.col));
      used[dir] -= shimChannelsSaved(*shimSpans, sides, extra,
                                     pinUse.channelsAt(tilePos, dir == 1));
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
    int ids =
        runtimeBdIds.shortfall(onTile, tilePos) * config.dmaPenaltyPerChannel;
    penalty += ids;
    if (bdIds)
      *bdIds = ids;
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
    if (positionOf(tile) == position)
      tiles.push_back(tile);
  return tiles;
}

int64_t SAPlacer::bdShortfall(TileID position, ArrayRef<Operation *> tiles,
                              WalkSets *walkSets,
                              SmallVector<int64_t> *walkBytes) {
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
  // Bounded, since tiles meet at positions in more ways than a design has
  // positions. Every tile at the position holds what its programs take; a tile
  // with fixed coordinates is only ever here, so the list keys the position
  // too.
  SmallVector<Operation *> holders(tiles);
  for (Operation *tile : logicalMemTiles)
    if (auto at = currentPlacement.find(tile);
        at != currentPlacement.end() && at->second == position &&
        !llvm::is_contained(holders, tile))
      holders.push_back(tile);
  for (Operation *tile : physicalTiles.lookup(position))
    if (!llvm::is_contained(holders, tile))
      holders.push_back(tile);
  std::vector<int64_t> depths;
  for (const VariableLink &variable : variableLinks)
    depths.push_back(variable.depth);
  std::tuple<TileID, std::vector<Operation *>, std::vector<int64_t>> key{
      layout->second, std::vector<Operation *>(holders.begin(), holders.end()),
      std::move(depths)};
  if (const BDShortfall *cached = bdShortfallOf.find(key)) {
    if (walkSets)
      *walkSets = cached->sets;
    if (walkBytes)
      *walkBytes = cached->bytes;
    return cached->ids;
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
  SmallVector<DMAChannelAnalysis::BDPlanEnd> planned;
  SmallVector<unsigned> poolOf;
  SmallVector<DMAChannelAnalysis::BDWalkPool> pools;
  SmallVector<std::pair<Operation *, int>> poolKeys;
  for (const MemTileEnd *end : ends) {
    std::pair<Operation *, int> poolKey{end->end.pool, end->end.id};
    auto *at = llvm::find(poolKeys, poolKey);
    if (at == poolKeys.end()) {
      poolKeys.push_back(poolKey);
      pools.push_back({end->end.depth, end->end.objectBytes, false});
      at = std::prev(poolKeys.end());
    }
    unsigned pool = at - poolKeys.begin();
    pools[pool].walkable |= end->end.walkable;
    poolOf.push_back(pool);
    planned.push_back({end->dir, end->end.bds,
                       channels[end->dir == DMAChannelDir::MM2S],
                       end->end.channel});
  }
  DMAChannelAnalysis::BDWalkSets found = DMAChannelAnalysis::walkSetsThatFit(
      *targetModel, col, row, planned, poolOf, pools, free, spent);
  WalkSets sets;
  for (ArrayRef<unsigned> set : found.sets)
    sets.push_back(llvm::map_to_vector(
        set, [&](unsigned pool) { return poolKeys[pool]; }));
  SmallVector<int64_t> bytes = found.bytes;
  int64_t shortfall = 0;
  if (found.channelShort || !found.fits) {
    // No set fits: every walkable pool is walked, and the ids still short
    // are charged; too few channels is the DMA term's to charge.
    if (!found.channelShort) {
      sets.emplace_back();
      bytes.push_back(0);
      for (auto [pool, walkable] : llvm::enumerate(pools))
        if (walkable.walkable) {
          sets.back().push_back(poolKeys[pool]);
          bytes.back() += walkable.depth * walkable.objectBytes;
        }
    }
    DenseSet<std::pair<Operation *, int>> whole;
    if (!sets.empty())
      whole.insert(sets.front().begin(), sets.front().end());
    int64_t total = 0;
    for (auto [end, pool] : llvm::zip_equal(planned, poolOf))
      total += whole.contains(poolKeys[pool]) ? 1 : end.bds;
    for (auto [channel, held] : taken)
      total += held;
    int64_t ids = targetModel->getNumBDs(col, row);
    shortfall = found.channelShort ? std::max<int64_t>(0, total - ids)
                                   : std::max<int64_t>(1, total - ids);
  }
  bdShortfallOf.insert(key, {shortfall, sets, bytes});
  if (walkSets)
    *walkSets = std::move(sets);
  if (walkBytes)
    *walkBytes = std::move(bytes);
  return shortfall;
}

// Penalizes mem tile positions whose ends' BD ids no channel plan fits.
int SAPlacer::computeBDCountPenalty() {
  bdShortfallAt.clear();
  int64_t penalty = 0;
  for (auto &[tile, ends] : memTileEnds) {
    std::optional<TileID> at = positionOf(tile);
    if (!at || bdShortfallAt.count(*at))
      continue;
    int64_t shortfall = bdShortfall(*at, memTilesAt(*at));
    bdShortfallAt[*at] = shortfall;
    penalty += shortfall * config.dmaPenaltyPerChannel;
  }
  return static_cast<int>(penalty);
}

int SAPlacer::computeAdjacencyPenalty(
    const Adjacency &adj, ArrayRef<std::pair<int, int>> validOffsets,
    int weight, SmallVectorImpl<TileID> *charged) const {
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
    if (minDist == 0)
      continue;
    penalty += weight * minDist;
    if (charged)
      charged->append({firstPos, secondPos});
  }
  return penalty;
}

int SAPlacer::computePairingPenalty(SmallVectorImpl<TileID> *charged) const {
  int penalty =
      computeAdjacencyPenalty(cascadeAdjacency, targetModel->getCascadeSteps(),
                              config.cascadeWeightPerDist, charged);
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
    if (charged)
      charged->append({a, b});
  }
  return penalty;
}

void SAPlacer::generateAllocates() {
  DenseSet<size_t> emittedFifoIndices;

  // Case A: the moves that planCoreFit plans.
  DenseMap<TileID, std::map<int64_t, int>> lent;
  for (TileID tilePos : overflowingCoreTiles()) {
    SmallVector<std::pair<size_t, TileID>> moves;
    (void)planCoreFit(tilePos, lent, &moves);
    for (auto [fifoIdx, nbrPos] : moves) {
      if (!emittedFifoIndices.insert(fifoIdx).second)
        continue;
      const auto &fb = fifoBuffers[fifoIdx];
      currentMemUsage[nbrPos] +=
          std::max(fb.producerSizeBytes, fb.consumerSizeBytes) * fb.sharedDepth;
      allocates.push_back({fb.fifoOp, nbrPos});
    }
  }

  // Case B: Shared-mem destination communication.
  // sharedMemDestination is populated by addFifoContribution during
  // initResourceTracking, which runs on the final placement before this.
  for (auto &[fifoIdx, destTile] : sharedMemDestination) {
    if (emittedFifoIndices.contains(fifoIdx))
      continue;
    const auto &fb = fifoBuffers[fifoIdx];
    std::optional<TileID> prodPos = positionOf(fb.producerTile);
    std::optional<TileID> consPos = positionOf(fb.consumerTiles[0]);
    if (!prodPos || !consPos || *prodPos == *consPos)
      continue;
    if (destTile != *prodPos)
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
  // A move that changes where a variable link's pool is shared changes its
  // depth: its participants come off with the old plans, on with the new.
  replannedLinks.clear();
  auto at = [this](Value tile) { return positionOf(tile.getDefiningOp()); };
  for (unsigned index = 0, e = variableLinks.size(); index < e; ++index) {
    const VariableLink &variable = variableLinks[index];
    if (llvm::none_of(oldPlacements, [&](const auto &moved) {
          return llvm::is_contained(variable.tiles, moved.first);
        }))
      continue;
    int64_t depth = variable.plan.depthAt(*targetModel, at);
    if (depth == variable.depth)
      continue;
    replannedLinks.push_back({index, variable.depth, depth});
    affectedFifos.insert(variable.fifos.begin(), variable.fifos.end());
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
  for (auto &[op, _] : oldPlacements) {
    for (size_t ei : tileToRouteEnds.lookup(op))
      addRouteEndContribution(ei, -1);
    for (const DMAChannelAnalysis::ChannelClaim &claim : claimsOf.lookup(op))
      pinUse.claim(currentPlacement[op], claim, -1);
  }

  // Restore new positions and add new contributions
  for (auto &[op, newPos] : savedPositions)
    currentPlacement[op] = newPos;
  replanLinks(llvm::map_to_vector(replannedLinks, [](ReplannedLink replanned) {
    return std::make_pair(replanned.index, replanned.to);
  }));
  for (size_t fi : affectedFifos)
    addFifoContribution(fi, +1);
  for (auto &[op, _] : oldPlacements) {
    for (size_t ei : tileToRouteEnds.lookup(op))
      addRouteEndContribution(ei, +1);
    for (const DMAChannelAnalysis::ChannelClaim &claim : claimsOf.lookup(op))
      pinUse.claim(currentPlacement[op], claim, 1);
  }

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
  coreUndo.clear();
  for (auto &[op, oldPos] : oldPlacements)
    if (coreContents.count(op)) {
      coreDirty.insert(oldPos);
      coreDirty.insert(currentPlacement[op]);
    }
  SmallVector<TileID> coreChanged = refreshCoreUnplaced();
  SmallVector<TileID> overflowing = overflowingCoreTiles();
  DenseSet<TileID> overflowReads = coreOverflowReads();
  if (overflowing != coreOverflowTiles ||
      llvm::any_of(coreChanged,
                   [&](TileID pos) { return overflowReads.contains(pos); })) {
    coreOverflowTiles = std::move(overflowing);
    coreOverflowPenalty = computeCoreOverflowPenalty();
  }

  auto reads = [&](const DenseSet<Operation *> &tiles) {
    return llvm::any_of(oldPlacements, [&](const auto &moved) {
      return tiles.contains(moved.first);
    });
  };
  if (reads(spillTiles) || !replannedLinks.empty())
    memSpilloverPenalty = computeMemSpilloverPenalty();
  bdUndo.clear();
  SmallVector<TileID> bdTouched;
  for (ReplannedLink replanned : replannedLinks)
    if (std::optional<TileID> pos =
            positionOf(variableLinks[replanned.index].tiles.front());
        pos &&
        targetModel->getTileType(pos->col, pos->row) == AIETileType::MemTile)
      bdTouched.push_back(*pos);
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

  cachedResourcePenalty = memSpilloverPenalty + coreOverflowPenalty +
                          computeLockPenalty() + dmaPenalty + bdCountPenalty +
                          delegatePenalty;
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

    // Shift onto an empty target, or onto an occupied one whose type several
    // tiles may share (a coin flip keeps swaps available too, since how many
    // sit on each could never change if sharing always won).
    if (!occupant ||
        (type != AIETileType::CoreTile && drawBoundedIndex(rng, 2) == 0)) {
      if (!isLegalPosition(tile, targetPos))
        continue;
      moves.clear();
      moves.push_back({tile, targetPos});
    } else {
      // Swap: both tiles must accept each other's position.
      if (tileTypes[occupant] != type)
        continue;
      if (!satisfiesConstraints(occupant, currentPos) ||
          namedClash(tile, namedAt(tile, targetPos, occupant)) ||
          namedClash(occupant, namedAt(occupant, currentPos, tile)))
        continue;
      moves.clear();
      moves.push_back({tile, targetPos});
      moves.push_back({occupant, currentPos});
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
  // Phase 6: Restore best, generate allocates, stats.
  return finalizePlacement(device);
}

LogicalResult SAPlacer::collectAndBuildModel(DeviceOp device) {
  collected = collectOperations(device);
  auto &logicalTiles = collected.logicalTiles;
  auto &objectFifos = collected.objectFifos;
  auto &objectFifoLinks = collected.objectFifoLinks;
  pinnedRule = PinnedChannels(*targetModel, packetSwitched);

  if (logicalTiles.empty())
    return success();

  // Cache tile types
  for (auto lt : logicalTiles)
    tileTypes[lt.getOperation()] = lt.getTileType();

  // Collect static buffer sizes per tile
  device.walk([&](BufferOp bufOp) {
    auto tile = dyn_cast_or_null<TileLike>(bufOp.getTile().getDefiningOp());
    if (!tile)
      return;
    int64_t bytes = bufOp.getAllocationSize();
    if (tile.getTileType() == AIETileType::MemTile)
      memTileFixed[tile].push_back(
          {bytes,
           bufOp.getAligned() ? targetModel->getMemTileLoadStoreBusWidth() / 8
                              : 1,
           bufOp.getAddress()});
    TileMemoryLimits limits =
        tileMemoryLimits(*targetModel, tile.getTileType());
    staticBufferSizes[tile] +=
        llvm::alignTo(bytes, bufferAlignBytes(bufOp, limits.tileAlignBitWidth,
                                              limits.maxVecAlignBits));
    if (tile.getTileType() == AIETileType::CoreTile)
      coreContents[tile].buffers.push_back(tileLayoutBuffer(bufOp, limits));
  });
  TileMemoryLimits coreLimits =
      tileMemoryLimits(*targetModel, AIETileType::CoreTile);
  // An unsplit pool becomes the buffers aie-objectfifo-allocate gives it.
  for (auto pool : device.getOps<ObjectFifoPoolOp>()) {
    auto tile = dyn_cast_or_null<TileLike>(pool.getTile().getDefiningOp());
    if (!tile || tile.getTileType() != AIETileType::CoreTile ||
        pool.getBuffers())
      continue;
    TileLayoutBuffer buffer;
    buffer.size = pool.getObjectSizeInBytes() * pool.getObjectsPerBuffer();
    buffer.alignBytes =
        requiredBufferAlignBits(coreLimits.tileAlignBitWidth,
                                coreLimits.maxVecAlignBits, buffer.size) /
        8;
    for (int i = 0, e = pool.getContiguous() ? 1 : pool.getDepth(); i < e; ++i)
      coreContents[tile].buffers.push_back(buffer);
  }

  for (LogicalTileOp tile : logicalTiles)
    for (Operation *user : tile->getUsers())
      if (isa<LockOp>(user))
        ++fixedLocks[tile];
  device.walk([&](LockOp lock) {
    if (auto placed = dyn_cast_or_null<TileOp>(lock.getTile().getDefiningOp()))
      ++placedLocks[placed.getTileID()];
  });
  for (auto tile : device.getOps<TileOp>())
    placedTiles.push_back(tile.getTileID());
  llvm::StringMap<Operation *> fifoNamed;
  for (ObjectFifoCreateOp fifo : collected.objectFifos)
    fifoNamed[fifo.getSymName()] = fifo;
  firstAcquire.clear();
  device.walk([&](ObjectFifoAcquireOp acquire) {
    if (Operation *fifo = fifoNamed.lookup(acquire.getObjFifoName()))
      firstAcquire.try_emplace(fifo, firstAcquire.size());
  });
  lockReach.clear();
  for (int col = 0; col < targetModel->columns(); ++col)
    for (int row = 0; row < targetModel->rows(); ++row) {
      TileID user{col, row};
      SmallVector<TileID> &reach = lockReach[user];
      for (int c = 0; c < targetModel->columns(); ++c)
        for (int r = 0; r < targetModel->rows(); ++r)
          if (TileID at{c, r};
              at == user ||
              LockPlan::reaches(*targetModel, user, at, /*localOnly=*/false))
            reach.push_back(at);
    }
  {
    PoolLocks poolLocks(device);
    bool semaphoreLocks =
        targetModel->hasProperty(AIETargetModel::UsesSemaphoreLocks);
    for (auto pool : device.getOps<ObjectFifoPoolOp>()) {
      if (!poolLocks.needsLocks(pool))
        continue;
      SmallVector<Operation *> groups;
      if (!semaphoreLocks && !pool.getLocks())
        groups.push_back(pool);
      if (semaphoreLocks)
        for (auto segment : pool.getSegmentOps())
          if (!(segment.getProduceLock() && segment.getConsumeLock()))
            groups.push_back(segment);
      SmallVector<int> counts =
          LockPlan::groupCounts(*targetModel, pool.getDepth(), groups.size());
      for (auto [index, entry] : llvm::enumerate(groups)) {
        Operation *owner = entry;
        InputLockGroup &group = inputLockGroups.emplace_back();
        group.home = pool.getTile().getDefiningOp();
        group.locks = counts[index];
        auto addUser = [&](Operation *endpoint, auto selected, bool dma) {
          if (semaphoreLocks && !llvm::is_contained(selected, owner))
            return;
          auto *tile = endpoint->getOperand(0).getDefiningOp();
          group.users.push_back(
              {tile, dma && LockPlan::dmaLocalOnly(
                                cast<TileLike>(tile).getTileType())});
        };
        for (auto endpoint : device.getOps<ObjectFifoCoreEndpointOp>())
          if (endpoint.getPoolOp() == pool)
            addUser(endpoint, endpoint.getSelectedSegments(), false);
        for (auto endpoint : device.getOps<ObjectFifoDmaEndpointOp>())
          if (endpoint.getPoolOp() == pool)
            addUser(endpoint, endpoint.getSelectedSegments(), true);
      }
    }
  }

  // Collect core stack and data regions (reserved by assign-buffer-addresses)
  for (auto [ordinal, coreOp] : llvm::enumerate(device.getOps<CoreOp>())) {
    auto *tileOp = coreOp.getTile().getDefiningOp();
    if (!isa_and_nonnull<TileLike>(tileOp))
      continue;
    // What addressing reserves for the core's own sections is as much its
    // memory as the stack.
    int64_t bytes = alignedBufferBytes(*targetModel, AIETileType::CoreTile,
                                       coreOp.getEffectiveStackSize());
    CoreTileContents &contents = coreContents[tileOp];
    contents.hasCore = true;
    contents.stackRun = coreOp.getStackRun();
    std::optional<uint32_t> stackBank = coreOp.getStackBank();
    if (!coreOp.getStackAddress() && stackBank) {
      contents.stackBank = static_cast<int>(*stackBank);
      contents.stackAlignBytes = targetModel->getCoreStackAlignment();
    }
    SmallVector<CoreSectionReservation> reservations =
        coreOp.getSectionReservations();
    if (!coreOp.getDataSize() && !coreOp.getMeasuredDataSize())
      if (auto charge = config.coreDataCharges.find(ordinal);
          charge != config.coreDataCharges.end() && charge->second.bytes)
        reservations.push_back({CoreSectionReservation::Kind::DataRegion,
                                charge->second.bytes, 0,
                                charge->second.alignBytes});
    for (auto [held, holder] : llvm::zip_equal(
             reservations, coreOp.getSectionReservationBuffers(reservations)))
      if (!holder) {
        TileLayoutBuffer buffer = tileLayoutBuffer(held, coreLimits);
        bytes += llvm::alignTo(buffer.size, buffer.alignBytes);
        contents.buffers.push_back(buffer);
      }
    stackSizes[tileOp] = bytes;
  }

  // Build net model, fifo buffer info, and cascade groups
  buildNetModel(objectFifos, objectFifoLinks);
  buildFifoBufferInfo(device, objectFifos, objectFifoLinks);
  if (pinnedRule.failed())
    return failure();
  llvm::ArrayRef<DMAChannelAnalysis::ChannelClaim> claims =
      channelAnalysis.emplace(device).getChannelClaims();
  buildRouteModel(device, collected.flows, collected.pktFlows, collected.routes,
                  claims);
  claimsOf.clear();
  for (const DMAChannelAnalysis::ChannelClaim &claim : claims)
    claimsOf[claim.tile.getDefiningOp()].push_back(claim);
  shareableShimEnds =
      collectShareableShimEnds(device, objectFifos, collected.rank);
  collectOverlayUses(device);
  overlayTiles = overlayTileOps();
  shimSpans.emplace(
      device, [&](StringAttr symbol) { return shimEndOf(device, symbol); },
      provenShimGates(device), controlOverlay);
  runtimeBdIds = RuntimeBdIds(device, *targetModel);
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

  // A pool is known by its objectFIFO as the split names it: a link's by the
  // fifo that owns its pool.
  chargedPools.clear();
  for (auto &[tile, ends] : memTileEnds)
    for (const MemTileEnd &end : ends) {
      std::optional<StringRef> fifo;
      if (auto create = dyn_cast_or_null<ObjectFifoCreateOp>(end.end.pool))
        fifo = create.name().getValue();
      else if (auto link = dyn_cast_or_null<ObjectFifoLinkOp>(end.end.pool))
        fifo = link.getPoolOwner().name().getValue();
      else if (auto pool = dyn_cast_or_null<ObjectFifoPoolOp>(end.end.pool))
        fifo = pool.getFifoName();
      if (fifo && llvm::is_contained(config.walkedFifos, *fifo))
        chargedPools.insert(end.end.pool);
    }

  return success();
}

void SAPlacer::buildFifoBufferInfo(DeviceOp device,
                                   ArrayRef<ObjectFifoCreateOp> objectFifos,
                                   ArrayRef<ObjectFifoLinkOp> objectFifoLinks) {
  mlir::DataLayout dataLayout(device->getParentOfType<ModuleOp>());

  memTileEnds.clear();
  PoolLocks poolLocks(device);
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
    int64_t producerBytes = getMemRefSizeInBytes(elemType, dataLayout);
    auto prodLike = dyn_cast_or_null<TileLike>(prodOp);
    AIETileType prodType =
        prodLike ? prodLike.getTileType() : AIETileType::CoreTile;
    fb.producerSizeBytes =
        alignedBufferBytes(*targetModel, prodType, producerBytes);
    fb.producerRawBytes = producerBytes;

    // Consumer element size (may differ with consumerElemType)
    int64_t consumerBytes = producerBytes;
    if (auto consType = ofOp.getConsumerElemType()) {
      auto consOFType = llvm::cast<AIEObjectFifoType>(consType.value());
      auto consMemref = llvm::cast<MemRefType>(consOFType.getElementType());
      consumerBytes = getMemRefSizeInBytes(consMemref, dataLayout);
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
    fb.consumerRawBytes = consumerBytes;

    fb.sharedDepth = ofOp.sharedDepth();
    fb.producerDepth = ofOp.dmaDepth(std::nullopt);
    for (unsigned ci = 0, e = fb.consumers.size(); ci < e; ++ci)
      fb.consumerDepths.push_back(ofOp.dmaDepth(ci));

    auto at = [this](Value tile) { return positionOf(tile.getDefiningOp()); };
    fb.producerEnd = ofOp.plannedDmaEnd(std::nullopt, at);
    for (unsigned ci = 0, e = fb.consumers.size(); ci < e; ++ci)
      fb.consumerEnds.push_back(ofOp.plannedDmaEnd(ci, at));
    auto onMemTile = [&](Operation *tile) {
      auto like = dyn_cast_or_null<TileLike>(tile);
      return like && like.getTileType() == AIETileType::MemTile;
    };
    auto drawsChannel = [&](Operation *tile, bool planned) {
      auto like = dyn_cast_or_null<TileLike>(tile);
      if (!like)
        return false;
      return isShimEndType(like.getTileType()) ? !ofOp.getPlio() : planned;
    };
    if (drawsChannel(fb.producerTile, fb.producerEnd.has_value()))
      fb.producerPin = PinnedChannels::endOf(ofOp, std::nullopt);
    if (fb.producerPin)
      pinnedRule.share(*fb.producerPin, collected.rank);
    for (unsigned ci = 0, e = fb.consumers.size(); ci < e; ++ci) {
      std::optional<PinnedChannels::End> pin;
      if (drawsChannel(fb.consumerTiles[ci], fb.consumerEnds[ci].has_value()))
        pin = PinnedChannels::endOf(ofOp, ci);
      if (pin)
        pinnedRule.share(*pin, collected.rank);
      fb.consumerPins.push_back(pin);
    }
    if (fb.producerEnd && onMemTile(fb.producerTile)) {
      SmallVector<MemTileEnd> &ends = memTileEnds[fb.producerTile];
      fb.producerMemTileEntry = ends.size();
      ends.push_back({DMAChannelDir::MM2S, *fb.producerEnd});
    }
    for (unsigned ci = 0, e = fb.consumers.size(); ci < e; ++ci) {
      std::optional<unsigned> &entry = fb.consumerMemTileEntries.emplace_back();
      if (const std::optional<PlannedDmaEnd> &end = fb.consumerEnds[ci];
          end && onMemTile(fb.consumerTiles[ci])) {
        SmallVector<MemTileEnd> &ends = memTileEnds[fb.consumerTiles[ci]];
        entry = ends.size();
        ends.push_back({DMAChannelDir::S2MM, *end});
      }
    }

    std::optional<ObjectFifoLinkOp> link = ofOp.getLink();
    fb.forcesDMA = ofOp.dmaReason().has_value();
    if (link)
      fb.linkTile = link->getOptionalSharedTile()->getDefiningOp();
    if (fb.delegate) {
      fb.allocateFallsBackToDMA = ofOp.allocateFallsBackToDMA();
      fb.fitsLinkPool = ofOp.linkPoolMisfit() == LinkPoolMisfit::None;
    }
    // A link output's producer objects are the link input's consumer pool.
    fb.linkSharedProd =
        link && llvm::is_contained(link->getOutputObjectFifos(), ofOp);
    fb.unsynchronized = ofOp.getDisableSynchronization();
    if (auto init = ofOp.getInitValues())
      fb.initObjects = init->size();
    fb.rearmed = poolLocks.rearms().contains(ofOp.getSymName());
    fb.producerCore = hasCoreAccess(device, ofOp.getProducerTile(), ofOp,
                                    ObjectFifoPort::Produce);
    for (Value tile : ofOp.getConsumerTiles())
      fb.consumerCore.push_back(
          hasCoreAccess(device, tile, ofOp, ObjectFifoPort::Consume));
    fifoBuffers.push_back(fb);
  }

  // Every end of a link on its tile shares the link's one pool, as the split
  // builds it once; the first end holds it. A merge's or dispatch's later
  // participants plan no DMA end of their own, so the pool's plan is the
  // first end's that names it.
  DenseMap<Operation *, PlannedDmaEnd> linkPlans;
  for (FifoBufferInfo &fb : fifoBuffers)
    for (const std::optional<PlannedDmaEnd> &end :
         llvm::concat<const std::optional<PlannedDmaEnd>>(
             fb.consumerEnds, ArrayRef(fb.producerEnd)))
      if (end && isa_and_nonnull<ObjectFifoLinkOp>(end->pool))
        linkPlans.try_emplace(end->pool, *end);
  DenseSet<Operation *> heldLinkPools;
  for (FifoBufferInfo &fb : fifoBuffers) {
    auto fifo = cast<ObjectFifoCreateOp>(fb.fifoOp);
    auto linkPoolOf = [&](std::optional<unsigned> consumer)
        -> std::pair<FifoBufferInfo::LinkPool, std::optional<PlannedDmaEnd>> {
      using LinkPool = FifoBufferInfo::LinkPool;
      ObjectFifoLinkOp link =
          consumer ? fifo.getConsumerLink(*consumer) : fifo.getProducerLink();
      auto plan = link ? linkPlans.find(link) : linkPlans.end();
      if (plan == linkPlans.end())
        return {LinkPool::None, std::nullopt};
      if (!heldLinkPools.insert(link).second)
        return {LinkPool::Shares, std::nullopt};
      return {LinkPool::Holds, plan->second};
    };
    for (unsigned ci = 0, e = fb.consumerEnds.size(); ci < e; ++ci) {
      auto [pool, plan] = linkPoolOf(ci);
      fb.consumerLinkPools.push_back(pool);
      fb.consumerLinkPlans.push_back(plan);
    }
    std::tie(fb.producerLinkPool, fb.producerLinkPlan) =
        linkPoolOf(std::nullopt);
    // A link's pool takes locks unless a participant disables
    // synchronization.
    auto linkSegments = [](const std::optional<PlannedDmaEnd> &plan) {
      if (!plan)
        return 0;
      auto link = cast<ObjectFifoLinkOp>(plan->pool);
      auto unsynchronized = [](ObjectFifoCreateOp participant) {
        return participant.getDisableSynchronization();
      };
      if (llvm::any_of(link.getInputObjectFifos(), unsynchronized) ||
          llvm::any_of(link.getOutputObjectFifos(), unsynchronized))
        return 0;
      return static_cast<int>(link.getPoolSegments().size());
    };
    for (const std::optional<PlannedDmaEnd> &plan : fb.consumerLinkPlans)
      fb.consumerLinkSegments.push_back(linkSegments(plan));
    fb.producerLinkSegments = linkSegments(fb.producerLinkPlan);
  }

  variableLinks.clear();
  DenseMap<Operation *, size_t> fifoIndex;
  for (auto [index, fb] : llvm::enumerate(fifoBuffers))
    fifoIndex[fb.fifoOp] = index;
  auto at = [this](Value tile) { return positionOf(tile.getDefiningOp()); };
  for (ObjectFifoLinkOp link : objectFifoLinks) {
    LinkPoolPlan plan = link.getPoolPlan();
    if (plan.sharers.empty())
      continue;
    Operation *point = plan.linkTile.getDefiningOp();
    VariableLink variable{link, std::move(plan), 0, {point}, {}, {}, {}};
    for (const LinkPoolSharer &sharer : variable.plan.sharers)
      variable.tiles.append({sharer.producer.getDefiningOp(),
                             sharer.consumer.getDefiningOp(),
                             sharer.delegate.getDefiningOp()});
    std::vector<ObjectFifoCreateOp> ins = link.getInputObjectFifos();
    std::vector<ObjectFifoCreateOp> outs = link.getOutputObjectFifos();
    for (ObjectFifoCreateOp fifo : llvm::concat<ObjectFifoCreateOp>(ins, outs))
      if (auto index = fifoIndex.find(fifo.getOperation());
          index != fifoIndex.end())
        variable.fifos.push_back(index->second);
    llvm::sort(variable.fifos);
    variable.fifos.erase(llvm::unique(variable.fifos), variable.fifos.end());
    using LinkPool = FifoBufferInfo::LinkPool;
    for (size_t index : variable.fifos) {
      FifoBufferInfo &fb = fifoBuffers[index];
      auto fifo = cast<ObjectFifoCreateOp>(fb.fifoOp);
      for (unsigned ci = 0, e = fb.consumerLinkPools.size(); ci < e; ++ci)
        if (fb.consumerLinkPools[ci] == LinkPool::Holds &&
            fifo.getConsumerLink(ci) == link)
          variable.holds.push_back({index, ci});
      if (fb.producerLinkPool == LinkPool::Holds &&
          fifo.getProducerLink() == link)
        variable.holds.push_back({index, std::nullopt});
    }
    variable.depth = variable.plan.depthAt(*targetModel, at);
    variableLinks.push_back(std::move(variable));
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

    // Each takes the next legal position of its pool, round robin.
    auto placeIn = [&](ArrayRef<Operation *> ops, ArrayRef<TileID> pool,
                       size_t &next) -> LogicalResult {
      for (Operation *op : ops) {
        std::optional<TileID> pos, named;
        for (size_t i = 0; i < pool.size() && !pos; ++i) {
          TileID at = pool[(next + i) % pool.size()];
          if (isLegalPosition(op, at))
            pos = at;
          else if (!named && satisfiesConstraints(op, at) &&
                   namedClash(op, namedAt(op, at)))
            named = at;
        }
        if (!pos && named)
          if (auto clash = namedClash(op, namedAt(op, *named)))
            return emitNamedClash(op, *named, *clash);
        if (!pos)
          return cast<LogicalTileOp>(op).emitError(
              "no available physical tile for placement");
        next++;
        currentPlacement[op] = *pos;
        movableTiles.push_back(op);
      }
      return success();
    };
    if (failed(placeIn(sortedMemTiles, availMem, memIdx)) ||
        failed(placeIn(shimTiles, availShim, shimIdx)))
      return failure();
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
  PinnedChannelUse savedPins = pinUse;
  auto savedSharedMemDest = sharedMemDestination;
  coreFifoLog.clear();
  lockUndo.clear();
  int oldCoreOverflow = coreOverflowPenalty;
  SmallVector<TileID> oldCoreOverflowTiles = coreOverflowTiles;

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
    replanLinks(
        llvm::map_to_vector(replannedLinks, [](ReplannedLink replanned) {
          return std::make_pair(replanned.index, replanned.from);
        }));
    currentMemUsage = savedMemUsage;
    currentDMAUsage = savedDMAUsage;
    pinUse = savedPins;
    sharedMemDestination = savedSharedMemDest;
    for (auto &[pos, delta] : lockUndo)
      currentLockDemand[pos] -= delta;
    cachedResourcePenalty = oldResPenalty;
    memSpilloverPenalty = oldMemSpillover;
    bdCountPenalty = oldBDCount;
    delegatePenalty = oldDelegate;
    dmaPenalty = oldDMA;
    for (auto &[tile, at] : dmaUndo)
      dmaPenaltyAt[tile] = at;
    for (auto &[tile, at] : llvm::reverse(bdUndo))
      bdShortfallAt[tile] = at;
    for (auto &[pos, bytes, count] : llvm::reverse(coreFifoLog)) {
      std::map<int64_t, int> &held = coreFifoBuffers[pos];
      if ((held[bytes] -= count) == 0)
        held.erase(bytes);
    }
    for (auto &[pos, unplaced] : llvm::reverse(coreUndo))
      coreUnplacedAt[pos] = unplaced;
    coreOverflowPenalty = oldCoreOverflow;
    coreOverflowTiles = std::move(oldCoreOverflowTiles);
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
  finalLockGroups = lockGroups().size();
  finalLockPenalty = computeLockPenalty(&finalLockStopped);

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
    InFlightDiagnostic diag =
        device.emitError("SA placer failed to find a legal placement (cascade "
                         "and shared memory penalty=")
        << finalPairing << ", delegate penalty=" << finalDelegate
        << "); try increasing --sa-effort";
    explainPenalties(diag);
    return failure();
  }
  if (int estimated = getResourcePenalty() - finalDelegate; estimated != 0) {
    InFlightDiagnostic diag =
        device.emitWarning("SA placer's resource estimate is over capacity "
                           "(penalty=")
        << estimated
        << "); later passes check exactly, and increasing --sa-effort may help";
    explainPenalties(diag);
  }

  result = currentPlacement;
  return success();
}

void SAPlacer::explainPenalties(InFlightDiagnostic &diag) {
  SmallVector<TileID> spill, core, locks, channels, bds, delegates, pairing;
  int spillPenalty = computeMemSpilloverPenalty(&spill);
  int corePenalty = computeCoreOverflowPenalty(&core);
  int lockPenalty = computeLockPenalty(nullptr, &locks);
  int channelPenalty = 0;
  int bdPenalty = computeBDCountPenalty();
  for (auto &[at, ids] : bdShortfallAt)
    if (ids)
      bds.push_back(at);
  for (auto &[at, _] : currentDMAUsage) {
    int ids = 0;
    int penalty = computeTileDMAPenalty(at, &ids);
    if (penalty != ids) {
      channelPenalty += penalty - ids;
      channels.push_back(at);
    }
    if (ids) {
      bdPenalty += ids;
      bds.push_back(at);
    }
  }
  int delegatePenalty = computeDelegatePenalty(&delegates);
  int pairingPenalty = computePairingPenalty(&pairing);

  // The tuner reads these names.
  std::tuple<StringRef, int, SmallVector<TileID> &> terms[] = {
      {"memory spillover", spillPenalty, spill},
      {"core overflow", corePenalty, core},
      {"locks", lockPenalty, locks},
      {"DMA channels", channelPenalty, channels},
      {"BD ids", bdPenalty, bds},
      {"delegates", delegatePenalty, delegates},
      {"cascade and shared memory pairing", pairingPenalty, pairing}};
  for (auto &[term, penalty, tiles] : terms) {
    if (!penalty)
      continue;
    llvm::sort(tiles);
    tiles.erase(llvm::unique(tiles), tiles.end());
    Operation *anchor = nullptr;
    auto consider = [&](Operation *op) {
      if (!anchor || collected.rank.lookup(op) < collected.rank.lookup(anchor))
        anchor = op;
    };
    if (!tiles.empty()) {
      for (auto &[op, at] : currentPlacement)
        if (at == tiles.front())
          consider(op);
      for (Operation *op : physicalTiles.lookup(tiles.front()))
        consider(op);
    }
    Diagnostic &note =
        anchor ? diag.attachNote(anchor->getLoc()) : diag.attachNote();
    note << term << " penalty=" << penalty;
    for (auto [i, at] : llvm::enumerate(tiles))
      note << (i ? ", " : " on ") << "(" << at.col << ", " << at.row << ")";
  }
}
