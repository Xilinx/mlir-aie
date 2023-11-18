//===- AIEPathfinder.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-pathfinder"
#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1

WireBundle xilinx::AIE::getConnectingBundle(WireBundle dir) {
  switch (dir) {
  case WireBundle::North:
    return WireBundle::South;
  case WireBundle::South:
    return WireBundle::North;
  case WireBundle::East:
    return WireBundle::West;
  case WireBundle::West:
    return WireBundle::East;
  default:
    return dir;
  }
}

void DynamicTileAnalysis::runAnalysis(DeviceOp &device) {
  LLVM_DEBUG(llvm::dbgs() << "\t---Begin DynamicTileAnalysis Constructor---\n");
  // find the maxCol and maxRow
  maxCol = 0;
  maxRow = 0;
  for (TileOp tileOp : device.getOps<TileOp>()) {
    maxCol = std::max(maxCol, tileOp.colIndex());
    maxRow = std::max(maxRow, tileOp.rowIndex());
  }

  pathfinder->initialize(maxCol, maxRow, device.getTargetModel());

  // for each flow in the device, add it to pathfinder
  // each source can map to multiple different destinations (fanout)
  for (FlowOp flowOp : device.getOps<FlowOp>()) {
    TileOp srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileOp dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    Port srcPort = {flowOp.getSourceBundle(), flowOp.getSourceChannel()};
    Port dstPort = {flowOp.getDestBundle(), flowOp.getDestChannel()};
    LLVM_DEBUG(llvm::dbgs()
               << "\tAdding Flow: (" << srcCoords.col << ", " << srcCoords.row
               << ")" << stringifyWireBundle(srcPort.bundle) << srcPort.channel
               << " -> (" << dstCoords.col << ", " << dstCoords.row << ")"
               << stringifyWireBundle(dstPort.bundle) << dstPort.channel
               << "\n");
    pathfinder->addFlow(srcCoords, srcPort, dstCoords, dstPort);
  }

  // add existing connections so Pathfinder knows which resources are
  // available search all existing SwitchBoxOps for exising connections
  for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
    for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
      TileID existingCoord = {switchboxOp.colIndex(), switchboxOp.rowIndex()};
      Port existingPort = {connectOp.getDestBundle(),
                           connectOp.getDestChannel()};
      if (!pathfinder->addFixedConnection(existingCoord, existingPort))
        switchboxOp.emitOpError(
            "Couldn't connect tile (" + std::to_string(existingCoord.col) +
            ", " + std::to_string(existingCoord.row) + ") to port (" +
            stringifyWireBundle(existingPort.bundle) + ", " +
            std::to_string(existingPort.channel) + ")\n");
    }
  }

  // all flows are now populated, call the congestion-aware pathfinder
  // algorithm
  // check whether the pathfinder algorithm creates a legal routing
  flowSolutions = pathfinder->findPaths(maxIterations);
  if (!pathfinder->isLegal())
    device.emitError("Unable to find a legal routing");

  // initialize all flows as unprocessed to prep for rewrite
  for (const auto &[pathEndPoint, switchSetting] : flowSolutions) {
    processedFlows[pathEndPoint] = false;
    LLVM_DEBUG(llvm::dbgs() << "Flow starting at (" << pathEndPoint.sb.col
                            << "," << pathEndPoint.sb.row << "):\t");
    LLVM_DEBUG(llvm::dbgs() << switchSetting);
  }

  // fill in coords to TileOps, SwitchboxOps, and ShimMuxOps
  for (auto tileOp : device.getOps<TileOp>()) {
    int col, row;
    col = tileOp.colIndex();
    row = tileOp.rowIndex();
    maxCol = std::max(maxCol, col);
    maxRow = std::max(maxRow, row);
    assert(coordToTile.count({col, row}) == 0);
    coordToTile[{col, row}] = tileOp;
  }
  for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
    int col, row;
    col = switchboxOp.colIndex();
    row = switchboxOp.rowIndex();
    assert(coordToSwitchbox.count({col, row}) == 0);
    coordToSwitchbox[{col, row}] = switchboxOp;
  }
  for (auto shimmuxOp : device.getOps<ShimMuxOp>()) {
    int col, row;
    col = shimmuxOp.colIndex();
    row = shimmuxOp.rowIndex();
    assert(coordToShimMux.count({col, row}) == 0);
    coordToShimMux[{col, row}] = shimmuxOp;
  }

  LLVM_DEBUG(llvm::dbgs() << "\t---End DynamicTileAnalysis Constructor---\n");
}

TileOp DynamicTileAnalysis::getTile(OpBuilder &builder, int col, int row) {
  if (coordToTile.count({col, row})) {
    return coordToTile[{col, row}];
  }
  auto tileOp = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
  coordToTile[{col, row}] = tileOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return tileOp;
}

SwitchboxOp DynamicTileAnalysis::getSwitchbox(OpBuilder &builder, int col,
                                              int row) {
  assert(col >= 0);
  assert(row >= 0);
  if (coordToSwitchbox.count({col, row})) {
    return coordToSwitchbox[{col, row}];
  }
  auto switchboxOp = builder.create<SwitchboxOp>(builder.getUnknownLoc(),
                                                 getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToSwitchbox[{col, row}] = switchboxOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return switchboxOp;
}

ShimMuxOp DynamicTileAnalysis::getShimMux(OpBuilder &builder, int col) {
  assert(col >= 0);
  int row = 0;
  if (coordToShimMux.count({col, row})) {
    return coordToShimMux[{col, row}];
  }
  assert(getTile(builder, col, row).isShimNOCTile());
  auto switchboxOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(),
                                               getTile(builder, col, row));
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                builder.getUnknownLoc());
  coordToShimMux[{col, row}] = switchboxOp;
  maxCol = std::max(maxCol, col);
  maxRow = std::max(maxRow, row);
  return switchboxOp;
}

void Pathfinder::initialize(int maxCol, int maxRow,
                            const AIETargetModel &targetModel) {
  // make grid of switchboxes
  for (int col = 0; col <= maxCol; col++) {
    for (int row = 0; row <= maxRow; row++) {
      auto nodeIt = grid.insert({{col, row}, SwitchboxNode{col, row}});
      (void)graph.addNode(nodeIt.first->second);
      SwitchboxNode &thisNode = grid.at({col, row});
      if (row > 0) { // if not in row 0 add channel to North/South
        SwitchboxNode &southernNeighbor = grid.at({col, row - 1});
        if (uint32_t maxCapacity = targetModel.getNumSourceSwitchboxConnections(
                col, row, WireBundle::South)) {
          edges.emplace_back(southernNeighbor, thisNode, WireBundle::North,
                             maxCapacity);
          (void)graph.connect(southernNeighbor, thisNode, edges.back());
        }
        if (uint32_t maxCapacity = targetModel.getNumDestSwitchboxConnections(
                col, row, WireBundle::South)) {
          edges.emplace_back(thisNode, southernNeighbor, WireBundle::South,
                             maxCapacity);
          (void)graph.connect(thisNode, southernNeighbor, edges.back());
        }
      }

      if (col > 0) { // if not in col 0 add channel to East/West
        SwitchboxNode &westernNeighbor = grid.at({col - 1, row});
        if (uint32_t maxCapacity = targetModel.getNumSourceSwitchboxConnections(
                col, row, WireBundle::West)) {
          edges.emplace_back(westernNeighbor, thisNode, WireBundle::East,
                             maxCapacity);
          (void)graph.connect(westernNeighbor, thisNode, edges.back());
        }
        if (uint32_t maxCapacity = targetModel.getNumDestSwitchboxConnections(
                col, row, WireBundle::West)) {
          edges.emplace_back(thisNode, westernNeighbor, WireBundle::West,
                             maxCapacity);
          (void)graph.connect(thisNode, westernNeighbor, edges.back());
        }
      }
    }
  }
}

// Add a flow from src to dst can have an arbitrary number of dst locations due
// to fanout.
void Pathfinder::addFlow(TileID srcCoords, Port srcPort, TileID dstCoords,
                         Port dstPort) {
  // check if a flow with this source already exists
  for (auto &flow : flows) {
    SwitchboxNode *existingSrc = flow.src.sb;
    assert(existingSrc && "nullptr flow source");
    if (Port existingPort = flow.src.port; existingSrc->col == srcCoords.col &&
                                           existingSrc->row == srcCoords.row &&
                                           existingPort == srcPort) {
      // find the vertex corresponding to the destination
      auto matchingSb = std::find_if(
          graph.begin(), graph.end(), [&](const SwitchboxNode *sb) {
            return sb->col == dstCoords.col && sb->row == dstCoords.row;
          });
      assert(matchingSb != graph.end() && "didn't find flow dest");
      flow.dsts.push_back({*matchingSb, dstPort});
      return;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  auto matchingSrcSb =
      std::find_if(graph.begin(), graph.end(), [&](const SwitchboxNode *sb) {
        return sb->col == srcCoords.col && sb->row == srcCoords.row;
      });
  assert(matchingSrcSb != graph.end() && "didn't find flow source");
  auto matchingDstSb =
      std::find_if(graph.begin(), graph.end(), [&](const SwitchboxNode *sb) {
        return sb->col == dstCoords.col && sb->row == dstCoords.row;
      });
  assert(matchingDstSb != graph.end() && "didn't add flow destinations");
  flows.push_back({PathEndPointNode{*matchingSrcSb, srcPort},
                   std::vector<PathEndPointNode>{{*matchingDstSb, dstPort}}});
}

// Keep track of connections already used in the AIE; Pathfinder algorithm will
// avoid using these.
bool Pathfinder::addFixedConnection(TileID coords, Port port) {
  // find the correct Channel and indicate the fixed direction
  auto matchingCh =
      std::find_if(edges.begin(), edges.end(), [&](ChannelEdge &ch) {
        return ch.src.col == coords.col && ch.src.row == coords.row &&
               ch.bundle == port.bundle;
      });
  if (matchingCh == edges.end())
    return false;

  matchingCh->fixedCapacity.insert(port.channel);
  return true;
}

static constexpr double INF = std::numeric_limits<double>::max();

std::map<SwitchboxNode *, SwitchboxNode *>
dijkstraShortestPaths(const SwitchboxGraph &graph, SwitchboxNode *src) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you overwrite
  // tombstones.
  auto demand = std::map<SwitchboxNode *, double>();
  auto preds = std::map<SwitchboxNode *, SwitchboxNode *>();
  for (SwitchboxNode *sb : graph)
    demand.emplace(sb, INF);
  demand[src] = 0.0;
  auto cmp = [](const std::pair<double, SwitchboxNode *> &p1,
                const std::pair<double, SwitchboxNode *> &p2) {
    return std::fabs(p1.first - p2.first) <
                   std::numeric_limits<double>::epsilon()
               ? std::less<Switchbox *>()(p1.second, p2.second)
               : p1.first < p2.first;
  };
  std::set<std::pair<double, SwitchboxNode *>, decltype(cmp)> priorityQueue(
      cmp);
  priorityQueue.insert({demand[src], src});

  while (!priorityQueue.empty()) {
    src = priorityQueue.begin()->second;
    priorityQueue.erase(priorityQueue.begin());
    for (ChannelEdge *e : src->getEdges()) {
      if (SwitchboxNode *dst = &e->getTargetNode(); demand[src] + e->demand < demand[dst]) {
        priorityQueue.erase({demand[dst], dst});

        demand[dst] = demand[src] + e->demand;
        preds[dst] = src;

        priorityQueue.insert({demand[dst], dst});
      }
    }
  }
  return preds;
}

// Perform congestion-aware routing for all flows which have been added.
// Use Dijkstra's shortest path to find routes, and use "demand" as the weights.
// If the routing finds too much congestion, update the demand weights
// and repeat the process until a valid solution is found.
// Returns a map specifying switchbox settings for all flows.
// If no legal routing can be found after maxIterations, returns empty vector.
std::map<PathEndPoint, SwitchSettings>
Pathfinder::findPaths(const int maxIterations) {
  LLVM_DEBUG(llvm::dbgs() << "Begin Pathfinder::findPaths\n");
  int iterationCount = 0;
  std::map<PathEndPoint, SwitchSettings> routingSolution;

  // initialize all Channel histories to 0
  for (auto &ch : edges)
    ch.overCapacityCount = 0;

  do {
    LLVM_DEBUG(llvm::dbgs()
               << "Begin findPaths iteration #" << iterationCount << "\n");
    // update demand on all channels
    for (auto &ch : edges) {
      if (ch.fixedCapacity.size() >=
          static_cast<unsigned int>(ch.maxCapacity)) {
        ch.demand = INF;
      } else {
        double history = 1.0 + OVER_CAPACITY_COEFF * ch.overCapacityCount;
        double congestion = 1.0 + USED_CAPACITY_COEFF * ch.usedCapacity;
        ch.demand = history * congestion;
      }
    }
    // if reach maxIterations, throw an error since no routing can be found
    // TODO: add error throwing mechanism
    if (++iterationCount > maxIterations) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Pathfinder: maxIterations has been exceeded ("
                 << maxIterations
                 << " iterations)...unable to find routing for flows.\n");
      //  return the invalid solution for debugging purposes
      maxIterReached = true;
      return routingSolution;
    }

    // "rip up" all routes, i.e. set used capacity in each Channel to 0
    routingSolution.clear();
    for (auto &ch : edges)
      ch.usedCapacity = 0;

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    for (const auto &flow : flows) {
      // Use dijkstra to find path given current demand from the start
      // switchbox; find the shortest paths to each other switchbox. Output is
      // in the predecessor map, which must then be processed to get individual
      // switchbox settings
      SwitchboxNode *src = flow.src.sb;
      assert(src && "nonexistent flow source");
      std::set<SwitchboxNode *> processed;
      std::map<SwitchboxNode *, SwitchboxNode *> preds =
          dijkstraShortestPaths(graph, src);

      // trace the path of the flow backwards via predecessors
      // increment used_capacity for the associated channels
      SwitchSettings switchSettings;
      // set the input bundle for the source endpoint
      switchSettings[*src].src = flow.src.port;
      processed.insert(src);
      for (const PathEndPointNode &endPoint : flow.dsts) {
        SwitchboxNode *curr = endPoint.sb;
        assert(curr && "endpoint has no source switchbox");
        // set the output bundle for this destination endpoint
        switchSettings[*curr].dsts.insert(endPoint.port);

        // trace backwards until a vertex already processed is reached
        while (!processed.count(curr)) {
          // find the edge from the pred to curr by searching incident edges
          SmallVector<ChannelEdge *, 10> channels;
          graph.findIncomingEdgesToNode(*curr, channels);
          auto matchingCh = std::find_if(
              channels.begin(), channels.end(),
              [&](ChannelEdge *ch) { return ch->src == *preds[curr]; });
          assert(matchingCh != channels.end() && "couldn't find ch");
          ChannelEdge *ch = *matchingCh;

          // don't use fixed channels
          while (ch->fixedCapacity.count(ch->usedCapacity))
            ch->usedCapacity++;

          // add the entrance port for this Switchbox
          switchSettings[*curr].src = {getConnectingBundle(ch->bundle),
                                       ch->usedCapacity};
          // add the current Switchbox to the map of the predecessor
          switchSettings[*preds[curr]].dsts.insert(
              {ch->bundle, ch->usedCapacity});

          ch->usedCapacity++;
          // if at capacity, bump demand to discourage using this Channel
          if (ch->usedCapacity >= ch->maxCapacity) {
            // this means the order matters!
            ch->demand *= DEMAND_COEFF;
          }

          processed.insert(curr);
          curr = preds[curr];
        }
      }
      // add this flow to the proposed solution
      routingSolution[flow.src] = switchSettings;
    }
  } while (!isLegal()); // continue iterations until a legal routing is found
  return routingSolution;
}

// Check that every channel does not exceed max capacity.
bool Pathfinder::isLegal() {
  bool legal = true; // assume legal until found otherwise
  // check if maximum number of iterations has been reached
  if (maxIterReached)
    legal = false;
  for (auto &e : edges)
    if (e.usedCapacity > e.maxCapacity) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Too much capacity on Edge (" << e.getTargetNode().col
                 << ", " << e.getTargetNode().row << ") . "
                 << stringifyWireBundle(e.bundle) << "\t: used_capacity = "
                 << e.usedCapacity << "\t: Demand = " << e.demand << "\n");
      e.overCapacityCount++;
      LLVM_DEBUG(llvm::dbgs()
                 << "over_capacity_count = " << e.overCapacityCount << "\n");
      legal = false;
    }
  return legal;
}
