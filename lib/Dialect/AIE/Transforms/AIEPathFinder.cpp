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

using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-pathfinder"
#define OVER_CAPACITY_COEFF 0.02
#define USED_CAPACITY_COEFF 0.02
#define DEMAND_COEFF 1.1

WireBundle getConnectingBundle(WireBundle dir) {
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

Pathfinder::Pathfinder(int maxCol, int maxRow, DeviceOp &d) {
  const auto &targetModel = d.getTargetModel();
  // make grid of switchboxes
  for (int col = 0; col <= maxCol; col++) {
    for (int row = 0; row <= maxRow; row++) {
      auto nodeIt = grid.insert({{col, row}, Switchbox{col, row}});
      (void)graph.addNode(nodeIt.first->second);
      Switchbox &thisNode = grid.at({col, row});
      if (row > 0) { // if not in row 0 add channel to North/South
        Switchbox &southernNeighbor = grid.at({col, row - 1});
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
        Switchbox &westernNeighbor = grid.at({col - 1, row});
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

  // initialize weights of all Channels to 1
  // initialize other variables
  for (auto &edge : edges) {
    edge.demand = 1.0;
    edge.usedCapacity = 0;
    edge.fixedCapacity.clear();
    edge.overCapacityCount = 0;
  }

  // initialize maximum iterations flag
  Pathfinder::maxIterReached = false;
}

// Add a flow from src to dst can have an arbitrary number of dst locations due
// to fanout.
void Pathfinder::addFlow(TileID srcCoords, Port srcPort, TileID dstCoords,
                         Port dstPort) {
  // check if a flow with this source already exists
  for (auto &flow : flows) {
    Switchbox *existingSrc = flow.first.first;
    assert(existingSrc && "nullptr flow source");
    Port existingPort = flow.first.second;
    if (existingSrc->col == srcCoords.first &&
        existingSrc->row == srcCoords.second && existingPort == srcPort) {
      // find the vertex corresponding to the destination
      auto matchingSb =
          std::find_if(graph.begin(), graph.end(), [&](const Switchbox *sb) {
            return sb->col == dstCoords.first && sb->row == dstCoords.second;
          });
      assert(matchingSb != graph.end() && "didn't find flow dest");
      flow.second.emplace_back(*matchingSb, dstPort);
      return;
    }
  }

  // If no existing flow was found with this source, create a new flow.
  auto matchingSrcSb =
      std::find_if(graph.begin(), graph.end(), [&](const Switchbox *sb) {
        return sb->col == srcCoords.first && sb->row == srcCoords.second;
      });
  assert(matchingSrcSb != graph.end() && "didn't find flow source");
  auto matchingDstSb =
      std::find_if(graph.begin(), graph.end(), [&](const Switchbox *sb) {
        return sb->col == dstCoords.first && sb->row == dstCoords.second;
      });
  assert(matchingDstSb != graph.end() && "didn't add flow destinations");
  flows.emplace_back(PathEndPoint{*matchingSrcSb, srcPort},
                     std::vector<PathEndPoint>{{*matchingDstSb, dstPort}});
}

// Keep track of connections already used in the AIE; Pathfinder algorithm will
// avoid using these.
bool Pathfinder::addFixedConnection(TileID coords, Port port) {
  // find the correct Channel and indicate the fixed direction
  auto matchingCh = std::find_if(edges.begin(), edges.end(), [&](Channel &ch) {
    return ch.src.col == coords.first && ch.src.row == coords.second &&
           ch.bundle == port.first;
  });
  if (matchingCh == edges.end())
    return false;

  matchingCh->fixedCapacity.insert((uint32_t)port.second);
  return true;
}

static constexpr double INF = std::numeric_limits<double>::max();

std::map<Switchbox *, Switchbox *>
dijkstraShortestPaths(const SwitchboxGraph &graph, Switchbox *src) {
  // Use std::map instead of DenseMap because DenseMap doesn't let you overwrite
  // tombstones.
  auto demand = std::map<Switchbox *, double>();
  auto preds = std::map<Switchbox *, Switchbox *>();
  for (Switchbox *sb : graph)
    demand.emplace(sb, INF);
  demand[src] = 0.0;
  auto cmp = [](const std::pair<double, Switchbox *> &p1,
                const std::pair<double, Switchbox *> &p2) {
    return std::fabs(p1.first - p2.first) <
                   std::numeric_limits<double>::epsilon()
               ? std::less<Switchbox *>()(p1.second, p2.second)
               : p1.first < p2.first;
  };
  std::set<std::pair<double, Switchbox *>, decltype(cmp)> priorityQueue(cmp);
  priorityQueue.insert({demand[src], src});

  while (!priorityQueue.empty()) {
    src = priorityQueue.begin()->second;
    priorityQueue.erase(priorityQueue.begin());
    for (Channel *e : src->getEdges()) {
      Switchbox *dst = &e->getTargetNode();
      if (demand[src] + e->demand < demand[dst]) {
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
      if (ch.fixedCapacity.size() >= ch.maxCapacity) {
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
    routingSolution = {};
    for (auto &ch : edges)
      ch.usedCapacity = 0;

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    for (const auto &flow : flows) {
      // Use dijkstra to find path given current demand from the start
      // switchbox; find the shortest paths to each other switchbox. Output is
      // in the predecessor map, which must then be processed to get individual
      // switchbox settings
      Switchbox *src = flow.first.first;
      assert(src && "nonexistent flow source");
      std::set<Switchbox *> processed;
      std::map<Switchbox *, Switchbox *> preds =
          dijkstraShortestPaths(graph, src);

      // trace the path of the flow backwards via predecessors
      // increment used_capacity for the associated channels
      SwitchSettings switchSettings;
      // set the input bundle for the source endpoint
      switchSettings[src].first = flow.first.second;
      processed.insert(src);
      for (const PathEndPoint &endPoint : flow.second) {
        Switchbox *curr = endPoint.first;
        assert(curr && "endpoint has no source switchbox");
        // set the output bundle for this destination endpoint
        switchSettings[curr].second.insert(endPoint.second);

        // trace backwards until a vertex already processed is reached
        while (!processed.count(curr)) {
          // find the edge from the pred to curr by searching incident edges
          SmallVector<Channel *, 10> channels;
          graph.findIncomingEdgesToNode(*curr, channels);
          auto matchingCh =
              std::find_if(channels.begin(), channels.end(), [&](Channel *ch) {
                return ch->src == *preds[curr];
              });
          assert(matchingCh != channels.end() && "couldn't find ch");
          Channel *ch = *matchingCh;

          // don't use fixed channels
          while (ch->fixedCapacity.count(ch->usedCapacity))
            ch->usedCapacity++;

          // add the entrance port for this Switchbox
          switchSettings[curr].first =
              std::make_pair(getConnectingBundle(ch->bundle), ch->usedCapacity);
          // add the current Switchbox to the map of the predecessor
          switchSettings[preds[curr]].second.insert(
              std::make_pair(ch->bundle, ch->usedCapacity));

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
      routingSolution[flow.first] = switchSettings;
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
