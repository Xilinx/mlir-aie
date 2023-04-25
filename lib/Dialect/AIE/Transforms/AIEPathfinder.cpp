//===- AIEPathfinder.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"
#include <iostream>

#include <aie/Dialect/AIE/Transforms/AIEPathfinder.h>

using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-pathfinder"

WireBundle getConnectingBundle(WireBundle dir) {
  switch (dir) {
  case WireBundle::North:
    return WireBundle::South;
    break;
  case WireBundle::South:
    return WireBundle::North;
    break;
  case WireBundle::East:
    return WireBundle::West;
    break;
  case WireBundle::West:
    return WireBundle::East;
    break;
  default:
    return dir;
  }
}

Pathfinder::Pathfinder() { initializeGraph(0, 0); }

Pathfinder::Pathfinder(int _maxcol, int _maxrow) {
  initializeGraph(_maxcol, _maxrow);
}

void Pathfinder::initializeGraph(int maxcol, int maxrow) {
  // make grid of switchboxes
  for (int row = 0; row <= maxrow; row++) {
    for (int col = 0; col <= maxcol; col++) {
      int id = add_vertex(graph);
      graph[id].row = row;
      graph[id].col = col;
      graph[id].pred = 0;
      graph[id].processed = false;
      if (row > 0) { // if not in row 0 add channel to North/South
        auto north_edge = add_edge(id - maxcol - 1, id, graph).first;
        graph[north_edge].bundle = WireBundle::North;
        graph[north_edge].max_capacity = 6;
        auto south_edge = add_edge(id, id - maxcol - 1, graph).first;
        graph[south_edge].bundle = WireBundle::South;
        graph[south_edge].max_capacity = 4;
      }
      if (col > 0) { // if not in col 0 add channel to East/West
        auto east_edge = add_edge(id - 1, id, graph).first;
        graph[east_edge].bundle = WireBundle::East;
        graph[east_edge].max_capacity = 4;
        auto west_edge = add_edge(id, id - 1, graph).first;
        graph[west_edge].bundle = WireBundle::West;
        graph[west_edge].max_capacity = 4;
      }
    }
  }

  // initialize weights of all Channels to 1
  // initialize other variables
  auto edge_pair = edges(graph);
  for (auto edge = edge_pair.first; edge != edge_pair.second; edge++) {
    graph[*edge].demand = 1;
    graph[*edge].used_capacity = 0;
    graph[*edge].fixed_capacity.clear();
    graph[*edge].over_capacity_count = 0;
  }

  // initialize maximum iterations flag
  Pathfinder::maxIterReached = false;
}

// Pathfinder::addFlow
// add a flow from src to dst
// can have an arbitrary number of dst locations due to fanout
void Pathfinder::addFlow(Coord srcCoords, Port srcPort, 
                        Coord dstCoords, Port dstPort, int flow_id) {
  // check if a flow with this source already exists
  for (unsigned int i = 0; i < flows.size(); i++) {
    Switchbox *otherSrc = std::get<0>(*flows[i]).first;
    Port otherPort = std::get<0>(*flows[i]).second;
    if (otherSrc->col == srcCoords.first && otherSrc->row == srcCoords.second &&
        otherPort == srcPort && std::get<2>(*flows[i]) == flow_id) {
      // find the vertex corresponding to the destination
      PathEndPoint dst;
      auto vpair = vertices(graph);
      for (vertex_iterator v = vpair.first; v != vpair.second; v++) {
        Switchbox *sb = &graph[*v];
        if (sb->col == dstCoords.first && sb->row == dstCoords.second) {
          dst = std::make_pair(sb, dstPort);
          break;
        }
      }
      // add the destination to this existing flow, and finish
      std::get<1>(*flows[i]).push_back(dst);
      return;
    }
  }

  // if no existing flow was found with this source, create a new flow
  Flow* flow = new Flow(); // new is important to generate a unique pointer!
  //std::unique_ptr<Flow> flow (new Flow);
  auto vpair = vertices(graph);
  for (vertex_iterator v = vpair.first; v != vpair.second; v++) {
    Switchbox *sb = &graph[*v];
    // check if this vertex matches the source
    if (sb->col == srcCoords.first && sb->row == srcCoords.second)
      std::get<0>(*flow) = std::make_pair(sb, srcPort);

    // check if this vertex matches the destination
    if (sb->col == dstCoords.first && sb->row == dstCoords.second)
      std::get<1>(*flow).push_back(std::make_pair(sb, dstPort));
  }

  std::get<2>(*flow) = flow_id;
  flows.push_back(flow);
  return;
}

// Pathfinder::addFixedConnection
// Keep track of connections already used in the AIE
// Pathfinder algorithm will avoid using these
void Pathfinder::addFixedConnection(Coord coords, Port port) {
  // find the correct Channel and indicate the fixed direction
  auto edge_pair = edges(graph);
  for (edge_iterator e = edge_pair.first; e != edge_pair.second; e++) {
    if (graph[source(*e, graph)].col == coords.first &&
        graph[source(*e, graph)].row == coords.second &&
        graph[*e].bundle == port.first) {
      graph[*e].fixed_capacity.insert(port.second);
      break;
    }
  }
}

// Pathfinder::findPaths
// Primary function for the class
// Perform congestion-aware routing for all flows which have been added.
// Use Dijkstra's shortest path to find routes, and use "demand" as the weights
// if the routing finds too much congestion, update the demand weights
// and repeat the process until a vaild solution is found
//
// returns a map specifying switchbox settings for all flows
// if no legal routing can be found after MAX_ITERATIONS, returns empty vector
void Pathfinder::findPaths(
        DenseMap<Flow*, SwitchSettings*> & flow_solutions,
        const int MAX_ITERATIONS) {

  LLVM_DEBUG(llvm::dbgs() << "Begin Pathfinder::findPaths()\n");
  int iteration_count = 0;
  // Each flow has a set of SwitchSettings which implements that flow
  // The implementations for all flows constitutes a routing solution.

  // initialize all Channel histories to 0
  auto edge_pair = edges(graph);
  for (auto edge = edge_pair.first; edge != edge_pair.second; edge++) {
    graph[*edge].over_capacity_count = 0;
  }

// Pathfinder iteration loop
#define over_capacity_coeff 0.02
#define used_capacity_coeff 0.02
  do {
    LLVM_DEBUG(llvm::dbgs()
               << "Begin findPaths iteration #" << iteration_count << "\n");
    // update demand on all channels
    edge_pair = edges(graph);
    for (edge_iterator it = edge_pair.first; it != edge_pair.second; it++) {
      Channel *ch = &graph[*it];
      // LLVM_DEBUG(llvm::dbgs() << "Pre update:\tEdge " << *it << "\t: used = "
      // << ch->used_capacity <<
      //   "\t demand = " << ch->demand << "\t over_capacity_count= " <<
      //   ch->over_capacity_count<< "\t");
      if (ch->fixed_capacity.size() >= ch->max_capacity) {
        ch->demand = std::numeric_limits<float>::max();
      } else {
        float history = 1 + over_capacity_coeff * ch->over_capacity_count;
        float congestion = 1 + used_capacity_coeff * ch->used_capacity;
        // std::max(0, ch->used_capacity - ch->max_capacity);
        ch->demand = history * congestion;
      }
    }
    // if reach MAX_ITERATIONS, throw an error since no routing can be found
    // TODO: add error throwing mechanism
    if (++iteration_count > MAX_ITERATIONS) {
      //assert(...)
      LLVM_DEBUG(llvm::dbgs()
                 << "Pathfinder: MAX_ITERATIONS has been exceeded ("
                 << MAX_ITERATIONS
                 << " iterations)...unable to find routing for flows.\n");
      // exit with the invalid solution for debugging purposes
      maxIterReached = true;
      return;
    }

    // "rip up" all routes, i.e. set used capacity in each Channel to 0
    flow_solutions.clear();
    auto edge_pair = edges(graph);
    for (edge_iterator e = edge_pair.first; e != edge_pair.second; e++) {
      graph[*e].used_capacity = 0;
    }

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    //for (auto iter = flows.begin(); iter != flows.end(); iter++) {
    //  Flow* flow = *iter;
    
    auto vpair = vertices(graph);
    for (Flow* flow : flows) {
      PathEndPoint start_point = std::get<0>(*flow);
      // Find the source vertex (switchbox)
      vertex_descriptor* src = nullptr;
      for (vertex_iterator v = vpair.first; v != vpair.second; v++) {
        Switchbox *sb = &graph[*v];
        sb->processed = false;
        if (sb->col == start_point.first->col &&
            sb->row == start_point.first->row) {
          src = new vertex_descriptor(*v);
        }
      }

      // use dijkstra to find path given current demand
      // from the start switchbox, find shortest path to each other switchbox
      // output is in the predecessor map, which must then be processed to get
      // individual switchbox settings
      dijkstra_shortest_paths(
          graph, *src,
          weight_map(get(&Channel::demand, graph))
              .predecessor_map(get(&Switchbox::pred, graph)));

      SwitchSettings* switchSettings = new SwitchSettings();
      int flow_ID = std::get<2>(*flow);
      LLVM_DEBUG(llvm::dbgs() << "Creating SwitchSettings for Flow " << flow_ID << "\n"); 

      // set the input bundle for the source endpoint
      std::get<0>((*switchSettings)[&graph[*src]]) = start_point.second;
      // set the packet ID for this flow
      std::get<2>((*switchSettings)[&graph[*src]]) = flow_ID;
      graph[*src].processed = true;

      // trace the path of the flow backwards via predecessors
      SmallVector<PathEndPoint> flow_endpoints = std::get<1>(*flow);
      for (unsigned int i = 0; i < flow_endpoints.size(); i++) {
        LLVM_DEBUG(llvm::dbgs() << "\tProcessing Endpoint (" 
                                <<  flow_endpoints[i].first->col << ", "
                                <<  flow_endpoints[i].first->row << ")\n");
        vertex_descriptor curr = 0;
        for (vertex_iterator v = vpair.first; v != vpair.second; v++)
          if (graph[*v].col == flow_endpoints[i].first->col &&
              graph[*v].row == flow_endpoints[i].first->row) {
            curr = *v;
            break;
          }

        Switchbox *sb = &graph[curr];
        // set the output bundle for this destination endpoint
        std::get<1>((*switchSettings)[sb]).push_back(flow_endpoints[i].second);
        std::get<2>((*switchSettings)[sb]) = flow_ID;

        // trace backwards until a vertex already processed is reached
        while (sb->processed == false) {
          LLVM_DEBUG(llvm::dbgs() << "\t\tProcessing Switchbox (" 
                                  << sb->col << ", " << sb->row << ")\n");
          // find the edge from the pred to curr by searching incident edges
          auto inedges = in_edges(curr, graph);
          Channel *ch = nullptr;
          for (in_edge_iterator it = inedges.first; it != inedges.second;
               it++) {
            if (source(*it, graph) == (unsigned)sb->pred) {
              // found the channel used in the path
              ch = &graph[*it];
              break;
            }
          }
          assert(ch != nullptr);

          // don't use fixed channels
          while (ch->fixed_capacity.count(ch->used_capacity))
            ch->used_capacity++;
          

          // add the entrance port for this Switchbox
          std::get<0>((*switchSettings)[sb]) = std::make_pair(
              getConnectingBundle(ch->bundle), ch->used_capacity);
          // add the current Switchbox to the map of the predecessor
          std::get<1>((*switchSettings)[&graph[sb->pred]]).push_back(
              std::make_pair(ch->bundle, ch->used_capacity));
          // add flow id for this connection
          std::get<2>((*switchSettings)[sb]) = flow_ID;

          // increment used_capacity for the associated channel
          if (flow_ID == -1) // allow packet flows to use the same channels
            ch->used_capacity++;

          // if at capacity, bump demand to discourage using this Channel
          if (ch->used_capacity >= ch->max_capacity) {
            // this means the order matters!
            ch->demand *= 1.1;
          }

          sb->processed = true;
          curr = sb->pred;
          sb = &graph[curr];
        }
      }
      // Add this flow to the proposed solution
      flow_solutions[flow] = switchSettings;

      // Print debugging info
      printFlow(flow);
      printSwitchSettings(switchSettings);
      LLVM_DEBUG(llvm::dbgs() << "flow_solutions.size(): " << flow_solutions.size() << "\n\n");
    }
  } while (!isLegal()); // continue iterations until a legal routing is found
  LLVM_DEBUG(llvm::dbgs() << "End Pathfinder::findPaths()\n");
}

// check that every channel does not exceed max capacity
bool Pathfinder::isLegal() {
  auto edge_pair = edges(graph);
  bool legal = true; // assume legal until found otherwise
  // check if maximum number of iterations has been reached
  if (maxIterReached)
    legal = false;
  for (edge_iterator e = edge_pair.first; e != edge_pair.second; e++) {
    if (graph[*e].used_capacity > graph[*e].max_capacity) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Too much capacity on Edge ("
                 << graph[source(*e, graph)].col << ", "
                 << graph[source(*e, graph)].row << ") -> "
                 << stringifyWireBundle(graph[*e].bundle)
                 << "\t: used_capacity = " << graph[*e].used_capacity
                 << "\t: Demand = " << graph[*e].demand << "\n");
      graph[*e].over_capacity_count++;
      LLVM_DEBUG(llvm::dbgs() << "over_capacity_count = "
                              << graph[*e].over_capacity_count << "\n");
      legal = false;
    }
  }
  return legal;
}

void Pathfinder::printFlows() {
  for (Flow* f : flows)
    printFlow(f);
}

void Pathfinder::printFlow(Flow* f) {
  PathEndPoint flow_start = std::get<0>(*f);
  Switchbox* flow_start_sb = flow_start.first;
  SmallVector<PathEndPoint> flow_end = std::get<1>(*f);

  LLVM_DEBUG(llvm::dbgs() << "Printing Flow ID " << std::get<2>(*f)<< "\n");
  LLVM_DEBUG(llvm::dbgs() << "\tstarting at: (" 
        << flow_start_sb->col << ", " << flow_start_sb->row << ") : "
        << stringifyWireBundle(flow_start.second.first) << flow_start.second.second << "\n");

  for (PathEndPoint p : flow_end) {
    Switchbox* flow_end_sb = p.first;
    LLVM_DEBUG(llvm::dbgs() << "\tending at: (" 
        << flow_end_sb->col << ", " << flow_end_sb->row << ") : "
        << stringifyWireBundle(p.second.first) << p.second.second << "\n");
  }
}

void Pathfinder::printSwitchSettings(SwitchSettings* settings) {
  for(auto settings_iter = settings->begin(); settings_iter != settings->end(); settings_iter++) {
    Switchbox* sb = settings_iter->first;
    SwitchConnection connection = settings_iter->second;
    LLVM_DEBUG(llvm::dbgs() << "SwitchConnection: Flow ID " << std::get<2>(connection) << "\n");
    LLVM_DEBUG(llvm::dbgs() << "\tSwitchbox (" << sb->col << ", " << sb->row << ")\n");
    Port srcPort = std::get<0>(connection);
    LLVM_DEBUG(llvm::dbgs() << "\tSource Port "
        << stringifyWireBundle(srcPort.first) << srcPort.second << "\n");
    SmallVector<Port> destPorts = std::get<1>(connection);
    for(Port p : destPorts) {
      LLVM_DEBUG(llvm::dbgs() << "\tDest Port "
        << stringifyWireBundle(p.first) << p.second << "\n");
    }
  }
    LLVM_DEBUG(llvm::dbgs() << "\n");
}

