//===- AIEPathfinder.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"

#include <aie/AIEPathfinder.h>

using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-pathfinder"

WireBundle getConnectingBundle(WireBundle dir) {
  switch(dir) {
    case WireBundle::North: return WireBundle::South; break;
    case WireBundle::South: return WireBundle::North; break;
    case WireBundle::East: return WireBundle::West; break;
    case WireBundle::West: return WireBundle::East; break;
    default: return dir;
  }
}

Pathfinder::Pathfinder() { initializeGraph(0, 0); }

Pathfinder::Pathfinder(int _maxcol, int _maxrow) { 
  initializeGraph(_maxcol, _maxrow); 
}

void Pathfinder::initializeGraph(int maxcol, int maxrow) {
  //make grid of switchboxes
  for(int row = 0; row <= maxrow; row++) {
    for(int col = 0; col <= maxcol; col++) {
      int id = add_vertex(graph);
      graph[id].row = row;
      graph[id].col = col;
      graph[id].pred = 0;
      graph[id].processed = false;
      if(row > 0) { // if not in row 0 add channel to North/South 
        auto north_edge = add_edge(id-maxcol-1, id, graph).first;
        graph[north_edge].bundle = WireBundle::North;
        graph[north_edge].max_capacity = 6;
        auto south_edge = add_edge(id, id-maxcol-1, graph).first;
        graph[south_edge].bundle = WireBundle::South;
        graph[south_edge].max_capacity = 4;
      }
      if(col > 0) { // if not in col 0 add channel to East/West
        auto east_edge = add_edge(id-1, id, graph).first;
        graph[east_edge].bundle = WireBundle::East;
        graph[east_edge].max_capacity = 4;
        auto west_edge = add_edge(id, id-1, graph).first;
        graph[west_edge].bundle = WireBundle::West;
        graph[west_edge].max_capacity = 4;
      }
    }
  }

  // initialize weights of all Channels to 1
  // initialize other variables
  auto edge_pair = edges(graph);
  for(auto edge = edge_pair.first; edge != edge_pair.second; edge++) {
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
                         Coord dstCoords, Port dstPort) {
  //check if a flow with this source already exists
  for(unsigned int i = 0; i < flows.size(); i++) {
    Switchbox* otherSrc = flows[i].first.first;
    Port otherPort = flows[i].first.second;
    if(otherSrc->col == srcCoords.first &&
      otherSrc->row == srcCoords.second &&
      otherPort == srcPort ) {
        //find the vertex corresponding to the destination
        PathEndPoint dst;
        auto vpair = vertices(graph);
        for(vertex_iterator v = vpair.first; v != vpair.second; v++) {
          Switchbox* sb = &graph[*v];
          if(sb->col == dstCoords.first && 
          sb->row == dstCoords.second) {
            dst = std::make_pair(sb, dstPort);
            break;
          }
        }
        // add the destination to this existing flow, and finish
        flows[i].second.push_back(dst);
        return;
    }
  }

  // if no existing flow was found with this source, create a new flow  
  Flow flow;
  auto vpair = vertices(graph);
  for(vertex_iterator v = vpair.first; v != vpair.second; v++) {
    Switchbox* sb = &graph[*v];
    //check if this vertex matches the source 
    if(sb->col == srcCoords.first && sb->row == srcCoords.second)
      flow.first = std::make_pair(sb, srcPort);
    
    //check if this vertex matches the destination 
    if(sb->col == dstCoords.first && sb->row == dstCoords.second)
      flow.second.push_back(std::make_pair(sb, dstPort));
  }

  flows.push_back(flow);
  return;
}

// Pathfinder::addFixedConnection
// Keep track of connections already used in the AIE
// Pathfinder algorithm will avoid using these
void Pathfinder::addFixedConnection(Coord coords, Port port) {
  //find the correct Channel and indicate the fixed direction
  auto edge_pair = edges(graph);
  for(edge_iterator e = edge_pair.first; e != edge_pair.second; e++) {
    if(graph[source(*e, graph)].col == coords.first &&
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
std::map< PathEndPoint, SwitchSettings >
Pathfinder::findPaths(const int MAX_ITERATIONS) {
  LLVM_DEBUG(llvm::dbgs() << "Begin Pathfinder::findPaths\n");
  int iteration_count = 0;
  std::map< PathEndPoint, SwitchSettings > routing_solution; 

  // initialize all Channel histories to 0
  auto edge_pair = edges(graph);
  for(auto edge = edge_pair.first; edge != edge_pair.second; edge++) {
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
    for(edge_iterator it = edge_pair.first; it != edge_pair.second; it++) {
      Channel *ch = &graph[*it];
      // LLVM_DEBUG(llvm::dbgs() << "Pre update:\tEdge " << *it << "\t: used = "
      // << ch->used_capacity <<
      //   "\t demand = " << ch->demand << "\t over_capacity_count= " <<
      //   ch->over_capacity_count<< "\t");
      if(ch->fixed_capacity.size() >= ch->max_capacity) {
        ch->demand = std::numeric_limits<float>::max();
      } else {
        float history = 1 + over_capacity_coeff*ch->over_capacity_count;
        float congestion = 1 + used_capacity_coeff*ch->used_capacity;
                            //std::max(0, ch->used_capacity - ch->max_capacity);
        ch->demand = history * congestion;
      }
    }
    // if reach MAX_ITERATIONS, throw an error since no routing can be found
    // TODO: add error throwing mechanism
    if(++iteration_count > MAX_ITERATIONS) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Pathfinder: MAX_ITERATIONS has been exceeded ("
                 << MAX_ITERATIONS
                 << " iterations)...unable to find routing for flows.\n");
      //return {};
      // return the invalid solution for debugging purposes
      maxIterReached = true;
      return routing_solution;
    }

    // "rip up" all routes, i.e. set used capacity in each Channel to 0
    routing_solution = {};
    auto edge_pair = edges(graph);
    for(edge_iterator e = edge_pair.first; e != edge_pair.second; e++) {
      graph[*e].used_capacity = 0;
    }

    // for each flow, find the shortest path from source to destination
    // update used_capacity for the path between them
    for(auto flow : flows) {
      auto vpair = vertices(graph);

      vertex_descriptor src;
      for(vertex_iterator v = vpair.first; v != vpair.second; v++) {
        Switchbox* sb = &graph[*v];
        sb->processed = false;
        if(sb->col == flow.first.first->col && sb->row == flow.first.first->row)
          src = *v;
      }

      // use dijkstra to find path given current demand
      // from the start switchbox, find shortest path to each other switchbox
      // output is in the predecessor map, which must then be processed to get
      // individual switchbox settings
      dijkstra_shortest_paths(graph, src,
                  weight_map(get(&Channel::demand, graph))
                  .predecessor_map(get(&Switchbox::pred, graph)));

      // trace the path of the flow backwards via predecessors
      // increment used_capacity for the associated channels
      SwitchSettings switchSettings = SwitchSettings();
      //set the input bundle for the source endpoint
      switchSettings[&graph[src]].first = flow.first.second;
      graph[src].processed = true;
      for(unsigned int i = 0; i < flow.second.size(); i++) {
        vertex_descriptor curr;
        for(vertex_iterator v = vpair.first; v != vpair.second; v++)
          if(graph[*v].col == flow.second[i].first->col && graph[*v].row == flow.second[i].first->row)
            curr = *v;
        Switchbox *sb = &graph[curr];

        //set the output bundle for this destination endpoint
        switchSettings[sb].second.insert(flow.second[i].second);

        // trace backwards until a vertex already processed is reached
        while(sb->processed == false) {
          // find the edge from the pred to curr by searching incident edges
          auto inedges = in_edges(curr, graph);
          Channel *ch = nullptr;
          for(in_edge_iterator it = inedges.first; it != inedges.second; it++) {
            if(source(*it, graph) == (unsigned)sb->pred) { 
              // found the channel used in the path
              ch = &graph[*it];
              break;
            }
          }
          assert(ch != nullptr);

          //don't use fixed channels
          while(ch->fixed_capacity.count(ch->used_capacity))
            ch->used_capacity++;

          // add the entrance port for this Switchbox
          switchSettings[sb].first = 
            std::make_pair(getConnectingBundle(ch->bundle), ch->used_capacity);
          // add the current Switchbox to the map of the predecessor
          switchSettings[&graph[sb->pred]].second.insert(
            std::make_pair(ch->bundle, ch->used_capacity));

          ch->used_capacity++;
          // if at capacity, bump demand to discourage using this Channel
          if(ch->used_capacity >= ch->max_capacity) {
            // this means the order matters!
            ch->demand *= 1.1;
          }

          sb->processed = true;
          curr = sb->pred;
          sb = &graph[curr];
        }
      }
      //add this flow to the proposed solution
      routing_solution[flow.first] = switchSettings;
    }
  } while (!isLegal()); // continue iterations until a legal routing is found
  return routing_solution;
}


// check that every channel does not exceed max capacity
bool Pathfinder::isLegal() {
  auto edge_pair = edges(graph);
  bool legal = true; // assume legal until found otherwise
  // check if maximum number of iterations has been reached
  if (maxIterReached)
    legal = false;
  for(edge_iterator e = edge_pair.first; e != edge_pair.second; e++) {
    if(graph[*e].used_capacity > graph[*e].max_capacity) {
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
