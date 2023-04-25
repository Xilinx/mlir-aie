//===- AIEPathfinder.h ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PATHFINDER_H
#define AIE_PATHFINDER_H

#include <algorithm>
#include <limits>
#include <utility> //for std::pair
#include <vector>
#include <memory>

// builds against at least boost graph 1.7.1
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedef"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#pragma GCC diagnostic pop

#include "aie/Dialect/AIE/IR/AIEDialect.h" // for WireBundle and Port

namespace xilinx {
namespace AIE {

using namespace boost;

struct Switchbox { // acts as a vertex
  unsigned short col, row;
  // int dist;
  unsigned int pred; // predecessor for dijkstra's
  bool processed;    // denotes this switchbox has already been processed
};

struct Channel { // acts as an edge
  float demand;  // indicates how many flows want to use this Channel
  unsigned short
      used_capacity;           // how many flows are actually using this Channel
  unsigned short max_capacity; // maximum number of routing resources
  std::set<short> fixed_capacity;     // channels not available to the algorithm
  unsigned short over_capacity_count; // history of Channel being over capacity
  WireBundle bundle;
};

// create a graph type that uses Switchboxes as vertices and Channels as edges
typedef adjacency_list<vecS, vecS, bidirectionalS, Switchbox, Channel>
    SwitchboxGraph;

typedef graph_traits<SwitchboxGraph>::vertex_descriptor vertex_descriptor;
typedef graph_traits<SwitchboxGraph>::edge_descriptor edge_descriptor;
typedef graph_traits<SwitchboxGraph>::vertex_iterator vertex_iterator;
typedef graph_traits<SwitchboxGraph>::edge_iterator edge_iterator;
typedef graph_traits<SwitchboxGraph>::in_edge_iterator in_edge_iterator;

typedef std::pair<int, int> Coord;
// A SwitchSetting defines the required settings for a Switchbox for a flow
// Trying out a std::tuple!
// std::get<0>(SwitchSetting) is the incoming signal
// std::get<1>(SwitchSetting) is the fanout
// std::get<2>(SwitchSetting) is the packet ID (set to -1 for circuit switched) 

typedef std::tuple<Port, std::set<Port>, int> SwitchConnection;
//typedef std::tuple<Port, SmallVector<Port>, int> SwitchConnection;
typedef std::tuple<Port, std::set<Port>, int> SwitchSetting; // TODO: change to SwitchConnection
typedef DenseMap<Switchbox *, SwitchConnection> SwitchSettings;

// A Flow defines source and destination vertices
// Only one source, but any number of destinations (fanout)
typedef std::pair<Switchbox *, Port> PathEndPoint;
typedef std::tuple<PathEndPoint, SmallVector<PathEndPoint>, int> Flow;

class Pathfinder {
private:
  SwitchboxGraph graph;
  std::vector<Flow*> flows;
  bool maxIterReached;

public:
  Pathfinder();
  Pathfinder(int maxcol, int maxrow);
  void initializeGraph(int maxcol, int maxrow);
  void addFlow(Coord srcCoords, Port srcPort, 
                Coord dstCoords, Port dstPort, int flow_id=-1);
  void addFixedConnection(Coord coord, Port port);
  bool isLegal();

  void findPaths(
          DenseMap<Flow*, SwitchSettings*> & flow_solutions,
         const int MAX_ITERATIONS = 1000);

  Switchbox *getSwitchbox(TileID coords) {
    auto vpair = vertices(graph);
    Switchbox *sb;
    for (vertex_iterator v = vpair.first; v != vpair.second; v++) {
      sb = &graph[*v];
      if (sb->col == coords.first && sb->row == coords.second)
        return sb;
    }
    return nullptr;
  }

  // Printing functions for debugging
  void printFlows();
  static void printFlow(Flow*);
  static void printSwitchSettings(SwitchSettings*);
};

} // namespace AIE
} // namespace xilinx
#endif
