//===- AIEObjectFifoStatefulTransform.cpp ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: October 18th 2021
//
//===----------------------------------------------------------------------===//

// Current limitations/To-Dos of this pass:
// - Does not support packet switching
// - Is only concerned with dmaBd routes. Routes straight from the core
//   can still lead nowhere/be disconnected.

#include <map>
#include <stack>
#include <vector>
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

//
// Utilities
//
template<class opT> 
bool regionContains(mlir::Region& r)
{
  return r.template op_begin<opT>() != r.template op_end<opT>();
}

// 
// Verify BDs Connected Pass
// 
struct AIEVerifyConnectionsPass 
	: public AIEVerifyConnectionsBase<AIEVerifyConnectionsPass>
{
private:

  struct endpoint {
    const TileID tile;
    const WireBundle bundle;
    const uint32_t channel;
    bool operator==(const endpoint &other) const {
      return tile.first == other.tile.first
             && tile.second == other.tile.second
             && bundle == other.bundle
             && channel == other.channel;
    }
    bool operator<(const endpoint &other) const {
      return (tile.first < other.tile.first
              || (tile.first == other.tile.first 
                  && (tile.second < other.tile.second
                      || (tile.second == other.tile.second
                          && (bundle < other.bundle
                              || (bundle == other.bundle 
                                  && channel < other.channel))))));
    }
  };

  struct link {
    endpoint src;
    endpoint dest;
    ConnectOp op;
  };

  typedef std::multimap<const endpoint, link> graph;

  // This analysis does not yet support packet-switched communications;
  bool analysisIsSupported(DeviceOp& device, bool emitWarnings = true) {
    bool ret = true;  // Don't return early; make sure we generate all warnings.
    Region &region = device.getRegion();
    for(SwitchboxOp switchbox : region.getOps<SwitchboxOp>()) {
      Region &switchboxRegion = switchbox.getRegion();
      if(regionContains<AMSelOp>(switchboxRegion)
        || regionContains<MasterSetOp>(switchboxRegion)
        || regionContains<PacketRulesOp>(switchboxRegion)
        || regionContains<PacketRuleOp>(switchboxRegion)) {
        switchbox->emitWarning() << getPassName() << " currently does not "
                              "support code containing packet-switched routing";
        ret = false;
      }
    }
    if(regionContains<FlowOp>(region)
       || regionContains<ObjectFifoCreateOp>(region))
    {
      device->emitWarning() << getPassName() << " must be applied after "
                            "lowering to switchboxes. Analysis does not work "
                            "higher abstraction levels.";
      ret = false;
    }
    return ret;
  }

  std::pair<short, short> getWireBundleTileOffset(const WireBundle bundle) const
  {
    switch(bundle) {
      case WireBundle::North: {
        return std::make_pair(0, 1);
      }
      case WireBundle::East: {
        return std::make_pair(1, 0);
      }
      case WireBundle::South: {
        return std::make_pair(0, -1);
      }
      case WireBundle::West: {
        return std::make_pair(-1, 0);
      }
      default: { // Tile-local connection to DMA or core
        return std::make_pair(0, 0);
      }
    }
  }

  WireBundle getMirroredWireBundle(const WireBundle outgoing) const {
    // How to read the following:
    // "If it goes out `outgoing` at the source, it must come in at
    //  `return value` at the destination."
    switch(outgoing) {
      case WireBundle::North: {
        return WireBundle::South;
      }
      case WireBundle::East: {
        return WireBundle::West;
      }
      case WireBundle::South: {
        return WireBundle::North;
      }
      case WireBundle::West: {
        return WireBundle::East;
      }
      default: {

      }
    }
    return WireBundle::North; // FIXME
  }

  void buildConnectionGraph(Region &region, 
                            graph &outgoing_edges, 
                            graph &incoming_edges) {
    for(SwitchboxOp switchbox : region.getOps<SwitchboxOp>()) {
      TileID tile = switchbox.getTileID();
      Region &switchbox_region = switchbox.getRegion();
      for(ConnectOp connect : switchbox_region.getOps<ConnectOp>()) {
        endpoint src_endpoint {tile, connect.getSourceBundle(), 
                               connect.getSourceChannel()};
        endpoint dst_endpoint {tile, connect.getDestBundle(),   
                               connect.getDestChannel()};
        link l = {src_endpoint, dst_endpoint, connect}; // will be copied
        incoming_edges.emplace(src_endpoint, l);
        outgoing_edges.emplace(dst_endpoint, l);
      }
    }
  }

  // This only works for North, East, South, West endpoints, as the opposite
  // end of a DMA, or Core is not uniquely defined (could be any of N, E, S, W)
  endpoint getOppositeEnd(const endpoint &src) {
    std::pair<short, short> dst_offset = getWireBundleTileOffset(src.bundle);
    TileID dst_tile = std::make_pair(src.tile.first + dst_offset.first,
                                      src.tile.second + dst_offset.second);
    WireBundle dst_bundle = getMirroredWireBundle(src.bundle);
    const endpoint dst {dst_tile, dst_bundle, src.channel};
    return dst;
  }

  void verifyAllConnectionsHaveMatchingEnds(graph &outgoing_edges, 
                                            graph &incoming_edges) {
    
    // Verify we find an incoming edge in all switchboxes for all outoging 
    // connections.
    for(auto &edge: outgoing_edges) {
      const endpoint &src = edge.first;
      link &link = edge.second;
      const endpoint dst = getOppositeEnd(src);
      if(src.tile == dst.tile) {
        // This is a tile-local connection to DMA or Core
        // TODO: see if there are any checks that should be performed here.
        continue;
      }
      if(0 == src.tile.second || 0 == dst.tile.second) {
        // Connection in/into the shim row. For now we don't check those.
        // TODO: Check these.
        continue;
      }
      if(0 > dst.tile.first || 0 > dst.tile.second) {
        // TODO: Also verify upper limit.
        link.op.emitError() << "Connection to tile (" << dst.tile.first << ", "
                            << dst.tile.second << ") is out of range.\n";
        signalPassFailure();
      }
      int incoming_count = incoming_edges.count(dst);
      if(incoming_count == 0) {
        link.op.emitError() << "There is no matching incoming edge in tile (" 
                            << dst.tile.first << ", " << dst.tile.second << ") "
                            "for this outgoing edge.";
        signalPassFailure();
      } else if(incoming_count < 1) {
        link.op.emitWarning() << "Duplicate incoming edges at the destination "
                                 "tile (" << dst.tile.first << ", " 
                                 << dst.tile.second << ") found.";
      }
    }
  }

  void verifyDMAsAreConnected(TileID tile, Region &region, 
                              graph &outgoing_edges, graph &incoming_edges) {
    // It really should not matter for this function whether we look at 
    // the outgoing_edges or incoming_edges graph, since the edge will have to
    // be internal to the tile (e.g. North -> DMA) and thus is guaranteed
    // to be present in both graphs (if the graph building routine above is 
    // correct).
    for(DMAStartOp dma : region.getOps<DMAStartOp>()) {
      DMAChannelDir dir = dma.getChannelDir();
      uint32_t channel = dma.getChannelIndex();
      endpoint dma_end {tile, WireBundle::DMA, channel};
      if(dir == DMAChannelDir::S2MM) {
        // There must be a connection going _into_ the DMA.
        if(outgoing_edges.find(dma_end) == outgoing_edges.end()) {
          dma.emitError() << "S2MM DMA defined, but no incoming connections "
                             "(AIE.connect()) to the DMA are defined.\n";
          signalPassFailure();
        }
      } else { // MM2S
        // There must be a connection going _out of_ the DMA.
        if(incoming_edges.find(dma_end) == incoming_edges.end()) {
          dma.emitError() << "MM2S DMA defined, but no connections out of the "
                             "DMA are defined.\n";
          signalPassFailure();
        }
      }
    }
  }

  void verifyContainsNoCycles(graph &incoming_edges)
  {
    // We will start a depth-first traversal from each node in this set, unless
    // a prior traversal already visited the node. This is used to explore all
    // disconnected components.
    std::set<std::pair<endpoint, ConnectOp &>> unexplored = {};
    for(auto node : incoming_edges) {
      unexplored.emplace(node.first, node.second.op);
    }
    // Perform a DFS for each of the components.
    while(unexplored.size() > 0) {
      auto start = *unexplored.begin();
      unexplored.erase(start);
      std::set<endpoint> visited = {};
      std::stack<std::pair<endpoint, ConnectOp &>> todo = {};
      todo.push(start);
      while(todo.size() > 0) {
        auto current = todo.top();
        endpoint current_endpoint = current.first;
        ConnectOp &current_op = current.second;
        todo.pop();

        if(visited.count(current.first) > 0) {
          // We've already been here -- cycle!
          current_op.emitError() << "There is a cycle in the route containing "
                                    "this connection.";
          signalPassFailure();
          return;
        }
        unexplored.erase(current);
        visited.insert(current_endpoint);

        // Add all connected nodes to visit next. Conceptually we make two
        // "hops" here: (1) Within the tile, we hop from the incoming port to
        // the outgoing port (i.e. the connection that AIE.connect(src, dst))
        // describes, and (2) we hop across tiles, to the neighboring tile
        // according to the output port (e.g. X+1 if the output port is East).
        auto children_range = incoming_edges.equal_range(current_endpoint);
        for(auto child_edge = children_range.first; 
            child_edge != children_range.second; ++child_edge) {
          endpoint hop_1_dst = child_edge->second.dest;
          endpoint hop_2_dst = getOppositeEnd(hop_1_dst);
          if(hop_1_dst.tile == hop_2_dst.tile) {
            // This is a local X -> DMA or X -> Core connection.
            continue;
          }
          if(incoming_edges.count(hop_2_dst) == 0) {
            // Only add next hops if they contain further connections, otherwise
            // we can stop exploring early here. Note that the 
            // verifyAllConnectionsHaveMatchingEnds pass should make sure this
            // case never happens.
            continue;
          }
          todo.emplace(hop_2_dst, child_edge->second.op);
        }
      }
    }
  }

public :
  void runOnOperation() override {
    graph outgoing_edges,  // map of AIE.connect()s, indexed by the dest port
          incoming_edges;  // map of AIE.connect()s, indexed by the src port
    DeviceOp device = getOperation();
    Region &region = device.getRegion();
    if(!analysisIsSupported(device)) {
      return;
    }
    // First, iterate over child operations to build our connectivity graph.
    buildConnectionGraph(region, outgoing_edges, incoming_edges);
    verifyAllConnectionsHaveMatchingEnds(outgoing_edges, incoming_edges);
    // Now, check if there are any BDs that are not connected to a source, and
    // check if there are BDs that are streaming with no one listening.
    for(MemOp mem : region.getOps<MemOp>()) {
      verifyDMAsAreConnected(mem.getTileID(), mem.getRegion(), 
                             outgoing_edges, incoming_edges);
    }
    // Lastly, check for cycles in the routes.
    verifyContainsNoCycles(incoming_edges);
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEVerifyConnectionsPass() {
  return std::make_unique<AIEVerifyConnectionsPass>();
}
