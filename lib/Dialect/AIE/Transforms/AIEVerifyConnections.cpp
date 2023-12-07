//===- AIEVerifyConnections.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

/*
This pass verifies that the connections between tiles are terminated at both
ends and contain no cycles.

This pass does not transform the MLIR whatsoever; it is purely analysis. It
issues errors if there are unterminated connections or cycles. This is done at
the switchbox connection (`AIE.connect`) and DMA (`AIE.dmaStart`) levels.

Specifically, this pass issues the following new errors after analysis:

- Cycle errors, along with some connection along the cycle.
   ```
    error: There is a cycle in the route containing this connection.
        AIE.connect<"North" : 0, "West" : 0>
    ````
- No outgoing connection errors: If there is a switchbox configured with an
  incoming connection from N, E, S, W direction, but in its neighboring tile's
  switchbox, there is no outgoing connection from S, W, N, E direction,
  respectively.
    ```
    error: There is no matching outgoing connection for <"East" : 2> in tile
    (0, 1) for this incoming connection.
        AIE.connect<"West" : 2, "DMA" : 1>
    ```

- No incoming connection errors: Essentially the analog of above, if a switchbox
  is configured to send traffic out towards N, E, S, W, but its neighboring tile
  is not configured to accept traffic from S, W, N, E, an error is issued.
    ````
    error: There is no matching incoming connection for <"West" : 1> in tile
    (1, 1) for this outgoing connection.
        AIE.connect<"DMA" : 1, "East" : 1>
    ````

- No connection for DMAs: If a MM2S DMA is configured, but no switchbox
   connection DMA->X is configured, or if a S2MM, but no switchbox connection
   X->DMA is configured, an error is issued.
    ```
    error: S2MM DMA defined, but no incoming connections to the DMA are defined.
        %dma = AIE.dmaStart("S2MM", 0, ^bd, ^end)
   ```

There are tests in `test/verify-connections` that demonstrate all of the above
examples.

Current limitations:
- Does not check if there is a matching configured DMA if a X->DMA
  `AIE.connect` is issued. (Essentially the opposite of the last error above.)
- Does not support packet based routing, only circuit switched. A warning is
  issued by the pass if packet based routing is used and the pass is invoked.
- Currently does not verify routing in the shim row.
*/

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <map>
#include <stack>
#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

//
// Utilities
//
template <class opT>
bool regionContains(mlir::Region &r) {
  return r.template op_begin<opT>() != r.template op_end<opT>();
}

//
// Verify BDs Connected Pass
//
struct AIEVerifyConnectionsPass
    : public AIEVerifyConnectionsBase<AIEVerifyConnectionsPass> {

private:
  struct endpoint {
    const TileID tile;
    const WireBundle bundle;
    const int32_t channel;
    bool operator==(const endpoint &other) const {
      return tile.col == other.tile.col && tile.row == other.tile.row &&
             bundle == other.bundle && channel == other.channel;
    }
    bool operator<(const endpoint &other) const {
      return (tile.col < other.tile.col ||
              (tile.col == other.tile.col &&
               (tile.row < other.tile.row ||
                (tile.row == other.tile.row &&
                 (bundle < other.bundle ||
                  (bundle == other.bundle && channel < other.channel))))));
    }
  };

  struct link {
    endpoint src;
    endpoint dest;
    ConnectOp op;
  };

  typedef std::multimap<const endpoint, link> graph;

  // This analysis does not yet support packet-switched communications;
  bool analysisIsSupported(DeviceOp &device, bool emitWarnings = true) {
    bool ret = true; // Don't return early; make sure we generate all warnings.
    Region &region = device.getRegion();
    for (SwitchboxOp switchbox : region.getOps<SwitchboxOp>()) {
      Region &switchboxRegion = switchbox.getRegion();
      if (regionContains<AMSelOp>(switchboxRegion) ||
          regionContains<MasterSetOp>(switchboxRegion) ||
          regionContains<PacketRulesOp>(switchboxRegion) ||
          regionContains<PacketRuleOp>(switchboxRegion)) {
        switchbox->emitWarning() << getPassName()
                                 << " currently does not "
                                    "support code containing packet-switched "
                                    "routing.";
        ret = false;
      }
    }
    if (regionContains<FlowOp>(region) ||
        regionContains<ObjectFifoCreateOp>(region)) {
      device->emitWarning()
          << getPassName()
          << " must be applied after "
             "lowering to switchboxes. Analysis does not work "
             "higher abstraction levels.";
      ret = false;
    }
    return ret;
  }

  std::pair<short, short>
  getWireBundleTileOffset(const WireBundle bundle) const {
    switch (bundle) {
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
    // (And vice versa.)
    switch (outgoing) {
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
      // This is not 100% accurate; it could also be DMA.
      // This function is only really useful for N, E, S, W inputs.
      return WireBundle::Core;
    }
    }
  }

  void buildConnectionGraph(Region &region, graph &outgoing_edges,
                            graph &incoming_edges) {
    for (SwitchboxOp switchbox : region.getOps<SwitchboxOp>()) {
      TileID tile = switchbox.getTileID();
      Region &switchbox_region = switchbox.getRegion();
      for (ConnectOp connect : switchbox_region.getOps<ConnectOp>()) {
        endpoint src_endpoint{tile, connect.getSourceBundle(),
                              connect.getSourceChannel()};
        endpoint dst_endpoint{tile, connect.getDestBundle(),
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
    TileID dst_tile = {src.tile.col + dst_offset.first,
                       src.tile.row + dst_offset.second};
    WireBundle dst_bundle = getMirroredWireBundle(src.bundle);
    const endpoint dst{dst_tile, dst_bundle, src.channel};
    return dst;
  }

  void verifyAllConnectionsHaveMatchingEnds(graph &outgoing_edges,
                                            graph &incoming_edges) {

    // Verify we find an incoming edge in all switchboxes for all outoging
    // connections.
    for (auto &edge : outgoing_edges) {
      const endpoint &src = edge.first;
      link &link = edge.second;
      const endpoint dst = getOppositeEnd(src);
      assert(0 <= dst.tile.col && 0 <= dst.tile.row);
      if (src.tile == dst.tile) {
        // This is a tile-local connection to DMA or Core
        // TODO: Verify there is a DMA that uses this connection.
        continue;
      }
      if (0 == src.tile.row || 0 == dst.tile.row) {
        // Connection in/into the shim row. For now we don't check those.
        // TODO: Check these.
        continue;
      }
      if (incoming_edges.count(dst) == 0) {
        link.op.emitError()
            << "There is no matching incoming connection for "
               "<\""
            << stringifyWireBundle(dst.bundle) << "\" : " << dst.channel
            << ">"
               " in tile "
               "("
            << dst.tile.col << ", " << dst.tile.row
            << ")"
               " for this outgoing connection.";
        signalPassFailure();
      }
    }

    // Opposite of above: Verify all incoming edges are matched by an outgoing
    // edge on the opposite end.
    for (auto &edge : incoming_edges) {
      const endpoint &src = edge.first;
      link &link = edge.second;
      const endpoint dst = getOppositeEnd(src);
      assert(0 <= dst.tile.col && 0 <= dst.tile.row);
      if (src.tile == dst.tile) {
        continue;
      }
      if (0 == src.tile.row || 0 == dst.tile.row) {
        continue; // Shim row
      }
      if (outgoing_edges.count(dst) == 0) {
        link.op.emitError()
            << "There is no matching outgoing connection for "
               "<\""
            << stringifyWireBundle(dst.bundle) << "\" : " << dst.channel
            << ">"
               " in tile "
               "("
            << dst.tile.col << ", " << dst.tile.row
            << ")"
               " for this incoming connection.";
        signalPassFailure();
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
    for (DMAStartOp dma : region.getOps<DMAStartOp>()) {
      DMAChannelDir dir = dma.getChannelDir();
      int32_t channel = dma.getChannelIndex();
      endpoint dma_end{tile, WireBundle::DMA, channel};
      if (dir == DMAChannelDir::S2MM) {
        // There must be a connection going _into_ the DMA.
        if (outgoing_edges.find(dma_end) == outgoing_edges.end()) {
          dma.emitError() << "S2MM DMA defined, but no incoming connections "
                             "to the DMA are defined.";
          signalPassFailure();
        }
      } else { // MM2S
        // There must be a connection going _out of_ the DMA.
        if (incoming_edges.find(dma_end) == incoming_edges.end()) {
          dma.emitError() << "MM2S DMA defined, but no outgoing connections "
                             "out of the DMA are defined.";
          signalPassFailure();
        }
      }
    }
  }

  void verifyContainsNoCycles(graph &incoming_edges) {
    // We will start a depth-first traversal from each node in this set, unless
    // a prior traversal already visited the node. This is used to explore all
    // disconnected components.
    std::set<std::pair<endpoint, ConnectOp &>> unexplored = {};
    for (auto node : incoming_edges) {
      unexplored.emplace(node.first, node.second.op);
    }
    // Perform a DFS for each of the components.
    while (unexplored.size() > 0) {
      auto start = *unexplored.begin();
      unexplored.erase(start);
      std::set<endpoint> visited = {};
      std::stack<std::pair<endpoint, ConnectOp &>> todo = {};
      todo.push(start);
      while (todo.size() > 0) {
        auto current = todo.top();
        endpoint current_endpoint = current.first;
        ConnectOp &current_op = current.second;
        todo.pop();

        if (visited.count(current.first) > 0) {
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
        for (auto child_edge = children_range.first;
             child_edge != children_range.second; ++child_edge) {
          endpoint hop_1_dst = child_edge->second.dest;
          endpoint hop_2_dst = getOppositeEnd(hop_1_dst);
          if (hop_1_dst.tile == hop_2_dst.tile) {
            // This is a local X -> DMA or X -> Core connection.
            continue;
          }
          if (incoming_edges.count(hop_2_dst) == 0) {
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

public:
  void runOnOperation() override {
    graph outgoing_edges, // map of AIE.connect()s, indexed by the dest port
        incoming_edges;   // map of AIE.connect()s, indexed by the src port
    DeviceOp device = getOperation();
    Region &region = device.getRegion();
    if (!analysisIsSupported(device)) {
      return;
    }
    // First, iterate over child operations to build our connectivity graph.
    buildConnectionGraph(region, outgoing_edges, incoming_edges);
    verifyAllConnectionsHaveMatchingEnds(outgoing_edges, incoming_edges);
    // Now, check if there are any BDs that are not connected to a source, and
    // check if there are BDs that are streaming with no one listening.
    for (MemOp mem : region.getOps<MemOp>()) {
      verifyDMAsAreConnected(mem.getTileID(), mem.getRegion(), outgoing_edges,
                             incoming_edges);
    }
    // Lastly, check for cycles in the routes.
    verifyContainsNoCycles(incoming_edges);
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEVerifyConnectionsPass() {
  return std::make_unique<AIEVerifyConnectionsPass>();
}
