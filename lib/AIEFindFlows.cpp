//===- AIEFindFlows.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aie-find-flows"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

typedef std::pair<int, int> MaskValue;
typedef std::pair<Operation *, Port> PortConnection;
typedef std::pair<Port, MaskValue> PortMaskValue;
typedef std::pair<PortConnection, MaskValue> PacketConnection;

struct vertex {
  // defining characteristics
  Operation *op_data;
  Port port_data;
  bool isDestPort;
  // graph data
  void *parent_key;
  bool isRoot;
};

class Graph {

public:
  std::map<void *, vertex> vertex_map;

  // default root node constructor
  Graph() {}

  bool isNotEmpty() { return vertex_map.size() > 0; }

  void add_root(Operation *op, Port port) {
    vertex_map[(void *)op] = {op, port, false, 0, true};
  }

  void add_child(void *parent_key, Operation *op, Port port,
                 bool isDestination) {
    // In the case of the ConnectOp.destPort, due to the Op address being the
    // same as ConnectOp.sourcePort, we add 8bits to the ConnectOp address to
    // represent the destPort key. Due to the word length of the system being
    // greater than 8bits we can maintain a unique key for each sourcePort and
    // destPort for the connections.
    void *key;
    if (isDestination)
      key = (void *)op + 8;
    else
      key = (void *)op;

    vertex_map[key] = {op, port, isDestination, parent_key, false};
  }

  // int get_index(Operation *op, Port port, bool isDestination,
  //               bool checkBundleChannel) {
  //   int idx = -1;
  //   for (unsigned int i = 0; i < vertices.size(); i++) {
  //     if (vertices[i].op_data == op && vertices[i].port_data == port) {
  //       // bind to node that is lowest in the hierarchy when
  //       checkBundleChannel
  //       // false this is for cases when in the same switch source and
  //       // destination are the same
  //       if (checkBundleChannel && vertices[i].isDestination == isDestination)
  //         idx = i;
  //       else if (!checkBundleChannel)
  //         idx = i;
  //     }
  //   }
  //   return idx;
  // }

  // std::vector<int> get_path_to_root(int start_node) {
  //   std::vector<int> valid_path;
  //   int work = start_node;

  //   while (true) {
  //     valid_path.push_back(work);
  //     int parent_idx = vertices[work].parent_idx;
  //     if (parent_idx != -1) {
  //       work = parent_idx;
  //     } else {
  //       break;
  //     }
  //   }
  //   return valid_path;
  // }
};

class ConnectivityAnalysis {
  ModuleOp &module;

public:
  ConnectivityAnalysis(ModuleOp &m) : module(m) {}

private:
  llvm::Optional<PortConnection>
  getConnectionThroughWire(Operation *op, Port masterPort) const {

    LLVM_DEBUG(llvm::dbgs()
               << "Wire:" << *op << " " << stringifyWireBundle(masterPort.first)
               << " " << masterPort.second << "\n");
    for (auto wireOp : module.getOps<WireOp>()) {
      if (wireOp.source().getDefiningOp() == op &&
          wireOp.sourceBundle() == masterPort.first) {
        Operation *other = wireOp.dest().getDefiningOp();
        Port otherPort = std::make_pair(wireOp.destBundle(), masterPort.second);
        LLVM_DEBUG(llvm::dbgs() << "Connects To:" << *other << " "
                                << stringifyWireBundle(otherPort.first) << " "
                                << otherPort.second << "\n");
        return std::make_pair(other, otherPort);
      }
      if (wireOp.dest().getDefiningOp() == op &&
          wireOp.destBundle() == masterPort.first) {
        Operation *other = wireOp.source().getDefiningOp();
        Port otherPort =
            std::make_pair(wireOp.sourceBundle(), masterPort.second);
        LLVM_DEBUG(llvm::dbgs() << "Connects To:" << *other << " "
                                << stringifyWireBundle(otherPort.first) << " "
                                << otherPort.second << "\n");
        return std::make_pair(other, otherPort);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "*** Missing Wire!\n");
    return None;
  }

  std::vector<PacketConnection>
  getConnectionsThroughSwitchbox(Operation *op, Port sourcePort, Graph &g,
                                 std::set<Operation *> &connections_set) const {
    LLVM_DEBUG(llvm::dbgs() << "Switchbox:\n");

    Region *r;
    if (auto switchOp = dyn_cast_or_null<SwitchboxOp>(op))
      r = &switchOp.connections();
    else if (auto switchOp = dyn_cast_or_null<ShimMuxOp>(op))
      r = &switchOp.connections();
    else
      LLVM_DEBUG(llvm::dbgs()
                 << "*** Connection Terminated at unknown operation: \n");

    Block &b = r->front();
    std::vector<PacketConnection> opportSet;
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        // remove accessed connectOp from set
        connections_set.erase(connectOp);
        // add to graph if in detection mode and op is connectOp
        if (g.isNotEmpty()) {
          // implicit type converion connectOp -> Operation*
          g.add_child(op, connectOp, connectOp.sourcePort(), false);
          g.add_child(connectOp, connectOp, connectOp.destPort(), true);
        }

        MaskValue maskValue = std::make_pair(0, 0);
        PortConnection portconnection =
            std::make_pair(connectOp.getOperation(), connectOp.destPort());
        opportSet.push_back(std::make_pair(portconnection, maskValue));
        LLVM_DEBUG(llvm::dbgs()
                   << "To:" << stringifyWireBundle(connectOp.destPort().first)
                   << " " << connectOp.destPort().second << "\n");
      }
    }
    for (auto connectOp : b.getOps<PacketRulesOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Packet From: "
                   << stringifyWireBundle(connectOp.sourcePort().first) << " "
                   << (int)sourcePort.second << "\n");
        for (auto masterSetOp : b.getOps<MasterSetOp>())
          for (Value amsel : masterSetOp.amsels())
            for (auto ruleOp :
                 connectOp.rules().front().getOps<PacketRuleOp>()) {
              if (ruleOp.amsel() == amsel) {
                LLVM_DEBUG(llvm::dbgs()
                           << "To:"
                           << stringifyWireBundle(masterSetOp.destPort().first)
                           << " " << masterSetOp.destPort().second << "\n");
                MaskValue maskValue =
                    std::make_pair(ruleOp.maskInt(), ruleOp.valueInt());
                PortConnection portconnection = std::make_pair(
                    masterSetOp.getOperation(), masterSetOp.destPort());
                opportSet.push_back(std::make_pair(portconnection, maskValue));
              }
            }
      }
    }
    return opportSet;
  }

  std::vector<PacketConnection>
  maskSwitchboxConnections(Operation *switchOp, MaskValue maskValue,
                           std::vector<PacketConnection> nextOpPortMaskValues,
                           Graph &g, std::vector<void *> &antennaKey) const {
    std::vector<PacketConnection> worklist;
    for (auto &nextOpPortMaskValue : nextOpPortMaskValues) {
      Operation *op = nextOpPortMaskValue.first.first;
      Port nextPort = nextOpPortMaskValue.first.second;
      MaskValue nextMaskValue = nextOpPortMaskValue.second;
      int maskConflicts = nextMaskValue.first & maskValue.first;
      LLVM_DEBUG(llvm::dbgs() << "Mask: " << maskValue.first << " "
                              << maskValue.second << "\n");
      LLVM_DEBUG(llvm::dbgs() << "NextMask: " << nextMaskValue.first << " "
                              << nextMaskValue.second << "\n");
      LLVM_DEBUG(llvm::dbgs() << maskConflicts << "\n");

      if ((maskConflicts & nextMaskValue.second) !=
          (maskConflicts & maskValue.second)) {
        // Incoming packets cannot match this rule. Skip it.
        continue;
      }
      MaskValue newMaskValue = std::make_pair(
          maskValue.first | nextMaskValue.first,
          maskValue.second | (nextMaskValue.first & nextMaskValue.second));
      auto nextConnection = getConnectionThroughWire(switchOp, nextPort);

      // If there is no wire to follow then bail out.
      // then add to antenna list
      if (!nextConnection.hasValue()) {
        antennaKey.push_back((void *)op + 8);
        if (g.isNotEmpty()) // continue if in antenna detection mode
          continue;
        else
          break; // break to restart in antenna detection mode
      }

      // add to graph if in detection mode and op is connectOp
      if (g.isNotEmpty() && dyn_cast_or_null<ConnectOp>(op))
        g.add_child((void *)op + 8, nextConnection.getValue().first,
                    nextConnection.getValue().second, false);

      worklist.push_back(
          std::make_pair(nextConnection.getValue(), newMaskValue));
    }
    return worklist;
  }

public:
  // Get the tiles connected to the given tile, starting from the given
  // output port of the tile.  This is 1:N relationship because each
  // switchbox can broadcast.
  std::vector<PacketConnection>
  getConnectedTiles(TileOp tileOp, Port port, bool &antenna_detection,
                    std::set<Operation *> &connections_set) const {

    // create graph
    Graph g;
    // root node if in detection mode
    if (antenna_detection)
      g.add_root(tileOp.getOperation(), port);

    LLVM_DEBUG(llvm::dbgs()
               << "getConnectedTile(" << stringifyWireBundle(port.first) << " "
               << (int)port.second << ")");
    LLVM_DEBUG(tileOp.dump());

    // The accumulated result;
    std::vector<PacketConnection> connectedTiles;
    // A worklist of PortConnections to visit.  These are all input ports of
    // some object (likely either a TileOp or a SwitchboxOp).
    std::vector<PacketConnection> worklist;
    // Start the worklist by traversing from the tile to its connected
    // switchbox.
    std::vector<int> connectedTilesIndex;
    std::vector<void *> antennaKey;
    auto t = getConnectionThroughWire(tileOp.getOperation(), port);

    // If there is no wire to traverse, then just return no connection
    if (!t.hasValue())
      return connectedTiles;

    // add node to graph if in detection mode
    if (g.isNotEmpty())
      g.add_child(tileOp.getOperation(), t.getValue().first,
                  t.getValue().second, false);

    PacketConnection connection =
        std::make_pair(t.getValue(), std::make_pair(0, 0));
    worklist.push_back(connection);

    while (!worklist.empty()) {
      PacketConnection t = worklist.back();
      worklist.pop_back();
      PortConnection portConnection = t.first;
      MaskValue maskValue = t.second;
      Operation *other = portConnection.first;
      Port otherPort = portConnection.second;
      if (isa<FlowEndPoint>(other)) {
        // If we got to a tile, then add it to the result.
        connectedTiles.push_back(t);
        // connectedTilesIndex.push_back(
        //     g.get_index(other, otherPort, false, true));
      } else if (isa<Interconnect>(other)) {
        // append to graph included with method
        std::vector<PacketConnection> nextOpPortMaskValues =
            getConnectionsThroughSwitchbox(other, otherPort, g,
                                           connections_set);
        // append to graph included with method
        std::vector<PacketConnection> newWorkList = maskSwitchboxConnections(
            other, maskValue, nextOpPortMaskValues, g, antennaKey);
        // append to the worklist
        worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        if (nextOpPortMaskValues.size() > 0 && newWorkList.size() == 0) {
          // No rule matched some incoming packet.  This is likely a
          // configuration error.
          LLVM_DEBUG(llvm::dbgs() << "No rule matched incoming packet here: ");
          LLVM_DEBUG(other->dump());
        }
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "*** Connection Terminated at unknown operation: ");
        LLVM_DEBUG(other->dump());
      }
      // break to restart in antenna detection mode
      if (!antenna_detection && antennaKey.size() > 0) {
        antenna_detection = true;
        return connectedTiles;
      }
    }
    // detect_antenna(g, connectedTilesIndex, antennaIndex);

    for (auto &key : antennaKey) {
      void *work = key;
      while (true) {
        llvm::dbgs() << "Printing tree\n";
        g.vertex_map.at(work).op_data->dump();
        if (g.vertex_map.at(work).isRoot) {
          break;
        } else {
          work = g.vertex_map.at(work).parent_key;
        }
      }
    }

    return connectedTiles;
  }
};

std::set<Operation *> create_connections_set(ModuleOp m) {
  // Dangling island antennas refers to ConnectOps that do not originate from a
  // FlowEndPoint. To be able to detect these antennas we create a set with the
  // addresses of the ConnectOp. During traversal from a FlowEndPoint we remove
  // ConnectOps that have been visited. Thus, remaining ConnectOps in the set
  // are Dangling Island antennas. We note that for Dangling island antennas,
  // both sourcePort and destPort are part of the antenna as Dangling island
  // antennas do not have a sourcePort connection.

  std::set<Operation *> new_set;
  for (auto switchbox : m.getOps<SwitchboxOp>()) {
    Region &r = switchbox.connections();
    Block &b = r.front();
    for (auto connectOp : b.getOps<ConnectOp>()) {
      // implicit type converion connectOp -> Operation*
      new_set.insert(connectOp);
    }
  }

  return new_set;
}

static void findFlowsFrom(AIE::TileOp op, ConnectivityAnalysis &analysis,
                          OpBuilder &rewriter,
                          std::set<Operation *> &connections_set) {
  Operation *Op = op.getOperation();
  rewriter.setInsertionPointToEnd(Op->getBlock());

  std::vector<WireBundle> bundles = {WireBundle::Core, WireBundle::DMA};
  for (WireBundle bundle : bundles) {
    for (int i = 0; i < op.getNumSourceConnections(bundle); i++) {
      bool antenna_detection = false;
      std::vector<PacketConnection> tiles;
      tiles = analysis.getConnectedTiles(op, std::make_pair(bundle, i),
                                         antenna_detection, connections_set);
      if (antenna_detection) { // get connected Tiles with antenna detection
        tiles = analysis.getConnectedTiles(op, std::make_pair(bundle, i),
                                           antenna_detection, connections_set);
      }
      LLVM_DEBUG(llvm::dbgs() << tiles.size() << " Flows\n");

      for (PacketConnection &c : tiles) {
        PortConnection portConnection = c.first;
        MaskValue maskValue = c.second;
        Operation *destOp = portConnection.first;
        Port destPort = portConnection.second;
        if (maskValue.first == 0) {
          rewriter.create<FlowOp>(Op->getLoc(), Op->getResult(0), bundle, i,
                                  destOp->getResult(0), destPort.first,
                                  destPort.second);
        } else {
          PacketFlowOp flowOp =
              rewriter.create<PacketFlowOp>(Op->getLoc(), maskValue.second);
          flowOp.ensureTerminator(flowOp.ports(), rewriter, Op->getLoc());
          OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
          rewriter.setInsertionPoint(flowOp.ports().front().getTerminator());
          rewriter.create<PacketSourceOp>(Op->getLoc(), Op->getResult(0),
                                          bundle, (int)i);
          rewriter.create<PacketDestOp>(Op->getLoc(), destOp->getResult(0),
                                        destPort.first, (int)destPort.second);
          rewriter.restoreInsertionPoint(ip);
        }
      }

      llvm::dbgs() << "\n\n\n";
    }
  }
}

struct AIEFindFlowsPass : public AIEFindFlowsBase<AIEFindFlowsPass> {
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }
  void runOnOperation() override {

    ModuleOp m = getOperation();
    ConnectivityAnalysis analysis(m);

    std::set<Operation *> connections_set = create_connections_set(m);

    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());
    for (auto tile : m.getOps<TileOp>()) {
      findFlowsFrom(tile, analysis, builder, connections_set);
    }

    // emit error when connections_set is not empty
    for (auto &k : connections_set) {
      Operation *parentops = k->getParentOp();
      k->emitWarning("Dangling Island Antenna\n")
          .attachNote(parentops->getLoc());
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> xilinx::AIE::createAIEFindFlowsPass() {
  return std::make_unique<AIEFindFlowsPass>();
}