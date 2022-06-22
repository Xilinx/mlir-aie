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
  int parent_idx = -1;
  bool isLeaf = true;
  bool isDestination = false;

  Operation *op_data;
  Port port_data;

  vertex(Operation *op, Port port) {
    op_data = op;
    port_data = port;
  }
};

class Graph {
public:
  std::vector<vertex> vertices;
  // default root node constructor
  Graph(Operation *op, Port port) { vertices.push_back(vertex(op, port)); }

  //(parent-op, parent-port, child-op, child-port)
  void add_child(Operation *op_p, Port port_p, Operation *op_c, Port port_c,
                 bool isDestination) {
    int idx = get_index(op_p, port_p, isDestination, false);
    if (idx == -1) {
      LLVM_DEBUG(llvm::dbgs() << "No parent in graph\n");
      return;
    }
    vertices.push_back(vertex(op_c, port_c));      // add new node
    vertices.back().parent_idx = idx;              // link parent
    vertices[idx].isLeaf = false;                  // link child
    vertices.back().isDestination = isDestination; // switch connection type
  }

  int get_index(Operation *op, Port port, bool isDestination,
                bool checkBundleChannel) {
    int idx = -1;
    for (unsigned int i = 0; i < vertices.size(); i++) {
      if (vertices[i].op_data == op && vertices[i].port_data == port) {
        // bind to node that is lowest in the hierarchy when checkBundleChannel
        // false this is for cases when in the same switch source and
        // destination are the same
        if (checkBundleChannel && vertices[i].isDestination == isDestination)
          idx = i;
        else if (!checkBundleChannel)
          idx = i;
      }
    }
    return idx;
  }

  std::vector<int> get_path_to_root(int start_node) {
    std::vector<int> valid_path;
    int work = start_node;

    while (true) {
      valid_path.push_back(work);
      int parent_idx = vertices[work].parent_idx;
      if (parent_idx != -1) {
        work = parent_idx;
      } else {
        break;
      }
    }
    return valid_path;
  }
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

  std::vector<PortMaskValue>
  getConnectionsThroughSwitchbox(Graph &g, Operation *op,
                                 Port sourcePort) const {
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
    std::vector<PortMaskValue> portSet;
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        MaskValue maskValue = std::make_pair(0, 0);
        portSet.push_back(std::make_pair(connectOp.destPort(), maskValue));
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
                   << (int)sourcePort.first << "\n");
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
                portSet.push_back(
                    std::make_pair(masterSetOp.destPort(), maskValue));
              }
            }
      }
    }
    // append to graph
    for (auto &port : portSet) {
      g.add_child(op, sourcePort, op, port.first, true);
    }
    return portSet;
  }

  std::vector<PacketConnection>
  maskSwitchboxConnections(Graph &g, std::vector<int> &antennaIndex,
                           Operation *switchOp,
                           std::vector<PortMaskValue> nextPortMaskValues,
                           MaskValue maskValue) const {
    std::vector<PacketConnection> worklist;
    for (auto &nextPortMaskValue : nextPortMaskValues) {
      Port nextPort = nextPortMaskValue.first;
      MaskValue nextMaskValue = nextPortMaskValue.second;
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
        antennaIndex.push_back(g.get_index(switchOp, nextPort, true, true));
        continue;
      }

      g.add_child(switchOp, nextPort, nextConnection.getValue().first,
                  nextConnection.getValue().second, false);
      worklist.push_back(
          std::make_pair(nextConnection.getValue(), newMaskValue));
    }
    return worklist;
  }

  void detect_antenna(Graph &g, std::vector<int> connectedTilesIndexList,
                      std::vector<int> antennaIndexList) const {

    std::vector<int> path_buffer;
    // create valid path from connected tiles
    std::vector<int> valid_path;
    for (auto &tile_idx : connectedTilesIndexList) {
      if (valid_path.empty()) {
        valid_path = g.get_path_to_root(tile_idx);
      } else {
        path_buffer = g.get_path_to_root(tile_idx);
        for (auto &node_idx : path_buffer) {
          if (!std::count(valid_path.begin(), valid_path.end(), node_idx)) {
            valid_path.push_back(node_idx);
          }
        }
      }
    }

    // output antenna path
    for (auto &antenna_idx : antennaIndexList) {
      std::vector<int> antenna_valid_path;
      std::vector<int> antenna_nonvalid_path;
      path_buffer = g.get_path_to_root(antenna_idx);
      for (auto &node_idx : path_buffer) {
        if (std::count(valid_path.begin(), valid_path.end(), node_idx)) {
          antenna_valid_path.push_back(node_idx);
        } else {
          antenna_nonvalid_path.push_back(node_idx);
        }
      }
      // emit warning message for antennas
      for (auto &v : antenna_nonvalid_path) {
        std::string connectionType = "";
        Operation *op = g.vertices[v].op_data;
        if (dyn_cast_or_null<SwitchboxOp>(op)) {
          if (g.vertices[v].isDestination)
            connectionType = "Connection Destination";
          else
            connectionType = "Connection Source";
        }
        op->emitWarning() << "Antenna\n"
                          << "at Port: "
                          << "("
                          << stringifyWireBundle(g.vertices[v].port_data.first)
                          << " " << (int)g.vertices[v].port_data.second << ") "
                          << connectionType << "\n";
      }
      // emit remarks for antenna traceback
      for (auto &v : antenna_valid_path) {
        std::string connectionType = "";
        Operation *op = g.vertices[v].op_data;
        if (dyn_cast_or_null<SwitchboxOp>(op)) {
          if (g.vertices[v].isDestination)
            connectionType = "Connection Destination";
          else
            connectionType = "Connection Source";
        }
        op->emitRemark() << "Traceback\n"
                         << "at Port: "
                         << "("
                         << stringifyWireBundle(g.vertices[v].port_data.first)
                         << " " << (int)g.vertices[v].port_data.second << ") "
                         << connectionType << "\n";
      }
    }
  }

public:
  // Get the tiles connected to the given tile, starting from the given
  // output port of the tile.  This is 1:N relationship because each
  // switchbox can broadcast.
  std::vector<PacketConnection> getConnectedTiles(TileOp tileOp,
                                                  Port port) const {

    // create graph and add root node
    Graph g(tileOp.getOperation(), port);

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
    std::vector<int> antennaIndex;
    auto t = getConnectionThroughWire(tileOp.getOperation(), port);

    // If there is no wire to traverse, then just return no connection
    if (!t.hasValue())
      return connectedTiles;

    // add node to graph
    g.add_child(tileOp.getOperation(), port, t.getValue().first,
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
        connectedTilesIndex.push_back(
            g.get_index(other, otherPort, false, true));
      } else if (isa<Interconnect>(other)) {
        // append to graph included with method
        std::vector<PortMaskValue> nextPortMaskValues =
            getConnectionsThroughSwitchbox(g, other, otherPort);
        // append to graph included with method
        std::vector<PacketConnection> newWorkList = maskSwitchboxConnections(
            g, antennaIndex, other, nextPortMaskValues, maskValue);
        // append to the worklist
        worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        // worklist.insert(worklist.end(), newWorkList.begin(),
        // newWorkList.end());
        if (nextPortMaskValues.size() > 0 && newWorkList.size() == 0) {
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
    }
    detect_antenna(g, connectedTilesIndex, antennaIndex);

    return connectedTiles;
  }
};

static void findFlowsFrom(AIE::TileOp op, ConnectivityAnalysis &analysis,
                          OpBuilder &rewriter) {
  Operation *Op = op.getOperation();
  rewriter.setInsertionPointToEnd(Op->getBlock());

  std::vector<WireBundle> bundles = {WireBundle::Core, WireBundle::DMA};
  for (WireBundle bundle : bundles) {
    for (int i = 0; i < op.getNumSourceConnections(bundle); i++) {
      std::vector<PacketConnection> tiles =
          analysis.getConnectedTiles(op, std::make_pair(bundle, i));
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

    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());
    for (auto tile : m.getOps<TileOp>()) {
      findFlowsFrom(tile, analysis, builder);
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> xilinx::AIE::createAIEFindFlowsPass() {
  return std::make_unique<AIEFindFlowsPass>();
}