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

class Graph {
  // The key of the map stores the operation pointer, it being the unique
  // identifier. The contents store the parent operation pointer and the Port of
  // the vertex.
  std::map<Operation *, PortConnection> vertex_map;

public:
  bool isNotEmpty() { return vertex_map.size() > 0; }

  void addVertex(Operation *parent_key, Operation *my_op, Port my_port) {
    vertex_map[my_op] = std::make_pair(parent_key, my_port);
  }

  StringRef getPortBundle(Operation *key) {
    return stringifyWireBundle(vertex_map.at(key).second.first);
  }

  std::string getPortChannel(Operation *key) {
    return std::to_string(vertex_map.at(key).second.second);
  }

  std::vector<Operation *> getPathToRoot(Operation *start_vertex) {
    std::vector<Operation *> path_to_root;

    // add start_vertex to path to prevent premature termination in the case the
    // start_vertex is a FlowEndPoint due to it being a connectedFlow.
    path_to_root.push_back(start_vertex);
    Operation *work = vertex_map.at(start_vertex).first;
    while (true) {
      path_to_root.push_back(work);
      if (isa<FlowEndPoint>(work)) {
        break;
      } else {
        work = vertex_map.at(work).first;
      }
    }
    return path_to_root;
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
        if (g.isNotEmpty())
          // implicit type converion connectOp -> Operation*
          g.addVertex(op, connectOp, connectOp.destPort());

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
                           Graph &g,
                           std::vector<Operation *> &antennaKeys) const {
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
        antennaKeys.push_back(op);
        if (g.isNotEmpty()) // continue if in antenna detection mode
          continue;
        else
          break; // break to restart in antenna detection mode
      }

      // add to graph if in detection mode and op is connectOp
      if (g.isNotEmpty())
        g.addVertex(op, nextConnection.getValue().first,
                    nextConnection.getValue().second);

      worklist.push_back(
          std::make_pair(nextConnection.getValue(), newMaskValue));
    }
    return worklist;
  }

  void detect_antenna(Graph &g, std::vector<Operation *> connectedTilesKeys,
                      std::vector<Operation *> antennaKeys) const {

    std::vector<Operation *> path_buffer;
    // create valid path from connected tiles
    std::set<Operation *> valid_path;
    for (auto &tile_key : connectedTilesKeys) {
      // todo: empty case
      path_buffer = g.getPathToRoot(tile_key);
      for (auto &vertex_key : path_buffer) {
        if (valid_path.find(vertex_key) == valid_path.end()) {
          valid_path.insert(vertex_key);
        }
      }
    }

    // generate messages for antennas
    for (auto &antenna_key : antennaKeys) {
      std::vector<Operation *> antenna_nonvalid_path;
      std::vector<Operation *> antenna_valid_path;
      path_buffer = g.getPathToRoot(antenna_key);
      for (auto &vertex_key : path_buffer) {
        if (valid_path.find(vertex_key) == valid_path.end()) {
          antenna_nonvalid_path.push_back(vertex_key);
        } else {
          antenna_valid_path.push_back(vertex_key);
        }
      }

      for (auto &key : antenna_nonvalid_path) {
        if (isa<ConnectOp>(key)) { // todo: else -> packet switched antenna
          // Generate Warning message for antennas inside a switchbox
          Operation *parent_op = key->getParentOp();
          key->emitWarning("Antenna\nAt Destination Port (" +
                           g.getPortBundle(key) + ":" + g.getPortChannel(key) +
                           ")\n")
                  .attachNote(parent_op->getLoc())
              << "in parent region\n";
        }
      }

      for (auto &key : antenna_valid_path) {
        // Generate Remarks for antenna path traceback
        key->emitRemark("Path Traceback\nUsing Port (" + g.getPortBundle(key) +
                        ":" + g.getPortChannel(key) + ")\n");
      }
    }
  }

public:
  // Get the tiles connected to the given tile, starting from the given
  // output port of the tile.  This is 1:N relationship because each
  // switchbox can broadcast.
  llvm::Optional<std::vector<PacketConnection>>
  getConnectedTiles(TileOp tileOp, Port port, bool &antenna_detection,
                    std::set<Operation *> &connections_set) const {

    // create graph
    Graph g;
    // root node if in detection mode
    if (antenna_detection)
      g.addVertex(nullptr, tileOp.getOperation(), port);

    LLVM_DEBUG(llvm::dbgs()
               << "getConnectedTile(" << stringifyWireBundle(port.first) << " "
               << (int)port.second << ")");
    LLVM_DEBUG(tileOp.dump());

    // The accumulated result;
    std::vector<PacketConnection> connectedTiles;
    // A worklist of PortConnections to visit.  These are all input ports of
    // some object (likely either a TileOp or a SwitchboxOp).
    std::vector<PacketConnection> worklist;
    // stores the leafs
    std::vector<Operation *> connectedTilesKeys;
    std::vector<Operation *> antennaKeys;

    // Start the worklist by traversing from the tile to its connected
    // switchbox.
    auto t = getConnectionThroughWire(tileOp.getOperation(), port);

    // If there is no wire to traverse, then just return no connection
    // emit error message for TileOp Antenna
    if (!t.hasValue()) {
      tileOp.getOperation()->emitWarning(
          "TileOp Antenna: " + stringifyWireBundle(port.first) + "\n");
      return None;
    }

    // add node to graph if in detection mode
    if (g.isNotEmpty())
      g.addVertex(tileOp.getOperation(), t.getValue().first,
                  t.getValue().second);

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
        connectedTilesKeys.push_back(other);
      } else if (isa<Interconnect>(other)) {
        // append to graph included with method
        std::vector<PacketConnection> nextOpPortMaskValues =
            getConnectionsThroughSwitchbox(other, otherPort, g,
                                           connections_set);
        // append to graph included with method
        std::vector<PacketConnection> newWorkList = maskSwitchboxConnections(
            other, maskValue, nextOpPortMaskValues, g, antennaKeys);
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
      if (!antenna_detection && antennaKeys.size() > 0) {
        antenna_detection = true;
        return connectedTiles;
      }
    }
    if (g.isNotEmpty())
      detect_antenna(g, connectedTilesKeys, antennaKeys);

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
      llvm::Optional<std::vector<PacketConnection>> tiles;
      tiles = analysis.getConnectedTiles(op, std::make_pair(bundle, i),
                                         antenna_detection, connections_set);
      if (!tiles.hasValue())
        // when no connections exist from the starting TileOP & bundle
        break;

      if (antenna_detection) { // get connected Tiles with antenna detection
        tiles = analysis.getConnectedTiles(op, std::make_pair(bundle, i),
                                           antenna_detection, connections_set);
      }
      LLVM_DEBUG(llvm::dbgs() << tiles.getValue().size() << " Flows\n");

      for (PacketConnection &c : tiles.getValue()) {
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

    std::set<Operation *> connections_set = create_connections_set(m);

    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());
    for (auto tile : m.getOps<TileOp>()) {
      findFlowsFrom(tile, analysis, builder, connections_set);
    }

    // emit error when connections_set is not empty
    for (auto &key : connections_set) {
      Operation *parent_op = key->getParentOp();
      key->emitWarning("Dangling Island Antenna\n")
          .attachNote(parent_op->getLoc());
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> xilinx::AIE::createAIEFindFlowsPass() {
  return std::make_unique<AIEFindFlowsPass>();
}