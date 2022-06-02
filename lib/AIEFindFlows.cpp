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
typedef std::pair<PortConnection, PacketConnection> Source_PacketConnection;

struct vertex {
  vertex *parent = NULL;
  std::vector<vertex*> children;

  Operation *op_data;
  Port port_data;

  vertex(Operation *op, Port port) {
    op_data = op;
    port_data = port;
  }
};

class Graph {
  std::vector<vertex> vertices;

  public:
    // default root node constructor
    Graph(Operation *op, Port port) {
      vertices.push_back(vertex(op, port));
    }

    //add node (parent-op, parent-port, child-op, child-port)
    void add_child(Operation *op_p, Port port_p, Operation *op_c, Port port_c) {

      bool found_target = false;
      unsigned int idx;
      for (unsigned int i = 0; i < vertices.size(); i++) { //loop vector by index
        if(vertices[i].op_data == op_p && vertices[i].port_data == port_p) {
          // bind to node that is lowest in the hierarchy
          found_target = true;
          idx = i;
        }     
      }
      if (found_target) {
        vertices.push_back(vertex(op_c, port_c)); //add new node
        vertices.back().parent = &vertices[idx]; //link parent
        vertices[idx].children.push_back(&vertices.back()); //link child
        return;
      }      
      LLVM_DEBUG(llvm::dbgs() << "No parent in graph\n");
    }

    std::vector<vertex*> get_leaf_nodes() {

      std::vector<vertex*> leafs;
      for (auto &node : vertices) {
        if (node.children.size() == 0) {
          leafs.push_back(&node);
        }
      }
      return leafs;
    }

    vertex* get_parent(vertex* node) {
      return node->parent;
    }

    std::vector<vertex*> get_path2root(vertex* start_node) {

      std::vector<vertex*> valid_path;
      std::vector<vertex*> worklist;      
      
      worklist.push_back(start_node);
      while (!worklist.empty()) {
        vertex* child = worklist.back();
        worklist.pop_back();
        valid_path.push_back(child);
        vertex* parent = get_parent(child);
        if (parent != NULL) {
          worklist.push_back(parent);
        }
      }

      return valid_path;
    }
    
    // debug methods
    void printNode(int idx) {
      LLVM_DEBUG(llvm::dbgs() << "Graph node: " << idx << "\n");
      LLVM_DEBUG(vertices[idx].op_data->dump());
      LLVM_DEBUG(llvm::dbgs() << stringifyWireBundle(vertices[idx].port_data.first) << " "
              << (int)vertices[idx].port_data.second << "\n");
    }

    void printAllNodes() {
      for (unsigned int i = 0; i < vertices.size(); i++) {
        printNode(i);
      }
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
  getConnectionsThroughSwitchbox(Region &r, Port sourcePort) const {
    LLVM_DEBUG(llvm::dbgs() << "Switchbox:\n");
    Block &b = r.front();
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
    return portSet;
  }

  std::vector<Source_PacketConnection>
  maskSwitchboxConnections(Operation *switchOp,
                           std::vector<PortMaskValue> nextPortMaskValues,
                           MaskValue maskValue) const {
    std::vector<Source_PacketConnection> worklist;
    // bool matched = false;
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
      // matched = true;
      MaskValue newMaskValue = std::make_pair(
          maskValue.first | nextMaskValue.first,
          maskValue.second | (nextMaskValue.first & nextMaskValue.second));
      auto nextConnection = getConnectionThroughWire(switchOp, nextPort);

      // If there is no wire to follow then bail out.
      if (!nextConnection.hasValue())
        continue;
      PortConnection sour = std::make_pair(switchOp, nextPort);
      PacketConnection dest = std::make_pair(nextConnection.getValue(), newMaskValue);
      worklist.push_back(
          std::make_pair(sour, dest));
    }
    return worklist;
  }

  void detect_antenna(Graph &g, std::vector<PacketConnection> &connectedTiles) const {

    // g.printAllNodes();
    std::vector<vertex*> leafs = g.get_leaf_nodes();

    std::vector<vertex*> connected_leafs;
    // remove leaf node that are connected
    for (auto &connectedTile : connectedTiles) {
      auto iter = leafs.begin();
      while (iter != leafs.end()) {
        if ((*iter)->op_data == connectedTile.first.first 
              && (*iter)->port_data == connectedTile.first.second) {
            connected_leafs.push_back(*iter);
            iter = leafs.erase(iter); // update iterator to next item
        } 
        else {
          iter++;
        }
      }
    }

    //create valid path from connected tiles
    std::vector<vertex*> valid_path;
    std::vector<vertex*> valid_path_buffer;
    for (auto &connected_leaf : connected_leafs) {
      valid_path_buffer = g.get_path2root(connected_leaf);      
      if (valid_path.empty()) {
          valid_path = valid_path_buffer;
      } 
      else { // check if path is already in vector
        for (auto &vpath_buf : valid_path_buffer) {
          for (unsigned int vp_i = 0; vp_i < valid_path.size(); vp_i++) { //loop vector by index
            if (!(vpath_buf == valid_path[vp_i])) {
              valid_path.push_back(vpath_buf);
            }        
          }
        }
      }
    }    

    // output antenna path
    std::vector<vertex*> antenna_valid_path;
    std::vector<vertex*> antenna_nonvalid_path;
    bool isvalid = false;
    for (auto &leaf : leafs) {
      antenna_valid_path = g.get_path2root(leaf);
      antenna_nonvalid_path.clear();
      if (2 < antenna_valid_path.size()) { //core/dma -> switch is not a antenna
        auto aiter = antenna_valid_path.begin();
        while (aiter != antenna_valid_path.end()) {
          isvalid = false;
          for (auto &vpath : valid_path) {
            if (*aiter == vpath) {
              isvalid = true;
              break;
            } 
          }

          if (isvalid) {
            aiter++;
          } 
          else {
            antenna_nonvalid_path.push_back(*aiter);
            aiter = antenna_valid_path.erase(aiter); // update iterator to next item

          }
        }
        for (auto &anvp : antenna_nonvalid_path) {
          llvm::errs() << "Antenna:" << "\n";
          llvm::errs() << "At port: " << stringifyWireBundle(anvp->port_data.first) << " "
                << (int)anvp->port_data.second << "\n";
          anvp->op_data->dump();
        }
        for (auto &vp : antenna_valid_path) {
          llvm::errs() << "Valid Route Shared With Antenna:" << "\n";
          llvm::errs() << "At port: " << stringifyWireBundle(vp->port_data.first) << " "
                  << (int)vp->port_data.second << "\n";
          vp->op_data->dump();                  
        } 


        // for (auto &anvp : antenna_nonvalid_path) {
        //   anvp->op_data->emitWarning() << "Antenna\n";
        // }        
        // for (auto &vp : antenna_valid_path) {
        //   vp->op_data->emitRemark() << "NonAntenna\n";
        // }
      }
    }
  }

public:
  // Get the tiles connected to the given tile, starting from the given
  // output port of the tile.  This is 1:N relationship because each
  // switchbox can broadcast.
  std::vector<PacketConnection> getConnectedTiles(TileOp tileOp,
                                                  Port port) const {

    //create graph and add root node
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
    

    auto t = getConnectionThroughWire(tileOp.getOperation(), port);

    // If there is no wire to traverse, then just return no connection
    if (!t.hasValue())
      return connectedTiles;

    //add node to graph
    g.add_child(tileOp.getOperation(), port, t.getValue().first, t.getValue().second);
    
    PacketConnection connection = std::make_pair(t.getValue(), std::make_pair(0, 0));
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
      } else if (auto switchOp = dyn_cast_or_null<SwitchboxOp>(other)) {

        std::vector<PortMaskValue> nextPortMaskValues =
            getConnectionsThroughSwitchbox(switchOp.connections(), otherPort);

        //add path to tree
        for (auto &nextPortMaskValue : nextPortMaskValues) {
          g.add_child(other, otherPort, other, nextPortMaskValue.first);
        }

        std::vector<Source_PacketConnection> newWorkList =
            maskSwitchboxConnections(switchOp, nextPortMaskValues, maskValue);

        // append to the worklist and graph
        for (auto &newWork : newWorkList) {
          g.add_child(newWork.first.first, newWork.first.second, 
              newWork.second.first.first, newWork.second.first.second);
          worklist.insert(worklist.end(), newWork.second);
        }
        // worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        if (nextPortMaskValues.size() > 0 && newWorkList.size() == 0) {
          // No rule matched some incoming packet.  This is likely a
          // configuration error.
          LLVM_DEBUG(llvm::dbgs() << "No rule matched incoming packet here: ");
          LLVM_DEBUG(other->dump());
        }
      } 
      else if (auto switchOp = dyn_cast_or_null<ShimMuxOp>(other)) {
        std::vector<PortMaskValue> nextPortMaskValues =
            getConnectionsThroughSwitchbox(switchOp.connections(), otherPort);

        //add path to tree
        for (auto &nextPortMaskValue : nextPortMaskValues) {
          g.add_child(other, otherPort, other, nextPortMaskValue.first);
        }

        std::vector<Source_PacketConnection> newWorkList =
            maskSwitchboxConnections(switchOp, nextPortMaskValues, maskValue);

        // append to the worklist and graph
        for (auto &newWork : newWorkList) {
          g.add_child(newWork.first.first, newWork.first.second, 
              newWork.second.first.first, newWork.second.first.second);
          worklist.insert(worklist.end(), newWork.second);
        }
        // worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        if (nextPortMaskValues.size() > 0 && newWorkList.size() == 0) {
          // No rule matched some incoming packet.  This is likely a
          // configuration error.
          LLVM_DEBUG(llvm::dbgs() << "No rule matched incoming packet here: ");
          LLVM_DEBUG(other->dump());
        }
      } 
      else {
        LLVM_DEBUG(llvm::dbgs()
                   << "*** Connection Terminated at unknown operation: ");
        LLVM_DEBUG(other->dump());
      }
    }

    detect_antenna(g, connectedTiles);

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