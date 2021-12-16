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

  std::vector<PacketConnection>
  maskSwitchboxConnections(Operation *switchOp,
                           std::vector<PortMaskValue> nextPortMaskValues,
                           MaskValue maskValue) const {
    std::vector<PacketConnection> worklist;
    bool matched = false;
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
      matched = true;
      MaskValue newMaskValue = std::make_pair(
          maskValue.first | nextMaskValue.first,
          maskValue.second | (nextMaskValue.first & nextMaskValue.second));
      auto nextConnection = getConnectionThroughWire(switchOp, nextPort);

      // If there is no wire to follow then bail out.
      if (!nextConnection.hasValue())
        continue;

      worklist.push_back(
          std::make_pair(nextConnection.getValue(), newMaskValue));
    }
    return worklist;
  }

public:
  // Get the tiles connected to the given tile, starting from the given
  // output port of the tile.  This is 1:N relationship because each
  // switchbox can broadcast.
  std::vector<PacketConnection> getConnectedTiles(TileOp tileOp,
                                                  Port port) const {

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
    worklist.push_back(std::make_pair(t.getValue(), std::make_pair(0, 0)));

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
        std::vector<PacketConnection> newWorkList =
            maskSwitchboxConnections(switchOp, nextPortMaskValues, maskValue);
        // append to the worklist
        worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        if (nextPortMaskValues.size() > 0 && newWorkList.size() == 0) {
          // No rule matched some incoming packet.  This is likely a
          // configuration error.
          LLVM_DEBUG(llvm::dbgs() << "No rule matched incoming packet here: ");
          LLVM_DEBUG(other->dump());
        }
      } else if (auto switchOp = dyn_cast_or_null<ShimMuxOp>(other)) {
        std::vector<PortMaskValue> nextPortMaskValues =
            getConnectionsThroughSwitchbox(switchOp.connections(), otherPort);
        std::vector<PacketConnection> newWorkList =
            maskSwitchboxConnections(switchOp, nextPortMaskValues, maskValue);
        // append to the worklist
        worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
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
    registry.insert<StandardOpsDialect>();
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
