//===- AIEFindFlows.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEFINDFLOWS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-find-flows"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

// The packet ids reaching a point, one bit per id. A set rather than a
// (mask, value) cube because first-match priority leaves a rule carrying the
// ids no earlier rule took, which is not in general a cube. At most 32 ids
// wide (AIETargetModel::getMaxPacketId).
typedef uint32_t IdSet;

typedef struct PortConnection {
  Operation *op;
  Port port;
} PortConnection;

typedef struct PortIdSet {
  Port port;
  IdSet ids;
  bool isPacket;
} PortIdSet;

typedef struct PacketConnection {
  PortConnection portConnection;
  IdSet ids;
  bool isPacket;
} PacketConnection;

class ConnectivityAnalysis {
  DeviceOp &device;

  // Clamped so every shift below stays in range; a target with a wider id
  // field needs a wider IdSet, not a silently truncated analysis.
  unsigned numIds() const {
    uint32_t maxPacketId = device.getTargetModel().getMaxPacketId();
    assert(maxPacketId < 32 && "packet id space wider than an IdSet");
    return std::min<uint32_t>(maxPacketId + 1, 32);
  }

  IdSet allIds() const {
    unsigned n = numIds();
    return n == 32 ? ~IdSet(0) : (IdSet(1) << n) - 1;
  }

  // The ids a rule matches on its own, ignoring priority: (id & mask) == value.
  IdSet ruleIds(int mask, int value) const {
    IdSet ids = 0;
    for (unsigned id = 0; id < numIds(); id++)
      if ((static_cast<int>(id) & mask) == value)
        ids |= IdSet(1) << id;
    return ids;
  }

public:
  ConnectivityAnalysis(DeviceOp &d) : device(d) {}

private:
  std::optional<PortConnection>
  getConnectionThroughWire(Operation *op, Port masterPort) const {
    LLVM_DEBUG(llvm::dbgs() << "Wire:" << *op << " "
                            << stringifyWireBundle(masterPort.bundle) << " "
                            << masterPort.channel << "\n");
    for (auto wireOp : device.getOps<WireOp>()) {
      if (wireOp.getSource().getDefiningOp() == op &&
          wireOp.getSourceBundle() == masterPort.bundle) {
        Operation *other = wireOp.getDest().getDefiningOp();
        Port otherPort = {wireOp.getDestBundle(), masterPort.channel};
        LLVM_DEBUG(llvm::dbgs() << "Connects To:" << *other << " "
                                << stringifyWireBundle(otherPort.bundle) << " "
                                << otherPort.channel << "\n");

        return PortConnection{other, otherPort};
      }
      if (wireOp.getDest().getDefiningOp() == op &&
          wireOp.getDestBundle() == masterPort.bundle) {
        Operation *other = wireOp.getSource().getDefiningOp();
        Port otherPort = {wireOp.getSourceBundle(), masterPort.channel};
        LLVM_DEBUG(llvm::dbgs() << "Connects To:" << *other << " "
                                << stringifyWireBundle(otherPort.bundle) << " "
                                << otherPort.channel << "\n");
        return PortConnection{other, otherPort};
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "*** Missing Wire!\n");
    return std::nullopt;
  }

  std::vector<PortIdSet> getConnectionsThroughSwitchbox(Region &r,
                                                        Port sourcePort) const {
    LLVM_DEBUG(llvm::dbgs() << "Switchbox:\n");
    Block &b = r.front();
    std::vector<PortIdSet> portSet;
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        portSet.push_back({connectOp.destPort(), allIds(), false});
        LLVM_DEBUG(llvm::dbgs()
                   << "To:" << stringifyWireBundle(connectOp.destPort().bundle)
                   << " " << connectOp.destPort().channel << "\n");
      }
    }
    for (auto connectOp : b.getOps<PacketRulesOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Packet From: "
                   << stringifyWireBundle(connectOp.sourcePort().bundle) << " "
                   << sourcePort.channel << "\n");
        // In rule order: a rule carries only what no earlier rule claimed.
        IdSet claimed = 0;
        for (auto ruleOp :
             connectOp.getRules().front().getOps<PacketRuleOp>()) {
          IdSet matched = ruleIds(ruleOp.maskInt(), ruleOp.valueInt());
          IdSet carried = matched & ~claimed;
          claimed |= matched;
          if (!carried) {
            // Fully shadowed by an earlier rule: dead on hardware.
            LLVM_DEBUG(llvm::dbgs() << "Shadowed rule, carries nothing\n");
            continue;
          }
          for (auto masterSetOp : b.getOps<MasterSetOp>())
            for (Value amsel : masterSetOp.getAmsels())
              if (ruleOp.getAmsel() == amsel) {
                LLVM_DEBUG(llvm::dbgs()
                           << "To:"
                           << stringifyWireBundle(masterSetOp.destPort().bundle)
                           << " " << masterSetOp.destPort().channel << "\n");
                portSet.push_back({masterSetOp.destPort(), carried, true});
              }
        }
      }
    }
    return portSet;
  }

  std::vector<PacketConnection>
  maskSwitchboxConnections(Operation *switchOp,
                           std::vector<PortIdSet> nextPortIdSets, IdSet ids,
                           bool isPacket) const {
    std::vector<PacketConnection> worklist;
    for (auto &nextPortIdSet : nextPortIdSets) {
      LLVM_DEBUG(llvm::dbgs() << "Ids: " << ids << "\n");
      LLVM_DEBUG(llvm::dbgs() << "NextIds: " << nextPortIdSet.ids << "\n");

      IdSet newIds = ids & nextPortIdSet.ids;
      if (!newIds) {
        // Incoming packets cannot match this rule. Skip it.
        continue;
      }
      auto nextConnection =
          getConnectionThroughWire(switchOp, nextPortIdSet.port);

      // If there is no wire to follow then bail out.
      if (!nextConnection)
        continue;

      worklist.push_back(
          {*nextConnection, newIds, isPacket || nextPortIdSet.isPacket});
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
               << "getConnectedTile(" << stringifyWireBundle(port.bundle) << " "
               << port.channel << ")");
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
    if (!t)
      return connectedTiles;
    worklist.push_back({*t, allIds(), false});

    while (!worklist.empty()) {
      PacketConnection t = worklist.back();
      worklist.pop_back();
      PortConnection portConnection = t.portConnection;
      Operation *other = portConnection.op;
      Port otherPort = portConnection.port;
      if (other && other->hasTrait<IsFlowEndPoint>()) {
        // If we got to a tile, then add it to the result.
        connectedTiles.push_back(t);
      } else if (auto switchOp = dyn_cast_or_null<SwitchboxOp>(other)) {
        std::vector<PortIdSet> nextPortIdSets = getConnectionsThroughSwitchbox(
            switchOp.getConnections(), otherPort);
        std::vector<PacketConnection> newWorkList = maskSwitchboxConnections(
            switchOp, nextPortIdSets, t.ids, t.isPacket);
        // append to the worklist
        worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        if (!nextPortIdSets.empty() && newWorkList.empty()) {
          // No rule matched some incoming packet.  This is likely a
          // configuration error.
          LLVM_DEBUG(llvm::dbgs() << "No rule matched incoming packet here: ");
          LLVM_DEBUG(other->dump());
        }
      } else if (auto switchOp = dyn_cast_or_null<ShimMuxOp>(other)) {
        std::vector<PortIdSet> nextPortIdSets = getConnectionsThroughSwitchbox(
            switchOp.getConnections(), otherPort);
        std::vector<PacketConnection> newWorkList = maskSwitchboxConnections(
            switchOp, nextPortIdSets, t.ids, t.isPacket);
        // append to the worklist
        worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        if (!nextPortIdSets.empty() && newWorkList.empty()) {
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

static void findFlowsFrom(TileOp op, ConnectivityAnalysis &analysis,
                          OpBuilder &rewriter) {
  Operation *Op = op.getOperation();
  rewriter.setInsertionPoint(Op->getBlock()->getTerminator());

  std::vector bundles = {WireBundle::Core, WireBundle::DMA};
  for (WireBundle bundle : bundles) {
    LLVM_DEBUG(llvm::dbgs()
               << op << stringifyWireBundle(bundle) << " has "
               << op.getNumSourceConnections(bundle) << " Connections\n");
    for (size_t i = 0; i < op.getNumSourceConnections(bundle); i++) {
      std::vector<PacketConnection> tiles =
          analysis.getConnectedTiles(op, {bundle, (int)i});
      LLVM_DEBUG(llvm::dbgs() << tiles.size() << " Flows\n");

      for (PacketConnection &c : tiles) {
        PortConnection portConnection = c.portConnection;
        Operation *destOp = portConnection.op;
        Port destPort = portConnection.port;
        if (!c.isPacket) {
          FlowOp::create(rewriter, Op->getLoc(), Op->getResult(0), bundle, i,
                         destOp->getResult(0), destPort.bundle,
                         destPort.channel);
        } else {
          // A route carries a set of ids, but PacketFlowOp names one. Take the
          // lowest reaching here. With relaxed masks that can be an id no
          // source sends, so treat the name as identifying the route rather
          // than a specific packet.
          auto flowOp =
              PacketFlowOp::create(rewriter, Op->getLoc(),
                                   llvm::countr_zero(c.ids), nullptr, nullptr);
          PacketFlowOp::ensureTerminator(flowOp.getPorts(), rewriter,
                                         Op->getLoc());
          OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
          rewriter.setInsertionPoint(flowOp.getPorts().front().getTerminator());
          PacketSourceOp::create(rewriter, Op->getLoc(), Op->getResult(0),
                                 bundle, i);
          PacketDestOp::create(rewriter, Op->getLoc(), destOp->getResult(0),
                               destPort.bundle, destPort.channel);
          rewriter.restoreInsertionPoint(ip);
        }
      }
    }
  }
}

struct AIEFindFlowsPass
    : public xilinx::AIE::impl::AIEFindFlowsBase<AIEFindFlowsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }
  void runOnOperation() override {

    DeviceOp d = getOperation();
    ConnectivityAnalysis analysis(d);
    d.getTargetModel().validate();

    OpBuilder builder = OpBuilder::atBlockTerminator(d.getBody());
    for (auto tile : d.getOps<TileOp>()) {
      findFlowsFrom(tile, analysis, builder);
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEFindFlowsPass() {
  return std::make_unique<AIEFindFlowsPass>();
}
