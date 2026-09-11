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
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEFINDFLOWS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-find-flows"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

using MaskValue = struct MaskValue {
  int mask;
  int value;
};

using PortConnection = struct PortConnection {
  Operation *op;
  Port port;
};

using PortMaskValue = struct PortMaskValue {
  Port port;
  MaskValue mv;
};

using PacketConnection = struct PacketConnection {
  PortConnection portConnection;
  MaskValue mv;
};

class ConnectivityAnalysis {
  DeviceOp &device;

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

  std::vector<PortMaskValue>
  getConnectionsThroughSwitchbox(Region &r, Port sourcePort) const {
    LLVM_DEBUG(llvm::dbgs() << "Switchbox:\n");
    Block &b = r.front();
    std::vector<PortMaskValue> portSet;
    for (auto connectOp : b.getOps<ConnectOp>()) {
      if (connectOp.sourcePort() == sourcePort) {
        MaskValue maskValue = {0, 0};
        portSet.push_back({connectOp.destPort(), maskValue});
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
        for (auto masterSetOp : b.getOps<MasterSetOp>())
          for (Value amsel : masterSetOp.getAmsels())
            for (auto ruleOp :
                 connectOp.getRules().front().getOps<PacketRuleOp>()) {
              if (ruleOp.getAmsel() == amsel) {
                LLVM_DEBUG(llvm::dbgs()
                           << "To:"
                           << stringifyWireBundle(masterSetOp.destPort().bundle)
                           << " " << masterSetOp.destPort().channel << "\n");
                MaskValue maskValue = {ruleOp.maskInt(), ruleOp.valueInt()};
                portSet.push_back({masterSetOp.destPort(), maskValue});
              }
            }
      }
    }
    return portSet;
  }

  std::vector<PacketConnection>
  maskSwitchboxConnections(Operation *switchOp,
                           const std::vector<PortMaskValue> &nextPortMaskValues,
                           MaskValue maskValue) const {
    std::vector<PacketConnection> worklist;
    for (auto &nextPortMaskValue : nextPortMaskValues) {
      Port nextPort = nextPortMaskValue.port;
      MaskValue nextMaskValue = nextPortMaskValue.mv;
      int maskConflicts = nextMaskValue.mask & maskValue.mask;
      LLVM_DEBUG(llvm::dbgs() << "Mask: " << maskValue.mask << " "
                              << maskValue.value << "\n");
      LLVM_DEBUG(llvm::dbgs() << "NextMask: " << nextMaskValue.mask << " "
                              << nextMaskValue.value << "\n");
      LLVM_DEBUG(llvm::dbgs() << maskConflicts << "\n");

      if ((maskConflicts & nextMaskValue.value) !=
          (maskConflicts & maskValue.value)) {
        // Incoming packets cannot match this rule. Skip it.
        continue;
      }
      MaskValue newMaskValue = {maskValue.mask | nextMaskValue.mask,
                                maskValue.value |
                                    (nextMaskValue.mask & nextMaskValue.value)};
      auto nextConnection = getConnectionThroughWire(switchOp, nextPort);

      // If there is no wire to follow then bail out.
      if (!nextConnection)
        continue;

      worklist.push_back({*nextConnection, newMaskValue});
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
    worklist.push_back({*t, {0, 0}});

    while (!worklist.empty()) {
      PacketConnection t = worklist.back();
      worklist.pop_back();
      PortConnection portConnection = t.portConnection;
      MaskValue maskValue = t.mv;
      Operation *other = portConnection.op;
      Port otherPort = portConnection.port;
      if (other && other->hasTrait<IsFlowEndPoint>()) {
        // If we got to a tile, then add it to the result.
        connectedTiles.push_back(t);
      } else if (auto switchOp = dyn_cast_or_null<SwitchboxOp>(other)) {
        std::vector<PortMaskValue> nextPortMaskValues =
            getConnectionsThroughSwitchbox(switchOp.getConnections(),
                                           otherPort);
        std::vector<PacketConnection> newWorkList =
            maskSwitchboxConnections(switchOp, nextPortMaskValues, maskValue);
        // append to the worklist
        worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        if (!nextPortMaskValues.empty() && newWorkList.empty()) {
          // No rule matched some incoming packet.  This is likely a
          // configuration error.
          LLVM_DEBUG(llvm::dbgs() << "No rule matched incoming packet here: ");
          LLVM_DEBUG(other->dump());
        }
      } else if (auto switchOp = dyn_cast_or_null<ShimMuxOp>(other)) {
        std::vector<PortMaskValue> nextPortMaskValues =
            getConnectionsThroughSwitchbox(switchOp.getConnections(),
                                           otherPort);
        std::vector<PacketConnection> newWorkList =
            maskSwitchboxConnections(switchOp, nextPortMaskValues, maskValue);
        // append to the worklist
        worklist.insert(worklist.end(), newWorkList.begin(), newWorkList.end());
        if (!nextPortMaskValues.empty() && newWorkList.empty()) {
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

// Identifies a flow by its two endpoints -- tile coordinates plus port -- and
// the packet ID it carries, using kCircuitFlow for circuit-switched flows.
// Coordinates rather than SSA values, so flows written against different
// aie.tile ops for the same tile still compare equal.
static constexpr int kCircuitFlow = -1;
using FlowKey = std::tuple<int, int, int, int, int, int, int, int, int>;
using FlowKeySet = std::set<FlowKey>;

// Returns nullopt when either endpoint's coordinates are unknown, in which
// case the caller cannot tell the flow apart from any other and must not
// dedupe it away.
static std::optional<FlowKey> tryGetFlowKey(Operation *srcOp, Port srcPort,
                                            Operation *destOp, Port destPort,
                                            int packetID) {
  auto coords = [](Operation *op) -> std::optional<std::pair<int, int>> {
    auto tile = llvm::dyn_cast_or_null<TileLike>(op);
    if (!tile)
      return std::nullopt;
    std::optional<int> col = tile.tryGetCol();
    std::optional<int> row = tile.tryGetRow();
    if (!col || !row)
      return std::nullopt;
    return std::make_pair(*col, *row);
  };
  std::optional<std::pair<int, int>> src = coords(srcOp);
  std::optional<std::pair<int, int>> dest = coords(destOp);
  if (!src || !dest)
    return std::nullopt;
  return FlowKey{src->first,
                 src->second,
                 static_cast<int>(srcPort.bundle),
                 srcPort.channel,
                 dest->first,
                 dest->second,
                 static_cast<int>(destPort.bundle),
                 destPort.channel,
                 packetID};
}

// Drops the flows the device already declares. This pass recovers the logical
// flows a routed design implements, so the flows it writes are the answer --
// leaving the originals in place next to them would just describe the same
// routing twice, which DeviceOp::verify now rejects. It really does happen:
// --aie-create-pathfinder-flows leaves behind every aie.flow it folded into an
// earlier flow with the same source, and it never consumes aie.packet_flow.
static void eraseExistingFlows(DeviceOp device) {
  SmallVector<Operation *> toErase;
  for (FlowOp flow : device.getOps<FlowOp>())
    toErase.push_back(flow);
  for (PacketFlowOp packetFlow : device.getOps<PacketFlowOp>())
    toErase.push_back(packetFlow);
  for (Operation *op : toErase)
    op->erase();
}

static void findFlowsFrom(TileOp op, ConnectivityAnalysis &analysis,
                          OpBuilder &rewriter, FlowKeySet &seen) {
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
        MaskValue maskValue = c.mv;
        Operation *destOp = portConnection.op;
        Port destPort = portConnection.port;
        // The traversal can reach the same endpoint more than once, for
        // instance when a broadcast re-converges on it. Those repeats all
        // describe one logical flow, and a flow declared twice is rejected by
        // DeviceOp::verify.
        std::optional<FlowKey> key =
            tryGetFlowKey(Op, {bundle, (int)i}, destOp, destPort,
                          maskValue.mask == 0 ? kCircuitFlow : maskValue.value);
        if (key && !seen.insert(*key).second)
          continue;
        if (maskValue.mask == 0) {
          FlowOp::create(rewriter, Op->getLoc(), Op->getResult(0), bundle, i,
                         destOp->getResult(0), destPort.bundle,
                         destPort.channel);
        } else {
          auto flowOp = PacketFlowOp::create(rewriter, Op->getLoc(),
                                             maskValue.value, nullptr, nullptr);
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

    // The analysis above reads only the switchboxes and shim muxes, so the
    // flows can be cleared before rebuilding them from that routing.
    eraseExistingFlows(d);

    OpBuilder builder = OpBuilder::atBlockTerminator(d.getBody());
    FlowKeySet seen;
    for (auto tile : d.getOps<TileOp>()) {
      findFlowsFrom(tile, analysis, builder, seen);
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEFindFlowsPass() {
  return std::make_unique<AIEFindFlowsPass>();
}
