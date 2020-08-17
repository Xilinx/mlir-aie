// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Translation.h"
#include "AIEDialect.h"
#include "AIENetlistAnalysis.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

typedef std::pair<WireBundle, int> PortTy;
typedef std::pair<PortTy, PortTy> ConnectTy;

template <typename MyOp>
struct AIEOpRemoval : public OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  ModuleOp &module;

  AIEOpRemoval(MLIRContext *context, ModuleOp &m,
    PatternBenefit benefit = 1
  ) : OpConversionPattern<MyOp>(context, benefit),
    module(m) {}

  LogicalResult matchAndRewrite(MyOp op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    rewriter.eraseOp(Op);
    return success();
  }
};

typedef std::pair<WireBundle, int> PortTy;
typedef std::pair<Operation *, PortTy> PhysPort;

struct AIECreatePacketFlowsPass : public PassWrapper<AIECreatePacketFlowsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder(m.getBody()->getTerminator());

    ConversionTarget target(getContext());

    // Some communication patterns:
    //  - one-to-one
    //  - one-to-many
    //    + same flow (same port, same ID): broadcast/multicast
    //    + different flows (same port, differnt IDs)
    //  - many-to-one
    //    + timeshare: single arbiter, different msels
    //  - many-to-many
    //
    // Compute the mask for each LUT entry of a slave port
    // Aim to generate as few LUT entries as possible

    // Avoid creating packetswitch config as much as possible
    // We will use circuit-switch to pre-route the flows from the source swboxes to the dest swboxes,
    // and only use packet-switch to route at the dest swboxes
    DenseMap<std::pair<int, int>, Operation *> tiles;
    DenseMap<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> packetFlows;
    SmallVector<std::pair<PhysPort, int>, 4> slavePorts;
    DenseMap<std::pair<PhysPort, int>, int> slaveAMSels;
    DenseMap<std::pair<Operation *, int>, SmallVector<PortTy, 4>> masterAMSels;

    for (auto tileOp : m.getOps<TileOp>()) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      tiles[std::make_pair(col, row)] = tileOp;
    }

    for (auto pktflow : m.getOps<PacketFlowOp>()) {
      Region &r = pktflow.ports();
      Block &b = r.front();
      Operation *source = nullptr;
      int flowID = pktflow.IDInt();
      std::pair<PhysPort, int> sourceFlow;

      for (Operation &Op : b.getOperations()) {
        if (PacketSourceOp sourcePort = dyn_cast<PacketSourceOp>(Op)) {
          source = sourcePort.tile().getDefiningOp();
          WireBundle sourceBundle = sourcePort.bundle();
          int sourceChannel = sourcePort.channelIndex();
          PortTy port = std::make_pair(sourceBundle, sourceChannel);
          sourceFlow = std::make_pair(std::make_pair(source, port), flowID);
        } else if (PacketDestOp destPort = dyn_cast<PacketDestOp>(Op)) {
          Operation *dest = destPort.tile().getDefiningOp();
          WireBundle destBundle = destPort.bundle();
          int destChannel = destPort.channelIndex();

          assert(source == dest && "Packet-switch routing between different tiles is not supported for now");
          packetFlows[sourceFlow].push_back(
            std::make_pair(dest, std::make_pair(destBundle, destChannel)));
          slavePorts.push_back(sourceFlow);
        }
      }
    }

    // amsel()
    // masterset()
    // packetrules()
    // rule()

    // Arbiter assignments. Each arbiter has four msels.
    // Therefore, the number of "logical" arbiters is 6 x 4 = 24
    // A master port can only be associated with one arbiter
    DenseMap<Operation *, int> amselValues;
    int numMsels = 4;
    int numArbiters = 6;

    // Check all multi-cast flows (same source, same ID). They should be assigned the same
    // arbiter and msel so that the flow can reach all the destination ports at the same time
    // For destination ports that appear in different (multicast) flows, it should have a different
    // <arbiterID, msel> value pair for each flow
    for (auto packetFlow : packetFlows) {
      Operation *tileOp = packetFlow.first.first.first;
      if (amselValues.count(tileOp) == 0)
        amselValues[tileOp] = 0;

      // arb0: 6*0,   6*1,   6*2,   6*3
      // arb1: 6*0+1, 6*1+1, 6*2+1, 6*3+1
      // arb2: 6*0+2, 6*1+2, 6*2+2, 6*3+2
      // arb3: 6*0+3, 6*1+3, 6*2+3, 6*3+3
      // arb4: 6*0+4, 6*1+4, 6*2+4, 6*3+4
      // arb5: 6*0+5, 6*1+5, 6*2+5, 6*3+5

      int amselValue = amselValues[tileOp];
      assert(amselValue < numArbiters && "Could not allocate new arbiter!");

      // Find existing arbiter assignment
      // If there is an assignment of an arbiter to a master port before, we assign
      // all the master ports here with the same arbiter but different msel
      bool foundMatchedDest = false;
      for (auto map : masterAMSels) {
        if (map.first.first != tileOp)
          continue;
        amselValue = map.first.second;

        // check if same destinations
        SmallVector<PortTy, 4> ports(masterAMSels[std::make_pair(tileOp, amselValue)]);
        if (ports.size() != packetFlow.second.size())
          continue;

        bool matched = true;
        for (auto dest : packetFlow.second) {
          PortTy port = std::make_pair(dest.second.first, dest.second.second);
          if (std::find(ports.begin(), ports.end(), port) == ports.end()) {
            matched = false;
            break;
          }
        }

        if (matched) {
          foundMatchedDest = true;
          break;
        }
      }

      if (!foundMatchedDest) {
        bool foundAMSelValue;
        for (int a = 0; a < numArbiters; a++) {
          for (int i = 0; i < numMsels; i++) {
            amselValue = a + i * numArbiters;
            if (masterAMSels.count(std::make_pair(tileOp, amselValue)) == 0) {
              foundAMSelValue = true;
              break;
            }
          }

          if (foundAMSelValue)
            break;
        }

        for (auto dest : packetFlow.second) {
          PortTy port = std::make_pair(dest.second.first, dest.second.second);
          masterAMSels[std::make_pair(tileOp, amselValue)].push_back(port);
        }
      }

      slaveAMSels[packetFlow.first] = amselValue;
      amselValues[tileOp] = amselValue % numArbiters;
    }

    DenseMap<PhysPort, SmallVector<int, 4>> mastersets;
    for (auto master : masterAMSels) {
      Operation * tileOp = master.first.first;
      int amselValue = master.first.second;
      for (auto port : master.second) {
        auto physPort = std::make_pair(tileOp, port);
        mastersets[physPort].push_back(amselValue);
      }
    }

    llvm::dbgs() << "CHECK mastersets\n";
    for (auto map : mastersets) {
      Operation *tileOp = map.first.first;
      WireBundle bundle = map.first.second.first;
      int channel = map.first.second.second;
      TileOp tile = dyn_cast<TileOp>(tileOp);
      llvm::dbgs() << "master " << tile << " " << stringifyWireBundle(bundle) << " : " << channel << '\n';
      for (auto value : map.second)
        llvm::dbgs() << "amsel: " << value << '\n';
    }

    // Compute mask values
    // Merging as many stream flows as possible
    // The flows must originate from the same source port and have different IDs
    // Two flows can be merged if they share the same destinations
    SmallVector<SmallVector<std::pair<PhysPort, int>, 4>, 4> slavesGroups;
    SmallVector<std::pair<PhysPort, int>, 4> workList(slavePorts);
    while (!workList.empty()) {
      auto slave1 = workList.pop_back_val();
      PortTy slavePort1 = std::make_pair(slave1.first.second.first, slave1.first.second.second);

      bool foundgroup = false;
      for (auto &group : slavesGroups) {
        auto slave2 = group.front();

        PortTy slavePort2 = std::make_pair(slave2.first.second.first, slave2.first.second.second);

        if (slavePort1 != slavePort2)
          continue;

        bool matched = true;

        auto dests1 = packetFlows[slave1];
        auto dests2 = packetFlows[slave2];

        if (dests1.size() != dests2.size())
          continue;

        for (auto dest1 : dests1) {
          if (std::find(dests2.begin(), dests2.end(), dest1) == dests2.end()) {
            matched = false;
            break;
          }
        }

        if (matched) {
          group.push_back(slave1);
          foundgroup = true;
          break;
        }
      }

      if (!foundgroup) {
        SmallVector<std::pair<PhysPort, int>, 4> group({slave1});
        slavesGroups.push_back(group);
      }
    }

    llvm::dbgs() << "slavesGroups: " << slavesGroups.size() << '\n';

    DenseMap<std::pair<PhysPort, int>, int> slaveMasks;
    for (auto group : slavesGroups) {
      int mask = -1;
      for (auto port : group) {
        int ID = port.second;
        if (mask == -1) {
          mask = ID;
          continue;
        }
        mask ^= ID;
      }

      if (group.size() == 1)
        mask =  0x1F;
      else
        mask = ~mask;

      for (auto port : group)
        slaveMasks[port] = mask & 0x1F;
    }

    llvm::dbgs() << "CHECK Slave Masks\n";
    for (auto map : slaveMasks) {
      auto port = map.first.first;
      TileOp tile = dyn_cast<TileOp>(port.first);
      WireBundle bundle = port.second.first;
      int channel = port.second.second;
      int ID = map.first.second;
      int mask = map.second;

      llvm::dbgs() << "Port " << tile << " " << stringifyWireBundle(bundle) << " " << channel << '\n';
      llvm::dbgs() << "Mask " << "0x" << llvm::utohexstr(mask) << '\n';
      llvm::dbgs() << "ID " << "0x" << llvm::utohexstr(ID) << '\n';
    }

    for (auto map : tiles) {
      Operation *tileOp = map.second;
      TileOp tile = dyn_cast<TileOp>(tileOp);
      builder.setInsertionPointAfter(tileOp);
      SwitchboxOp swbox = builder.create<SwitchboxOp>(builder.getUnknownLoc(), tile);
      swbox.ensureTerminator(swbox.connections(), builder, builder.getUnknownLoc());
      Block &b = swbox.connections().front();
      builder.setInsertionPoint(b.getTerminator());
      DenseMap<int, Value> amselOps;
      for (auto map : mastersets) {
        if (tileOp != map.first.first)
          continue;

        WireBundle bundle = map.first.second.first;
        int channel = map.first.second.second;
        SmallVector<Value, 4> amsels;
        for (auto value : map.second) {
          if (amselOps.count(value) == 1) {
            amsels.push_back(amselOps[value]);
            continue;
          }

          int arbiterID = value % numArbiters;
          int msel = value / numArbiters;
          AMSelOp amsel = builder.create<AMSelOp>(builder.getUnknownLoc(), arbiterID, msel);
          amselOps[value] = amsel;
          amsels.push_back(amsel);
        }

        MasterSetOp masters = builder.create<MasterSetOp>(builder.getUnknownLoc(),
                                                          builder.getIndexType(),
                                                          bundle, APInt(32, channel), amsels);
      }

      DenseMap<PortTy, PacketRulesOp> slaveRules;
      for (auto group : slavesGroups) {
      //for (auto map : slaveMasks) {
        builder.setInsertionPoint(b.getTerminator());

        //auto port = map.first.first;
        auto port = group.front().first;
        TileOp tile = dyn_cast<TileOp>(port.first);
        WireBundle bundle = port.second.first;
        int channel = port.second.second;
        auto slave = std::make_pair(bundle, channel);

        int ID = group.front().second;
        int mask = slaveMasks[group.front()];
        Value amsel = amselOps[slaveAMSels[group.front()]];

        PacketRulesOp packetrules;
        if (slaveRules.count(slave) == 0) {
          packetrules = builder.create<PacketRulesOp>(builder.getUnknownLoc(), bundle, channel);
          packetrules.ensureTerminator(packetrules.rules(), builder, builder.getUnknownLoc());
          slaveRules[slave] = packetrules;
        } else
          packetrules = slaveRules[slave];

        Block &rules = packetrules.rules().front();
        builder.setInsertionPoint(rules.getTerminator());
        PacketRuleOp rule = builder.create<PacketRuleOp>(builder.getUnknownLoc(), mask, ID, amsel);
      }
    }

    OwningRewritePatternList patterns;
    patterns.insert<AIEOpRemoval<PacketFlowOp>
                   >(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, patterns)))
      signalPassFailure();
  }
};

void xilinx::AIE::registerAIECreatePacketFlowsPass() {
    PassRegistration<AIECreatePacketFlowsPass>(
      "aie-create-packet-flows",
      "Lowering PacketFlow ops to Switchbox ops with packet-switch routing");
}
