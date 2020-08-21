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

typedef std::pair<Operation *, Port> PhysPort;

int getAvailableDestChannel(
  SmallVector<std::pair<Connect, int>, 8> &connects,
  Port sourcePort, int flowID,
  WireBundle destBundle) {

  if (connects.size() == 0)
    return 0;

  int numChannels;

  if (destBundle == WireBundle::North)
    numChannels = 6;
  else if (destBundle == WireBundle::South ||
           destBundle == WireBundle::East  ||
           destBundle == WireBundle::West)
    numChannels = 4;
  else
    numChannels = 2;

  int availableChannel = -1;

  // look for existing connect that has a matching destination
  for (int i = 0; i < numChannels; i++) {
    Port port = std::make_pair(destBundle, i);
    int countFlows = 0;
    for (auto conn : connects) {
      Port connDest = conn.first.second;
      // Since we are doing packet-switched routing, dest ports can be shared among multiple sources.
      // Therefore, we don't need to worry about checking the same source
      if (connDest == port)
        countFlows++;
    }

    // Since a mask has 5 bits, there can only be 32 logical streams flow through a port
    // TODO: what about packet-switched flow that uses nested header?
    if (countFlows > 0 && countFlows < 32)
      return i;
  }

  // if not, look for available destination port
  for (int i = 0; i < numChannels; i++) {
    Port port = std::make_pair(destBundle, i);
    SmallVector<Port, 8> ports;
    for (auto connect : connects)
      ports.push_back(connect.first.second);

    if (std::find(ports.begin(), ports.end(), port) == ports.end())
      return i;
  }

  return -1;
}

// build packet-switched route
void buildPSRoute(int xSrc, int ySrc, int xDest, int yDest,
  Port sourcePort,
  Port destPort,
  int flowID,
  DenseMap<std::pair<int, int>, SmallVector<std::pair<Connect, int>, 8>> &switchboxes) {

  int xCnt = 0;
  int yCnt = 0;

  int xCur = xSrc;
  int yCur = ySrc;
  WireBundle curBundle;
  int curChannel;
  int xLast, yLast;
  WireBundle lastBundle;
  Port lastPort = sourcePort;

  SmallVector<std::pair<int, int>, 4> congestion;

  llvm::dbgs() << "Build route: " << xSrc << " " << ySrc << " --> " << xDest << " " << yDest << '\n';
  llvm::dbgs() << "flowID " << flowID << '\n';
  // traverse horizontally, then vertically
  while (!((xCur == xDest) && (yCur == yDest))) {
    llvm::dbgs() << "coord " << xCur << " " << yCur << '\n';

    auto curCoord = std::make_pair(xCur, yCur);
    xLast = xCur;
    yLast = yCur;

    SmallVector<WireBundle, 4> moves;

    if (xCur < xDest)
      moves.push_back(WireBundle::East);
    if (xCur > xDest)
      moves.push_back(WireBundle::West);
    if (yCur < yDest)
      moves.push_back(WireBundle::North);
    if (yCur > yDest)
      moves.push_back(WireBundle::South);

    if (std::find(moves.begin(), moves.end(), WireBundle::East) == moves.end())
      moves.push_back(WireBundle::East);
    if (std::find(moves.begin(), moves.end(), WireBundle::West) == moves.end())
      moves.push_back(WireBundle::West);
    if (std::find(moves.begin(), moves.end(), WireBundle::North) == moves.end())
      moves.push_back(WireBundle::North);
    if (std::find(moves.begin(), moves.end(), WireBundle::South) == moves.end())
      moves.push_back(WireBundle::South);

    for (unsigned i = 0; i < moves.size(); i++) {
      WireBundle move = moves[i];
      curChannel = getAvailableDestChannel(switchboxes[curCoord], lastPort, flowID, move);
      if (curChannel == -1)
        continue;

      if (move == lastBundle)
        continue;

      if (move == WireBundle::East) {
        xCur = xCur + 1;
        yCur = yCur;
      } else if (move == WireBundle::West) {
        xCur = xCur - 1;
        yCur = yCur;
      } else if (move == WireBundle::North) {
        xCur = xCur;
        yCur = yCur + 1;
      } else if (move == WireBundle::South) {
        xCur = xCur;
        yCur = yCur - 1;
      }

      if (std::find(congestion.begin(), congestion.end(), std::make_pair(xCur, yCur)) != congestion.end())
        continue;

      curBundle = move;
      lastBundle = (move == WireBundle::East)  ? WireBundle::West :
                   (move == WireBundle::West)  ? WireBundle::East :
                   (move == WireBundle::North) ? WireBundle::South :
                   (move == WireBundle::South) ? WireBundle::North : lastBundle;
      break;
    }

    assert(curChannel >= 0 && "Could not find available destination port!");

    if (curChannel == -1) {
      congestion.push_back(std::make_pair(xLast, yLast)); // this switchbox is congested
      switchboxes[curCoord].pop_back(); // back up, remove the last connection
    } else {
      llvm::dbgs() << "[" << stringifyWireBundle(lastPort.first) << " : " << lastPort.second << "], "
                      "[" << stringifyWireBundle(curBundle) << " : " << curChannel << "]\n";

      Port curPort = std::make_pair(curBundle, curChannel);
      Connect connect = std::make_pair(lastPort, curPort);
      if (std::find(switchboxes[curCoord].begin(),
                    switchboxes[curCoord].end(),
                    std::make_pair(connect, flowID)) == switchboxes[curCoord].end())
        switchboxes[curCoord].push_back(std::make_pair(connect, flowID));
      lastPort = std::make_pair(lastBundle, curChannel);
    }
  }

  llvm::dbgs() << "coord " << xCur << " " << yCur << '\n';
  llvm::dbgs() << "[" << stringifyWireBundle(lastPort.first) << " : " << lastPort.second << "], "
                  "[" << stringifyWireBundle(destPort.first) << " : " << destPort.second << "]\n";

  switchboxes[std::make_pair(xCur, yCur)].push_back(
    std::make_pair(std::make_pair(lastPort, destPort), flowID));
}

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
    DenseMap<std::pair<Operation *, int>, SmallVector<Port, 4>> masterAMSels;

    for (auto tileOp : m.getOps<TileOp>()) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      tiles[std::make_pair(col, row)] = tileOp;
    }

    DenseMap<std::pair<int, int>, SmallVector<std::pair<Connect, int>, 8>> switchboxes;
    for (auto pktflow : m.getOps<PacketFlowOp>()) {
      Region &r = pktflow.ports();
      Block &b = r.front();
      int flowID = pktflow.IDInt();
      int xSrc, ySrc;
      Port sourcePort;

      for (Operation &Op : b.getOperations()) {
        if (PacketSourceOp pktSource = dyn_cast<PacketSourceOp>(Op)) {
          TileOp srcTile = dyn_cast<TileOp>(pktSource.tile().getDefiningOp());
          xSrc = srcTile.colIndex();
          ySrc = srcTile.rowIndex();
          sourcePort = pktSource.port();
        } else if (PacketDestOp pktDest = dyn_cast<PacketDestOp>(Op)) {
          TileOp destTile = dyn_cast<TileOp>(pktDest.tile().getDefiningOp());
          int xDest = destTile.colIndex();
          int yDest = destTile.rowIndex();
          Port destPort = pktDest.port();

          buildPSRoute(xSrc, ySrc, xDest, yDest, sourcePort, destPort, flowID, switchboxes);
        }
      }
    }

    llvm::dbgs() << "Check switchboxes\n";
     
    for (auto swbox : switchboxes) {
      int col = swbox.first.first;
      int row = swbox.first.second;
      Operation *tileOp = tiles[std::make_pair(col, row)];

      llvm::dbgs() << "***switchbox*** " << col << " " << row << '\n';
      SmallVector<std::pair<Connect, int>, 8> connects(swbox.second);
      for (auto connect : connects) {
        Port sourcePort = connect.first.first;
        Port destPort = connect.first.second;
        int flowID = connect.second;

        llvm::dbgs() << "sourcePort: " << stringifyWireBundle(sourcePort.first) << " " << sourcePort.second << '\n';
        llvm::dbgs() << "destPort: "   << stringifyWireBundle(destPort.first) << " " << destPort.second << '\n';
        llvm::dbgs() << "flowID " << flowID << '\n';

        auto sourceFlow = std::make_pair(std::make_pair(tileOp, sourcePort), flowID);
        packetFlows[sourceFlow].push_back(std::make_pair(tileOp, destPort));
        slavePorts.push_back(sourceFlow);
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
        SmallVector<Port, 4> ports(masterAMSels[std::make_pair(tileOp, amselValue)]);
        if (ports.size() != packetFlow.second.size())
          continue;

        bool matched = true;
        for (auto dest : packetFlow.second) {
          Port port = dest.second;
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
          Port port = dest.second;
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
      Port slavePort1 = slave1.first.second;

      bool foundgroup = false;
      for (auto &group : slavesGroups) {
        auto slave2 = group.front();
        Port slavePort2 = slave2.first.second;
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

    DenseMap<std::pair<PhysPort, int>, int> slaveMasks;
    for (auto group : slavesGroups) {
      // Iterate over all the ID values in a group
      // If bit n-th (n <= 5) of an ID value differs from bit n-th of another ID value,
      // the bit position should be "don't care", and we will set the mask bit of that position to 0
      int mask[5] = {-1, -1, -1, -1, -1};
      for (auto port : group) {
        int ID = port.second;
        for (int i = 0; i < 5; i++) {
          if (mask[i] == -1)
            mask[i] = ((ID >> i) & 0x1);
          else if (mask[i] != ((ID >> i) & 0x1))
            mask[i] = 2; // found bit difference --> mark as "don't care"
        }
      }

      int maskValue = 0;
      for (int i = 4; i >= 0; i--) {
        if (mask[i] == 2) // don't care
          mask[i] = 0;
        else
          mask[i] = 1;
        maskValue = (maskValue << 1) + mask[i];
      }
      for (auto port : group)
        slaveMasks[port] = maskValue;
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

      DenseMap<Port, PacketRulesOp> slaveRules;
      for (auto group : slavesGroups) {
        builder.setInsertionPoint(b.getTerminator());

        auto port = group.front().first;
        if (tileOp != port.first)
          continue;

        WireBundle bundle = port.second.first;
        int channel = port.second.second;
        auto slave = port.second;

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
