//===- AIECreatePacketFlows.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/AIEDialect.h"
#include "aie/AIENetlistAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"

#define DEBUG_TYPE "aie-create-packet-flows"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

template <typename MyOp>
struct AIEOpRemoval : public OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  using OpAdaptor = typename MyOp::Adaptor;
  ModuleOp &module;

  AIEOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit), module(m) {}

  LogicalResult
  matchAndRewrite(MyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    rewriter.eraseOp(Op);
    return success();
  }
};

// A port on a switch is identified by the tile and port name.
typedef std::pair<Operation *, Port> PhysPort;

int getAvailableDestChannel(SmallVector<std::pair<Connect, int>, 8> &connects,
                            Port sourcePort, int flowID,
                            WireBundle destBundle) {

  if (connects.size() == 0)
    return 0;

  int numChannels;

  if (destBundle == WireBundle::North)
    numChannels = 6;
  else if (destBundle == WireBundle::South || destBundle == WireBundle::East ||
           destBundle == WireBundle::West)
    numChannels = 4;
  else
    numChannels = 2;

  // look for existing connect that has a matching destination
  for (int i = 0; i < numChannels; i++) {
    Port port = std::make_pair(destBundle, i);
    int countFlows = 0;
    for (auto conn : connects) {
      Port connDest = conn.first.second;
      // Since we are doing packet-switched routing, dest ports can be shared
      // among multiple sources. Therefore, we don't need to worry about
      // checking the same source
      if (connDest == port)
        countFlows++;
    }

    // Since a mask has 5 bits, there can only be 32 logical streams flow
    // through a port
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

void update_coordinates(int &xCur, int &yCur, WireBundle move) {
  if (move == WireBundle::East) {
    xCur = xCur + 1;
    // yCur = yCur;
  } else if (move == WireBundle::West) {
    xCur = xCur - 1;
    // yCur = yCur;
  } else if (move == WireBundle::North) {
    // xCur = xCur;
    yCur = yCur + 1;
  } else if (move == WireBundle::South) {
    // xCur = xCur;
    yCur = yCur - 1;
  }
}

// Build a packet-switched route from the sourse to the destination with the
// given ID. The route is recorded in the given map of switchboxes.
void buildPSRoute(
    int xSrc, int ySrc, Port sourcePort, int xDest, int yDest, Port destPort,
    int flowID,
    DenseMap<std::pair<int, int>, SmallVector<std::pair<Connect, int>, 8>>
        &switchboxes) {
  int xCur = xSrc;
  int yCur = ySrc;
  WireBundle curBundle;
  int curChannel;
  int xLast, yLast;
  WireBundle lastBundle;
  Port lastPort = sourcePort;

  SmallVector<std::pair<int, int>, 4> congestion;

  LLVM_DEBUG(llvm::dbgs() << "Build route ID " << flowID << ": " << xSrc << " "
                          << ySrc << " --> " << xDest << " " << yDest << '\n');
  // traverse horizontally, then vertically
  while (!((xCur == xDest) && (yCur == yDest))) {
    LLVM_DEBUG(llvm::dbgs() << "Tile " << xCur << " " << yCur << " ");

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
      curChannel = getAvailableDestChannel(switchboxes[curCoord], lastPort,
                                           flowID, move);
      if (curChannel == -1)
        continue;

      if (move == lastBundle)
        continue;

      update_coordinates(xCur, yCur, move);

      if (std::find(congestion.begin(), congestion.end(),
                    std::make_pair(xCur, yCur)) != congestion.end())
        continue;

      curBundle = move;
      lastBundle = (move == WireBundle::East)    ? WireBundle::West
                   : (move == WireBundle::West)  ? WireBundle::East
                   : (move == WireBundle::North) ? WireBundle::South
                   : (move == WireBundle::South) ? WireBundle::North
                                                 : lastBundle;
      break;
    }

    assert(curChannel >= 0 && "Could not find available destination port!");

    if (curChannel == -1) {
      congestion.push_back(
          std::make_pair(xLast, yLast)); // this switchbox is congested
      switchboxes[curCoord].pop_back();  // back up, remove the last connection
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << stringifyWireBundle(lastPort.first) << " "
                 << lastPort.second << " -> " << stringifyWireBundle(curBundle)
                 << " " << curChannel << "\n");

      Port curPort = std::make_pair(curBundle, curChannel);
      Connect connect = std::make_pair(lastPort, curPort);
      // If there is no connection with this ID going where we want to go..
      if (std::find(switchboxes[curCoord].begin(), switchboxes[curCoord].end(),
                    std::make_pair(connect, flowID)) ==
          switchboxes[curCoord].end())
        // then add one.
        switchboxes[curCoord].push_back(std::make_pair(connect, flowID));
      lastPort = std::make_pair(lastBundle, curChannel);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Tile " << xCur << " " << yCur << " ");
  LLVM_DEBUG(llvm::dbgs() << stringifyWireBundle(lastPort.first) << " "
                          << lastPort.second << " -> "
                          << stringifyWireBundle(curBundle) << " " << curChannel
                          << "\n");

  switchboxes[std::make_pair(xCur, yCur)].push_back(
      std::make_pair(std::make_pair(lastPort, destPort), flowID));
}

SwitchboxOp getOrCreateSwitchbox(OpBuilder &builder, TileOp tile) {
  for (auto i : tile.result().getUsers()) {
    if (llvm::isa<SwitchboxOp>(*i)) {
      return llvm::cast<SwitchboxOp>(*i);
    }
  }
  return builder.create<SwitchboxOp>(builder.getUnknownLoc(), tile);
}
struct AIERoutePacketFlowsPass
    : public AIERoutePacketFlowsBase<AIERoutePacketFlowsPass> {
  // Map from tile coordinates to TileOp
  DenseMap<std::pair<int, int>, Operation *> tiles;
  Operation *getOrCreateTile(OpBuilder &builder, int col, int row) {
    auto index = std::make_pair(col, row);
    Operation *tileOp = tiles[index];
    if (!tileOp) {
      auto tile = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
      tileOp = tile.getOperation();
      tiles[index] = tileOp;
    }
    return tileOp;
  }
  void runOnOperation() override {

    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

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
    // We will use circuit-switch to pre-route the flows from the source swboxes
    // to the dest swboxes, and only use packet-switch to route at the dest
    // swboxes

    // Map from a port and flowID to
    DenseMap<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> packetFlows;
    SmallVector<std::pair<PhysPort, int>, 4> slavePorts;
    DenseMap<std::pair<PhysPort, int>, int> slaveAMSels;

    for (auto tileOp : m.getOps<TileOp>()) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      tiles[std::make_pair(col, row)] = tileOp;
    }

    // The logical model of all the switchboxes.
    DenseMap<std::pair<int, int>, SmallVector<std::pair<Connect, int>, 8>>
        switchboxes;
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

          buildPSRoute(xSrc, ySrc, sourcePort, xDest, yDest, destPort, flowID,
                       switchboxes);
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Check switchboxes\n");

    for (auto swbox : switchboxes) {
      int col = swbox.first.first;
      int row = swbox.first.second;
      Operation *tileOp = getOrCreateTile(builder, col, row);

      LLVM_DEBUG(llvm::dbgs()
                 << "***switchbox*** " << col << " " << row << '\n');
      SmallVector<std::pair<Connect, int>, 8> connects(swbox.second);
      for (auto connect : connects) {
        Port sourcePort = connect.first.first;
        Port destPort = connect.first.second;
        int flowID = connect.second;

        int nextCol = col, nextRow = row;
        update_coordinates(nextCol, nextRow, sourcePort.first);
        LLVM_DEBUG(llvm::dbgs() << "flowID " << flowID << ':'
                                << stringifyWireBundle(sourcePort.first) << " "
                                << sourcePort.second << " -> "
                                << stringifyWireBundle(destPort.first) << " "
                                << destPort.second << " tile " << nextCol << " "
                                << nextRow << "\n");

        auto sourceFlow =
            std::make_pair(std::make_pair(tileOp, sourcePort), flowID);
        packetFlows[sourceFlow].push_back(std::make_pair(tileOp, destPort));
        slavePorts.push_back(sourceFlow);
      }
    }

    // amsel()
    // masterset()
    // packetrules()
    // rule()

    // Compute arbiter assignments. Each arbiter has four msels.
    // Therefore, the number of "logical" arbiters is 6 x 4 = 24
    // A master port can only be associated with one arbiter

    // A map from Tile and master selectValue to the ports targetted by that
    // master select.
    DenseMap<std::pair<Operation *, int>, SmallVector<Port, 4>> masterAMSels;

    // Count of currently used logical arbiters for each tile.
    DenseMap<Operation *, int> amselValues;
    int numMsels = 4;
    int numArbiters = 6;

    // Check all multi-cast flows (same source, same ID). They should be
    // assigned the same arbiter and msel so that the flow can reach all the
    // destination ports at the same time For destination ports that appear in
    // different (multicast) flows, it should have a different <arbiterID, msel>
    // value pair for each flow
    for (auto packetFlow : packetFlows) {
      // The Source Tile of the flow
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
      // If there is an assignment of an arbiter to a master port before, we
      // assign all the master ports here with the same arbiter but different
      // msel
      bool foundMatchedDest = false;
      for (auto map : masterAMSels) {
        if (map.first.first != tileOp)
          continue;
        amselValue = map.first.second;

        // check if same destinations
        // SmallVector<Port, 4> ports(map.second);
        SmallVector<Port, 4> ports(
            masterAMSels[std::make_pair(tileOp, amselValue)]);
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
        bool foundAMSelValue = false;
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

    // Compute the master set IDs
    // A map from a switchbox output port to the number of that port.
    DenseMap<PhysPort, SmallVector<int, 4>> mastersets;
    for (auto master : masterAMSels) {
      Operation *tileOp = master.first.first;
      assert(tileOp);
      int amselValue = master.first.second;
      for (auto port : master.second) {
        auto physPort = std::make_pair(tileOp, port);
        mastersets[physPort].push_back(amselValue);
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "CHECK mastersets\n");
    for (auto map : mastersets) {
      Operation *tileOp = map.first.first;
      WireBundle bundle = map.first.second.first;
      int channel = map.first.second.second;
      assert(tileOp);
      TileOp tile = dyn_cast<TileOp>(tileOp);
      LLVM_DEBUG(llvm::dbgs()
                 << "master " << tile << " " << stringifyWireBundle(bundle)
                 << " : " << channel << '\n');
      for (auto value : map.second)
        LLVM_DEBUG(llvm::dbgs() << "amsel: " << value << '\n');
    }

    // Compute mask values
    // Merging as many stream flows as possible
    // The flows must originate from the same source port and have different IDs
    // Two flows can be merged if they share the same destinations
    SmallVector<SmallVector<std::pair<PhysPort, int>, 4>, 4> slaveGroups;
    SmallVector<std::pair<PhysPort, int>, 4> workList(slavePorts);
    while (!workList.empty()) {
      auto slave1 = workList.pop_back_val();
      Port slavePort1 = slave1.first.second;

      bool foundgroup = false;
      for (auto &group : slaveGroups) {
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
        slaveGroups.push_back(group);
      }
    }

    DenseMap<std::pair<PhysPort, int>, int> slaveMasks;
    for (auto group : slaveGroups) {
      // Iterate over all the ID values in a group
      // If bit n-th (n <= 5) of an ID value differs from bit n-th of another ID
      // value, the bit position should be "don't care", and we will set the
      // mask bit of that position to 0
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

    LLVM_DEBUG(llvm::dbgs() << "CHECK Slave Masks\n");
    for (auto map : slaveMasks) {
      auto port = map.first.first;
      TileOp tile = dyn_cast<TileOp>(port.first);
      WireBundle bundle = port.second.first;
      int channel = port.second.second;
      int ID = map.first.second;
      int mask = map.second;

      LLVM_DEBUG(llvm::dbgs()
                 << "Port " << tile << " " << stringifyWireBundle(bundle) << " "
                 << channel << '\n');
      LLVM_DEBUG(llvm::dbgs() << "Mask "
                              << "0x" << llvm::Twine::utohexstr(mask) << '\n');
      LLVM_DEBUG(llvm::dbgs() << "ID "
                              << "0x" << llvm::Twine::utohexstr(ID) << '\n');
      for (int i = 0; i < 31; i++) {
        if ((i & mask) == (ID & mask))
          LLVM_DEBUG(llvm::dbgs() << "matches flow ID "
                                  << "0x" << llvm::Twine::utohexstr(i) << '\n');
      }
    }

    // Realize the routes in MLIR
    for (auto map : tiles) {
      Operation *tileOp = map.second;
      TileOp tile = dyn_cast<TileOp>(tileOp);

      // Create a switchbox for the routes and insert inside it.
      builder.setInsertionPointAfter(tileOp);
      SwitchboxOp swbox = getOrCreateSwitchbox(builder, tile);
      swbox.ensureTerminator(swbox.connections(), builder,
                             builder.getUnknownLoc());
      Block &b = swbox.connections().front();
      builder.setInsertionPoint(b.getTerminator());

      std::vector<bool> amselOpNeededVector(32);
      for (auto map : mastersets) {
        if (tileOp != map.first.first)
          continue;

        for (auto value : map.second) {
          amselOpNeededVector[value] = true;
        }
      }
      // Create all the amsel Ops
      DenseMap<int, AMSelOp> amselOps;
      for (int i = 0; i < 32; i++) {
        if (amselOpNeededVector[i]) {
          int arbiterID = i % numArbiters;
          int msel = i / numArbiters;
          AMSelOp amsel =
              builder.create<AMSelOp>(builder.getUnknownLoc(), arbiterID, msel);
          amselOps[i] = amsel;
        }
      }
      // Create all the master set Ops
      // First collect the master sets for this tile.
      SmallVector<Port, 4> tileMasters;
      for (auto map : mastersets) {
        if (tileOp != map.first.first)
          continue;
        tileMasters.push_back(map.first.second);
      }
      // Sort them so we get a reasonable order
      std::sort(tileMasters.begin(), tileMasters.end());
      for (auto tileMaster : tileMasters) {
        WireBundle bundle = tileMaster.first;
        int channel = tileMaster.second;
        SmallVector<int, 4> msels =
            mastersets[std::make_pair(tileOp, tileMaster)];
        SmallVector<Value, 4> amsels;
        for (auto msel : msels) {
          assert(amselOps.count(msel) == 1);
          amsels.push_back(amselOps[msel]);
        }

        builder.create<MasterSetOp>(builder.getUnknownLoc(),
                                    builder.getIndexType(), bundle, channel,
                                    amsels);
      }

      // Generate the packet rules
      DenseMap<Port, PacketRulesOp> slaveRules;
      for (auto group : slaveGroups) {
        builder.setInsertionPoint(b.getTerminator());

        auto port = group.front().first;
        if (tileOp != port.first)
          continue;

        WireBundle bundle = port.second.first;
        int channel = port.second.second;
        auto slave = port.second;

        int mask = slaveMasks[group.front()];
        int ID = group.front().second & mask;

        // Verify that we actually map all the ID's correctly.
        for (auto slave : group) {
          assert((slave.second & mask) == ID);
        }
        Value amsel = amselOps[slaveAMSels[group.front()]];

        PacketRulesOp packetrules;
        if (slaveRules.count(slave) == 0) {
          packetrules = builder.create<PacketRulesOp>(builder.getUnknownLoc(),
                                                      bundle, channel);
          packetrules.ensureTerminator(packetrules.rules(), builder,
                                       builder.getUnknownLoc());
          slaveRules[slave] = packetrules;
        } else
          packetrules = slaveRules[slave];

        Block &rules = packetrules.rules().front();
        builder.setInsertionPoint(rules.getTerminator());
        builder.create<PacketRuleOp>(builder.getUnknownLoc(), mask, ID, amsel);
      }
    }

    // Add support for shimDMA
    // From shimDMA to BLI: 1) shimDMA 0 --> North 3
    //                      2) shimDMA 1 --> North 7
    // From BLI to shimDMA: 1) North   2 --> shimDMA 0
    //                      2) North   3 --> shimDMA 1

    for (auto switchbox : llvm::make_early_inc_range(m.getOps<SwitchboxOp>())) {
      auto retVal = switchbox->getOperand(0);
      auto tileOp = retVal.getDefiningOp<TileOp>();

      // Check if it is a shim Tile
      if (!tileOp.isShimNOCTile())
        continue;

      // Check if it the switchbox is empty
      if (&switchbox.getBody()->front() == switchbox.getBody()->getTerminator())
        continue;

      Region &r = switchbox.connections();
      Block &b = r.front();

      // Find if the corresponding shimmux exsists or not
      int shim_exist = 0;
      ShimMuxOp shimOp;
      for (auto shimmux : m.getOps<ShimMuxOp>()) {
        if (shimmux.tile() == tileOp) {
          shim_exist = 1;
          shimOp = shimmux;
          break;
        }
      }

      for (Operation &Op : b.getOperations()) {
        if (PacketRulesOp pktrules = dyn_cast<PacketRulesOp>(Op)) {

          // check if there is MM2S DMA in the switchbox of the 0th row
          if (pktrules.sourceBundle() == WireBundle::DMA) {

            // If there is, then it should be put into the corresponding shimmux
            // If shimmux not defined then create shimmux
            if (!shim_exist) {
              builder.setInsertionPointAfter(tileOp);
              shimOp =
                  builder.create<ShimMuxOp>(builder.getUnknownLoc(), tileOp);
              Region &r1 = shimOp.connections();
              Block *b1 = builder.createBlock(&r1);
              builder.setInsertionPointToEnd(b1);
              builder.create<EndOp>(builder.getUnknownLoc());
              shim_exist = 1;
            }

            Region &r0 = shimOp.connections();
            Block &b0 = r0.front();
            builder.setInsertionPointToStart(&b0);

            pktrules->removeAttr("sourceBundle");
            pktrules->setAttr(
                "sourceBundle",
                builder.getI32IntegerAttr(3)); // WireBundle::South
            if (pktrules.sourceChannel() == 0) {
              pktrules->removeAttr("sourceChannel");
              pktrules->setAttr("sourceChannel",
                                builder.getI32IntegerAttr(3)); // Channel 3
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::DMA, 0, WireBundle::North,
                                        3);
            }
            if (pktrules.sourceChannel() == 1) {
              pktrules->removeAttr("sourceChannel");
              pktrules->setAttr("sourceChannel",
                                builder.getI32IntegerAttr(7)); // Channel 7
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::DMA, 1, WireBundle::North,
                                        7);
            }
          }
        }

        if (MasterSetOp mtset = dyn_cast<MasterSetOp>(Op)) {

          // check if there is S2MM DMA in the switchbox of the 0th row
          if (mtset.destBundle() == WireBundle::DMA) {

            // If there is, then it should be put into the corresponding shimmux
            // If shimmux not defined then create shimmux
            if (!shim_exist) {
              builder.setInsertionPointAfter(tileOp);
              shimOp =
                  builder.create<ShimMuxOp>(builder.getUnknownLoc(), tileOp);
              Region &r1 = shimOp.connections();
              Block *b1 = builder.createBlock(&r1);
              builder.setInsertionPointToEnd(b1);
              builder.create<EndOp>(builder.getUnknownLoc());
              shim_exist = 1;
            }

            Region &r0 = shimOp.connections();
            Block &b0 = r0.front();
            builder.setInsertionPointToStart(&b0);

            mtset->removeAttr("destBundle");
            mtset->setAttr("destBundle",
                           builder.getI32IntegerAttr(3)); // WireBundle::South
            if (mtset.destChannel() == 0) {
              mtset->removeAttr("destChannel");
              mtset->setAttr("destChannel",
                             builder.getI32IntegerAttr(2)); // Channel 2
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::North, 2, WireBundle::DMA,
                                        0);
            }
            if (mtset.destChannel() == 1) {
              mtset->removeAttr("destChannel");
              mtset->setAttr("destChannel",
                             builder.getI32IntegerAttr(3)); // Channel 3
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::North, 3, WireBundle::DMA,
                                        1);
            }
          }
        }
      }
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<AIEOpRemoval<PacketFlowOp>>(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIERoutePacketFlowsPass() {
  return std::make_unique<AIERoutePacketFlowsPass>();
}
