//===- AIECreatePacketFlows.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Attributes.h"
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
struct AIEOpRemoval : OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  using OpAdaptor = typename MyOp::Adaptor;

  explicit AIEOpRemoval(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit) {}

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

std::optional<int>
getAvailableDestChannel(SmallVector<std::pair<Connect, int>, 8> &connects,
                        WireBundle destBundle) {

  if (connects.empty())
    return {0};

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
    Port port = {destBundle, i};
    int countFlows = 0;
    for (const auto &[conn, _flowID] : connects) {
      // Since we are doing packet-switched routing, dest ports can be shared
      // among multiple sources. Therefore, we don't need to worry about
      // checking the same source
      if (conn.dst == port)
        countFlows++;
    }

    // Since a mask has 5 bits, there can only be 32 logical streams flow
    // through a port
    // TODO: what about packet-switched flow that uses nested header?
    if (countFlows > 0 && countFlows < 32)
      return {i};
  }

  // if not, look for available destination port
  for (int i = 0; i < numChannels; i++) {
    Port port = {destBundle, i};
    SmallVector<Port, 8> ports;
    for (const auto &[conn, _flowID] : connects)
      ports.push_back(conn.dst);

    if (std::find(ports.begin(), ports.end(), port) == ports.end())
      return {i};
  }

  return std::nullopt;
}

// Same function as above, but scanning from the last connect backwards
std::optional<int> getAvailableDestChannelReverseOrder(
    SmallVector<std::pair<Connect, int>, 8> &connects, WireBundle destBundle) {

  int numChannels;

  if (destBundle == WireBundle::North)
    numChannels = 6;
  else if (destBundle == WireBundle::South || destBundle == WireBundle::East ||
           destBundle == WireBundle::West)
    numChannels = 4;
  else
    numChannels = 2;

  if (connects.empty()) {
    assert(numChannels > 0 && "numChannels <= 0");
    return numChannels - 1;
  }

  for (int i = numChannels - 1; i >= 0; i--) {
    Port port = {destBundle, i};
    int countFlows = 0;
    for (const auto &[conn, _flowID] : connects) {
      if (Port connDest = conn.dst; connDest == port)
        countFlows++;
    }
    if (countFlows > 0 && countFlows < 32)
      return {i};
  }
  for (int i = numChannels - 1; i >= 0; i--) {
    Port port = {destBundle, i};
    SmallVector<Port, 8> ports;
    for (auto [connect, _] : connects)
      ports.push_back(connect.dst);

    if (std::find(ports.begin(), ports.end(), port) == ports.end())
      return {i};
  }

  return std::nullopt;
}

void updateCoordinates(int &xCur, int &yCur, WireBundle move) {
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
    TileOp srcTile, Port sourcePort, TileOp destTile, Port destPort, int flowID,
    DenseMap<TileID, SmallVector<std::pair<Connect, int>, 8>> &switchboxes,
    bool reverseOrder = false) {

  int xSrc = srcTile.colIndex();
  int ySrc = srcTile.rowIndex();
  int xDest = destTile.colIndex();
  int yDest = destTile.rowIndex();

  const auto &targetModel = getTargetModel(srcTile);

  int xCur = xSrc;
  int yCur = ySrc;
  WireBundle curBundle = {};
  int curChannel = 0;
  WireBundle lastBundle = {};
  Port lastPort = sourcePort;

  SmallVector<TileID, 4> congestion;

  LLVM_DEBUG(llvm::dbgs() << "Build route ID " << flowID << ": " << xSrc << " "
                          << ySrc << " --> " << xDest << " " << yDest << '\n');
  // traverse horizontally, then vertically
  while (!(xCur == xDest && yCur == yDest)) {
    LLVM_DEBUG(llvm::dbgs() << "Tile " << xCur << " " << yCur << " ");

    TileID curCoord = {xCur, yCur};
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

    for (auto move : moves) {
      if (reverseOrder) {
        if (auto maybeDestChannel = getAvailableDestChannelReverseOrder(
                switchboxes[curCoord], move))
          curChannel = maybeDestChannel.value();
        else
          continue;
      } else if (auto maybeDestChannel =
                     getAvailableDestChannel(switchboxes[curCoord], move))
        curChannel = maybeDestChannel.value();
      else
        continue;

      if (move == lastBundle)
        continue;

      // If the source port is a trace port, we need to validate the destination
      if (xCur == xSrc && yCur == ySrc &&
          sourcePort.bundle == WireBundle::Trace &&
          !targetModel.isValidTraceMaster(xSrc, ySrc, move, curChannel)) {
        continue;
      }

      updateCoordinates(xCur, yCur, move);

      if (std::find(congestion.begin(), congestion.end(), TileID{xCur, yCur}) !=
          congestion.end())
        continue;

      curBundle = move;
      lastBundle = move == WireBundle::East    ? WireBundle::West
                   : move == WireBundle::West  ? WireBundle::East
                   : move == WireBundle::North ? WireBundle::South
                   : move == WireBundle::South ? WireBundle::North
                                               : lastBundle;
      break;
    }

    assert(curChannel >= 0 && "Could not find available destination port!");

    LLVM_DEBUG(llvm::dbgs()
               << stringifyWireBundle(lastPort.bundle) << " "
               << lastPort.channel << " -> " << stringifyWireBundle(curBundle)
               << " " << curChannel << "\n");

    Port curPort = {curBundle, curChannel};
    // If there is no connection with this ID going where we want to go.
    if (Connect connect = {lastPort, curPort};
        std::find(switchboxes[curCoord].begin(), switchboxes[curCoord].end(),
                  std::pair{connect, flowID}) == switchboxes[curCoord].end())
      // then add one.
      switchboxes[curCoord].push_back({connect, flowID});
    lastPort = {lastBundle, curChannel};
  }

  LLVM_DEBUG(llvm::dbgs() << "Tile " << xCur << " " << yCur << " ");
  LLVM_DEBUG(llvm::dbgs() << stringifyWireBundle(lastPort.bundle) << " "
                          << lastPort.channel << " -> "
                          << stringifyWireBundle(curBundle) << " " << curChannel
                          << "\n");

  switchboxes[{xCur, yCur}].push_back(
      std::make_pair(Connect{lastPort, destPort}, flowID));
}

SwitchboxOp getOrCreateSwitchbox(OpBuilder &builder, TileOp tile) {
  for (auto i : tile.getResult().getUsers()) {
    if (llvm::isa<SwitchboxOp>(*i)) {
      return llvm::cast<SwitchboxOp>(*i);
    }
  }
  return builder.create<SwitchboxOp>(builder.getUnknownLoc(), tile);
}
struct AIERoutePacketFlowsPass
    : AIERoutePacketFlowsBase<AIERoutePacketFlowsPass> {
  // Map from tile coordinates to TileOp
  DenseMap<TileID, Operation *> tiles;
  Operation *getOrCreateTile(OpBuilder &builder, int col, int row) {
    TileID index = {col, row};
    Operation *tileOp = tiles[index];
    if (!tileOp) {
      auto tile = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
      tileOp = tile.getOperation();
      tiles[index] = tileOp;
    }
    return tileOp;
  }
  void runOnOperation() override {

    DeviceOp device = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(device.getBody());

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
    // Map from a port to
    DenseMap<PhysPort, Attribute> keepPktHeaderAttr;

    for (auto tileOp : device.getOps<TileOp>()) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      tiles[{col, row}] = tileOp;
    }

    // The logical model of all the switchboxes.
    DenseMap<TileID, SmallVector<std::pair<Connect, int>, 8>> switchboxes;
    for (auto pktflow : device.getOps<PacketFlowOp>()) {
      Region &r = pktflow.getPorts();
      Block &b = r.front();
      int flowID = pktflow.IDInt();
      Port sourcePort, destPort;
      TileOp srcTile, destTile;

      for (Operation &Op : b.getOperations()) {
        if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
          srcTile = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
          sourcePort = pktSource.port();
        } else if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
          destTile = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
          destPort = pktDest.port();

          buildPSRoute(srcTile, sourcePort, destTile, destPort, flowID,
                       switchboxes, true);

          // Assign "keep_pkt_header flag"
          if (pktflow->hasAttr("keep_pkt_header"))
            keepPktHeaderAttr[{destTile, destPort}] =
                StringAttr::get(Op.getContext(), "true");
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Check switchboxes\n");

    for (const auto &[tileId, connects] : switchboxes) {
      int col = tileId.col;
      int row = tileId.row;
      Operation *tileOp = getOrCreateTile(builder, col, row);

      LLVM_DEBUG(llvm::dbgs()
                 << "***switchbox*** " << col << " " << row << '\n');
      for (const auto &[conn, flowID] : connects) {
        Port sourcePort = conn.src;
        Port destPort = conn.dst;
        int nextCol = col, nextRow = row;
        updateCoordinates(nextCol, nextRow, sourcePort.bundle);
        LLVM_DEBUG(llvm::dbgs() << "flowID " << flowID << ':'
                                << stringifyWireBundle(sourcePort.bundle) << " "
                                << sourcePort.channel << " -> "
                                << stringifyWireBundle(destPort.bundle) << " "
                                << destPort.channel << " tile " << nextCol
                                << " " << nextRow << "\n");

        auto sourceFlow =
            std::make_pair(std::make_pair(tileOp, sourcePort), flowID);
        packetFlows[sourceFlow].push_back({tileOp, destPort});
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
    for (const auto &packetFlow : packetFlows) {
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
      for (const auto &map : masterAMSels) {
        if (map.first.first != tileOp)
          continue;
        amselValue = map.first.second;

        // check if same destinations
        SmallVector<Port, 4> ports(masterAMSels[{tileOp, amselValue}]);
        if (ports.size() != packetFlow.second.size())
          continue;

        bool matched = true;
        for (auto dest : packetFlow.second) {
          if (Port port = dest.second;
              std::find(ports.begin(), ports.end(), port) == ports.end()) {
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
            if (masterAMSels.count({tileOp, amselValue}) == 0) {
              foundAMSelValue = true;
              break;
            }
          }

          if (foundAMSelValue)
            break;
        }

        for (auto dest : packetFlow.second) {
          Port port = dest.second;
          masterAMSels[{tileOp, amselValue}].push_back(port);
        }
      }

      slaveAMSels[packetFlow.first] = amselValue;
      amselValues[tileOp] = amselValue % numArbiters;
    }

    // Compute the master set IDs
    // A map from a switchbox output port to the number of that port.
    DenseMap<PhysPort, SmallVector<int, 4>> mastersets;
    for (const auto &[physPort, ports] : masterAMSels) {
      Operation *tileOp = physPort.first;
      assert(tileOp);
      int amselValue = physPort.second;
      for (auto port : ports) {
        PhysPort physPort = {tileOp, port};
        mastersets[physPort].push_back(amselValue);
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "CHECK mastersets\n");
#ifndef NDEBUG
    for (const auto &[physPort, values] : mastersets) {
      Operation *tileOp = physPort.first;
      WireBundle bundle = physPort.second.bundle;
      int channel = physPort.second.channel;
      assert(tileOp);
      auto tile = dyn_cast<TileOp>(tileOp);
      LLVM_DEBUG(llvm::dbgs()
                 << "master " << tile << " " << stringifyWireBundle(bundle)
                 << " : " << channel << '\n');
      for (auto value : values)
        LLVM_DEBUG(llvm::dbgs() << "amsel: " << value << '\n');
    }
#endif

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
        if (Port slavePort2 = slave2.first.second; slavePort1 != slavePort2)
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
    for (const auto &group : slaveGroups) {
      // Iterate over all the ID values in a group
      // If bit n-th (n <= 5) of an ID value differs from bit n-th of another ID
      // value, the bit position should be "don't care", and we will set the
      // mask bit of that position to 0
      int mask[5] = {-1, -1, -1, -1, -1};
      for (auto port : group) {
        int ID = port.second;
        for (int i = 0; i < 5; i++) {
          if (mask[i] == -1)
            mask[i] = ID >> i & 0x1;
          else if (mask[i] != (ID >> i & 0x1))
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

#ifndef NDEBUG
    LLVM_DEBUG(llvm::dbgs() << "CHECK Slave Masks\n");
    for (auto map : slaveMasks) {
      auto port = map.first.first;
      auto tile = dyn_cast<TileOp>(port.first);
      WireBundle bundle = port.second.bundle;
      int channel = port.second.channel;
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
#endif

    // Realize the routes in MLIR
    for (auto map : tiles) {
      Operation *tileOp = map.second;
      auto tile = dyn_cast<TileOp>(tileOp);

      // Create a switchbox for the routes and insert inside it.
      builder.setInsertionPointAfter(tileOp);
      SwitchboxOp swbox = getOrCreateSwitchbox(builder, tile);
      SwitchboxOp::ensureTerminator(swbox.getConnections(), builder,
                                    builder.getUnknownLoc());
      Block &b = swbox.getConnections().front();
      builder.setInsertionPoint(b.getTerminator());

      std::vector<bool> amselOpNeededVector(32);
      for (const auto &map : mastersets) {
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
          auto amsel =
              builder.create<AMSelOp>(builder.getUnknownLoc(), arbiterID, msel);
          amselOps[i] = amsel;
        }
      }
      // Create all the master set Ops
      // First collect the master sets for this tile.
      SmallVector<Port, 4> tileMasters;
      for (const auto &map : mastersets) {
        if (tileOp != map.first.first)
          continue;
        tileMasters.push_back(map.first.second);
      }
      // Sort them so we get a reasonable order
      std::sort(tileMasters.begin(), tileMasters.end());
      for (auto tileMaster : tileMasters) {
        WireBundle bundle = tileMaster.bundle;
        int channel = tileMaster.channel;
        SmallVector<int, 4> msels = mastersets[{tileOp, tileMaster}];
        SmallVector<Value, 4> amsels;
        for (auto msel : msels) {
          assert(amselOps.count(msel) == 1);
          amsels.push_back(amselOps[msel]);
        }

        auto msOp = builder.create<MasterSetOp>(builder.getUnknownLoc(),
                                                builder.getIndexType(), bundle,
                                                channel, amsels);
        if (auto pktFlowAttrs = keepPktHeaderAttr[{tileOp, tileMaster}])
          msOp->setAttr("keep_pkt_header", pktFlowAttrs);
      }

      // Generate the packet rules
      DenseMap<Port, PacketRulesOp> slaveRules;
      for (auto group : slaveGroups) {
        builder.setInsertionPoint(b.getTerminator());

        auto port = group.front().first;
        if (tileOp != port.first)
          continue;

        WireBundle bundle = port.second.bundle;
        int channel = port.second.channel;
        auto slave = port.second;

        int mask = slaveMasks[group.front()];
        int ID = group.front().second & mask;

        // Verify that we actually map all the ID's correctly.
#ifndef NDEBUG
        for (auto slave : group)
          assert((slave.second & mask) == ID);
#endif
        Value amsel = amselOps[slaveAMSels[group.front()]];

        PacketRulesOp packetrules;
        if (slaveRules.count(slave) == 0) {
          packetrules = builder.create<PacketRulesOp>(builder.getUnknownLoc(),
                                                      bundle, channel);
          PacketRulesOp::ensureTerminator(packetrules.getRules(), builder,
                                          builder.getUnknownLoc());
          slaveRules[slave] = packetrules;
        } else
          packetrules = slaveRules[slave];

        Block &rules = packetrules.getRules().front();
        builder.setInsertionPoint(rules.getTerminator());
        builder.create<PacketRuleOp>(builder.getUnknownLoc(), mask, ID, amsel);
      }
    }

    // Add support for shimDMA
    // From shimDMA to BLI: 1) shimDMA 0 --> North 3
    //                      2) shimDMA 1 --> North 7
    // From BLI to shimDMA: 1) North   2 --> shimDMA 0
    //                      2) North   3 --> shimDMA 1

    for (auto switchbox : make_early_inc_range(device.getOps<SwitchboxOp>())) {
      auto retVal = switchbox->getOperand(0);
      auto tileOp = retVal.getDefiningOp<TileOp>();

      // Check if it is a shim Tile
      if (!tileOp.isShimNOCTile())
        continue;

      // Check if the switchbox is empty
      if (&switchbox.getBody()->front() == switchbox.getBody()->getTerminator())
        continue;

      Region &r = switchbox.getConnections();
      Block &b = r.front();

      // Find if the corresponding shimmux exsists or not
      int shimExist = 0;
      ShimMuxOp shimOp;
      for (auto shimmux : device.getOps<ShimMuxOp>()) {
        if (shimmux.getTile() == tileOp) {
          shimExist = 1;
          shimOp = shimmux;
          break;
        }
      }

      for (Operation &Op : b.getOperations()) {
        if (auto pktrules = dyn_cast<PacketRulesOp>(Op)) {

          // check if there is MM2S DMA in the switchbox of the 0th row
          if (pktrules.getSourceBundle() == WireBundle::DMA) {

            // If there is, then it should be put into the corresponding shimmux
            // If shimmux not defined then create shimmux
            if (!shimExist) {
              builder.setInsertionPointAfter(tileOp);
              shimOp =
                  builder.create<ShimMuxOp>(builder.getUnknownLoc(), tileOp);
              Region &r1 = shimOp.getConnections();
              Block *b1 = builder.createBlock(&r1);
              builder.setInsertionPointToEnd(b1);
              builder.create<EndOp>(builder.getUnknownLoc());
              shimExist = 1;
            }

            Region &r0 = shimOp.getConnections();
            Block &b0 = r0.front();
            builder.setInsertionPointToStart(&b0);

            pktrules.setSourceBundle(WireBundle::South);
            if (pktrules.getSourceChannel() == 0) {
              pktrules.setSourceChannel(3);
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::DMA, 0, WireBundle::North,
                                        3);
            }
            if (pktrules.getSourceChannel() == 1) {
              pktrules.setSourceChannel(7);
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::DMA, 1, WireBundle::North,
                                        7);
            }
          }
        }

        if (auto mtset = dyn_cast<MasterSetOp>(Op)) {

          // check if there is S2MM DMA in the switchbox of the 0th row
          if (mtset.getDestBundle() == WireBundle::DMA) {

            // If there is, then it should be put into the corresponding shimmux
            // If shimmux not defined then create shimmux
            if (!shimExist) {
              builder.setInsertionPointAfter(tileOp);
              shimOp =
                  builder.create<ShimMuxOp>(builder.getUnknownLoc(), tileOp);
              Region &r1 = shimOp.getConnections();
              Block *b1 = builder.createBlock(&r1);
              builder.setInsertionPointToEnd(b1);
              builder.create<EndOp>(builder.getUnknownLoc());
              shimExist = 1;
            }

            Region &r0 = shimOp.getConnections();
            Block &b0 = r0.front();
            builder.setInsertionPointToStart(&b0);

            mtset.setDestBundle(WireBundle::South);
            if (mtset.getDestChannel() == 0) {
              mtset.setDestChannel(2);
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::North, 2, WireBundle::DMA,
                                        0);
            }
            if (mtset.getDestChannel() == 1) {
              mtset.setDestChannel(3);
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::North, 3, WireBundle::DMA,
                                        1);
            }
          }
        }
      }
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<AIEOpRemoval<PacketFlowOp>>(device.getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIERoutePacketFlowsPass() {
  return std::make_unique<AIERoutePacketFlowsPass>();
}
