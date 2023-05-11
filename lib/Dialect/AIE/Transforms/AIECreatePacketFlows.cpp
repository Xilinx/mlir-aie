//===- AIECreatePacketFlows.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/AIENetlistAnalysis.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"
#include <aie/Dialect/AIE/Transforms/AIEPathfinder.h>

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

int computeMaskValueBetter(SmallVector<std::pair<PhysPort, int>, 4> group) {
  // compute masks using bitwise XOR operation ^
  // std::bitset<4> name ("1100");
  // std::bitset<4> name (my_int);
  SmallVector<int, 4> XORs;
  LLVM_DEBUG(llvm::dbgs() << "\n\nGroup:\n");

  for (unsigned int i = 0; i < group.size(); i++) {
    int mask = 0;
    LLVM_DEBUG(llvm::dbgs() << "mask: " << mask << "\n");
    int first = group[i].second;;
    LLVM_DEBUG(llvm::dbgs() << "first: " << first << "\n");
    for (unsigned int j = i; j < group.size(); j++) {
      int ID = group[j].second;
      LLVM_DEBUG(llvm::dbgs() << "ID: " << ID << "\n");
      int xor_result = first ^ ID;
      LLVM_DEBUG(llvm::dbgs() << "xor_result: " << xor_result << "\n");
      mask |= xor_result;
    }
    LLVM_DEBUG(llvm::dbgs() << "final mask: " << mask << "\n\n");
    XORs.push_back(mask);
  }

  // Combine all XOR results with bitwise OR
  int maskValue = XORs[0];
  LLVM_DEBUG(llvm::dbgs() << "\nXORs:\n");
  LLVM_DEBUG(llvm::dbgs() << XORs[0] << "\n");
  for (unsigned int i = 1; i < XORs.size(); i++) {
    LLVM_DEBUG(llvm::dbgs() << XORs[i] << "\n");
    maskValue |= XORs[i];
  }

  // Invert with bitwise not, but keep only 5 LSB
  return (~maskValue) & 31;
}

int computeMaskValue(SmallVector<std::pair<PhysPort, int>, 4> group) {
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
       return maskValue;

}

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
    LLVM_DEBUG(llvm::dbgs() << "\nTile " << xCur << " " << yCur << " ");

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
  for (auto i : tile.getResult().getUsers()) {
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

  void createWireOps(OpBuilder &builder, SwitchboxOp sw, ModuleOp &m) {

    int col = sw.colIndex();
    int row = sw.rowIndex();

    //TileOp tile = cast<TileOp>(tiles[std::make_pair(col, row)]);
    TileOp tile = cast<TileOp>(sw.getTileOp());

    // add wires between Core and Switchbox
    builder.create<WireOp>(builder.getUnknownLoc(), 
          tile, WireBundle::Core,
          sw,   WireBundle::Core);

    // if the tile to the west exists, add wires
    if(tiles.count(std::make_pair(col-1, row))) {
      TileOp west_tile = cast<TileOp>(tiles[std::make_pair(col-1, row)]);
      SwitchboxOp west_sw = getOrCreateSwitchbox(builder, west_tile);
      builder.create<WireOp>(builder.getUnknownLoc(), 
                    west_sw, WireBundle::East,
                    sw,      WireBundle::West);
    }

    if (tile.isShimNOCTile()) {
      for (auto shimmux : m.getOps<ShimMuxOp>()) {
        if (shimmux.getTile() == tile) {
          // add wire from tile DMA to ShimMuxOp
          builder.create<WireOp>(builder.getUnknownLoc(),
                tile,     WireBundle::DMA,
                shimmux,  WireBundle::DMA);

          // add wire from ShimMuxOp to SwitchboxOp
          builder.create<WireOp>(builder.getUnknownLoc(), 
                        shimmux, WireBundle::North,
                        sw,       WireBundle::South);
        }
      }
    } else { // it is normal tile (not in shim)

      // add wires between DMA and Switchbox
      builder.create<WireOp>(builder.getUnknownLoc(),
            tile, WireBundle::DMA,
            sw,   WireBundle::DMA);

      // if the tile to the south exists, add wires
      if(tiles.count(std::make_pair(col, row-1))) {
        TileOp south_tile = cast<TileOp>(tiles[std::make_pair(col, row-1)]);
        SwitchboxOp south_sw = getOrCreateSwitchbox(builder, south_tile);
        builder.create<WireOp>(builder.getUnknownLoc(), 
                      south_sw, WireBundle::North,
                      sw,       WireBundle::South);
      }
      // wires on north and east will be added by other tiles, if they exist
    }
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

    int maxcol = 0, maxrow = 0;
    for (auto tileOp : m.getOps<TileOp>()) {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      maxcol = std::max(maxcol, col);
      maxrow = std::max(maxrow, row);
      tiles[std::make_pair(col, row)] = tileOp;
    }


    // The logical model of all the switchboxes.
    // Each "Connect" is a Port-to-Port switchbox setting.
    // The SmallVector of Connects defines a Packet Route
    DenseMap<std::pair<int, int>, SmallVector<std::pair<Connect, int>, 8>>
        switchboxes;

    Pathfinder pathfinder = Pathfinder(maxcol, maxrow);
    // Add all PacketRoutes to Pathfinder object
    // each source can map to multiple different destinations (fanout)
    for (auto pktflow : m.getOps<PacketFlowOp>()) {
      Region &r = pktflow.getPorts();
      Block &b = r.front();
      int flowID = pktflow.IDInt();
      int xSrc, ySrc;
      Port sourcePort;

      for (Operation &Op : b.getOperations()) {
        if (PacketSourceOp pktSource = dyn_cast<PacketSourceOp>(Op)) {
          TileOp srcTile =
              dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
          xSrc = srcTile.colIndex();
          ySrc = srcTile.rowIndex();
          sourcePort = pktSource.port();
        } else if (PacketDestOp pktDest = dyn_cast<PacketDestOp>(Op)) {
          TileOp destTile = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
          int xDest = destTile.colIndex();
          int yDest = destTile.rowIndex();
          Port destPort = pktDest.port();

          pathfinder.addFlow( std::make_pair(xSrc, ySrc), sourcePort, 
                              std::make_pair(xDest, yDest), destPort, flowID);

          //buildPSRoute(xSrc, ySrc, sourcePort, xDest, yDest, destPort, flowID, switchboxes);
        }
      }
    }

    // all flows are populated, call the congestion-aware Pathfinder algorithm
    DenseMap<Flow*, SwitchSettings*> flow_solutions;
    pathfinder.findPaths(flow_solutions, 1000);

    // Print out flow_solutions for debugging
    LLVM_DEBUG(llvm::dbgs() << "###Pathfinder packet routing results###" <<"\n");
    LLVM_DEBUG(llvm::dbgs() << "flow_solutions.size(): " << flow_solutions.size() << "\n");
    //for (auto iter = flow_solutions.begin(); iter != flow_solutions.end(); iter++) {
    //  // Print info about the flow
    //  Flow* f = iter->first;
    //  Pathfinder::printFlow(f);

    //  // Print info about the SwitchSettings which implement this flow
    //  SwitchSettings* settings = iter->second;
    //  Pathfinder::printSwitchSettings(settings);
    //}

    //Pathfinder::convertFlowSolutionsToDenseMap(flow_solutions, switchboxes);

    // check whether the pathfinder algorithm creates a legal routing
    if (!pathfinder.isLegal())
      m.emitError("Unable to find a legal routing");

    //LLVM_DEBUG(llvm::dbgs() << "Check switchboxes\n");

    //DenseMap<std::pair<int, int>, SmallVector<std::pair<Connect, int>, 8>> switchboxes;
    //for (auto swbox : switchboxes) {
    for (auto sol : flow_solutions) {
      Flow* flow = sol.first;
      PathEndPoint flow_start = std::get<0>(*flow);
      Switchbox* flow_start_sb = flow_start.first;
      SmallVector<PathEndPoint> flow_end = std::get<1>(*flow);
      //int flowID = std::get<2>(*flow);

      int col = flow_start_sb->col; //swbox.first.first;
      int row = flow_start_sb->row; //swbox.first.second;

      LLVM_DEBUG(llvm::dbgs() << "Flow starting at: (" 
                              << col << ", " << row << ") flowID: "
                              << std::get<2>(*flow) << "\nConnections:\n");

      SwitchSettings* settings = sol.second;
      //SmallVector<std::pair<Connect, int>, 8> connects(swbox.second);
      for (auto item : *settings) {
        Switchbox* sb = item.first;
        SwitchConnection connect = item.second;
        Port sourcePort = std::get<0>(connect); //connect.first.first;
        SmallVector<Port> destPorts = std::get<1>(connect); //connect.first.second;
        //Port destPort = *destPorts.begin(); //*std::next(destPorts.begin(), 0);
        int flowID = std::get<2>(connect); //connect.second;

        //int nextCol = sb->col, nextRow = sb->row;
        //update_coordinates(nextCol, nextRow, sourcePort.first);

        //update tile of interest
        Operation* op = getOrCreateTile(builder, sb->col, sb->row);

        auto sourceFlow = std::make_pair(std::make_pair(op, sourcePort), flowID);
        LLVM_DEBUG(llvm::dbgs() << "\tFlow source: (" << sb->col << ", " << sb->row
                << stringifyWireBundle(sourcePort.first) << ":" << sourcePort.second 
                << ") ID: " << flowID << "\n");

        for (Port destPort : destPorts) {
          packetFlows[sourceFlow].push_back(std::make_pair(op, destPort));
          LLVM_DEBUG(llvm::dbgs() << "\ttile (" << sb->col << ", " << sb->row << "): "
                                << stringifyWireBundle(sourcePort.first) << " "
                                << sourcePort.second << " -> "
                                << stringifyWireBundle(destPort.first) << " "
                                << destPort.second << "\n");
        }
        slavePorts.push_back(sourceFlow);
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
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
            amselValue = i * numArbiters + a;
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

    // Print slavePorts for debugging
    for (auto item : slavePorts) {
      PhysPort physPort = item.first;
      int flowID = item.second;
      Operation* op = physPort.first;
      Port port = physPort.second;
      TileOp tileOp = dyn_cast<TileOp>(op);

      LLVM_DEBUG(llvm::dbgs() << "slavePort: (" 
              << tileOp.colIndex() << ", " << tileOp.rowIndex() << ") "
              << "\tport: " << stringifyWireBundle(port.first) << ":" 
              << port.second << "\tflowID: " << flowID << "\n");
    }

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

        // Removing this clause makes all the rules seperate
        // i.e. mask = 31
        //if (false)
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
      for (auto port : group) {
        int maskValue = computeMaskValue(group);
        int maskValueBetter = computeMaskValueBetter (group);
        if (maskValue == maskValueBetter)
          LLVM_DEBUG(llvm::dbgs() << "@@@mask values MATCH!\n");
        else
          LLVM_DEBUG(llvm::dbgs() << "@@@mask values DON'T match!\n");
          LLVM_DEBUG(llvm::dbgs() << "Orig: " << maskValue << "\tNew: " << maskValueBetter << "\n");
        slaveMasks[port] = maskValue;
      }
    }

    // Sometimes packet rules can interfere with each other.
    // e.g.
    // AIE.packetrules(DMA : 0) {
    //   AIE.rule(24, 0, %40)
    //   AIE.rule(24, 0, %41) // No packets will ever reach this rule!
    // }
    // Let's try to make that not happen!
    LLVM_DEBUG(llvm::dbgs() << "CHECK Slave Masks for precluding:\n");

    // Organize the slave groups by PhysPorts
    DenseMap< PhysPort, 
      SmallVector<SmallVector<std::pair<PhysPort, int>, 4>, 4> > 
      slaveGroupsByCoord;

    for (auto group : slaveGroups) {
      slaveGroupsByCoord[group.front().first].push_back(group); 
    }

    // PhysPort, ID, masterset_num
    DenseMap<std::pair<PhysPort, int>, std::pair<Port, int> > priorityMap;

    for (auto item : slaveGroupsByCoord) {
      PhysPort physPort = item.first;
      TileOp tile = dyn_cast<TileOp>(physPort.first);
      Coord coord = std::make_pair(tile.colIndex(), tile.rowIndex());

      LLVM_DEBUG(llvm::dbgs() << "\n&&&\n");
      SmallVector<Port, 4> tileMasters;
      for (auto map : mastersets) {
        if (physPort.first != map.first.first)
          continue;
        tileMasters.push_back(map.first.second);
      }

      LLVM_DEBUG(llvm::dbgs() << "tileMasters:\n");
      for (Port port : tileMasters)
        LLVM_DEBUG(llvm::dbgs() << "(" << coord.first << ", " << coord.second << ") "
                << "Port: " << stringifyWireBundle(port.first) << " "
                << port.second << '\n');

      auto groupsAtCoord = item.second;
      SmallVector<int, 8> matchedIDs;
      SmallVector<int, 8> precludedIDs;
      for(auto group : groupsAtCoord) {
        std::pair<PhysPort, int> pair = group.front();
        PhysPort physPort = pair.first;
        WireBundle bundle = physPort.second.first;
        int mask = slaveMasks[pair];
        int channel = physPort.second.second;
        int match = pair.second & mask;
      // auto port = map.first.first;
      // TileOp tile = dyn_cast<TileOp>(port.first);
      // WireBundle bundle = port.second.first;
      // int channel = port.second.second;
      // int match = map.first.second;
      // int mask = map.second;

        LLVM_DEBUG(llvm::dbgs() << "Slave:\n(" << coord.first << ", " << coord.second << ")"
                  << " Port " << stringifyWireBundle(bundle) << ":"
                  << channel << '\n');
        LLVM_DEBUG(llvm::dbgs() << "Mask: " << mask << ", ");
                                //<< "0x" << llvm::Twine::utohexstr(mask) << '\t');
        LLVM_DEBUG(llvm::dbgs() << "match: " << match << ", ");
                                //<< "0x" << llvm::Twine::utohexstr(match) << '\t');
        LLVM_DEBUG(llvm::dbgs() << "Flow ID matches: ");
        for (int i = 0; i < 31; i++) {
          if ((i & mask) == (match & mask)) {
            if(std::count(matchedIDs.begin(), matchedIDs.end(), i)) {
              precludedIDs.push_back(i);
            }
            LLVM_DEBUG(llvm::dbgs() << i << ", ");
            matchedIDs.push_back(i);
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "\nslaveAMSels[<"
          << "(" << coord.first << ", " << coord.second << "), "
          << " Port " << stringifyWireBundle(bundle) << ":" << channel
          << ">]: " << slaveAMSels[pair] << "\t");
        
        // search mastersets for a PhysPort that matches AMSel
        for (auto item : mastersets) {
          PhysPort p = item.first;
          TileOp tileOp = dyn_cast<TileOp>(p.first);
          Port masterPort = p.second;
          WireBundle masterBundle = masterPort.first;
          int masterChannel = masterPort.second;

          SmallVector<int, 4> masterAMsels = item.second;

          // if same coord
          if (std::make_pair(tileOp.colIndex(), tileOp.rowIndex()) == coord){
            for (int AMsel : masterAMsels) {
              if (AMsel == slaveAMSels[pair])
                LLVM_DEBUG(llvm::dbgs() << "masterAMsel Port: "
                      << stringifyWireBundle(masterBundle) << ":" 
                      << masterChannel << "\n");
            }
          }
        }

        LLVM_DEBUG(llvm::dbgs() << "\nPrecluded IDs:\n");
        for( int i : precludedIDs){
            LLVM_DEBUG(llvm::dbgs() << i << ", ");
            Flow* flow_ptr = Pathfinder::getPacketFlow(flow_solutions, i);
            SwitchSettings* settings_ptr = flow_solutions[flow_ptr];
            Switchbox* sb_ptr = Pathfinder::getSwitchbox(settings_ptr, coord);
            SwitchConnection connection = (*settings_ptr)[sb_ptr];
            Pathfinder::printSwitchConnection(connection);
            
            // We now have a SwitchConnection which tells us where all these 
            // packets are intended to go!
            // Since some of these packets are precluded by a previous rule, 
            // we want to make a special packet rule to accomodate them.
            // So we make a list of precluded rules which need priority:
            auto key = std::make_pair(physPort, i);
            //Port srcPort = std::get<0>(connection);
            SmallVector<Port> destPorts = std::get<1>(connection);
            int ID = std::get<2>(connection);
            assert (i == ID);

            for (Port dp : destPorts) {
              for (auto item : mastersets) {
                Port masterPort = item.first.second;
                if (dp == masterPort)
                  priorityMap[key] = std::make_pair(masterPort, slaveAMSels[key]);
              }
            }


            // --OR--
            // So we modify slaveGroups:
            //SmallVector<std::pair<PhysPort, int>, 4> newGroup;
            //newGroup.push_back(std::make_pair(physPort, i));
            //slaveGroups.insert(slaveGroups.begin(), newGroup);
        }
        LLVM_DEBUG(llvm::dbgs() << "\n\n");
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "\n^^^priorityMap:\n");
    for (auto item : priorityMap) {
      auto key = item.first;
      PhysPort physPort = key.first;
      TileOp slaveTileOp = dyn_cast<TileOp>(physPort.first);
      WireBundle bundle = physPort.second.first;
      int channel = physPort.second.second;
      int ID = key.second;
      Port masterPort = item.second.first;
      int slaveAMSel = item.second.second;
      LLVM_DEBUG(llvm::dbgs() << "(" << slaveTileOp.colIndex() << ", " 
                              << slaveTileOp.rowIndex() << ")");
      LLVM_DEBUG(llvm::dbgs() << "\tSource Port (slave):" << stringifyWireBundle(bundle)
                              << ":" << channel);
      LLVM_DEBUG(llvm::dbgs() << "\tID: " << ID 
                              << "\tAMsel: " << slaveAMSel << "\t");
      LLVM_DEBUG(llvm::dbgs() << "\tmasterPort: "
                    << stringifyWireBundle(masterPort.first) << ":" 
                    << masterPort.second << "\t");

      //for (auto pair : mastersets) {
      //  PhysPort p = pair.first;
      //  SmallVector<int, 4> masterAMsels = pair.second;

      //  TileOp masterTileOp = dyn_cast<TileOp>(p.first);
      //  Port port = p.second;
      //  WireBundle masterBundle = port.first;
      //  int masterChannel = port.second;

      //  if (slaveTileOp == masterTileOp) {
      //    for (int AMsel : masterAMsels) {
      //      if (AMsel == slaveAMSel)
      //        LLVM_DEBUG(llvm::dbgs() << "Dest Port (master): "
      //              << stringifyWireBundle(masterBundle) << ":" 
      //              << masterChannel << "\t");
      //    }
      //  }
      //}
        LLVM_DEBUG(llvm::dbgs() << "\n\n");
    }

    // Realize the routes in MLIR
    for (auto map : tiles) {
      Operation *tileOp = map.second;
      TileOp tile = dyn_cast<TileOp>(tileOp);

      // Create a switchbox for the routes and insert inside it.
      builder.setInsertionPointAfter(tileOp);
      SwitchboxOp swbox = getOrCreateSwitchbox(builder, tile);
      swbox.ensureTerminator(swbox.getConnections(), builder,
                             builder.getUnknownLoc());
      Block &b = swbox.getConnections().front();
      builder.setInsertionPoint(b.getTerminator());

      std::vector<bool> amselOpNeededVector(32);
      for (auto map : mastersets) {
        if (tileOp != map.first.first)
          continue;

        WireBundle bundle = map.first.second.first;
        int channel = map.first.second.second;
        assert(tileOp);
        TileOp tile = dyn_cast<TileOp>(tileOp);
        LLVM_DEBUG(llvm::dbgs()
                  << "master " << tile << " " << stringifyWireBundle(bundle)
                  << " : " << channel << '\n');
        for (auto value : map.second)
          LLVM_DEBUG(llvm::dbgs() << "amsel: " << value << '\n');


        for (auto value : map.second) {
          amselOpNeededVector[value] = true;
          LLVM_DEBUG(llvm::dbgs() << "amselOpNeededVector[" << value << "]: true\n");
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
        //data type reminder: SmallVector<std::pair<PhysPort, int>, 4> group;
        builder.setInsertionPoint(b.getTerminator());

        std::pair<PhysPort, int> front = group.front();

        PhysPort physPort = front.first;
        if (tileOp != physPort.first)
          continue;


        Port slavePort  = physPort.second;
        WireBundle bundle = slavePort .first;
        int channel = slavePort .second;
        int mask = slaveMasks[front];
        int ID = front.second & mask;

        // Verify that we actually map all the ID's correctly.
        for (auto s : group) {
          assert((s.second & mask) == ID);
        }
        Value amsel = amselOps[slaveAMSels[front]];

        PacketRulesOp packetrules;
        LLVM_DEBUG(llvm::dbgs() << "\nslavePort: "
                  << stringifyWireBundle(slavePort.first)
                  << ":" << slavePort.second << "\n");
        if (slaveRules.count(slavePort) == 0) {
          packetrules = builder.create<PacketRulesOp>(builder.getUnknownLoc(),
                                                      bundle, channel);
          packetrules.ensureTerminator(packetrules.getRules(), builder,
                                       builder.getUnknownLoc());
          slaveRules[slavePort] = packetrules;
        } else { //When will this else execute?
          packetrules = slaveRules[slavePort];
        }

        Block &rules = packetrules.getRules().front();

        //generate priority packet rules
        for (auto pair : group) {
          if(priorityMap.count(pair)) {
            int priorityID = pair.second;
            Port priorityPort = priorityMap[pair].first;
            int priorityAMsel = priorityMap[pair].second;
            LLVM_DEBUG(llvm::dbgs() << "Generate Priority Packet rule: AIE.rule(31, "
              << priorityID << ", " << priorityAMsel << ")\n");
            LLVM_DEBUG(llvm::dbgs() << "Amsel: " << amselOps[priorityAMsel] << "\n");
            LLVM_DEBUG(llvm::dbgs() << "priorityPort: " << stringifyWireBundle(priorityPort.first) << ":"
                          << priorityPort.second << "\n\n");

            builder.setInsertionPointToStart(&rules);
            builder.create<PacketRuleOp>(builder.getUnknownLoc(), 31, priorityID, amselOps[priorityAMsel]);
          }
        }

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

      Region &r = switchbox.getConnections();
      Block &b = r.front();

      // Find if the corresponding shimmux exsists or not
      int shim_exist = 0;
      ShimMuxOp shimOp;
      for (auto shimmux : m.getOps<ShimMuxOp>()) {
        if (shimmux.getTile() == tileOp) {
          shim_exist = 1;
          shimOp = shimmux;
          break;
        }
      }

      for (Operation &Op : b.getOperations()) {
        if (PacketRulesOp pktrules = dyn_cast<PacketRulesOp>(Op)) {

          // check if there is MM2S DMA in the switchbox of the 0th row
          if (pktrules.getSourceBundle() == WireBundle::DMA) {

            // If there is, then it should be put into the corresponding shimmux
            // If shimmux not defined then create shimmux
            if (!shim_exist) {
              builder.setInsertionPointAfter(tileOp);
              shimOp =
                  builder.create<ShimMuxOp>(builder.getUnknownLoc(), tileOp);
              Region &r1 = shimOp.getConnections();
              Block *b1 = builder.createBlock(&r1);
              builder.setInsertionPointToEnd(b1);
              builder.create<EndOp>(builder.getUnknownLoc());
              shim_exist = 1;
            }

            Region &r0 = shimOp.getConnections();
            Block &b0 = r0.front();
            builder.setInsertionPointToStart(&b0);

            pktrules->removeAttr("sourceBundle");
            pktrules->setAttr(
                "sourceBundle",
                builder.getI32IntegerAttr(3)); // WireBundle::South
            if (pktrules.getSourceChannel() == 0) {
              pktrules->removeAttr("sourceChannel");
              pktrules->setAttr("sourceChannel",
                                builder.getI32IntegerAttr(3)); // Channel 3
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::DMA, 0, WireBundle::North,
                                        3);
            }
            if (pktrules.getSourceChannel() == 1) {
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
          if (mtset.getDestBundle() == WireBundle::DMA) {

            // If there is, then it should be put into the corresponding shimmux
            // If shimmux not defined then create shimmux
            if (!shim_exist) {
              builder.setInsertionPointAfter(tileOp);
              shimOp =
                  builder.create<ShimMuxOp>(builder.getUnknownLoc(), tileOp);
              Region &r1 = shimOp.getConnections();
              Block *b1 = builder.createBlock(&r1);
              builder.setInsertionPointToEnd(b1);
              builder.create<EndOp>(builder.getUnknownLoc());
              shim_exist = 1;
            }

            Region &r0 = shimOp.getConnections();
            Block &b0 = r0.front();
            builder.setInsertionPointToStart(&b0);

            mtset->removeAttr("destBundle");
            mtset->setAttr("destBundle",
                           builder.getI32IntegerAttr(3)); // WireBundle::South
            if (mtset.getDestChannel() == 0) {
              mtset->removeAttr("destChannel");
              mtset->setAttr("destChannel",
                             builder.getI32IntegerAttr(2)); // Channel 2
              builder.create<ConnectOp>(builder.getUnknownLoc(),
                                        WireBundle::North, 2, WireBundle::DMA,
                                        0);
            }
            if (mtset.getDestChannel() == 1) {
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


    // add WireOps between all tiles used.
    for (auto map : tiles) {
      Operation *tileOp = map.second;
      TileOp tile = dyn_cast<TileOp>(tileOp);
      SwitchboxOp swbox = getOrCreateSwitchbox(builder, tile);
      builder.setInsertionPointAfter(swbox);
      createWireOps(builder, swbox, m);
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