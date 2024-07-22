//===- AIECreatePathfindFlows.cpp -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#define DEBUG_TYPE "aie-create-pathfinder-flows"

namespace {
std::vector<Operation *> flowOps;

// allocates channels between switchboxes ( but does not assign them)
// instantiates shim-muxes AND allocates channels ( no need to rip these up in )
struct ConvertFlowsToInterconnect : OpConversionPattern<FlowOp> {
  using OpConversionPattern::OpConversionPattern;
  DeviceOp &device;
  DynamicTileAnalysis &analyzer;
  bool keepFlowOp;
  ConvertFlowsToInterconnect(MLIRContext *context, DeviceOp &d,
                             DynamicTileAnalysis &a, bool keepFlowOp,
                             PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), device(d), analyzer(a),
        keepFlowOp(keepFlowOp) {}

  LogicalResult match(FlowOp op) const override { return success(); }

  void addConnection(ConversionPatternRewriter &rewriter,
                     // could be a shim-mux or a switchbox.
                     Interconnect op, FlowOp flowOp, WireBundle inBundle,
                     int inIndex, WireBundle outBundle, int outIndex) const {

    Region &r = op.getConnections();
    Block &b = r.front();
    auto point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(b.getTerminator());

    rewriter.create<ConnectOp>(rewriter.getUnknownLoc(), inBundle, inIndex,
                               outBundle, outIndex);

    rewriter.restoreInsertionPoint(point);

    LLVM_DEBUG(llvm::dbgs()
               << "\t\taddConnection() (" << op.colIndex() << ","
               << op.rowIndex() << ") " << stringifyWireBundle(inBundle)
               << inIndex << " -> " << stringifyWireBundle(outBundle)
               << outIndex << "\n");
  }

  void rewrite(FlowOp flowOp, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    Operation *Op = flowOp.getOperation();

    auto srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    auto srcBundle = flowOp.getSourceBundle();
    auto srcChannel = flowOp.getSourceChannel();
    Port srcPort = {srcBundle, srcChannel};

    if (keepFlowOp) {
      auto *clonedOp = Op->clone();
      flowOps.push_back(clonedOp);
    }

#ifndef NDEBUG
    auto dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    auto dstBundle = flowOp.getDestBundle();
    auto dstChannel = flowOp.getDestChannel();
    LLVM_DEBUG(llvm::dbgs()
               << "\n\t---Begin rewrite() for flowOp: (" << srcCoords.col
               << ", " << srcCoords.row << ")" << stringifyWireBundle(srcBundle)
               << srcChannel << " -> (" << dstCoords.col << ", "
               << dstCoords.row << ")" << stringifyWireBundle(dstBundle)
               << dstChannel << "\n\t");
#endif

    // if the flow (aka "net") for this FlowOp hasn't been processed yet,
    // add all switchbox connections to implement the flow
    Switchbox srcSB = {srcCoords.col, srcCoords.row};
    if (PathEndPoint srcPoint = {srcSB, srcPort};
        !analyzer.processedFlows[srcPoint]) {
      SwitchSettings settings = analyzer.flowSolutions[srcPoint];
      // add connections for all the Switchboxes in SwitchSettings
      for (const auto &[curr, setting] : settings) {
        SwitchboxOp swOp = analyzer.getSwitchbox(rewriter, curr.col, curr.row);
        int shimCh = srcChannel;
        // TODO: must reserve N3, N7, S2, S3 for DMA connections
        if (curr == srcSB &&
            analyzer.getTile(rewriter, srcSB.col, srcSB.row).isShimNOCTile()) {
          // shim DMAs at start of flows
          if (srcBundle == WireBundle::DMA) {
            shimCh = srcChannel == 0
                         ? 3
                         : 7; // must be either DMA0 -> N3 or DMA1 -> N7
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
            addConnection(rewriter,
                          cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          } else if (srcBundle ==
                     WireBundle::NOC) { // must be NOC0/NOC1 -> N2/N3 or
                                        // NOC2/NOC3 -> N6/N7
            shimCh = srcChannel >= 2 ? srcChannel + 4 : srcChannel + 2;
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
            addConnection(rewriter,
                          cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          } else if (srcBundle ==
                     WireBundle::PLIO) { // PLIO at start of flows with mux
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
            addConnection(rewriter,
                          cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          }
        } else if (curr == srcSB &&
                   analyzer.getTile(rewriter, srcSB.col, srcSB.row)
                       .isShimNOCorPLTile()) {
          if (srcBundle ==
              WireBundle::PLIO) { // PLIO at start of flows with mux
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
            addConnection(rewriter,
                          cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          }
        }
        for (const auto &[bundle, channel] : setting.dsts) {
          // handle special shim connectivity
          if (curr == srcSB && analyzer.getTile(rewriter, srcSB.col, srcSB.row)
                                   .isShimNOCorPLTile()) {
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                          flowOp, WireBundle::South, shimCh, bundle, channel);
          } else if (analyzer.getTile(rewriter, curr.col, curr.row)
                         .isShimNOCorPLTile() &&
                     (bundle == WireBundle::DMA || bundle == WireBundle::PLIO ||
                      bundle == WireBundle::NOC)) {
            shimCh = channel;
            if (analyzer.getTile(rewriter, curr.col, curr.row)
                    .isShimNOCTile()) {
              // shim DMAs at end of flows
              if (bundle == WireBundle::DMA) {
                shimCh = channel == 0
                             ? 2
                             : 3; // must be either N2 -> DMA0 or N3 -> DMA1
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr.col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              } else if (bundle == WireBundle::NOC) {
                shimCh = channel + 2; // must be either N2/3/4/5 -> NOC0/1/2/3
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr.col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              } else if (channel >=
                         2) { // must be PLIO...only PLIO >= 2 require mux
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr.col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              }
            }
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                          flowOp, setting.src.bundle, setting.src.channel,
                          WireBundle::South, shimCh);
          } else {
            // otherwise, regular switchbox connection
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                          flowOp, setting.src.bundle, setting.src.channel,
                          bundle, channel);
          }
        }

        LLVM_DEBUG(llvm::dbgs() << curr << ": " << setting << " | "
                                << "\n");
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "\n\t\tFinished adding ConnectOps to implement flowOp.\n");
      analyzer.processedFlows[srcPoint] = true;
    } else
      LLVM_DEBUG(llvm::dbgs() << "Flow already processed!\n");

    rewriter.eraseOp(Op);
  }
};

} // namespace

namespace xilinx::AIE {

void AIEPathfinderPass::runOnFlow(DeviceOp d, OpBuilder &builder) {
  // Apply rewrite rule to switchboxes to add assignments to every 'connect'
  // operation inside
  ConversionTarget target(getContext());
  target.addLegalOp<TileOp>();
  target.addLegalOp<ConnectOp>();
  target.addLegalOp<SwitchboxOp>();
  target.addLegalOp<ShimMuxOp>();
  target.addLegalOp<EndOp>();

  RewritePatternSet patterns(&getContext());
  patterns.insert<ConvertFlowsToInterconnect>(d.getContext(), d, analyzer,
                                              clKeepFlowOp);
  if (failed(applyPartialConversion(d, target, std::move(patterns))))
    return signalPassFailure();

  // Keep for visualization
  if (clKeepFlowOp)
    for (auto op : flowOps)
      builder.insert(op);

  // Populate wires between switchboxes and tiles.
  for (int col = 0; col <= analyzer.getMaxCol(); col++) {
    for (int row = 0; row <= analyzer.getMaxRow(); row++) {
      TileOp tile;
      if (analyzer.coordToTile.count({col, row}))
        tile = analyzer.coordToTile[{col, row}];
      else
        continue;
      SwitchboxOp sw;
      if (analyzer.coordToSwitchbox.count({col, row}))
        sw = analyzer.coordToSwitchbox[{col, row}];
      else
        continue;
      if (col > 0) {
        // connections east-west between stream switches
        if (analyzer.coordToSwitchbox.count({col - 1, row})) {
          auto westsw = analyzer.coordToSwitchbox[{col - 1, row}];
          builder.create<WireOp>(builder.getUnknownLoc(), westsw,
                                 WireBundle::East, sw, WireBundle::West);
        }
      }
      if (row > 0) {
        // connections between abstract 'core' of tile
        builder.create<WireOp>(builder.getUnknownLoc(), tile, WireBundle::Core,
                               sw, WireBundle::Core);
        // connections between abstract 'dma' of tile
        builder.create<WireOp>(builder.getUnknownLoc(), tile, WireBundle::DMA,
                               sw, WireBundle::DMA);
        // connections north-south inside array ( including connection to shim
        // row)
        if (analyzer.coordToSwitchbox.count({col, row - 1})) {
          auto southsw = analyzer.coordToSwitchbox[{col, row - 1}];
          builder.create<WireOp>(builder.getUnknownLoc(), southsw,
                                 WireBundle::North, sw, WireBundle::South);
        }
      } else if (row == 0) {
        if (tile.isShimNOCTile()) {
          if (analyzer.coordToShimMux.count({col, 0})) {
            auto shimsw = analyzer.coordToShimMux[{col, 0}];
            builder.create<WireOp>(
                builder.getUnknownLoc(), shimsw,
                WireBundle::North, // Changed to connect into the north
                sw, WireBundle::South);
            // PLIO is attached to shim mux
            if (analyzer.coordToPLIO.count(col)) {
              auto plio = analyzer.coordToPLIO[col];
              builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                     WireBundle::North, shimsw,
                                     WireBundle::South);
            }

            // abstract 'DMA' connection on tile is attached to shim mux ( in
            // row 0 )
            builder.create<WireOp>(builder.getUnknownLoc(), tile,
                                   WireBundle::DMA, shimsw, WireBundle::DMA);
          }
        } else if (tile.isShimPLTile()) {
          // PLIO is attached directly to switch
          if (analyzer.coordToPLIO.count(col)) {
            auto plio = analyzer.coordToPLIO[col];
            builder.create<WireOp>(builder.getUnknownLoc(), plio,
                                   WireBundle::North, sw, WireBundle::South);
          }
        }
      }
    }
  }
}

Operation *AIEPathfinderPass::getOrCreateTile(OpBuilder &builder, int col,
                                              int row) {
  TileID index = {col, row};
  Operation *tileOp = tiles[index];
  if (!tileOp) {
    auto tile = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
    tileOp = tile.getOperation();
    tiles[index] = tileOp;
  }
  return tileOp;
}

SwitchboxOp AIEPathfinderPass::getOrCreateSwitchbox(OpBuilder &builder,
                                                    TileOp tile) {
  for (auto i : tile.getResult().getUsers()) {
    if (llvm::isa<SwitchboxOp>(*i)) {
      return llvm::cast<SwitchboxOp>(*i);
    }
  }
  return builder.create<SwitchboxOp>(builder.getUnknownLoc(), tile);
}

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

bool AIEPathfinderPass::findPathToDest(SwitchSettings settings, TileID currTile,
                                       WireBundle currDestBundle,
                                       int currDestChannel, TileID finalTile,
                                       WireBundle finalDestBundle,
                                       int finalDestChannel) {

  if ((currTile == finalTile) && (currDestBundle == finalDestBundle) &&
      (currDestChannel == finalDestChannel)) {
    return true;
  }

  WireBundle neighbourSourceBundle;
  TileID neighbourTile;
  if (currDestBundle == WireBundle::East) {
    neighbourSourceBundle = WireBundle::West;
    neighbourTile = {currTile.col + 1, currTile.row};
  } else if (currDestBundle == WireBundle::West) {
    neighbourSourceBundle = WireBundle::East;
    neighbourTile = {currTile.col - 1, currTile.row};
  } else if (currDestBundle == WireBundle::North) {
    neighbourSourceBundle = WireBundle::South;
    neighbourTile = {currTile.col, currTile.row + 1};
  } else if (currDestBundle == WireBundle::South) {
    neighbourSourceBundle = WireBundle::North;
    neighbourTile = {currTile.col, currTile.row - 1};
  } else {
    return false;
  }

  int neighbourSourceChannel = currDestChannel;
  for (const auto &[tile, setting] : settings) {
    if ((tile == neighbourTile) &&
        (setting.src.bundle == neighbourSourceBundle) &&
        (setting.src.channel == neighbourSourceChannel)) {
      for (const auto &[bundle, channel] : setting.dsts) {
        if (findPathToDest(settings, neighbourTile, bundle, channel, finalTile,
                           finalDestBundle, finalDestChannel)) {
          return true;
        }
      }
    }
  }

  return false;
}

void AIEPathfinderPass::runOnPacketFlow(DeviceOp device, OpBuilder &builder) {

  ConversionTarget target(getContext());

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
  for (PacketFlowOp pktFlowOp : device.getOps<PacketFlowOp>()) {
    Region &r = pktFlowOp.getPorts();
    Block &b = r.front();
    int flowID = pktFlowOp.IDInt();
    Port srcPort, destPort;
    TileOp srcTile, destTile;
    TileID srcCoords, destCoords;

    for (Operation &Op : b.getOperations()) {
      if (auto pktSource = dyn_cast<PacketSourceOp>(Op)) {
        srcTile = dyn_cast<TileOp>(pktSource.getTile().getDefiningOp());
        srcPort = pktSource.port();
        srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      } else if (auto pktDest = dyn_cast<PacketDestOp>(Op)) {
        destTile = dyn_cast<TileOp>(pktDest.getTile().getDefiningOp());
        destPort = pktDest.port();
        destCoords = {destTile.colIndex(), destTile.rowIndex()};
        // Assign "keep_pkt_header flag"
        if (pktFlowOp->hasAttr("keep_pkt_header"))
          keepPktHeaderAttr[{destTile, destPort}] =
              StringAttr::get(Op.getContext(), "true");
        Switchbox srcSB = {srcCoords.col, srcCoords.row};
        if (PathEndPoint srcPoint = {srcSB, srcPort};
            !analyzer.processedFlows[srcPoint]) {
          SwitchSettings settings = analyzer.flowSolutions[srcPoint];
          // add connections for all the Switchboxes in SwitchSettings
          for (const auto &[curr, setting] : settings) {
            for (const auto &[bundle, channel] : setting.dsts) {
              // reject false broadcast
              if (!findPathToDest(settings, curr, bundle, channel, destCoords,
                                  destPort.bundle, destPort.channel))
                continue;
              Connect connect = {{setting.src.bundle, setting.src.channel},
                                 {bundle, channel}};
              if (std::find(switchboxes[curr].begin(), switchboxes[curr].end(),
                            std::pair{connect, flowID}) ==
                  switchboxes[curr].end())
                switchboxes[curr].push_back({connect, flowID});
            }
          }
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Check switchboxes\n");

  for (const auto &[tileId, connects] : switchboxes) {
    int col = tileId.col;
    int row = tileId.row;
    Operation *tileOp = getOrCreateTile(builder, col, row);
    LLVM_DEBUG(llvm::dbgs() << "***switchbox*** " << col << " " << row << '\n');
    for (const auto &[conn, flowID] : connects) {
      Port sourcePort = conn.src;
      Port destPort = conn.dst;
      auto sourceFlow =
          std::make_pair(std::make_pair(tileOp, sourcePort), flowID);
      packetFlows[sourceFlow].push_back({tileOp, destPort});
      slavePorts.push_back(sourceFlow);
      LLVM_DEBUG(llvm::dbgs() << "flowID " << flowID << ':'
                              << stringifyWireBundle(sourcePort.bundle) << " "
                              << sourcePort.channel << " -> "
                              << stringifyWireBundle(destPort.bundle) << " "
                              << destPort.channel << "\n");
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

  std::vector<std::pair<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>>>
      sortedPacketFlows(packetFlows.begin(), packetFlows.end());

  // To get determinsitic behaviour
  std::sort(sortedPacketFlows.begin(), sortedPacketFlows.end(),
            [](const auto &lhs, const auto &rhs) {
              auto lhsFlowID = lhs.first.second;
              auto rhsFlowID = rhs.first.second;
              return lhsFlowID < rhsFlowID;
            });

  // Check all multi-cast flows (same source, same ID). They should be
  // assigned the same arbiter and msel so that the flow can reach all the
  // destination ports at the same time For destination ports that appear in
  // different (multicast) flows, it should have a different <arbiterID, msel>
  // value pair for each flow
  for (const auto &packetFlow : sortedPacketFlows) {
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
            shimOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(), tileOp);
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
            builder.create<ConnectOp>(builder.getUnknownLoc(), WireBundle::DMA,
                                      0, WireBundle::North, 3);
          }
          if (pktrules.getSourceChannel() == 1) {
            pktrules.setSourceChannel(7);
            builder.create<ConnectOp>(builder.getUnknownLoc(), WireBundle::DMA,
                                      1, WireBundle::North, 7);
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
            shimOp = builder.create<ShimMuxOp>(builder.getUnknownLoc(), tileOp);
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
                                      WireBundle::North, 2, WireBundle::DMA, 0);
          }
          if (mtset.getDestChannel() == 1) {
            mtset.setDestChannel(3);
            builder.create<ConnectOp>(builder.getUnknownLoc(),
                                      WireBundle::North, 3, WireBundle::DMA, 1);
          }
        }
      }
    }
  }

  RewritePatternSet patterns(&getContext());

  if (!clKeepFlowOp)
    patterns.add<AIEOpRemoval<PacketFlowOp>>(device.getContext());

  if (failed(applyPartialConversion(device, target, std::move(patterns))))
    signalPassFailure();
}

void AIEPathfinderPass::runOnOperation() {

  // create analysis pass with routing graph for entire device
  LLVM_DEBUG(llvm::dbgs() << "---Begin AIEPathfinderPass---\n");

  DeviceOp d = getOperation();
  if (failed(analyzer.runAnalysis(d)))
    return signalPassFailure();
  OpBuilder builder = OpBuilder::atBlockEnd(d.getBody());

  if (clRouteCircuit)
    runOnFlow(d, builder);
  if (clRoutePacket)
    runOnPacketFlow(d, builder);

  // If the routing violates architecture-specific routing constraints, then
  // attempt to partially reroute.
  const auto &targetModel = d.getTargetModel();
  std::vector<SwConnection> problemConnects;
  d.walk([&](ConnectOp connect) {
    if (auto sw = connect->getParentOfType<SwitchboxOp>()) {
      // Constraint: memtile stream switch constraints
      if (auto tile = sw.getTileOp();
          tile.isMemTile() &&
          !targetModel.isLegalTileConnection(
              tile.colIndex(), tile.rowIndex(), connect.getSourceBundle(),
              connect.getSourceChannel(), connect.getDestBundle(),
              connect.getDestChannel())) {
        problemConnects.push_back(
            {sw, connect.sourcePort(), connect.destPort()});
      }
    }
  });

  d.walk([&](AMSelOp amsel) {
    if (auto sw = amsel->getParentOfType<SwitchboxOp>()) {
      std::vector<MasterSetOp> mstrs;
      std::vector<PacketRulesOp> slvs;
      for (auto *user : amsel.getResult().getUsers()) {
        if (auto s = dyn_cast<PacketRuleOp>(user)) {
          auto pktRules = dyn_cast<PacketRulesOp>(s->getParentOp());
          slvs.push_back(pktRules);
        } else if (auto m = dyn_cast<MasterSetOp>(user))
          mstrs.push_back(m);
      }
      for (auto m : mstrs) {
        for (auto s : slvs) {
          if (auto tile = sw.getTileOp();
              tile.isMemTile() &&
              !targetModel.isLegalTileConnection(
                  tile.colIndex(), tile.rowIndex(), s.sourcePort().bundle,
                  s.sourcePort().channel, m.destPort().bundle,
                  m.destPort().channel))
            problemConnects.push_back({sw, s.sourcePort(), m.destPort()});
        }
      }
    }
  });

  for (SwConnection swConnect : problemConnects) {
    if (!attemptFixupMemTileRouting(d, swConnect))
      return signalPassFailure();
  }
}

bool AIEPathfinderPass::attemptFixupMemTileRouting(
    DeviceOp &d, SwConnection &problemConnect) {
  int northChannel;
  int southChannel;
  if (problemConnect.sourcePort.bundle == WireBundle::North &&
      problemConnect.destPort.bundle == WireBundle::South) {
    northChannel = problemConnect.sourcePort.channel;
    southChannel = problemConnect.destPort.channel;
  } else if (problemConnect.sourcePort.bundle == WireBundle::South &&
             problemConnect.destPort.bundle == WireBundle::North) {
    northChannel = problemConnect.destPort.channel;
    southChannel = problemConnect.sourcePort.channel;
  } else
    return false; // Problem is not about n-s routing

  SwitchboxOp &swBox = problemConnect.sw;

  // Attempt to reroute northern channel and neighbouring sw
  if (auto neighbourSw =
          getSwitchbox(d, swBox.colIndex(), swBox.rowIndex() + 1)) {
    WireBundle problemBundle = WireBundle::North;
    WireBundle neighbourBundle = WireBundle::South;
    int problemChannel = northChannel;
    int candidateChannel = southChannel;
    if (checkChannelEmpty(neighbourSw, neighbourBundle, candidateChannel)) {
      replaceRoutingChannel(swBox, problemBundle, problemChannel,
                            candidateChannel);
      replaceRoutingChannel(neighbourSw, neighbourBundle, problemChannel,
                            candidateChannel);
      return true;
    }
  }
  // Attempt to reroute southern channel and neighbouring sw
  if (auto neighbourSw =
          getSwitchbox(d, swBox.colIndex(), swBox.rowIndex() - 1)) {
    WireBundle problemBundle = WireBundle::South;
    WireBundle neighbourBundle = WireBundle::North;
    int problemChannel = southChannel;
    int candidateChannel = northChannel;
    if (checkChannelEmpty(neighbourSw, neighbourBundle, candidateChannel)) {
      replaceRoutingChannel(swBox, problemBundle, problemChannel,
                            candidateChannel);
      replaceRoutingChannel(neighbourSw, neighbourBundle, problemChannel,
                            candidateChannel);
      return true;
    }
  }

  return false;
}

bool AIEPathfinderPass::checkChannelEmpty(SwitchboxOp swOp, WireBundle bundle,
                                          int channel) {
  // Check circuit-switched
  for (auto connect : swOp.getOps<ConnectOp>()) {
    if (connect.getSourceBundle() == bundle &&
        connect.getSourceChannel() == channel)
      return false;
    if (connect.getDestBundle() == bundle &&
        connect.getDestChannel() == channel)
      return false;
  }

  // Check packet-switched
  for (auto pktRules : swOp.getOps<PacketRulesOp>()) {
    if (pktRules.sourcePort().bundle == bundle &&
        pktRules.sourcePort().channel == channel)
      return false;
  }
  for (auto masterSet : swOp.getOps<MasterSetOp>()) {
    if (masterSet.destPort().bundle == bundle &&
        masterSet.destPort().channel == channel)
      return false;
  }

  return true;
}

void AIEPathfinderPass::replaceRoutingChannel(SwitchboxOp &swOp,
                                              WireBundle bundle, int oldChannel,
                                              int newChannel) {
  // replace any macthed circuit-switched
  for (auto connect : swOp.getOps<ConnectOp>()) {
    if (connect.getSourceBundle() == bundle &&
        connect.getSourceChannel() == oldChannel)
      connect.setSourceChannel(newChannel);
    if (connect.getDestBundle() == bundle &&
        connect.getDestChannel() == oldChannel)
      connect.setDestChannel(newChannel);
  }

  // replace any macthed packet-switched
  std::vector<PacketRulesOp> newSourcePacketRules;
  std::vector<MasterSetOp> newDestAMSels;
  for (auto pktRules : swOp.getOps<PacketRulesOp>()) {
    if (pktRules.sourcePort().bundle == bundle &&
        pktRules.sourcePort().channel == oldChannel)
      pktRules.setSourceChannel(newChannel);
  }
  for (auto masterSet : swOp.getOps<MasterSetOp>()) {
    if (masterSet.destPort().bundle == bundle &&
        masterSet.destPort().channel == oldChannel)
      masterSet.setDestChannel(newChannel);
  }
}

SwitchboxOp AIEPathfinderPass::getSwitchbox(DeviceOp &d, int col, int row) {
  SwitchboxOp output = nullptr;
  d.walk([&](SwitchboxOp swBox) {
    if (swBox.colIndex() == col && swBox.rowIndex() == row) {
      output = swBox;
    }
  });
  return output;
}

std::unique_ptr<OperationPass<DeviceOp>> createAIEPathfinderPass() {
  return std::make_unique<AIEPathfinderPass>();
}

} // namespace xilinx::AIE
