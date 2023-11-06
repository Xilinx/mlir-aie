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

#include "mlir/IR/Attributes.h"
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

std::string stringifyDirs(std::set<Port> dirs) {
  unsigned int count = 0;
  std::string out = "{";
  for (Port dir : dirs) {
    switch (dir.bundle) {
    case WireBundle::Core:
      out += "Core";
      break;
    case WireBundle::DMA:
      out += "DMA";
      break;
    case WireBundle::North:
      out += "N";
      break;
    case WireBundle::East:
      out += "E";
      break;
    case WireBundle::South:
      out += "S";
      break;
    case WireBundle::West:
      out += "W";
      break;
    default:
      out += "X";
    }
    out += std::to_string(dir.channel);
    if (++count < dirs.size())
      out += ",";
  }
  out += "}";
  return out;
}

std::string stringifyDir(Port dir) {
  return stringifyDirs(std::set<Port>({dir}));
}
std::string stringifySwitchSettings(SwitchSettings settings) {
  std::string out = "\tSwitchSettings: ";
  for (const auto &[sb, setting] : settings) {
    out += (std::string) "(" + std::to_string(sb->col) + ", " +
           std::to_string(sb->row) + ") " + stringifyDir(setting.src) + " -> " +
           stringifyDirs(setting.dsts) + " | ";
  }
  return out + "\n";
}

// DynamicTileAnalysis integrates the Pathfinder class into the MLIR
// environment. It passes flows to the Pathfinder as ordered pairs of ints.
// Detailed routing is received as SwitchboxSettings
// It then converts these settings to MLIR operations
class DynamicTileAnalysis {
public:
  DeviceOp &device;
  int maxCol, maxRow;
  Pathfinder pathfinder;
  std::map<PathEndPoint, SwitchSettings> flowSolutions;
  std::map<PathEndPoint, bool> processedFlows;

  DenseMap<TileID, TileOp> coordToTile;
  DenseMap<TileID, SwitchboxOp> coordToSwitchbox;
  DenseMap<TileID, ShimMuxOp> coordToShimMux;
  DenseMap<int, PLIOOp> coordToPLIO;

  const int maxIterations = 1000; // how long until declared unroutable

  DynamicTileAnalysis(DeviceOp &d) : device(d) {
    LLVM_DEBUG(llvm::dbgs()
               << "\t---Begin DynamicTileAnalysis Constructor---\n");
    // find the maxCol and maxRow
    maxCol = 0;
    maxRow = 0;
    for (TileOp tileOp : d.getOps<TileOp>()) {
      maxCol = std::max(maxCol, tileOp.colIndex());
      maxRow = std::max(maxRow, tileOp.rowIndex());
    }

    pathfinder = Pathfinder(maxCol, maxRow, d);

    // for each flow in the device, add it to pathfinder
    // each source can map to multiple different destinations (fanout)
    for (FlowOp flowOp : device.getOps<FlowOp>()) {
      TileOp srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
      TileOp dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
      TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
      TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
      Port srcPort = {flowOp.getSourceBundle(), flowOp.getSourceChannel()};
      Port dstPort = {flowOp.getDestBundle(), flowOp.getDestChannel()};
      LLVM_DEBUG(llvm::dbgs()
                 << "\tAdding Flow: (" << srcCoords.col << ", " << srcCoords.row
                 << ")" << stringifyWireBundle(srcPort.bundle)
                 << srcPort.channel << " -> (" << dstCoords.col << ", "
                 << dstCoords.row << ")" << stringifyWireBundle(dstPort.bundle)
                 << (int)dstPort.channel << "\n");
      pathfinder.addFlow(srcCoords, srcPort, dstCoords, dstPort);
    }

    // add existing connections so Pathfinder knows which resources are
    // available search all existing SwitchBoxOps for exising connections
    for (SwitchboxOp switchboxOp : device.getOps<SwitchboxOp>()) {
      for (ConnectOp connectOp : switchboxOp.getOps<ConnectOp>()) {
        TileID existingCoord = {switchboxOp.colIndex(), switchboxOp.rowIndex()};
        Port existingPort = {connectOp.getDestBundle(),
                             connectOp.getDestChannel()};
        if (!pathfinder.addFixedConnection(existingCoord, existingPort))
          switchboxOp.emitOpError(
              "Couldn't connect tile (" + std::to_string(existingCoord.col) +
              ", " + std::to_string(existingCoord.row) + ") to port (" +
              stringifyWireBundle(existingPort.bundle) + ", " +
              std::to_string(existingPort.channel) + ")\n");
      }
    }

    // all flows are now populated, call the congestion-aware pathfinder
    // algorithm
    // check whether the pathfinder algorithm creates a legal routing
    flowSolutions = pathfinder.findPaths(maxIterations);
    if (!pathfinder.isLegal())
      d.emitError("Unable to find a legal routing");

    // initialize all flows as unprocessed to prep for rewrite
    for (const auto &[pathEndPoint, switchSetting] : flowSolutions) {
      processedFlows[pathEndPoint] = false;
      LLVM_DEBUG(llvm::dbgs() << "Flow starting at (" << pathEndPoint.sb->col
                              << "," << pathEndPoint.sb->row << "):\t");
      LLVM_DEBUG(llvm::dbgs() << stringifySwitchSettings(switchSetting));
    }

    // fill in coords to TileOps, SwitchboxOps, and ShimMuxOps
    for (auto tileOp : device.getOps<TileOp>()) {
      int col, row;
      col = tileOp.colIndex();
      row = tileOp.rowIndex();
      maxCol = std::max(maxCol, col);
      maxRow = std::max(maxRow, row);
      assert(coordToTile.count({col, row}) == 0);
      coordToTile[{col, row}] = tileOp;
    }
    for (auto switchboxOp : device.getOps<SwitchboxOp>()) {
      int col, row;
      col = switchboxOp.colIndex();
      row = switchboxOp.rowIndex();
      assert(coordToSwitchbox.count({col, row}) == 0);
      coordToSwitchbox[{col, row}] = switchboxOp;
    }
    for (auto shimmuxOp : device.getOps<ShimMuxOp>()) {
      int col, row;
      col = shimmuxOp.colIndex();
      row = shimmuxOp.rowIndex();
      assert(coordToShimMux.count({col, row}) == 0);
      coordToShimMux[{col, row}] = shimmuxOp;
    }

    LLVM_DEBUG(llvm::dbgs() << "\t---End DynamicTileAnalysis Constructor---\n");
  }

  int getMaxCol() { return maxCol; }
  int getMaxRow() { return maxRow; }

  TileOp getTile(OpBuilder &builder, int col, int row) {
    if (coordToTile.count({col, row})) {
      return coordToTile[{col, row}];
    } else {
      TileOp tileOp = builder.create<TileOp>(builder.getUnknownLoc(), col, row);
      coordToTile[{col, row}] = tileOp;
      maxCol = std::max(maxCol, col);
      maxRow = std::max(maxRow, row);
      return tileOp;
    }
  }

  SwitchboxOp getSwitchbox(OpBuilder &builder, int col, int row) {
    assert(col >= 0);
    assert(row >= 0);
    if (coordToSwitchbox.count({col, row})) {
      return coordToSwitchbox[{col, row}];
    } else {
      SwitchboxOp switchboxOp = builder.create<SwitchboxOp>(
          builder.getUnknownLoc(), getTile(builder, col, row));
      switchboxOp.ensureTerminator(switchboxOp.getConnections(), builder,
                                   builder.getUnknownLoc());
      coordToSwitchbox[{col, row}] = switchboxOp;
      maxCol = std::max(maxCol, col);
      maxRow = std::max(maxRow, row);
      return switchboxOp;
    }
  }

  ShimMuxOp getShimMux(OpBuilder &builder, int col) {
    assert(col >= 0);
    int row = 0;
    if (coordToShimMux.count({col, row})) {
      return coordToShimMux[{col, row}];
    } else {
      assert(getTile(builder, col, row).isShimNOCTile());
      ShimMuxOp switchboxOp = builder.create<ShimMuxOp>(
          builder.getUnknownLoc(), getTile(builder, col, row));
      switchboxOp.ensureTerminator(switchboxOp.getConnections(), builder,
                                   builder.getUnknownLoc());
      coordToShimMux[{col, row}] = switchboxOp;
      maxCol = std::max(maxCol, col);
      maxRow = std::max(maxRow, row);
      return switchboxOp;
    }
  }
};

// allocates channels between switchboxes ( but does not assign them)
// instantiates shim-muxes AND allocates channels ( no need to rip these up in )
struct ConvertFlowsToInterconnect : public OpConversionPattern<AIE::FlowOp> {
  using OpConversionPattern<AIE::FlowOp>::OpConversionPattern;
  DeviceOp &device;
  DynamicTileAnalysis &analyzer;
  ConvertFlowsToInterconnect(MLIRContext *context, DeviceOp &d,
                             DynamicTileAnalysis &a, PatternBenefit benefit = 1)
      : OpConversionPattern<AIE::FlowOp>(context, benefit), device(d),
        analyzer(a) {}

  LogicalResult match(AIE::FlowOp op) const override { return success(); }

  void addConnection(ConversionPatternRewriter &rewriter,
                     // could be a shim-mux or a switchbox.
                     Interconnect op, FlowOp flowOp, WireBundle inBundle,
                     int inIndex, WireBundle outBundle, int outIndex) const {

    Region &r = op.getConnections();
    Block &b = r.front();
    auto point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(b.getTerminator());

    rewriter.template create<ConnectOp>(rewriter.getUnknownLoc(), inBundle,
                                        inIndex, outBundle, outIndex);

    rewriter.restoreInsertionPoint(point);

    LLVM_DEBUG(llvm::dbgs()
               << "\t\taddConnection() (" << op.colIndex() << ","
               << op.rowIndex() << ") " << stringifyWireBundle(inBundle)
               << inIndex << " -> " << stringifyWireBundle(outBundle)
               << outIndex << "\n");
  }

  void rewrite(AIE::FlowOp flowOp, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    Operation *Op = flowOp.getOperation();

    TileOp srcTile = cast<TileOp>(flowOp.getSource().getDefiningOp());
    TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
    auto srcBundle = flowOp.getSourceBundle();
    auto srcChannel = flowOp.getSourceChannel();
    Port srcPort = {srcBundle, srcChannel};

#ifndef NDEBUG
    TileOp dstTile = cast<TileOp>(flowOp.getDest().getDefiningOp());
    TileID dstCoords = {dstTile.colIndex(), dstTile.rowIndex()};
    auto dstBundle = flowOp.getDestBundle();
    auto dstChannel = flowOp.getDestChannel();
    LLVM_DEBUG(llvm::dbgs()
               << "\n\t---Begin rewrite() for flowOp: (" << srcCoords.col
               << ", " << srcCoords.row << ")" << stringifyWireBundle(srcBundle)
               << srcChannel << " -> (" << dstCoords.col << ", "
               << dstCoords.row << ")" << stringifyWireBundle(dstBundle)
               << (int)dstChannel << "\n\t");
#endif

    // if the flow (aka "net") for this FlowOp hasn't been processed yet,
    // add all switchbox connections to implement the flow
    Switchbox *srcSB = analyzer.pathfinder.getSwitchbox(srcCoords);
    PathEndPoint srcPoint = {srcSB, srcPort};
    if (!analyzer.processedFlows[srcPoint]) {
      SwitchSettings settings = analyzer.flowSolutions[srcPoint];
      // add connections for all the Switchboxes in SwitchSettings
      for (const auto &[curr, setting] : settings) {
        SwitchboxOp swOp =
            analyzer.getSwitchbox(rewriter, curr->col, curr->row);
        int shimCh = srcChannel;
        // TODO: must reserve N3, N7, S2, S3 for DMA connections
        if (*curr == *srcSB &&
            analyzer.getTile(rewriter, srcSB->col, srcSB->row)
                .isShimNOCTile()) {
          // shim DMAs at start of flows
          if (srcBundle == WireBundle::DMA) {
            shimCh = (srcChannel == 0
                          ? 3
                          : 7); // must be either DMA0 -> N3 or DMA1 -> N7
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB->col);
            addConnection(rewriter,
                          cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          } else if (srcBundle ==
                     WireBundle::NOC) { // must be NOC0/NOC1 -> N2/N3 or
                                        // NOC2/NOC3 -> N6/N7
            shimCh = (srcChannel >= 2 ? srcChannel + 4 : srcChannel + 2);
            ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB->col);
            addConnection(rewriter,
                          cast<Interconnect>(shimMuxOp.getOperation()), flowOp,
                          srcBundle, srcChannel, WireBundle::North, shimCh);
          } else if (srcBundle ==
                     WireBundle::PLIO) { // PLIO at start of flows with mux
            if ((srcChannel == 2) || (srcChannel == 3) || (srcChannel == 6) ||
                (srcChannel == 7)) { // Only some PLIO requrie mux
              ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB->col);
              addConnection(
                  rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                  flowOp, srcBundle, srcChannel, WireBundle::North, shimCh);
            }
          }
        }
        for (const auto &[bundle, channel] : setting.dsts) {
          // handle special shim connectivity
          if (*curr == *srcSB &&
              analyzer.getTile(rewriter, srcSB->col, srcSB->row)
                  .isShimNOCorPLTile()) {
            addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                          flowOp, WireBundle::South, shimCh, bundle, channel);
          } else if (analyzer.getTile(rewriter, curr->col, curr->row)
                         .isShimNOCorPLTile() &&
                     (bundle == WireBundle::DMA || bundle == WireBundle::PLIO ||
                      bundle == WireBundle::NOC)) {
            shimCh = channel;
            if (analyzer.getTile(rewriter, curr->col, curr->row)
                    .isShimNOCTile()) {
              // shim DMAs at end of flows
              if (bundle == WireBundle::DMA) {
                shimCh = (channel == 0
                              ? 2
                              : 3); // must be either N2 -> DMA0 or N3 -> DMA1
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr->col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              } else if (bundle == WireBundle::NOC) {
                shimCh = (channel + 2); // must be either N2/3/4/5 -> NOC0/1/2/3
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr->col);
                addConnection(
                    rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                    flowOp, WireBundle::North, shimCh, bundle, channel);
              } else if (channel >=
                         2) { // must be PLIO...only PLIO >= 2 require mux
                ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, curr->col);
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

        LLVM_DEBUG(llvm::dbgs() << " (" << curr->col << "," << curr->row << ") "
                                << stringifyDir(setting.src) << " -> "
                                << stringifyDirs(setting.dsts) << " | ");
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "\n\t\tFinished adding ConnectOps to implement flowOp.\n");
      analyzer.processedFlows[srcPoint] = true;
    } else
      LLVM_DEBUG(llvm::dbgs() << "Flow already processed!\n");

    rewriter.eraseOp(Op);
  }
};

struct AIEPathfinderPass
    : public AIERoutePathfinderFlowsBase<AIEPathfinderPass> {

  /* Overall Flow
   rewrite switchboxes to assign unassigned connections, ensure this can be done
   concurrently ( by different threads)

   // Goal is to rewrite all flows in the device into switchboxes + shim-mux

   // multiple passes of the rewrite pattern rewriting streamswitch
   configurations to routes

   // rewrite flows to stream-switches using 'weights' from analysis pass.

   // check a region is legal

   // rewrite stream-switches (within a bounding box) back to flows */

  void runOnOperation() override {

    // create analysis pass with routing graph for entire device
    LLVM_DEBUG(llvm::dbgs() << "---Begin AIEPathfinderPass---\n");

    DeviceOp d = getOperation();
    DynamicTileAnalysis analyzer(d);
    OpBuilder builder = OpBuilder::atBlockEnd(d.getBody());

    // Apply rewrite rule to switchboxes to add assignments to every 'connect'
    // operation inside
    ConversionTarget target(getContext());
    target.addLegalOp<TileOp>();
    target.addLegalOp<ConnectOp>();
    target.addLegalOp<SwitchboxOp>();
    target.addLegalOp<ShimMuxOp>();
    target.addLegalOp<EndOp>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertFlowsToInterconnect>(d.getContext(), d, analyzer);
    if (failed(applyPartialConversion(d, target, std::move(patterns))))
      signalPassFailure();

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
          builder.create<WireOp>(builder.getUnknownLoc(), tile,
                                 WireBundle::Core, sw, WireBundle::Core);
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

    // If the routing violates architecture-specific routing constraints, then
    // attempt to partially reroute.
    const auto &targetModel = d.getTargetModel();
    std::vector<ConnectOp> problemConnects;
    d.walk([&](ConnectOp connect) {
      if (auto sw = connect->getParentOfType<SwitchboxOp>()) {
        auto tile = sw.getTileOp();
        // Constraint: memtile stream switch constraints
        if (tile.isMemTile() &&
            !targetModel.isLegalMemtileConnection(
                connect.getSourceBundle(), connect.getSourceChannel(),
                connect.getDestBundle(), connect.getDestChannel())) {
          problemConnects.push_back(connect);
        }
      }
    });
    for (auto connect : problemConnects) {
      auto swBox = connect->getParentOfType<SwitchboxOp>();
      builder.setInsertionPoint(connect);
      auto northSw = getSwitchbox(d, swBox.colIndex(), swBox.rowIndex() + 1);
      auto southSw = getSwitchbox(d, swBox.colIndex(), swBox.rowIndex() - 1);
      attemptFixupMemTileRouting(builder, swBox, northSw, southSw, connect);
    }

    return;
  }

  bool attemptFixupMemTileRouting(OpBuilder builder, SwitchboxOp memtileSwOp,
                                  SwitchboxOp northSwOp, SwitchboxOp southSwOp,
                                  ConnectOp &problemConnect) {
    unsigned problemNorthChannel;
    if (problemConnect.getSourceBundle() == WireBundle::North) {
      problemNorthChannel = problemConnect.getSourceChannel();
    } else if (problemConnect.getDestBundle() == WireBundle::North) {
      problemNorthChannel = problemConnect.getDestChannel();
    } else
      return false; // Problem is not about n-s routing
    unsigned problemSouthChannel;
    if (problemConnect.getSourceBundle() == WireBundle::South) {
      problemSouthChannel = problemConnect.getSourceChannel();
    } else if (problemConnect.getDestBundle() == WireBundle::South) {
      problemSouthChannel = problemConnect.getDestChannel();
    } else
      return false; // Problem is not about n-s routing

    // Attempt to reroute northern neighbouring sw
    if (reconnectConnectOps(builder, northSwOp, problemConnect, true,
                            WireBundle::South, problemNorthChannel,
                            problemSouthChannel))
      return true;
    if (reconnectConnectOps(builder, northSwOp, problemConnect, false,
                            WireBundle::South, problemNorthChannel,
                            problemSouthChannel))
      return true;
    // Otherwise, attempt to reroute southern neighbouring sw
    if (reconnectConnectOps(builder, southSwOp, problemConnect, true,
                            WireBundle::North, problemSouthChannel,
                            problemNorthChannel))
      return true;
    if (reconnectConnectOps(builder, southSwOp, problemConnect, false,
                            WireBundle::North, problemSouthChannel,
                            problemNorthChannel))
      return true;
    return false;
  }

  bool reconnectConnectOps(OpBuilder builder, SwitchboxOp sw,
                           ConnectOp problemConnect, bool isIncomingToSW,
                           WireBundle problemBundle, unsigned ProblemChan,
                           unsigned emptyChan) {
    bool hasEmptyChannelSlot = true;
    bool foundCandidateForFixup = false;
    ConnectOp candidate;
    if (isIncomingToSW) {
      for (ConnectOp connect : sw.getOps<ConnectOp>()) {
        if (connect.getDestBundle() == problemBundle &&
            connect.getDestChannel() == ProblemChan) {
          candidate = connect;
          foundCandidateForFixup = true;
        }
        if (connect.getDestBundle() == problemBundle &&
            connect.getDestChannel() == emptyChan) {
          hasEmptyChannelSlot = false;
        }
      }
    } else {
      for (ConnectOp connect : sw.getOps<ConnectOp>()) {
        if (connect.getSourceBundle() == problemBundle &&
            connect.getSourceChannel() == ProblemChan) {
          candidate = connect;
          foundCandidateForFixup = true;
        }
        if (connect.getSourceBundle() == problemBundle &&
            connect.getSourceChannel() == emptyChan) {
          hasEmptyChannelSlot = false;
        }
      }
    }
    if (foundCandidateForFixup && hasEmptyChannelSlot) {
      WireBundle problemBundleOpposite = (problemBundle == WireBundle::North)
                                             ? (WireBundle::South)
                                             : (WireBundle::North);
      // Found empty channel slot, perform reroute
      if (isIncomingToSW) {
        replaceConnectOpWithNewDest(builder, candidate, problemBundle,
                                    emptyChan);
        replaceConnectOpWithNewSource(builder, problemConnect,
                                      problemBundleOpposite, emptyChan);
      } else {
        replaceConnectOpWithNewSource(builder, candidate, problemBundle,
                                      emptyChan);
        replaceConnectOpWithNewDest(builder, problemConnect,
                                    problemBundleOpposite, emptyChan);
      }
      return true;
    }
    return false;
  }

  // Replace connect op
  ConnectOp replaceConnectOpWithNewDest(OpBuilder builder, ConnectOp connect,
                                        WireBundle newBundle, int newChannel) {
    builder.setInsertionPoint(connect);
    auto newOp = builder.create<ConnectOp>(
        builder.getUnknownLoc(), connect.getSourceBundle(),
        connect.getSourceChannel(), newBundle, newChannel);
    connect.erase();
    return newOp;
  }
  ConnectOp replaceConnectOpWithNewSource(OpBuilder builder, ConnectOp connect,
                                          WireBundle newBundle,
                                          int newChannel) {
    builder.setInsertionPoint(connect);
    auto newOp = builder.create<ConnectOp>(builder.getUnknownLoc(), newBundle,
                                           newChannel, connect.getDestBundle(),
                                           connect.getDestChannel());
    connect.erase();
    return newOp;
  }

  SwitchboxOp getSwitchbox(DeviceOp &d, int col, int row) {
    SwitchboxOp output = nullptr;
    d.walk([&](SwitchboxOp swBox) {
      if (swBox.colIndex() == col && swBox.rowIndex() == row) {
        output = swBox;
      }
    });
    return output;
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEPathfinderPass() {
  return std::make_unique<AIEPathfinderPass>();
}
