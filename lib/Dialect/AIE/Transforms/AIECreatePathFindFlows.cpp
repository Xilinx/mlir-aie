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

// allocates channels between switchboxes ( but does not assign them)
// instantiates shim-muxes AND allocates channels ( no need to rip these up in )
struct ConvertFlowsToInterconnect : OpConversionPattern<FlowOp> {
  using OpConversionPattern::OpConversionPattern;
  DeviceOp &device;
  DynamicTileAnalysis &analyzer;
  ConvertFlowsToInterconnect(MLIRContext *context, DeviceOp &d,
                             DynamicTileAnalysis &a, PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), device(d), analyzer(a) {}

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
            if (srcChannel == 2 || srcChannel == 3 || srcChannel == 6 ||
                srcChannel == 7) { // Only some PLIO requrie mux
              ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, srcSB.col);
              addConnection(
                  rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                  flowOp, srcBundle, srcChannel, WireBundle::North, shimCh);
            }
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

void AIEPathfinderPass::runOnOperation() {

  // create analysis pass with routing graph for entire device
  LLVM_DEBUG(llvm::dbgs() << "---Begin AIEPathfinderPass---\n");

  DeviceOp d = getOperation();
  if (failed(analyzer.runAnalysis(d)))
    return signalPassFailure();
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
    return signalPassFailure();

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

  // If the routing violates architecture-specific routing constraints, then
  // attempt to partially reroute.
  const auto &targetModel = d.getTargetModel();
  std::vector<ConnectOp> problemConnects;
  d.walk([&](ConnectOp connect) {
    if (auto sw = connect->getParentOfType<SwitchboxOp>()) {
      // Constraint: memtile stream switch constraints
      if (auto tile = sw.getTileOp();
          tile.isMemTile() &&
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
    if (auto southSw = getSwitchbox(d, swBox.colIndex(), swBox.rowIndex() - 1);
        !attemptFixupMemTileRouting(builder, northSw, southSw, connect))
      return signalPassFailure();
  }
}

bool AIEPathfinderPass::attemptFixupMemTileRouting(const OpBuilder &builder,
                                                   SwitchboxOp northSwOp,
                                                   SwitchboxOp southSwOp,
                                                   ConnectOp &problemConnect) {
  int problemNorthChannel;
  if (problemConnect.getSourceBundle() == WireBundle::North) {
    problemNorthChannel = problemConnect.getSourceChannel();
  } else if (problemConnect.getDestBundle() == WireBundle::North) {
    problemNorthChannel = problemConnect.getDestChannel();
  } else
    return false; // Problem is not about n-s routing
  int problemSouthChannel;
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

bool AIEPathfinderPass::reconnectConnectOps(const OpBuilder &builder,
                                            SwitchboxOp sw,
                                            ConnectOp problemConnect,
                                            bool isIncomingToSW,
                                            WireBundle problemBundle,
                                            int problemChan, int emptyChan) {
  bool hasEmptyChannelSlot = true;
  bool foundCandidateForFixup = false;
  ConnectOp candidate;
  if (isIncomingToSW) {
    for (ConnectOp connect : sw.getOps<ConnectOp>()) {
      if (connect.getDestBundle() == problemBundle &&
          connect.getDestChannel() == problemChan) {
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
          connect.getSourceChannel() == problemChan) {
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
    WireBundle problemBundleOpposite = problemBundle == WireBundle::North
                                           ? WireBundle::South
                                           : WireBundle::North;
    // Found empty channel slot, perform reroute
    if (isIncomingToSW) {
      replaceConnectOpWithNewDest(builder, candidate, problemBundle, emptyChan);
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
ConnectOp AIEPathfinderPass::replaceConnectOpWithNewDest(OpBuilder builder,
                                                         ConnectOp connect,
                                                         WireBundle newBundle,
                                                         int newChannel) {
  builder.setInsertionPoint(connect);
  auto newOp = builder.create<ConnectOp>(
      builder.getUnknownLoc(), connect.getSourceBundle(),
      connect.getSourceChannel(), newBundle, newChannel);
  connect.erase();
  return newOp;
}
ConnectOp AIEPathfinderPass::replaceConnectOpWithNewSource(OpBuilder builder,
                                                           ConnectOp connect,
                                                           WireBundle newBundle,
                                                           int newChannel) {
  builder.setInsertionPoint(connect);
  auto newOp = builder.create<ConnectOp>(builder.getUnknownLoc(), newBundle,
                                         newChannel, connect.getDestBundle(),
                                         connect.getDestChannel());
  connect.erase();
  return newOp;
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
