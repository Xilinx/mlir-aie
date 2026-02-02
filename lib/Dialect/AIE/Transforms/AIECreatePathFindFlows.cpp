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

  void addConnection(ConversionPatternRewriter &rewriter,
                     // could be a shim-mux or a switchbox.
                     Interconnect op, FlowOp flowOp, WireBundle inBundle,
                     int inIndex, WireBundle outBundle, int outIndex) const {

    Region &r = op.getConnections();
    Block &b = r.front();
    auto point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(b.getTerminator());

    ConnectOp::create(rewriter, rewriter.getUnknownLoc(), inBundle, inIndex,
                      outBundle, outIndex);

    rewriter.restoreInsertionPoint(point);

    LLVM_DEBUG(llvm::dbgs()
               << "\t\taddConnection() (" << op.colIndex() << ","
               << op.rowIndex() << ") " << stringifyWireBundle(inBundle)
               << inIndex << " -> " << stringifyWireBundle(outBundle)
               << outIndex << "\n");
  }

  mlir::LogicalResult
  matchAndRewrite(FlowOp flowOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = flowOp.getOperation();
    DeviceOp d = flowOp->getParentOfType<DeviceOp>();
    if (!d) {
      flowOp->emitOpError("This operation must be contained within a device");
      return failure();
    }
    rewriter.setInsertionPoint(d.getBody()->getTerminator());

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
    TileID srcSbId = {srcCoords.col, srcCoords.row};
    PathEndPoint srcPoint = {srcSbId, srcPort};
    if (analyzer.processedFlows[srcPoint]) {
      LLVM_DEBUG(llvm::dbgs() << "Flow already processed!\n");
      rewriter.eraseOp(Op);
      return failure();
    }
    // std::map<TileID, SwitchSetting>
    SwitchSettings settings = analyzer.flowSolutions[srcPoint];
    // add connections for all the Switchboxes in SwitchSettings
    for (const auto &[tileId, setting] : settings) {
      int col = tileId.col;
      int row = tileId.row;
      SwitchboxOp swOp = analyzer.getSwitchbox(rewriter, col, row);
      int shimCh = srcChannel;
      bool isShim = analyzer.getTile(rewriter, tileId).isShimNOCorPLTile();

      // TODO: must reserve N3, N7, S2, S3 for DMA connections
      if (isShim && tileId == srcSbId) {

        // shim DMAs at start of flows
        if (srcBundle == WireBundle::DMA)
          // must be either DMA0 -> N3 or DMA1 -> N7
          shimCh = srcChannel == 0 ? 3 : 7;
        else if (srcBundle == WireBundle::NOC)
          // must be NOC0/NOC1 -> N2/N3 or NOC2/NOC3 -> N6/N7
          shimCh = srcChannel >= 2 ? srcChannel + 4 : srcChannel + 2;
        else if (srcBundle == WireBundle::PLIO)
          shimCh = srcChannel;

        ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, col);
        addConnection(rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                      flowOp, srcBundle, srcChannel, WireBundle::North, shimCh);
      }
      assert(setting.srcs.size() == setting.dsts.size());
      for (size_t i = 0; i < setting.srcs.size(); i++) {
        Port src = setting.srcs[i];
        Port dest = setting.dsts[i];

        // handle special shim connectivity
        if (isShim && tileId == srcSbId) {
          addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                        flowOp, WireBundle::South, shimCh, dest.bundle,
                        dest.channel);
        } else if (isShim && (dest.bundle == WireBundle::DMA ||
                              dest.bundle == WireBundle::PLIO ||
                              dest.bundle == WireBundle::NOC)) {

          // shim DMAs at end of flows
          if (dest.bundle == WireBundle::DMA)
            // must be either N2 -> DMA0 or N3 -> DMA1
            shimCh = dest.channel == 0 ? 2 : 3;
          else if (dest.bundle == WireBundle::NOC)
            // must be either N2/3/4/5 -> NOC0/1/2/3
            shimCh = dest.channel + 2;
          else if (dest.bundle == WireBundle::PLIO)
            shimCh = dest.channel;

          ShimMuxOp shimMuxOp = analyzer.getShimMux(rewriter, col);
          addConnection(rewriter, cast<Interconnect>(shimMuxOp.getOperation()),
                        flowOp, WireBundle::North, shimCh, dest.bundle,
                        dest.channel);
          addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                        flowOp, src.bundle, src.channel, WireBundle::South,
                        shimCh);
        } else {
          // otherwise, regular switchbox connection
          addConnection(rewriter, cast<Interconnect>(swOp.getOperation()),
                        flowOp, src.bundle, src.channel, dest.bundle,
                        dest.channel);
        }
      }

      LLVM_DEBUG(llvm::dbgs() << tileId << ": " << setting << " | "
                              << "\n");
    }

    LLVM_DEBUG(llvm::dbgs()
               << "\n\t\tFinished adding ConnectOps to implement flowOp.\n");

    analyzer.processedFlows[srcPoint] = true;
    rewriter.eraseOp(Op);
    return success();
  }
};

} // namespace

namespace xilinx::AIE {

LogicalResult AIEPathfinderPass::runOnFlow(DeviceOp d,
                                           DynamicTileAnalysis &analyzer) {
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
    return failure();
  return success();
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
  for (const auto &[sbNode, setting] : settings) {
    TileID tile = {sbNode.col, sbNode.row};
    if (tile == neighbourTile) {
      assert(setting.srcs.size() == setting.dsts.size());
      for (size_t i = 0; i < setting.srcs.size(); i++) {
        Port src = setting.srcs[i];
        Port dest = setting.dsts[i];
        if ((src.bundle == neighbourSourceBundle) &&
            (src.channel == neighbourSourceChannel)) {
          if (findPathToDest(settings, neighbourTile, dest.bundle, dest.channel,
                             finalTile, finalDestBundle, finalDestChannel)) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

LogicalResult
AIEPathfinderPass::runOnPacketFlow(DeviceOp device, OpBuilder &builder,
                                   DynamicTileAnalysis &analyzer) {

  ConversionTarget target(getContext());

  std::map<TileID, mlir::Operation *> tiles;

  // Map from a port and flowID to
  std::map<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> packetFlows;
  std::map<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> ctrlPacketFlows;
  SmallVector<std::pair<PhysPort, int>, 4> slavePorts;
  DenseMap<std::pair<PhysPort, int>, int> slaveAMSels;
  // Flag to keep packet header at packet flow destination
  DenseMap<PhysPort, BoolAttr> keepPktHeaderAttr;
  // Map from tileID and master ports to flags labelling control packet flows
  DenseMap<std::pair<PhysPort, int>, bool> ctrlPktFlows;

  for (auto tileOp : device.getOps<TileOp>()) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    tiles[{col, row}] = tileOp;
  }

  // The logical model of all the switchboxes.
  std::map<TileID, SmallVector<std::pair<Connect, int>, 8>> switchboxes;
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
        auto keep = pktFlowOp.getKeepPktHeader();
        keepPktHeaderAttr[{destTile.getTileID(), destPort}] =
            keep ? BoolAttr::get(Op.getContext(), *keep) : nullptr;

        TileID srcSB = {srcCoords.col, srcCoords.row};
        if (PathEndPoint srcPoint = {srcSB, srcPort};
            !analyzer.processedFlows[srcPoint]) {
          SwitchSettings settings = analyzer.flowSolutions[srcPoint];
          // add connections for all the Switchboxes in SwitchSettings
          for (const auto &[curr, setting] : settings) {
            assert(setting.srcs.size() == setting.dsts.size());
            TileID currTile = {curr.col, curr.row};
            for (size_t i = 0; i < setting.srcs.size(); i++) {
              Port src = setting.srcs[i];
              Port dest = setting.dsts[i];
              // reject false broadcast
              if (!findPathToDest(settings, currTile, dest.bundle, dest.channel,
                                  destCoords, destPort.bundle,
                                  destPort.channel))
                continue;
              Connect connect = {{src.bundle, src.channel},
                                 {dest.bundle, dest.channel}};
              if (std::find(switchboxes[currTile].begin(),
                            switchboxes[currTile].end(),
                            std::pair{connect, flowID}) ==
                  switchboxes[currTile].end())
                switchboxes[currTile].push_back({connect, flowID});
              // Assign "control packet flows" flag per switchbox, based on
              // packet flow op attribute
              auto ctrlPkt = pktFlowOp.getPriorityRoute();
              ctrlPktFlows[{{currTile, dest}, flowID}] =
                  ctrlPkt ? *ctrlPkt : false;
            }
          }
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Check switchboxes\n");

  for (const auto &[tileId, connects] : switchboxes) {
    LLVM_DEBUG(llvm::dbgs() << "***switchbox*** " << tileId.col << " "
                            << tileId.row << '\n');
    for (const auto &[conn, flowID] : connects) {
      Port sourcePort = conn.src;
      Port destPort = conn.dst;
      auto sourceFlow =
          std::make_pair(std::make_pair(tileId, sourcePort), flowID);
      if (ctrlPktFlows[{{tileId, destPort}, flowID}])
        ctrlPacketFlows[sourceFlow].push_back({tileId, destPort});
      else
        packetFlows[sourceFlow].push_back({tileId, destPort});
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

  // Constants for arbiter configuration
  constexpr int INVALID_AMSEL_VALUE = -1;
  constexpr int INVALID_ARBITER_VALUE = -1;

  // A map from Tile and master selectValue to the ports targetted by that
  // master select.
  std::map<std::pair<TileID, int>, SmallVector<Port, 4>> masterAMSels;

  // Track which arbiter each port is assigned to (to prevent conflicts)
  std::map<PhysPort, int> portToArbiter;

  // Count of currently used logical arbiters for each tile.
  DenseMap<Operation *, int> amselValues;
  int numMselsPerArbiter = 4;
  int numArbiters = 6;

  // Get arbiter id from amsel
  auto getArbiterIDFromAmsel = [numArbiters](int amsel) {
    return amsel % numArbiters;
  };
  // Get amsel from arbiter id and msel
  auto getAmselFromArbiterIDAndMsel = [numArbiters](int arbiter, int msel) {
    return arbiter + msel * numArbiters;
  };
  // Get a new unique amsel from masterAMSels on tile op. Prioritize on
  // incrementing arbiter id, before incrementing msel
  auto getNewUniqueAmsel =
      [&](const std::map<std::pair<TileID, int>, SmallVector<Port, 4>>
              &masterAMSels,
          TileOp tileOp, bool isCtrlPkt) {
        if (isCtrlPkt) { // Higher AMsel first
          for (int i = numMselsPerArbiter - 1; i >= 0; i--)
            for (int a = numArbiters - 1; a >= 0; a--)
              if (!masterAMSels.count(
                      {tileOp.getTileID(), getAmselFromArbiterIDAndMsel(a, i)}))
                return getAmselFromArbiterIDAndMsel(a, i);
        } else { // Lower AMsel first
          for (int i = 0; i < numMselsPerArbiter; i++)
            for (int a = 0; a < numArbiters; a++)
              if (!masterAMSels.count(
                      {tileOp.getTileID(), getAmselFromArbiterIDAndMsel(a, i)}))
                return getAmselFromArbiterIDAndMsel(a, i);
        }
        tileOp->emitOpError(
            "tile op has used up all arbiter-msel combinations");
        return INVALID_AMSEL_VALUE;
      };
  // Get a new unique amsel from masterAMSels on tile op with given arbiter id
  auto getNewUniqueAmselPerArbiterID =
      [&](const std::map<std::pair<TileID, int>, SmallVector<Port, 4>>
              &masterAMSels,
          TileOp tileOp, int arbiter) {
        for (int i = 0; i < numMselsPerArbiter; i++)
          if (!masterAMSels.count({tileOp.getTileID(),
                                   getAmselFromArbiterIDAndMsel(arbiter, i)}))
            return getAmselFromArbiterIDAndMsel(arbiter, i);
        tileOp->emitOpError("tile op arbiter ")
            << std::to_string(arbiter) << " has used up all its msels";
        return INVALID_AMSEL_VALUE;
      };

  // Order the packet flows in order to get determinsitic amsel allocation;
  // allocate amsels for control packet flows before others to ensure
  // consistency in control packet flow overlay.
  auto getSortedPacketFlows =
      [&](std::map<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>> pktFlows,
          std::map<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>>
              ctrlPktFlows) {
        std::vector<
            std::pair<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>>>
            sortedpktFlows(pktFlows.begin(), pktFlows.end());
        std::vector<
            std::pair<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>>>
            sortedctrlpktFlows(ctrlPktFlows.begin(), ctrlPktFlows.end());
        sortedctrlpktFlows.insert(sortedctrlpktFlows.end(),
                                  sortedpktFlows.begin(), sortedpktFlows.end());
        return sortedctrlpktFlows;
      };

  std::vector<std::pair<std::pair<PhysPort, int>, SmallVector<PhysPort, 4>>>
      sortedPacketFlows = getSortedPacketFlows(packetFlows, ctrlPacketFlows);

  packetFlows.insert(ctrlPacketFlows.begin(), ctrlPacketFlows.end());

  // Helper function to assign ports to a given amsel value
  // Updates masterAMSels and portToArbiter, skipping ports already on different
  // arbiters
  auto assignPortsToAmsel = [&](TileID tileId, int amselValue,
                                const SmallVector<PhysPort, 4> &destinations) {
    int targetArbiter = getArbiterIDFromAmsel(amselValue);
    for (auto dest : destinations) {
      Port port = dest.second;
      PhysPort physPort = {tileId, port};

      // Skip this port if it's already assigned to a different arbiter
      if (portToArbiter.count(physPort) &&
          portToArbiter[physPort] != targetArbiter) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skipping port " << stringifyWireBundle(port.bundle)
                   << ":" << port.channel << " - already on arbiter "
                   << portToArbiter[physPort] << ", target is " << targetArbiter
                   << "\n");
        continue;
      }

      masterAMSels[{tileId, amselValue}].push_back(port);
      portToArbiter[physPort] = targetArbiter;
    }
  };

  // Helper function to find existing arbiter assignment for ports in a flow
  // Returns -1 if no existing assignment found
  auto findExistingArbiter =
      [&](TileID tileId, const SmallVector<PhysPort, 4> &destinations) -> int {
    for (auto dest : destinations) {
      Port port = dest.second;
      PhysPort physPort = {tileId, port};
      if (portToArbiter.count(physPort)) {
        return portToArbiter[physPort];
      }
    }
    return INVALID_ARBITER_VALUE;
  };

  // Check all multi-cast flows (same source, same ID). They should be
  // assigned the same arbiter and msel so that the flow can reach all the
  // destination ports at the same time For destination ports that appear in
  // different (multicast) flows, it should have a different <arbiterID, msel>
  // value pair for each flow
  for (const auto &packetFlow : sortedPacketFlows) {
    // The Source Tile of the flow
    TileID tileId = packetFlow.first.first.first;
    TileOp tileOp = analyzer.getTile(builder, tileId);
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

    // Find existing arbiter and amsel assignments for this flow
    // Strategy: Look for existing amsel entries that match the flow's
    // destinations
    // - Complete match: reuse existing amsel
    // - Partial match: create new amsel on same arbiter
    // - No match: create new amsel (on existing arbiter if ports already
    // assigned)
    bool hasMatchingAmselEntry = false;
    int partialMatchArbiterID = INVALID_ARBITER_VALUE;

    // Check if any ports in this flow already have arbiter assignments
    int existingArbiter = findExistingArbiter(tileId, packetFlow.second);

    // Search for matching amsel entries, prioritizing those that match the
    // existing arbiter
    for (const auto &amselEntry : masterAMSels) {
      if (amselEntry.first.first != tileId)
        continue;
      amselValue = amselEntry.first.second;
      int thisArbiter = getArbiterIDFromAmsel(amselValue);

      // If we have an existing arbiter assignment, only consider amsels from
      // that arbiter
      if (existingArbiter != INVALID_ARBITER_VALUE &&
          existingArbiter != thisArbiter) {
        continue;
      }

      // Check if destinations match (completely or partially)
      const SmallVector<Port, 4> &existingPorts =
          masterAMSels[{tileId, amselValue}];
      bool hasOverlap = false;
      bool hasNonOverlap = false;

      for (auto dest : packetFlow.second) {
        Port port = dest.second;
        if (std::find(existingPorts.begin(), existingPorts.end(), port) ==
            existingPorts.end())
          hasNonOverlap = true;
        else
          hasOverlap = true;
      }

      if (hasOverlap) {
        hasMatchingAmselEntry = true;
        // Partial match if some ports don't overlap or sizes differ
        if (hasNonOverlap || existingPorts.size() != packetFlow.second.size())
          partialMatchArbiterID = thisArbiter;
        break;
      }
    }

    if (!hasMatchingAmselEntry) {
      // This packet flow switchbox's output ports completely mismatches with
      // any existing amsel. Creating a new amsel.

      // Determine target arbiter: use existing if available, otherwise allocate
      // based on priority
      int targetArbiter = existingArbiter;

      if (targetArbiter == INVALID_ARBITER_VALUE) {
        // No existing assignment, choose based on control packet priority
        bool isCtrlPkt =
            llvm::any_of(packetFlow.second, [&](PhysPort destPhysPort) {
              Port port = destPhysPort.second;
              return ctrlPktFlows[{{tileId, port}, packetFlow.first.second}];
            });

        amselValue = getNewUniqueAmsel(masterAMSels, tileOp, isCtrlPkt);
        if (amselValue == INVALID_AMSEL_VALUE)
          return failure();
      } else {
        // Use existing arbiter to maintain consistency
        amselValue =
            getNewUniqueAmselPerArbiterID(masterAMSels, tileOp, targetArbiter);
        if (amselValue == INVALID_AMSEL_VALUE) {
          // No more msels available on this arbiter - this is a routing
          // conflict
          tileOp->emitOpError("cannot assign flow: arbiter ")
              << targetArbiter
              << " has no free msels, but flow requires this arbiter due to "
                 "existing port assignments";
          return failure();
        }
      }

      // Update masterAMSels with new amsel, skipping ports already assigned to
      // different arbiters
      assignPortsToAmsel(tileId, amselValue, packetFlow.second);
    } else if (partialMatchArbiterID != INVALID_ARBITER_VALUE) {
      // This packet flow switchbox's output ports partially overlaps with
      // some existing amsel. Create a NEW amsel with the SAME arbiter for this
      // flow. The comment states: "destination ports that appear in different
      // (multicast) flows should have a different <arbiterID, msel> value pair
      // for each flow" but use the same arbiter to maintain the constraint.

      // Use the arbiter we found (which should match existingArbiter if set)
      int targetArbiter = partialMatchArbiterID;
      if (existingArbiter != INVALID_ARBITER_VALUE &&
          existingArbiter != targetArbiter) {
        // Conflict detected - should have been caught earlier, but add safety
        // check
        tileOp->emitOpError(
            "internal error: arbiter conflict in partial match");
        return failure();
      }

      amselValue =
          getNewUniqueAmselPerArbiterID(masterAMSels, tileOp, targetArbiter);

      // Update masterAMSels with new amsel, skipping ports already assigned to
      // different arbiters
      assignPortsToAmsel(tileId, amselValue, packetFlow.second);
    } else {
      // Complete match - reuse the existing amsel
      // Track arbiter assignments for all ports in this flow
      int arbiter = getArbiterIDFromAmsel(amselValue);
      for (auto dest : packetFlow.second) {
        Port port = dest.second;
        PhysPort physPort = {tileId, port};
        // Update tracking even for reused amsels
        if (!portToArbiter.count(physPort)) {
          portToArbiter[physPort] = arbiter;
        }
      }
    }

    slaveAMSels[packetFlow.first] = amselValue;
    amselValues[tileOp] = getArbiterIDFromAmsel(amselValue);
  }

  // Compute the master set IDs
  // A map from a switchbox output port to its associated amsel values
  std::map<PhysPort, SmallVector<int, 4>> mastersets;
  for (const auto &[physPort, ports] : masterAMSels) {
    TileID tileId = physPort.first;
    int amselValue = physPort.second;
    for (auto port : ports) {
      PhysPort physPort = {tileId, port};
      mastersets[physPort].push_back(amselValue);
    }
  }

  // Validate that each port only has amsels from a single arbiter
  for (const auto &[physPort, amselList] : mastersets) {
    int assignedArbiter = INVALID_ARBITER_VALUE;
    for (auto amsel : amselList) {
      int thisArbiter = getArbiterIDFromAmsel(amsel);
      if (assignedArbiter != INVALID_ARBITER_VALUE &&
          assignedArbiter != thisArbiter) {
        TileID tileId = physPort.first;
        Port port = physPort.second;
        TileOp tileOp = analyzer.getTile(builder, tileId);
        tileOp->emitOpError("port ")
            << stringifyWireBundle(port.bundle) << ":" << port.channel
            << " assigned to multiple arbiters: " << assignedArbiter << " and "
            << thisArbiter << " (amsels: " << amselList[0];
        for (size_t i = 1; i < amselList.size(); i++) {
          llvm::errs() << ", " << amselList[i];
        }
        llvm::errs() << ")\n";
        return failure();
      }
      assignedArbiter = thisArbiter;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "CHECK mastersets\n");
#ifndef NDEBUG
  for (const auto &[physPort, values] : mastersets) {
    TileID tileId = physPort.first;
    WireBundle bundle = physPort.second.bundle;
    int channel = physPort.second.channel;
    LLVM_DEBUG(llvm::dbgs()
               << "master " << tileId << " " << stringifyWireBundle(bundle)
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

  std::map<std::pair<PhysPort, int>, int> slaveMasks;
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
    PhysPort port = map.first.first;
    TileOp tile = analyzer.getTile(builder, port.first);
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

  // Update tiles map if any new tile op declaration is needed for constructing
  // the flow.
  for (const auto &swMap : mastersets) {
    TileID tileId = swMap.first.first;
    TileOp tileOp = analyzer.getTile(builder, tileId);
    if (std::none_of(tiles.begin(), tiles.end(),
                     [&tileOp](const std::pair<const xilinx::AIE::TileID,
                                               Operation *> &tileMapEntry) {
                       return tileMapEntry.second == tileOp.getOperation();
                     })) {
      tiles[{tileOp.colIndex(), tileOp.rowIndex()}] = tileOp;
    }
  }

  for (auto map : tiles) {
    Operation *tileOp = map.second;
    TileOp tile = cast<TileOp>(map.second);
    TileID tileId = tile.getTileID();

    // Create a switchbox for the routes and insert inside it.
    builder.setInsertionPointAfter(tileOp);
    SwitchboxOp swbox =
        analyzer.getSwitchbox(builder, tile.colIndex(), tile.rowIndex());
    SwitchboxOp::ensureTerminator(swbox.getConnections(), builder,
                                  builder.getUnknownLoc());
    Block &b = swbox.getConnections().front();
    builder.setInsertionPoint(b.getTerminator());

    std::vector<bool> amselOpNeededVector(numMselsPerArbiter * numArbiters);
    for (const auto &map : mastersets) {
      if (tileId != map.first.first)
        continue;

      for (auto value : map.second) {
        amselOpNeededVector[value] = true;
      }
    }
    // Create all the amsel Ops
    std::map<int, AMSelOp> amselOps;
    for (int i = 0; i < numMselsPerArbiter; i++) {
      for (int a = 0; a < numArbiters; a++) {
        auto amselValue = getAmselFromArbiterIDAndMsel(a, i);
        if (amselOpNeededVector[amselValue]) {
          int arbiterID = a;
          int msel = i;
          auto amsel = AMSelOp::create(builder, builder.getUnknownLoc(),
                                       arbiterID, msel);
          amselOps[amselValue] = amsel;
        }
      }
    }
    // Create all the master set Ops
    // First collect the master sets for this tile.
    SmallVector<Port, 4> tileMasters;
    for (const auto &map : mastersets) {
      if (tileId != map.first.first)
        continue;
      tileMasters.push_back(map.first.second);
    }
    // Sort them so we get a reasonable order
    std::sort(tileMasters.begin(), tileMasters.end());
    for (auto tileMaster : tileMasters) {
      WireBundle bundle = tileMaster.bundle;
      int channel = tileMaster.channel;
      SmallVector<int, 4> msels = mastersets[{tileId, tileMaster}];
      SmallVector<Value, 4> amsels;
      for (auto msel : msels) {
        assert(amselOps.count(msel) == 1);
        amsels.push_back(amselOps[msel]);
      }

      MasterSetOp::create(builder, builder.getUnknownLoc(),
                          builder.getIndexType(), bundle, channel, amsels,
                          keepPktHeaderAttr[{tileId, tileMaster}]);
    }

    // Generate the packet rules
    DenseMap<Port, PacketRulesOp> slaveRules;
    for (auto group : slaveGroups) {
      builder.setInsertionPoint(b.getTerminator());

      auto port = group.front().first;
      if (tileId != port.first)
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
        packetrules = PacketRulesOp::create(builder, builder.getUnknownLoc(),
                                            bundle, channel);
        PacketRulesOp::ensureTerminator(packetrules.getRules(), builder,
                                        builder.getUnknownLoc());
        slaveRules[slave] = packetrules;
      } else
        packetrules = slaveRules[slave];

      Block &rules = packetrules.getRules().front();

      // Verify ID mapping against all other rules of the same slave.
      for (auto rule : rules.getOps<PacketRuleOp>()) {
        auto verifyMask = rule.maskInt();
        auto verifyValue = rule.valueInt();
        if ((group.front().second & verifyMask) == verifyValue) {
          rule->emitOpError("can lead to false packet id match for id ")
              << ID << ", which is not supposed to pass through this port.";
          rule->emitRemark("Please consider changing all uses of packet id ")
              << ID << " to avoid deadlock.";
          return failure();
        }
      }

      builder.setInsertionPoint(rules.getTerminator());
      PacketRuleOp::create(builder, builder.getUnknownLoc(), mask, ID, amsel);
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
            shimOp = analyzer.getShimMux(builder, tileOp.colIndex());
            shimExist = 1;
          }

          Region &r0 = shimOp.getConnections();
          Block &b0 = r0.front();
          builder.setInsertionPointToStart(&b0);

          pktrules.setSourceBundle(WireBundle::South);
          if (pktrules.getSourceChannel() == 0) {
            pktrules.setSourceChannel(3);
            ConnectOp::create(builder, builder.getUnknownLoc(), WireBundle::DMA,
                              0, WireBundle::North, 3);
          }
          if (pktrules.getSourceChannel() == 1) {
            pktrules.setSourceChannel(7);
            ConnectOp::create(builder, builder.getUnknownLoc(), WireBundle::DMA,
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
            shimOp = analyzer.getShimMux(builder, tileOp.colIndex());
            shimExist = 1;
          }

          Region &r0 = shimOp.getConnections();
          Block &b0 = r0.front();
          builder.setInsertionPointToStart(&b0);

          mtset.setDestBundle(WireBundle::South);
          if (mtset.getDestChannel() == 0) {
            mtset.setDestChannel(2);
            ConnectOp::create(builder, builder.getUnknownLoc(),
                              WireBundle::North, 2, WireBundle::DMA, 0);
          }
          if (mtset.getDestChannel() == 1) {
            mtset.setDestChannel(3);
            ConnectOp::create(builder, builder.getUnknownLoc(),
                              WireBundle::North, 3, WireBundle::DMA, 1);
          }
        }
      }
    }
  }

  RewritePatternSet patterns(&getContext());

  if (failed(applyPartialConversion(device, target, std::move(patterns))))
    return failure();

  return success();
}

void AIEPathfinderPass::runOnOperation() {

  // create analysis pass with routing graph for entire device
  LLVM_DEBUG(llvm::dbgs() << "---Begin AIEPathfinderPass---\n");

  DeviceOp d = getOperation();
  DynamicTileAnalysis &analyzer = getAnalysis<DynamicTileAnalysis>();
  if (failed(analyzer.runAnalysis(d))) {
    signalPassFailure();
    return;
  }
  OpBuilder builder = OpBuilder::atBlockTerminator(d.getBody());

  if (clRouteCircuit && failed(runOnFlow(d, analyzer))) {
    signalPassFailure();
    return;
  }
  if (clRoutePacket && failed(runOnPacketFlow(d, builder, analyzer))) {
    signalPassFailure();
    return;
  }

  // Populate wires between switchboxes and tiles.
  builder.setInsertionPoint(d.getBody()->getTerminator());
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
          WireOp::create(builder, builder.getUnknownLoc(), westsw,
                         WireBundle::East, sw, WireBundle::West);
        }
      }
      if (row > 0) {
        // connections between abstract 'core' of tile
        WireOp::create(builder, builder.getUnknownLoc(), tile, WireBundle::Core,
                       sw, WireBundle::Core);
        // connections between abstract 'dma' of tile
        WireOp::create(builder, builder.getUnknownLoc(), tile, WireBundle::DMA,
                       sw, WireBundle::DMA);
        // connections north-south inside array ( including connection to shim
        // row)
        if (analyzer.coordToSwitchbox.count({col, row - 1})) {
          auto southsw = analyzer.coordToSwitchbox[{col, row - 1}];
          WireOp::create(builder, builder.getUnknownLoc(), southsw,
                         WireBundle::North, sw, WireBundle::South);
        }
      } else if (row == 0) {
        if (tile.isShimNOCTile()) {
          if (analyzer.coordToShimMux.count({col, 0})) {
            auto shimsw = analyzer.coordToShimMux[{col, 0}];
            WireOp::create(
                builder, builder.getUnknownLoc(), shimsw,
                WireBundle::North, // Changed to connect into the north
                sw, WireBundle::South);
            // PLIO is attached to shim mux
            if (analyzer.coordToPLIO.count(col)) {
              auto plio = analyzer.coordToPLIO[col];
              WireOp::create(builder, builder.getUnknownLoc(), plio,
                             WireBundle::North, shimsw, WireBundle::South);
            }

            // abstract 'DMA' connection on tile is attached to shim mux ( in
            // row 0 )
            WireOp::create(builder, builder.getUnknownLoc(), tile,
                           WireBundle::DMA, shimsw, WireBundle::DMA);
          }
        } else if (tile.isShimPLTile()) {
          // PLIO is attached directly to switch
          if (analyzer.coordToPLIO.count(col)) {
            auto plio = analyzer.coordToPLIO[col];
            WireOp::create(builder, builder.getUnknownLoc(), plio,
                           WireBundle::North, sw, WireBundle::South);
          }
        }
      }
    }
  }
}

std::unique_ptr<OperationPass<DeviceOp>> createAIEPathfinderPass() {
  return std::make_unique<AIEPathfinderPass>();
}

} // namespace xilinx::AIE
