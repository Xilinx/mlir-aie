//===- AIEHerdRouting.cpp ---------------------------------------*- C++ -*-===//
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
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

template <typename MyOp>
struct AIEOpRemoval : public OpConversionPattern<MyOp> {
  using OpConversionPattern<MyOp>::OpConversionPattern;
  using OpAdaptor = typename MyOp::Adaptor;

  AIEOpRemoval(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(MyOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    rewriter.eraseOp(Op);
    return success();
  }
};

int getAvailableDestChannel(SmallVector<Connect, 8> &connects, Port sourcePort,
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

  // look for existing connect
  for (int i = 0; i < numChannels; i++) {
    Port port = std::make_pair(destBundle, i);
    if (std::find(connects.begin(), connects.end(),
                  std::make_pair(sourcePort, port)) != connects.end())
      return i;
  }

  // if not, look for available destination port
  for (int i = 0; i < numChannels; i++) {
    Port port = std::make_pair(destBundle, i);
    SmallVector<Port, 8> ports;
    for (auto connect : connects)
      ports.push_back(connect.second);

    if (std::find(ports.begin(), ports.end(), port) == ports.end())
      return i;
  }

  return -1;
}

void buildRoute(int xSrc, int ySrc, int xDest, int yDest,
                WireBundle sourceBundle, int sourceChannel,
                WireBundle destBundle, int destChannel, Operation *herdOp,
                DenseMap<std::pair<Operation *, std::pair<int, int>>,
                         SmallVector<Connect, 8>> &switchboxes) {

  int xCur = xSrc;
  int yCur = ySrc;
  WireBundle curBundle;
  int curChannel;
  int xLast, yLast;
  WireBundle lastBundle;
  Port lastPort = std::make_pair(sourceBundle, sourceChannel);

  SmallVector<std::pair<int, int>, 4> congestion;

  llvm::dbgs() << "Build route: " << xSrc << " " << ySrc << " --> " << xDest
               << " " << yDest << '\n';
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
      curChannel = getAvailableDestChannel(
          switchboxes[std::make_pair(herdOp, curCoord)], lastPort, move);
      if (curChannel == -1)
        continue;

      if (move == lastBundle)
        continue;

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
      switchboxes[std::make_pair(herdOp, curCoord)]
          .pop_back(); // back up, remove the last connection
    } else {
      llvm::dbgs() << "[" << stringifyWireBundle(lastPort.first) << " : "
                   << lastPort.second
                   << "], "
                      "["
                   << stringifyWireBundle(curBundle) << " : " << curChannel
                   << "]\n";

      Port curPort = std::make_pair(curBundle, curChannel);
      Connect connect = std::make_pair(lastPort, curPort);
      if (std::find(switchboxes[std::make_pair(herdOp, curCoord)].begin(),
                    switchboxes[std::make_pair(herdOp, curCoord)].end(),
                    connect) ==
          switchboxes[std::make_pair(herdOp, curCoord)].end())
        switchboxes[std::make_pair(herdOp, curCoord)].push_back(connect);
      lastPort = std::make_pair(lastBundle, curChannel);
    }
  }

  llvm::dbgs() << "coord " << xCur << " " << yCur << '\n';
  llvm::dbgs() << "[" << stringifyWireBundle(lastPort.first) << " : "
               << lastPort.second
               << "], "
                  "["
               << stringifyWireBundle(destBundle) << " : " << destChannel
               << "]\n";

  switchboxes[std::make_pair(herdOp, std::make_pair(xCur, yCur))].push_back(
      std::make_pair(lastPort, std::make_pair(destBundle, destChannel)));
}

struct AIEHerdRoutingPass : public AIEHerdRoutingBase<AIEHerdRoutingPass> {
  void runOnOperation() override {

    DeviceOp device = getOperation();
    OpBuilder builder(device.getBody()->getTerminator());

    SmallVector<HerdOp, 4> herds;
    SmallVector<Operation *, 4> placeOps;
    SmallVector<Operation *, 4> routeOps;
    DenseMap<std::pair<Operation *, Operation *>, std::pair<int, int>>
        distances;
    SmallVector<std::pair<std::pair<int, int>, std::pair<int, int>>, 4> routes;
    DenseMap<std::pair<Operation *, std::pair<int, int>>,
             SmallVector<Connect, 8>>
        switchboxes;

    for (auto herd : device.getOps<HerdOp>()) {
      herds.push_back(herd);
    }

    for (auto placeOp : device.getOps<PlaceOp>()) {
      placeOps.push_back(placeOp);
      Operation *sourceHerd = placeOp.getSourceHerd().getDefiningOp();
      Operation *destHerd = placeOp.getDestHerd().getDefiningOp();
      int distX = placeOp.getDistXValue();
      int distY = placeOp.getDistYValue();
      distances[std::make_pair(sourceHerd, destHerd)] =
          std::make_pair(distX, distY);
    }

    // FIXME: multiple route ops with different sourceHerds does not seem to be
    // aware of the routes done before
    for (auto routeOp : device.getOps<RouteOp>()) {
      routeOps.push_back(routeOp);

      AIEX::SelectOp sourceHerds =
          dyn_cast<AIEX::SelectOp>(routeOp.getSourceHerds().getDefiningOp());
      AIEX::SelectOp destHerds =
          dyn_cast<AIEX::SelectOp>(routeOp.getDestHerds().getDefiningOp());
      WireBundle sourceBundle = routeOp.getSourceBundle();
      WireBundle destBundle = routeOp.getDestBundle();
      int sourceChannel = routeOp.getSourceChannelValue();
      int destChannel = routeOp.getDestChannelValue();

      HerdOp sourceHerd =
          dyn_cast<HerdOp>(sourceHerds.getStartHerd().getDefiningOp());
      IterOp sourceIterX =
          dyn_cast<IterOp>(sourceHerds.getIterX().getDefiningOp());
      IterOp sourceIterY =
          dyn_cast<IterOp>(sourceHerds.getIterY().getDefiningOp());

      HerdOp destHerd =
          dyn_cast<HerdOp>(destHerds.getStartHerd().getDefiningOp());
      IterOp destIterX = dyn_cast<IterOp>(destHerds.getIterX().getDefiningOp());
      IterOp destIterY = dyn_cast<IterOp>(destHerds.getIterY().getDefiningOp());

      int sourceStartX = sourceIterX.getStartValue();
      int sourceEndX = sourceIterX.getEndValue();
      int sourceStrideX = sourceIterX.getStrideValue();
      int sourceStartY = sourceIterY.getStartValue();
      int sourceEndY = sourceIterY.getEndValue();
      int sourceStrideY = sourceIterY.getStrideValue();

      int destStartX = destIterX.getStartValue();
      int destEndX = destIterX.getEndValue();
      int destStrideX = destIterX.getStrideValue();
      int destStartY = destIterY.getStartValue();
      int destEndY = destIterY.getEndValue();
      int destStrideY = destIterY.getStrideValue();

      assert(distances.count(std::make_pair(sourceHerd, destHerd)) == 1);

      std::pair<int, int> distance =
          distances[std::make_pair(sourceHerd, destHerd)];
      int distX = distance.first;
      int distY = distance.second;
      // FIXME: this looks like it can be improved further ...
      for (int xSrc = sourceStartX; xSrc < sourceEndX; xSrc += sourceStrideX) {
        for (int ySrc = sourceStartY; ySrc < sourceEndY;
             ySrc += sourceStrideY) {
          for (int xDst = destStartX; xDst < destEndX; xDst += destStrideX) {
            for (int yDst = destStartY; yDst < destEndY; yDst += destStrideY) {
              // Build route (x0, y0) --> (x1, y1)
              int x0 = xSrc;
              int y0 = ySrc;
              int x1 = xDst;
              int y1 = yDst;
              if (destIterX == sourceIterX)
                x1 = x0;
              if (destIterY == sourceIterX)
                y1 = x0;
              if (destIterX == sourceIterY)
                x1 = y0;
              if (destIterY == sourceIterY)
                y1 = y0;

              auto route = std::make_pair(
                  std::make_pair(x0, y0),
                  std::make_pair(distX + x1 - x0, distY + y1 - y0));
              if (std::find(routes.begin(), routes.end(), route) !=
                  routes.end())
                continue;

              buildRoute(x0, y0, x1 + distX, y1 + distY, sourceBundle,
                         sourceChannel, destBundle, destChannel, sourceHerd,
                         switchboxes);

              routes.push_back(route);
            }
          }
        }
      }
    }

    for (auto swboxCfg : switchboxes) {
      Operation *herdOp = swboxCfg.first.first;
      int x = swboxCfg.first.second.first;
      int y = swboxCfg.first.second.second;
      auto connects = swboxCfg.second;
      HerdOp herd = dyn_cast<HerdOp>(herdOp);

      builder.setInsertionPoint(device.getBody()->getTerminator());

      IterOp iterx =
          builder.create<IterOp>(builder.getUnknownLoc(), x, x + 1, 1);
      IterOp itery =
          builder.create<IterOp>(builder.getUnknownLoc(), y, y + 1, 1);
      AIEX::SelectOp sel = builder.create<AIEX::SelectOp>(
          builder.getUnknownLoc(), herd, iterx, itery);
      SwitchboxOp swbox =
          builder.create<SwitchboxOp>(builder.getUnknownLoc(), sel);
      swbox.ensureTerminator(swbox.getConnections(), builder,
                             builder.getUnknownLoc());
      Block &b = swbox.getConnections().front();
      builder.setInsertionPoint(b.getTerminator());

      for (auto connect : connects) {
        Port sourcePort = connect.first;
        Port destPort = connect.second;
        WireBundle sourceBundle = sourcePort.first;
        int sourceChannel = sourcePort.second;
        WireBundle destBundle = destPort.first;
        int destChannel = destPort.second;

        builder.create<ConnectOp>(builder.getUnknownLoc(), sourceBundle,
                                  sourceChannel, destBundle, destChannel);
      }
    }

    ConversionTarget target(getContext());

    RewritePatternSet patterns(&getContext());
    patterns.insert<AIEOpRemoval<PlaceOp>, AIEOpRemoval<RouteOp>>(
        device.getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIEHerdRoutingPass() {
  return std::make_unique<AIEHerdRoutingPass>();
}
