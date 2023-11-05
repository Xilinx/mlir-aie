//===- AIEHerdRouting.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Attributes.h"
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

  explicit AIEOpRemoval(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<MyOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(MyOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();

    rewriter.eraseOp(Op);
    return success();
  }
};

std::optional<uint32_t>
getAvailableDestChannel(SmallVector<Connect, 8> &connects, Port sourcePort,
                        WireBundle destBundle) {

  if (connects.empty())
    return {0};

  uint32_t numChannels;

  if (destBundle == WireBundle::North)
    numChannels = 6;
  else if (destBundle == WireBundle::South || destBundle == WireBundle::East ||
           destBundle == WireBundle::West)
    numChannels = 4;
  else
    numChannels = 2;

  // look for existing connect
  for (uint32_t i = 0; i < numChannels; i++) {
    Port port = {destBundle, i};
    if (std::find(connects.begin(), connects.end(),
                  Connect{sourcePort, port}) != connects.end())
      return {i};
  }

  // if not, look for available destination port
  for (uint32_t i = 0; i < numChannels; i++) {
    Port port = {destBundle, i};
    SmallVector<Port, 8> ports;
    for (auto connect : connects)
      ports.push_back(connect.dst);

    if (std::find(ports.begin(), ports.end(), port) == ports.end())
      return {i};
  }

  return std::nullopt;
}

void buildRoute(uint32_t xSrc, uint32_t ySrc, uint32_t xDest, uint32_t yDest,
                WireBundle sourceBundle, uint32_t sourceChannel,
                WireBundle destBundle, uint32_t destChannel, Operation *herdOp,
                DenseMap<std::pair<Operation *, TileID>,
                         SmallVector<Connect, 8>> &switchboxes) {

  uint32_t xCur = xSrc;
  uint32_t yCur = ySrc;
  WireBundle curBundle;
  uint32_t curChannel;
  uint32_t xLast, yLast;
  WireBundle lastBundle;
  Port lastPort = {sourceBundle, sourceChannel};

  SmallVector<TileID, 4> congestion;

  llvm::dbgs() << "Build route: " << xSrc << " " << ySrc << " --> " << xDest
               << " " << yDest << '\n';
  // traverse horizontally, then vertically
  while (!((xCur == xDest) && (yCur == yDest))) {
    llvm::dbgs() << "coord " << xCur << " " << yCur << '\n';

    TileID curCoord = {xCur, yCur};
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

    for (auto move : moves) {
      if (auto maybeDestChannel = getAvailableDestChannel(
              switchboxes[std::make_pair(herdOp, curCoord)], lastPort, move))
        curChannel = maybeDestChannel.value();
      else
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

      if (std::find(congestion.begin(), congestion.end(), TileID{xCur, yCur}) !=
          congestion.end())
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
      congestion.push_back({xLast, yLast}); // this switchbox is congested
      switchboxes[std::make_pair(herdOp, curCoord)]
          .pop_back(); // back up, remove the last connection
    } else {
      llvm::dbgs() << "[" << stringifyWireBundle(lastPort.bundle) << " : "
                   << lastPort.channel
                   << "], "
                      "["
                   << stringifyWireBundle(curBundle) << " : " << curChannel
                   << "]\n";

      Port curPort = {curBundle, curChannel};
      Connect connect = {lastPort, curPort};
      if (std::find(switchboxes[std::make_pair(herdOp, curCoord)].begin(),
                    switchboxes[std::make_pair(herdOp, curCoord)].end(),
                    connect) ==
          switchboxes[std::make_pair(herdOp, curCoord)].end())
        switchboxes[std::make_pair(herdOp, curCoord)].push_back(connect);
      lastPort = {lastBundle, curChannel};
    }
  }

  llvm::dbgs() << "coord " << xCur << " " << yCur << '\n';
  llvm::dbgs() << "[" << stringifyWireBundle(lastPort.bundle) << " : "
               << lastPort.channel
               << "], "
                  "["
               << stringifyWireBundle(destBundle) << " : " << destChannel
               << "]\n";

  switchboxes[std::make_pair(herdOp, TileID{xCur, yCur})].push_back(
      {lastPort, Port{destBundle, destChannel}});
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
    SmallVector<
        std::pair<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint32_t>>,
        4>
        routes;
    DenseMap<std::pair<Operation *, TileID>, SmallVector<Connect, 8>>
        switchboxes;

    for (auto herd : device.getOps<HerdOp>()) {
      herds.push_back(herd);
    }

    for (auto placeOp : device.getOps<PlaceOp>()) {
      placeOps.push_back(placeOp);
      Operation *sourceHerd = placeOp.getSourceHerd().getDefiningOp();
      Operation *destHerd = placeOp.getDestHerd().getDefiningOp();
      uint32_t distX = placeOp.getDistXValue();
      uint32_t distY = placeOp.getDistYValue();
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
      uint32_t sourceChannel = routeOp.getSourceChannelValue();
      uint32_t destChannel = routeOp.getDestChannelValue();

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

      uint32_t sourceStartX = sourceIterX.getStartValue();
      uint32_t sourceEndX = sourceIterX.getEndValue();
      uint32_t sourceStrideX = sourceIterX.getStrideValue();
      uint32_t sourceStartY = sourceIterY.getStartValue();
      uint32_t sourceEndY = sourceIterY.getEndValue();
      uint32_t sourceStrideY = sourceIterY.getStrideValue();

      uint32_t destStartX = destIterX.getStartValue();
      uint32_t destEndX = destIterX.getEndValue();
      uint32_t destStrideX = destIterX.getStrideValue();
      uint32_t destStartY = destIterY.getStartValue();
      uint32_t destEndY = destIterY.getEndValue();
      uint32_t destStrideY = destIterY.getStrideValue();

      assert(distances.count(std::make_pair(sourceHerd, destHerd)) == 1);

      std::pair<int, int> distance =
          distances[std::make_pair(sourceHerd, destHerd)];
      int distX = distance.first;
      int distY = distance.second;
      // FIXME: this looks like it can be improved further ...
      for (uint32_t xSrc = sourceStartX; xSrc < sourceEndX;
           xSrc += sourceStrideX) {
        for (uint32_t ySrc = sourceStartY; ySrc < sourceEndY;
             ySrc += sourceStrideY) {
          for (uint32_t xDst = destStartX; xDst < destEndX;
               xDst += destStrideX) {
            for (uint32_t yDst = destStartY; yDst < destEndY;
                 yDst += destStrideY) {
              // Build route (x0, y0) --> (x1, y1)
              uint32_t x0 = xSrc;
              uint32_t y0 = ySrc;
              uint32_t x1 = xDst;
              uint32_t y1 = yDst;
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

    for (const auto &swboxCfg : switchboxes) {
      Operation *herdOp = swboxCfg.first.first;
      uint32_t x = swboxCfg.first.second.col;
      uint32_t y = swboxCfg.first.second.row;
      auto connects = swboxCfg.second;
      HerdOp herd = dyn_cast<HerdOp>(herdOp);

      builder.setInsertionPoint(device.getBody()->getTerminator());

      auto iterx = builder.create<IterOp>(builder.getUnknownLoc(), x, x + 1, 1);
      auto itery = builder.create<IterOp>(builder.getUnknownLoc(), y, y + 1, 1);
      auto sel = builder.create<AIEX::SelectOp>(builder.getUnknownLoc(), herd,
                                                iterx, itery);
      auto swbox = builder.create<SwitchboxOp>(builder.getUnknownLoc(), sel);
      SwitchboxOp::ensureTerminator(swbox.getConnections(), builder,
                                    builder.getUnknownLoc());
      Block &b = swbox.getConnections().front();
      builder.setInsertionPoint(b.getTerminator());

      for (auto connect : connects) {
        Port sourcePort = connect.src;
        Port destPort = connect.dst;
        WireBundle sourceBundle = sourcePort.bundle;
        uint32_t sourceChannel = sourcePort.channel;
        WireBundle destBundle = destPort.bundle;
        uint32_t destChannel = destPort.channel;

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
