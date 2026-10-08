//===- AIESplitFlowVias.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIESPLITFLOWVIAS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// True when (srcTile, srcBundle:srcChannel) and (dstTile, dstBundle:dstChannel)
// are the two ends of a single physical inter-switchbox wire. Such a segment is
// realized by the wire itself and must not become a routable flow: doing so
// would double-drive the port that the adjacent local via flow produces.
static bool isDirectWire(Value srcTile, WireBundle srcBundle, int srcChannel,
                         Value dstTile, WireBundle dstBundle, int dstChannel) {
  if (srcChannel != dstChannel) {
    return false;
  }
  auto src = srcTile.getDefiningOp<TileOp>();
  auto dst = dstTile.getDefiningOp<TileOp>();
  if (!src || !dst) {
    return false;
  }
  int sc = src.colIndex(), sr = src.rowIndex();
  int dc = dst.colIndex(), dr = dst.rowIndex();
  if (sc == dc) {
    if (srcBundle == WireBundle::North && dstBundle == WireBundle::South &&
        dr == sr + 1) {
      return true;
    }
    if (srcBundle == WireBundle::South && dstBundle == WireBundle::North &&
        dr == sr - 1) {
      return true;
    }
  }
  if (sr == dr) {
    if (srcBundle == WireBundle::East && dstBundle == WireBundle::West &&
        dc == sc + 1) {
      return true;
    }
    if (srcBundle == WireBundle::West && dstBundle == WireBundle::East &&
        dc == sc - 1) {
      return true;
    }
  }
  return false;
}

static bool isDirectShimMux(Value srcTile, WireBundle srcBundle, int srcChannel,
                            Value dstTile, WireBundle dstBundle,
                            int dstChannel) {
  auto tile = srcTile.getDefiningOp<TileOp>();
  if (srcTile != dstTile || !tile || !tile.isShimNOCorPLTile())
    return false;
  if (dstBundle == WireBundle::South)
    return dstChannel == shimMuxChannelFrom({srcBundle, srcChannel});
  if (srcBundle == WireBundle::South)
    return srcChannel == shimMuxChannelTo({dstBundle, dstChannel});
  return false;
}

static void emitPacketFlow(OpBuilder &builder, Location loc, Operation *anchor,
                           Value srcTile, WireBundle srcBundle, int srcChannel,
                           Value dstTile, WireBundle dstBundle, int dstChannel,
                           int packetID, IntegerAttr maskAttr,
                           BoolAttr keepPktHeaderAttr,
                           BoolAttr priorityRouteAttr) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(anchor);
  auto pf = PacketFlowOp::create(builder, loc, static_cast<uint8_t>(packetID),
                                 keepPktHeaderAttr, priorityRouteAttr);
  if (maskAttr)
    pf.setMaskAttr(maskAttr);
  PacketFlowOp::ensureTerminator(pf.getPorts(), builder, loc);
  builder.setInsertionPoint(pf.getPorts().front().getTerminator());
  PacketSourceOp::create(builder, loc, srcTile, srcBundle, srcChannel);
  PacketDestOp::create(builder, loc, dstTile, dstBundle, dstChannel);
}

// Emit the routable portion of a segment between two pinned ports, eliding the
// parts the wires and vias already realize: a degenerate same-port segment or
// a single inter-switchbox wire. Whatever remains is a gap the router must fill
// and becomes a flow -- circuit when `packetID` < 0, otherwise a
// packet_flow(packetID).
static void emitRoutableSegment(OpBuilder &builder, Location loc,
                                Operation *anchor, Value srcTile,
                                WireBundle srcBundle, int srcChannel,
                                Value dstTile, WireBundle dstBundle,
                                int dstChannel, int packetID,
                                mlir::IntegerAttr maskAttr = {},
                                mlir::BoolAttr keepPktHeaderAttr = {},
                                mlir::BoolAttr priorityRouteAttr = {}) {
  if (srcTile == dstTile && srcBundle == dstBundle &&
      srcChannel == dstChannel) {
    return;
  }
  if (isDirectWire(srcTile, srcBundle, srcChannel, dstTile, dstBundle,
                   dstChannel)) {
    return;
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(anchor);
  if (packetID < 0) {
    FlowOp::create(builder, loc, srcTile, srcBundle, srcChannel, dstTile,
                   dstBundle, dstChannel);
    return;
  }
  // Preserve the mask across every section so the router derives the same
  // packet claim for the gaps and local via flows.
  emitPacketFlow(builder, loc, anchor, srcTile, srcBundle, srcChannel, dstTile,
                 dstBundle, dstChannel, packetID, maskAttr, keepPktHeaderAttr,
                 priorityRouteAttr);
}

struct AIESplitFlowViasPass
    : public xilinx::AIE::impl::AIESplitFlowViasBase<AIESplitFlowViasPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device.getContext());

    SmallVector<FlowOp> flowsWithVias;
    for (auto flow : device.getOps<FlowOp>()) {
      if (!flow.getVias().empty()) {
        flowsWithVias.push_back(flow);
      }
    }

    for (FlowOp flow : flowsWithVias) {
      ArrayRef<int32_t> ingressBundles =
          flow.getViaIngressBundlesAttr().asArrayRef();
      ArrayRef<int32_t> ingressChannels =
          flow.getViaIngressChannelsAttr().asArrayRef();
      ArrayRef<int32_t> egressBundles =
          flow.getViaEgressBundlesAttr().asArrayRef();
      ArrayRef<int32_t> egressChannels =
          flow.getViaEgressChannelsAttr().asArrayRef();
      Location loc = flow.getLoc();

      // Running source of the next segment, starting at the flow's source.
      Value srcTile = flow.getSource();
      WireBundle srcBundle = flow.getSourceBundle();
      int srcChannel = flow.getSourceChannel();

      auto emitSegment = [&](Value dstTile, WireBundle dstBundle,
                             int dstChannel) {
        emitRoutableSegment(builder, loc, flow, srcTile, srcBundle, srcChannel,
                            dstTile, dstBundle, dstChannel, /*packetID=*/-1);
      };

      for (size_t i = 0, e = flow.getVias().size(); i < e; i++) {
        Value viaTile = flow.getVias()[i];
        auto ingressBundle = static_cast<WireBundle>(ingressBundles[i]);
        int ingressChannel = ingressChannels[i];
        auto egressBundle = static_cast<WireBundle>(egressBundles[i]);
        int egressChannel = egressChannels[i];

        bool foldIngress =
            isDirectShimMux(srcTile, srcBundle, srcChannel, viaTile,
                            ingressBundle, ingressChannel);
        if (!foldIngress)
          emitSegment(viaTile, ingressBundle, ingressChannel);

        Value localSrcTile = foldIngress ? srcTile : viaTile;
        WireBundle localSrcBundle = foldIngress ? srcBundle : ingressBundle;
        int localSrcChannel = foldIngress ? srcChannel : ingressChannel;

        Value nextTile = i + 1 < e ? flow.getVias()[i + 1] : flow.getDest();
        WireBundle nextBundle =
            i + 1 < e ? static_cast<WireBundle>(ingressBundles[i + 1])
                      : flow.getDestBundle();
        int nextChannel =
            i + 1 < e ? ingressChannels[i + 1] : flow.getDestChannel();
        bool foldEgress = isDirectShimMux(viaTile, egressBundle, egressChannel,
                                          nextTile, nextBundle, nextChannel);

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(flow);
        FlowOp::create(builder, loc, localSrcTile, localSrcBundle,
                       localSrcChannel, foldEgress ? nextTile : viaTile,
                       foldEgress ? nextBundle : egressBundle,
                       foldEgress ? nextChannel : egressChannel);

        srcTile = foldEgress ? nextTile : viaTile;
        srcBundle = foldEgress ? nextBundle : egressBundle;
        srcChannel = foldEgress ? nextChannel : egressChannel;
      }

      emitSegment(flow.getDest(), flow.getDestBundle(), flow.getDestChannel());
      flow.erase();
    }

    // Packet sections pin their route with vias too. Each via becomes a local
    // packet flow so the router can plan its rules, master sets, and arbiters
    // with the other packet flows.
    SmallVector<PacketFlowOp> pktFlowsWithVias;
    for (auto pf : device.getOps<PacketFlowOp>()) {
      if (!pf.getVias().empty()) {
        pktFlowsWithVias.push_back(pf);
      }
    }

    for (PacketFlowOp pf : pktFlowsWithVias) {
      Location loc = pf.getLoc();
      int id = pf.IDInt();
      ArrayRef<int32_t> inB = pf.getViaIngressBundlesAttr().asArrayRef();
      ArrayRef<int32_t> inC = pf.getViaIngressChannelsAttr().asArrayRef();
      ArrayRef<int32_t> egB = pf.getViaEgressBundlesAttr().asArrayRef();
      ArrayRef<int32_t> egC = pf.getViaEgressChannelsAttr().asArrayRef();

      // A section has exactly one source and one dest; the gaps between the
      // pinned vias (and before the first / after the last) are routed.
      PacketSourceOp source;
      PacketDestOp dest;
      for (Operation &op : pf.getPorts().front()) {
        if (auto s = dyn_cast<PacketSourceOp>(op)) {
          source = s;
        } else if (auto d = dyn_cast<PacketDestOp>(op)) {
          dest = d;
        }
      }
      Value srcTile = source.getTile();
      WireBundle srcBundle = source.getBundle();
      int srcChannel = source.getChannel();
      auto emitSegment = [&](Value dstTile, WireBundle dstBundle,
                             int dstChannel, BoolAttr keepPktHeader = {}) {
        emitRoutableSegment(builder, loc, pf, srcTile, srcBundle, srcChannel,
                            dstTile, dstBundle, dstChannel, id,
                            pf.getMaskAttr(), keepPktHeader,
                            pf.getPriorityRouteAttr());
      };

      for (size_t i = 0, e = pf.getVias().size(); i < e; i++) {
        Value viaTile = pf.getVias()[i];
        auto ingress = static_cast<WireBundle>(inB[i]);
        int ingressChannel = inC[i];
        auto egress = static_cast<WireBundle>(egB[i]);
        int egressChannel = egC[i];

        bool foldIngress = isDirectShimMux(srcTile, srcBundle, srcChannel,
                                           viaTile, ingress, ingressChannel);
        if (!foldIngress)
          emitSegment(viaTile, ingress, ingressChannel);

        Value localSrcTile = foldIngress ? srcTile : viaTile;
        WireBundle localSrcBundle = foldIngress ? srcBundle : ingress;
        int localSrcChannel = foldIngress ? srcChannel : ingressChannel;

        Value nextTile = i + 1 < e ? pf.getVias()[i + 1] : dest.getTile();
        WireBundle nextBundle =
            i + 1 < e ? static_cast<WireBundle>(inB[i + 1]) : dest.getBundle();
        int nextChannel = i + 1 < e ? inC[i + 1] : dest.getChannel();
        bool foldEgress = isDirectShimMux(viaTile, egress, egressChannel,
                                          nextTile, nextBundle, nextChannel);

        BoolAttr keepPktHeader;
        if ((foldEgress && nextTile == dest.getTile() &&
             nextBundle == dest.getBundle() &&
             nextChannel == dest.getChannel()) ||
            (viaTile == dest.getTile() && egress == dest.getBundle() &&
             egressChannel == dest.getChannel()))
          keepPktHeader = pf.getKeepPktHeaderAttr();
        emitPacketFlow(
            builder, loc, pf, localSrcTile, localSrcBundle, localSrcChannel,
            foldEgress ? nextTile : viaTile, foldEgress ? nextBundle : egress,
            foldEgress ? nextChannel : egressChannel, id, pf.getMaskAttr(),
            keepPktHeader, pf.getPriorityRouteAttr());

        srcTile = foldEgress ? nextTile : viaTile;
        srcBundle = foldEgress ? nextBundle : egress;
        srcChannel = foldEgress ? nextChannel : egressChannel;
      }

      emitSegment(dest.getTile(), dest.getBundle(), dest.getChannel(),
                  pf.getKeepPktHeaderAttr());
      pf.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIESplitFlowViasPass() {
  return std::make_unique<AIESplitFlowViasPass>();
}
