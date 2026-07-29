//===- AIESplitFlowVias.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

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
// would double-drive the port that the adjacent via's fixed connection already
// produces.
static bool isDirectWire(Value srcTile, WireBundle srcBundle, int srcChannel,
                         Value dstTile, WireBundle dstBundle, int dstChannel) {
  if (srcChannel != dstChannel)
    return false;
  auto src = srcTile.getDefiningOp<TileOp>();
  auto dst = dstTile.getDefiningOp<TileOp>();
  if (!src || !dst)
    return false;
  int sc = src.colIndex(), sr = src.rowIndex();
  int dc = dst.colIndex(), dr = dst.rowIndex();
  if (sc == dc) {
    if (srcBundle == WireBundle::North && dstBundle == WireBundle::South &&
        dr == sr + 1)
      return true;
    if (srcBundle == WireBundle::South && dstBundle == WireBundle::North &&
        dr == sr - 1)
      return true;
  }
  if (sr == dr) {
    if (srcBundle == WireBundle::East && dstBundle == WireBundle::West &&
        dc == sc + 1)
      return true;
    if (srcBundle == WireBundle::West && dstBundle == WireBundle::East &&
        dc == sc - 1)
      return true;
  }
  return false;
}

// Return the switchbox for `tile`, reusing an existing one or creating an empty
// one immediately after the tile so its operand dominates the connections.
static SwitchboxOp getOrCreateSwitchbox(OpBuilder &builder, Value tile,
                                        DenseMap<Value, SwitchboxOp> &cache) {
  if (auto it = cache.find(tile); it != cache.end())
    return it->second;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(tile);
  auto switchboxOp = SwitchboxOp::create(builder, tile.getLoc(), tile);
  SwitchboxOp::ensureTerminator(switchboxOp.getConnections(), builder,
                                tile.getLoc());
  cache[tile] = switchboxOp;
  return switchboxOp;
}

// Return the shim-mux for `tile`, reusing an existing one or creating an empty
// one immediately after the tile so its operand dominates the connections.
static ShimMuxOp getOrCreateShimMux(OpBuilder &builder, Value tile,
                                    DenseMap<Value, ShimMuxOp> &cache) {
  if (auto it = cache.find(tile); it != cache.end())
    return it->second;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(tile);
  auto shimMuxOp = ShimMuxOp::create(builder, tile.getLoc(), tile);
  ShimMuxOp::ensureTerminator(shimMuxOp.getConnections(), builder,
                              tile.getLoc());
  cache[tile] = shimMuxOp;
  return shimMuxOp;
}

// Allocate a fresh (arbiter, msel) amsel in `sb` that is not already used by its
// configuration, creating the amsel at the top of the switchbox block so it
// dominates the master sets and rules that reference it.
static AMSelOp allocateAmsel(OpBuilder &builder, Location loc, SwitchboxOp sb) {
  Block &block = sb.getConnections().front();
  llvm::SmallSet<std::pair<int, int>, 24> used;
  for (auto a : block.getOps<AMSelOp>())
    used.insert({a.arbiterIndex(), a.getMselValue()});
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&block);
  for (int msel = 0; msel < 4; msel++)
    for (int arb = 0; arb < 6; arb++)
      if (!used.count({arb, msel}))
        return AMSelOp::create(builder, loc, arb, msel);
  return nullptr;
}

// Return the packet_rules for `ingress` in `sb`, creating an empty one if none
// exists, so several rules for the same ingress port accumulate in one op.
static PacketRulesOp getOrCreatePacketRules(OpBuilder &builder, Location loc,
                                            SwitchboxOp sb, WireBundle ingress,
                                            int ingressChannel) {
  Block &block = sb.getConnections().front();
  for (auto rules : block.getOps<PacketRulesOp>())
    if (rules.getSourceBundle() == ingress &&
        rules.getSourceChannel() == ingressChannel)
      return rules;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(block.getTerminator());
  auto rules = PacketRulesOp::create(builder, loc, ingress, ingressChannel);
  PacketRulesOp::ensureTerminator(rules.getRules(), builder, loc);
  return rules;
}

// Emit the routable portion of a segment between two pinned ports, eliding the
// parts the wires and vias already realize: a degenerate same-port segment, a
// single inter-switchbox wire, or an intra-shim hop (realized through the
// shim-mux, since the switchbox South:N port appears as mux North:N). Whatever
// remains is a gap the router must fill and becomes a flow -- circuit when
// `packetID` < 0, otherwise a packet_flow(packetID).
static void emitRoutableSegment(OpBuilder &builder, Location loc,
                                Operation *anchor, Value srcTile,
                                WireBundle srcBundle, int srcChannel,
                                Value dstTile, WireBundle dstBundle,
                                int dstChannel, int packetID,
                                DenseMap<Value, ShimMuxOp> &shimMuxes,
                                mlir::IntegerAttr maskAttr = {}) {
  if (srcTile == dstTile && srcBundle == dstBundle && srcChannel == dstChannel)
    return;
  if (isDirectWire(srcTile, srcBundle, srcChannel, dstTile, dstBundle,
                   dstChannel))
    return;
  if (srcTile == dstTile) {
    if (auto tileOp = srcTile.getDefiningOp<TileOp>();
        tileOp && tileOp.isShimNOCorPLTile()) {
      auto toMux = [](WireBundle b) {
        return b == WireBundle::South ? WireBundle::North : b;
      };
      ShimMuxOp mux = getOrCreateShimMux(builder, srcTile, shimMuxes);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(mux.getConnections().front().getTerminator());
      ConnectOp::create(builder, loc, toMux(srcBundle), srcChannel,
                        toMux(dstBundle), dstChannel);
      return;
    }
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(anchor);
  if (packetID < 0) {
    FlowOp::create(builder, loc, srcTile, srcBundle, srcChannel, dstTile,
                   dstBundle, dstChannel);
    return;
  }
  auto pf = PacketFlowOp::create(builder, loc,
                                 static_cast<uint8_t>(packetID), BoolAttr(),
                                 BoolAttr());
  // Preserve the flow's pinned packet-rule mask on the routed remainder; without
  // it the pathfinder derives a full 0x1f mask that mismatches the pinned via
  // segments' rules for the same flow and the packet is never accepted.
  if (maskAttr)
    pf.setMaskAttr(maskAttr);
  PacketFlowOp::ensureTerminator(pf.getPorts(), builder, loc);
  builder.setInsertionPoint(pf.getPorts().front().getTerminator());
  PacketSourceOp::create(builder, loc, srcTile, srcBundle, srcChannel);
  PacketDestOp::create(builder, loc, dstTile, dstBundle, dstChannel);
}

struct AIESplitFlowViasPass
    : public xilinx::AIE::impl::AIESplitFlowViasBase<AIESplitFlowViasPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device.getContext());

    DenseMap<Value, SwitchboxOp> switchboxes;
    for (auto switchboxOp : device.getOps<SwitchboxOp>())
      switchboxes[switchboxOp.getTile()] = switchboxOp;
    DenseMap<Value, ShimMuxOp> shimMuxes;
    for (auto shimMuxOp : device.getOps<ShimMuxOp>())
      shimMuxes[shimMuxOp.getTile()] = shimMuxOp;

    SmallVector<FlowOp> flowsWithVias;
    for (auto flow : device.getOps<FlowOp>())
      if (!flow.getVias().empty())
        flowsWithVias.push_back(flow);

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
                            dstTile, dstBundle, dstChannel, /*packetID=*/-1,
                            shimMuxes);
      };

      for (size_t i = 0, e = flow.getVias().size(); i < e; i++) {
        Value viaTile = flow.getVias()[i];
        auto ingressBundle = static_cast<WireBundle>(ingressBundles[i]);
        int ingressChannel = ingressChannels[i];
        auto egressBundle = static_cast<WireBundle>(egressBundles[i]);
        int egressChannel = egressChannels[i];

        emitSegment(viaTile, ingressBundle, ingressChannel);

        SwitchboxOp switchboxOp = getOrCreateSwitchbox(builder, viaTile,
                                                       switchboxes);
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(
              switchboxOp.getConnections().front().getTerminator());
          ConnectOp::create(builder, loc, ingressBundle, ingressChannel,
                            egressBundle, egressChannel);
        }

        srcTile = viaTile;
        srcBundle = egressBundle;
        srcChannel = egressChannel;
      }

      emitSegment(flow.getDest(), flow.getDestBundle(), flow.getDestChannel());
      flow.erase();
    }

    // Packet sections pin their route with vias too. Each via becomes a
    // stream-switch rule/master-set pair carrying the flow's ID; the amsel is
    // allocated locally (its numeric value is not pinned). Fan-out/fan-in nodes
    // are already materialized in the IR, so a section only needs its own hops.
    SmallVector<PacketFlowOp> pktFlowsWithVias;
    for (auto pf : device.getOps<PacketFlowOp>())
      if (!pf.getVias().empty())
        pktFlowsWithVias.push_back(pf);

    for (PacketFlowOp pf : pktFlowsWithVias) {
      Location loc = pf.getLoc();
      int id = pf.IDInt();
      int mask = pf.getMask() ? static_cast<int>(*pf.getMask()) : 0x1f;
      ArrayRef<int32_t> inB = pf.getViaIngressBundlesAttr().asArrayRef();
      ArrayRef<int32_t> inC = pf.getViaIngressChannelsAttr().asArrayRef();
      ArrayRef<int32_t> egB = pf.getViaEgressBundlesAttr().asArrayRef();
      ArrayRef<int32_t> egC = pf.getViaEgressChannelsAttr().asArrayRef();

      // A section has exactly one source and one dest; the gaps between the
      // pinned vias (and before the first / after the last) are routed.
      PacketSourceOp source;
      PacketDestOp dest;
      for (Operation &op : pf.getPorts().front()) {
        if (auto s = dyn_cast<PacketSourceOp>(op))
          source = s;
        else if (auto d = dyn_cast<PacketDestOp>(op))
          dest = d;
      }
      Value srcTile = source.getTile();
      WireBundle srcBundle = source.getBundle();
      int srcChannel = source.getChannel();
      auto emitSegment = [&](Value dstTile, WireBundle dstBundle,
                             int dstChannel) {
        emitRoutableSegment(builder, loc, pf, srcTile, srcBundle, srcChannel,
                            dstTile, dstBundle, dstChannel, id, shimMuxes,
                            pf.getMaskAttr());
      };

      for (size_t i = 0, e = pf.getVias().size(); i < e; i++) {
        Value viaTile = pf.getVias()[i];
        auto ingress = static_cast<WireBundle>(inB[i]);
        int ingressChannel = inC[i];
        auto egress = static_cast<WireBundle>(egB[i]);
        int egressChannel = egC[i];

        emitSegment(viaTile, ingress, ingressChannel);

        SwitchboxOp sb = getOrCreateSwitchbox(builder, viaTile, switchboxes);
        AMSelOp amsel = allocateAmsel(builder, loc, sb);
        if (!amsel) {
          pf.emitOpError("via tile has no free arbiter-msel combination");
          signalPassFailure();
          return;
        }
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(sb.getConnections().front().getTerminator());
          // Restore keep_pkt_header on the master set that drives the flow's
          // destination; the intermediate hops always keep the header.
          BoolAttr keepPktHeader;
          if (viaTile == dest.getTile() && egress == dest.getBundle() &&
              egressChannel == dest.getChannel())
            keepPktHeader = pf.getKeepPktHeaderAttr();
          MasterSetOp::create(builder, loc, builder.getIndexType(), egress,
                              egressChannel, ValueRange{amsel}, keepPktHeader);
        }
        PacketRulesOp rules =
            getOrCreatePacketRules(builder, loc, sb, ingress, ingressChannel);
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(rules.getRules().front().getTerminator());
          PacketRuleOp::create(builder, loc, mask, id, amsel);
        }

        srcTile = viaTile;
        srcBundle = egress;
        srcChannel = egressChannel;
      }

      emitSegment(dest.getTile(), dest.getBundle(), dest.getChannel());
      pf.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIESplitFlowViasPass() {
  return std::make_unique<AIESplitFlowViasPass>();
}
