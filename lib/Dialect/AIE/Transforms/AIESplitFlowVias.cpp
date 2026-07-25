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
        // A flow that already begins or ends on a via port produces a segment
        // whose two ends are the same port; there is nothing to route.
        if (srcTile == dstTile && srcBundle == dstBundle &&
            srcChannel == dstChannel)
          return;
        // A segment that is a single inter-switchbox wire is realized by the
        // wire; emitting a flow for it would fight the via's fixed connection.
        if (isDirectWire(srcTile, srcBundle, srcChannel, dstTile, dstBundle,
                         dstChannel))
          return;
        // A shim tile reaches its switchbox through a shim-mux that the via
        // chain does not record. The head/tail of a shim flow is a same-tile
        // hop between a tile-side port (DMA/NOC/PLIO) and the mux-facing South
        // port, which the mux realizes: the switchbox South:N port appears as
        // North:N on the mux side.
        if (srcTile == dstTile) {
          if (auto tileOp = srcTile.getDefiningOp<TileOp>();
              tileOp && tileOp.isShimNOCorPLTile()) {
            auto toMux = [](WireBundle b) {
              return b == WireBundle::South ? WireBundle::North : b;
            };
            ShimMuxOp mux = getOrCreateShimMux(builder, srcTile, shimMuxes);
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(
                mux.getConnections().front().getTerminator());
            ConnectOp::create(builder, loc, toMux(srcBundle), srcChannel,
                              toMux(dstBundle), dstChannel);
            return;
          }
        }
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(flow);
        FlowOp::create(builder, loc, srcTile, srcBundle, srcChannel, dstTile,
                       dstBundle, dstChannel);
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
      int mask = pf.getViaMask() ? static_cast<int>(*pf.getViaMask()) : 0x1f;
      ArrayRef<int32_t> inB = pf.getViaIngressBundlesAttr().asArrayRef();
      ArrayRef<int32_t> inC = pf.getViaIngressChannelsAttr().asArrayRef();
      ArrayRef<int32_t> egB = pf.getViaEgressBundlesAttr().asArrayRef();
      ArrayRef<int32_t> egC = pf.getViaEgressChannelsAttr().asArrayRef();

      for (size_t i = 0, e = pf.getVias().size(); i < e; i++) {
        Value viaTile = pf.getVias()[i];
        auto ingress = static_cast<WireBundle>(inB[i]);
        int ingressChannel = inC[i];
        auto egress = static_cast<WireBundle>(egB[i]);
        int egressChannel = egC[i];

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
          MasterSetOp::create(builder, loc, builder.getIndexType(), egress,
                              egressChannel, ValueRange{amsel}, BoolAttr());
        }
        PacketRulesOp rules =
            getOrCreatePacketRules(builder, loc, sb, ingress, ingressChannel);
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(rules.getRules().front().getTerminator());
        PacketRuleOp::create(builder, loc, mask, id, amsel);
      }
      pf.erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIESplitFlowViasPass() {
  return std::make_unique<AIESplitFlowViasPass>();
}
