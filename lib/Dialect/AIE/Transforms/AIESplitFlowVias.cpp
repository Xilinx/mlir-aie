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

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIESPLITFLOWVIAS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

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

struct AIESplitFlowViasPass
    : public xilinx::AIE::impl::AIESplitFlowViasBase<AIESplitFlowViasPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder builder(device.getContext());

    DenseMap<Value, SwitchboxOp> switchboxes;
    for (auto switchboxOp : device.getOps<SwitchboxOp>())
      switchboxes[switchboxOp.getTile()] = switchboxOp;

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
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIESplitFlowViasPass() {
  return std::make_unique<AIESplitFlowViasPass>();
}
