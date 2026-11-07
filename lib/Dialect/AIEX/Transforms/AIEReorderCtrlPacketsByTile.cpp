//===- AIEReorderCtrlPacketsByTile.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "CtrlPacketClassifier.h"
#include "CtrlPacketDevices.h"
#include "CtrlPacketRuns.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SetVector.h"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEREORDERCTRLPACKETSBYTILE
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIEReorderCtrlPacketsByTilePass
    : xilinx::AIEX::impl::AIEReorderCtrlPacketsByTileBase<
          AIEReorderCtrlPacketsByTilePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ConfiguredDevices devices(module);
    for (auto device : module.getOps<AIE::DeviceOp>()) {
      PacketClassifier classifier(device.getTargetModel());
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>())
        for (GroupRun &run : groupRuns(seq))
          if (failed(reorder(run, classifier, devices)))
            return signalPassFailure();
    }
  }

  // Puts the run's packets in the order the pass description gives.
  LogicalResult reorder(GroupRun &run, PacketClassifier &classifier,
                        ConfiguredDevices &devices) {
    using Key = std::tuple<WriteKind, uint32_t, uint32_t>;
    SmallVector<std::pair<Key, GroupedPacket>> keyed;
    for (GroupedPacket entry : run) {
      NpuControlPacketOp packet = entry.packet;
      FailureOr<Classified> c = classifier.classify(packet);
      if (failed(c))
        return failure();
      AIE::TileID tile = packet.getTileFromAddr();
      WriteKind part = c->kind;
      // With gated-enable, a proved core's enable goes last among its
      // column's other writes.
      uint32_t row = tile.row;
      FlatSymbolRefAttr device = entry.group.getDeviceRefAttr();
      if (part == WriteKind::ChannelStart &&
          devices.startGated(device, {tile, c->channel})) {
        part = WriteKind::Other;
      } else if (part == WriteKind::CoreEnable && gatedEnable &&
                 devices.startGated(device, {tile, -1})) {
        part = WriteKind::Other;
        row = std::numeric_limits<uint32_t>::max();
      }
      Key key{part, 0, 0};
      if (byTile)
        key = {part, static_cast<uint32_t>(tile.col), row};
      keyed.push_back({key, entry});
    }
    std::stable_sort(
        keyed.begin(), keyed.end(),
        [](const auto &a, const auto &b) { return a.first < b.first; });

    // The packets go back after the run, each in a group like its old one; a
    // barrier puts the enable phase after the rest.
    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(run.back().group);
    llvm::SetVector<Operation *> oldGroups;
    for (GroupedPacket entry : run)
      oldGroups.insert(entry.group);
    ControlPacketGroupOp current;
    bool placed = false, barrier = false;
    for (auto &[key, entry] : keyed) {
      if (std::get<0>(key) != WriteKind::Other && placed && !barrier) {
        ControlPacketBarrierOp::create(builder, entry.packet.getLoc());
        current = nullptr;
        barrier = true;
      }
      placed = true;
      if (!current) {
        current = ControlPacketGroupOp::create(builder, entry.group.getLoc(),
                                               entry.group.getKindAttr(),
                                               entry.group.getDeviceRefAttr());
        current.getBody().emplaceBlock();
      }
      Block &body = current.getBody().front();
      entry.packet->moveBefore(&body, body.end());
    }
    for (Operation *group : oldGroups)
      group->erase();
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEReorderCtrlPacketsByTilePass() {
  return std::make_unique<AIEReorderCtrlPacketsByTilePass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEReorderCtrlPacketsByTilePass(
    const AIEReorderCtrlPacketsByTileOptions &options) {
  return std::make_unique<AIEReorderCtrlPacketsByTilePass>(options);
}
