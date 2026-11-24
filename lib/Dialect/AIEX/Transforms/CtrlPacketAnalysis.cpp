//===- CtrlPacketAnalysis.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CtrlPacketRuns.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

SmallVector<PacketInterval> AIEX::packetIntervals(Block &block) {
  SmallVector<PacketInterval> intervals;
  PacketInterval current;
  ArrayAttr multicasts;
  auto finish = [&] {
    if (!current.packets.empty()) {
      intervals.push_back(current);
      current.packets.clear();
    }
  };
  for (Operation &operation : block) {
    if (auto packet = dyn_cast<NpuControlPacketOp>(operation)) {
      bool ordinary = packet.isOrdinaryWrite();
      if (auto id = packet.getMulticastId()) {
        auto group = multicasts && *id < multicasts.size()
                         ? dyn_cast<DictionaryAttr>(multicasts[*id])
                         : DictionaryAttr{};
        auto tiles = group ? group.getAs<ArrayAttr>("tiles") : ArrayAttr{};
        ordinary =
            ordinary && tiles && !tiles.empty() &&
            llvm::all_of(tiles, [&](Attribute attribute) {
              auto tile = dyn_cast<AIE::TileAttr>(attribute);
              return tile &&
                     llvm::all_of(
                         llvm::seq<size_t>(0, packet.getDataAttr().size()),
                         [&](size_t word) {
                           return AIE::getTargetModel(packet)
                               .isReorderableRegisterWrite(
                                   tile.getTileID(),
                                   packet.getLocalAddress() + 4 * word);
                         });
            });
      }
      if (ordinary != current.ordinary) {
        finish();
        current.ordinary = ordinary;
      }
      current.packets.push_back(packet);
      continue;
    }
    finish();
    auto barrier = dyn_cast<ControlPacketBarrierOp>(operation);
    current.ordinary = false;
    current.independent =
        barrier && (barrier.getIndependent() ||
                    (barrier.getCompilerOnly() && current.independent));
    if (!barrier || !barrier.getCompilerOnly())
      ++current.delivery;
    if (auto context = dyn_cast<SetDeviceOp>(operation)) {
      current.device = context.getDeviceRefAttr();
      auto module = context->getParentOfType<ModuleOp>();
      auto device = module.lookupSymbol<AIE::DeviceOp>(current.device);
      auto overlay =
          device && device.getOverlayAttr()
              ? module.lookupSymbol<AIE::DeviceOp>(device.getOverlayAttr())
              : AIE::DeviceOp{};
      multicasts =
          overlay
              ? AIE::AIEDialect::CtrlPktMulticastAttrHelper(module.getContext())
                    .getAttr(overlay)
              : ArrayAttr{};
    } else if (isa<NpuLoadPdiOp>(operation)) {
      current.device = {};
      multicasts = {};
    }
  }
  finish();
  return intervals;
}
