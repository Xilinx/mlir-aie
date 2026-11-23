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
  auto finish = [&] {
    if (!current.packets.empty()) {
      intervals.push_back(current);
      current.packets.clear();
    }
  };
  for (Operation &operation : block) {
    if (auto packet = dyn_cast<NpuControlPacketOp>(operation)) {
      bool ordinary = packet.isOrdinaryWrite();
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
    if (auto context = dyn_cast<SetDeviceOp>(operation))
      current.device = context.getDeviceRefAttr();
    else if (isa<NpuLoadPdiOp>(operation))
      current.device = {};
  }
  finish();
  return intervals;
}

ArrayAttr AIEX::controlPacketMulticasts(ModuleOp module,
                                        FlatSymbolRefAttr reference) {
  auto device = reference ? module.lookupSymbol<AIE::DeviceOp>(reference)
                          : AIE::DeviceOp{};
  auto overlay =
      device && device.getOverlayAttr()
          ? module.lookupSymbol<AIE::DeviceOp>(device.getOverlayAttr())
          : AIE::DeviceOp{};
  return overlay
             ? AIE::AIEDialect::CtrlPktMulticastAttrHelper(module.getContext())
                   .getAttr(overlay)
             : ArrayAttr{};
}