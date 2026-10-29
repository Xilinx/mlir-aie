//===- AIEPacketIdSpace.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPacketIdSpace.h"

using namespace xilinx;
using namespace xilinx::AIE;

PacketIdSpace::PacketIdSpace(DeviceOp device)
    : max(static_cast<int>(device.getTargetModel().getMaxPacketId())) {
  device.walk([&](PacketFlowOp flow) { taken.insert(flow.IDInt()); });
  device.walk([&](TracePacketOp packet) {
    if (std::optional<int32_t> id = packet.getId()) {
      taken.insert(*id);
    }
  });
  auto pinned = [&](PacketInfoAttr header) {
    if (header && header.isAssigned()) {
      taken.insert(header.assignedId());
    }
  };
  for (auto route : device.getOps<RouteOp>()) {
    pinned(route.getPacketAttr());
  }
  for (auto fifo : device.getOps<ObjectFifoCreateOp>()) {
    pinned(fifo.packetHeader());
  }
}

std::optional<int> PacketIdSpace::takeLowestFrom(int from) {
  for (int id = from; id <= max; id++) {
    if (taken.insert(id).second) {
      return id;
    }
  }
  return std::nullopt;
}
