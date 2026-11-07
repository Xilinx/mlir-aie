//===- AIEPacketIdSpace.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIEPacketIdSpace.h"

using namespace xilinx;
using namespace xilinx::AIE;

PacketIdSpace::PacketIdSpace(DeviceOp device, bool reserveControllerIds)
    : max(static_cast<int>(device.getTargetModel().getMaxPacketId())) {
  if (reserveControllerIds ||
      AIEDialect::ReserveControllerPacketIdsAttrHelper(device.getContext())
          .isAttrPresent(device)) {
    for (auto [tile, id] :
         device.getTargetModel().getTileToControllerIdMap(true))
      taken.insert(id);
    for (auto tile : device.getOps<TileOp>())
      if (auto id = tile->getAttrOfType<PacketInfoAttr>("controller_id");
          id && id.isAssigned())
        taken.insert(id.assignedId());
  }
  device.walk([&](PacketFlowOp flow) { taken.insert(flow.IDInt()); });
  // An id a rule already matches would reach that rule's master too, at any
  // slave port the rule's rules sit on; an imported overlay's rules match
  // whole cubes of ids.
  device.walk([&](PacketRuleOp rule) {
    for (int id = 0; id <= max; ++id)
      if ((id & rule.maskInt()) == rule.valueInt())
        taken.insert(id);
  });
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
