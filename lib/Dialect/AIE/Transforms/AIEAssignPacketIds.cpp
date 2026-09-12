//===- AIEAssignPacketIds.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPacketIdSpace.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEASSIGNPACKETIDS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

struct AIEAssignPacketIdsPass
    : public xilinx::AIE::impl::AIEAssignPacketIdsBase<AIEAssignPacketIdsPass> {
  using Base::Base;

  void runOnOperation() override {
    DeviceOp device = getOperation();
    PacketIdSpace space(device);

    // Pinned ids are checked before any are handed out, so a bad pin is
    // reported as one rather than surfacing as an exhausted field.
    for (auto route : device.getOps<RouteOp>()) {
      std::optional<PacketInfoAttr> packet = route.getPacket();
      if (packet && packet->isAssigned() &&
          packet->assignedId() > space.maxId()) {
        route.emitOpError("pkt_id ")
            << packet->assignedId() << " is out of range (max " << space.maxId()
            << ")";
        return signalPassFailure();
      }
    }

    for (auto route : device.getOps<RouteOp>()) {
      std::optional<PacketInfoAttr> packet = route.getPacket();
      // The pass flag is a default for routes that express no preference, so
      // a device may mix circuit- and packet-switched connections.
      if (!packet && !clPacketSwObjectFifos) {
        continue;
      }
      if (packet && packet->isAssigned()) {
        continue;
      }
      std::optional<int> id = space.takeLowestFrom();
      if (!id) {
        route.emitOpError("max number of packet IDs reached");
        return signalPassFailure();
      }
      uint16_t type = packet ? packet->getPktType() : 0;
      route.setPacketAttr(PacketInfoAttr::get(device.getContext(), type, *id));
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEAssignPacketIdsPass() {
  return std::make_unique<AIEAssignPacketIdsPass>();
}

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIE::createAIEAssignPacketIdsPass(bool packetSwitched) {
  AIEAssignPacketIdsOptions options;
  options.clPacketSwObjectFifos = packetSwitched;
  return std::make_unique<AIEAssignPacketIdsPass>(options);
}
