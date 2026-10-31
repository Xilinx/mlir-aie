//===- AIEAssignPacketIds.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPacketIdSpace.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEStructuralRank.h"

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
    if (clReserveControllerPacketIds)
      device->setAttr("aie.reserve_controller_packet_ids",
                      UnitAttr::get(device.getContext()));
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

    // Ids go out in structural order, so a design declared in another order
    // or under other names gets the same ones.
    // The pass flag is a default for routes that express no preference, so a
    // device may mix circuit- and packet-switched connections.
    SmallVector<RouteOp> unpinned;
    for (auto route : device.getOps<RouteOp>()) {
      std::optional<PacketInfoAttr> packet = route.getPacket();
      if ((packet || clPacketSwObjectFifos) &&
          !(packet && packet->isAssigned()))
        unpinned.push_back(route);
    }
    if (unpinned.size() > 1) {
      DenseMap<Operation *, unsigned> rank = StructuralGraph::of(device).rank();
      llvm::stable_sort(unpinned, [&](RouteOp a, RouteOp b) {
        return rank.lookup(a) < rank.lookup(b);
      });
    }

    // A DMA program states its BDs' header itself, so it cannot use an id
    // handed out here. Indexed once instead of a device-wide
    // SymbolTable::getSymbolUses walk per unpinned route's source: the first
    // DMAStartOp naming a given endpoint is all the loop below needs.
    DenseMap<StringRef, DMAStartOp> dmaStartByEndpoint;
    device.walk([&](DMAStartOp start) {
      if (FlatSymbolRefAttr name = start.getEndpoint())
        dmaStartByEndpoint.try_emplace(name.getValue(), start);
    });

    for (auto route : unpinned) {
      std::optional<PacketInfoAttr> packet = route.getPacket();
      for (StringRef source : route.getSourceNames()) {
        auto it = dmaStartByEndpoint.find(source);
        if (it == dmaStartByEndpoint.end())
          continue;
        it->second.emitOpError("names @")
            << source
            << ", the source of a packet-switched route without a pinned "
               "pkt_id; pin pkt_id on each route it sources and state that "
               "header in each of the program's BDs";
        return signalPassFailure();
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
xilinx::AIE::createAIEAssignPacketIdsPass(bool packetSwitched,
                                          bool reserveControllerPacketIds) {
  AIEAssignPacketIdsOptions options;
  options.clPacketSwObjectFifos = packetSwitched;
  options.clReserveControllerPacketIds = reserveControllerPacketIds;
  return std::make_unique<AIEAssignPacketIdsPass>(options);
}
