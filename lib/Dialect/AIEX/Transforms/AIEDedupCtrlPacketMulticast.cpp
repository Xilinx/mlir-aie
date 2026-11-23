//===- AIEDedupCtrlPacketMulticast.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "CtrlPacketRuns.h"
#include "mlir/Pass/Pass.h"

#include <map>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEDEDUPCTRLPACKETMULTICAST
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

struct AIEDedupCtrlPacketMulticastPass
    : xilinx::AIEX::impl::AIEDedupCtrlPacketMulticastBase<
          AIEDedupCtrlPacketMulticastPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](AIE::RuntimeSequenceOp sequence) {
      sequence.walk([&](Block *block) {
        for (PacketInterval interval : packetIntervals(*block)) {
          if ((!interval.unobserved && !interval.independent) ||
              !interval.device ||
              llvm::any_of(interval.packets, [](NpuControlPacketOp packet) {
                return packet.getMulticastId().has_value();
              }))
            continue;
          ArrayAttr groups = controlPacketMulticasts(module, interval.device);
          if (!groups)
            continue;
          std::map<AIE::TileID, SmallVector<NpuControlPacketOp>> byTile;
          for (NpuControlPacketOp packet : interval.packets)
            byTile[packet.getTileFromAddr()].push_back(packet);
          OpBuilder builder(interval.packets.front());
          auto anchor = ControlPacketBarrierOp::create(
              builder, interval.packets.front().getLoc(),
              builder.getUnitAttr());
          for (Attribute attribute : groups) {
            auto group = cast<DictionaryAttr>(attribute);
            SmallVector<SmallVector<NpuControlPacketOp> *> copies;
            for (auto tile :
                 group.getAs<ArrayAttr>("tiles").getAsRange<AIE::TileAttr>())
              copies.push_back(&byTile[tile.getTileID()]);
            if (copies.size() < 2)
              continue;
            size_t length = 0;
            for (NpuControlPacketOp packet : *copies.front()) {
              if (packet.getOpcode() != 0 || !packet.getData())
                break;
              if (llvm::any_of(llvm::drop_begin(copies), [&](auto *other) {
                    if (length >= other->size())
                      return true;
                    NpuControlPacketOp candidate = (*other)[length];
                    return candidate.getOpcode() != 0 ||
                           packet.getLocalAddress() !=
                               candidate.getLocalAddress() ||
                           packet.getDataAttr() != candidate.getDataAttr() ||
                           packet.getStreamId() != candidate.getStreamId() ||
                           packet.getLength() != candidate.getLength();
                  }))
                break;
              ++length;
            }
            for (size_t index = 0; index < length; ++index) {
              NpuControlPacketOp packet = (*copies.front())[index];
              packet.setMulticastId(group.getAs<IntegerAttr>("id").getInt());
              packet->moveBefore(anchor);
              for (auto *other : llvm::drop_begin(copies))
                (*other)[index].erase();
            }
            for (auto *copy : copies)
              copy->erase(copy->begin(), copy->begin() + length);
          }
          anchor.erase();
        }
      });
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
AIEX::createAIEDedupCtrlPacketMulticastPass() {
  return std::make_unique<AIEDedupCtrlPacketMulticastPass>();
}
