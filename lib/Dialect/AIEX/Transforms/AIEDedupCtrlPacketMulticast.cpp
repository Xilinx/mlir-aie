//===- AIEDedupCtrlPacketMulticast.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "CtrlPacketMulticast.h"
#include "CtrlPacketRuns.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"

#include <map>
#include <optional>
#include <tuple>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEDEDUPCTRLPACKETMULTICAST
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

// A control packet's write without the address's column and row bits, so the
// copies of one write to a multicast's tiles compare equal.
using Write = std::tuple<uint32_t, uint32_t, uint32_t, std::optional<uint32_t>,
                         SmallVector<int32_t>>;

static Write appliedWrite(NpuControlPacketOp op) {
  SmallVector<int32_t> data;
  if (auto d = op.getData())
    data.assign(d->begin(), d->end());
  return {op.getOpcode(), op.getStreamId(), op.getLocalAddress(),
          op.getLength(), std::move(data)};
}

struct AIEDedupCtrlPacketMulticastPass
    : xilinx::AIEX::impl::AIEDedupCtrlPacketMulticastBase<
          AIEDedupCtrlPacketMulticastPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OverlayMulticasts multicasts(module, false);
    for (auto device : module.getOps<AIE::DeviceOp>()) {
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>()) {
        for (PacketRun &run : packetRuns(seq)) {
          const OverlayMulticasts::Info *info =
              multicasts.of(run.front().device);
          llvm::DenseSet<uint32_t> written;
          if (!info || llvm::any_of(run, [&](ContextualPacket entry) {
                NpuControlPacketOp packet = entry.packet;
                if (packet.getStreamId() != run.front().packet.getStreamId() ||
                    packet.getOpcode() != 0 || !packet.getData() ||
                    packet.getMulticastId())
                  return true;
                for (size_t index = 0; index < packet.getData()->size();
                     ++index)
                  if (!written.insert(packet.getAddress() + 4 * index).second)
                    return true;
                return false;
              }))
            continue;
          std::map<AIE::TileID, SmallVector<NpuControlPacketOp>> byTile;
          for (ContextualPacket entry : run)
            byTile[entry.packet.getTileFromAddr()].push_back(entry.packet);
          for (const auto &[column, multicast] : info->multicasts) {
            if (multicast.tiles.size() < 2 ||
                llvm::any_of(multicast.tiles, [&](AIE::TileID tile) {
                  return !byTile.count(tile);
                }))
              continue;
            auto &base = byTile.at(multicast.tiles.front());
            size_t count = 0;
            for (; count < base.size(); ++count) {
              NpuControlPacketOp packet = base[count];
              if (packet.getOpcode() != 0 || !packet.getData() ||
                  packet.getMulticastId())
                break;
              Write write = appliedWrite(packet);
              if (llvm::any_of(llvm::drop_begin(multicast.tiles),
                               [&](AIE::TileID tile) {
                                 auto &copies = byTile.at(tile);
                                 return count >= copies.size() ||
                                        copies[count].getMulticastId() ||
                                        appliedWrite(copies[count]) != write;
                               }))
                break;
            }
            if (!count)
              continue;
            for (size_t index = 0; index < count; ++index) {
              base[index].setMulticastId(multicast.id);
              for (AIE::TileID tile : llvm::drop_begin(multicast.tiles))
                byTile.at(tile)[index].erase();
            }
            OpBuilder builder(base.front());
            ControlPacketBarrierOp::create(builder, base.front().getLoc());
            builder.setInsertionPointAfter(base[count - 1]);
            ControlPacketBarrierOp::create(builder, base[count - 1].getLoc());
          }
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEDedupCtrlPacketMulticastPass() {
  return std::make_unique<AIEDedupCtrlPacketMulticastPass>();
}
