//===- AIEReorderCtrlPacketsByTile.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CtrlPacketRuns.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

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

  void normalize(SmallVectorImpl<NpuControlPacketOp> &packets) {
    if (packets.size() < 2 || (!addressOrder && !eliminateDeadWrites)) {
      packets.clear();
      return;
    }
    Block *block = packets.back()->getBlock();
    auto end = std::next(packets.back()->getIterator());
    llvm::DenseSet<uint32_t> written;
    SmallVector<NpuControlPacketOp> surviving;
    for (NpuControlPacketOp packet : llvm::reverse(packets)) {
      if (!eliminateDeadWrites) {
        surviving.push_back(packet);
        continue;
      }
      auto data = packet.getData().value();
      SmallVector<bool, 4> live;
      for (size_t index = 0; index < data.size(); ++index)
        live.push_back(written.insert(packet.getAddress() + 4 * index).second);
      if (llvm::all_of(live, [](bool value) { return value; })) {
        surviving.push_back(packet);
        continue;
      }
      OpBuilder builder(packet);
      for (size_t begin = 0; begin < data.size();) {
        if (!live[begin]) {
          ++begin;
          continue;
        }
        size_t limit = begin + 1;
        while (limit < data.size() && live[limit])
          ++limit;
        surviving.push_back(NpuControlPacketOp::create(
            builder, packet.getLoc(),
            builder.getUI32IntegerAttr(packet.getAddress() + 4 * begin),
            IntegerAttr{}, packet.getOpcodeAttr(), packet.getStreamIdAttr(),
            builder.getDenseI32ArrayAttr(data.slice(begin, limit - begin))));
        begin = limit;
      }
      packet.erase();
    }
    if (addressOrder) {
      llvm::sort(surviving,
                 [](NpuControlPacketOp left, NpuControlPacketOp right) {
                   return left.getAddress() < right.getAddress();
                 });
      for (NpuControlPacketOp packet : surviving)
        packet->moveBefore(block, end);
    }
    packets.clear();
  }

  void runOnOperation() override {
    getOperation().walk([&](AIE::RuntimeSequenceOp sequence) {
      sequence.walk([&](Block *block) {
        for (PacketInterval interval : packetIntervals(*block)) {
          if (addressOrder && interval.independent &&
              llvm::none_of(interval.packets, [](NpuControlPacketOp packet) {
                return packet.getMulticastId().has_value();
              })) {
            auto end = std::next(interval.packets.back()->getIterator());
            llvm::stable_sort(interval.packets, [](NpuControlPacketOp left,
                                                   NpuControlPacketOp right) {
              return left.getTileFromAddr() < right.getTileFromAddr();
            });
            for (NpuControlPacketOp packet : interval.packets)
              packet->moveBefore(block, end);
          }
          if (!interval.ordinary)
            continue;
          SmallVector<NpuControlPacketOp> packets;
          llvm::DenseSet<uint32_t> written;
          auto finish = [&] {
            normalize(packets);
            written.clear();
          };
          for (NpuControlPacketOp packet : interval.packets) {
            if (packet.getMulticastId()) {
              finish();
              continue;
            }
            if (!packets.empty() &&
                packets.back().getStreamId() != packet.getStreamId())
              finish();
            if (addressOrder && !eliminateDeadWrites) {
              for (size_t index = 0; index < packet.getData()->size(); ++index)
                if (written.contains(packet.getAddress() + 4 * index)) {
                  finish();
                  break;
                }
              for (size_t index = 0; index < packet.getData()->size(); ++index)
                written.insert(packet.getAddress() + 4 * index);
            }
            packets.push_back(packet);
          }
          finish();
        }
      });
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
AIEX::createAIEReorderCtrlPacketsByTilePass() {
  return std::make_unique<AIEReorderCtrlPacketsByTilePass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
AIEX::createAIEReorderCtrlPacketsByTilePass(
    const AIEReorderCtrlPacketsByTileOptions &options) {
  return std::make_unique<AIEReorderCtrlPacketsByTilePass>(options);
}