//===- AIEDedupCtrlPacketMulticast.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "CtrlPacketClassifier.h"
#include "CtrlPacketMulticast.h"
#include "CtrlPacketRuns.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"

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
    OverlayMulticasts multicasts(module);
    for (auto device : module.getOps<AIE::DeviceOp>()) {
      PacketClassifier classifier(device.getTargetModel());
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>()) {
        SmallVector<Block *> blocks;
        seq.walk([&](Block *block) {
          if (!isa<ControlPacketGroupOp>(block->getParentOp()))
            blocks.push_back(block);
        });
        for (Block *block : blocks) {
          for (GroupRun &run : groupRuns(*block)) {
            const OverlayMulticasts::Info *info =
                multicasts.of(run.front().group.getDeviceRefAttr());
            SmallVector<NpuControlPacketOp> packets = llvm::map_to_vector(
                run, [](GroupedPacket &p) { return p.packet; });
            if (info && failed(dedupRun(packets, *info, classifier)))
              return signalPassFailure();
          }
          compactGroups(*block);
        }
      }
    }
  }

  // Dedups each multicast whose every tile has packets in the run.
  LogicalResult dedupRun(ArrayRef<NpuControlPacketOp> run,
                         const OverlayMulticasts::Info &info,
                         PacketClassifier &classifier) {
    // By the multicast's column.
    std::map<int, std::map<AIE::TileID, SmallVector<NpuControlPacketOp>>>
        members;
    for (NpuControlPacketOp packet : run) {
      AIE::TileID tile = packet.getTileFromAddr();
      auto it = info.byTile.find(tile);
      if (it != info.byTile.end())
        members[it->second][tile].push_back(packet);
    }
    llvm::DenseMap<Operation *, size_t> position;
    for (auto [i, packet] : llvm::enumerate(run))
      position[packet] = i;
    // The copies go after every multicast is done, since each reads the run.
    SmallVector<Operation *> copies;
    SmallVector<SmallVector<NpuControlPacketOp>> first;
    for (auto &[column, tiles] : members) {
      const OverlayMulticasts::Multicast &m = info.multicasts.at(column);
      if (tiles.size() != m.tiles.size())
        continue;
      if (failed(dedupMulticast(position, tiles, m, classifier, info, copies,
                                first)))
        return failure();
    }
    for (Operation *op : copies)
      op->erase();
    if (!first.empty())
      sendFirst(run.front(), first);
    return success();
  }

  // Moves each multicast's packets to send first to the stretch's start, each
  // ahead of a barrier of its own.
  static void sendFirst(NpuControlPacketOp start,
                        ArrayRef<SmallVector<NpuControlPacketOp>> first) {
    auto old = cast<ControlPacketGroupOp>(start->getParentOp());
    OpBuilder builder(old);
    for (const SmallVector<NpuControlPacketOp> &packets : first) {
      auto group = ControlPacketGroupOp::create(
          builder, old.getLoc(), old.getKindAttr(), old.getDeviceRefAttr());
      group.getBody().emplaceBlock();
      Block &body = group.getBody().front();
      for (NpuControlPacketOp packet : packets)
        packet->moveBefore(&body, body.end());
      ControlPacketBarrierOp::create(builder, old.getLoc());
    }
  }

  // Erases the emptied groups and joins the adjacent ones of one kind and
  // device that leaves.
  static void compactGroups(Block &block) {
    for (auto group :
         llvm::make_early_inc_range(block.getOps<ControlPacketGroupOp>())) {
      Block &body = group.getBody().front();
      if (body.empty()) {
        group.erase();
        continue;
      }
      auto prev = dyn_cast_or_null<ControlPacketGroupOp>(group->getPrevNode());
      if (prev && sameKindAndDevice(prev, group)) {
        Block &into = prev.getBody().front();
        into.getOperations().splice(into.end(), body.getOperations());
        group.erase();
      }
    }
  }

  // Sends the leading packets every tile of the multicast has in common once,
  // from the lowest tile, collecting those to send first where the pass
  // description says.
  LogicalResult dedupMulticast(
      const llvm::DenseMap<Operation *, size_t> &position,
      std::map<AIE::TileID, SmallVector<NpuControlPacketOp>> &tiles,
      const OverlayMulticasts::Multicast &m, PacketClassifier &classifier,
      const OverlayMulticasts::Info &info, SmallVectorImpl<Operation *> &copies,
      SmallVectorImpl<SmallVector<NpuControlPacketOp>> &first) {
    SmallVector<SmallVector<NpuControlPacketOp> *> lists;
    for (auto &[tile, packets] : tiles)
      lists.push_back(&packets);
    SmallVector<NpuControlPacketOp> &base = *lists.front();
    // Each tile's prefix is its first packets in the stretch, so the
    // multicast changes no tile's order of its own packets. Only writes go
    // on it, since every tile would answer a read.
    size_t len = 0;
    for (; len < base.size() && base[len].getOpcode() == 0; ++len) {
      Write w = appliedWrite(base[len]);
      bool common = llvm::all_of(llvm::drop_begin(lists), [&](auto *packets) {
        return len < packets->size() && appliedWrite((*packets)[len]) == w &&
               position.lookup(base[len]) < position.lookup((*packets)[len]);
      });
      if (!common)
        break;
    }
    if (len == 0)
      return success();

    // The prefix stays in place only if every other packet to a member that
    // lands after it does so in order, along the multicast's own trail from
    // the same shim channel, or commutes with it.
    SmallVector<Classified> shared;
    for (size_t i = 0; i < len; ++i) {
      FailureOr<Classified> c = classifier.classify(base[i]);
      if (failed(c))
        return failure();
      shared.push_back(std::move(*c));
    }
    bool moveFirst = false;
    for (auto &[tile, packets] : tiles) {
      if (info.sharesTrail(m, tile))
        continue;
      for (NpuControlPacketOp packet : llvm::drop_begin(packets, len)) {
        if (!drainMerge) {
          moveFirst = true;
          break;
        }
        FailureOr<Classified> c = classifier.classify(packet);
        if (failed(c))
          return failure();
        if (llvm::any_of(shared, [&](const Classified &s) {
              return !s.commutesWith(*c);
            })) {
          moveFirst = true;
          break;
        }
      }
      if (moveFirst)
        break;
    }

    SmallVector<NpuControlPacketOp> prefix;
    for (size_t i = 0; i < len; ++i) {
      base[i].setMulticastId(m.id);
      prefix.push_back(base[i]);
      for (auto *packets : llvm::drop_begin(lists))
        copies.push_back((*packets)[i]);
    }
    if (moveFirst)
      first.push_back(std::move(prefix));
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEDedupCtrlPacketMulticastPass() {
  return std::make_unique<AIEDedupCtrlPacketMulticastPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEDedupCtrlPacketMulticastPass(
    const AIEDedupCtrlPacketMulticastOptions &options) {
  return std::make_unique<AIEDedupCtrlPacketMulticastPass>(options);
}
