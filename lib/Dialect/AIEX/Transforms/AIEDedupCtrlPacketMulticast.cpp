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
#include "CtrlPacketDevices.h"
#include "CtrlPacketMulticast.h"
#include "CtrlPacketRuns.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <map>
#include <optional>
#include <set>
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
    ConfiguredDevices devices(module);
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
          for (GroupRun &run : groupRuns(*block, {KindOrDeviceChange::NextPhase,
                                                  /*atBarriers=*/false}))
            if (failed(dedupRun(run, multicasts, classifier, devices)))
              return signalPassFailure();
          compactGroups(*block);
        }
      }
    }
  }

  // Dedups each phase of a delivery run, which aie-ctrl-packet-to-dma
  // awaits before the next, knowing the arbiters each packet's streams may
  // take once started.
  LogicalResult dedupRun(GroupRun &run, OverlayMulticasts &multicasts,
                         PacketClassifier &classifier,
                         ConfiguredDevices &devices) {
    llvm::DenseMap<Operation *, std::set<Arbiter>> started;
    for (GroupedPacket &p : run) {
      FailureOr<Classified> c = classifier.classify(p.packet);
      if (failed(c))
        return failure();
      std::set<Arbiter> held = devices.startedArbiters(
          p.group.getDeviceRefAttr(), p.packet.getTileFromAddr(), *c);
      if (!held.empty())
        started[p.packet] = std::move(held);
    }
    // What the streams of the phases before may hold.
    std::set<Arbiter> earlier;
    for (size_t begin = 0, end; begin < run.size(); begin = end) {
      SmallVector<NpuControlPacketOp> phase;
      std::set<Arbiter> held;
      for (end = begin; end < run.size() && run[end].phase == run[begin].phase;
           ++end) {
        phase.push_back(run[end].packet);
        if (auto it = started.find(run[end].packet); it != started.end())
          held.insert(it->second.begin(), it->second.end());
      }
      if (const OverlayMulticasts::Info *info =
              multicasts.of(run[begin].group.getDeviceRefAttr()))
        if (failed(dedupPhase(phase, *info, classifier, devices, started,
                              earlier)))
          return failure();
      earlier.insert(held.begin(), held.end());
    }
    return success();
  }

  // Dedups each multicast whose every tile has packets in the phase `run`.
  LogicalResult
  dedupPhase(ArrayRef<NpuControlPacketOp> run,
             const OverlayMulticasts::Info &info, PacketClassifier &classifier,
             ConfiguredDevices &devices,
             const llvm::DenseMap<Operation *, std::set<Arbiter>> &started,
             const std::set<Arbiter> &earlier) {
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
      if (failed(dedupMulticast(run, position, tiles, m, classifier, devices,
                                started, earlier, info, copies, first)))
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

  // Erases the emptied groups and a barrier right after another, and joins
  // the adjacent groups of one kind and device that leaves.
  static void compactGroups(Block &block) {
    for (auto group :
         llvm::make_early_inc_range(block.getOps<ControlPacketGroupOp>()))
      if (group.getBody().front().empty())
        group.erase();
    for (auto barrier :
         llvm::make_early_inc_range(block.getOps<ControlPacketBarrierOp>()))
      if (isa_and_present<ControlPacketBarrierOp>(barrier->getPrevNode()))
        barrier.erase();
    for (auto group :
         llvm::make_early_inc_range(block.getOps<ControlPacketGroupOp>())) {
      auto prev = dyn_cast_or_null<ControlPacketGroupOp>(group->getPrevNode());
      if (prev && sameKindAndDevice(prev, group)) {
        Block &into = prev.getBody().front();
        Block &body = group.getBody().front();
        into.getOperations().splice(into.end(), body.getOperations());
        group.erase();
      }
    }
  }

  // Sends the leading packets every tile of the multicast has in common once,
  // from the lowest tile, collecting those to send first where the pass
  // description says.
  LogicalResult
  dedupMulticast(ArrayRef<NpuControlPacketOp> run,
                 const llvm::DenseMap<Operation *, size_t> &position,
                 std::map<AIE::TileID, SmallVector<NpuControlPacketOp>> &tiles,
                 const OverlayMulticasts::Multicast &m,
                 PacketClassifier &classifier, ConfiguredDevices &devices,
                 const llvm::DenseMap<Operation *, std::set<Arbiter>> &started,
                 const std::set<Arbiter> &earlier,
                 const OverlayMulticasts::Info &info,
                 SmallVectorImpl<Operation *> &copies,
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

    // The prefix stays in place only if every other packet to a member that
    // lands after it does so in order, along the multicast's own trail from
    // the same shim channel, or commutes with it.
    auto sendsFirst = [&]() -> FailureOr<bool> {
      if (parallelShims && llvm::any_of(m.tiles, [&](AIE::TileID t) {
            return t.col != m.column;
          }))
        return true;
      SmallVector<Classified> shared;
      for (size_t i = 0; i < len; ++i) {
        FailureOr<Classified> c = classifier.classify(base[i]);
        if (failed(c))
          return failure();
        shared.push_back(std::move(*c));
      }
      for (auto &[tile, packets] : tiles) {
        if (info.keepsOrder(m, tile, parallelShims))
          continue;
        for (NpuControlPacketOp packet : llvm::drop_begin(packets, len)) {
          if (!drainMerge)
            return true;
          FailureOr<Classified> c = classifier.classify(packet);
          if (failed(c))
            return failure();
          if (llvm::any_of(shared, [&](const Classified &s) {
                return !s.commutesWith(*c);
              }))
            return true;
        }
      }
      return false;
    };

    // Sending packet `i` of every tile from the base's position, or the
    // stretch's start, moves each copy ahead of the packets between, and onto
    // the base's shim channel with the packets before it in the stretch, none
    // of which is awaited first. So it must start no stream one of those may
    // wait behind (see aie-reorder-ctrl-packets-by-tile).
    llvm::DenseSet<Operation *> merged;
    auto startsAheadOfBlocked = [&](size_t i) {
      for (auto *packets : lists) {
        NpuControlPacketOp copy = (*packets)[i];
        auto held = started.find(copy);
        if (held == started.end())
          continue;
        for (size_t k = 0, end = position.lookup(copy); k < end; ++k)
          if (!merged.contains(run[k]) &&
              arbiterTaken(run[k], &info, devices, held->second))
            return true;
      }
      return false;
    };
    for (size_t i = 0; i < len; ++i) {
      for (auto *packets : lists)
        merged.insert((*packets)[i]);
      if (startsAheadOfBlocked(i)) {
        len = i;
        break;
      }
    }
    if (len == 0)
      return success();

    // On the multicast the prefix takes the tree's arbiters, which no stream
    // started in a phase before, or by a packet of this one, the prefix's own
    // included, may hold.
    if (auto tree = info.trees.find(m.column); tree != info.trees.end()) {
      auto takes = [&](const std::set<Arbiter> &held) {
        return llvm::any_of(tree->second,
                            [&](const Arbiter &a) { return held.count(a); });
      };
      if (takes(earlier) || llvm::any_of(run, [&](NpuControlPacketOp p) {
            auto held = started.find(p);
            return held != started.end() && takes(held->second);
          }))
        return success();
    }
    FailureOr<bool> moveFirst = sendsFirst();
    if (failed(moveFirst))
      return failure();

    SmallVector<NpuControlPacketOp> prefix;
    for (size_t i = 0; i < len; ++i) {
      base[i].setMulticastId(m.id);
      prefix.push_back(base[i]);
      for (auto *packets : llvm::drop_begin(lists))
        copies.push_back((*packets)[i]);
    }
    if (*moveFirst)
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
