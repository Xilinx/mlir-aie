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

  struct Prefix {
    SmallVector<SmallVector<NpuControlPacketOp>> copies;
    const OverlayMulticasts::Multicast *multicast;
    bool wait = false;
  };

  llvm::DenseMap<Operation *, PacketEffects> effects;
  llvm::DenseMap<Operation *, Classified> writes;

  static SmallVector<PacketRun> matchingRuns(AIE::RuntimeSequenceOp sequence) {
    SmallVector<PacketRun> runs;
    sequence.walk([&](Block *block) {
      PacketRun run;
      FlatSymbolRefAttr device;
      auto finish = [&] {
        if (!run.empty())
          runs.push_back(std::move(run));
        run.clear();
      };
      for (Operation &operation : *block) {
        if (auto barrier = dyn_cast<ControlPacketBarrierOp>(operation);
            barrier && barrier.getCompilerOnly() && barrier.getScopeAttr())
          continue;
        if (auto packet = dyn_cast<NpuControlPacketOp>(operation)) {
          run.push_back({packet, device, 0});
          continue;
        }
        finish();
        if (auto context = dyn_cast<SetDeviceOp>(operation))
          device = context.getDeviceRefAttr();
        else if (isa<NpuLoadPdiOp>(operation))
          device = {};
      }
      finish();
    });
    return runs;
  }

  static std::optional<Prefix>
  matchPrefix(const PacketRun &run,
              const OverlayMulticasts::Multicast &multicast) {
    Prefix prefix{{}, &multicast};
    for (AIE::TileID tile : multicast.tiles) {
      auto &packets = prefix.copies.emplace_back();
      for (ContextualPacket entry : run)
        if (entry.packet.getTileFromAddr() == tile)
          packets.push_back(entry.packet);
      if (packets.empty())
        return std::nullopt;
    }
    size_t length = 0;
    for (NpuControlPacketOp packet : prefix.copies.front()) {
      if (packet.getOpcode() != 0 || !packet.getData() ||
          packet.getMulticastId() || !packetEffects(packet).scope)
        break;
      Write write = appliedWrite(packet);
      if (llvm::any_of(llvm::drop_begin(prefix.copies), [&](auto &copies) {
            return length >= copies.size() || copies[length].getMulticastId() ||
                   appliedWrite(copies[length]) != write;
          }))
        break;
      ++length;
    }
    if (!length)
      return std::nullopt;
    for (auto &copies : prefix.copies)
      copies.resize(length);
    return prefix;
  }

  bool preservesDependencies(const PacketRun &run, const Prefix &prefix) {
    llvm::DenseMap<Operation *, int> selected;
    for (const auto &copies : prefix.copies)
      for (auto [index, packet] : llvm::enumerate(copies))
        selected[packet] = index;
    using Module = std::pair<StringRef, AIE::TileID>;
    using Ranks = std::map<Module, int>;
    Ranks targets, accesses, fences;
    auto maximum = [](const Ranks &ranks, DictionaryAttr scope) {
      int result = -1;
      for (NamedAttribute entry : scope)
        for (auto tile :
             cast<ArrayAttr>(entry.getValue()).getAsRange<AIE::TileAttr>()) {
          auto found =
              ranks.find({entry.getName().getValue(), tile.getTileID()});
          if (found != ranks.end())
            result = std::max(result, found->second);
        }
      return result;
    };
    auto record = [](Ranks &ranks, DictionaryAttr scope, int rank) {
      for (NamedAttribute entry : scope)
        for (auto tile :
             cast<ArrayAttr>(entry.getValue()).getAsRange<AIE::TileAttr>()) {
          auto [found, inserted] = ranks.try_emplace(
              Module{entry.getName().getValue(), tile.getTileID()}, rank);
          if (!inserted)
            found->second = std::max(found->second, rank);
        }
    };
    size_t remaining = selected.size();
    int unselected = prefix.copies.front().size();
    for (Operation *operation = run.front().packet; operation;
         operation = operation->getNextNode()) {
      if (auto barrier = dyn_cast<ControlPacketBarrierOp>(operation)) {
        if (!barrier.getCompilerOnly() || !barrier.getScopeAttr())
          return false;
        if (!barrier.getPacketEffect())
          record(fences, barrier.getScopeAttr(), unselected);
        continue;
      }
      auto packet = dyn_cast<NpuControlPacketOp>(operation);
      if (!packet)
        return false;
      PacketEffects effect = effects.lookup(packet);
      if (!effect.scope || !effect.target)
        return false;
      auto chosen = selected.find(packet);
      int rank = chosen == selected.end() ? unselected : chosen->second;
      if (chosen != selected.end()) {
        if (maximum(targets, effect.scope) >= rank ||
            maximum(accesses, effect.target) >= rank ||
            maximum(fences, effect.scope) >= rank)
          return false;
        if (!--remaining)
          return true;
      }
      record(targets, effect.target, rank);
      record(accesses, effect.scope, rank);
    }
    return false;
  }

  bool planDelivery(const PacketRun &run, Prefix &prefix,
                    const OverlayMulticasts::Info &info,
                    ConfiguredDevices &devices,
                    const std::set<Arbiter> &earlier) {
    const auto &multicast = *prefix.multicast;
    auto tree = info.trees.find(multicast.column);
    if (tree != info.trees.end()) {
      auto conflicts = [&](const std::set<Arbiter> &held) {
        return llvm::any_of(tree->second, [&](const Arbiter &arbiter) {
          return held.count(arbiter);
        });
      };
      if (conflicts(earlier))
        return false;
      for (ContextualPacket entry : run)
        for (AIE::TileID tile : tilesReached(entry.packet, &info))
          if (conflicts(devices.startedArbiters(entry.device, tile,
                                                writes.lookup(entry.packet))))
            return false;
    }
    llvm::DenseSet<Operation *> selected;
    std::set<Arbiter> started;
    for (const auto &copies : prefix.copies)
      for (NpuControlPacketOp packet : copies) {
        selected.insert(packet);
        auto held = devices.startedArbiters(run.front().device,
                                            packet.getTileFromAddr(),
                                            writes.lookup(packet));
        started.insert(held.begin(), held.end());
      }
    if (!started.empty())
      for (ContextualPacket entry : run)
        if (!selected.contains(entry.packet) &&
            devices.controlArbiterIn(entry.device,
                                     entry.packet.getTileFromAddr(), started))
          return false;
    for (const auto &copies : prefix.copies) {
      NpuControlPacketOp first = copies.front();
      AIE::TileID tile = first.getTileFromAddr();
      SmallVector<DictionaryAttr> scopes, targets;
      for (NpuControlPacketOp packet : copies) {
        scopes.push_back(effects.lookup(packet).scope);
        targets.push_back(effects.lookup(packet).target);
      }
      DictionaryAttr scope = unionScopes(first.getContext(), scopes);
      DictionaryAttr target = unionScopes(first.getContext(), targets);
      for (ContextualPacket entry : run) {
        if (selected.contains(entry.packet))
          continue;
        AIE::TileID otherTile = entry.packet.getTileFromAddr();
        if (otherTile != tile) {
          if (!parallelShims && otherTile.col != multicast.column)
            continue;
          PacketEffects other = effects.lookup(entry.packet);
          if (scopesOverlap(target, other.scope) ||
              scopesOverlap(scope, other.target)) {
            prefix.wait = true;
            return true;
          }
          continue;
        }
        if (info.keepsOrder(multicast, tile, parallelShims))
          continue;
        Classified other = writes.lookup(entry.packet);
        for (NpuControlPacketOp packet : copies) {
          if (!writes.lookup(packet).commutesWith(other) ||
              !effects.lookup(packet).reorderable ||
              !effects.lookup(entry.packet).reorderable) {
            prefix.wait = true;
            return true;
          }
        }
      }
    }
    return true;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OverlayMulticasts multicasts(module);
    ConfiguredDevices devices(module);
    for (auto device : module.getOps<AIE::DeviceOp>()) {
      PacketClassifier classifier(device.getTargetModel());
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>()) {
        effects.clear();
        WalkResult classified = seq.walk([&](Block *block) {
          auto result = packetEffects(*block);
          if (failed(result))
            return WalkResult::interrupt();
          effects.insert(result->begin(), result->end());
          return WalkResult::advance();
        });
        if (classified.wasInterrupted())
          return signalPassFailure();
        auto runs = matchingRuns(seq);
        llvm::DenseSet<Operation *> starts;
        for (const PacketRun &run : runs)
          starts.insert(run.front().packet);
        llvm::DenseMap<Operation *, std::set<Arbiter>> earlierAt;
        for (PacketRun &run : packetRuns(seq, true)) {
          std::set<Arbiter> held;
          for (ContextualPacket entry : run) {
            if (starts.contains(entry.packet))
              earlierAt[entry.packet] = held;
            auto write = classifier.classify(entry.packet);
            if (failed(write))
              return signalPassFailure();
            for (AIE::TileID tile :
                 tilesReached(entry.packet, multicasts.of(entry.device))) {
              auto started =
                  devices.startedArbiters(entry.device, tile, *write);
              held.insert(started.begin(), started.end());
            }
          }
        }
        for (PacketRun &run : runs) {
          const OverlayMulticasts::Info *info =
              multicasts.of(run.front().device);
          if (!info || llvm::any_of(run, [&](ContextualPacket entry) {
                return entry.packet.getStreamId() !=
                       run.front().packet.getStreamId();
              }))
            continue;
          writes.clear();
          for (ContextualPacket entry : run) {
            auto write = classifier.classify(entry.packet);
            if (failed(write))
              return signalPassFailure();
            writes[entry.packet] = *write;
          }
          SmallVector<Prefix> prefixes;
          for (const auto &[column, multicast] : info->multicasts) {
            if (multicast.tiles.size() < 2)
              continue;
            auto prefix = matchPrefix(run, multicast);
            if (!prefix)
              continue;
            size_t fullLength = prefix->copies.front().size();
            size_t low = 0, high = fullLength;
            Prefix trial = *prefix;
            while (low < high) {
              size_t length = (low + high + 1) / 2;
              trial = *prefix;
              for (auto &copies : trial.copies)
                copies.resize(length);
              if (preservesDependencies(run, trial))
                low = length;
              else
                high = length - 1;
            }
            if (!low)
              continue;
            for (auto &copies : prefix->copies)
              copies.resize(low);
            if (!planDelivery(run, *prefix, *info, devices,
                              earlierAt.lookup(run.front().packet)))
              continue;
            if (parallelShims)
              for (Prefix &previous : prefixes) {
                auto left = info->trees.find(previous.multicast->column);
                auto right = info->trees.find(column);
                if (left != info->trees.end() && right != info->trees.end() &&
                    llvm::any_of(left->second, [&](const Arbiter &arbiter) {
                      return right->second.count(arbiter);
                    }))
                  previous.wait = true;
              }
            prefixes.push_back(std::move(*prefix));
          }
          OpBuilder start(run.front().packet);
          auto insertion = ControlPacketBarrierOp::create(
              start, run.front().packet.getLoc(), start.getUnitAttr());
          for (Prefix &prefix : prefixes) {
            auto &base = prefix.copies.front();
            for (auto [index, packet] : llvm::enumerate(base)) {
              SmallVector<DictionaryAttr> scopes;
              for (const auto &copies : prefix.copies)
                scopes.push_back(effects.lookup(copies[index]).scope);
              DictionaryAttr scope = unionScopes(packet.getContext(), scopes);
              OpBuilder builder(insertion);
              if (!effects.lookup(packet).reorderable)
                ControlPacketBarrierOp::create(builder, packet.getLoc(),
                                               builder.getUnitAttr(), scope,
                                               builder.getUnitAttr());
              packet.setMulticastId(prefix.multicast->id);
              packet->moveBefore(insertion);
              if (!effects.lookup(packet).reorderable)
                ControlPacketBarrierOp::create(builder, packet.getLoc(),
                                               builder.getUnitAttr(), scope,
                                               builder.getUnitAttr());
            }
            if (prefix.wait) {
              OpBuilder builder(insertion);
              ControlPacketBarrierOp::create(builder, base.back().getLoc());
            }
            for (auto &copies : llvm::drop_begin(prefix.copies))
              for (NpuControlPacketOp packet : copies)
                packet.erase();
          }
          insertion.erase();
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

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEDedupCtrlPacketMulticastPass(
    const AIEDedupCtrlPacketMulticastOptions &options) {
  return std::make_unique<AIEDedupCtrlPacketMulticastPass>(options);
}
