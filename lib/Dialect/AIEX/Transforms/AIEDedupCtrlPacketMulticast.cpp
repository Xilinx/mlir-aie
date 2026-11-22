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
    bool atEnd = false;
  };

  llvm::DenseMap<Operation *, PacketEffects> effects;
  llvm::DenseMap<Operation *, Classified> writes;

  static SmallVector<PacketRun> matchingRuns(AIE::RuntimeSequenceOp sequence,
                                             bool internal = false) {
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
          if (internal && packet.getMulticastId())
            finish();
          else
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

  static Operation *endpoint(const Prefix &prefix, bool last) {
    NpuControlPacketOp base =
        last ? prefix.copies.front().back() : prefix.copies.front().front();
    Operation *result = base;
    for (const auto &copies : prefix.copies) {
      NpuControlPacketOp packet = last ? copies.back() : copies.front();
      if (last ? result->isBeforeInBlock(packet)
               : packet->isBeforeInBlock(result))
        result = packet;
    }
    return result;
  }

  bool preservesDependencies(const PacketRun &run, const Prefix &prefix,
                             bool internal = false) {
    llvm::DenseMap<Operation *, int> selected;
    llvm::DenseSet<uint32_t> selectedWords;
    SmallVector<DictionaryAttr> actions;
    int length = prefix.copies.front().size();
    for (const auto &copies : prefix.copies)
      for (auto [index, selectedPacket] : llvm::enumerate(copies)) {
        NpuControlPacketOp packet = selectedPacket;
        selected[packet] = prefix.atEnd ? length - 1 - index : index;
        if (internal) {
          if (!effects.lookup(packet).reorderable)
            actions.push_back(effects.lookup(packet).scope);
          for (size_t word = 0; word < packet.getData()->size(); ++word)
            selectedWords.insert(packet.getAddress() + 4 * word);
        }
      }
    DictionaryAttr actionScope = actions.empty()
                                     ? DictionaryAttr{}
                                     : unionScopes(&getContext(), actions);
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
    NpuControlPacketOp first = run.front().packet;
    Operation *start =
        internal ? endpoint(prefix, prefix.atEnd) : first.getOperation();
    for (Operation *operation = start; operation;
         operation = prefix.atEnd ? operation->getPrevNode()
                                  : operation->getNextNode()) {
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
      if (internal && chosen == selected.end() && effect.reorderable &&
          (!actionScope || !scopesOverlap(actionScope, effect.scope))) {
        bool overlaps = false;
        for (size_t word = 0; word < packet.getData()->size(); ++word)
          overlaps |= selectedWords.contains(packet.getAddress() + 4 * word);
        if (!overlaps)
          continue;
      }
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

  std::optional<Prefix>
  matchSubstring(const PacketRun &run,
                 const OverlayMulticasts::Multicast &multicast,
                 const OverlayMulticasts::Info &info,
                 ConfiguredDevices &devices, const std::set<Arbiter> &earlier) {
    SmallVector<SmallVector<NpuControlPacketOp>> lists;
    SmallVector<std::map<Write, int>> positions;
    for (AIE::TileID tile : multicast.tiles) {
      auto &packets = lists.emplace_back();
      auto &index = positions.emplace_back();
      for (ContextualPacket entry : run)
        if (entry.packet.getTileFromAddr() == tile) {
          NpuControlPacketOp packet = entry.packet;
          if (packet.getOpcode() == 0 && packet.getData() &&
              effects.lookup(packet).scope) {
            auto [found, inserted] =
                index.try_emplace(appliedWrite(packet), packets.size());
            if (!inserted)
              found->second = -1;
          }
          packets.push_back(packet);
        }
      if (packets.empty())
        return std::nullopt;
    }
    std::optional<Prefix> best;
    uint64_t bestScore = 0;
    unsigned examined = 0;
    for (size_t base = 0; base < lists.front().size() && examined < 64;
         ++base) {
      Write key = appliedWrite(lists.front()[base]);
      SmallVector<int> offsets;
      for (const auto &index : positions) {
        auto found = index.find(key);
        if (found == index.end() || found->second < 0)
          break;
        offsets.push_back(found->second);
      }
      if (offsets.size() != lists.size())
        continue;
      ++examined;
      Prefix candidate{{}, &multicast};
      candidate.copies.resize(lists.size());
      uint64_t score = 0;
      for (size_t length = 0;; ++length) {
        bool common = true;
        for (auto [member, packets] : llvm::enumerate(lists)) {
          size_t position = offsets[member] + length;
          if (position >= packets.size() ||
              !effects.lookup(packets[position]).scope ||
              appliedWrite(packets[position]) != key) {
            common = false;
            break;
          }
        }
        if (!common)
          break;
        for (auto [member, packets] : llvm::enumerate(lists))
          candidate.copies[member].push_back(packets[offsets[member] + length]);
        score += (2 + std::get<4>(key).size()) * (lists.size() - 1);
        if (base + length + 1 == lists.front().size())
          break;
        key = appliedWrite(lists.front()[base + length + 1]);
      }
      base += candidate.copies.front().size() - 1;
      if (score <= bestScore)
        continue;
      if (!preservesDependencies(run, candidate, true)) {
        candidate.atEnd = true;
        if (!preservesDependencies(run, candidate, true))
          continue;
      }
      candidate.wait = true;
      if (!planDelivery(run, candidate, info, devices, earlier))
        continue;
      bestScore = score;
      best = std::move(candidate);
    }
    return best;
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
    if (prefix.wait)
      return true;
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
        for (bool internal : {false, true}) {
          if (internal && !substrings)
            continue;
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
          auto runs = matchingRuns(seq, internal);
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
            if (internal && !parallelShims &&
                llvm::any_of(run, [&](ContextualPacket entry) {
                  return entry.packet.getColumnFromAddr() !=
                         run.front().packet.getColumnFromAddr();
                }))
              continue;
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
            uint64_t bestScore = 0;
            for (const auto &[column, multicast] : info->multicasts) {
              if (multicast.tiles.size() < 2)
                continue;
              if (internal) {
                auto candidate =
                    matchSubstring(run, multicast, *info, devices,
                                   earlierAt.lookup(run.front().packet));
                if (!candidate)
                  continue;
                uint64_t score = 0;
                for (NpuControlPacketOp packet : candidate->copies.front())
                  score += (2 + packet.getData()->size()) *
                           (candidate->copies.size() - 1);
                if (score > bestScore) {
                  bestScore = score;
                  prefixes.clear();
                  prefixes.push_back(std::move(*candidate));
                }
                continue;
              }
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
            if (prefixes.empty())
              continue;
            Operation *anchor =
                internal ? endpoint(prefixes.front(), prefixes.front().atEnd)
                         : run.front().packet.getOperation();
            OpBuilder start(anchor);
            if (internal && prefixes.front().atEnd)
              start.setInsertionPointAfter(anchor);
            auto insertion = ControlPacketBarrierOp::create(
                start, run.front().packet.getLoc(), start.getUnitAttr());
            for (Prefix &prefix : prefixes) {
              if (internal) {
                OpBuilder builder(insertion);
                ControlPacketBarrierOp::create(builder, insertion.getLoc());
              }
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
