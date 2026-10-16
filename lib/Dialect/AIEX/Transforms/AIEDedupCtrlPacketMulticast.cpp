//===- AIEDedupCtrlPacketMulticast.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sends the control packets a column multicast group's tiles have in common
// once, on the group's multicast packet id, instead of once per tile.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "ControlPacketClassifier.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <map>
#include <tuple>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEDEDUPCTRLPACKETMULTICAST
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

// What a tile applies from a control packet. It leaves out the address's column
// and row bits, which the multicast does not change.
using Write = std::tuple<uint32_t, uint32_t, uint32_t, std::optional<uint32_t>,
                         SmallVector<int32_t>>;

static Write appliedWrite(NpuControlPacketOp op, uint32_t rowShift) {
  SmallVector<int32_t> data;
  if (auto d = op.getData())
    data.assign(d->begin(), d->end());
  return {op.getOpcode(), op.getStreamId(),
          op.getAddress() & ((1u << rowShift) - 1), op.getLength(),
          std::move(data)};
}

struct AIEDedupCtrlPacketMulticastPass
    : xilinx::AIEX::impl::AIEDedupCtrlPacketMulticastBase<
          AIEDedupCtrlPacketMulticastPass> {
  using Base::Base;

  // A multicast group, as the device whose configuration the packets carry and
  // the group's multicast id.
  using GroupKey = std::pair<StringRef, int64_t>;
  using TileKey = std::pair<int, int>;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    AIE::AIEDialect::CtrlPktMulticastIdAttrHelper tileStamp(&getContext());
    AIEXDialect::CtrlPktDeviceAttrHelper deviceStamp(&getContext());

    // (column, row) -> multicast id, per configured device.
    llvm::DenseMap<StringAttr, std::map<std::pair<int, int>, int64_t>> ids;
    // A packet's multicast id and the number of tiles its multicast reaches.
    auto idOf = [&](NpuControlPacketOp packet)
        -> std::optional<std::pair<int64_t, size_t>> {
      FlatSymbolRefAttr device = deviceStamp.getAttr(packet);
      if (!device)
        return std::nullopt;
      auto it = ids.find(device.getAttr());
      if (it == ids.end()) {
        std::map<std::pair<int, int>, int64_t> tiles;
        if (auto d = module.lookupSymbol<AIE::DeviceOp>(device))
          d.walk([&](AIE::TileOp t) {
            if (auto id = tileStamp.getAttr(t))
              tiles[{t.getCol(), t.getRow()}] = id.getInt();
          });
        it = ids.try_emplace(device.getAttr(), std::move(tiles)).first;
      }
      auto tile = it->second.find({static_cast<int>(packet.getColumnFromAddr()),
                                   static_cast<int>(packet.getRowFromAddr())});
      if (tile == it->second.end())
        return std::nullopt;
      size_t size = llvm::count_if(it->second, [&](const auto &entry) {
        return entry.second == tile->second;
      });
      return std::make_pair(tile->second, size);
    };

    for (auto device : module.getOps<AIE::DeviceOp>()) {
      uint32_t rowShift = device.getTargetModel().getRowShift();
      PacketClassifier classifier(device.getTargetModel());
      bool failed = false;
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>())
        seq.walk([&](Block *block) {
          SmallVector<NpuControlPacketOp> run;
          for (Operation &op : *block) {
            if (auto packet = dyn_cast<NpuControlPacketOp>(op)) {
              run.push_back(packet);
              continue;
            }
            failed |= mlir::failed(dedupRun(run, idOf, rowShift, classifier));
            run.clear();
          }
          failed |= mlir::failed(dedupRun(run, idOf, rowShift, classifier));
        });
      if (failed)
        return signalPassFailure();
    }
  }

  // Dedups each group within one run, the enable phase apart from the rest.
  template <typename IdOf>
  LogicalResult dedupRun(ArrayRef<NpuControlPacketOp> run, IdOf &idOf,
                         uint32_t rowShift, PacketClassifier &classifier) {
    AIEXDialect::CtrlPktDeviceAttrHelper deviceStamp(&getContext());
    AIEXDialect::CtrlPktEnablePhaseAttrHelper enablePhase(&getContext());
    // (group, enable phase) -> tile -> the tile's packets in order.
    std::map<std::pair<GroupKey, bool>,
             std::map<TileKey, SmallVector<NpuControlPacketOp>>>
        members;
    std::map<GroupKey, size_t> sizes;
    for (NpuControlPacketOp packet : run) {
      auto id = idOf(packet);
      if (!id)
        continue;
      GroupKey key{deviceStamp.getAttr(packet).getValue(), id->first};
      sizes[key] = id->second;
      members[{key, enablePhase.isAttrPresent(packet)}]
             [{static_cast<int>(packet.getColumnFromAddr()),
               static_cast<int>(packet.getRowFromAddr())}]
                 .push_back(packet);
    }
    // The copies go after every group is done, since each reads the run.
    SmallVector<Operation *> copies;
    for (auto &[key, tiles] : members)
      if (tiles.size() == sizes[key.first] &&
          failed(dedupGroup(run, tiles, key.first.second, key.second, rowShift,
                            classifier, copies)))
        return failure();
    for (Operation *op : copies)
      op->erase();
    return success();
  }

  // Sends the leading packets every tile of the group has in common once, from
  // the lowest tile, in a base phase where the pass description says.
  LogicalResult
  dedupGroup(ArrayRef<NpuControlPacketOp> run,
             std::map<TileKey, SmallVector<NpuControlPacketOp>> &members,
             int64_t id, bool inEnablePhase, uint32_t rowShift,
             PacketClassifier &classifier,
             SmallVectorImpl<Operation *> &copies) {
    if (members.size() < 2)
      return success();
    SmallVector<SmallVector<NpuControlPacketOp> *> tiles;
    for (auto &[tile, packets] : members)
      tiles.push_back(&packets);
    SmallVector<NpuControlPacketOp> &first = *tiles.front();
    AIEXDialect::CtrlPktMulticastIdAttrHelper stamp(&getContext());
    size_t baseLen = 0;
    for (; baseLen < first.size(); ++baseLen) {
      Write w = appliedWrite(first[baseLen], rowShift);
      bool common = llvm::all_of(llvm::drop_begin(tiles), [&](auto *packets) {
        return baseLen < packets->size() &&
               appliedWrite((*packets)[baseLen], rowShift) == w &&
               first[baseLen]->isBeforeInBlock((*packets)[baseLen]);
      });
      if (!common)
        break;
    }
    if (baseLen == 0)
      return success();
    bool spansColumns =
        members.begin()->first.first != std::prev(members.end())->first.first;
    bool ordered = false;
    if (spansColumns) {
      SmallVector<Classified> shared;
      for (size_t i = 0; i < baseLen; ++i) {
        FailureOr<Classified> c = classifier.classify(first[i]);
        if (failed(c))
          return failure();
        shared.push_back(std::move(*c));
      }
      llvm::DenseSet<Operation *> baseOps;
      for (auto *packets : tiles)
        for (size_t i = 0; i < baseLen; ++i)
          baseOps.insert((*packets)[i]);
      AIEXDialect::CtrlPktEnablePhaseAttrHelper enablePhase(&getContext());
      ordered = !drainMerge;
      for (NpuControlPacketOp packet : run) {
        if (ordered)
          break;
        if (baseOps.contains(packet) ||
            enablePhase.isAttrPresent(packet) != inEnablePhase ||
            !members.count({static_cast<int>(packet.getColumnFromAddr()),
                            static_cast<int>(packet.getRowFromAddr())}))
          continue;
        FailureOr<Classified> c = classifier.classify(packet);
        if (failed(c))
          return failure();
        ordered = llvm::any_of(
            shared, [&](const Classified &s) { return !s.commutesWith(*c); });
      }
    }
    AIEXDialect::CtrlPktBasePhaseAttrHelper basePhaseStamp(&getContext());
    for (size_t i = 0; i < baseLen; ++i) {
      stamp.setAttr(first[i],
                    IntegerAttr::get(IntegerType::get(&getContext(), 32), id));
      if (ordered)
        basePhaseStamp.setAttr(first[i], UnitAttr::get(&getContext()));
      for (auto *packets : llvm::drop_begin(tiles))
        copies.push_back((*packets)[i]);
    }
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
