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

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"

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
  // A multicast group, as the device whose configuration the packets carry, the
  // column, and the multicast id.
  using GroupKey = std::tuple<StringRef, int, int64_t>;

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
      int col = packet.getColumnFromAddr();
      auto tile =
          it->second.find({col, static_cast<int>(packet.getRowFromAddr())});
      if (tile == it->second.end())
        return std::nullopt;
      size_t size = llvm::count_if(it->second, [&](const auto &entry) {
        return entry.first.first == col && entry.second == tile->second;
      });
      return std::make_pair(tile->second, size);
    };

    for (auto device : module.getOps<AIE::DeviceOp>()) {
      uint32_t rowShift = device.getTargetModel().getRowShift();
      for (auto seq : device.getOps<AIE::RuntimeSequenceOp>())
        seq.walk([&](Block *block) {
          SmallVector<NpuControlPacketOp> run;
          for (Operation &op : *block) {
            if (auto packet = dyn_cast<NpuControlPacketOp>(op)) {
              run.push_back(packet);
              continue;
            }
            dedupRun(run, idOf, rowShift);
            run.clear();
          }
          dedupRun(run, idOf, rowShift);
        });
    }
  }

  // Dedups each group within one run, the enable phase apart from the rest.
  template <typename IdOf>
  void dedupRun(ArrayRef<NpuControlPacketOp> run, IdOf &idOf,
                uint32_t rowShift) {
    AIEXDialect::CtrlPktDeviceAttrHelper deviceStamp(&getContext());
    AIEXDialect::CtrlPktEnablePhaseAttrHelper enablePhase(&getContext());
    // (group, enable phase) -> row -> the row's packets in order.
    std::map<std::pair<GroupKey, bool>,
             std::map<int, SmallVector<NpuControlPacketOp>>>
        members;
    std::map<GroupKey, size_t> sizes;
    for (NpuControlPacketOp packet : run) {
      auto id = idOf(packet);
      if (!id)
        continue;
      GroupKey key{deviceStamp.getAttr(packet).getValue(),
                   static_cast<int>(packet.getColumnFromAddr()), id->first};
      sizes[key] = id->second;
      members[{key, enablePhase.isAttrPresent(packet)}]
             [static_cast<int>(packet.getRowFromAddr())]
                 .push_back(packet);
    }
    for (auto &[key, rows] : members)
      if (rows.size() == sizes[key.first])
        dedupGroup(rows, std::get<2>(key.first), rowShift);
  }

  // The leading packets every tile of the group has at the same position are
  // sent once, from the lowest tile; each tile still applies the same writes
  // in order. Every tile must have packets, since the multicast reaches all.
  void dedupGroup(std::map<int, SmallVector<NpuControlPacketOp>> &rows,
                  int64_t id, uint32_t rowShift) {
    if (rows.size() < 2)
      return;
    SmallVector<SmallVector<NpuControlPacketOp> *> tiles;
    for (auto &[row, packets] : rows)
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
    for (size_t i = 0; i < baseLen; ++i) {
      stamp.setAttr(first[i],
                    IntegerAttr::get(IntegerType::get(&getContext(), 32), id));
      for (auto *packets : llvm::drop_begin(tiles))
        (*packets)[i]->erase();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIEX::createAIEDedupCtrlPacketMulticastPass() {
  return std::make_unique<AIEDedupCtrlPacketMulticastPass>();
}
