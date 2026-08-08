//===- AIEVerifyPacketRules.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rejects an aie.packet_rules whose rules send a packet id the design actually
// produces to two different routes. A stream switch matches a slave port's
// rules in order and routes on the first hit, so such an id silently takes the
// earlier rule's route and the later rule never sees it.
//
// The check is deliberately restricted to *live* ids -- those some op in the
// device injects. Overlap on an id nothing produces is both normal and wanted:
// a slave port has only getNumSlaveSlots() slots, so the router relaxes masks
// to merge ids into one rule, and a relaxed rule claims ids no flow uses. Only
// ids that can arrive make an overlap observable.
//
// The live set is device-wide, which over-approximates per-port arrival: an id
// live only on some disjoint port still counts here. That direction is safe for
// the router, which never leaves a live id ambiguous on the port it arrives at,
// and it keeps the pass from having to re-derive routing.
//
// This lives in AIEX because a hand-routed design names its ids only in the
// runtime sequence (aiex.npu.dma_memcpy_nd / aiex.npu.writebd), which the AIE
// dialect cannot see. Run it after routing and before those ops are lowered.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEVERIFYPACKETRULES
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

struct AIEVerifyPacketRulesPass
    : xilinx::AIEX::impl::AIEVerifyPacketRulesBase<AIEVerifyPacketRulesPass> {

  // Packet ids the device can put on the stream. Missing a source only weakens
  // the check, so err towards leaving one out rather than guessing.
  llvm::SmallSet<int, 16> collectLiveIds(DeviceOp dev) {
    llvm::SmallSet<int, 16> live;
    dev.walk([&](Operation *op) {
      if (auto flow = dyn_cast<PacketFlowOp>(op))
        live.insert(flow.getID());
      else if (auto bd = dyn_cast<DMABDPACKETOp>(op))
        live.insert(bd.getPacketId());
      else if (auto bd = dyn_cast<DMABDOp>(op)) {
        if (auto pkt = bd.getPacket())
          live.insert(pkt->getPktId());
      } else if (auto alloc = dyn_cast<ShimDMAAllocationOp>(op)) {
        if (auto pkt = alloc.getPacket())
          live.insert(pkt->getPktId());
      } else if (auto memcpy = dyn_cast<NpuDmaMemcpyNdOp>(op)) {
        if (auto pkt = memcpy.getPacket())
          live.insert(pkt->getPktId());
      } else if (auto task = dyn_cast<DMAConfigureTaskOp>(op)) {
        if (auto pkt = task.getPacket())
          live.insert(pkt->getPktId());
      } else if (auto bd = dyn_cast<NpuWriteBdOp>(op)) {
        // Only reachable through aie-opt: the runtime sequence is still
        // memcpy-shaped when this pass runs inside aiecc.
        if (bd.getEnablePacket())
          live.insert(bd.getPacketId());
      } else if (auto tile = dyn_cast<TileOp>(op)) {
        // Control packets are addressed by the target tile's controller id.
        if (auto pkt = tile->getAttrOfType<PacketInfoAttr>("controller_id"))
          live.insert(pkt.getPktId());
      }
    });
    return live;
  }

  // Where an amsel actually sends a packet: the master ports of every
  // masterset naming an amsel with the same (arbiter, msel), since that pair is
  // all the switch matches on. Two rules whose destinations agree route alike
  // and may overlap freely -- one masterset can list several amsels, and
  // distinct AMSelOps can spell the same pair.
  static SmallVector<Port, 4> destPorts(PacketRuleOp rule) {
    SmallVector<Port, 4> dests;
    auto amsel = rule.getAmsel().getDefiningOp<AMSelOp>();
    if (!amsel)
      return dests;
    Block *sb = rule->getParentOp()->getBlock();
    for (auto ms : sb->getOps<MasterSetOp>())
      for (Value v : ms.getAmsels()) {
        auto other = v.getDefiningOp<AMSelOp>();
        if (other && other.arbiterIndex() == amsel.arbiterIndex() &&
            other.getMselValue() == amsel.getMselValue()) {
          dests.push_back(ms.destPort());
          break;
        }
      }
    llvm::sort(dests);
    return dests;
  }

  static bool sameRoute(PacketRuleOp a, PacketRuleOp b) {
    if (a.getAmsel() == b.getAmsel())
      return true;
    SmallVector<Port, 4> da = destPorts(a), db = destPorts(b);
    // An unresolvable amsel yields no destinations; do not call that a match.
    return !da.empty() && da == db;
  }

  void runOnOperation() override {
    DeviceOp dev = getOperation();
    llvm::SmallSet<int, 16> live = collectLiveIds(dev);
    if (live.empty())
      return;

    SmallVector<int, 16> liveSorted(live.begin(), live.end());
    llvm::sort(liveSorted);

    WalkResult wr = dev.walk([&](PacketRulesOp rulesOp) -> WalkResult {
      Region &body = rulesOp.getRules();
      if (body.empty())
        return WalkResult::advance();
      SmallVector<PacketRuleOp, 4> rules(body.front().getOps<PacketRuleOp>());
      for (size_t i = 0; i < rules.size(); i++)
        for (size_t j = i + 1; j < rules.size(); j++) {
          if (sameRoute(rules[i], rules[j]))
            continue;
          int maskI = rules[i].maskInt(), valueI = rules[i].valueInt();
          int maskJ = rules[j].maskInt(), valueJ = rules[j].valueInt();
          for (int id : liveSorted) {
            if ((id & maskI) != valueI || (id & maskJ) != valueJ)
              continue;
            rules[j].emitOpError("is shadowed for packet id ")
                << id
                << ": an earlier rule matches it too, and the switch routes "
                   "on the first match";
            rules[i].emitRemark("this is the rule that claims packet id ")
                << id;
            return WalkResult::interrupt();
          }
        }
      return WalkResult::advance();
    });

    if (wr.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIEVerifyPacketRulesPass() {
  return std::make_unique<AIEVerifyPacketRulesPass>();
}
