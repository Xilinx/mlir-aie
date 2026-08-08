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
      else if (auto bd = dyn_cast<DMABDOp>(op)) {
        if (auto pkt = bd.getPacket())
          live.insert(pkt->getPktId());
      } else if (auto memcpy = dyn_cast<NpuDmaMemcpyNdOp>(op)) {
        if (auto pkt = memcpy.getPacket())
          live.insert(pkt->getPktId());
      } else if (auto bd = dyn_cast<NpuWriteBdOp>(op)) {
        if (bd.getEnablePacket())
          live.insert(bd.getPacketId());
      }
    });
    return live;
  }

  // Two rules route alike, and so may overlap freely, when their amsels name
  // the same (arbiter, msel). Distinct AMSelOps can spell the same pair.
  static bool sameRoute(PacketRuleOp a, PacketRuleOp b) {
    if (a.getAmsel() == b.getAmsel())
      return true;
    auto amselA = a.getAmsel().getDefiningOp<AMSelOp>();
    auto amselB = b.getAmsel().getDefiningOp<AMSelOp>();
    return amselA && amselB && amselA.arbiterIndex() == amselB.arbiterIndex() &&
           amselA.getMselValue() == amselB.getMselValue();
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
