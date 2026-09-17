//===- AIESortControlPacketsByTile.cpp -------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/CtrlPktUtils.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIESORTCONTROLPACKETSBYTILE
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {
struct AIESortControlPacketsByTilePass
    : xilinx::AIEX::impl::AIESortControlPacketsByTileBase<
          AIESortControlPacketsByTilePass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    // Sorting groups every packet destined for the same tile into one
    // contiguous run, so the downstream AIECtrlPacketToDma pass can pack each
    // tile's run into a single linear shim DMA (one BD per tile) instead of
    // one BD per packet. This is behavior-preserving: the writeset of a
    // control-packet run is invariant to the delivery order across DISTINCT
    // tiles (each tile only observes writes addressed to itself), and
    // std::stable_sort preserves the original relative order of packets that
    // target the SAME tile -- so no tile ever sees its own packets reordered.
    auto byTile = [](NpuControlPacketOp a, NpuControlPacketOp b) {
      int ca = a.getColumnFromAddr(), cb = b.getColumnFromAddr();
      if (ca != cb)
        return ca < cb;
      return a.getRowFromAddr() < b.getRowFromAddr();
    };

    // Stable-sort a contiguous run of control packets by destination tile and
    // re-thread the block in the new order. When partitionAtEnable is set, the
    // run is split at the first core-enable (enables stay a trailing phase);
    // teardown runs pass false (no enables, single phase).
    auto sortRun = [&](SmallVector<NpuControlPacketOp> &run,
                       bool partitionAtEnable) {
      if (run.size() < 2)
        return;
      Operation *anchor = run.front()->getPrevNode(); // original run position
      Block *blk = run.front()->getBlock();
      size_t bnd = run.size();
      if (partitionAtEnable)
        for (size_t i = 0; i < run.size(); ++i)
          if (isCoreEnableControlPacket(run[i])) {
            bnd = i;
            break;
          }
      std::stable_sort(run.begin(), run.begin() + bnd, byTile);
      std::stable_sort(run.begin() + bnd, run.end(), byTile);
      for (NpuControlPacketOp cp : run) {
        if (anchor)
          cp->moveAfter(anchor);
        else
          cp->moveBefore(blk, blk->begin());
        anchor = cp;
      }
    };

    for (auto seq : device.getOps<AIE::RuntimeSequenceOp>()) {
      // RuntimeSequenceOp is NoTerminator, so an empty sequence body is valid
      // IR -- and a body written as `{ }` parses to a region with NO blocks
      // at all (not a block with zero ops), so `Region::front()` would be UB.
      // Guard that first, then also guard the (in-principle-possible) case
      // of one block with zero ops, since `Block::front()` on that is UB too.
      // This loop runs over every runtime sequence on the device, not just
      // ctrlpkt ones, so it must tolerate both cases.
      if (seq.getBody().empty())
        continue;
      Block &entry = seq.getBody().front();

      // Collect every maximal contiguous run of control packets. For a real
      // ctrlpkt config sequence there are two: the leading config(+enable) run
      // and, after the app-run, the trailing teardown run.
      SmallVector<SmallVector<NpuControlPacketOp>> runs;
      if (entry.empty())
        continue;
      Operation *o = &entry.front();
      while (o) {
        if (!isa<NpuControlPacketOp>(o)) {
          o = o->getNextNode();
          continue;
        }
        SmallVector<NpuControlPacketOp> run;
        while (o && isa<NpuControlPacketOp>(o)) {
          run.push_back(cast<NpuControlPacketOp>(o));
          o = o->getNextNode();
        }
        runs.push_back(std::move(run));
      }

      if (runs.empty())
        continue;
      // Only the first (config+enable) and last (teardown) runs are sorted.
      // A hypothetical middle run (neither) would be left un-coalesced by
      // AIECtrlPacketToDma -- a throughput miss only, still correct.
      sortRun(runs.front(), /*partitionAtEnable=*/true); // config + enable
      if (runs.size() >= 2)
        sortRun(runs.back(), /*partitionAtEnable=*/false); // teardown
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
xilinx::AIEX::createAIESortControlPacketsByTilePass() {
  return std::make_unique<AIESortControlPacketsByTilePass>();
}
