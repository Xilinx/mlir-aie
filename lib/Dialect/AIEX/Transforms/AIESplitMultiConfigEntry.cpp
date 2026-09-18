//===- AIESplitMultiConfigEntry.cpp -----------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// On the `aiex.entrypoint`-marked host device of a `--reconfig-method` fold,
// this pass synthesizes the shared standup `init` sequence and normalizes each
// per-config entrypoint's load_pdi re-arm, according to the delivery method
// recorded in the marker's `reconfig_method` key (loadpdi | write32 | ctrlpkt).
//
// It runs AFTER the per-device DMA lowering (so load_pdi ops are materialized
// and, for ctrlpkt, AIECtrlPacketToDma has appended the trailing ctrl-pkt
// stream buffer as each entrypoint's last block arg) and after PDI-id
// assignment -- distinct from `aie-split-configure-entries`, which runs earlier
// on the pre-materialize `aiex.configure` monolith.
//
// The method drives three load_pdi shapes:
//   * ctrlpkt / write32 (expectInit): each entrypoint carries exactly one
//     load_pdi post-expansion; a single shared `init` is synthesized (from the
//     overlay for ctrlpkt, from the truncated first entrypoint for write32) and
//     every entrypoint's own load_pdi re-arm is stripped (the in-band
//     self-clear supplies each per-config reset).
//   * loadpdi (loadPdiNoInit): no shared `init`; each entrypoint keeps its sole
//     self-reset load_pdi (a load_pdi is a full reset every time it is
//     applied).
// An unexpected load_pdi count for the mode fails loud rather than silently
// mis-splitting. A no-op for a module with no marked entry device.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIESPLITMULTICONFIGENTRY
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-split-multi-config-entry"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

// Split the marked entry device's entrypoint sequences per the method-derived
// policy. Mutates `dev` in place. Returns failure (with an emitted diagnostic)
// on any out-of-contract input.
static LogicalResult splitMarkedEntryDevice(DeviceOp dev, bool stripRearm,
                                            bool expectInit, bool loadPdiNoInit,
                                            bool ctrlPkt) {
  // Entrypoint sequences in block order (= chain order). An init method
  // (expectInit) carries exactly one load_pdi per entrypoint post-expansion, so
  // select the load_pdi-carrying config entrypoints; every other mode takes
  // every non-empty entrypoint. The per-mode load_pdi count is validated below.
  // Entrypoint NAMES ARE KEPT VERBATIM: the sym_name IS the dispatch name
  // (main:<name>), chosen by the design; the toolchain never renames them.
  SmallVector<RuntimeSequenceOp> seqs;
  for (RuntimeSequenceOp s : dev.getOps<RuntimeSequenceOp>()) {
    if (s.getBody().empty())
      continue;
    if (expectInit) {
      if (llvm::any_of(s.getBody().front(), [](Operation &op) {
            return llvm::isa<NpuLoadPdiOp>(op);
          }))
        seqs.push_back(s);
    } else {
      seqs.push_back(s);
    }
  }
  if (seqs.empty())
    return success();

  // Loud-fail on duplicate entrypoint names within the marked device -- the
  // multi-design fold names each entrypoint with its design's name=, and two
  // designs sharing a name would silently collide on one dispatch kernel.
  {
    llvm::StringSet<> seen;
    for (RuntimeSequenceOp s : seqs)
      if (!seen.insert(s.getSymName()).second)
        return s.emitError() << "duplicate entry sequence name '"
                             << s.getSymName() << "' -- set name= per design";
  }

  // Validate the per-mode load_pdi count on every selected entrypoint.
  for (RuntimeSequenceOp s : seqs) {
    unsigned nLoadPdi = 0;
    for (Operation &op : s.getBody().front())
      if (llvm::isa<NpuLoadPdiOp>(op))
        ++nLoadPdi;
    if (expectInit) {
      if (nLoadPdi != 1)
        return s.emitError() << "expected exactly one load_pdi in runtime "
                                "sequence '"
                             << s.getSymName() << "', found " << nLoadPdi;
    } else if (loadPdiNoInit) {
      if (nLoadPdi != 1)
        return s.emitError()
               << "expected exactly one load_pdi in runtime sequence '"
               << s.getSymName() << "' (no-init loadpdi), found " << nLoadPdi;
    } else if (nLoadPdi != 0) {
      // Defensive guard for a future no-load_pdi mode (no current method
      // routes here).
      return s.emitError() << "expected zero load_pdi in runtime sequence '"
                           << s.getSymName() << "', found " << nLoadPdi;
    }
  }

  if (expectInit) {
    RuntimeSequenceOp first = seqs.front();
    OpBuilder builder(first);
    builder.setInsertionPoint(first);
    if (ctrlPkt) {
      // ctrlpkt: synthesize ONE shared `init` from the overlay, design-
      // independent. Its signature is only the uniform trailing ctrl-pkt-stream
      // buffer that AIECtrlPacketToDma appended as the LAST block arg of every
      // entrypoint; its body is that entrypoint's own load_pdi (validated
      // exactly-one above), cloned so its device_ref/id/expand_mode come along.
      if (first.getBody().getArguments().empty())
        return first.emitError()
               << "ctrlpkt entry sequence '" << first.getSymName()
               << "' has no block arguments -- expected a trailing ctrl-pkt "
                  "stream buffer appended by aie-ctrl-packet-to-dma; run this "
                  "pass after DMA lowering";
      Type ctrlArgType = first.getBody().getArguments().back().getType();
      NpuLoadPdiOp firstLoadPdi;
      for (Operation &op : first.getBody().front())
        if (auto loadPdi = llvm::dyn_cast<NpuLoadPdiOp>(op)) {
          firstLoadPdi = loadPdi;
          break;
        }
      assert(firstLoadPdi &&
             "expectInit validated exactly one load_pdi per entrypoint above");

      auto initSeq = RuntimeSequenceOp::create(builder, first.getLoc(),
                                               StringAttr{}, BoolAttr{});
      initSeq.setSymName("init");
      initSeq.getBody().push_back(new Block);
      initSeq.getBody().addArgument(ctrlArgType, first.getLoc());
      builder.setInsertionPointToStart(&initSeq.getBody().front());
      builder.clone(*firstLoadPdi);
    } else {
      // write32: no ctrl-pkt lowering ran, so there is no uniform trailing arg
      // to key off of. Clone the first entrypoint and truncate right after its
      // first load_pdi -- the shared `init` keeps the sole @empty reset standup
      // and streams nothing.
      auto initSeq = cast<RuntimeSequenceOp>(builder.clone(*first));
      initSeq.setSymName("init");
      Block &b = initSeq.getBody().front();
      bool afterLoadPdi = false;
      SmallVector<Operation *> toErase;
      for (Operation &op : b) {
        if (afterLoadPdi)
          toErase.push_back(&op);
        else if (llvm::isa<NpuLoadPdiOp>(op))
          afterLoadPdi = true;
      }
      for (Operation *op : llvm::reverse(toErase))
        op->erase();
    }
  }

  // `stripRearm` STRIPS every per-entrypoint load_pdi re-arm: the `init`
  // synthesized above keeps the sole real standup load_pdi and the entrypoints
  // stay load_pdi-free (the in-band self-clear supplies each per-config reset).
  // Otherwise each entrypoint keeps its own load_pdi so a separately-dispatched
  // config re-arms via a PDI reload. Names are untouched.
  if (stripRearm)
    for (RuntimeSequenceOp s : seqs) {
      Block &b = s.getBody().front();
      for (Operation &op : llvm::make_early_inc_range(b))
        if (llvm::isa<NpuLoadPdiOp>(op))
          op.erase();
    }
  return success();
}

struct AIESplitMultiConfigEntryPass
    : public xilinx::AIEX::impl::AIESplitMultiConfigEntryBase<
          AIESplitMultiConfigEntryPass> {
  using AIESplitMultiConfigEntryBase::AIESplitMultiConfigEntryBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<memref::MemRefDialect, AIE::AIEDialect, AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The entry device carries the aiex.entrypoint marker (kEntrypointAttr) --
    // the SOLE discriminator. Its dictionary records the reconfig_method
    // spelling, from which the split policy is derived. No marker -> no-op.
    for (DeviceOp dev : module.getOps<DeviceOp>()) {
      auto marker = dev->getAttrOfType<DictionaryAttr>(kEntrypointAttr);
      if (!marker)
        continue;

      StringRef method;
      if (auto m = marker.getAs<StringAttr>(kReconfigMethodKey))
        method = m.getValue();
      const bool ctrlPkt = method == "ctrlpkt";
      const bool write32 = method == "write32";
      const bool loadPdiNoInit = method == "loadpdi";
      if (!ctrlPkt && !write32 && !loadPdiNoInit) {
        dev->emitError() << "aie-split-multi-config-entry: unrecognized "
                         << kReconfigMethodKey << " '" << method
                         << "' on the entrypoint marker (expected "
                            "loadpdi|write32|ctrlpkt)";
        signalPassFailure();
        return;
      }
      // ctrlpkt and write32 both synthesize a shared `init` and strip each
      // entrypoint's load_pdi re-arm; loadpdi keeps per-config self-resets.
      const bool expectInit = ctrlPkt || write32;
      if (failed(splitMarkedEntryDevice(dev, /*stripRearm=*/expectInit,
                                        expectInit, loadPdiNoInit, ctrlPkt))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIESplitMultiConfigEntryPass() {
  return std::make_unique<AIESplitMultiConfigEntryPass>();
}
