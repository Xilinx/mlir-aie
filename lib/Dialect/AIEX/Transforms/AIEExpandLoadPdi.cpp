//===- AIEExpandLoadPdi.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands `npu.load_pdi` operations that reference a device. There
// are two output modes, controlled by the `ctrl-pkt` option:
//
// 1. Default (ctrl-pkt=false): replaces each `load_pdi @device` with
//    a. an empty device PDI load (`load_pdi @empty_N`), which causes the
//       firmware to reset the device, and
//    b. explicit `aiex.npu.write32`/`aiex.npu.blockwrite` configuration ops.
// 2. With ctrl-pkt=true: replaces each `load_pdi @device` with
//    a. a `load_pdi @ctrl_pkt_overlay`, which configures the NPU to stream
//       further configuration as control packets, and
//    b. a sequence of `aiex.npu.control_packet` ops carrying the device's
//       configuration.
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <tuple>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEEXPANDLOADPDI
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-expand-load-pdi"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;
using namespace xilinx::AIE;

namespace {

// Name of the overlay device loaded ahead of streaming configuration as
// control packets. The device is expected to exist in the module (e.g.,
// emitted by the `aie-generate-column-control-overlay` pass).
static constexpr llvm::StringLiteral kCtrlPktOverlayName = "ctrl_pkt_overlay";
// Name of an alternating copy of the overlay device, used to avoid PDI
// address caching when the same overlay would otherwise be loaded twice in a
// row (the firmware would treat the second load as a no-op).
static constexpr llvm::StringLiteral kCtrlPktOverlayCopyName =
    "ctrl_pkt_overlay_copy";

// Look up `kCtrlPktOverlayName` and return a clone of it named
// `kCtrlPktOverlayCopyName`, creating it once if needed. Returns nullptr on
// error.
static AIE::DeviceOp getOrCreateCtrlPktOverlayCopy(ModuleOp moduleOp,
                                                   Operation *errLocOp) {
  if (auto existing =
          moduleOp.lookupSymbol<AIE::DeviceOp>(kCtrlPktOverlayCopyName))
    return existing;

  auto orig = moduleOp.lookupSymbol<AIE::DeviceOp>(kCtrlPktOverlayName);
  if (!orig) {
    errLocOp->emitError("ctrl-pkt mode requires a `@")
        << kCtrlPktOverlayName << "` device in the module";
    return nullptr;
  }

  OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointAfter(orig);
  auto *cloned = builder.clone(*orig.getOperation());
  auto clonedDev = cast<AIE::DeviceOp>(cloned);
  clonedDev.setSymName(kCtrlPktOverlayCopyName);
  return clonedDev;
}

// The empty device whose PDI load makes the firmware reset the array. `parity`
// alternates so that two consecutive loads never name the same PDI (see
// transformLoadPdi).
static AIE::DeviceOp getOrCreateEmptyDevice(ModuleOp moduleOp,
                                            AIE::AIEDevice deviceType,
                                            unsigned parity) {
  std::string emptyName = "empty_" + std::to_string(parity);
  if (auto existing = moduleOp.lookupSymbol<AIE::DeviceOp>(emptyName))
    return existing;

  OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto loc = builder.getUnknownLoc();
  auto emptyDevice = AIE::DeviceOp::create(builder, loc, deviceType,
                                           builder.getStringAttr(emptyName));
  emptyDevice.getRegion().emplaceBlock();
  builder.setInsertionPointToEnd(&emptyDevice.getRegion().front());
  AIE::EndOp::create(builder, loc);
  return emptyDevice;
}

// Helper to transform a single load_pdi operation
static LogicalResult
transformLoadPdi(NpuLoadPdiOp loadPdiOp, ModuleOp moduleOp, unsigned index,
                 AIEX::ExpandMode defaultMode, bool resetFree, bool selfClear,
                 bool withReset, unsigned loadPdisInBlock) {
  OpBuilder builder(loadPdiOp);
  // The three self-clear teardowns (switch, circuit, DMA) are no longer
  // independently selectable: self-clear (now unconditional for
  // ctrlpkt/write32) emits the complete protocol. Each generator stays
  // demand-scoped (empty when the config uses none of that resource class), so
  // the unconditional behavior only widens WHICH teardowns are attempted, not
  // whether an unused one fires.
  bool selfClearCircuit = selfClear;
  bool selfClearDma = selfClear;

  // Only process load_pdi ops that reference a device
  auto deviceRefAttr = loadPdiOp.getDeviceRefAttr();
  if (!deviceRefAttr) {
    return success();
  }

  // Per-op annotation takes precedence; fall back to pass default
  AIEX::ExpandMode mode = loadPdiOp.getExpandMode().value_or(defaultMode);
  if (mode == AIEX::ExpandMode::none)
    return success();
  bool ctrlPkt = (mode == AIEX::ExpandMode::ctrlpkt);
  // ctrlpkt keeps the resident @ctrl_pkt_overlay preload and skips the
  // overlay's own switch writes. The reset-free policy rides write32 delivery
  // (no resident overlay), so it never uses the overlay here.
  bool useOverlay = ctrlPkt;

  auto referencedDevice = moduleOp.lookupSymbol<AIE::DeviceOp>(deviceRefAttr);
  if (!referencedDevice) {
    loadPdiOp.emitError("Referenced symbol '")
        << deviceRefAttr.getValue() << "' is not a device";
    return failure();
  }

  // The reset-free policy without with-reset skips the init preload entirely:
  // the firmware resets the partition on context teardown, so there is no
  // @empty reset to (re-)establish at the start of a config. with-reset
  // restores the reset behavior (preload @empty, like plain write32).
  bool skipPreload = resetFree && !withReset;

  FlatSymbolRefAttr preloadRef;
  if (useOverlay) {
    // Overlay device PDI
    // Alternate between the original overlay and a clone of it on every
    // other load. Loading the same PDI twice in a row gets cached by the
    // firmware (the second load becomes a no-op), so we need two distinct
    // PDI addresses that carry the same overlay configuration.
    StringRef overlayName =
        (index % 2 == 0) ? kCtrlPktOverlayName : kCtrlPktOverlayCopyName;
    if (index % 2 != 0) {
      AIE::DeviceOp copy =
          getOrCreateCtrlPktOverlayCopy(moduleOp, loadPdiOp.getOperation());
      if (!copy)
        return failure();
    } else if (!moduleOp.lookupSymbol<AIE::DeviceOp>(kCtrlPktOverlayName)) {
      loadPdiOp.emitError("overlay expand mode requires a `@")
          << kCtrlPktOverlayName << "` device in the module";
      return failure();
    }
    preloadRef = FlatSymbolRefAttr::get(builder.getContext(), overlayName);
  } else if (!skipPreload) {
    // Empty device PDI (triggers firmware reset)
    AIE::DeviceOp emptyDevice = getOrCreateEmptyDevice(
        moduleOp, referencedDevice.getDevice(), index % 2);
    preloadRef = FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr());
  }

  builder.setInsertionPoint(loadPdiOp);

  // Emit the preload load_pdi (either empty-device reset or ctrl_pkt_overlay).
  if (useOverlay) {
    NpuLoadPdiOp::create(builder, loadPdiOp.getLoc(), preloadRef,
                         /*id=*/nullptr, /*size=*/nullptr,
                         /*address=*/nullptr,
                         /*expand_mode=*/
                         AIEX::ExpandModeAttr::get(builder.getContext(),
                                                   AIEX::ExpandMode::none));
  } else if (!skipPreload) {
    NpuLoadPdiOp::create(builder, loadPdiOp.getLoc(), preloadRef,
                         loadPdiOp.getIdAttr(), loadPdiOp.getSizeAttr(),
                         loadPdiOp.getAddressAttr(),
                         /*expand_mode=*/
                         AIEX::ExpandModeAttr::get(builder.getContext(),
                                                   AIEX::ExpandMode::none));
  }

  // Step 2: generate and insert configuration ops. Only ctrlpkt emits control
  // packets; write32 delivery (plain and reset-free) emits direct
  // write32/blockwrite ops. skipCtrlPktOverlay follows the resident-overlay
  // preload (only ctrlpkt keeps it and skips the overlay's own switch writes;
  // write32 resets to empty and must include them).
  auto outputType = ctrlPkt ? AIEToConfigurationOutputType::ControlPacket
                            : AIEToConfigurationOutputType::Transaction;
  std::string prefix =
      ctrlPkt     ? ("loadpdi_ctrlpkt_" + std::to_string(index) + "_")
      : resetFree ? ("loadpdi_write32_" + std::to_string(index) + "_")
                  : ("loadpdi_" + std::to_string(index));
  if (failed(xilinx::AIE::generateAndInsertConfigOps(
          builder, referencedDevice, /*clElfDir=*/"", outputType, prefix,
          /*skipCtrlPktOverlay=*/useOverlay))) {
    loadPdiOp.emitError("Failed to generate configuration operations");
    return failure();
  }

  // Self-clear epilogue: once this config's data completes, disable the ports
  // it enabled that the overlay does not own (config ports minus overlay
  // ports). Under ctrl-pkt this restores the control-plane-only state
  // main:init left behind, since the resident overlay only ever holds
  // control ports, and it stops the stream-switch arbiter bindings from
  // accruing across reconfigurations. Under the reset-free policy there is no
  // resident overlay to restore to (init leaves an @empty reset instead), so
  // this is deadlock-avoidance rather than a pristine restore: it tears down
  // the config's own data ports so the next config's direct writes do not
  // wedge on state this config left enabled. The disables ride the same
  // transport as the config: control packets on the resident control network
  // under ctrl-pkt (fully in-band, no per-config load_pdi re-arm and no
  // privileged reset), or write32/blockwrite direct writes under the reset-free
  // policy (the out-of-band arm delivers config AND self-clear OOB). Empty (a
  // no-op) for non-switch reconfigs, where the config enables no
  // exclusively-data ports.
  if ((useOverlay || resetFree) && selfClear) {
    // Precondition: the epilogue below finds "the last NpuDmaWaitOp in this
    // config's runtime-sequence block" and assumes that block holds exactly
    // one load_pdi (this one). With >1 load_pdi sharing a block, "last wait
    // in the block" would grab a wait belonging to a different config's data,
    // silently misplacing the teardown. Fail loudly instead of guessing.
    if (loadPdisInBlock > 1) {
      loadPdiOp.emitError(
          "ctrl-pkt self-clear requires exactly one load_pdi per "
          "runtime-sequence block, found ")
          << loadPdisInBlock;
      return failure();
    }

    // Insert the teardown after the last dma_wait in this config's runtime
    // sequence (the block also holds its dma_memcpy_nd and dma_wait), i.e. once
    // the transfer completes, so each config only undoes its own state: O(1)
    // per config and order-independent, with no cross-config union.
    mlir::Block *seqBlock = loadPdiOp->getBlock();
    AIEX::NpuDmaWaitOp lastWait;
    for (auto waitOp : seqBlock->getOps<AIEX::NpuDmaWaitOp>())
      lastWait = waitOp;

    OpBuilder::InsertionGuard guard(builder);
    if (lastWait)
      builder.setInsertionPointAfter(lastWait);
    else
      builder.setInsertionPointToEnd(seqBlock);

    // Route the teardown through the same transport as the config (outputType,
    // computed above): control packets under ctrl-pkt (in-band), direct writes
    // under the reset-free policy (OOB).

    // Switch teardown: disable the exclusively-data ports the config enabled
    // (config ports minus the overlay's), so the resident overlay does not
    // accrue stream-switch bindings across reconfigurations. Under the
    // reset-free policy there is no resident overlay device at all (the whole
    // point of the no-overlay arm), so there is nothing to carve out of the
    // exclude set:
    // leave it empty and disable every packet-switch port this config's own
    // connect enabled -- still a whole-array-safe teardown, scoped to exactly
    // this config's used data ports.
    if (selfClear) {
      llvm::DenseSet<std::tuple<int, int, int, int, int>> excludePorts;
      auto overlayDev =
          moduleOp.lookupSymbol<AIE::DeviceOp>(kCtrlPktOverlayName);
      if (overlayDev) {
        // Exclude every master/slave packet-switch port the overlay uses
        // (both its control-only ports and the ports it shares with data),
        // keyed by (col, row, bundle, index, isSlave), leaving the
        // exclusively-data ports.
        for (auto sb : overlayDev.getOps<AIE::SwitchboxOp>()) {
          int col = sb.colIndex();
          int row = sb.rowIndex();
          mlir::Block &conns = sb.getConnections().front();
          for (auto ms : conns.getOps<AIE::MasterSetOp>())
            excludePorts.insert(
                std::make_tuple(col, row, static_cast<int>(ms.getDestBundle()),
                                ms.destIndex(), 0));
          for (auto pr : conns.getOps<AIE::PacketRulesOp>())
            excludePorts.insert(std::make_tuple(
                col, row, static_cast<int>(pr.getSourceBundle()),
                pr.sourceIndex(), 1));
        }
      }
      if (failed(xilinx::AIE::generateAndInsertSwitchDisableOps(
              builder, referencedDevice, excludePorts, outputType,
              "selfclear_disable_" + std::to_string(index) + "_",
              selfClearCircuit))) {
        loadPdiOp.emitError("Failed to generate self-clear switch-disable ops");
        return failure();
      }
    }

    // DMA teardown: reset the config's active non-shim DMA channels so a
    // busy/enqueued channel is drained before the next config reconfigures it.
    if (selfClearDma &&
        failed(xilinx::AIE::generateAndInsertDmaChannelResetOps(
            builder, referencedDevice, outputType,
            "selfclear_dma_reset_" + std::to_string(index) + "_"))) {
      loadPdiOp.emitError("Failed to generate self-clear DMA reset ops");
      return failure();
    }
  }

  // Erase the original load_pdi operation
  loadPdiOp.erase();

  return success();
}

struct AIEExpandLoadPdiPass
    : public xilinx::AIEX::impl::AIEExpandLoadPdiBase<AIEExpandLoadPdiPass> {
  using AIEExpandLoadPdiBase::AIEExpandLoadPdiBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<memref::MemRefDialect, AIE::AIEDialect, AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Collect all load_pdi operations in program order;
    // need to collect once, then transform all collected ops;
    // since the transform inserts a new preload load_pdi, we can't transform
    // as we walk or it'd infinitely recurse.
    SmallVector<NpuLoadPdiOp> loadPdiOps;

    module.walk(
        [&](NpuLoadPdiOp loadPdiOp) { loadPdiOps.push_back(loadPdiOp); });

    // Per enclosing runtime sequence: the parity of its FIRST and LAST reset,
    // plus the device type to reset. `transformLoadPdi` picks the empty device
    // as `index % 2` over the module-wide op order, and that order includes
    // ops this pass skips, so a sequence's first reset is not necessarily
    // `@empty_0` and its parity cannot be recovered from a count. Captured
    // BEFORE transforming, which erases the ops.
    struct ResetParity {
      unsigned firstParity = 0;
      unsigned lastParity = 0;
      bool seen = false;
      AIE::AIEDevice device = {};
    };
    // Map the pass bool options to the ExpandMode enum: ctrlpkt keeps the
    // resident overlay, otherwise write32 delivery. The reset-free arm is a
    // write32 variant selected by clResetFree (not a distinct mode); it resets
    // to @empty only under with-reset and otherwise skips the preload entirely.
    AIEX::ExpandMode defaultMode =
        clCtrlPkt ? AIEX::ExpandMode::ctrlpkt : AIEX::ExpandMode::write32;

    // Classify which load_pdis are @empty-reset configs needing parity
    // alternation. Plain write32 (--expand-load-pdis) preloads @empty per
    // config and qualifies. The reset-free arm never resets to @empty per
    // config (with-reset aside, it still gets no trailing append -- matching
    // the pre-consolidation reset-free mode, which was likewise excluded here),
    // so clResetFree drops out of this classification entirely.
    llvm::MapVector<AIE::RuntimeSequenceOp, ResetParity> resetsPerSequence;
    for (auto [index, loadPdiOp] : llvm::enumerate(loadPdiOps)) {
      auto deviceRefAttr = loadPdiOp.getDeviceRefAttr();
      if (!deviceRefAttr)
        continue;
      if (clResetFree || loadPdiOp.getExpandMode().value_or(defaultMode) !=
                             AIEX::ExpandMode::write32)
        continue;
      auto seq = loadPdiOp->getParentOfType<AIE::RuntimeSequenceOp>();
      auto dev = module.lookupSymbol<AIE::DeviceOp>(deviceRefAttr);
      if (!seq || !dev)
        continue;
      auto &entry = resetsPerSequence[seq];
      if (!entry.seen) {
        entry.firstParity = index % 2;
        entry.seen = true;
      }
      entry.lastParity = index % 2;
      entry.device = dev.getDevice();
    }

    // Pre-count load_pdi ops per block (before any op is erased/inserted by
    // the transform below) so the self-clear path can check its one-load_pdi-
    // per-runtime-sequence-block assumption.
    llvm::DenseMap<mlir::Block *, unsigned> loadPdisPerBlock;
    for (auto loadPdiOp : loadPdiOps)
      loadPdisPerBlock[loadPdiOp->getBlock()]++;

    // Transform load_pdi ops
    unsigned idx = 0;
    for (auto loadPdiOp : loadPdiOps) {
      if (failed(transformLoadPdi(loadPdiOp, module, idx, defaultMode,
                                  clResetFree, clSelfClear, clWithReset,
                                  loadPdisPerBlock[loadPdiOp->getBlock()]))) {
        signalPassFailure();
        return;
      }
      idx++;
    }

    // The `index % 2` alternation above keeps two consecutive resets on
    // different PDIs WITHIN a sequence. But the host re-runs the whole sequence
    // on every dispatch, so the alternation has to hold across that boundary
    // too: when a sequence's last reset lands on the same empty PDI as its
    // first, the firmware caches the address and no-ops the next dispatch's
    // first load, and the configuration that follows lands on a device that was
    // never reset. That is silent -- the first dispatch is correct and later
    // ones return non-deterministic garbage.
    //
    // Append one more reset, of the parity opposite the last one, so the
    // sequence ends somewhere its own start will not repeat. The empty PDI is a
    // few hundred bytes, so the cost is negligible next to the configuration it
    // guards.
    for (auto &[seq, info] : resetsPerSequence) {
      if (!info.seen || info.firstParity != info.lastParity)
        continue;
      AIE::DeviceOp emptyDevice =
          getOrCreateEmptyDevice(module, info.device, 1 - info.lastParity);
      OpBuilder builder(seq.getContext());
      Block &body = seq.getBody().front();
      if (!body.empty() && body.back().hasTrait<OpTrait::IsTerminator>())
        builder.setInsertionPoint(&body.back());
      else
        builder.setInsertionPointToEnd(&body);
      NpuLoadPdiOp::create(
          builder, seq.getLoc(),
          FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr()),
          /*id=*/nullptr, /*size=*/nullptr, /*address=*/nullptr,
          /*expand_mode=*/
          AIEX::ExpandModeAttr::get(seq.getContext(), AIEX::ExpandMode::none));
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIEExpandLoadPdiPass() {
  return std::make_unique<AIEExpandLoadPdiPass>();
}
