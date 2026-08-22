//===- AIEExpandLoadPdi.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands `npu.load_pdi` operations that reference a device. There
// are three output modes, controlled by the `ctrl-pkt` / `register-reset`
// options:
//
// 1. Default (both false): replaces each `load_pdi @device` with
//    a. an empty device PDI load (`load_pdi @empty_N`), which causes the
//       firmware to reset the device, and
//    b. explicit `aiex.npu.write32`/`aiex.npu.blockwrite` configuration ops.
// 2. With ctrl-pkt=true: replaces each `load_pdi @device` with
//    a. a `load_pdi @ctrl_pkt_overlay`, which configures the NPU to stream
//       further configuration as control packets, and
//    b. a sequence of `aiex.npu.control_packet` ops carrying the device's
//       configuration.
// 3. With register-reset=true: for every `load_pdi` after the first,
//    replaces the empty-device firmware reset with `aiex.core_reset` /
//    `aiex.dma_channel_reset_for` register writes targeting the PREVIOUS
//    `load_pdi`'s device, instead of a firmware partition reset, then emits
//    the same write32/blockwrite configuration sequence as the default mode.
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

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

// Find (or create) a device-local `aie.lock(tile, lockID)` in `device`,
// inserted just before `insertBefore`. Mirrors `AIE::TileOp::getOrCreate`,
// which this relies on to have already placed `tile` before `insertBefore`.
static AIE::LockOp getOrCreateLock(OpBuilder &builder, AIE::DeviceOp device,
                                   Operation *insertBefore, AIE::TileOp tile,
                                   int lockID) {
  for (auto lock : device.getOps<AIE::LockOp>()) {
    auto lockTile =
        dyn_cast_or_null<AIE::TileOp>(lock.getTile().getDefiningOp());
    if (lockTile && lockTile.getCol() == tile.getCol() &&
        lockTile.getRow() == tile.getRow() && lock.getLockID() &&
        *lock.getLockID() == lockID)
      return lock;
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertBefore);
  return AIE::LockOp::create(builder, device.getLoc(), tile.getResult(), lockID,
                             /*init=*/0);
}

// Replace the empty-device firmware reset with register writes: `core_reset`
// for every core of `outgoingDevice`, and `dma_channel_reset_for` for every
// resident objectFIFO of `outgoingDevice`. An SSA value cannot cross into a
// sibling `aie.device` region, so each `aie.tile`/`aie.lock` the reset needs
// is re-declared in `mainDevice` (deduplicated by (col, row[, lockID])); the
// `aie.objectfifo_rearm_binding` itself is re-declared alongside them under a
// fresh name, since `aiex.dma_channel_reset_for` resolves its symbol only
// within its own enclosing device. Reset ops land at `builder`'s current
// insertion point (inside `runtimeSeqOp`); the tile/lock/binding
// declarations they reference go directly in `mainDevice`, before
// `runtimeSeqOp`. `bindingCounter` gives each re-declared binding a unique
// name across the whole pass run.
static LogicalResult emitRegisterReset(OpBuilder &builder, Location loc,
                                       AIE::DeviceOp mainDevice,
                                       AIE::RuntimeSequenceOp runtimeSeqOp,
                                       AIE::DeviceOp outgoingDevice,
                                       unsigned &bindingCounter) {
  for (auto coreOp : outgoingDevice.getOps<AIE::CoreOp>()) {
    AIE::TileOp origTile = coreOp.getTileOp();
    AIE::TileOp freshTile = AIE::TileOp::getOrCreate(
        builder, mainDevice, origTile.getCol(), origTile.getRow());
    AIEX::CoreResetOp::create(builder, loc, freshTile.getResult());
  }

  for (auto binding : outgoingDevice.getOps<AIE::ObjectFifoRearmBindingOp>()) {
    SmallVector<Value> freshTiles;
    for (Value t : binding.getChannelTiles()) {
      auto orig = dyn_cast_or_null<AIE::TileOp>(t.getDefiningOp());
      if (!orig)
        return binding.emitOpError(
            "channel_tiles operand is not defined by an aie.tile op");
      freshTiles.push_back(AIE::TileOp::getOrCreate(builder, mainDevice,
                                                    orig.getCol(),
                                                    orig.getRow())
                               .getResult());
    }
    SmallVector<Value> freshLocks;
    for (Value l : binding.getLocks()) {
      auto origLock = dyn_cast_or_null<AIE::LockOp>(l.getDefiningOp());
      auto origTile = origLock ? dyn_cast_or_null<AIE::TileOp>(
                                     origLock.getTile().getDefiningOp())
                               : nullptr;
      if (!origLock || !origTile || !origLock.getLockID())
        return binding.emitOpError(
            "locks operand is not a fully-resolved aie.lock(aie.tile, id)");
      AIE::TileOp freshTile = AIE::TileOp::getOrCreate(
          builder, mainDevice, origTile.getCol(), origTile.getRow());
      freshLocks.push_back(getOrCreateLock(builder, mainDevice, runtimeSeqOp,
                                           freshTile, *origLock.getLockID())
                               .getResult());
    }

    std::string newName =
        (Twine(binding.getSymName()) + "_regreset_" + Twine(bindingCounter++))
            .str();
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(runtimeSeqOp);
      AIE::ObjectFifoRearmBindingOp::create(
          builder, loc, builder.getStringAttr(newName), freshTiles, freshLocks,
          binding.getChannelDirsAttr(), binding.getChannelIndicesAttr(),
          binding.getLockInitsAttr(), binding.getHeadBdIdsAttr(),
          binding.getRepeatCountsAttr());
    }
    AIEX::DmaChannelResetForOp::create(
        builder, loc, FlatSymbolRefAttr::get(builder.getContext(), newName));
  }
  return success();
}

// Helper to transform a single load_pdi operation
static LogicalResult transformLoadPdi(NpuLoadPdiOp loadPdiOp, ModuleOp moduleOp,
                                      unsigned index,
                                      AIEX::ExpandMode defaultMode,
                                      AIE::DeviceOp outgoingDevice,
                                      unsigned &bindingCounter) {
  static unsigned long i = 0;
  OpBuilder builder(loadPdiOp);

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
  // The first load_pdi in the module has no known outgoing device, so it
  // always falls back to the default empty-device reset.
  bool useRegisterReset =
      (mode == AIEX::ExpandMode::regreset && outgoingDevice != nullptr);

  auto referencedDevice = moduleOp.lookupSymbol<AIE::DeviceOp>(deviceRefAttr);
  if (!referencedDevice) {
    loadPdiOp.emitError("Referenced symbol '")
        << deviceRefAttr.getValue() << "' is not a device";
    return failure();
  }

  if (useRegisterReset) {
    auto mainDevice = loadPdiOp->getParentOfType<AIE::DeviceOp>();
    auto runtimeSeqOp = loadPdiOp->getParentOfType<AIE::RuntimeSequenceOp>();
    if (!mainDevice || !runtimeSeqOp) {
      loadPdiOp.emitError("register-reset mode requires load_pdi inside an "
                          "aie.runtime_sequence inside an aie.device");
      return failure();
    }
    if (failed(emitRegisterReset(builder, loadPdiOp.getLoc(), mainDevice,
                                 runtimeSeqOp, outgoingDevice, bindingCounter)))
      return failure();
  }

  FlatSymbolRefAttr preloadRef;
  if (useRegisterReset) {
    // No preload load_pdi: emitRegisterReset above already cleared the
    // outgoing device's residual core/DMA state with register writes, so
    // there is nothing to load before streaming the incoming configuration.
  } else if (ctrlPkt) {
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
      loadPdiOp.emitError("ctrl-pkt mode requires a `@")
          << kCtrlPktOverlayName << "` device in the module";
      return failure();
    }
    preloadRef = FlatSymbolRefAttr::get(builder.getContext(), overlayName);
  } else {
    // Empty device PDI (triggers firmware reset)
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    std::string emptyName = "empty_" + std::to_string(index % 2);
    AIE::DeviceOp emptyDevice = moduleOp.lookupSymbol<AIE::DeviceOp>(emptyName);
    if (!emptyDevice) {
      auto deviceType = referencedDevice.getDevice();
      auto loc = builder.getUnknownLoc();
      emptyDevice = AIE::DeviceOp::create(builder, loc, deviceType,
                                          builder.getStringAttr(emptyName));
      emptyDevice.getRegion().emplaceBlock();
      Block *deviceBlock = &emptyDevice.getRegion().front();
      builder.setInsertionPointToEnd(deviceBlock);
      AIE::EndOp::create(builder, loc);
    }
    preloadRef = FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr());
  }

  builder.setInsertionPoint(loadPdiOp);

  // Emit the preload load_pdi (empty-device reset or ctrl_pkt_overlay); the
  // register-reset path has no preload step, see above.
  if (useRegisterReset) {
    // Nothing to emit here.
  } else if (ctrlPkt) {
    NpuLoadPdiOp::create(builder, loadPdiOp.getLoc(), preloadRef,
                         /*id=*/nullptr, /*size=*/nullptr,
                         /*address=*/nullptr,
                         /*expand_mode=*/
                         AIEX::ExpandModeAttr::get(builder.getContext(),
                                                   AIEX::ExpandMode::none));
  } else {
    NpuLoadPdiOp::create(builder, loadPdiOp.getLoc(), preloadRef,
                         loadPdiOp.getIdAttr(), loadPdiOp.getSizeAttr(),
                         loadPdiOp.getAddressAttr(),
                         /*expand_mode=*/
                         AIEX::ExpandModeAttr::get(builder.getContext(),
                                                   AIEX::ExpandMode::none));
  }

  // Step 2: generate and insert configuration ops. register-reset reuses the
  // write32/blockwrite (Transaction) output, same as the default mode.
  auto outputType = ctrlPkt ? AIEToConfigurationOutputType::ControlPacket
                            : AIEToConfigurationOutputType::Transaction;
  std::string prefix = ctrlPkt ? ("loadpdi_ctrlpkt_" + std::to_string(i) + "_")
                               : ("loadpdi_" + std::to_string(i));
  if (failed(xilinx::AIE::generateAndInsertConfigOps(
          builder, referencedDevice, /*clElfDir=*/"", outputType, prefix,
          /*skipCtrlPktOverlay=*/ctrlPkt))) {
    loadPdiOp.emitError("Failed to generate configuration operations");
    return failure();
  }

  // Erase the original load_pdi operation
  loadPdiOp.erase();

  i++;

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

    if (clCtrlPkt && clRegisterReset) {
      module.emitError("aie-expand-load-pdi: ctrl-pkt and register-reset are "
                       "mutually exclusive");
      signalPassFailure();
      return;
    }

    // Collect all load_pdi operations in program order;
    // need to collect once, then transform all collected ops;
    // since the transform inserts a new preload load_pdi, we can't transform
    // as we walk or it'd infinitely recurse.
    SmallVector<NpuLoadPdiOp> loadPdiOps;

    module.walk(
        [&](NpuLoadPdiOp loadPdiOp) { loadPdiOps.push_back(loadPdiOp); });

    // Map the legacy bool options to the new ExpandMode enum.
    AIEX::ExpandMode defaultMode = clCtrlPkt ? AIEX::ExpandMode::ctrlpkt
                                   : clRegisterReset
                                       ? AIEX::ExpandMode::regreset
                                       : AIEX::ExpandMode::write32;

    // Transform load_pdi ops. outgoingDevice tracks, in program order, which
    // device the PREVIOUS load_pdi made resident -- register-reset mode
    // resets that device's state instead of firmware-resetting the NPU.
    unsigned idx = 0;
    unsigned bindingCounter = 0;
    AIE::DeviceOp outgoingDevice = nullptr;
    for (auto loadPdiOp : loadPdiOps) {
      auto deviceRefAttr = loadPdiOp.getDeviceRefAttr();
      if (failed(transformLoadPdi(loadPdiOp, module, idx, defaultMode,
                                  outgoingDevice, bindingCounter))) {
        signalPassFailure();
        return;
      }
      if (deviceRefAttr)
        outgoingDevice = module.lookupSymbol<AIE::DeviceOp>(deviceRefAttr);
      idx++;
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIEExpandLoadPdiPass() {
  return std::make_unique<AIEExpandLoadPdiPass>();
}
