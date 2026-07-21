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

#include "llvm/ADT/StringRef.h"
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

// Helper to transform a single load_pdi operation
static LogicalResult transformLoadPdi(NpuLoadPdiOp loadPdiOp, ModuleOp moduleOp,
                                      unsigned index,
                                      AIEX::ExpandMode defaultMode) {
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

  auto referencedDevice = moduleOp.lookupSymbol<AIE::DeviceOp>(deviceRefAttr);
  if (!referencedDevice) {
    loadPdiOp.emitError("Referenced symbol '")
        << deviceRefAttr.getValue() << "' is not a device";
    return failure();
  }

  FlatSymbolRefAttr preloadRef;
  if (ctrlPkt) {
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

  // Emit the preload load_pdi (either empty-device reset or ctrl_pkt_overlay).
  if (ctrlPkt) {
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

  // Step 2: generate and insert configuration ops.
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

    // Collect all load_pdi operations in program order;
    // need to collect once, then transform all collected ops;
    // since the transform inserts a new preload load_pdi, we can't transform
    // as we walk or it'd infinitely recurse.
    SmallVector<NpuLoadPdiOp> loadPdiOps;

    module.walk(
        [&](NpuLoadPdiOp loadPdiOp) { loadPdiOps.push_back(loadPdiOp); });

    // Map the legacy bool option to the new ExpandMode enum.
    AIEX::ExpandMode defaultMode =
        clCtrlPkt ? AIEX::ExpandMode::ctrlpkt : AIEX::ExpandMode::write32;

    // Transform load_pdi ops
    unsigned idx = 0;
    for (auto loadPdiOp : loadPdiOps) {
      if (failed(transformLoadPdi(loadPdiOp, module, idx, defaultMode))) {
        signalPassFailure();
        return;
      }
      idx++;
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIEExpandLoadPdiPass() {
  return std::make_unique<AIEExpandLoadPdiPass>();
}
