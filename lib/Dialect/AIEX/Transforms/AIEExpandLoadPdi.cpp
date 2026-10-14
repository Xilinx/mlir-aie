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
//    b. an `aiex.npu.write_config @device`, whose register writes the NPU
//       translation emits (or, with inline-config, those writes as ops).
// 2. With ctrl-pkt=true: replaces each `load_pdi @device` with
//    a. a `load_pdi @ctrl_pkt_overlay`, which configures the NPU to stream
//       further configuration as control packets, and
//    b. a sequence of `aiex.npu.control_packet` ops carrying the device's
//       configuration.
//
//===----------------------------------------------------------------------===//

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/AIEUtils.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <map>
#include <set>

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

// A stream-switch setting: a port, or with `slot` >= 0 one of its packet rule
// slots.
struct SwitchPort {
  AIE::TileID tile;
  bool master;
  AIE::Port port;
  int slot;

  bool operator<(const SwitchPort &other) const {
    return std::tie(tile, master, port, slot) <
           std::tie(other.tile, other.master, other.port, other.slot);
  }

  std::string describe() const {
    std::string s =
        llvm::formatv("tile ({0}, {1}) {2} {3} : {4}", tile.col, tile.row,
                      master ? "master" : "slave",
                      AIE::stringifyWireBundle(port.bundle), port.channel);
    if (slot >= 0)
      s += llvm::formatv(" slot {0}", slot).str();
    return s;
  }
};

struct PortSetting {
  SwitchPort port;
  std::string value;
  Operation *op;
  bool skipped;
};

// The settings `op` writes, and whether a reload skips each. Each packet rule
// fills one slot, and a reload skips the rules marked
// is_ctrl_pkt_overlay. It skips enabling the port if it skips any of them.
static SmallVector<PortSetting> portSettings(Operation &op,
                                             AIE::SwitchboxOp sb) {
  auto key = [&](bool master, AIE::Port port, int slot = -1) {
    return SwitchPort{{sb.colIndex(), sb.rowIndex()}, master, port, slot};
  };
  auto amsel = [](Value v) {
    auto a = v.getDefiningOp<AIE::AMSelOp>();
    return llvm::formatv("{0}.{1}", a.arbiterIndex(), a.getMselValue()).str();
  };
  AIE::AIEDialect::IsCtrlPktOverlayAttrHelper overlay(op.getContext());
  bool skipped = overlay.isAttrPresent(&op);
  SmallVector<PortSetting> settings;
  if (auto connect = dyn_cast<AIE::ConnectOp>(op)) {
    AIE::Port src = connect.sourcePort();
    settings.push_back(
        {key(true, connect.destPort()),
         llvm::formatv("circuit {0} : {1}",
                       AIE::stringifyWireBundle(src.bundle), src.channel),
         &op, skipped});
    settings.push_back({key(false, src), "circuit", &op, skipped});
  } else if (auto masterSet = dyn_cast<AIE::MasterSetOp>(op)) {
    int arbiter = -1;
    int mask = 0;
    for (Value v : masterSet.getAmsels()) {
      auto a = cast<AIE::AMSelOp>(v.getDefiningOp());
      arbiter = a.arbiterIndex();
      mask |= 1 << a.getMselValue();
    }
    std::string value =
        llvm::formatv("packet {0}/{1} {2}", arbiter, mask,
                      masterSet.keepsPktHeader() ? "keep" : "drop")
            .str();
    settings.push_back({key(true, masterSet.destPort()), value, &op, skipped});
  } else if (auto rules = dyn_cast<AIE::PacketRulesOp>(op)) {
    // An empty packet_rules op writes nothing to the hardware.
    auto ruleOps = rules.getRules().front().getOps<AIE::PacketRuleOp>();
    if (ruleOps.empty())
      return settings;
    bool anySkipped = skipped;
    for (auto [slot, rule] : llvm::enumerate(ruleOps)) {
      bool ruleSkipped = skipped || overlay.isAttrPresent(rule);
      anySkipped |= ruleSkipped;
      settings.push_back({key(false, rules.sourcePort(), slot),
                          llvm::formatv("{0}/{1}>{2}", rule.valueInt(),
                                        rule.maskInt(), amsel(rule.getAmsel())),
                          rule, ruleSkipped});
    }
    settings.insert(settings.begin(), {key(false, rules.sourcePort()), "packet",
                                       &op, anySkipped});
  }
  return settings;
}

// A control-packet reload streams `device`'s configuration over the routes
// `overlay` set up, skipping what is marked is_ctrl_pkt_overlay. So the
// overlay must already write every setting the reload skips, the same way,
// and the reload may write nothing the overlay writes.
static LogicalResult verifyReloadKeepsOverlay(AIE::DeviceOp device,
                                              AIE::DeviceOp overlay) {
  StringRef overlayName = overlay.getSymName();
  std::map<SwitchPort, std::string> overlaySettings;
  overlay.walk([&](AIE::SwitchboxOp sb) {
    for (Operation &op : sb.getConnections().front())
      for (PortSetting &setting : portSettings(op, sb))
        overlaySettings[setting.port] = setting.value;
  });
  WalkResult result = device.walk([&](AIE::SwitchboxOp sb) {
    for (Operation &op : sb.getConnections().front()) {
      for (auto &[port, value, setter, skipped] : portSettings(op, sb)) {
        auto it = overlaySettings.find(port);
        if (skipped && it == overlaySettings.end()) {
          setter->emitError()
              << "a control-packet reload skips this op, but @" << overlayName
              << " does not set " << port.describe();
          return WalkResult::interrupt();
        }
        if (skipped && it->second != value) {
          setter->emitError()
              << "a control-packet reload skips this op, but @" << overlayName
              << " sets " << port.describe() << " differently";
          return WalkResult::interrupt();
        }
        if (!skipped && it != overlaySettings.end()) {
          setter->emitError()
              << "a control-packet reload rewrites " << port.describe()
              << ", which the control packets of @" << overlayName
              << " route through";
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

// A control-packet reload reaches a tile only if `overlay` routes control
// packets to its TileControl port. `packets` are the reload's.
static LogicalResult
verifyOverlayReachesTiles(NpuLoadPdiOp loadPdiOp, AIE::DeviceOp overlay,
                          iterator_range<Block::iterator> packets) {
  std::set<AIE::TileID> reached;
  overlay.walk([&](AIE::SwitchboxOp sb) {
    for (Operation &op : sb.getConnections().front()) {
      std::optional<AIE::Port> dest;
      if (auto master = dyn_cast<AIE::MasterSetOp>(op))
        dest = master.destPort();
      else if (auto connect = dyn_cast<AIE::ConnectOp>(op))
        dest = connect.destPort();
      if (dest == AIE::Port{AIE::WireBundle::TileControl, 0})
        reached.insert({sb.colIndex(), sb.rowIndex()});
    }
  });
  for (Operation &op : packets) {
    auto packet = dyn_cast<NpuControlPacketOp>(op);
    if (!packet)
      continue;
    AIE::TileID tile{static_cast<int>(packet.getColumnFromAddr()),
                     static_cast<int>(packet.getRowFromAddr())};
    if (reached.count(tile))
      continue;
    return loadPdiOp.emitError()
           << "a control-packet reload configures tile (" << tile.col << ", "
           << tile.row << "), but @" << overlay.getSymName()
           << " routes no control packets to it";
  }
  return success();
}

// The empty device whose PDI load makes the firmware reset the array. `parity`
// alternates so that two consecutive loads never name the same PDI (see
// transformLoadPdi).
static AIE::DeviceOp getOrCreateEmptyDevice(ModuleOp moduleOp,
                                            AIE::AIEDevice deviceType,
                                            unsigned parity, Location loc) {
  std::string emptyName = "empty_" + std::to_string(parity);
  if (auto existing = moduleOp.lookupSymbol<AIE::DeviceOp>(emptyName)) {
    // Shared reset PDIs retain every load and trailing sequence's provenance.
    loc = FusedLoc::get(moduleOp.getContext(), {existing.getLoc(), loc});
    existing->setLoc(loc);
    existing.getRegion().front().getTerminator()->setLoc(loc);
    return existing;
  }

  OpBuilder builder(moduleOp.getContext());
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto emptyDevice = AIE::DeviceOp::create(builder, loc, deviceType,
                                           builder.getStringAttr(emptyName));
  emptyDevice.getRegion().emplaceBlock();
  builder.setInsertionPointToEnd(&emptyDevice.getRegion().front());
  AIE::EndOp::create(builder, loc);
  return emptyDevice;
}

// A control-packet reload: the load_pdi it replaces, the device it streams,
// and the overlay load ahead of it.
struct CtrlPktReload {
  NpuLoadPdiOp load;
  AIE::DeviceOp device;
  NpuLoadPdiOp preload;
  AIE::DeviceOp overlay;
};

// Set `builder` after the last wait before the next load_pdi, or at that
// load_pdi (or the block's end) if there is none.
static void setInsertionPointToSegmentEnd(OpBuilder &builder,
                                          NpuLoadPdiOp loadPdiOp) {
  Operation *lastWait = nullptr;
  Operation *op = loadPdiOp->getNextNode();
  for (; op && !isa<NpuLoadPdiOp>(op) && !op->hasTrait<OpTrait::IsTerminator>();
       op = op->getNextNode())
    if (isa<NpuSyncOp, NpuDmaWaitOp, DMAAwaitTaskOp>(op))
      lastWait = op;
  if (lastWait)
    builder.setInsertionPointAfter(lastWait);
  else if (op)
    builder.setInsertionPoint(op);
  else
    builder.setInsertionPointToEnd(loadPdiOp->getBlock());
}

// Helper to transform a single load_pdi operation. Without `keepPreload` the
// empty-device reset is left out. With `teardownData` the configuration's
// writes are inserted here, since aiex.npu.write_config needs a reset ahead of
// it, and its teardown goes where it ends.
static LogicalResult
transformLoadPdi(NpuLoadPdiOp loadPdiOp, ModuleOp moduleOp, unsigned index,
                 AIEX::ExpandMode defaultMode, bool keepPreload,
                 AIEX::BlockwriteData *configData,
                 AIEX::BlockwriteData *teardownData,
                 SmallVectorImpl<CtrlPktReload> &ctrlPktReloads) {
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
  AIE::DeviceOp overlay;
  if (ctrlPkt) {
    // Overlay device PDI
    // Alternate between the original overlay and a clone of it on every
    // other load. Loading the same PDI twice in a row gets cached by the
    // firmware (the second load becomes a no-op), so we need two distinct
    // PDI addresses that carry the same overlay configuration.
    StringRef overlayName =
        (index % 2 == 0) ? kCtrlPktOverlayName : kCtrlPktOverlayCopyName;
    if (index % 2 != 0) {
      overlay =
          getOrCreateCtrlPktOverlayCopy(moduleOp, loadPdiOp.getOperation());
      if (!overlay)
        return failure();
    } else {
      overlay = moduleOp.lookupSymbol<AIE::DeviceOp>(kCtrlPktOverlayName);
      if (!overlay) {
        loadPdiOp.emitError("ctrl-pkt mode requires a `@")
            << kCtrlPktOverlayName << "` device in the module";
        return failure();
      }
    }
    if (failed(verifyReloadKeepsOverlay(referencedDevice, overlay)))
      return failure();
    preloadRef = FlatSymbolRefAttr::get(builder.getContext(), overlayName);
  } else if (keepPreload) {
    // Empty device PDI (triggers firmware reset)
    AIE::DeviceOp emptyDevice = getOrCreateEmptyDevice(
        moduleOp, referencedDevice.getDevice(), index % 2, loadPdiOp.getLoc());
    preloadRef = FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr());
  }

  builder.setInsertionPoint(loadPdiOp);

  // Emit the preload load_pdi (either empty-device reset or ctrl_pkt_overlay).
  if (ctrlPkt) {
    NpuLoadPdiOp preload =
        NpuLoadPdiOp::create(builder, loadPdiOp.getLoc(), preloadRef,
                             /*id=*/nullptr, /*size=*/nullptr,
                             /*address=*/nullptr,
                             /*expand_mode=*/
                             AIEX::ExpandModeAttr::get(builder.getContext(),
                                                       AIEX::ExpandMode::none));
    // The control packets go in once every reload is known, so that a
    // sequence's reloads share their constants.
    ctrlPktReloads.push_back({loadPdiOp, referencedDevice, preload, overlay});
    return success();
  }
  if (keepPreload)
    NpuLoadPdiOp::create(builder, loadPdiOp.getLoc(), preloadRef,
                         loadPdiOp.getIdAttr(), loadPdiOp.getSizeAttr(),
                         loadPdiOp.getAddressAttr(),
                         /*expand_mode=*/
                         AIEX::ExpandModeAttr::get(builder.getContext(),
                                                   AIEX::ExpandMode::none));
  if (!teardownData) {
    NpuWriteConfigOp::create(builder, loadPdiOp.getLoc(), deviceRefAttr);
  } else if (failed(xilinx::AIE::generateAndInsertConfigOps(
                 builder, referencedDevice, *configData))) {
    loadPdiOp.emitError("Failed to generate configuration operations");
    return failure();
  }

  if (teardownData) {
    setInsertionPointToSegmentEnd(builder, loadPdiOp);
    if (failed(xilinx::AIE::generateAndInsertTeardownOps(
            builder, referencedDevice, *teardownData))) {
      loadPdiOp.emitError("Failed to generate teardown operations");
      return failure();
    }
  }
  loadPdiOp.erase();
  return success();
}

struct AIEExpandLoadPdiPass
    : public xilinx::AIEX::impl::AIEExpandLoadPdiBase<AIEExpandLoadPdiPass> {
  using AIEExpandLoadPdiBase::AIEExpandLoadPdiBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, AIE::AIEDialect,
                    AIEX::AIEXDialect>();
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

    if (clInitEntry && !clTeardown) {
      module.emitError("aie-expand-load-pdi: init-entry requires teardown");
      return signalPassFailure();
    }
    auto expandMode = [&](NpuLoadPdiOp loadPdiOp) {
      return loadPdiOp.getDeviceRefAttr()
                 ? loadPdiOp.getExpandMode().value_or(defaultMode)
                 : AIEX::ExpandMode::none;
    };
    if (clTeardown)
      for (NpuLoadPdiOp loadPdiOp : loadPdiOps)
        if (expandMode(loadPdiOp) == AIEX::ExpandMode::ctrlpkt) {
          loadPdiOp.emitError(
              "teardown is not implemented for control packet reconfiguration");
          return signalPassFailure();
        }

    // A shim DMA channel has no reset, so no teardown can stop one a
    // configuration starts itself.
    if (clTeardown)
      for (NpuLoadPdiOp loadPdiOp : loadPdiOps) {
        auto dev = expandMode(loadPdiOp) != AIEX::ExpandMode::none
                       ? module.lookupSymbol<AIE::DeviceOp>(
                             loadPdiOp.getDeviceRefAttr())
                       : nullptr;
        if (dev && dev.walk([](AIE::DMAStartOp start) {
                        if (!isa<AIE::ShimDMAOp>(start->getParentOp()))
                          return WalkResult::advance();
                        start.emitOpError("starts a shim DMA channel, which "
                                          "teardown cannot stop");
                        return WalkResult::interrupt();
                      })
                       .wasInterrupted())
          return signalPassFailure();
      }

    // Per enclosing runtime sequence, the parity of the FIRST and LAST of its
    // empty-device resets and of its overlay loads, plus the device type.
    // `transformLoadPdi` picks the preload as `index % 2` over the module-wide
    // op order, and that order includes ops this pass skips, so a sequence's
    // first preload is not necessarily the even one and its parity cannot be
    // recovered from a count. Captured BEFORE transforming, which erases the
    // ops.
    struct PreloadParity {
      unsigned firstParity = 0;
      unsigned lastParity = 0;
      bool seen = false;
      AIE::AIEDevice device = {};
    };
    llvm::MapVector<AIE::RuntimeSequenceOp, std::array<PreloadParity, 2>>
        preloadsPerSequence;
    for (auto [index, loadPdiOp] : llvm::enumerate(loadPdiOps)) {
      auto deviceRefAttr = loadPdiOp.getDeviceRefAttr();
      if (!deviceRefAttr)
        continue;
      AIEX::ExpandMode mode = loadPdiOp.getExpandMode().value_or(defaultMode);
      if (mode == AIEX::ExpandMode::none)
        continue;
      auto seq = loadPdiOp->getParentOfType<AIE::RuntimeSequenceOp>();
      auto dev = module.lookupSymbol<AIE::DeviceOp>(deviceRefAttr);
      if (!seq || !dev)
        continue;
      auto &entry = preloadsPerSequence[seq][mode == AIEX::ExpandMode::ctrlpkt];
      if (!entry.seen) {
        entry.firstParity = index % 2;
        entry.seen = true;
      }
      entry.lastParity = index % 2;
      entry.device = dev.getDevice();
    }

    // Teardown leaves the array ready for the next configuration, so only a
    // sequence's first load needs a reset. With an init entry, `reconfig_init`
    // holds that reset. A load this pass leaves alone sets up a configuration
    // no teardown undoes, so the load after it keeps its reset.
    llvm::MapVector<AIE::DeviceOp, AIE::AIEDevice> initEntries;
    for (NpuLoadPdiOp loadPdiOp : loadPdiOps) {
      auto seq = loadPdiOp->getParentOfType<AIE::RuntimeSequenceOp>();
      if (clInitEntry && seq &&
          expandMode(loadPdiOp) == AIEX::ExpandMode::write32)
        if (auto dev = module.lookupSymbol<AIE::DeviceOp>(
                loadPdiOp.getDeviceRefAttr()))
          initEntries.try_emplace(seq->getParentOfType<AIE::DeviceOp>(),
                                  dev.getDevice());
    }

    // Transform load_pdi ops
    SmallVector<CtrlPktReload> ctrlPktReloads;
    llvm::DenseMap<AIE::DeviceOp, AIEX::BlockwriteData> configData,
        teardownData;
    // The kind of each sequence's last load, which left the array as it is.
    llvm::DenseMap<Operation *, std::optional<AIEX::ExpandMode>> arrayState;
    std::optional<AIEX::ExpandMode> start;
    if (clInitEntry)
      start = AIEX::ExpandMode::write32;
    for (auto [idx, loadPdiOp] : llvm::enumerate(loadPdiOps)) {
      auto seq = loadPdiOp->getParentOfType<AIE::RuntimeSequenceOp>();
      std::optional<AIEX::ExpandMode> &state =
          arrayState.try_emplace(seq, start).first->second;
      bool keepPreload = !clTeardown || state != expandMode(loadPdiOp);
      state = expandMode(loadPdiOp);
      AIEX::BlockwriteData *config = nullptr, *teardown = nullptr;
      if (clTeardown) {
        auto device = loadPdiOp->getParentOfType<AIE::DeviceOp>();
        config =
            &configData.try_emplace(device, device, "loadpdi_").first->second;
        teardown = &teardownData.try_emplace(device, device, "teardown_")
                        .first->second;
      }
      if (failed(transformLoadPdi(loadPdiOp, module, idx, defaultMode,
                                  keepPreload, config, teardown,
                                  ctrlPktReloads))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<std::pair<Operation *, AIE::DeviceOp>> ctrlPktSites;
    for (CtrlPktReload &reload : ctrlPktReloads)
      ctrlPktSites.emplace_back(reload.load, reload.device);
    if (failed(AIE::insertConfigOps(
            ctrlPktSites, AIEToConfigurationOutputType::ControlPacket))) {
      signalPassFailure();
      return;
    }
    for (CtrlPktReload &reload : ctrlPktReloads) {
      if (failed(verifyOverlayReachesTiles(
              reload.load, reload.overlay,
              {std::next(reload.preload->getIterator()),
               reload.load->getIterator()}))) {
        signalPassFailure();
        return;
      }
      reload.load.erase();
    }

    for (auto [device, deviceType] : initEntries) {
      if (Operation *existing = device.lookupSymbol(initEntrySequenceName)) {
        existing->emitOpError()
            << "is named '" << initEntrySequenceName
            << "', which aie-expand-load-pdi needs for the device's init entry";
        return signalPassFailure();
      }
      OpBuilder builder(device.getContext());
      builder.setInsertionPoint(
          *device.getOps<AIE::RuntimeSequenceOp>().begin());
      auto init = AIE::RuntimeSequenceOp::create(
          builder, device.getLoc(),
          builder.getStringAttr(initEntrySequenceName), BoolAttr{},
          AIE::TraceBufferAttr{}, ArrayAttr{});
      builder.setInsertionPointToStart(&init.getBody().emplaceBlock());
      AIE::DeviceOp emptyDevice =
          getOrCreateEmptyDevice(module, deviceType, 0, device.getLoc());
      NpuLoadPdiOp::create(builder, device.getLoc(),
                           FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr()),
                           /*id=*/nullptr, /*size=*/nullptr,
                           /*address=*/nullptr,
                           /*expand_mode=*/
                           AIEX::ExpandModeAttr::get(builder.getContext(),
                                                     AIEX::ExpandMode::none));
    }

    if (clInlineConfig && failed(AIE::inlineWriteConfigs(module)))
      return signalPassFailure();

    // Under teardown each sequence loads at most one reset, so none needs the
    // append.
    if (clTeardown)
      return;

    // The `index % 2` alternation above keeps two consecutive preloads on
    // different PDIs WITHIN a sequence. But the host re-runs the whole sequence
    // on every dispatch, so the alternation has to hold across that boundary
    // too. When a sequence's last preload lands on the same PDI as its first,
    // the firmware caches the address and no-ops the next dispatch's first
    // load, and the configuration that follows lands on a device that was never
    // reset. That is silent -- the first dispatch is correct and later ones
    // return non-deterministic garbage.
    //
    // Append one more preload, of the parity opposite the last one, so the
    // sequence ends somewhere its own start will not repeat. The empty PDI is a
    // few hundred bytes, so the cost is negligible next to the configuration it
    // guards.
    for (auto &[seq, kinds] : preloadsPerSequence) {
      for (auto [ctrlPkt, info] : llvm::enumerate(kinds)) {
        if (!info.seen || info.firstParity != info.lastParity)
          continue;
        unsigned parity = 1 - info.lastParity;
        FlatSymbolRefAttr preloadRef;
        if (ctrlPkt) {
          if (parity == 1 && !getOrCreateCtrlPktOverlayCopy(module, seq)) {
            signalPassFailure();
            return;
          }
          preloadRef = FlatSymbolRefAttr::get(
              seq.getContext(),
              parity == 0 ? kCtrlPktOverlayName : kCtrlPktOverlayCopyName);
        } else {
          AIE::DeviceOp emptyDevice =
              getOrCreateEmptyDevice(module, info.device, parity, seq.getLoc());
          preloadRef = FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr());
        }
        OpBuilder builder(seq.getContext());
        Block &body = seq.getBody().front();
        if (!body.empty() && body.back().hasTrait<OpTrait::IsTerminator>())
          builder.setInsertionPoint(&body.back());
        else
          builder.setInsertionPointToEnd(&body);
        NpuLoadPdiOp::create(builder, seq.getLoc(), preloadRef,
                             /*id=*/nullptr, /*size=*/nullptr,
                             /*address=*/nullptr,
                             /*expand_mode=*/
                             AIEX::ExpandModeAttr::get(seq.getContext(),
                                                       AIEX::ExpandMode::none));
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
xilinx::AIEX::createAIEExpandLoadPdiPass() {
  return std::make_unique<AIEExpandLoadPdiPass>();
}
