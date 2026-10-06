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

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

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
  bool skipped = op.hasAttr(AIE::kCtrlPktOverlayAttrName);
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
    bool keepHeader = masterSet.getKeepPktHeader().value_or(
        masterSet.getDestBundle() != AIE::WireBundle::DMA &&
        (sb.rowIndex() != 0 ||
         masterSet.getDestBundle() != AIE::WireBundle::South));
    std::string value = llvm::formatv("packet {0}/{1} {2}", arbiter, mask,
                                      keepHeader ? "keep" : "drop")
                            .str();
    settings.push_back({key(true, masterSet.destPort()), value, &op, skipped});
  } else if (auto rules = dyn_cast<AIE::PacketRulesOp>(op)) {
    // An empty packet_rules op writes nothing to the hardware.
    auto ruleOps = rules.getRules().front().getOps<AIE::PacketRuleOp>();
    if (ruleOps.empty())
      return settings;
    bool anySkipped = skipped;
    for (auto [slot, rule] : llvm::enumerate(ruleOps)) {
      bool ruleSkipped = skipped || rule->hasAttr(AIE::kCtrlPktOverlayAttrName);
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
          setter->emitError() << "a control-packet reload skips this op, but @"
                              << overlayName << " does not set "
                              << port.describe();
          return WalkResult::interrupt();
        }
        if (skipped && it->second != value) {
          setter->emitError() << "a control-packet reload skips this op, but @"
                              << overlayName << " sets " << port.describe()
                              << " differently";
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
  } else {
    // Empty device PDI (triggers firmware reset)
    AIE::DeviceOp emptyDevice = getOrCreateEmptyDevice(
        moduleOp, referencedDevice.getDevice(), index % 2);
    preloadRef = FlatSymbolRefAttr::get(emptyDevice.getSymNameAttr());
  }

  builder.setInsertionPoint(loadPdiOp);

  // Emit the preload load_pdi (either empty-device reset or ctrl_pkt_overlay).
  NpuLoadPdiOp preload;
  if (ctrlPkt) {
    preload =
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
  if (ctrlPkt &&
      failed(verifyOverlayReachesTiles(
          loadPdiOp, overlay,
          {std::next(preload->getIterator()), loadPdiOp->getIterator()})))
    return failure();

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
    llvm::MapVector<AIE::RuntimeSequenceOp, ResetParity> resetsPerSequence;
    for (auto [index, loadPdiOp] : llvm::enumerate(loadPdiOps)) {
      auto deviceRefAttr = loadPdiOp.getDeviceRefAttr();
      if (!deviceRefAttr)
        continue;
      if (loadPdiOp.getExpandMode().value_or(clCtrlPkt
                                                 ? AIEX::ExpandMode::ctrlpkt
                                                 : AIEX::ExpandMode::write32) !=
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
