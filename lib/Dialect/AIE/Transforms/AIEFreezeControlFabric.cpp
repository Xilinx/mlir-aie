//===- AIEFreezeControlFabric.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Module-level capture + annotate of the reconfiguration-safe control fabric.
// A per-device sibling read is unsafe: DeviceOp is IsolatedFromAbove so the
// per-device pathfinder parallelizes across devices, the standalone
// @ctrl_pkt_overlay is emitted last, and canonical control routing does not
// exist before the pathfinder runs (the overlay pass emits only packet_flow
// declarations). So this module pass runs a read-only pathfinder analysis on
// the data-free @ctrl_pkt_overlay ONCE and ANNOTATES each participating
// config's control packet_flow op with its captured route (a per-device IR
// attribute, never shared mutable state). Control stays a co-routed
// packet_flow -- nothing is materialized or removed. The per-device pathfinder
// (AIEPathFinder.cpp) decodes the annotation in runAnalysis and PINS the flow
// so findPaths replays the captured route instead of re-deriving it, while the
// native emitClass merges control and data for free on a shared slave port.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIE/Transforms/AIEPathFinder.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#include <array>

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEFREEZECONTROLFABRIC
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-freeze-control-fabric"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

// A control packet flow is one the overlay pass tagged priority_route (the
// `ctrl_pkt_flow` argument of createPacketFlowOp); the pathfinder keys its
// is_ctrl_pkt_overlay tags off the same attribute.
static bool isControlPacketFlow(AIE::PacketFlowOp flow) {
  return flow.getPriorityRoute().value_or(false);
}

// Extra demand a design-aware freeze puts on every cell config data occupies,
// so control routes around it. Finite (never INF, which is reserved for pinned
// priority): big enough that one avoided data cell outweighs a few extra
// control hops, small enough that control can still overlap when a direction is
// fully occupied (it then degrades to the shortest path).
static constexpr double DESIGN_AVOID_PENALTY = 100.0;

// Seed one occupied crossbar cell into the design-demand field, mirroring the
// pathfinder's replay/build neighbor logic (AIEPathFinder.cpp
// replayPinnedRoute): the intra-tile crossbar hop sp->dp, plus the implicit
// inter-tile hop leaving output port dp toward its neighbor (only a directional
// N/S/E/W output has one; terminal ports match none). Assignment (not
// accumulation) = binary union: a cell any config occupies gets one fixed
// penalty regardless of how many share it (count-ranking would be backwards --
// a widely shared flexible master is safer to overlap than one design's sole
// egress). Shared by the routed-flow and pre-placed-connection accumulators.
static void seedDemandCell(TileID coords, Port sp, Port dp,
                           AIE::DesignField &field) {
  field[{coords, coords, sp, dp}] = DESIGN_AVOID_PENALTY;
  for (const auto &[neighborCoords, neighborPort] :
       getCardinalNeighbors(coords, dp.channel)) {
    if (dp.bundle != getConnectingBundle(neighborPort.bundle))
      continue;
    field[{coords, neighborCoords, dp, neighborPort}] = DESIGN_AVOID_PENALTY;
    break;
  }
}

// Seed every switchbox-connect cell each routed data flow occupies.
static void accumulateDesignDemand(
    const std::map<AIE::PathEndPoint, AIE::SwitchSettings> &flowSolutions,
    AIE::DesignField &field) {
  for (const auto &[src, settings] : flowSolutions)
    for (const auto &[coords, setting] : settings)
      for (size_t k = 0; k < setting.srcs.size(); k++)
        seedDemandCell(coords, setting.srcs[k], setting.dsts[k], field);
}

// Pre-placed (manual) circuit connections -- aie.switchbox / aie.shim_mux with
// hand-written aie.connect children -- are fixed obstacles the design-demand
// capture would otherwise miss (they are not routed flows, so they never appear
// in flowSolutions). Seed their occupied cells too, or a captured control route
// can pick a channel a manual circuit already holds and collide with it on the
// per-device replay (a fixed circuit connection cannot be shared).
static void accumulatePreplacedDemand(AIE::DeviceOp cfg,
                                      AIE::DesignField &field) {
  auto seedConnects = [&](TileID coords, auto connectOps) {
    for (auto connectOp : connectOps)
      seedDemandCell(coords, connectOp.sourcePort(), connectOp.destPort(),
                     field);
  };
  for (auto sw : cfg.getOps<AIE::SwitchboxOp>())
    seedConnects({sw.getTileOp().colIndex(), sw.getTileOp().rowIndex()},
                 sw.getOps<AIE::ConnectOp>());
  for (auto mux : cfg.getOps<AIE::ShimMuxOp>())
    seedConnects({mux.getTileOp().colIndex(), mux.getTileOp().rowIndex()},
                 mux.getOps<AIE::ConnectOp>());
}

struct AIEFreezeControlFabricPass
    : xilinx::AIE::impl::AIEFreezeControlFabricBase<
          AIEFreezeControlFabricPass> {
  AIEFreezeControlFabricPass() = default;
  AIEFreezeControlFabricPass(const AIEFreezeControlFabricOptions &options)
      : AIEFreezeControlFabricBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Find the standalone control-only overlay device. Without one there is no
    // canonical control routing to freeze.
    DeviceOp overlay;
    for (auto dev : module.getOps<DeviceOp>()) {
      if (dev.getSymName() == "ctrl_pkt_overlay") {
        overlay = dev;
        break;
      }
    }
    if (!overlay)
      return;

    // Config devices that carry control packet flows to freeze.
    SmallVector<DeviceOp> configs;
    for (auto dev : module.getOps<DeviceOp>()) {
      if (dev == overlay)
        continue;
      bool hasControl = false;
      for (auto flow : dev.getOps<AIE::PacketFlowOp>())
        if (isControlPacketFlow(flow)) {
          hasControl = true;
          break;
        }
      if (hasControl)
        configs.push_back(dev);
    }
    if (configs.empty())
      return;

    // Design-aware freeze (eager avoidance): route each config's DATA demand
    // (control excluded) and aggregate it into a field so the captured control
    // route steers OFF the ports the designs use, minimizing the overlay's
    // imposition on already-routable designs. Control still freezes to ONE
    // route across all configs, so it avoids the UNION of their demand. Off by
    // default the field stays empty and the overlay routes blind
    // (byte-identical Layer 0).
    DesignField designField;
    if (designAware) {
      for (auto cfg : configs) {
        // Fresh analyzer per config (each owns its Pathfinder; read-only, no
        // shared state). skipControlFlows routes the config's circuit + data
        // packet flows only -- on the config's OWN model, so no cross-model
        // replay happens here and the device-model guard below (in the annotate
        // loop) is the one that matters. A config whose data alone will not
        // route is broken regardless of the overlay -> fail loud.
        DynamicTileAnalysis dataAnalyzer;
        if (failed(dataAnalyzer.runAnalysis(cfg, /*skipControlFlows=*/true)))
          return signalPassFailure();
        accumulateDesignDemand(dataAnalyzer.flowSolutions, designField);
        // Also avoid the config's pre-placed (manual) circuit connections,
        // which are fixed obstacles absent from the routed flowSolutions.
        accumulatePreplacedDemand(cfg, designField);
      }
    }

    // Capture the canonical control routing: run the pathfinder analysis on the
    // data-free @ctrl_pkt_overlay directly. runAnalysis is read-only (it
    // creates no ops), so the overlay keeps its packet_flow declarations for
    // the per-device pathfinder to route -- data-free and deterministic, it
    // reproduces this exact routing. flowSolutions maps each control source to
    // its routed SwitchSettings (the captured canonical route). Under
    // design-aware freeze the seeded field bends this route around config data.
    DynamicTileAnalysis analyzer;
    if (failed(analyzer.runAnalysis(overlay, /*skipControlFlows=*/false,
                                    designAware ? &designField : nullptr)))
      return signalPassFailure();

    // Annotate each config's control packet_flow op with its captured route,
    // one entry per source. Control stays a CO-ROUTED packet_flow (decl NOT
    // removed, switch ops NOT materialized); the per-device pathfinder decodes
    // the annotation and PINS the flow to the captured route so it cannot
    // drift, while the native emitClass merges control+data for free.
    //
    // Design-aware freeze also pins @ctrl_pkt_overlay itself: its captured
    // route differs from what the per-device pathfinder derives BLIND, so the
    // resident overlay (routed downstream to build the fabric) must replay the
    // SAME route as the configs, or the two would disagree on the physical
    // control port. In blind mode the overlay reproduces the capture
    // deterministically, so it is left unpinned and OFF stays byte-identical.
    SmallVector<DeviceOp> annotate(configs.begin(), configs.end());
    if (designAware)
      annotate.push_back(overlay);
    MLIRContext *ctx = module.getContext();
    for (auto cfg : annotate) {
      // A captured route encodes physical ports valid only for the overlay's
      // target model; replaying it into a config of a different device model
      // would be silently wrong. Require the models match.
      if (cfg.getDevice() != overlay.getDevice()) {
        cfg.emitOpError("device model differs from @ctrl_pkt_overlay; cannot "
                        "replay the captured control route");
        return signalPassFailure();
      }
      for (auto flow : cfg.getOps<AIE::PacketFlowOp>()) {
        if (!isControlPacketFlow(flow))
          continue;
        SmallVector<Attribute> perSource;
        for (Operation &op : flow.getPorts().front().getOperations()) {
          auto pktSource = dyn_cast<AIE::PacketSourceOp>(op);
          if (!pktSource)
            continue;
          auto srcTile = cast<AIE::TileOp>(pktSource.getTile().getDefiningOp());
          TileID srcCoords = {srcTile.colIndex(), srcTile.rowIndex()};
          Port srcPort = pktSource.port();
          auto it = analyzer.flowSolutions.find({srcCoords, srcPort});
          if (it == analyzer.flowSolutions.end()) {
            // The overlay analysis produced no route for this control source,
            // so it would ship UNPINNED and could drift config-to-config (the
            // freeze silently fails for that source). Fail loud instead.
            pktSource.emitOpError()
                << "control source (" << srcCoords.col << ", " << srcCoords.row
                << ") " << stringifyWireBundle(srcPort.bundle)
                << srcPort.channel
                << " has no captured route in @ctrl_pkt_overlay";
            return signalPassFailure();
          }
          perSource.push_back(
              encodePinnedRoute(ctx, srcCoords, srcPort, it->second));
        }
        if (!perSource.empty())
          flow->setAttr(kPinnedRouteAttr, ArrayAttr::get(ctx, perSource));
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
AIE::createAIEFreezeControlFabricPass() {
  return std::make_unique<AIEFreezeControlFabricPass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
AIE::createAIEFreezeControlFabricPass(bool designAware) {
  AIEFreezeControlFabricOptions options;
  options.designAware = designAware;
  return std::make_unique<AIEFreezeControlFabricPass>(options);
}
