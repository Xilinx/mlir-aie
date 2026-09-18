//===- AIEAutoPacketizeControlIngress.cpp -----------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEAUTOPACKETIZECONTROLINGRESS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

namespace {

struct AIEAutoPacketizeControlIngressPass
    : public xilinx::AIE::impl::AIEAutoPacketizeControlIngressBase<
          AIEAutoPacketizeControlIngressPass> {

  // The union view of one physical shim column across every config in the
  // module. `shimMuxBlocked`/`dataPinned` are unions over configs: a channel is
  // shim-mux-blocked if a manual aie.shim_mux hard-reserves it in ANY config,
  // and data-pinned if a circuit leg pins prod_dma_channel to it in ANY config.
  struct ColumnUnion {
    int numMM2S = 0;
    llvm::SmallSet<int, 4> shimMuxBlocked;
    llvm::SmallSet<int, 4> dataPinned;
    SmallVector<DeviceOp> devices;
  };

  // Traffic proxy for a shim-ingress leg: element byte-size * producer depth.
  // Element size uses the objectFifo's memref element type; depth is the
  // producer entry of elemNumber (mirrors AIESAPlacer's producer-side sizing).
  // Element bit-width comes from a DataLayout (not getElementTypeBitWidth) so
  // custom element types such as !aiex.bfp block-float sizes resolve instead of
  // asserting.
  static int64_t ingressTrafficProxy(ObjectFifoCreateOp ofo) {
    auto fifoType = dyn_cast<AIEObjectFifoType>(ofo.getElemType());
    if (!fifoType)
      return 0;
    MemRefType memref = fifoType.getElementType();
    if (!memref)
      return 0;
    mlir::DataLayout dataLayout(ofo->getParentOfType<ModuleOp>());
    // Round bits up to whole bytes so sub-byte element types (i4, i1) don't
    // truncate to a 0-byte proxy (leg ranking uses this size).
    uint64_t elemBits = dataLayout.getTypeSizeInBits(memref.getElementType());
    int64_t elemBytes =
        memref.getNumElements() * (int64_t)llvm::divideCeil(elemBits, 8);
    int depth = 1;
    if (auto arr = dyn_cast<ArrayAttr>(ofo.getElemNumber()))
      depth = arr.getValue().empty()
                  ? 1
                  : (int)cast<IntegerAttr>(arr.getValue()[0]).getInt();
    else if (auto i = dyn_cast<IntegerAttr>(ofo.getElemNumber()))
      depth = (int)i.getInt();
    return elemBytes * depth;
  }

  // Fan-out of a shim-ingress leg = the number of destination endpoints it
  // creates at the shim slave port, proxied by its consumer-tile count. This is
  // the least-disruptive ranking key for packetization: fan-out, not data
  // volume, drives shim slave-port packet-rule slot pressure (a two-consumer
  // ingress needs more distinct switchbox slots than a one-consumer ingress, so
  // packetizing it is likelier to overflow the 4-slot arbiter when control
  // co-tenants K). A downstream aie.objectfifo.link redistributes from the
  // consumer tile (e.g. memtile) onward, NOT from the shim, so link fan-out is
  // not shim slave-port pressure and is deliberately excluded.
  static int ingressFanOut(ObjectFifoCreateOp ofo) {
    return (int)ofo.getConsumerTiles().size();
  }

  // Scan every config in the module and build the per-column union view. A
  // column is "covered" once any config has a shim-ingress objectFifo there;
  // manual aie.shim_mux reservations are recorded only for covered columns.
  static void buildColumnUnion(ModuleOp module,
                               llvm::MapVector<int, ColumnUnion> &unionByCol) {
    for (DeviceOp device : module.getOps<DeviceOp>()) {
      const auto &tm = device.getTargetModel();
      if (tm.getTargetArch() == AIEArch::AIE1)
        continue; // shim control overlay is AIE2+ only.
      llvm::SmallSet<int, 8> devColsSeen;
      for (auto ofo : device.getOps<ObjectFifoCreateOp>()) {
        TileOp prod = ofo.getProducerTileOp();
        if (!prod.isShimNOCorPLTile())
          continue; // only shim ingress competes for MM2S.
        int col = prod.colIndex();
        ColumnUnion &view = unionByCol[col];
        if (view.numMM2S == 0)
          view.numMM2S =
              tm.getNumSourceShimMuxConnections(col, 0, WireBundle::DMA);
        bool isPacket = (bool)ofo.getPacket();
        int explicitChanPin = -1;
        if (auto pin = ofo.getProdDmaChannel())
          explicitChanPin = *pin;
        if (!isPacket && explicitChanPin >= 0)
          view.dataPinned.insert(explicitChanPin);
        if (devColsSeen.insert(col).second)
          view.devices.push_back(device);
      }
    }
    // Manual aie.shim_mux reservations (mirrors the overlay's union-scan idiom:
    // a ConnectOp with a DMA source bundle hard-reserves that MM2S channel,
    // circuit routing control packets cannot time-share).
    for (DeviceOp device : module.getOps<DeviceOp>()) {
      const auto &tm = device.getTargetModel();
      if (tm.getTargetArch() == AIEArch::AIE1)
        continue;
      for (auto tile : device.getOps<TileOp>()) {
        if (!tm.isShimNOCTile(tile.colIndex(), tile.rowIndex()))
          continue;
        auto it = unionByCol.find(tile.colIndex());
        if (it == unionByCol.end())
          continue; // shim_mux on an un-covered column is not this pass's
                    // concern (no control ingress needed there yet).
        for (auto *user : tile.getResult().getUsers()) {
          auto muxOp = dyn_cast<ShimMuxOp>(user);
          if (!muxOp)
            continue;
          for (auto connectOp : muxOp.getOps<ConnectOp>())
            if (connectOp.getSourceBundle() == WireBundle::DMA)
              it->second.shimMuxBlocked.insert(connectOp.sourceIndex());
        }
      }
    }
  }

  // Choose ONE control trunk channel K for a column, consistent across every
  // config. Mirrors chooseCtrlShimChan's two-tier order in
  // AIEGenerateColumnControlOverlay.cpp: prefer a fully-free channel (no manual
  // routing AND no pinned data leg in any config), else the lowest channel free
  // of manual routing (control may co-tenant a packet-flipped data leg there;
  // each config is conformed to K). Returns -1 when manual shim_mux routing
  // covers every channel in some config, leaving no shareable trunk.
  static int chooseUnionTrunkChan(const ColumnUnion &view) {
    for (int c = 0; c < view.numMM2S; c++)
      if (!view.shimMuxBlocked.count(c) && !view.dataPinned.count(c))
        return c;
    for (int c = 0; c < view.numMM2S; c++)
      if (!view.shimMuxBlocked.count(c))
        return c;
    return -1;
  }

  // Confine the packet segment of a packetized shim-ingress leg to the
  // contested shim hop by synthesizing a local memtile relay. A single-hop shim
  // -> core packet leg rides the resident control overlay's switchbox arbiters
  // all the way to the core, which on a multi-column grid deadlocks the packet
  // arbiters. Splitting it into a shim ->
  // memtile PACKET fifo (this op, symbol retained so the runtime_sequence DMA
  // task and the K/packet pin stay valid) plus a NEW memtile -> core CIRCUIT
  // relay fifo, joined by an aie.objectfifo.link, drops the header at the
  // memtile master port so the data leg takes NO arbiter grant past the shim --
  // no data-side packet deadlock is possible regardless of placement / column
  // count / arbiter assignment. This is the same boundary split multi-hop
  // two-input designs already ride on device.
  //
  // No-op (the caller's bare setPacket stands) when the trunk already
  // terminates at a memtile (boundary split already present), when the target
  // has no memtile row, or for the rare strided/per-consumer-depth legs whose
  // split would need dim/depth redistribution (kept on today's path rather than
  // risk miscompiling them; no design that wedges here hits this).
  static void synthesizeMemtileRelay(DeviceOp device,
                                     ObjectFifoCreateOp trunk) {
    const auto &tm = device.getTargetModel();

    // The core consumers, captured before we retarget the trunk to the memtile.
    SmallVector<Value> cores(trunk.getConsumerTiles().begin(),
                             trunk.getConsumerTiles().end());
    if (cores.empty())
      return;
    // Already multi-hop: the leg terminates at a memtile, so the packet/circuit
    // boundary already exists and a bare setPacket confines it correctly.
    if (auto c0 = dyn_cast_or_null<TileOp>(cores.front().getDefiningOp()))
      if (c0.isMemTile())
        return;
    // Per-consumer depths (array elemNumber) or consumer-side stream dims would
    // both need redistribution across the split; leave those legs on the bare
    // packet path (unreached by designs that wedge here, all of which use a
    // scalar depth and no objectFifo-level consumer dims -- their striding
    // lives in the host DMA BD, which rides the retained shim producer fifo).
    if (isa<ArrayAttr>(trunk.getElemNumber()))
      return;
    // getDimensionsFromStreamPerConsumer() holds one (possibly-empty) entry per
    // consumer, so test the per-consumer lists, not the outer array.
    for (auto consDims : trunk.getDimensionsFromStreamPerConsumer())
      if (!consDims.empty())
        return;

    int col = trunk.getProducerTileOp().colIndex();
    int memRow = -1;
    for (int r = 0; r < tm.rows(); r++)
      if (tm.isMemTile(col, r)) {
        memRow = r;
        break;
      }
    if (memRow < 0)
      return; // no memtile in this column -> cannot relay; keep bare packet.

    MLIRContext *ctx = trunk.getContext();
    OpBuilder builder(trunk);
    TileOp memtile = TileOp::getOrCreate(builder, device, col, memRow);
    std::string relayName = (trunk.getSymName() + "_relay").str();

    // memtile -> cores CIRCUIT relay (no packet, no prod_dma_channel: K pins
    // the shim MM2S, not the memtile's). Same element type and (scalar) depth.
    builder.setInsertionPointAfter(trunk);
    ObjectFifoCreateOp::create(builder, trunk.getLoc(),
                               builder.getStringAttr(relayName),
                               memtile.getResult(), ValueRange(cores),
                               trunk.getElemNumberAttr(), trunk.getElemType());

    // The trunk keeps its symbol, {packet} and prod_dma_channel=K but now feeds
    // the memtile; the link forwards memtile -> cores as circuit.
    trunk.getConsumerTilesMutable().assign(memtile.getResult());
    ObjectFifoLinkOp::create(
        builder, trunk.getLoc(),
        builder.getArrayAttr({SymbolRefAttr::get(ctx, trunk.getSymName())}),
        builder.getArrayAttr({SymbolRefAttr::get(ctx, relayName)}),
        builder.getI64ArrayAttr({}), builder.getI64ArrayAttr({0}));

    // The cores now consume from the circuit relay, not the packet shim leg.
    StringRef trunkName = trunk.getSymName();
    auto relayRef = FlatSymbolRefAttr::get(ctx, relayName);
    device.walk([&](Operation *op) {
      if (auto acq = dyn_cast<ObjectFifoAcquireOp>(op)) {
        if (acq.getObjFifoName() == trunkName)
          acq.setObjFifoNameAttr(relayRef);
      } else if (auto rel = dyn_cast<ObjectFifoReleaseOp>(op)) {
        if (rel.getObjFifoName() == trunkName)
          rel.setObjFifoNameAttr(relayRef);
      }
    });
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Module-level pre-pass: pick one control trunk channel K per column that
    // is consistent across every config, and stamp it on each covered column's
    // row-0 shim tile so the overlay pass can assert against it. This
    // runs BEFORE the per-config flip loop; the flip is unchanged, so
    // single-config behavior is preserved.
    llvm::MapVector<int, ColumnUnion> unionByCol;
    buildColumnUnion(module, unionByCol);
    Builder builder(module.getContext());
    // The union-chosen control trunk channel K per covered column, consumed by
    // the per-config conform decision below.
    llvm::MapVector<int, int> trunkByCol;
    for (auto &kv : unionByCol) {
      int col = kv.first;
      ColumnUnion &view = kv.second;
      int K = chooseUnionTrunkChan(view);
      if (K < 0) {
        Operation *reporter =
            view.devices.empty()
                ? static_cast<Operation *>(module)
                : static_cast<Operation *>(view.devices.front());
        reporter->emitError()
            << "no shim mm2s channel is free of manual routing across all "
               "configs for column "
            << col
            << "; free a shim ingress or reduce manual shim_mux routing so a "
               "control trunk channel can be shared across all configs";
        return signalPassFailure();
      }
      trunkByCol[col] = K;
      IntegerAttr kAttr = builder.getI32IntegerAttr(K);
      for (DeviceOp device : module.getOps<DeviceOp>()) {
        if (device.getTargetModel().getTargetArch() == AIEArch::AIE1)
          continue;
        for (auto tile : device.getOps<TileOp>())
          if (tile.colIndex() == col && tile.rowIndex() == 0 &&
              tile.isShimNOCorPLTile())
            tile->setAttr("ctrl_pkt_trunk_chan", kAttr);
      }
    }

    for (DeviceOp device : module.getOps<DeviceOp>()) {
      const auto &tm = device.getTargetModel();
      if (tm.getTargetArch() == AIEArch::AIE1)
        continue; // shim control overlay is AIE2+ only; skip this device only

      // Every shim-ingress leg, per column, packet and circuit alike -- the
      // conform decision below picks which single leg K carries.
      llvm::MapVector<int, SmallVector<ObjectFifoCreateOp>> ingressByCol;
      for (auto ofo : device.getOps<ObjectFifoCreateOp>()) {
        TileOp prod = ofo.getProducerTileOp();
        if (!prod.isShimNOCorPLTile())
          continue; // only shim ingress competes for MM2S.
        ingressByCol[prod.colIndex()].push_back(ofo);
      }

      for (auto &kv : ingressByCol) {
        int col = kv.first;
        SmallVector<ObjectFifoCreateOp> &fifos = kv.second;
        auto kIt = trunkByCol.find(col);
        if (kIt == trunkByCol.end())
          continue; // no trunk was chosen for this column (every covered
                    // column got a K above; guard for a column that carries
                    // only egress or is otherwise uncovered).
        int K = kIt->second;
        int numMM2S =
            tm.getNumSourceShimMuxConnections(col, 0, WireBundle::DMA);
        if (numMM2S <= 1)
          continue; // a shim with <=1 MM2S can't host both a data leg and
                    // control anyway (unreachable today: AIE2 shim DMA
                    // reports numMM2S == 2).

        // Deterministic order by symbol name; the least-disruptive tie-break
        // below relies on it to pick the same leg across configs.
        llvm::sort(fifos, [](ObjectFifoCreateOp a, ObjectFifoCreateOp b) {
          return a.getSymName() < b.getSymName();
        });

        // Conform this config to the union trunk K. K carries at
        // most one leg, and only a PACKET leg -- so control ingress can
        // co-tenant K in every config. Pick that leg (the "trunk"):
        //  - an existing packet leg (author's or a prior run's) is the trunk;
        //    it swaps onto K and nothing new is packetized;
        //  - else, only when the column is saturated (every channel would
        //    otherwise carry a circuit data leg, leaving none free for
        //    control) is the least-disruptive circuit leg packetized onto K --
        //    the FEWEST fan-out (fewest shim slave-port destinations),
        //    tie-broken to the smallest trafficProxy, then the trailing symbol
        //    name for determinism;
        //  - else K already has a free channel, so no leg is packetized and no
        //    leg is the trunk; every circuit leg is simply pinned off K.
        ObjectFifoCreateOp trunk = nullptr;
        for (auto ofo : fifos)
          if (ofo.getPacket()) {
            trunk = ofo;
            break;
          }
        if (!trunk && (int)fifos.size() >= numMM2S) {
          int bestFan = 0;
          int64_t bestTp = 0;
          for (auto ofo : fifos) {
            int fan = ingressFanOut(ofo);
            int64_t tp = ingressTrafficProxy(ofo);
            // fifos is name-ascending, so `<=` on the trafficProxy tie keeps
            // the trailing symbol name; fewest fan-out wins first, then
            // smallest trafficProxy.
            if (!trunk || fan < bestFan || (fan == bestFan && tp <= bestTp)) {
              trunk = ofo;
              bestFan = fan;
              bestTp = tp;
            }
          }
          trunk.setPacket(true);
          trunk.emitWarning()
              << "auto-packetized objectFifo '" << trunk.getSymName()
              << "' on column " << col
              << " from circuit -> packet for resident control coexistence ("
              << fifos.size() << " circuit shim-ingress legs on column " << col
              << " leave no free channel for the control overlay)";
        }

        // Pin the trunk onto K and every remaining (circuit) leg onto a non-K
        // channel, round-robin. K then carries either nothing or the single
        // packet trunk; the other channel(s) carry the remaining circuit legs.
        SmallVector<int> nonK;
        for (int c = 0; c < numMM2S; c++)
          if (c != K)
            nonK.push_back(c);
        if (trunk)
          trunk->setAttr("prod_dma_channel", builder.getI32IntegerAttr(K));
        // Confine the packet segment to the shim hop: for a single-hop trunk,
        // synthesize a memtile relay so the data leg goes circuit to its
        // core(s) and never rides the control overlay's arbiters to the compute
        // tile.
        if (trunk)
          synthesizeMemtileRelay(device, trunk);
        // This round-robin assumes no manual shim_mux or data-pin already
        // occupies a non-K channel in this config; an objectFifo+shim_mux mix
        // on the same column is Phase-2, out of the Phase-1 contract here --
        // downstream assignChannels surfaces a conflict if that ever occurs.
        // More than numMM2S circuit legs on the column (e.g. >2 on a 2-MM2S
        // target) is out-of-contract oversubscription, unreached on npu2
        // where numMM2S == 2.
        unsigned idx = 0;
        for (auto ofo : fifos) {
          if (ofo == trunk)
            continue;
          int chan = nonK[idx % nonK.size()];
          ofo->setAttr("prod_dma_channel", builder.getI32IntegerAttr(chan));
          idx++;
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::AIE::createAIEAutoPacketizeControlIngressPass() {
  return std::make_unique<AIEAutoPacketizeControlIngressPass>();
}
