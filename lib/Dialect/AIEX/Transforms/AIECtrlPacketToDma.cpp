//===- AIECtrlPacketToDma.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEGenerateColumnControlOverlay.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/CtrlPktUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIECTRLPACKETTODMA
#define GEN_PASS_DEF_AIECTRLPACKETINFERTILES
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-ctrl-packet-to-dma"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

// Control-packet transfer size in i32 words: the payload (data element count or
// explicit length) plus the control info word and the packet header.
static int64_t ctrlPacketSize(NpuControlPacketOp op) {
  int64_t sz = 0;
  auto data = op.getData();
  auto length = op.getLength();
  if (data)
    sz = data->size();
  else if (length)
    sz = *length;
  sz++; // Ctrl info word
  sz++; // Packet header
  return sz;
}

struct AIECtrlPacketInferTilesPass
    : xilinx::AIEX::impl::AIECtrlPacketInferTilesBase<
          AIECtrlPacketInferTilesPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    OpBuilder devBuilder = OpBuilder::atBlockBegin(device.getBody());

    // The tile -> controller-id map depends only on the target model; build it
    // once for the whole device rather than once per control packet.
    auto tileIDMap = getTileToControllerIdMap(true, targetModel);
    auto sequenceOps = device.getOps<AIE::RuntimeSequenceOp>();
    for (auto f : sequenceOps) {
      auto ctrlPktOps = f.getOps<AIEX::NpuControlPacketOp>();
      for (auto ctrlPktOp : ctrlPktOps) {
        auto tOp = TileOp::getOrCreate(devBuilder, device,
                                       (int)ctrlPktOp.getColumnFromAddr(),
                                       (int)ctrlPktOp.getRowFromAddr());
        // Assign controller id
        if (tOp->hasAttr("controller_id"))
          continue;
        auto pktInfoAttr = AIE::PacketInfoAttr::get(
            tOp->getContext(), /*pkt_type*/ 0,
            /*pkt_id*/ tileIDMap[{tOp.colIndex(), tOp.rowIndex()}]);
        tOp->setAttr("controller_id", pktInfoAttr);
      }
    }
  }
};

struct AIECtrlPacketToDmaPass
    : xilinx::AIEX::impl::AIECtrlPacketToDmaBase<AIECtrlPacketToDmaPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    const auto &targetModel = device.getTargetModel();
    auto *ctx = device->getContext();
    auto loc = device->getLoc();

    if (targetModel.getTargetArch() == AIEArch::AIE1)
      return; // Disable this pass for AIE1; AIE1 support NYI.

    // Resolve control's shim allocation by physical (col, row, dir, chan)
    // rather than assuming the `ctrlpkt_col<col>_mm2s_chan<chan>` naming
    // convention: AIEGenerateColumnControlOverlay may have reused an
    // existing data aie.shim_dma_allocation (arbitrary symbol name) for a
    // channel shared between control and data instead of materializing a
    // `ctrlpkt_...` allocation of its own. Built once per device -- shim
    // allocations are already all present by this point in the pipeline.
    std::map<std::tuple<int, int, int, int>, StringRef> shimAllocByLoc;
    for (auto allocOp : device.getOps<AIE::ShimDMAAllocationOp>()) {
      AIE::TileOp tile = allocOp.getTileOp();
      shimAllocByLoc[{tile.colIndex(), tile.rowIndex(),
                      (int)allocOp.getChannelDir(),
                      (int)allocOp.getChannelIndex()}] = allocOp.getSymName();
    }

    // (col, row) -> shim MM2S channel the control overlay chose for this
    // controlled tile, recorded by AIEGenerateColumnControlOverlay as the
    // `ctrl_pkt_shim_chan` attribute. Read this instead of recomputing the
    // fixed round-robin map: occupancy-aware channel selection may have
    // relocated control off the mandated channel, and recomputing would then
    // resolve the wrong allocation (dangling symbol) and wait on the wrong
    // completion channel (host hang).
    std::map<std::pair<int, int>, int> ctrlChanByColRow;
    for (auto tileOp : device.getOps<AIE::TileOp>())
      if (auto a = tileOp->getAttrOfType<IntegerAttr>("ctrl_pkt_shim_chan"))
        ctrlChanByColRow[{tileOp.colIndex(), tileOp.rowIndex()}] =
            (int)a.getInt();

    // Resolve a controlled tile's shim MM2S channel and the allocation that
    // owns it: prefer the channel the overlay actually chose (occupancy-aware
    // selection may have relocated control off the fixed round-robin map), fall
    // back to the fixed map otherwise, then look up whatever allocation owns
    // that physical (col, 0, MM2S, chan) -- the auto-generated `ctrlpkt_...`
    // one, or a reused data allocation when the channel is shared. Shared by
    // the serial and column-parallel delivery paths so their source offsets
    // agree.
    auto resolveShimAlloc = [&](int col,
                                int row) -> std::pair<int, std::string> {
      int shimChan;
      auto cIt = ctrlChanByColRow.find({col, row});
      if (cIt != ctrlChanByColRow.end())
        shimChan = cIt->second;
      else
        shimChan = getRowToShimChanMap(targetModel, WireBundle::DMA)[row];

      auto it = shimAllocByLoc.find(
          {col, 0, (int)AIE::DMAChannelDir::MM2S, shimChan});
      std::string allocName;
      if (it != shimAllocByLoc.end()) {
        allocName = it->second.str();
      } else {
        // Should not happen: AIEGenerateColumnControlOverlay always
        // materializes or reuses an allocation for this channel. Fail loud in
        // asserts-enabled builds (a synthesized name here would dangle if no
        // matching allocation exists); fall back to the historical
        // deterministic name otherwise so a release build still emits a
        // best-effort reference rather than crashing.
        assert(false && "ctrl-packet channel has no shim_dma_allocation; "
                        "AIEGenerateColumnControlOverlay should have "
                        "materialized or reused one");
        allocName = "ctrlpkt";
        allocName += "_col" + std::to_string(col);
        allocName += "_mm2s";
        allocName += "_chan" + std::to_string(shimChan);
      }
      return {shimChan, allocName};
    };

    SmallVector<Operation *> erased;
    // Monotonic suffix keeping every emitted `aie.bd_chain` symbol unique
    // across phases and across runtime sequences on this device.
    int chainCounter = 0;
    auto sequenceOps = device.getOps<AIE::RuntimeSequenceOp>();
    for (auto f : sequenceOps) {

      auto controlPacketOps = f.getOps<AIEX::NpuControlPacketOp>();
      if (controlPacketOps.empty())
        continue;

      OpBuilder builder(f);

      IRMapping mapping;

      auto newSeq = AIE::RuntimeSequenceOp::create(
          builder, loc, f.getSymNameAttr(), BoolAttr{});
      newSeq.getBody().push_back(new Block);

      // Copy the arguments from the old sequence to the new one.
      for (auto arg : f.getBody().getArguments()) {
        // Add the argument to the new sequence.
        auto newArg = newSeq.getBody().addArgument(arg.getType(), arg.getLoc());
        // Replace all uses of the old argument with the new one.
        arg.replaceAllUsesWith(newArg);
        // Add the mapping for the argument.
        mapping.map(arg, newArg);
      }

      // Using dynamic shape for ctrl pkt stream.
      auto ctrlPktMemrefType = MemRefType::get(
          ShapedType::kDynamic, IntegerType::get(ctx, 32), nullptr, nullptr);
      auto newBlockArg = newSeq.getBody().addArgument(ctrlPktMemrefType, loc);

      builder.setInsertionPointToStart(&newSeq.getBody().front());

      Block &entry = f.getBody().front();

      // Opt-in overlay-gated parallel-columns delivery. Requires both the
      // `parallel-columns` option AND the ctrl-pkt overlay (the chained
      // schedule below relies on a resident control overlay giving each column
      // its own shim MM2S trunk): with either absent, fall through to the
      // unmodified serial per-(col,row) emission so flag-off stays
      // byte-identical. The engaged path delivers each column's per-tile
      // control BDs as one per-column `next_bd`-linked `aie.bd_chain` (one
      // linear BD per tile), issued with one push and one deferred await per
      // column per phase. The independent per-column shim MM2S channels overlap
      // because each column rides its own trunk; the config phase is drained
      // before the enable phase.
      bool engage =
          clParallelColumns &&
          device->hasAttrOfType<BoolAttr>("has_ctrl_pkt_overlay") &&
          device->getAttrOfType<BoolAttr>("has_ctrl_pkt_overlay").getValue();

      if (!engage) {
        // ------------------------------------------------------------------
        // Existing serial path (flag off / no overlay): unchanged.
        // ------------------------------------------------------------------

        // Collect all npu.control_packet ops, grouped by location in
        // 'batches'
        struct BatchInfo {
          TileID tileId;
          int64_t startOffset;
          int64_t totalSize;
          std::string shimDmaAllocName;
          int shimChan;
          Operation *first;
        };
        std::vector<BatchInfo> batches;

        int64_t ddrOffset = 0;

        // First pass: collect and batch control packet operations
        bool new_batch = true;
        for (Operation &o : entry) {
          auto ctrlPktOp = dyn_cast<NpuControlPacketOp>(&o);

          // A non-control_packet op ends the current batch
          if (!ctrlPktOp) {
            new_batch = true;
            continue;
          }
          int col = ctrlPktOp.getColumnFromAddr();
          int row = ctrlPktOp.getRowFromAddr();

          int64_t ctrlPktSize = ctrlPacketSize(ctrlPktOp);

          // Check if we can batch with the previous packet
          if (targetModel.getTargetArch() == AIEArch::AIE2p && !new_batch &&
              batches.back().tileId == TileID{col, row}) {
            // Add to existing batch
            batches.back().totalSize += ctrlPktSize;
          } else {
            // Start a new batch on the channel + allocation this tile resolves
            // to (shared with the column-parallel path).
            auto [shimChan, shimDmaAllocName] = resolveShimAlloc(col, row);
            batches.push_back({TileID{col, row}, ddrOffset, ctrlPktSize,
                               shimDmaAllocName, shimChan, &o});
            new_batch = false;
          }
          ddrOffset += ctrlPktSize;
        }

        // Second pass: emit batched operations in original order
        auto batchIt = batches.begin();

        for (Operation &o : entry) {
          auto ctrlPktOp = dyn_cast<NpuControlPacketOp>(&o);
          if (!ctrlPktOp) {
            builder.clone(o, mapping);
            continue;
          }

          // There are no more control packet batches to emit
          if (batchIt == batches.end())
            continue;

          // Check if this is the first packet of a new batch, otherwise skip
          // it.
          if (batchIt->first != &o)
            continue;

          int col = ctrlPktOp.getColumnFromAddr();

          // Emit the batched DMA operation for this (col, row) pair
          const std::vector<int64_t> staticOffsets = {0, 0, 0,
                                                      batchIt->startOffset};
          const std::vector<int64_t> staticSizes = {1, 1, 1,
                                                    batchIt->totalSize};
          const std::vector<int64_t> staticStrides = {0, 0, 0, 1};

          SymbolRefAttr metadata = SymbolRefAttr::get(
              builder.getContext(), batchIt->shimDmaAllocName);
          NpuDmaMemcpyNdOp::create(
              builder, loc, newBlockArg, SmallVector<Value>{},
              SmallVector<Value>{}, SmallVector<Value>{},
              ArrayRef(staticOffsets), ArrayRef(staticSizes),
              ArrayRef(staticStrides), nullptr, metadata, 0, true, 0, 0, 0, 0,
              0, 0,
              /*burst_length=*/0,
              /*axcache=*/IntegerAttr(),
              /*offset_parameter=*/FlatSymbolRefAttr(),
              /*offset_state_table_idx=*/IntegerAttr());

          Value shimRow = AIEX::createConstantI32(builder, loc, 0);
          Value shimCol = AIEX::createConstantI32(builder, loc, col);
          Value dir = AIEX::createConstantI32(builder, loc, 1); // MM2S
          Value chan = AIEX::createConstantI32(builder, loc, batchIt->shimChan);
          Value col_num = AIEX::createConstantI32(builder, loc, 1);
          Value row_num = AIEX::createConstantI32(builder, loc, 1);
          AIEX::NpuSyncOp::create(builder, loc, shimCol, shimRow, dir, chan,
                                  col_num, row_num);
          ++batchIt;
        }
      } else {
        // ------------------------------------------------------------------
        // Overlay-gated two-phase per-column collapse (design correction:
        // scoped to the LEADING contiguous control-packet run only -- a real
        // ctrlpkt config sequence is not a pure control-packet run; an
        // overlay preamble (load_pdi) may lead it and an app-run plus a
        // teardown control-packet block legitimately follow. The app-run is
        // walked in place below (trailing-region loop): app ops are cloned
        // verbatim in program order, and each maximal run of teardown control
        // packets encountered along the way is delivered as its own parallel
        // phase -- a per-column bd_chain (buildGroups + emitPhase, the same
        // machinery used for the leading config run) issued and awaited right
        // there, in place, rather than hoisted out or left serial. This keeps
        // ordering intact w.r.t. any op following the teardown, and continues
        // the SAME program-order ddrOffset space so DMA source offsets stay
        // consistent with the .ctrldata payload (AIETargetNPU walks program
        // order).
        // ------------------------------------------------------------------

        // ctrlPacketSize / resolveShimAlloc are shared with the serial path
        // (the file-scope helper and the device-scope lambda above). A column
        // whose rows resolve to different channels is caught by the per-column
        // fail-loud guard in pushColumnChains (a per-column bd_chain requires
        // one channel per column), so a column's channel/alloc is well-defined
        // by its first tile.

        // Locate the leading contiguous control-packet run: skip any
        // preamble (e.g. a load_pdi standing up the overlay) to reach the
        // first control packet; the run ends at the first non-control op
        // after it (an app DMA, a teardown control-packet block, or the end
        // of the sequence).
        Operation *runStart = nullptr;
        for (Operation &o : entry)
          if (isa<NpuControlPacketOp>(&o)) {
            runStart = &o;
            break;
          }
        assert(runStart &&
               "controlPacketOps non-empty implies a leading control packet");

        Operation *runEnd = runStart;
        while (isa_and_nonnull<NpuControlPacketOp>(runEnd))
          runEnd = runEnd->getNextNode();

        // Clone the preamble (ops strictly before the leading run, if any)
        // verbatim, same as the serial path's unconditional clone-every-
        // non-control-op walk over the whole block.
        for (Operation &o : entry) {
          if (&o == runStart)
            break;
          builder.clone(o, mapping);
        }

        // Group the leading run into one linear DMA per controlled TILE
        // (col,row), partitioned at the first core-enable into a config region
        // and a trailing enable region. One BD per tile is a hardware routing
        // REQUIREMENT, not a heuristic: a shim control BD is one
        // TLAST-delimited AXI-stream packet, and the stream switch routes that
        // packet to a single tile by the first header after the preceding TLAST
        // (AIE2P ArchSpec sec.4.3.2 Master Port Drop Header; sec.3.7.5
        // TLAST_Suppress is a per-BD field). TLAST is a stream sideband, not a
        // payload word (sec.4.3.3), so the only way to raise TLAST at a tile
        // boundary is to end a BD there. A BD spanning >1 tile routes the whole
        // transfer to the first tile and leaves the rest unconfigured --
        // verified by decoding the payload (per-tile BD boundaries align
        // exactly with the pkt_id boundaries) and on device (collapsing to 1
        // BD/col yields all-zero output; merging even 2 tiles/BD fails).
        // Multiple contiguous SAME-tile packets still pack into one linear BD
        // (TLAST-suppressed between them, one terminal TLAST at the tile
        // boundary -- the throughput win the serial path already uses). Column
        // parallelism comes from one per-column `bd_chain` issued with
        // `issue_token` and a deferred completion await, which lets the
        // independent per-column shim MM2S channels overlap.
        struct TileGroup {
          int col;
          int row;
          int64_t startOffset;
          int64_t totalSize;
          int shimChan;
          std::string allocName;
        };
        std::vector<TileGroup> configGroups, enableGroups;
        int64_t ddrOffset = 0;

        // Merge a control-packet run [begin,end) into per-tile TileGroups
        // (contiguous same-tile packets coalesce into one linear BD), advancing
        // the shared ddrOffset. When splitAtEnable, packets at/after the first
        // core-enable go to `enable`; otherwise everything goes to `primary`.
        auto buildGroups = [&](Operation *begin, Operation *end,
                               bool splitAtEnable,
                               std::vector<TileGroup> &primary,
                               std::vector<TileGroup> &enable) {
          bool sawEnable = false;
          for (Operation *o = begin; o != end; o = o->getNextNode()) {
            auto cp = cast<NpuControlPacketOp>(o);
            if (splitAtEnable && !sawEnable && isCoreEnableControlPacket(cp))
              sawEnable = true;
            std::vector<TileGroup> &cur =
                (splitAtEnable && sawEnable) ? enable : primary;
            int col = cp.getColumnFromAddr();
            int row = cp.getRowFromAddr();
            int64_t sz = ctrlPacketSize(cp);
            auto [shimChan, allocName] = resolveShimAlloc(col, row);

            // Merge only CONTIGUOUS same-tile packets into one linear BD, and
            // only on AIE2P: same-tile packing rides AIE2P TLAST_Suppress
            // (ArchSpec 3.7.5), which AIE2/npu1 lacks, so the serial path gates
            // the identical merge on AIE2P too (see the batching guard above)
            // and npu1 keeps one BD per packet. Any change of tile (or a re-
            // appearance after another tile) opens a new per-tile BD. No
            // column-contiguity is assumed, so no sort is needed.
            if (targetModel.getTargetArch() == AIEArch::AIE2p && !cur.empty() &&
                cur.back().col == col && cur.back().row == row)
              cur.back().totalSize += sz;
            else
              cur.push_back({col, row, ddrOffset, sz, shimChan, allocName});
            ddrOffset += sz;
          }
        };

        buildGroups(runStart, runEnd, /*splitAtEnable=*/true, configGroups,
                    enableGroups);

        // Build one column's `aie.bd_chain`: a linear no-dims `aie.dma_bd` per
        // controlled tile (offset/len drawn straight from the TileGroup, so the
        // payload words each BD reads are byte-identical to the serial memcpy
        // emission), `aie.next_bd`-linked head-to-tail, terminated by
        // `aie.end`. Each BD raises its own terminal TLAST (the default), which
        // is the per-tile routing the hardware requires; the whole column is
        // delivered by ONE queue-push that auto-advances the chain, one
        // completion sync. The chain def lives at device scope (before the new
        // sequence).
        auto buildColumnChain =
            [&](ArrayRef<TileGroup *> tiles) -> AIE::BDChainOp {
          OpBuilder devBuilder(newSeq);
          std::string name =
              "ctrlpkt_bd_chain_" + std::to_string(chainCounter++);
          auto chain = AIE::BDChainOp::create(devBuilder, loc,
                                              devBuilder.getStringAttr(name));
          Region &body = chain.getBody();
          Block *entry = devBuilder.createBlock(&body, body.begin(),
                                                {ctrlPktMemrefType}, {loc});
          Value chainArg = entry->getArgument(0);
          SmallVector<Block *> blocks;
          blocks.push_back(entry);
          for (size_t i = 1; i < tiles.size(); ++i)
            blocks.push_back(devBuilder.createBlock(&body, body.end()));
          for (size_t i = 0; i < tiles.size(); ++i) {
            devBuilder.setInsertionPointToEnd(blocks[i]);
            AIE::DMABDOp::create(devBuilder, loc, chainArg,
                                 (int)tiles[i]->startOffset,
                                 (int)tiles[i]->totalSize);
            if (i + 1 < tiles.size())
              AIE::NextBDOp::create(devBuilder, loc, blocks[i + 1]);
            else
              AIE::EndOp::create(devBuilder, loc);
          }
          return chain;
        };

        // Emit one `aiex.dma_start_bd_chain_for @chain(%payload) for @alloc`
        // per column (issue_token = true -> a single chain-tail TCT). Returns
        // the pushed task values so the caller can defer the awaits (all pushes
        // first, then all awaits) and let independent columns overlap.
        auto pushColumnChains =
            [&](std::vector<TileGroup> &groups) -> SmallVector<Value> {
          std::map<int, std::vector<TileGroup *>> byCol;
          std::vector<int> colOrder;
          for (auto &g : groups) {
            if (byCol.find(g.col) == byCol.end())
              colOrder.push_back(g.col);
            byCol[g.col].push_back(&g);
          }
          SmallVector<Value> tasks;
          for (int col : colOrder) {
            auto &tiles = byCol[col];
            // Fail-loud: one `dma_start_bd_chain_for @chain(%payload) for
            // @alloc` runs on ONE shim DMA channel/alloc, so a column whose
            // rows resolved to different channels/allocations has no
            // well-defined single chain endpoint. resolveShimAlloc keys on
            // (col,row), so this CAN happen; the serial path tolerated it
            // (per-tile DMA on its own alloc) but the chain cannot. Do not
            // silently pick the first tile's alloc.
            int chan0 = tiles.front()->shimChan;
            StringRef alloc0 = tiles.front()->allocName;
            for (auto *t : tiles)
              if (t->shimChan != chan0 || StringRef(t->allocName) != alloc0) {
                device.emitError()
                    << "control-packet column " << col
                    << " resolves to multiple shim channels/allocations; a "
                       "per-column bd_chain requires a single channel per "
                       "column";
                signalPassFailure();
                return tasks;
              }
            AIE::BDChainOp chain = buildColumnChain(tiles);
            auto startOp = AIEX::DMAStartBdChainForOp::create(
                builder, loc, builder.getIndexType(), chain.getSymName(),
                ValueRange{newBlockArg}, alloc0,
                /*issue_token=*/true, /*repeat_count=*/0);
            tasks.push_back(startOp.getResult());
          }
          return tasks;
        };

        // One phase: push every column's chain (issue_token) then defer all
        // awaits, so independent per-column shim MM2S channels overlap.
        auto emitPhase = [&](std::vector<TileGroup> &groups) {
          SmallVector<Value> tasks = pushColumnChains(groups);
          for (Value t : tasks)
            AIEX::DMAAwaitTaskOp::create(builder, loc, t);
        };

        // Two-phase per-column chained emission. Config first, fully drained,
        // then the enable phase (enable-last is the one load-bearing barrier --
        // enables only after all config has landed). Each phase pushes every
        // column's next_bd chain, then defers all completion syncs so the
        // independent per-column shim MM2S channels overlap. Peak live control
        // BDs per column is the physical tile count (<= 6, a shim + a memtile +
        // <= 4 cores), well under the clean per-column device depth. The
        // allocator (AIEAssignRuntimeSequenceBDIDs) remains the authoritative
        // fail-loud on real over-budget, counting ALL co-resident live BDs on
        // the shim.
        emitPhase(configGroups); // config first, fully drained
        emitPhase(enableGroups); // then enable (enable-last barrier)

        // Trailing region = app-run ops interleaved with the teardown control-
        // packet run(s). Emit IN PROGRAM ORDER (like the serial path it
        // replaces): clone app ops verbatim; when a maximal run of teardown
        // control packets appears, deliver it in place as a parallel phase
        // (per-column chains, push-all/await-all) via the same buildGroups/
        // emitPhase used for config. Emitting in place (not hoisting to the
        // block end) preserves order w.r.t. any op that follows the teardown
        // (e.g. dma_free_task), so no clean-trailing assumption or guard is
        // needed. ddrOffset continues from the config/enable phases.
        for (Operation *o = runEnd; o;) {
          if (!isa<NpuControlPacketOp>(o)) {
            builder.clone(*o, mapping);
            o = o->getNextNode();
            continue;
          }
          Operation *end = o;
          while (isa_and_nonnull<NpuControlPacketOp>(end))
            end = end->getNextNode();
          std::vector<TileGroup> teardownGroups, teardownUnused;
          // teardownUnused is a throwaway: with splitAtEnable=false this
          // single-phase teardown never sets sawEnable, so the `enable`
          // output vector is never written.
          buildGroups(o, end, /*splitAtEnable=*/false, teardownGroups,
                      teardownUnused);
          emitPhase(teardownGroups);
          o = end;
        }
      }

      erased.push_back(f);
    }

    for (auto *e : erased)
      e->erase();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIEX::createAIECtrlPacketInferTilesPass() {
  return std::make_unique<AIECtrlPacketInferTilesPass>();
}
std::unique_ptr<OperationPass<DeviceOp>> AIEX::createAIECtrlPacketToDmaPass() {
  return std::make_unique<AIECtrlPacketToDmaPass>();
}
