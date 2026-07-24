//===- AIELowerDmaChannelResetFor.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIELOWERDMACHANNELRESETFOR
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-lower-dma-channel-reset-for"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

namespace {

// One resident endpoint resolved from the binding, ready to re-arm.
struct ResolvedEndpoint {
  Value tile;
  DMAChannelDir dir;
  int channel;
  int col;
  int row;
  uint32_t headBdId;
  int32_t repeatCount; // already the N-1 biased value on the resident dma_start
};

// The outcome of resolving a resident channel: either found (with head BD id +
// repeat), or a specific reason it could not be, for a precise diagnostic.
enum class ResolveResult { Found, NoChannel, NoBdId };

// Find the resident DMA channel (dir, channel) on `tile` and read its head BD
// id and (biased) repeat count from the placed aie.mem / aie.memtile_dma chain,
// exactly as the load-time start-queue push does. Note: only the block form
// (aie.dma_start) is scanned; the objectFIFO transform always emits that form.
static ResolveResult resolveResidentChannel(DeviceOp device, TileOp tile,
                                            DMAChannelDir dir, int channel,
                                            uint32_t &headBdId,
                                            int32_t &repeatCount) {
  auto scanRegion = [&](Region &region) -> ResolveResult {
    for (Block &block : region) {
      for (DMAStartOp startOp : block.getOps<DMAStartOp>()) {
        if (startOp.getChannelDir() != dir ||
            startOp.getChannelIndex() != channel)
          continue;
        Block *dest = startOp.getDest();
        if (!dest)
          return ResolveResult::NoChannel;
        auto bds = dest->getOps<DMABDOp>();
        if (bds.empty())
          return ResolveResult::NoChannel;
        DMABDOp head = *bds.begin();
        if (!head.getBdId().has_value())
          return ResolveResult::NoBdId;
        headBdId = head.getBdId().value();
        repeatCount = startOp.getRepeatCount();
        return ResolveResult::Found;
      }
    }
    return ResolveResult::NoChannel;
  };
  for (MemOp mem : device.getOps<MemOp>())
    if (mem.getTile() == tile.getResult())
      return scanRegion(mem->getRegion(0));
  for (MemTileDMAOp mem : device.getOps<MemTileDMAOp>())
    if (mem.getTile() == tile.getResult())
      return scanRegion(mem->getRegion(0));
  return ResolveResult::NoChannel;
}

struct AIELowerDmaChannelResetForPass
    : public xilinx::AIEX::impl::AIELowerDmaChannelResetForBase<
          AIELowerDmaChannelResetForPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();
    SymbolTable symbolTable(device);

    SmallVector<DmaChannelResetForOp> toErase;
    WalkResult res = device.walk([&](DmaChannelResetForOp op) -> WalkResult {
      auto binding =
          symbolTable.lookup<ObjectFifoRearmBindingOp>(op.getObjfifo());
      if (!binding) {
        op.emitOpError("could not resolve '")
            << op.getObjfifo()
            << "' to an aie.objectfifo_rearm_binding; run "
               "--aie-objectFifo-stateful-transform first";
        return WalkResult::interrupt();
      }

      ArrayRef<int32_t> dirs = binding.getChannelDirs();
      ArrayRef<int32_t> chans = binding.getChannelIndices();
      ValueRange tiles = binding.getChannelTiles();

      OpBuilder builder(op);
      Location loc = op.getLoc();
      MLIRContext *ctx = builder.getContext();

      // Resolve every endpoint up front so a failure aborts before we emit
      // anything for this op.
      SmallVector<ResolvedEndpoint> endpoints;
      for (unsigned i = 0; i < tiles.size(); ++i) {
        auto tileOp = tiles[i].getDefiningOp<TileOp>();
        if (!tileOp) {
          op.emitOpError("re-arm binding channel tile is not an aie.tile");
          return WalkResult::interrupt();
        }
        DMAChannelDir dir =
            dirs[i] == 0 ? DMAChannelDir::S2MM : DMAChannelDir::MM2S;
        ResolvedEndpoint ep;
        ep.tile = tiles[i];
        ep.dir = dir;
        ep.channel = chans[i];
        ep.col = tileOp.getCol();
        ep.row = tileOp.getRow();
        switch (resolveResidentChannel(device, tileOp, dir, chans[i],
                                       ep.headBdId, ep.repeatCount)) {
        case ResolveResult::Found:
          break;
        case ResolveResult::NoChannel:
          op.emitOpError(
              "could not find a resident DMA channel for endpoint on "
              "tile (")
              << ep.col << ", " << ep.row << ")";
          return WalkResult::interrupt();
        case ResolveResult::NoBdId:
          op.emitOpError("the resident DMA channel for endpoint on tile (")
              << ep.col << ", " << ep.row
              << ") has no assigned BD id; run --aie-assign-bd-ids first";
          return WalkResult::interrupt();
        }
        endpoints.push_back(ep);
      }

      // Guard the lock operands (the verifier already requires them to be
      // aie.lock, but stay defensive so a malformed binding cannot make the
      // set_lock builder cast a non-lock operand and abort).
      for (Value lock : binding.getLocks())
        if (!lock.getDefiningOp<AIE::LockOp>()) {
          op.emitOpError("re-arm binding lock operand is not an aie.lock");
          return WalkResult::interrupt();
        }

      // The re-arm, ordered so a channel is only restarted once its locks are
      // re-armed:
      //   1. reset every channel (drain the queue, clear the run FSM),
      //   2. re-arm the fifo locks to their init values,
      //   3. re-push each channel's start queue (required on aie2p: a DMA
      //      channel has no enable bit, so the only way to restart it is a
      //      START_QUEUE write).

      // 1. dma_channel_reset per endpoint.
      for (const ResolvedEndpoint &ep : endpoints)
        DmaChannelResetOp::create(builder, loc, ep.tile,
                                  DMAChannelDirAttr::get(ctx, ep.dir),
                                  builder.getI32IntegerAttr(ep.channel));

      // 2. set_lock per lock, to the value the fifo was armed with.
      ValueRange locks = binding.getLocks();
      ArrayRef<int32_t> lockInits = binding.getLockInits();
      for (unsigned j = 0; j < locks.size(); ++j)
        SetLockOp::create(builder, loc, locks[j],
                          builder.getI32IntegerAttr(lockInits[j]));

      // 3. START_QUEUE re-push per endpoint, emitted as aiex.npu.push_queue so
      // the command-word encoding, the queue address, and the bd_id/repeat
      // range checks all live in one place -- the aie-dma-to-npu lowering and
      // NpuPushQueueOp::verify -- reproducing the channel's load-time arming
      // exactly instead of hand-rolling it here. This pass therefore runs
      // before aie-dma-to-npu (see getNpuDmaLoweringPipeline). The token bit is
      // only set for a shim S2MM channel; these endpoints are core/mem, so it
      // is always false.
      for (const ResolvedEndpoint &ep : endpoints) {
        bool issueToken = ep.row == 0 && ep.dir == DMAChannelDir::S2MM;
        // Materialize the operands in fixed statements before the create() so
        // their emission order is deterministic (function-argument evaluation
        // order is unspecified in C++).
        Value repeatVal = createConstantI32(
            builder, loc, static_cast<uint32_t>(ep.repeatCount));
        Value bdVal = createConstantI32(builder, loc, ep.headBdId);
        NpuPushQueueOp::create(builder, loc, builder.getI32IntegerAttr(ep.col),
                               builder.getI32IntegerAttr(ep.row),
                               DMAChannelDirAttr::get(ctx, ep.dir),
                               builder.getI32IntegerAttr(ep.channel),
                               builder.getBoolAttr(issueToken), repeatVal,
                               bdVal);
      }

      toErase.push_back(op);
      return WalkResult::advance();
    });

    if (res.wasInterrupted())
      return signalPassFailure();

    for (DmaChannelResetForOp op : toErase)
      op.erase();

    // Drop re-arm bindings that are now unreferenced (their reset_for users
    // were lowered). They are device-body metadata like
    // aie.shim_dma_allocation; erasing the dead ones keeps the lowered module
    // clean.
    SmallVector<ObjectFifoRearmBindingOp> deadBindings;
    for (auto binding : device.getOps<ObjectFifoRearmBindingOp>())
      if (SymbolTable::symbolKnownUseEmpty(binding.getSymNameAttr(), device))
        deadBindings.push_back(binding);
    for (ObjectFifoRearmBindingOp binding : deadBindings)
      binding.erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIELowerDmaChannelResetForPass() {
  return std::make_unique<AIELowerDmaChannelResetForPass>();
}
