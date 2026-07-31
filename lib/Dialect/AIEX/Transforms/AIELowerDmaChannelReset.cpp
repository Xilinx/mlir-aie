//===- AIELowerDmaChannelReset.cpp ------------------------------*- C++ -*-===//
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
#include "mlir/Transforms/DialectConversion.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIELOWERDMACHANNELRESET
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

#define DEBUG_TYPE "aie-lower-dma-channel-reset"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

// Expand every aiex.dma_channel_reset_for in `device` into the resident re-arm
// trio of its objectFIFO -- one aiex.dma_channel_reset + START_QUEUE re-push
// (aiex.npu.push_queue) per non-shim channel and one aiex.set_lock per
// producer/consumer lock -- resolved through the fifo's
// aie.objectfifo_rearm_binding. The head BD id + repeat of each channel are
// read straight off the binding (populated by --aie-assign-bd-ids), so this
// does not re-scan the objectFIFO-emitted aie.mem chain. The emitted
// dma_channel_reset ops are lowered to npu.maskwrite32 by the conversion below
// in the same pass; the push_queue / set_lock ops are lowered by aie-dma-to-npu
// / aie-lower-set- lock later in the pipeline, so this pass runs before them.
static LogicalResult expandDmaChannelResetForOps(DeviceOp device) {
  SymbolTable symbolTable(device);

  // One resident endpoint resolved from the binding, ready to re-arm.
  struct ResolvedEndpoint {
    Value tile;
    DMAChannelDir dir;
    int channel;
    int col;
    int row;
    uint32_t headBdId;
    int32_t
        repeatCount; // already the N-1 biased value on the resident dma_start
  };

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

    // The per-channel head BD id + repeat live on the binding, put there by
    // --aie-assign-bd-ids (or set directly on a hand-authored binding). A
    // binding still missing them cannot be re-pushed: either that pass has not
    // run, or it could not resolve an endpoint's resident DMA channel.
    std::optional<ArrayRef<int32_t>> headBdIds = binding.getHeadBdIds();
    std::optional<ArrayRef<int32_t>> repeats = binding.getRepeatCounts();
    if (!headBdIds || !repeats) {
      op.emitOpError("objectFIFO re-arm binding '")
          << op.getObjfifo()
          << "' has no head_bd_ids/repeat_counts; run --aie-assign-bd-ids "
             "before this pass, or set them on the binding";
      return WalkResult::interrupt();
    }

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
      ep.headBdId = static_cast<uint32_t>((*headBdIds)[i]);
      ep.repeatCount = (*repeats)[i];
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

    // 1. dma_channel_reset per endpoint (lowered to maskwrite32 below).
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
    // the command-word encoding, the queue address, and the bd_id/repeat range
    // checks all live in one place -- the aie-dma-to-npu lowering and
    // NpuPushQueueOp::verify. The token bit is only set for a shim S2MM
    // channel; these endpoints are core/mem, so it is always false.
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
                             builder.getBoolAttr(issueToken), repeatVal, bdVal);
    }

    toErase.push_back(op);
    return WalkResult::advance();
  });

  if (res.wasInterrupted())
    return failure();

  for (DmaChannelResetForOp op : toErase)
    op.erase();

  // Drop re-arm bindings that are now unreferenced (their reset_for users were
  // lowered). They are device-body metadata like aie.shim_dma_allocation;
  // erasing the dead ones keeps the lowered module clean.
  SmallVector<ObjectFifoRearmBindingOp> deadBindings;
  for (auto binding : device.getOps<ObjectFifoRearmBindingOp>())
    if (SymbolTable::symbolKnownUseEmpty(binding.getSymNameAttr(), device))
      deadBindings.push_back(binding);
  for (ObjectFifoRearmBindingOp binding : deadBindings)
    binding.erase();

  return success();
}

struct DmaChannelResetToMaskWrite32Pattern
    : OpConversionPattern<DmaChannelResetOp> {
  using OpConversionPattern<DmaChannelResetOp>::OpConversionPattern;

  DmaChannelResetToMaskWrite32Pattern(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(DmaChannelResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    const AIE::AIETargetModel &tm = dev.getTargetModel();

    AIE::TileOp tile = op.getTileOp();
    int col = tile.getCol();
    int row = tile.getRow();
    int channel = op.getChannel();
    AIE::DMAChannelDir dir = op.getDirection();

    // getDmaControlAddress returns the absolute address (col/row folded in);
    // NpuMaskWrite32Op re-folds col/row, so pass the local offset + col/row, as
    // AIELowerSetLock does for the lock address.
    uint32_t ctrlAddrLocal =
        tm.getDmaControlAddress(col, row, channel, dir) & 0xFFFFF;

    // Resolve the channel's reset bit from the register database rather than
    // restating the layout here. The DMA control registers live in a tile's
    // memory module (mem tiles resolve to their own module regardless), so the
    // lookup is isMem; shim tiles are rejected by the verifier.
    std::string ctrlRegName =
        (dir == AIE::DMAChannelDir::S2MM ? "DMA_S2MM_" : "DMA_MM2S_") +
        std::to_string(channel) + "_Ctrl";
    const AIE::RegisterInfo *ctrlReg =
        tm.lookupRegister(ctrlRegName, tile.getTileID(), /*isMem=*/true);
    if (!ctrlReg)
      return op.emitOpError("target has no ")
             << ctrlRegName << " register in its register database";
    const AIE::BitFieldInfo *resetField = ctrlReg->getField("Reset");
    if (!resetField)
      return op.emitOpError()
             << ctrlRegName << " has no Reset field in the register database";
    std::optional<uint32_t> resetMask = tm.getFieldMask(*resetField);
    if (!resetMask)
      return op.emitOpError()
             << ctrlRegName << " Reset field does not fit in a 32-bit register";

    Location loc = op.getLoc();
    IntegerAttr colAttr = rewriter.getI32IntegerAttr(col);
    IntegerAttr rowAttr = rewriter.getI32IntegerAttr(row);

    // Reset pulse: assert the reset bit, then clear it. Both writes mask to the
    // reset bit only, so the pulse preserves the other CTRL fields
    // (DECOMPRESSION_ENABLE, ENABLE_OUT_OF_ORDER, CONTROLLER_ID, FOT_MODE)
    // instead of clobbering them. This mirrors aie-rt's XAie_DmaChannelReset,
    // which drives the reset bit with a MaskWrite32. Constants are materialized
    // in named locals so the emitted IR order does not depend on unspecified
    // C++ argument-evaluation order.
    Value assertAddr = createConstantI32(rewriter, loc, ctrlAddrLocal);
    Value assertVal =
        createConstantI32(rewriter, loc, tm.encodeFieldValue(*resetField, 1));
    Value assertMask = createConstantI32(rewriter, loc, *resetMask);
    rewriter.create<NpuMaskWrite32Op>(loc, assertAddr, assertVal, assertMask,
                                      nullptr, colAttr, rowAttr);

    Value clearAddr = createConstantI32(rewriter, loc, ctrlAddrLocal);
    Value clearVal = createConstantI32(rewriter, loc, 0u);
    Value clearMask = createConstantI32(rewriter, loc, *resetMask);
    rewriter.replaceOpWithNewOp<NpuMaskWrite32Op>(
        op, clearAddr, clearVal, clearMask, nullptr, colAttr, rowAttr);

    return success();
  };
};

struct AIELowerDmaChannelResetPass
    : public xilinx::AIEX::impl::AIELowerDmaChannelResetBase<
          AIELowerDmaChannelResetPass> {
  void runOnOperation() override {
    DeviceOp device = getOperation();

    // First expand any aiex.dma_channel_reset_for into its re-arm trio; the
    // dma_channel_reset ops it emits are then lowered by the conversion below,
    // together with any standalone ones.
    if (failed(expandDmaChannelResetForOps(device)))
      return signalPassFailure();

    ConversionTarget target(getContext());
    target.addLegalOp<NpuMaskWrite32Op>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<DmaChannelResetOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<DmaChannelResetToMaskWrite32Pattern>(&getContext());

    if (failed(applyPartialConversion(device, target, std::move(patterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
xilinx::AIEX::createAIELowerDmaChannelResetPass() {
  return std::make_unique<AIELowerDmaChannelResetPass>();
}
