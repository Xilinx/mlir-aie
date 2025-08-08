//===- AIEDMATasksToNPU.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct DMAStartTaskOpPattern : OpConversionPattern<DMAStartTaskOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DMAStartTaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DMAConfigureTaskOp task_op = op.getTaskOp();
    if (!task_op) {
      // Cannot rewrite this; probably points to a DMAStartTaskForOp,
      // which we will lower once it has been rewritten into a DMAStartTaskOp.
      return failure();
    }
    AIE::TileOp tile = task_op.getTileOp();
    std::optional<uint32_t> first_bd_id = task_op.getFirstBdId();
    if (!first_bd_id) {
      auto err = op.emitOpError(
          "First buffer descriptor in chain has not been assigned an ID");
      err.attachNote() << "Run the `aie-assign-runtime-buffer-descriptor-ids` "
                          "pass first or manually assign an ID.";
      return failure();
    }
    rewriter.replaceOpWithNewOp<NpuPushQueueOp>(
        op, tile.getCol(), tile.getRow(), task_op.getDirection(),
        task_op.getChannel(), task_op.getIssueToken(), task_op.getRepeatCount(),
        *first_bd_id);
    return success();
  }
};

struct DMAAwaitTaskOpPattern : OpConversionPattern<DMAAwaitTaskOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DMAAwaitTaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DMAConfigureTaskOp task_op = op.getTaskOp();
    if (!task_op) {
      return failure();
    }
    if (!task_op.getIssueToken()) {
      auto err = op.emitOpError(
          "Cannot wait on a BD that is not configured to issue a token.");
      err.attachNote(task_op.getLoc())
          << "Consider adding attribute `issue_token=true` here.";
      return err;
    }
    AIE::TileOp tile = task_op.getTileOp();
    rewriter.replaceOpWithNewOp<NpuSyncOp>(op, tile.getCol(), tile.getRow(),
                                           (uint32_t)task_op.getDirection(),
                                           task_op.getChannel(), 1, 1);
    return success();
  }
};

struct AIEDMATasksToNPUPass : AIEDMATasksToNPUBase<AIEDMATasksToNPUPass> {

  bool shouldSkipBlock(Block &block) {
    // Allow blocks in the input IR that contain nothing but a next_bd operation
    // as the entry block. We will skip these blocks and not lower them to
    // anything.
    auto it = block.without_terminator();
    return block.isEntryBlock() && it.begin() == it.end();
  }

  LogicalResult verifyBdInBlock(Block &block) {
    auto bd_ops = block.getOps<AIE::DMABDOp>();
    // Exactly one BD op per block
    int n_bd_ops = std::distance(bd_ops.begin(), bd_ops.end());
    if (n_bd_ops < 1) {
      auto error = block.getTerminator()->emitError(
          "Block ending in this terminator does not contain a required "
          "aie.dma_bd operation.");
      error.attachNote(block.getParentOp()->getLoc())
          << "Error encountered while lowering this BD configuration.";
      return failure();
    } else if (n_bd_ops > 1) {
      auto error = block.getTerminator()->emitOpError(
          "This block contains multiple aie.dma_bd operations. Exactly one is "
          "required.");
      auto it = bd_ops.begin();
      ++it;
      for (; it != bd_ops.end(); ++it) {
        error.attachNote((*it)->getLoc()) << "Extra aie.dma_bd operation here.";
      }
      return failure();
    }
    AIE::DMABDOp bd_op = *bd_ops.begin();
    if (!bd_op.getBdId().has_value()) {
      auto error = bd_op.emitOpError(
          "Cannot lower buffer descriptor without assigned ID.");
      error.attachNote()
          << "Run the `--aie-assign-runtime-sequence-bd-ids` pass first or "
             "manually assign an ID to this buffer descriptor.";
      error.attachNote(block.getParentOp()->getLoc())
          << "Error encountered while lowering this BD configuration.";
      return failure();
    }
    return success();
  }

  LogicalResult verifyOptionalLocksInBlock(Block &block) {
    auto lock_ops = block.getOps<AIE::UseLockOp>();
    // Exactly 0 or 2 lock ops
    int n_lock_ops = std::distance(lock_ops.begin(), lock_ops.end());
    if (n_lock_ops > 0) {
      // TODO: Not yet implemented
      AIE::UseLockOp lock_op = *lock_ops.begin();
      lock_op.emitOpError("Lowering for lock operations in NPU runtime "
                          "configuration is not yet implemented.");
      return failure();
    }
    return success();
  }

  LogicalResult verifyNoUnsupportedOpsInBlock(Block &block) {
    WalkResult unsupported_ops = block.walk([&](Operation *inner_op) {
      return llvm::TypeSwitch<Operation *, WalkResult>(inner_op)
          .Case<AIE::DMABDOp>(
              [&](AIE::DMABDOp bd_op) { return WalkResult::advance(); })
          .Case<AIE::UseLockOp>(
              [&](AIE::UseLockOp lock_op) { return WalkResult::advance(); })
          .Case<AIE::NextBDOp>(
              [&](AIE::NextBDOp lock_op) { return WalkResult::advance(); })
          .Case<AIE::EndOp>(
              [&](AIE::EndOp lock_op) { return WalkResult::advance(); })
          .Default([&](Operation *inner_op) {
            auto error = block.getParentOp()->emitOpError(
                "Unsupported operation within BD block.");
            error.attachNote(inner_op->getLoc())
                << "No lowering to NPU instructions available for this "
                   "operation.";
            return WalkResult::interrupt();
          });
    });
    if (unsupported_ops.wasInterrupted()) {
      return failure();
    }
    return success();
  }

  AIE::DMABDOp getBdForBlock(Block &block) {
    auto bd_ops = block.getOps<AIE::DMABDOp>();
    AIE::DMABDOp bd_op = *bd_ops.begin(); // Dereference first (and only, after
                                          // previous checks) bd op iterator
    return bd_op;
  }

  std::optional<std::pair<AIE::UseLockOp, AIE::UseLockOp>>
  getOptionalLockOpsForBlock(Block &block) {
    // auto lock_ops = block.getOps<AIE::UseLockOp>();
    return std::nullopt; // Not yet implemented
  }

  LogicalResult setAddressForSingleBD(OpBuilder &builder, AIE::DMABDOp &bd_op,
                                      AIE::TileOp &tile) {
    uint32_t bd_id = bd_op.getBdId().value();
    const AIE::AIETargetModel &target_model = AIE::getTargetModel(bd_op);
    auto buf = bd_op.getBuffer();
    uint64_t register_addr =
        target_model.getDmaBdAddress(tile.getCol(), tile.getRow(), bd_id) +
        target_model.getDmaBdAddressOffset(tile.getCol(), tile.getRow());
    if (mlir::BlockArgument buf_arg =
            llvm::dyn_cast<mlir::BlockArgument>(buf)) {
      if (!target_model.isShimNOCTile(tile.getCol(), tile.getRow())) {
        return bd_op->emitOpError("DDR memory (runtime input arguments) can "
                                  "only be referred to on shim tiles.");
      }
      unsigned arg_idx = buf_arg.getArgNumber();
      int64_t offset = bd_op.getOffsetInBytes();
      builder.create<NpuAddressPatchOp>(bd_op.getLoc(),
                                        /*addr*/ register_addr,
                                        /*arg_idx*/ arg_idx,
                                        /*arg_plus*/ offset);
    } else if (AIE::BufferOp buffer =
                   llvm::dyn_cast<AIE::BufferOp>(buf.getDefiningOp())) {
      uint64_t buf_addr;
      if (!buffer.getAddress().has_value()) {
        return bd_op->emitOpError(
            "Cannot lower buffer without associated address. Run pass "
            "--aie-assign-buffer-addresses first or manually assign an "
            "address.");
      }
      buf_addr = *buffer.getAddress();
      builder.create<NpuWrite32Op>(bd_op.getLoc(), register_addr, buf_addr,
                                   nullptr, nullptr, nullptr);
    } else {
      return bd_op->emitOpError("Buffer argument must be either a constant "
                                "aie.buffer or a runtime "
                                "sequence input argument.");
    }
    return success();
  }

  LogicalResult
  rewriteSingleBD(OpBuilder &builder, Block &block, AIE::TileOp &tile,
                  AIE::DMAChannelDir channelDir,
                  std::optional<xilinx::AIE::PacketInfoAttr> packet) {
    AIE::DMABDOp bd_op = getBdForBlock(block);
    const auto &target_model = AIE::getTargetModel(bd_op);
    MemRefType buffer_type = bd_op.getBuffer().getType();
    uint32_t addr_granularity = target_model.getAddressGenGranularity();

    uint32_t bd_id = bd_op.getBdId().value();
    int64_t offset = bd_op.getOffsetInBytes();
    uint64_t len = bd_op.getLenInBytes();
    uint64_t len_addr_granularity = len * 8 / addr_granularity;

    if (offset * 8 % addr_granularity != 0) {
      return bd_op->emitOpError("Offset must be aligned to ")
             << (addr_granularity / 8) << " byte boundary.";
    }

    if (len < addr_granularity / 8) {
      return bd_op->emitOpError("Transfer size of ")
             << len << " bytes falls below minimum hardware transfer unit of "
             << (addr_granularity / 8) << " bytes.";
    }
    // Process strides/wraps
    std::optional<llvm::ArrayRef<AIE::BDDimLayoutAttr>> dims =
        bd_op.getDimensions();
    llvm::SmallVector<int64_t, 4> sizes = llvm::SmallVector<int64_t, 4>(4, 0);
    llvm::SmallVector<int64_t, 4> strides = llvm::SmallVector<int64_t, 4>(4, 0);

    // Padding
    std::optional<llvm::ArrayRef<AIE::BDPadLayoutAttr>> padDims =
        bd_op.getPadDimensions();
    llvm::SmallVector<int64_t, 4> padBefore =
        llvm::SmallVector<int64_t, 4>(4, 0);
    llvm::SmallVector<int64_t, 4> padAfter =
        llvm::SmallVector<int64_t, 4>(4, 0);
    std::fill(padBefore.begin(), padBefore.end(), 0);
    std::fill(padAfter.begin(), padAfter.end(), 0);

    auto enable_packet = 0;
    auto out_of_order_id = 0;
    auto packet_id = 0;
    auto packet_type = 0;
    auto d0size = 0;
    auto d0stride = 0;
    auto d1size = 0;
    auto d1stride = 0;
    auto d2size = 0;
    auto d2stride = 0;
    auto iteration_size = 0;
    auto iteration_stride = 0;

    if (dims && dims->size() > 0) {
      llvm::SmallVector<int64_t, 4> input_sizes =
          llvm::SmallVector<int64_t, 4>(4, 1);
      llvm::SmallVector<int64_t, 4> input_strides =
          llvm::SmallVector<int64_t, 4>(4, 0);
      if (dims->size() > 4) {
        return bd_op->emitOpError("At most four data layout transformation "
                                  "dimensions may be provided.");
      }

      for (size_t i = 0; i < dims->size(); i++) {
        // Pass down dimensions in reverse order; in the MLIR, this allows
        // us to specify step sizes/wraps in the same order as we would
        // access a multi-dim C array, with the highest dimension first.
        int j = dims->size() - i - 1;
        input_sizes[i] = (*dims)[j].getSize();
        input_strides[i] = (*dims)[j].getStride();
      }

      // Do not check input_sizes[3] because a repeat can still be considered a
      // linear transfer
      bool isLinearTransfer = (input_sizes[0] >= 1) && (input_sizes[1] == 1) &&
                              (input_sizes[2] == 1);

      if (dims->size() > 2) {
        d2size = (target_model.isMemTile(tile.getCol(), tile.getRow()))
                     ? (*dims)[2].getSize()
                     : 0;
      }
      if (padDims.has_value()) {
        if (!target_model.isMemTile(tile.getCol(), tile.getRow()))
          return bd_op->emitOpError()
                 << "Padding is only supported by memtile dma bds.";
        if (padDims->size() > dims->size())
          return bd_op->emitOpError()
                 << "Mismatch number of dimensions between padding(s)"
                 << " and wrap(s) and stride(s).";
        if (channelDir == AIE::DMAChannelDir::MM2S) {
          for (size_t i = 0; i < padDims->size(); i++) {
            int j = padDims->size() - i - 1;
            padBefore[i] = (*padDims)[j].getConstPadBefore();
            padAfter[i] = (*padDims)[j].getConstPadAfter();
          }
          for (size_t i = padDims->size(); i < dims->size(); i++) {
            padBefore[i] = 0;
            padAfter[i] = 0;
          }
        } else
          return bd_op->emitOpError()
                 << "supports padding only for MM2S direction on MemTiles.";
      }
      getHardwareStridesWraps(target_model, bd_op, buffer_type, input_sizes,
                              input_strides, sizes, strides);

      if (failed(verifyStridesWraps(bd_op, buffer_type, tile.getCol(),
                                    tile.getRow(), input_sizes, input_strides,
                                    sizes, strides, isLinearTransfer))) {
        return failure();
      }

      iteration_size = sizes[3];
      iteration_stride = strides[3];

      if (!isLinearTransfer) {
        // d0_size, d0_stride
        d0size = sizes[0];
        d0stride = strides[0];

        // d1_size, d1_stride
        d1size = sizes[1];
        d1stride = strides[1];

        // d2_stride
        d2stride = strides[2];
        // d2_size set elsewhere
      }
      if (input_sizes[3] > 1 && input_strides[3] == 0) {
        // We allow users to encode the repeat_count as a dimension 3 stride
        // of 0. This must lower to a iteration wrap of 0, so no stride is
        // ever added. We then repeat the BD using the repeat_count in
        // NpuPushQueueOp.
        iteration_size = 0;
        iteration_stride = 0;
      }

      // Ensure the total transfer length and the length expressed in the lowest
      // three dimensions of strides/wraps agree. (Fourth dimension is
      // iteration/repeat count and repeats the whole BD, so should not be
      // incorporated in length of a single BD invocation.)
      uint64_t len_dims_addr_granularity = 1;
      for (size_t i = 0; i < 3; i++) {
        len_dims_addr_granularity *= sizes[i];
      }
      if (len_dims_addr_granularity != len_addr_granularity) {
        auto err =
            bd_op->emitOpError(
                "Buffer descriptor length does not match length of transfer "
                "expressed by lowest three dimensions of data layout "
                "transformation strides/wraps. ")
            << "BD length is " << (len_addr_granularity * addr_granularity / 8)
            << " bytes. "
            << "Lowest three dimensions of data layout transformation would "
               "result in transfer of "
            << (len_dims_addr_granularity * addr_granularity / 8) << " bytes. ";
        err.attachNote() << "Do not include the highest dimension size in "
                            "transfer length, as this is the BD repeat count.";
        return failure();
      }
    } else {
      if (padDims && target_model.isMemTile(tile.getCol(), tile.getRow()) &&
          channelDir == AIE::DMAChannelDir::MM2S) {
        return bd_op->emitOpError()
               << "Padding requires n-d data layouts expressed as "
               << "wrap(s) and stride(s).";
      } else if (padDims) {
        return bd_op->emitOpError() << "Padding is supported only on MemTiles.";
      }
    }
    // find next BD ID, if any
    uint32_t use_next_bd = 0;
    uint32_t next_bd_id = 0;
    if (bd_op.getNextBdId().has_value()) {
      next_bd_id = bd_op.getNextBdId().value();
      use_next_bd = 1;
    }

    // enable_packet
    // auto info = bd_op.getPacket() ? bd_op.getPacket() : packet;
    auto info = bd_op.getPacket().value_or(packet.value_or(nullptr));
    if (info) {
      enable_packet = 1;
      packet_type = info.getPktType();
      packet_id = info.getPktId();
    }

    builder.create<NpuWriteBdOp>(
        bd_op.getLoc(), tile.getCol(), bd_id, len_addr_granularity, offset,
        /*enable_packet=*/enable_packet,
        /*out_of_order_id=*/out_of_order_id,
        /*packet_id=*/packet_id,
        /*packet_type=*/packet_type,
        /* TODO: Strides/Wraps */
        /*d0_size=*/d0size, /*d0_stride=*/d0stride,
        /*d1_size=*/d1size, /*d1_stride=*/d1stride,
        /*d2_size=*/d2size, /*d2_stride=*/d2stride,
        /*iteration_current=*/0, /*iteration_size=*/iteration_size,
        /*iteration_stride=*/iteration_stride,
        /* TODO: Next BD */
        /*next_bd=*/next_bd_id,
        /*row=*/tile.getRow(),
        /*use_next_bd=*/use_next_bd,
        /*valid_bd=*/1,
        /* TODO: Locks */
        /*lock_rel_val=*/0, /*lock_rel_id=*/0, /*lock_acq_enable=*/0,
        /*lock_acq_val=*/0, /*lock_ackq_id=*/0, /*d0_zero_before=*/padBefore[0],
        /*d1_zero_before=*/padBefore[1], /*d2_zero_before=*/padBefore[2],
        /*d0_zero_after=*/padAfter[0], /*d1_zero_after=*/padAfter[1],
        /*d2_zero_after=*/padAfter[2],
        /*burst_length=*/bd_op.getBurstLength());
    return setAddressForSingleBD(builder, bd_op, tile);
  }

  LogicalResult hoistNextBdOpsIntoAttrs(DMAConfigureTaskOp op) {
    Region &body = op.getBody();
    for (auto it = body.begin(); it != body.end(); ++it) {
      Block &block = *it;
      if (shouldSkipBlock(block)) {
        continue;
      }
      AIE::DMABDOp bd_op = getBdForBlock(block);
      if (AIE::NextBDOp next_bd_op =
              llvm::dyn_cast<AIE::NextBDOp>(block.getTerminator())) {
        if (bd_op.getNextBdId().has_value()) {
          auto error =
              bd_op.emitOpError("Cannot specify both next_bd_id attribute and "
                                "aie.next_bd operation.");
          error.attachNote(next_bd_op.getLoc())
              << "Potentially conflicting next buffer descriptor ID specified "
                 "here.";
          return failure();
        }
        Block &next_bd_block = *next_bd_op.getDest();
        AIE::DMABDOp next_dma_bd_op = getBdForBlock(next_bd_block);
        assert(next_dma_bd_op.getBdId()
                   .has_value()); // Next BD should have assigned ID, and this
                                  // should have been checked by earlier
                                  // verifyBdInBlock() call
        bd_op.setNextBdId(next_dma_bd_op.getBdId().value());
        OpBuilder builder(next_bd_op);
        builder.create<AIE::EndOp>(next_bd_op.getLoc());
        next_bd_op.erase();
      }
    }
    return success();
  }

  LogicalResult rewriteSingleDMAConfigureTaskOp(DMAConfigureTaskOp op) {
    OpBuilder builder(op);
    AIE::TileOp tile = op.getTileOp();

    if (!op.use_empty()) {
      auto err = op.emitOpError("Cannot lower while op still has uses.");
      mlir::Operation::use_range uses = op.getOperation()->getUses();
      for (auto it = uses.begin(); it != uses.end(); ++it) {
        err.attachNote(it->getOwner()->getLoc()) << "Used here.";
      }
      return failure();
    }

    Region &body = op.getBody();

    // Verify each BD block first; subsequent functions rely on them being
    // well-formed
    for (auto it = body.begin(); it != body.end(); ++it) {
      if (shouldSkipBlock(*it)) {
        continue;
      }
      if (failed(verifyNoUnsupportedOpsInBlock(*it))) {
        return failure();
      }
      if (failed(verifyBdInBlock(*it))) {
        return failure();
      }
      if (failed(verifyOptionalLocksInBlock(*it))) {
        return failure();
      }
    }

    // Hoist next_bd operations into next_bd_id attribute of the dma_bd
    if (failed(hoistNextBdOpsIntoAttrs(op))) {
      return failure();
    }

    auto channelDir = op.getDirection();
    auto packet = op.getPacket();

    // Lower all BDs
    for (auto it = body.begin(); it != body.end(); ++it) {
      Block &block = *it;
      if (shouldSkipBlock(block)) {
        continue;
      }
      if (failed(rewriteSingleBD(builder, block, tile, channelDir, packet))) {
        return failure();
      }
    }

    op.erase();

    return success();
  }

  LogicalResult rewriteDMAConfigureTaskOp(AIE::DeviceOp device) {
    WalkResult result = device.walk([&](DMAConfigureTaskOp op) {
      if (failed(rewriteSingleDMAConfigureTaskOp(op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return failure();
    }
    return success();
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    // Convert DMAStartBD and DMAAwaitBD ops
    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addIllegalOp<DMAStartTaskOp>();
    target.addIllegalOp<DMAAwaitTaskOp>();
    RewritePatternSet patterns(&getContext());
    patterns.insert<DMAStartTaskOpPattern>(&getContext());
    patterns.insert<DMAAwaitTaskOpPattern>(&getContext());
    if (failed(applyPartialConversion(device, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // Lower the configuration for the BDs
    if (failed(rewriteDMAConfigureTaskOp(device))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEDMATasksToNPUPass() {
  return std::make_unique<AIEDMATasksToNPUPass>();
}
