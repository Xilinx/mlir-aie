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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct DMAConfigureTaskForOpPattern : RewritePattern {

  DMAConfigureTaskForOpPattern(MLIRContext *ctx)
      : RewritePattern(DMAConfigureTaskForOp::getOperationName(),
                       PatternBenefit(1), ctx) {}

  LogicalResult matchAndRewrite(Operation *op_any,
                                PatternRewriter &rewriter) const override {
    DMAConfigureTaskForOp op = llvm::dyn_cast<DMAConfigureTaskForOp>(op_any);
    if (!op) {
      return failure();
    }
    AIE::DeviceOp device = op->getParentOfType<AIE::DeviceOp>();

    AIE::ShimDMAAllocationOp alloc_op =
        AIE::ShimDMAAllocationOp::getForSymbol(device, op.getAlloc());
    if (!alloc_op) {
      return op.emitOpError("no shim DMA allocation found for symbol");
    }

    const int col = alloc_op.getCol();
    AIE::TileOp tile = AIE::TileOp::getOrCreate(rewriter, device, col, 0);
    DMAConfigureTaskOp new_op = rewriter.create<DMAConfigureTaskOp>(
        op.getLoc(), rewriter.getIndexType(), tile.getResult(),
        alloc_op.getChannelDir(), (int32_t)alloc_op.getChannelIndex(),
        op.getIssueToken(), op.getRepeatCount());
    rewriter.replaceAllUsesWith(op.getResult(), new_op.getResult());
    rewriter.inlineRegionBefore(op.getBody(), new_op.getBody(),
                                new_op.getBody().begin());
    rewriter.eraseOp(op);
    return success();
  }
};

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
    uint64_t register_addr = getBufferDescriptorAddressRegisterAddress(
        target_model, bd_id, tile.getCol(), tile.getRow());
    if (mlir::BlockArgument buf_arg =
            llvm::dyn_cast<mlir::BlockArgument>(buf)) {
      if (!target_model.isShimNOCTile(tile.getCol(), tile.getRow())) {
        return bd_op->emitOpError("DDR memory (runtime input arguments) can "
                                  "only be referred to on shim tiles.");
      }
      unsigned arg_idx = buf_arg.getArgNumber();
      int64_t offset = bd_op.getOffsetInBytes();
      builder.create<NpuAddressPatchOp>(bd_op.getLoc(), /*addr*/ register_addr,
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
      return bd_op->emitOpError(
          "Buffer argument must be either a constant aie.buffer or a runtime "
          "sequence input argument.");
    }
    return success();
  }

  LogicalResult rewriteSingleBD(OpBuilder &builder, Block &block,
                                AIE::TileOp &tile) {
    AIE::DMABDOp bd_op = getBdForBlock(block);
    const auto &target_model = AIE::getTargetModel(bd_op);
    MemRefType buffer_type = bd_op.getBuffer().getType();
    uint32_t addr_granularity = target_model.getAddressGenGranularity();

    uint32_t bd_id = bd_op.getBdId().value();
    int64_t offset = bd_op.getOffsetInBytes();
    uint32_t len = bd_op.getLenInBytes();
    uint32_t len_addr_granularity = len * 8 / addr_granularity;

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
      getHardwareStridesWraps(target_model, buffer_type, input_sizes,
                              input_strides, sizes, strides);
      if (failed(verifyStridesWraps(bd_op, buffer_type, tile.getCol(),
                                    tile.getRow(), input_sizes, input_strides,
                                    sizes, strides))) {
        return failure();
      }
      // Ensure the total transfer length and the length expressed in the lowest
      // three dimensions of strides/wraps agree. (Fourth dimension is
      // iteration/repeat count and repeats the whole BD, so should not be
      // incorporated in length of a single BD invocation.)
      uint32_t len_dims_addr_granularity = 1;
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
    }

    // find next BD ID, if any
    uint32_t use_next_bd = 0;
    uint32_t next_bd_id = 0;
    if (bd_op.getNextBdId().has_value()) {
      next_bd_id = bd_op.getNextBdId().value();
      use_next_bd = 1;
    }

    builder.create<NpuWriteBdOp>(
        bd_op.getLoc(), tile.getCol(), bd_id, len_addr_granularity, offset, 0,
        0, 0, 0,
        /* TODO: Strides/Wraps */
        /*d0_size=*/sizes[0], /*d0_stride=*/strides[0],
        /*d1_size=*/sizes[1], /*d1_stride=*/strides[1],
        /*d2_stride=*/strides[2],
        /*iteration_current=*/0, /*iteration_size=*/sizes[3],
        /*iteration_stride=*/strides[3],
        /* TODO: Next BD */
        /*next_bd=*/next_bd_id,
        /*row=*/tile.getRow(),
        /*use_next_bd=*/use_next_bd,
        /*valid_bd=*/1,
        /* TODO: Locks */
        /*lock_rel_val=*/0, /*lock_rel_id=*/0, /*lock_acq_enable=*/0,
        /*lock_acq_val=*/0, /*lock_ackq_id=*/0);

    return setAddressForSingleBD(builder, bd_op, tile);
  }

  LogicalResult hoistNextBdOpsIntoAttrs(DMAConfigureTaskOp op) {
    Region &body = op.getBody();
    for (auto it = body.begin(); it != body.end(); ++it) {
      Block &block = *it;
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
      if (failed(verifyNoUnsupportedOpsInBlock(*it))) {
        return failure();
      }
      if (failed(verifyBdInBlock(*it))) {
        return failure();
      } else {
      }
      if (failed(verifyOptionalLocksInBlock(*it))) {
        return failure();
      }
    }

    // Hoist next_bd operations into next_bd_id attribute of the dma_bd
    if (failed(hoistNextBdOpsIntoAttrs(op))) {
      return failure();
    }

    // Lower all BDs
    for (auto it = body.begin(); it != body.end(); ++it) {
      Block &block = *it;
      if (failed(rewriteSingleBD(builder, block, tile))) {
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

    // Convert DMAConfigureTaskForOps that reference shim DMA allocations
    // to regular DMAConfigureTaskOps
    ConversionTarget target_0(getContext());
    target_0.addLegalDialect<AIEXDialect>();
    target_0.addIllegalOp<DMAConfigureTaskForOp>();
    RewritePatternSet patterns_0(&getContext());
    patterns_0.insert<DMAConfigureTaskForOpPattern>(&getContext());

    GreedyRewriteConfig rewriter_config = GreedyRewriteConfig();
    if (failed(applyPatternsAndFoldGreedily(device, std::move(patterns_0),
                                            rewriter_config))) {
      signalPassFailure();
    }

    // Convert DMAStartBD and DMAAwaitBD ops
    ConversionTarget target_1(getContext());
    target_1.addLegalDialect<AIEXDialect>();
    target_1.addIllegalOp<DMAStartTaskOp>();
    target_1.addIllegalOp<DMAAwaitTaskOp>();
    RewritePatternSet patterns_1(&getContext());
    patterns_1.insert<DMAStartTaskOpPattern>(&getContext());
    patterns_1.insert<DMAAwaitTaskOpPattern>(&getContext());
    if (failed(
            applyPartialConversion(device, target_1, std::move(patterns_1)))) {
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
