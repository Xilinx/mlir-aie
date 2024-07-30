//===- AIEBDChainsToNPU.cpp ---------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <iterator>
#include <algorithm>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct DMAStartBDsPattern : OpConversionPattern<DMAStartBDs> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(DMAStartBDs op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    DMAConfigureBDs bds_op = op.getBdsOp();
    AIE::TileOp tile = bds_op.getTileOp();
    std::optional<uint32_t> first_bd_id = bds_op.getFirstBdId();
    if(!first_bd_id) {
        auto err = op.emitOpError("First buffer descriptor in chain has not been assigned an ID");
        err.attachNote() << "Run the `aie-assign-runtime-buffer-descriptor-ids` pass first or manually assign an ID.";
        return failure();
    }
    rewriter.replaceOpWithNewOp<NpuPushQueueOp>(
        op, tile.getCol(), tile.getRow(), bds_op.getDirection(), bds_op.getChannel(), bds_op.getIssueToken(), bds_op.getRepeatCount(), *first_bd_id);
    return success();
  }
};

struct DMAAwaitBDsPattern : OpConversionPattern<DMAAwaitBDs> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(DMAAwaitBDs op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    DMAConfigureBDs bds_op = op.getBdsOp();
    if(!bds_op.getIssueToken()) {
      auto err = op.emitOpError("Cannot wait on a BD that is not configured to issue a token.");
      err.attachNote(bds_op.getLoc()) << "Consider adding attribute `issue_token=true` here.";
      return failure();
    }
    AIE::TileOp tile = bds_op.getTileOp();
    rewriter.replaceOpWithNewOp<NpuSyncOp>(
        op, tile.getCol(), tile.getRow(), (uint32_t)bds_op.getDirection(), bds_op.getChannel(), 1, 1);
    return success();
  }
};

struct AIEBDChainsToNPUPass
    : AIEBDChainsToNPUBase<AIEBDChainsToNPUPass> {
  

  LogicalResult verifyBdInBlock(Block &block) {
    auto bd_ops = block.getOps<AIE::DMABDOp>();
    // Exactly one BD op per block
    int n_bd_ops = std::distance(bd_ops.begin(), bd_ops.end());
    if(n_bd_ops < 1) {
      auto error = block.getTerminator()->emitError("Block ending in this terminator does not contain a required aie.dma_bd operation.");
      error.attachNote(block.getParentOp()->getLoc()) << "Error encountered while lowering this BD configuration.";
      return failure();
    } else if(n_bd_ops > 1) {
      auto error = block.getTerminator()->emitOpError("This block contains multiple aie.dma_bd operations. Exactly one is required.");
     // auto it = bd_ops.begin();
     // ++it;
     // for(; it != it.end(); ++it) {
     //   error.attachNote(it->getLoc()) << "Extra aie.dma_bd operation here.";
     // }
      return failure();
    }
    AIE::DMABDOp bd_op = *bd_ops.begin();
    if(!bd_op.getBdId().has_value()) {
      auto error = bd_op.emitOpError("Cannot lower buffer descriptor without assigned ID.");
      error.attachNote() << "Run the `aie-assign-runtime-buffer-descriptor-ids` pass first or manually assign an ID to this buffer descriptor.";
      error.attachNote(block.getParentOp()->getLoc()) << "Error encountered while lowering this BD configuration.";
      return failure();
    }
    return success();
  }

  LogicalResult verifyOptionalLocksInBlock(Block &block) {
    auto lock_ops = block.getOps<AIE::UseLockOp>();
    // Exactly 0 or 2 lock ops
    int n_lock_ops = std::distance(lock_ops.begin(), lock_ops.end());
    if(n_lock_ops > 0) {
      // TODO: Not yet implemented
      AIE::UseLockOp lock_op = *lock_ops.begin();
      lock_op.emitOpError("Lowering for lock operations in NPU runtime configuration is not yet implemented.");
      return failure();
    }
    return success();
  }

  LogicalResult verifyNoUnsupportedOpsInBlock(Block &block) {
    WalkResult unsupported_ops = block.walk([&](Operation *inner_op) {
      return llvm::TypeSwitch<Operation *, WalkResult>(inner_op)
        .Case<AIE::DMABDOp>([&](AIE::DMABDOp bd_op) { return WalkResult::advance(); })
        .Case<AIE::UseLockOp>([&](AIE::UseLockOp lock_op) { return WalkResult::advance(); })
        .Case<AIE::NextBDOp>([&](AIE::NextBDOp lock_op) { return WalkResult::advance(); })
        .Case<AIE::EndOp>([&](AIE::EndOp lock_op) { return WalkResult::advance(); })
        .Default([&](Operation *inner_op){
          auto error = block.getParentOp()->emitOpError("Unsupported operation within BD block.");
          error.attachNote(inner_op->getLoc()) << "No lowering to NPU instructions available for this operation.";
          return WalkResult::interrupt();
        });
    });
    if(unsupported_ops.wasInterrupted()) {
      return failure();
    }
    return success();
  }

  AIE::DMABDOp getBdForBlock(Block &block) {
    auto bd_ops = block.getOps<AIE::DMABDOp>();
    AIE::DMABDOp bd_op = *bd_ops.begin(); // Dereference first (and only, after previous checks) bd op iterator
    return bd_op;
  }

  std::optional<std::pair<AIE::UseLockOp, AIE::UseLockOp>> getOptionalLockOpsForBlock(Block &block) {
    //auto lock_ops = block.getOps<AIE::UseLockOp>();
    return std::nullopt; // Not yet implemented
  }

  LogicalResult setAddressForSingleBD(OpBuilder &builder, AIE::DMABDOp &bd_op, AIE::TileOp &tile) {
    uint32_t bd_id = bd_op.getBdId().value();
    auto buf = bd_op.getBuffer();
    const AIE::AIETargetModel &target_model = AIE::getTargetModel(bd_op);
    AIEX::RuntimeSequenceOp parentOp = buf.getDefiningOp<AIEX::RuntimeSequenceOp>();
    if(!parentOp) {
      // TODO: Not yet implemented (for AIE.buffer references)
      // Implementing this would involve taking the allocated buffer address and plugging it as a register write to `addr` below.
      bd_op->emitOpError("At present, buffer arguments must be an input to the runtime sequence. Constant buffer arguments not yet implemented.");
      return failure();
    }
    Region &parentRegion = parentOp.getBody();
    auto arg_idx_it = std::find(parentRegion.args_begin(), parentRegion.args_end(), buf);
    if(arg_idx_it == parentRegion.args_end()) {
      auto err = bd_op->emitOpError("Unable to determine argument index for input buffer.");
      err.attachNote(parentRegion.getLoc()) << "In this region.";
      return failure();
    }
    int arg_idx = std::distance(parentRegion.args_begin(), arg_idx_it);
    uint64_t addr = AIEXDialect::getBufferDescriptorAddressRegisterAddress(target_model, bd_id, tile.getCol());
    int64_t offset = bd_op.getOffsetInBytes();
    builder.create<NpuAddressPatchOp>(bd_op.getLoc(), /*addr*/addr, /*arg_idx*/arg_idx, /*arg_plus*/offset);
    return success();
  }

  LogicalResult rewriteSingleBD(OpBuilder &builder, Block &block, AIE::TileOp &tile) {
    AIE::DMABDOp bd_op = getBdForBlock(block);
    const auto &target_model = AIE::getTargetModel(bd_op);
    MemRefType buffer_type = bd_op.getBuffer().getType();

    uint32_t bd_id = bd_op.getBdId().value();
    int64_t offset = bd_op.getOffsetInBytes();
    uint32_t len = bd_op.getLenInBytes();

    // Process strides/wraps
    std::optional<llvm::ArrayRef<AIE::BDDimLayoutAttr>> dims = bd_op.getDimensions();
    llvm::SmallVector<int64_t, 4> input_sizes = llvm::SmallVector<int64_t, 4>(4, 0);
    llvm::SmallVector<int64_t, 4> input_strides = llvm::SmallVector<int64_t, 4>(4, 0);
    if(dims) {
      for (size_t i = 0; i < dims->size(); i++) {
        // Pass down dimensions in reverse order; in the MLIR, this allows
        // us to specify step sizes/wraps in the same order as we would
        // access a multi-dim C array, with the highest dimension first.
        int j = dims->size() - i - 1;
        input_sizes[j] = dims.value()[i].getSize();
        input_strides[j] = dims.value()[i].getStride();
      }
    }
    auto [sizes, strides] = AIEXDialect::getHardwareStridesWraps(target_model, buffer_type, input_sizes, input_strides);
    if(failed(AIEXDialect::verifyStridesWraps((Operation *)&bd_op, buffer_type, tile.getCol(), tile.getRow(), input_sizes, input_strides, sizes, strides))) {
      return failure();
    }

    // find next BD ID, if any
    uint32_t use_next_bd = 0;
    uint32_t next_bd_id = 0;
    auto next_bd_ops = block.getOps<AIE::NextBDOp>();
    if(!next_bd_ops.empty()) {
      AIE::NextBDOp next_bd_op = *next_bd_ops.begin();
      Block *next_bd_block = next_bd_op.getDest(); 
      auto next_bd_bd_ops = next_bd_block->getOps<AIE::DMABDOp>();
      if(next_bd_bd_ops.empty()) {
        auto err = bd_op->emitOpError("Referenced next BD block contains no BD operation.");
        return failure();
      }
      AIE::DMABDOp next_bd_bd_op = *next_bd_bd_ops.begin();
      next_bd_id = next_bd_bd_op.getBdId().value();
      use_next_bd = 1;
    }

    builder.create<NpuWriteBdOp>(bd_op.getLoc(),
        tile.getCol(), bd_id, len, offset, 0, 0, 0, 0,
        /* TODO: Strides/Wraps */
        /*d0_size=*/sizes[0], /*d0_stride=*/strides[0], 
        /*d1_size=*/sizes[1], /*d1_stride=*/strides[1], 
        /*d2_stride=*/strides[2],
        /*iteration_current=*/0, /*iteration_size=*/sizes[3], /*iteration_stride=*/strides[3],
        /* TODO: Next BD */
        /*next_bd=*/next_bd_id, 
        /*row=*/tile.getRow(), 
        /*use_next_bd=*/use_next_bd,
        /*valid_bd=*/1,
        /* TODO: Locks */
        /*lock_rel_val=*/0, /*lock_rel_id=*/0, /*lock_acq_enable=*/0, /*lock_acq_val=*/0, /*lock_ackq_id=*/0);

    return setAddressForSingleBD(builder, bd_op, tile);
  }

  LogicalResult hoistNextBdOpsIntoAttrs(DMAConfigureBDs op) {
    Region &body = op.getBody();
    for(auto it = body.begin(); it != body.end(); ++it) {
      Block &block = *it; 
      AIE::DMABDOp bd_op = getBdForBlock(block);
      Operation *terminator = block.getTerminator();
      if(llvm::isa<AIE::NextBDOp>(terminator)) {
        if(bd_op.getNextBdId().has_value()) {
          auto error = bd_op.emitOpError("Cannot specify both next_bd_id attribute and aie.next_bd operation."); 
          error.attachNote(terminator->getLoc()) << "Potentially conflicting next buffer descriptor ID specified here.";
          return failure();
        }
        AIE::NextBDOp next_bd_op = llvm::cast<AIE::NextBDOp>(*terminator);
        Block &next_bd_block = *next_bd_op.getDest();
        AIE::DMABDOp next_dma_bd_op = getBdForBlock(next_bd_block);
        assert(next_dma_bd_op.getBdId().has_value()); // Next BD should have assigned ID, and this should have been checked by earlier verifyBdInBlock() call
        bd_op.setNextBdId(next_dma_bd_op.getBdId().value());
        OpBuilder builder(next_bd_op);
        builder.create<AIE::EndOp>(next_bd_op.getLoc());
        next_bd_op.erase();
      } else if(!llvm::isa<AIE::EndOp>(terminator)) {
        terminator->emitOpError("Invalid terminator for a buffer descriptor block. Must be either aie.end or aie.next_bd.");
        return failure();
      }
    }
    return success();
  }
  
  LogicalResult rewriteSingleDMAConfigureBDs(DMAConfigureBDs op) {
    OpBuilder builder(op);
    AIE::TileOp tile = op.getTileOp();

    if(!op.use_empty()) {
      auto err = op.emitOpError("Cannot lower while op still has uses.");
      mlir::Operation::use_range uses = op.getOperation()->getUses();
      for(auto it = uses.begin(); it != uses.end(); ++it) {
        err.attachNote(it->getOwner()->getLoc()) << "Used here.";
      }
      return failure();
    }

    Region &body = op.getBody();

    // Verify each BD block first; subsequent functions rely on them being well-formed
    for(auto it = body.begin(); it != body.end(); ++it) {
      if(failed(verifyNoUnsupportedOpsInBlock(*it))) {
        return failure();
      }
      if(failed(verifyBdInBlock(*it))) {
        return failure();
      } else {
        
      }
      if(failed(verifyOptionalLocksInBlock(*it))) {
        return failure();
      }
    }

    // Hoist next_bd operations into next_bd_id attribute of the dma_bd
    if(failed(hoistNextBdOpsIntoAttrs(op))) {
      return failure();
    }

    // Lower all BDs
    for(auto it = body.begin(); it != body.end(); ++it) {
      Block &block = *it;
      if(failed(rewriteSingleBD(builder, block, tile))) {
        return failure();
      }
    }

    op.erase();

    return success();
  }

  LogicalResult rewriteDMAConfigureBDs(AIE::DeviceOp device) {
    WalkResult result = device.walk([&](DMAConfigureBDs op) {
      if(failed(rewriteSingleDMAConfigureBDs(op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if(result.wasInterrupted()) {
      return failure();
    }
    return success();
  }

  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();

    // Convert DMAStartBD and DMAAwaitBD ops
    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addIllegalOp<DMAStartBDs>();
    target.addIllegalOp<DMAAwaitBDs>();
    RewritePatternSet patterns(&getContext());
    patterns.insert<DMAStartBDsPattern>(&getContext());
    patterns.insert<DMAAwaitBDsPattern>(&getContext());
    if (failed(applyPartialConversion(device, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // Lower the configuration for the BDs
    if (failed(rewriteDMAConfigureBDs(device))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEBDChainsToNPUPass() {
  return std::make_unique<AIEBDChainsToNPUPass>();
}
