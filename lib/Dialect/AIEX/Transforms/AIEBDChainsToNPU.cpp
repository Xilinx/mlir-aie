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

  LogicalResult rewriteSingleBD(OpBuilder &builder, Block &block, AIE::TileOp &tile) {
    
    AIE::DMABDOp bd_op = getBdForBlock(block);

    uint32_t bd_id = bd_op.getBdId().value();
    uint32_t offset = bd_op.getOffset();
    uint32_t len = 0;
    if(bd_op.getLen().has_value()) {
        len = bd_op.getLen().value();
    } else {
        // TODO use memref/buf size
    }
    //NpuWriteBdOp npu_op = 
    builder.create<NpuWriteBdOp>(bd_op.getLoc(),
        tile.getCol(), bd_id, len, offset, 0, 0, 0, 0,
        /* TODO: Strides/Wraps */
        /*d0_size=*/1, /*d0_stride=*/1, /*d1_size=*/1, /*d1_stride=*/0, /*d2_stride=*/0,
        /*iteration_current=*/0, /*iteration_size=*/0, /*iteration_stride=*/0,
        /* TODO: Next BD */
        /*next_bd=*/0, /*row=*/tile.getRow(), /*use_next_bd=*/0,
        /*valid_bd=*/1,
        /* TODO: Locks */
        /*lock_rel_val=*/0, /*lock_rel_id=*/0, /*lock_acq_enable=*/0, /*lock_acq_val=*/0, /*lock_ackq_id=*/0);
    //rewriter.insert((mlir::Operation*)&write_bd_op);

    return success();
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

    // Convert DMAStartBD ops
    ConversionTarget target(getContext());
    target.addLegalDialect<AIEXDialect>();
    target.addIllegalOp<DMAStartBDs>();
    RewritePatternSet start_patterns(&getContext());
    start_patterns.insert<DMAStartBDsPattern>(&getContext());
    if (failed(applyPartialConversion(device, target, std::move(start_patterns)))) {
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
