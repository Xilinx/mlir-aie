//===- AIEAssignRuntimeSequenceBDIDs.cpp ---------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

struct AIEAssignRuntimeSequenceBDIDsPass
    : AIEAssignRuntimeSequenceBDIDsBase<AIEAssignRuntimeSequenceBDIDsPass> {

  BdIdGenerator &
  getGeneratorForTile(AIE::TileOp tile,
                      std::map<AIE::TileOp, BdIdGenerator> &gens) {
    const AIETargetModel &targetModel =
        tile->getParentOfType<AIE::DeviceOp>().getTargetModel();
    auto genIt = gens.find(tile);
    if (genIt == gens.end()) {
      gens.insert(std::pair(
          tile, BdIdGenerator(tile.getCol(), tile.getRow(), targetModel)));
    }
    BdIdGenerator &gen = std::get<1>(*gens.find(tile));
    return gen;
  }

  LogicalResult runOnConfigureBDs(DMAConfigureTaskOp op,
                                  std::map<AIE::TileOp, BdIdGenerator> &gens) {
    AIE::TileOp tile = op.getTileOp();
    BdIdGenerator &gen = getGeneratorForTile(tile, gens);

    // First, honor all the user-specified BD IDs
    WalkResult result = op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
      if (bd_op.getBdId().has_value()) {
        if (gen.bdIdAlreadyAssigned(bd_op.getBdId().value())) {
          op.emitOpError("Specified buffer descriptor ID ")
              << bd_op.getBdId().value()
              << " is already in use. Emit an aiex.dma_free_task operation to "
                 "reuse BDs.";
          return WalkResult::interrupt();
        }
        gen.assignBdId(bd_op.getBdId().value());
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return failure();
    }

    // Now, allocate BD IDs for all unspecified BDs
    result =
        op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
          if (bd_op.getBdId().has_value()) {
            return WalkResult::advance();
          }
          // FIXME: use correct channelIndex here
          std::optional<int32_t> next_id = gen.nextBdId(/*channelIndex=*/0);
          if (!next_id) {
            op.emitOpError()
                << "Allocator exhausted available buffer descriptor IDs.";
            return WalkResult::interrupt();
          }
          bd_op.setBdId(*next_id);
          return WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      return failure();
    }

    return success();
  }

  LogicalResult runOnFreeBDs(DMAFreeTaskOp op,
                             std::map<AIE::TileOp, BdIdGenerator> &gens) {
    DMAConfigureTaskOp task_op = op.getTaskOp();
    if (!task_op) {
      auto err = op.emitOpError(
          "does not reference a valid configure_task operation.");
      Operation *task_op = op.getTask().getDefiningOp();
      if (llvm::isa<DMAStartBdChainOp>(task_op)) {
        err.attachNote(task_op->getLoc())
            << "Lower this operation first using the "
               "--aie-materialize-bd-chains pass.";
      }
      return err;
    }

    AIE::TileOp tile = task_op.getTileOp();
    BdIdGenerator &gen = getGeneratorForTile(tile, gens);

    // First, honor all the hard-coded BD IDs
    WalkResult result =
        task_op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
          if (!bd_op.getBdId().has_value()) {
            bd_op.emitOpError("Free called on BD chain with unassigned IDs.");
            return WalkResult::interrupt();
          }
          assert(gen.bdIdAlreadyAssigned(bd_op.getBdId().value()) &&
                 "MLIR state and BdIDGenerator state out of sync.");
          gen.freeBdId(bd_op.getBdId().value());
          return WalkResult::advance();
        });
    if (result.wasInterrupted()) {
      return failure();
    }

    op.erase();

    return success();
  }

  void runOnOperation() override {

    // This pass currently assigns BD IDs with a simple linear pass. IDs are
    // assigned in sequence, and issuing an aiex.free_bds or aiex.await_bds op
    // kills the correspondings IDs use. If in the future we support
    // branching/jumping in the sequence function, a proper liveness analysis
    // will become necessary here.

    AIE::DeviceOp device = getOperation();
    std::map<AIE::TileOp, BdIdGenerator> gens;

    // Insert a free_bds operation for each await_bds
    // After waiting for BD IDs, they can definitely safely be reused
    device.walk([&](DMAAwaitTaskOp op) {
      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);
      builder.create<DMAFreeTaskOp>(op.getLoc(), op.getTask());
    });

    // TODO: Only walk the sequence function
    device.walk([&](Operation *op) {
      LogicalResult result =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<DMAConfigureTaskOp>([&](DMAConfigureTaskOp op) {
                return runOnConfigureBDs(op, gens);
              })
              .Case<DMAFreeTaskOp>(
                  [&](DMAFreeTaskOp op) { return runOnFreeBDs(op, gens); })
              .Default([](Operation *op) { return success(); });
      if (failed(result)) {
        return signalPassFailure();
      }
    });
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEAssignRuntimeSequenceBDIDsPass() {
  return std::make_unique<AIEAssignRuntimeSequenceBDIDsPass>();
}
