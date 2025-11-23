//===- AIECoalesceWrite32s.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// This pass coalesces consecutive npu.write32 operations into npu.blockwrite
// operations when their addresses are contiguous (4-byte increments).
//
// Since register writes cannot be reordered, we only coalesce operations that
// are immediately adjacent in the instruction stream with no intervening ops.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "aie-coalesce-write32s"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;
using namespace xilinx::AIE;

namespace {

// Represents a sequence of consecutive write32 operations
struct Write32Sequence {
  SmallVector<NpuWrite32Op, 8> ops;
  uint32_t startAddress = 0;
  SmallVector<uint32_t, 8> values;
  std::optional<Location> firstLoc;
  
  bool isContiguous() const {
    return ops.size() >= 2;
  }
  
  size_t size() const {
    return ops.size();
  }
};

struct AIECoalesceWrite32sPass
    : public AIECoalesceWrite32sBase<AIECoalesceWrite32sPass> {

  void runOnOperation() override {
    AIE::DeviceOp deviceOp = getOperation();
    
    // Collect all runtime sequences in the device
    SmallVector<RuntimeSequenceOp> runtimeSeqs;
    deviceOp.walk([&](RuntimeSequenceOp seqOp) {
      runtimeSeqs.push_back(seqOp);
    });
    
    // Process each runtime sequence
    for (auto seqOp : runtimeSeqs) {
      coalesceWrite32sInSequence(seqOp);
    }
  }
  
private:
  void coalesceWrite32sInSequence(RuntimeSequenceOp seqOp) {
    OpBuilder builder(seqOp.getContext());
    
    // Walk through the sequence and find consecutive write32 operations
    for (Block &block : seqOp.getBody()) {
      SmallVector<Write32Sequence> sequences;
      Write32Sequence currentSeq;
      uint32_t expectedNextAddr = 0;
      bool inSequence = false;
      
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (auto write32Op = dyn_cast<NpuWrite32Op>(&op)) {
          // Check if this write32 has the required attributes
          // (must have address but no buffer/column/row for coalescing)
          if (write32Op.getBuffer() || write32Op.getColumn() || 
              write32Op.getRow()) {
            // Cannot coalesce symbolic writes
            if (inSequence && currentSeq.isContiguous()) {
              sequences.push_back(currentSeq);
            }
            currentSeq = Write32Sequence();
            inSequence = false;
            expectedNextAddr = 0;
            continue;
          }
          
          uint32_t addr = write32Op.getAddress();
          uint32_t value = write32Op.getValue();
          
          if (!inSequence) {
            // Start a new sequence
            currentSeq = Write32Sequence();
            currentSeq.startAddress = addr;
            currentSeq.ops.push_back(write32Op);
            currentSeq.values.push_back(value);
            currentSeq.firstLoc = write32Op.getLoc();
            expectedNextAddr = addr + 4;
            inSequence = true;
          } else if (addr == expectedNextAddr) {
            // Continue the sequence
            currentSeq.ops.push_back(write32Op);
            currentSeq.values.push_back(value);
            expectedNextAddr = addr + 4;
          } else {
            // Address not contiguous, save current sequence if valid
            if (currentSeq.isContiguous()) {
              sequences.push_back(currentSeq);
            }
            // Start a new sequence
            currentSeq = Write32Sequence();
            currentSeq.startAddress = addr;
            currentSeq.ops.push_back(write32Op);
            currentSeq.values.push_back(value);
            currentSeq.firstLoc = write32Op.getLoc();
            expectedNextAddr = addr + 4;
          }
        } else {
          // Non-write32 operation interrupts the sequence
          if (inSequence && currentSeq.isContiguous()) {
            sequences.push_back(currentSeq);
          }
          currentSeq = Write32Sequence();
          inSequence = false;
          expectedNextAddr = 0;
        }
      }
      
      // Don't forget the last sequence
      if (inSequence && currentSeq.isContiguous()) {
        sequences.push_back(currentSeq);
      }
      
      // Now replace each sequence with a blockwrite
      for (auto &seq : sequences) {
        replaceSequenceWithBlockWrite(builder, seq, seqOp);
      }
    }
  }
  
  void replaceSequenceWithBlockWrite(OpBuilder &builder, 
                                     const Write32Sequence &seq,
                                     RuntimeSequenceOp seqOp) {
    if (seq.ops.empty())
      return;
      
    MLIRContext *ctx = builder.getContext();
    Location loc = seq.firstLoc.value();
    
    // Get or create the module to hold the global
    auto moduleOp = seqOp->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return;
    
    // Get the device to hold the global
    auto deviceOp = seqOp->getParentOfType<AIE::DeviceOp>();
    if (!deviceOp)
      return;
    
    // Generate a unique name for the global memref
    static unsigned globalCounter = 0;
    std::string globalName = "coalesced_write32_" + 
                            std::to_string(seq.startAddress) + "_" +
                            std::to_string(globalCounter++);
    
    // Create memref type: memref<NxI32>
    auto i32Type = IntegerType::get(ctx, 32);
    auto memrefType = MemRefType::get({static_cast<int64_t>(seq.values.size())}, 
                                     i32Type);
    
    // Create DenseIntElementsAttr for initial values
    SmallVector<int32_t> signedValues;
    for (uint32_t val : seq.values) {
      signedValues.push_back(static_cast<int32_t>(val));
    }
    auto tensorType = RankedTensorType::get({static_cast<int64_t>(seq.values.size())}, 
                                           i32Type);
    auto valuesAttr = DenseElementsAttr::get<int32_t>(
        tensorType, ArrayRef<int32_t>(signedValues));
    
    // Insert the global at the beginning of the device
    builder.setInsertionPointToStart(deviceOp.getBody());
    builder.create<memref::GlobalOp>(
        loc, 
        globalName,
        /*sym_visibility=*/builder.getStringAttr("private"),
        /*type=*/memrefType,
        /*initial_value=*/valuesAttr,
        /*constant=*/true,
        /*alignment=*/nullptr);
    
    // Create a GetGlobalOp and BlockWriteOp at the position of the first write32
    builder.setInsertionPoint(seq.ops[0]);
    auto getGlobalOp = builder.create<memref::GetGlobalOp>(
        loc, memrefType, globalName);
    
    builder.create<NpuBlockWriteOp>(
        loc,
        seq.startAddress,
        getGlobalOp.getResult(),
        nullptr, // buffer
        nullptr, // column
        nullptr  // row
    );
    
    // Erase all the original write32 operations
    for (auto op : seq.ops) {
      op.erase();
    }
  }
};

} // namespace

namespace xilinx::AIEX {
std::unique_ptr<OperationPass<AIE::DeviceOp>> createAIECoalesceWrite32sPass() {
  return std::make_unique<AIECoalesceWrite32sPass>();
}
} // namespace xilinx::AIEX
