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

// Represents a write operation that can be either write32 or blockwrite
struct WriteOp {
  Operation *op = nullptr;
  uint32_t address = 0;
  SmallVector<uint32_t, 8> values;
  
  WriteOp() = default;
  WriteOp(NpuWrite32Op w32) : op(w32.getOperation()), address(w32.getAddress()) {
    values.push_back(w32.getValue());
  }
  
  WriteOp(NpuBlockWriteOp bw) : op(bw.getOperation()) {
    auto addr = bw.getAbsoluteAddress();
    if (!addr) return;
    address = *addr;
    
    auto dataWords = bw.getDataWords();
    if (!dataWords) return;
    
    for (auto val : dataWords.getValues<APInt>()) {
      values.push_back(val.getZExtValue());
    }
  }
  
  bool isValid() const {
    return op != nullptr && !values.empty();
  }
  
  size_t numWords() const {
    return values.size();
  }
};

// Represents a sequence of consecutive write operations (write32 or blockwrite)
struct WriteSequence {
  SmallVector<WriteOp, 0> ops;  // Don't inline - WriteOp is large
  uint32_t startAddress = 0;
  SmallVector<uint32_t, 32> values;
  std::optional<Location> firstLoc;
  
  bool isCoalescable() const {
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
    
    // Walk through the sequence and find consecutive write operations
    for (Block &block : seqOp.getBody()) {
      SmallVector<WriteSequence> sequences;
      WriteSequence currentSeq;
      uint32_t expectedNextAddr = 0;
      bool inSequence = false;
      
      for (Operation &op : llvm::make_early_inc_range(block)) {
        WriteOp writeOp;
        
        // Skip get_global operations as they don't interrupt sequences
        if (isa<memref::GetGlobalOp>(&op)) {
          continue;
        }
        
        if (auto write32Op = dyn_cast<NpuWrite32Op>(&op)) {
          // Check if this write32 has the required attributes
          // (must have address but no buffer/column/row for coalescing)
          if (write32Op.getBuffer() || write32Op.getColumn() || 
              write32Op.getRow()) {
            // Cannot coalesce symbolic writes
            if (inSequence && currentSeq.isCoalescable()) {
              sequences.push_back(currentSeq);
            }
            currentSeq = WriteSequence();
            inSequence = false;
            expectedNextAddr = 0;
            continue;
          }
          
          writeOp = WriteOp(write32Op);
        } else if (auto blockWriteOp = dyn_cast<NpuBlockWriteOp>(&op)) {
          // Check if this blockwrite has the required attributes
          if (blockWriteOp.getBuffer() || blockWriteOp.getColumn() || 
              blockWriteOp.getRow()) {
            // Cannot coalesce symbolic writes
            if (inSequence && currentSeq.isCoalescable()) {
              sequences.push_back(currentSeq);
            }
            currentSeq = WriteSequence();
            inSequence = false;
            expectedNextAddr = 0;
            continue;
          }
          
          writeOp = WriteOp(blockWriteOp);
          if (!writeOp.isValid()) {
            // Invalid blockwrite (couldn't get address or data), skip coalescing
            if (inSequence && currentSeq.isCoalescable()) {
              sequences.push_back(currentSeq);
            }
            currentSeq = WriteSequence();
            inSequence = false;
            expectedNextAddr = 0;
            continue;
          }
        } else {
          // Non-write operation interrupts the sequence
          if (inSequence && currentSeq.isCoalescable()) {
            sequences.push_back(currentSeq);
          }
          currentSeq = WriteSequence();
          inSequence = false;
          expectedNextAddr = 0;
          continue;
        }
        
        // Process the write operation
        if (!inSequence) {
          // Start a new sequence
          currentSeq = WriteSequence();
          currentSeq.startAddress = writeOp.address;
          currentSeq.ops.push_back(writeOp);
          currentSeq.values.append(writeOp.values.begin(), writeOp.values.end());
          currentSeq.firstLoc = writeOp.op->getLoc();
          expectedNextAddr = writeOp.address + writeOp.numWords() * 4;
          inSequence = true;
        } else if (writeOp.address == expectedNextAddr) {
          // Continue the sequence
          currentSeq.ops.push_back(writeOp);
          currentSeq.values.append(writeOp.values.begin(), writeOp.values.end());
          expectedNextAddr = writeOp.address + writeOp.numWords() * 4;
        } else {
          // Address not contiguous, save current sequence if valid
          if (currentSeq.isCoalescable()) {
            sequences.push_back(currentSeq);
          }
          // Start a new sequence
          currentSeq = WriteSequence();
          currentSeq.startAddress = writeOp.address;
          currentSeq.ops.push_back(writeOp);
          currentSeq.values.append(writeOp.values.begin(), writeOp.values.end());
          currentSeq.firstLoc = writeOp.op->getLoc();
          expectedNextAddr = writeOp.address + writeOp.numWords() * 4;
        }
      }
      
      // Don't forget the last sequence
      if (inSequence && currentSeq.isCoalescable()) {
        sequences.push_back(currentSeq);
      }
      
      // Now replace each sequence with a blockwrite
      for (auto &seq : sequences) {
        replaceSequenceWithBlockWrite(builder, seq, seqOp);
      }
    }
  }
  
  void replaceSequenceWithBlockWrite(OpBuilder &builder, 
                                     const WriteSequence &seq,
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
    
    // Create a GetGlobalOp and BlockWriteOp at the position of the first operation
    builder.setInsertionPoint(seq.ops[0].op);
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
    
    // Erase all the original operations
    for (const auto &writeOp : seq.ops) {
      writeOp.op->erase();
    }
  }
};

} // namespace

namespace xilinx::AIEX {
std::unique_ptr<OperationPass<AIE::DeviceOp>> createAIECoalesceWrite32sPass() {
  return std::make_unique<AIECoalesceWrite32sPass>();
}
} // namespace xilinx::AIEX
