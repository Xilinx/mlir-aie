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
// The pass can reorder non-special register writes within slices. Special
// register writes act as barriers and cannot be reordered. Duplicate writes
// to the same address are eliminated (keeping the last value).
//
// A configurable threshold controls the minimum number of contiguous writes
// required for coalescing into a blockwrite.
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

// Represents a single 32-bit write to memory
struct WriteWord {
  uint32_t address;
  uint32_t value;
  Operation *op;  // The original operation that created this write
  
  bool operator<(const WriteWord &other) const {
    return address < other.address;
  }
};

// Represents a slice of execution between special register barriers
// All writes within a slice can be reordered
struct WriteSlice {
  SmallVector<WriteWord, 0> writes;  // All writes in this slice (can be reordered)
  SmallVector<Operation *, 0> specialOps;  // Special register writes (cannot be reordered)
  SmallVector<Operation *, 0> otherOps;  // Non-write operations that create barriers
  
  bool isEmpty() const {
    return writes.empty() && specialOps.empty() && otherOps.empty();
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
    
    // Get the device and target model
    auto deviceOp = seqOp->getParentOfType<AIE::DeviceOp>();
    if (!deviceOp) return;
    
    const auto &tm = deviceOp.getTargetModel();
    
    // Walk through the sequence and partition into slices
    for (Block &block : seqOp.getBody()) {
      SmallVector<WriteSlice> slices;
      WriteSlice currentSlice;
      
      for (Operation &op : block) {
        // Skip get_global operations - they don't create barriers
        if (isa<memref::GetGlobalOp>(&op)) {
          continue;
        }
        
        bool isSpecialWrite = false;
        
        // Check if this is a write32 operation
        if (auto write32Op = dyn_cast<NpuWrite32Op>(&op)) {
          // Only handle writes with absolute addresses (no buffer/col/row)
          if (!write32Op.getBuffer() && !write32Op.getColumn() && 
              !write32Op.getRow()) {
            uint32_t addr = write32Op.getAddress();
            isSpecialWrite = tm.isSpecialRegister(addr);
            
            if (isSpecialWrite) {
              currentSlice.specialOps.push_back(&op);
            } else {
              WriteWord word{addr, write32Op.getValue(), &op};
              currentSlice.writes.push_back(word);
            }
          } else {
            // Symbolic write creates a barrier
            if (!currentSlice.isEmpty()) {
              slices.push_back(std::move(currentSlice));
              currentSlice = WriteSlice();
            }
            currentSlice.otherOps.push_back(&op);
          }
        }
        // Check if this is a blockwrite operation
        else if (auto blockWriteOp = dyn_cast<NpuBlockWriteOp>(&op)) {
          // Only handle writes with absolute addresses
          if (!blockWriteOp.getBuffer() && !blockWriteOp.getColumn() && 
              !blockWriteOp.getRow()) {
            auto addr = blockWriteOp.getAbsoluteAddress();
            auto dataWords = blockWriteOp.getDataWords();
            
            if (addr && dataWords) {
              uint32_t baseAddr = *addr;
              
              // Check if any word writes to a special register
              int64_t numWords = dataWords.size();
              for (int64_t i = 0; i < numWords; ++i) {
                if (tm.isSpecialRegister(baseAddr + i * 4)) {
                  isSpecialWrite = true;
                  break;
                }
              }
              
              if (isSpecialWrite) {
                currentSlice.specialOps.push_back(&op);
              } else {
                // Add each word in the blockwrite as a separate WriteWord
                int64_t idx = 0;
                for (auto val : dataWords.getValues<APInt>()) {
                  WriteWord word{
                    static_cast<uint32_t>(baseAddr + idx * 4),
                    static_cast<uint32_t>(val.getZExtValue()),
                    &op
                  };
                  currentSlice.writes.push_back(word);
                  ++idx;
                }
              }
            } else {
              // Invalid blockwrite creates a barrier
              if (!currentSlice.isEmpty()) {
                slices.push_back(std::move(currentSlice));
                currentSlice = WriteSlice();
              }
              currentSlice.otherOps.push_back(&op);
            }
          } else {
            // Symbolic write creates a barrier
            if (!currentSlice.isEmpty()) {
              slices.push_back(std::move(currentSlice));
              currentSlice = WriteSlice();
            }
            currentSlice.otherOps.push_back(&op);
          }
        }
        // Any other operation creates a barrier
        else {
          if (!currentSlice.isEmpty()) {
            slices.push_back(std::move(currentSlice));
            currentSlice = WriteSlice();
          }
          currentSlice.otherOps.push_back(&op);
        }
        
        // Special register writes create slice boundaries
        if (isSpecialWrite) {
          slices.push_back(std::move(currentSlice));
          currentSlice = WriteSlice();
        }
      }
      
      // Don't forget the last slice
      if (!currentSlice.isEmpty()) {
        slices.push_back(std::move(currentSlice));
      }
      
      // Now process each slice: sort writes and coalesce
      for (auto &slice : slices) {
        processSlice(builder, slice, deviceOp);
      }
    }
  }
  
  void processSlice(OpBuilder &builder, WriteSlice &slice, 
                    AIE::DeviceOp deviceOp) {
    // Non-write operations (barriers) are kept in their original positions
    // Nothing to do here - they're already in the IR
    
    // If there are no writes to coalesce, nothing to do
    if (slice.writes.empty()) {
      return;
    }
    
    // Eliminate duplicate writes - keep only the last write to each address
    // Map from address to the index of the last write to that address
    DenseMap<uint32_t, size_t> lastWriteIndex;
    DenseSet<Operation *> supersededOps;  // Operations that are superseded by later writes
    
    for (size_t i = 0; i < slice.writes.size(); ++i) {
      uint32_t addr = slice.writes[i].address;
      
      // If there was a previous write to this address, mark it as superseded
      if (lastWriteIndex.count(addr)) {
        supersededOps.insert(slice.writes[lastWriteIndex[addr]].op);
      }
      
      lastWriteIndex[addr] = i;
    }
    
    // Filter writes to keep only the last write to each address
    SmallVector<WriteWord> uniqueWrites;
    for (size_t i = 0; i < slice.writes.size(); ++i) {
      if (lastWriteIndex[slice.writes[i].address] == i) {
        uniqueWrites.push_back(slice.writes[i]);
      }
    }
    
    // Update slice.writes with unique writes
    slice.writes = std::move(uniqueWrites);
    
    // Sort writes by address to enable coalescing
    llvm::sort(slice.writes);
    
    // Find contiguous sequences and coalesce them
    SmallVector<SmallVector<WriteWord>> sequences;
    SmallVector<WriteWord> currentSeq;
    
    for (auto &write : slice.writes) {
      if (currentSeq.empty()) {
        currentSeq.push_back(write);
      } else if (write.address == currentSeq.back().address + 4) {
        // Contiguous with previous write
        currentSeq.push_back(write);
      } else {
        // Not contiguous, start new sequence
        // Only save sequences that meet the minimum threshold
        if (currentSeq.size() >= minWritesToCoalesce) {
          sequences.push_back(currentSeq);
        }
        currentSeq.clear();
        currentSeq.push_back(write);
      }
    }
    
    // Don't forget the last sequence (also check threshold)
    if (currentSeq.size() >= minWritesToCoalesce) {
      sequences.push_back(currentSeq);
    }
    
    // Track which operations to erase
    DenseSet<Operation *> toErase = supersededOps;  // Start with superseded ops
    
    // Create blockwrites for each sequence
    for (auto &seq : sequences) {
      createBlockWrite(builder, seq, deviceOp);
      
      // Mark operations for erasure
      for (auto &write : seq) {
        toErase.insert(write.op);
      }
    }
    
    // For writes that couldn't be coalesced, keep them as write32 ops
    for (auto &write : slice.writes) {
      if (!toErase.contains(write.op)) {
        // This write wasn't coalesced - keep it but possibly reorder
        // The write is already in the IR, we just need to move it if needed
      }
    }
    
    // Special register writes stay in their original positions
    // Nothing to do here - they're already in the IR
    
    // Erase operations that were coalesced or superseded
    for (auto *op : toErase) {
      op->erase();
    }
  }
  
  void createBlockWrite(OpBuilder &builder, 
                       const SmallVector<WriteWord> &sequence,
                       AIE::DeviceOp deviceOp) {
    if (sequence.empty()) return;
    
    MLIRContext *ctx = builder.getContext();
    uint32_t startAddr = sequence[0].address;
    
    // Generate a unique name for the global memref
    static unsigned globalCounter = 0;
    std::string globalName = "coalesced_write32_" + 
                            std::to_string(startAddr) + "_" +
                            std::to_string(globalCounter++);
    
    // Collect values
    SmallVector<int32_t> values;
    for (auto &write : sequence) {
      values.push_back(static_cast<int32_t>(write.value));
    }
    
    // Create memref type
    auto i32Type = IntegerType::get(ctx, 32);
    auto memrefType = MemRefType::get({static_cast<int64_t>(values.size())}, 
                                     i32Type);
    
    // Create DenseIntElementsAttr for initial values
    auto tensorType = RankedTensorType::get({static_cast<int64_t>(values.size())}, 
                                           i32Type);
    auto valuesAttr = DenseElementsAttr::get<int32_t>(
        tensorType, ArrayRef<int32_t>(values));
    
    // Get location from first operation
    Location loc = sequence[0].op->getLoc();
    
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
    
    // Create a GetGlobalOp and BlockWriteOp at the position of the first write
    builder.setInsertionPoint(sequence[0].op);
    auto getGlobalOp = builder.create<memref::GetGlobalOp>(
        loc, memrefType, globalName);
    
    builder.create<NpuBlockWriteOp>(
        loc,
        startAddr,
        getGlobalOp.getResult(),
        nullptr, // buffer
        nullptr, // column
        nullptr  // row
    );
  }
};

} // namespace

namespace xilinx::AIEX {
std::unique_ptr<OperationPass<AIE::DeviceOp>> createAIECoalesceWrite32sPass() {
  return std::make_unique<AIECoalesceWrite32sPass>();
}
} // namespace xilinx::AIEX
