//===- AIEVectorToPointerLoops.cpp -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms vector.load/store operations with loop-carried indices
// to use ptr dialect operations (ptr.to_ptr, ptr.ptr_add, ptr.from_ptr).
//
// Goal: Make pointer increment patterns explicit to help LLVM backend generate
// efficient post-increment addressing modes (GEP fusion).
//
// Transformation:
//   Before:
//     scf.for iter_args(%idx = %0) -> (index) {
//       %vec = vector.load %memref[%idx]
//       %next_idx = arith.addi %idx, %stride
//       scf.yield %next_idx
//     }
//
//   After:
//     %base_ptr = ptr.to_ptr %memref
//     %init_ptr = ptr.ptr_add %base_ptr, %0
//     scf.for iter_args(%ptr = %init_ptr) -> (!ptr.ptr<...>) {
//       %memref_tmp = ptr.from_ptr %ptr
//       %vec = vector.load %memref_tmp[%c0]
//       %next_ptr = ptr.ptr_add %ptr, %stride
//       scf.yield %next_ptr
//     }
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "aie-vector-to-pointer-loops"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

/// Check if a value is an iter_arg of an scf.for loop
static bool isLoopCarriedValue(Value val, scf::ForOp forOp) {
  auto blockArgs = forOp.getRegionIterArgs();
  return llvm::is_contained(blockArgs, val);
}

/// Structure to track memref base and its uses in vector ops
struct MemrefVectorAccess {
  Value memref;
  SmallVector<vector::LoadOp> loads;
  SmallVector<vector::StoreOp> stores;
  SmallVector<Value> indices; // Loop-carried indices used
};

/// Analyze loop to find vector load/store patterns with loop-carried indices
static bool analyzeLoopForVectorAccesses(
    scf::ForOp forOp, DenseMap<Value, MemrefVectorAccess> &memrefAccesses) {

  bool foundPattern = false;

  forOp.walk([&](vector::LoadOp loadOp) {
    Value base = loadOp.getBase();
    auto indices = loadOp.getIndices();

    // Only handle 1D access for now
    if (indices.size() != 1)
      return;

    Value idx = indices[0];

    // Check if index is loop-carried
    if (isLoopCarriedValue(idx, forOp)) {
      memrefAccesses[base].memref = base;
      memrefAccesses[base].loads.push_back(loadOp);
      memrefAccesses[base].indices.push_back(idx);
      foundPattern = true;
    }
  });

  forOp.walk([&](vector::StoreOp storeOp) {
    Value base = storeOp.getBase();
    auto indices = storeOp.getIndices();

    if (indices.size() != 1)
      return;

    Value idx = indices[0];

    if (isLoopCarriedValue(idx, forOp)) {
      memrefAccesses[base].memref = base;
      memrefAccesses[base].stores.push_back(storeOp);
      memrefAccesses[base].indices.push_back(idx);
      foundPattern = true;
    }
  });

  return foundPattern;
}

/// Transform an scf.for loop to use pointer iter_args
struct VectorToPointerLoopsPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {

    // Skip if loop already uses pointer iter_args (already transformed)
    for (Value iterArg : forOp.getRegionIterArgs()) {
      if (llvm::isa<ptr::PtrType>(iterArg.getType()))
        return failure(); // Already transformed
    }

    // Analyze the loop for vector access patterns
    DenseMap<Value, MemrefVectorAccess> memrefAccesses;
    if (!analyzeLoopForVectorAccesses(forOp, memrefAccesses))
      return failure();

    if (memrefAccesses.empty())
      return failure();

    Location loc = forOp.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forOp);

    // Step 1: Convert memrefs to pointers before the loop
    DenseMap<Value, Value> memrefToPtrMap;
    DenseMap<Value, Type>
        memrefToGenericTypeMap; // Track generic-space memref types
    DenseMap<Value, Value>
        memrefToConvertedMap; // Track the converted memref value (for metadata)
    auto genericSpace = ptr::GenericSpaceAttr::get(rewriter.getContext());

    for (auto &[memref, access] : memrefAccesses) {
      // Get the memory space from the memref type
      auto memrefType = cast<MemRefType>(memref.getType());
      Attribute memorySpace = memrefType.getMemorySpace();

      Value memrefToConvert = memref;
      Type genericMemrefType = memrefType;

      // If memref has a different memory space, cast it to generic_space first
      if (memorySpace && !llvm::isa<ptr::GenericSpaceAttr>(memorySpace)) {
        // Create new memref type with generic_space
        auto newMemrefType =
            MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                            memrefType.getLayout(), genericSpace);

        // Insert unrealized_conversion_cast to convert to generic_space
        auto castOp = rewriter.create<UnrealizedConversionCastOp>(
            loc, newMemrefType, memref);
        memrefToConvert = castOp.getResult(0);
        genericMemrefType = newMemrefType;
      }

      // Create pointer type with generic_space
      auto ptrType = ptr::PtrType::get(rewriter.getContext(), genericSpace);
      auto ptrOp = rewriter.create<ptr::ToPtrOp>(loc, ptrType, memrefToConvert);
      memrefToPtrMap[memref] = ptrOp.getResult();
      memrefToGenericTypeMap[memref] = genericMemrefType;
      memrefToConvertedMap[memref] =
          memrefToConvert; // Store the converted memref
    }

    // Step 2: Identify which iter_args are indices used in vector ops
    SmallVector<unsigned> indexIterArgPositions;
    SmallVector<Value> correspondingMemrefs;

    for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      for (auto &[memref, access] : memrefAccesses) {
        if (llvm::is_contained(access.indices, iterArg)) {
          indexIterArgPositions.push_back(idx);
          correspondingMemrefs.push_back(memref);
          break;
        }
      }
    }

    if (indexIterArgPositions.empty())
      return failure();

    // Step 3: Build new init args with pointers
    SmallVector<Value> newInitArgs;
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
      auto it = llvm::find(indexIterArgPositions, idx);
      if (it != indexIterArgPositions.end()) {
        // This is an index iter_arg - convert to pointer
        size_t pos = std::distance(indexIterArgPositions.begin(), it);
        Value memref = correspondingMemrefs[pos];
        Value basePtr = memrefToPtrMap[memref];

        // Get element size in bytes
        auto memrefType = cast<MemRefType>(memref.getType());
        unsigned elementSizeBits = memrefType.getElementTypeBitWidth();
        unsigned elementSizeBytes = (elementSizeBits + 7) / 8;

        // Scale index by element size: byteOffset = initArg * elementSizeBytes
        Value byteOffset = initArg;
        if (elementSizeBytes != 1) {
          Value elementSize =
              rewriter.create<arith::ConstantIndexOp>(loc, elementSizeBytes);
          byteOffset =
              rewriter.create<arith::MulIOp>(loc, initArg, elementSize);
        }

        // Create: ptr.ptr_add basePtr, byteOffset
        auto initPtrOp =
            rewriter.create<ptr::PtrAddOp>(loc, basePtr, byteOffset);
        newInitArgs.push_back(initPtrOp.getResult());
      } else {
        // Keep as-is
        newInitArgs.push_back(initArg);
      }
    }

    // Step 4: Create new loop with updated signature
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep(), newInitArgs);

    // Step 5: Transform loop body (simplified - doesn't handle all cases yet)
    IRMapping mapper;
    mapper.map(forOp.getInductionVar(), newForOp.getInductionVar());

    for (auto [oldArg, newArg] :
         llvm::zip(forOp.getRegionIterArgs(), newForOp.getRegionIterArgs())) {
      mapper.map(oldArg, newArg);
    }

    rewriter.setInsertionPointToStart(newForOp.getBody());

    // Clone operations with transformation (c0 already created above)
    for (Operation &op : forOp.getBody()->without_terminator()) {
      // Transform vector.load operations
      if (auto loadOp = dyn_cast<vector::LoadOp>(&op)) {
        Value idx = loadOp.getIndices()[0];
        Value mappedIdx = mapper.lookup(idx);

        // Check if the index is now a pointer (was transformed)
        if (llvm::isa<ptr::PtrType>(mappedIdx.getType())) {
          // Get the generic-space memref type for this base
          Type genericType = memrefToGenericTypeMap[loadOp.getBase()];

          // Get metadata from the converted memref (with generic_space)
          Value convertedMemref = memrefToConvertedMap[loadOp.getBase()];
          auto metadataOp = rewriter.create<ptr::GetMetadataOp>(
              loadOp.getLoc(), convertedMemref);

          // Transform: vector.load %memref[%ptr] -> ptr.from_ptr + vector.load
          // [...[0]]
          auto fromPtrOp = rewriter.create<ptr::FromPtrOp>(
              loadOp.getLoc(), genericType, mappedIdx, metadataOp.getResult());
          auto newLoad = rewriter.create<vector::LoadOp>(
              loadOp.getLoc(), loadOp.getVectorType(), fromPtrOp.getResult(),
              ValueRange{c0});
          mapper.map(loadOp.getResult(), newLoad.getResult());
          continue;
        }
      }

      // Transform vector.store operations
      if (auto storeOp = dyn_cast<vector::StoreOp>(&op)) {
        Value idx = storeOp.getIndices()[0];
        Value mappedIdx = mapper.lookup(idx);

        if (llvm::isa<ptr::PtrType>(mappedIdx.getType())) {
          // Get the generic-space memref type for this base
          Type genericType = memrefToGenericTypeMap[storeOp.getBase()];

          // Get metadata from the converted memref (with generic_space)
          Value convertedMemref = memrefToConvertedMap[storeOp.getBase()];
          auto metadataOp = rewriter.create<ptr::GetMetadataOp>(
              storeOp.getLoc(), convertedMemref);

          // Transform: vector.store %val, %memref[%ptr] -> ptr.from_ptr +
          // vector.store[0]
          auto fromPtrOp = rewriter.create<ptr::FromPtrOp>(
              storeOp.getLoc(), genericType, mappedIdx, metadataOp.getResult());
          Value valueToStore =
              mapper.lookupOrDefault(storeOp.getValueToStore());
          rewriter.create<vector::StoreOp>(storeOp.getLoc(), valueToStore,
                                           fromPtrOp.getResult(),
                                           ValueRange{c0});
          continue;
        }
      }

      // Transform arith.addi to ptr.ptr_add when operating on pointers
      if (auto addiOp = dyn_cast<arith::AddIOp>(&op)) {
        Value lhs = mapper.lookupOrDefault(addiOp.getLhs());
        Value rhs = mapper.lookupOrDefault(addiOp.getRhs());

        // If LHS is a pointer, convert to ptr.ptr_add
        if (llvm::isa<ptr::PtrType>(lhs.getType())) {
          // Find which memref this pointer corresponds to
          Value memrefForPtr = nullptr;
          for (auto [memref, access] : memrefAccesses) {
            // Check if lhs comes from this memref's pointer chain
            // For now, find any memref being accessed (simplified)
            memrefForPtr = memref;
            break;
          }

          Value byteOffset = rhs;
          if (memrefForPtr) {
            auto memrefType = cast<MemRefType>(memrefForPtr.getType());
            unsigned elementSizeBits = memrefType.getElementTypeBitWidth();
            unsigned elementSizeBytes = (elementSizeBits + 7) / 8;

            // Scale offset by element size: byteOffset = rhs * elementSizeBytes
            if (elementSizeBytes != 1) {
              Value elementSize = rewriter.create<arith::ConstantIndexOp>(
                  addiOp.getLoc(), elementSizeBytes);
              byteOffset = rewriter.create<arith::MulIOp>(addiOp.getLoc(), rhs,
                                                          elementSize);
            }
          }

          auto ptrAddOp =
              rewriter.create<ptr::PtrAddOp>(addiOp.getLoc(), lhs, byteOffset);
          mapper.map(addiOp.getResult(), ptrAddOp.getResult());
          continue;
        }
      }

      // Default: clone the operation
      rewriter.clone(op, mapper);
    }

    // Clone yield
    auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    SmallVector<Value> newYieldOperands;
    for (Value operand : oldYield.getOperands()) {
      newYieldOperands.push_back(mapper.lookupOrDefault(operand));
    }
    rewriter.create<scf::YieldOp>(loc, newYieldOperands);

    // Step 6: Replace old loop
    rewriter.replaceOp(forOp, newForOp.getResults());

    // NOTE: This is a simplified implementation
    // Full version needs to properly transform vector.load/store and arith.addi

    return success();
  }
};

struct AIEVectorToPointerLoopsPass
    : public PassWrapper<AIEVectorToPointerLoopsPass, OperationPass<DeviceOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AIEVectorToPointerLoopsPass)

  StringRef getArgument() const override {
    return "aie-vector-to-pointer-loops";
  }

  StringRef getDescription() const override {
    return "Transform vector.load/store with loop-carried indices to use ptr "
           "dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ptr::PtrDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    DeviceOp deviceOp = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<VectorToPointerLoopsPattern>(&getContext());

    // Apply patterns to the entire device
    // The pattern will match scf.for loops in aie.core regions
    if (failed(applyPatternsGreedily(deviceOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace xilinx {
namespace AIE {

std::unique_ptr<OperationPass<DeviceOp>> createAIEVectorToPointerLoopsPass() {
  return std::make_unique<AIEVectorToPointerLoopsPass>();
}

} // namespace AIE
} // namespace xilinx
