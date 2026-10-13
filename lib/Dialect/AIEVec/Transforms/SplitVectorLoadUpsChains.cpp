//===- SplitVectorLoadUpsChains.cpp - Split Load+UPS Chains ----*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Splits 1024-bit vector loads and stores into 512-bit halves where the
// value is widened or narrowed one half at a time on AIE2P: vector.load +
// aievec.ups, aievec.srs + vector.store, and bf16 loads feeding the LHS of
// aievec.matmul_aie2p. Each half then maps to its own memory access instead of
// a shuffle of the full-width value.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/Sequence.h"

#define DEBUG_TYPE "aievec-split-load-ups-chains"

using namespace mlir;
using namespace xilinx::aievec;

namespace {

/// Pattern to optimize vector.load + aievec.ups chains by splitting them.
///
/// This pattern detects cases where a 1024-bit vector is loaded and then
/// passed to an aievec.ups operation that produces a 2048-bit result.
/// Instead of the inefficient approach of:
///   1. Load 1024 bits
///   2. Shuffle to split into 2×512 bits
///   3. Apply 2× UPS operations
///   4. Shuffle to concatenate results
///
/// It transforms to:
///   1. Load 2×512 bits directly
///   2. Apply 2× UPS operations immediately
///   3. Shuffle once to concatenate results
///
/// This reduces shuffle operations from 3 to 1.
struct SplitVectorLoadUpsChainPattern : public OpRewritePattern<UPSOp> {
  using OpRewritePattern<UPSOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UPSOp upsOp,
                                PatternRewriter &rewriter) const override {
    // Get source value and its type
    Value source = upsOp.getSource();
    auto srcVecTy = dyn_cast<VectorType>(source.getType());
    if (!srcVecTy)
      return failure();

    // Get result type
    auto resultVecTy = dyn_cast<VectorType>(upsOp.getResult().getType());
    if (!resultVecTy)
      return failure();

    // Check if this is a 1024-bit -> 2048-bit integer UPS
    Type srcElemTy = srcVecTy.getElementType();
    Type resultElemTy = resultVecTy.getElementType();

    if (!srcElemTy.isInteger() || !resultElemTy.isInteger())
      return failure();

    unsigned srcBitWidth = srcElemTy.getIntOrFloatBitWidth();
    unsigned resultBitWidth = resultElemTy.getIntOrFloatBitWidth();
    int64_t srcLanes = srcVecTy.getNumElements();
    int64_t resultLanes = resultVecTy.getNumElements();

    int64_t srcVectorSize = srcBitWidth * srcLanes;
    int64_t resultVectorSize = resultBitWidth * resultLanes;

    // Only optimize the 1024-bit -> 2048-bit case
    // (e.g., v64int16 -> v64acc32)
    if (srcVectorSize != 1024 || resultVectorSize != 2048)
      return failure();

    // Check that the UPS result width is 32 and source width is 16
    if (resultBitWidth != 32 || srcBitWidth != 16)
      return failure();

    // Check if source is directly from a vector.load
    auto loadOp = source.getDefiningOp<vector::LoadOp>();
    if (!loadOp)
      return failure();

    // Ensure the load is only used by this UPS operation
    if (!loadOp.getResult().hasOneUse())
      return failure();

    Location loc = upsOp.getLoc();

    // Get load operation details
    Value memRef = loadOp.getBase();
    ValueRange indices = loadOp.getIndices();

    // Create element type for half-sized vector (v32int16)
    int64_t halfLanes = srcLanes / 2;
    auto halfSrcVecTy = VectorType::get({halfLanes}, srcElemTy);
    auto halfResultVecTy = VectorType::get({halfLanes}, resultElemTy);

    // Calculate offset for second half load
    // For v64int16, we need to offset by 32 elements (64 bytes for i16)
    int64_t elementOffset = halfLanes;

    // Create indices for first half load (same as original)
    SmallVector<Value> firstHalfIndices(indices.begin(), indices.end());

    // Create indices for second half load
    SmallVector<Value> secondHalfIndices(indices.begin(), indices.end());

    // Adjust the last index by the element offset
    if (!indices.empty()) {
      Value lastIdx = indices.back();
      Value offsetVal =
          arith::ConstantIndexOp::create(rewriter, loc, elementOffset);
      Value newLastIdx =
          arith::AddIOp::create(rewriter, loc, lastIdx, offsetVal);
      secondHalfIndices.back() = newLastIdx;
    }

    // Create first half load
    auto loadHalf0 = vector::LoadOp::create(rewriter, loc, halfSrcVecTy, memRef,
                                            firstHalfIndices);

    // Create second half load
    auto loadHalf1 = vector::LoadOp::create(rewriter, loc, halfSrcVecTy, memRef,
                                            secondHalfIndices);

    // Create UPS for first half
    auto upsHalf0 = UPSOp::create(rewriter, loc, halfResultVecTy,
                                  loadHalf0.getResult(), upsOp.getShift());

    // Create UPS for second half
    auto upsHalf1 = UPSOp::create(rewriter, loc, halfResultVecTy,
                                  loadHalf1.getResult(), upsOp.getShift());

    // Concatenate the two halves using vector.shuffle
    // The mask is sequential from 0 to 63 to concatenate [half0; half1]
    SmallVector<int64_t> concatMask;
    for (int64_t i = 0; i < resultLanes; ++i) {
      concatMask.push_back(i);
    }

    auto concatOp = vector::ShuffleOp::create(
        rewriter, loc, upsHalf0.getResult(), upsHalf1.getResult(), concatMask);

    // Replace the original UPS operation with the concatenated result
    rewriter.replaceOp(upsOp, concatOp.getResult());

    // The original load will be removed by dead code elimination
    // since it no longer has any uses

    return success();
  }
};

/// Pattern to optimize aievec.srs + vector.store chains by splitting them.
///
/// This pattern detects cases where a 2048-bit vector is passed to an
/// aievec.srs operation that produces a 1024-bit result, which is then stored.
/// Instead of the inefficient approach of:
///   1. Shuffle to split 2048-bit into 2×1024 bits
///   2. Apply 2× SRS operations
///   3. Shuffle to concatenate results
///   4. Store 1024 bits
///
/// It transforms to:
///   1. Split source via shuffle into 2×1024 bits (for SRS input)
///   2. Apply 2× SRS operations to get 2×512 bits
///   3. Store 2×512 bits directly
///
/// This reduces shuffle operations from 3 to 1.
struct SplitVectorSrsStoreChainPattern
    : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    // Get the value being stored and its type
    Value valueToStore = storeOp.getValueToStore();
    auto storeVecTy = dyn_cast<VectorType>(valueToStore.getType());
    if (!storeVecTy)
      return failure();

    // Check if the value comes from an aievec.srs operation
    auto srsOp = valueToStore.getDefiningOp<SRSOp>();
    if (!srsOp)
      return failure();

    // Ensure the SRS is only used by this store operation
    if (!srsOp.getResult().hasOneUse())
      return failure();

    // Get source and result types of SRS
    Value srsSource = srsOp.getSource();
    auto srcVecTy = dyn_cast<VectorType>(srsSource.getType());
    if (!srcVecTy)
      return failure();

    Type srcElemTy = srcVecTy.getElementType();
    Type resultElemTy = storeVecTy.getElementType();

    if (!srcElemTy.isInteger() || !resultElemTy.isInteger())
      return failure();

    unsigned srcBitWidth = srcElemTy.getIntOrFloatBitWidth();
    unsigned resultBitWidth = resultElemTy.getIntOrFloatBitWidth();
    int64_t srcLanes = srcVecTy.getNumElements();
    int64_t resultLanes = storeVecTy.getNumElements();

    int64_t srcVectorSize = srcBitWidth * srcLanes;
    int64_t resultVectorSize = resultBitWidth * resultLanes;

    // Only optimize the 2048-bit -> 1024-bit case
    // (e.g., v64acc32 -> v64int16)
    if (srcVectorSize != 2048 || resultVectorSize != 1024)
      return failure();

    // Check that the SRS source width is 32 and result width is 16
    if (srcBitWidth != 32 || resultBitWidth != 16)
      return failure();

    Location loc = storeOp.getLoc();

    // Get store operation details
    Value memRef = storeOp.getBase();
    ValueRange indices = storeOp.getIndices();

    // Create element types for half-sized vectors
    int64_t halfSrcLanes = srcLanes / 2;
    int64_t halfResultLanes = resultLanes / 2;
    auto halfResultVecTy = VectorType::get({halfResultLanes}, resultElemTy);

    // Split the SRS source into two halves using shuffle
    SmallVector<int64_t> firstHalfMask, secondHalfMask;
    for (int64_t i = 0; i < halfSrcLanes; ++i) {
      firstHalfMask.push_back(i);
      secondHalfMask.push_back(halfSrcLanes + i);
    }

    auto srcHalf0 = vector::ShuffleOp::create(rewriter, loc, srsSource,
                                              srsSource, firstHalfMask);
    auto srcHalf1 = vector::ShuffleOp::create(rewriter, loc, srsSource,
                                              srsSource, secondHalfMask);

    // Create SRS for first half
    auto srsHalf0 = SRSOp::create(rewriter, loc, halfResultVecTy,
                                  srcHalf0.getResult(), srsOp.getShift());

    // Create SRS for second half
    auto srsHalf1 = SRSOp::create(rewriter, loc, halfResultVecTy,
                                  srcHalf1.getResult(), srsOp.getShift());

    // Calculate offset for second half store
    int64_t elementOffset = halfResultLanes;

    // Create indices for first half store (same as original)
    SmallVector<Value> firstHalfIndices(indices.begin(), indices.end());

    // Create indices for second half store
    SmallVector<Value> secondHalfIndices(indices.begin(), indices.end());

    // Adjust the last index by the element offset
    if (!indices.empty()) {
      Value lastIdx = indices.back();
      Value offsetVal =
          arith::ConstantIndexOp::create(rewriter, loc, elementOffset);
      Value newLastIdx =
          arith::AddIOp::create(rewriter, loc, lastIdx, offsetVal);
      secondHalfIndices.back() = newLastIdx;
    }

    // Create first half store
    vector::StoreOp::create(rewriter, loc, srsHalf0.getResult(), memRef,
                            firstHalfIndices);

    // Create second half store
    vector::StoreOp::create(rewriter, loc, srsHalf1.getResult(), memRef,
                            secondHalfIndices);

    // Erase the original store operation
    rewriter.eraseOp(storeOp);

    // The original SRS will be removed by dead code elimination
    // since it no longer has any uses

    return success();
  }
};

/// Splits a 1024-bit bf16 vector.load that only feeds the LHS of bf16 8x8x8
/// aievec.matmul_aie2p ops into two 512-bit loads joined by a vector.shuffle.
///
/// The matmul lowering widens the LHS to accfloat one 32-lane half at a time.
/// With a single 64-lane load, each half is extracted by a shuffle before it is
/// widened. After this split, and once LLVM folds the halves of the joining
/// shuffle back to the two loads, each widening reads its own 512-bit load,
/// which the backend can fuse into a single load-and-convert.
struct SplitVectorLoadMatMulLhsPattern
    : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    VectorType loadTy = loadOp.getVectorType();
    if (loadTy.getRank() != 1 || loadTy.getNumElements() != 64 ||
        !loadTy.getElementType().isBF16())
      return failure();

    auto bf16Tile = VectorType::get({8, 8}, rewriter.getBF16Type());
    auto f32Tile = VectorType::get({8, 8}, rewriter.getF32Type());
    auto feedsOnlyMatMulLhs = [&](Operation *user) {
      auto cast = dyn_cast<vector::ShapeCastOp>(user);
      if (!cast || cast.getResultVectorType() != bf16Tile)
        return false;
      return llvm::all_of(cast->getUses(), [&](OpOperand &use) {
        auto matmul = dyn_cast<MatMulOp_AIE2P>(use.getOwner());
        return matmul && use.getOperandNumber() == 0 &&
               matmul.getRhs().getType() == bf16Tile &&
               matmul.getAcc().getType() == f32Tile;
      });
    };
    if (loadOp->use_empty() ||
        !llvm::all_of(loadOp->getUsers(), feedsOnlyMatMulLhs))
      return failure();

    Location loc = loadOp.getLoc();
    auto halfTy = VectorType::get({32}, rewriter.getBF16Type());
    SmallVector<Value> hiIndices(loadOp.getIndices());
    Value c32 = arith::ConstantIndexOp::create(rewriter, loc, 32);
    hiIndices.back() =
        arith::AddIOp::create(rewriter, loc, hiIndices.back(), c32);
    Value lo = vector::LoadOp::create(rewriter, loc, halfTy, loadOp.getBase(),
                                      loadOp.getIndices());
    Value hi = vector::LoadOp::create(rewriter, loc, halfTy, loadOp.getBase(),
                                      hiIndices);
    SmallVector<int64_t> concatMask =
        llvm::to_vector(llvm::seq<int64_t>(0, 64));
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(loadOp, lo, hi, concatMask);
    return success();
  }
};

/// Pass to split vector.load + aievec.ups chains for better performance
struct SplitVectorLoadUpsChainsPass
    : public PassWrapper<SplitVectorLoadUpsChainsPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitVectorLoadUpsChainsPass)

  StringRef getArgument() const final { return "aievec-split-load-ups-chains"; }

  StringRef getDescription() const final {
    return "Split 1024-bit vector loads and stores whose halves are widened "
           "or narrowed separately";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, arith::ArithDialect,
                    memref::MemRefDialect, affine::AffineDialect,
                    xilinx::aievec::AIEVecDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns
        .add<SplitVectorLoadUpsChainPattern, SplitVectorSrsStoreChainPattern,
             SplitVectorLoadMatMulLhsPattern>(context);

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace xilinx {
namespace aievec {

std::unique_ptr<::mlir::Pass> createSplitVectorLoadUpsChainsPass() {
  return std::make_unique<SplitVectorLoadUpsChainsPass>();
}

} // namespace aievec
} // namespace xilinx
