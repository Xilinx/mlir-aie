//===- AIEDecomposeLargeDmaBd.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decomposes oversized non-contiguous aiex.npu.dma_memcpy_nd ops into one or
// more hardware-legal ND transfers before aie-dma-to-npu lowering.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/DmaDecomposition.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEDECOMPOSELARGEDMABD
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

static bool allConstant(NpuDmaMemcpyNdOp op) {
  return llvm::all_of(op.getMixedSizes(), [](OpFoldResult s) {
           return getConstantIntValue(s).has_value();
         }) &&
         llvm::all_of(op.getMixedStrides(), [](OpFoldResult s) {
           return getConstantIntValue(s).has_value();
         }) &&
         llvm::all_of(op.getMixedOffsets(), [](OpFoldResult s) {
           return getConstantIntValue(s).has_value();
         });
}

static NdDmaPattern patternFromOp(NpuDmaMemcpyNdOp op) {
  NdDmaPattern pattern;
  pattern.offsets = llvm::map_to_vector(llvm::reverse(op.getMixedOffsets()),
                                        [](OpFoldResult s) {
                                          return getConstantIntValue(s).value();
                                        });
  pattern.sizes = llvm::map_to_vector(llvm::reverse(op.getMixedSizes()),
                                      [](OpFoldResult s) {
                                        return getConstantIntValue(s).value();
                                      });
  pattern.strides = llvm::map_to_vector(llvm::reverse(op.getMixedStrides()),
                                        [](OpFoldResult s) {
                                          return getConstantIntValue(s).value();
                                        });
  return pattern;
}

static SmallVector<int64_t, 4> toOuter(ArrayRef<int64_t> inner) {
  return llvm::map_to_vector(llvm::reverse(inner),
                             [](int64_t v) { return v; });
}

static NpuDmaMemcpyNdOp
createDecomposedOp(PatternRewriter &rewriter, NpuDmaMemcpyNdOp op,
                   const NdDmaPattern &pattern, int64_t id, bool issueToken) {
  auto outerOffsets = toOuter(pattern.offsets);
  auto outerSizes = toOuter(pattern.sizes);
  auto outerStrides = toOuter(pattern.strides);

  return rewriter.create<NpuDmaMemcpyNdOp>(
      op.getLoc(), op.getMemref(),
      /*offsets=*/ValueRange{}, /*sizes=*/ValueRange{},
      /*strides=*/ValueRange{},
      DenseI64ArrayAttr::get(op.getContext(), outerOffsets),
      DenseI64ArrayAttr::get(op.getContext(), outerSizes),
      DenseI64ArrayAttr::get(op.getContext(), outerStrides), op.getPacketAttr(),
      op.getMetadata(), rewriter.getI64IntegerAttr(id),
      rewriter.getBoolAttr(issueToken), op.getD0ZeroBeforeAttr(),
      op.getD1ZeroBeforeAttr(), op.getD2ZeroBeforeAttr(),
      op.getD0ZeroAfterAttr(), op.getD1ZeroAfterAttr(), op.getD2ZeroAfterAttr(),
      op.getBurstLengthAttr(), op.getOffsetParameterAttr(),
      op.getOffsetStateTableIdxAttr());
}

static int64_t allocateNextId(NpuDmaMemcpyNdOp op, int64_t startId,
                              llvm::DenseSet<int64_t> &usedIds) {
  int64_t id = startId;
  while (usedIds.contains(id))
    ++id;
  usedIds.insert(id);
  return id;
}

struct DecomposeLargeDmaBdPattern : OpRewritePattern<NpuDmaMemcpyNdOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NpuDmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {
    if (!allConstant(op))
      return failure();

    NdDmaPattern pattern = patternFromOp(op);
    if (isContiguousTransfer(pattern.sizes, pattern.strides))
      return failure();

    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return failure();

    auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(
        dev, op.getMetadata().getRootReference());
    if (!allocOp)
      return failure();

    AIE::TileOp tile = allocOp.getTileOp();
    if (!tile)
      return failure();

    int col = tile.getCol();
    int row = tile.getRow();
    const AIE::AIETargetModel &targetModel = AIE::getTargetModel(op);
    auto bufferType = cast<BaseMemRefType>(op.getMemref().getType());

    if (isNdDmaPatternLegal(op, bufferType, targetModel, col, row, pattern))
      return failure();

    auto decomposed = decomposeNdDmaPattern(op, bufferType, pattern,
                                            targetModel, col, row);
    if (failed(decomposed) || decomposed->empty())
      return failure();

    if (decomposed->size() == 1) {
      rewriter.replaceOpWithNewOp<NpuDmaMemcpyNdOp>(
          op, op.getMemref(), ValueRange{}, ValueRange{}, ValueRange{},
          DenseI64ArrayAttr::get(op.getContext(),
                                 toOuter(decomposed->front().offsets)),
          DenseI64ArrayAttr::get(op.getContext(),
                                 toOuter(decomposed->front().sizes)),
          DenseI64ArrayAttr::get(op.getContext(),
                                 toOuter(decomposed->front().strides)),
          op.getPacketAttr(), op.getMetadata(), op.getIdAttr(),
          op.getIssueTokenAttr(), op.getD0ZeroBeforeAttr(),
          op.getD1ZeroBeforeAttr(), op.getD2ZeroBeforeAttr(),
          op.getD0ZeroAfterAttr(), op.getD1ZeroAfterAttr(),
          op.getD2ZeroAfterAttr(), op.getBurstLengthAttr(),
          op.getOffsetParameterAttr(), op.getOffsetStateTableIdxAttr());
      return success();
    }

    llvm::DenseSet<int64_t> usedIds;
    if (auto seq = op->getParentOfType<AIE::RuntimeSequenceOp>()) {
      seq.walk([&](NpuDmaMemcpyNdOp other) {
        if (other == op)
          return;
        if (other.getMetadata() == op.getMetadata())
          usedIds.insert(other.getId());
      });
    }

    int64_t nextId = op.getId();
    rewriter.setInsertionPoint(op);
    for (auto [idx, subPattern] : llvm::enumerate(*decomposed)) {
      bool last = idx + 1 == decomposed->size();
      int64_t id = allocateNextId(op, nextId, usedIds);
      nextId = id + 1;
      createDecomposedOp(rewriter, op, subPattern, id, last && op.getIssueToken());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEDecomposeLargeDmaBdPass
    : xilinx::AIEX::impl::AIEDecomposeLargeDmaBdBase<AIEDecomposeLargeDmaBdPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeLargeDmaBdPattern>(&getContext());
    if (failed(applyPatternsGreedily(device, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEDecomposeLargeDmaBdPass() {
  return std::make_unique<AIEDecomposeLargeDmaBdPass>();
}
