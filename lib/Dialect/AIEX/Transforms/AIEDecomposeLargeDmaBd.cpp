//===- AIEDecomposeLargeDmaBd.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Decomposes oversized non-contiguous aiex.npu.dma_memcpy_nd ops and task-path
// aie.dma_bd ops into one or more hardware-legal ND transfers before
// aie-dma-to-npu / aie-dma-tasks-to-npu lowering.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/Dialect/AIEX/Utils/DmaDecomposition.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

// Safety cap: avoid exploding a single BD into an unbounded chain.
static constexpr unsigned kMaxDecomposedSubBDs = 64;

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

static bool allConstant(AIE::DMABDOp op) {
  if (op.getPadDimensions().has_value())
    return false;
  if (op.getOffset() && !op.getConstantOffset())
    return false;
  if (op.getLen() && !op.getConstantLen())
    return false;
  if (op.getMixedSizes().empty())
    return false;
  return llvm::all_of(op.getMixedSizes(), [](OpFoldResult s) {
           return getConstantIntValue(s).has_value();
         }) &&
         llvm::all_of(op.getMixedStrides(), [](OpFoldResult s) {
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

static NdDmaPattern patternFromDmaBd(AIE::DMABDOp op) {
  SmallVector<int64_t, 4> outerSizes;
  SmallVector<int64_t, 4> outerStrides;
  for (OpFoldResult s : op.getMixedSizes())
    outerSizes.push_back(getConstantIntValue(s).value());
  for (OpFoldResult s : op.getMixedStrides())
    outerStrides.push_back(getConstantIntValue(s).value());
  while (outerSizes.size() < 4) {
    outerSizes.insert(outerSizes.begin(), 1);
    outerStrides.insert(outerStrides.begin(), 0);
  }

  NdDmaPattern pattern;
  pattern.offsets = {0, 0, 0, 0};
  pattern.sizes = llvm::map_to_vector(llvm::reverse(outerSizes),
                                      [](int64_t v) { return v; });
  pattern.strides = llvm::map_to_vector(llvm::reverse(outerStrides),
                                        [](int64_t v) { return v; });
  return pattern;
}

static SmallVector<int64_t, 4> toOuter(ArrayRef<int64_t> inner) {
  return llvm::map_to_vector(llvm::reverse(inner),
                             [](int64_t v) { return v; });
}

static int64_t flatOffsetFromPattern(int64_t baseFlatOffset,
                                     const NdDmaPattern &pattern) {
  int64_t flat = baseFlatOffset;
  for (unsigned k = 0; k < 4; ++k)
    flat += pattern.offsets[k] * pattern.strides[k];
  return flat;
}

static int64_t lenFromInnermost3(ArrayRef<int64_t> sizesInnermostFirst) {
  int64_t len = 1;
  for (unsigned i = 0; i < 3; ++i)
    len *= sizesInnermostFirst[i];
  return len;
}

static AIE::BDDimLayoutArrayAttr
outerDimsAttr(MLIRContext *ctx, ArrayRef<int64_t> outerSizes,
              ArrayRef<int64_t> outerStrides) {
  SmallVector<AIE::BDDimLayoutAttr> dims;
  dims.reserve(outerSizes.size());
  for (auto [s, t] : llvm::zip(outerSizes, outerStrides))
    dims.push_back(AIE::BDDimLayoutAttr::get(
        ctx, static_cast<uint32_t>(s), static_cast<uint32_t>(t)));
  return AIE::BDDimLayoutArrayAttr::get(ctx, dims);
}

static void updateTaskBdInPlace(AIE::DMABDOp bd, int32_t offset, int32_t len,
                                ArrayRef<int64_t> outerSizes,
                                ArrayRef<int64_t> outerStrides) {
  bd.getOffsetMutable().clear();
  bd.setStaticOffset(offset);
  bd.getLenMutable().clear();
  bd.setStaticLen(len);
  bd.getSizesMutable().clear();
  bd.getStridesMutable().clear();
  bd.setStaticSizes(DenseI64ArrayAttr::get(bd.getContext(), outerSizes));
  bd.setStaticStrides(DenseI64ArrayAttr::get(bd.getContext(), outerStrides));
}

static AIE::DMABDOp createTaskBd(PatternRewriter &rewriter, Location loc,
                                 AIE::DMABDOp tmpl, int32_t offset,
                                 int32_t len, ArrayRef<int64_t> outerSizes,
                                 ArrayRef<int64_t> outerStrides) {
  auto dims = outerDimsAttr(rewriter.getContext(), outerSizes, outerStrides);
  auto bd = AIE::DMABDOp::create(rewriter, loc, tmpl.getBuffer(), offset, len,
                                 dims);
  if (tmpl.getPacketAttr())
    bd.setPacketAttr(tmpl.getPacketAttr());
  if (tmpl.getBurstLengthAttr())
    bd.setBurstLengthAttr(tmpl.getBurstLengthAttr());
  if (tmpl.getOffsetParameterAttr())
    bd.setOffsetParameterAttr(tmpl.getOffsetParameterAttr());
  return bd;
}

static unsigned countTaskBds(Operation *taskOp) {
  unsigned n = 0;
  taskOp->walk([&](AIE::DMABDOp) { ++n; });
  return n;
}

static Region *getTaskBody(Operation *taskOp) {
  if (auto cfg = dyn_cast<DMAConfigureTaskOp>(taskOp))
    return &cfg.getBody();
  if (auto cfgFor = dyn_cast<DMAConfigureTaskForOp>(taskOp))
    return &cfgFor.getBody();
  return nullptr;
}

static bool isUnderRuntimeControlFlow(AIE::DMABDOp op) {
  auto seq = op->getParentOfType<AIE::RuntimeSequenceOp>();
  if (!seq)
    return false;
  for (Operation *parent = op->getParentOp(); parent && parent != seq;
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(parent))
      return true;
  }
  return false;
}

static std::optional<std::pair<AIE::TileOp, Operation *>>
resolveTaskAndTile(AIE::DMABDOp op) {
  if (auto cfg = op->getParentOfType<DMAConfigureTaskOp>()) {
    AIE::TileOp tile = cfg.getTileOp();
    if (!tile)
      return std::nullopt;
    return std::make_pair(tile, cfg.getOperation());
  }
  if (auto cfgFor = op->getParentOfType<DMAConfigureTaskForOp>()) {
    AIE::DeviceOp dev = op->getParentOfType<AIE::DeviceOp>();
    if (!dev)
      return std::nullopt;
    auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(
        dev, cfgFor.getAlloc().getRootReference());
    if (!allocOp)
      return std::nullopt;
    AIE::TileOp tile = allocOp.getTileOp();
    if (!tile)
      return std::nullopt;
    return std::make_pair(tile, cfgFor.getOperation());
  }
  return std::nullopt;
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
    if (decomposed->size() > kMaxDecomposedSubBDs)
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

struct DecomposeLargeDmaBdTaskPattern : OpRewritePattern<AIE::DMABDOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AIE::DMABDOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<AIE::MemOp>() ||
        op->getParentOfType<AIE::ShimDMAOp>() ||
        op->getParentOfType<AIE::MemTileDMAOp>() ||
        op->getParentOfType<AIE::DMAOp>())
      return failure();

    auto taskAndTile = resolveTaskAndTile(op);
    if (!taskAndTile)
      return failure();

    AIE::TileOp tile = taskAndTile->first;
    Operation *taskOp = taskAndTile->second;

    if (!allConstant(op))
      return failure();
    if (countTaskBds(taskOp) != 1)
      return failure();

    NdDmaPattern pattern = patternFromDmaBd(op);
    if (isContiguousTransfer(pattern.sizes, pattern.strides))
      return failure();

    int col = tile.getCol();
    int row = tile.getRow();
    const AIE::AIETargetModel &targetModel = AIE::getTargetModel(op);
    auto bufferType = cast<BaseMemRefType>(op.getBuffer().getType());

    if (isNdDmaPatternLegal(op, bufferType, targetModel, col, row, pattern))
      return failure();

    auto decomposed = decomposeNdDmaPattern(op, bufferType, pattern,
                                            targetModel, col, row);
    if (failed(decomposed) || decomposed->empty())
      return failure();
    if (decomposed->size() > kMaxDecomposedSubBDs)
      return failure();

    if (decomposed->size() > 1 && isUnderRuntimeControlFlow(op)) {
      op.emitRemark()
          << "deferring multi-BD decomposition under runtime control flow "
             "(dynamic BD pool supports single-BD tasks only)";
      return failure();
    }

    int64_t baseFlatOffset = op.getConstantOffset().value_or(0);

    if (decomposed->size() == 1) {
      const NdDmaPattern &sub = decomposed->front();
      int64_t flatOffset = flatOffsetFromPattern(baseFlatOffset, sub);
      auto outerSizes = toOuter(sub.sizes);
      auto outerStrides = toOuter(sub.strides);
      rewriter.modifyOpInPlace(op, [&]() {
        op.getOffsetMutable().clear();
        op.setStaticOffset(static_cast<int32_t>(flatOffset));
        op.getSizesMutable().clear();
        op.getStridesMutable().clear();
        op.setStaticSizes(DenseI64ArrayAttr::get(op.getContext(), outerSizes));
        op.setStaticStrides(
            DenseI64ArrayAttr::get(op.getContext(), outerStrides));
      });
      return success();
    }

    Region *body = getTaskBody(taskOp);
    if (!body || body->empty())
      return failure();

    SmallVector<Block *> blocks;
    blocks.push_back(op->getBlock());
    for (unsigned i = 1; i < decomposed->size(); ++i)
      blocks.push_back(rewriter.createBlock(body));

    for (auto [idx, subPattern] : llvm::enumerate(*decomposed)) {
      Block *block = blocks[idx];
      int64_t flatOffset = flatOffsetFromPattern(baseFlatOffset, subPattern);
      auto outerSizes = toOuter(subPattern.sizes);
      auto outerStrides = toOuter(subPattern.strides);
      int32_t len =
          static_cast<int32_t>(lenFromInnermost3(subPattern.sizes));

      if (idx == 0) {
        rewriter.modifyOpInPlace(op, [&]() {
          updateTaskBdInPlace(op, static_cast<int32_t>(flatOffset), len,
                              outerSizes, outerStrides);
        });
        Operation *oldTerm = block->getTerminator();
        rewriter.setInsertionPoint(oldTerm);
        if (idx + 1 < decomposed->size())
          AIE::NextBDOp::create(rewriter, op.getLoc(), blocks[idx + 1]);
        else
          AIE::EndOp::create(rewriter, op.getLoc());
        rewriter.eraseOp(oldTerm);
      } else {
        rewriter.setInsertionPointToStart(block);
        createTaskBd(rewriter, op.getLoc(), op,
                     static_cast<int32_t>(flatOffset), len, outerSizes,
                     outerStrides);
        rewriter.setInsertionPointToEnd(block);
        if (idx + 1 < decomposed->size())
          AIE::NextBDOp::create(rewriter, op.getLoc(), blocks[idx + 1]);
        else
          AIE::EndOp::create(rewriter, op.getLoc());
      }
    }

    return success();
  }
};

struct AIEDecomposeLargeDmaBdPass
    : xilinx::AIEX::impl::AIEDecomposeLargeDmaBdBase<AIEDecomposeLargeDmaBdPass> {
  void runOnOperation() override {
    AIE::DeviceOp device = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<DecomposeLargeDmaBdPattern, DecomposeLargeDmaBdTaskPattern>(
        &getContext());
    if (failed(applyPatternsGreedily(device, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEDecomposeLargeDmaBdPass() {
  return std::make_unique<AIEDecomposeLargeDmaBdPass>();
}
