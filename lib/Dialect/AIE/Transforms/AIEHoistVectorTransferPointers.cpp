//===- AIEHoistVectorTransferPointers.cpp -----------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass hoists vector transfer operations with IV-dependent pointers
// out of scf.for loops by using iter_args to track pointer updates. This
// optimization reduces address computation overhead in loops by maintaining
// a running pointer offset rather than recomputing addresses each iteration.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEHOISTVECTORTRANSFERPOINTERS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-hoist-vector-transfer-pointers"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// The coefficient of `iv` in the linear expression `expr`, given the
/// coefficient of each of the map's operands (dims first, then symbols).
/// nullopt if `expr` is not linear in `iv`.
static std::optional<int64_t> getExprCoefficient(AffineExpr expr,
                                                 ArrayRef<int64_t> operandCoefs,
                                                 unsigned numDims) {
  if (auto dim = dyn_cast<AffineDimExpr>(expr))
    return operandCoefs[dim.getPosition()];
  if (auto sym = dyn_cast<AffineSymbolExpr>(expr))
    return operandCoefs[numDims + sym.getPosition()];
  if (isa<AffineConstantExpr>(expr))
    return 0;
  auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin)
    return std::nullopt;
  std::optional<int64_t> lhs =
      getExprCoefficient(bin.getLHS(), operandCoefs, numDims);
  std::optional<int64_t> rhs =
      getExprCoefficient(bin.getRHS(), operandCoefs, numDims);
  if (!lhs || !rhs)
    return std::nullopt;
  switch (bin.getKind()) {
  case AffineExprKind::Add:
    return *lhs + *rhs;
  case AffineExprKind::Mul:
    // In an affine expression one side of a multiplication is a constant.
    if (auto c = dyn_cast<AffineConstantExpr>(bin.getRHS()))
      return *lhs * c.getValue();
    if (auto c = dyn_cast<AffineConstantExpr>(bin.getLHS()))
      return *rhs * c.getValue();
    return (*lhs == 0 && *rhs == 0) ? std::optional<int64_t>(0) : std::nullopt;
  default:
    // mod, floordiv and ceildiv are linear only if neither side depends on
    // the IV.
    return (*lhs == 0 && *rhs == 0) ? std::optional<int64_t>(0) : std::nullopt;
  }
}

/// Writes `v` as `coef * iv + inv`, where `inv` does not change across
/// iterations of `forOp` and can be recomputed before it, and returns `coef`.
/// nullopt when `v` is not of that form: it reads an iter_arg, loads from
/// memory, or depends on the IV non-linearly.
static std::optional<int64_t>
getIVCoefficient(Value v, scf::ForOp forOp,
                 DenseMap<Value, std::optional<int64_t>> &cache) {
  if (v == forOp.getInductionVar())
    return 1;
  if (forOp.isDefinedOutsideOfLoop(v))
    return 0;
  auto it = cache.find(v);
  if (it != cache.end())
    return it->second;

  std::optional<int64_t> result;
  Operation *def = v.getDefiningOp();
  // A block argument other than the IV (an iter_arg) changes every iteration.
  if (def && isPure(def)) {
    SmallVector<int64_t> coefs;
    bool linearOperands = true;
    for (Value operand : def->getOperands()) {
      std::optional<int64_t> c = getIVCoefficient(operand, forOp, cache);
      if (!c) {
        linearOperands = false;
        break;
      }
      coefs.push_back(*c);
    }
    if (linearOperands) {
      bool invariant = llvm::all_of(coefs, [](int64_t c) { return c == 0; });
      if (isa<arith::AddIOp>(def)) {
        result = coefs[0] + coefs[1];
      } else if (isa<arith::SubIOp>(def)) {
        result = coefs[0] - coefs[1];
      } else if (auto mul = dyn_cast<arith::MulIOp>(def)) {
        if (std::optional<int64_t> c = getConstantIntValue(mul.getRhs()))
          result = coefs[0] * *c;
        else if (std::optional<int64_t> c = getConstantIntValue(mul.getLhs()))
          result = coefs[1] * *c;
        else if (invariant)
          result = 0;
      } else if (auto apply = dyn_cast<affine::AffineApplyOp>(def)) {
        AffineMap map = apply.getAffineMap();
        result = getExprCoefficient(map.getResult(0), coefs, map.getNumDims());
      } else if (invariant) {
        // Any other pure op over loop-invariant operands is loop-invariant.
        result = 0;
      }
    }
  }
  cache[v] = result;
  return result;
}

/// Recomputes `v` (of the form getIVCoefficient accepts) before `forOp` with
/// the IV replaced by the loop's lower bound.
static Value cloneAtLowerBound(Value v, scf::ForOp forOp, OpBuilder &builder,
                               IRMapping &mapping) {
  if (v == forOp.getInductionVar())
    return forOp.getLowerBound();
  if (forOp.isDefinedOutsideOfLoop(v))
    return v;
  if (Value mapped = mapping.lookupOrNull(v))
    return mapped;
  Operation *def = v.getDefiningOp();
  for (Value operand : def->getOperands())
    mapping.map(operand, cloneAtLowerBound(operand, forOp, builder, mapping));
  builder.clone(*def, mapping);
  return mapping.lookup(v);
}

/// Get the total number of elements in a vector type
static int64_t getVectorNumElements(VectorType vectorType) {
  int64_t numElements = 1;
  for (int64_t dim : vectorType.getShape()) {
    numElements *= dim;
  }
  return numElements;
}

//===----------------------------------------------------------------------===//
// HoistVectorTransferPointers Pattern
//===----------------------------------------------------------------------===//

/// Information about a vector transfer operation
struct TransferOpInfo {
  Operation *op;
  Value base;
  MemRefType memrefType;
  VectorType vectorType;
  SmallVector<Value> indices;
  int64_t constantStride; // Total constant stride per iteration
  bool hasIVDependentIndices;
};

/// Pattern to hoist vector transfer operations with IV-dependent pointers
/// out of scf.for loops by using iter_args to track pointer updates
struct HoistVectorTransferPointersPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    Location loc = forOp.getLoc();

    // Collect all vector transfer operations with IV-dependent indices
    SmallVector<TransferOpInfo> transferOps;
    DenseMap<Value, std::optional<int64_t>> ivCoefs;

    for (Operation &op : forOp.getBody()->without_terminator()) {
      Value base;
      VectorType vectorType;
      SmallVector<Value> indices;

      if (auto readOp = dyn_cast<vector::TransferReadOp>(&op)) {
        base = readOp.getBase();
        vectorType = readOp.getVectorType();
        indices.assign(readOp.getIndices().begin(), readOp.getIndices().end());
      } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(&op)) {
        base = writeOp.getBase();
        vectorType = writeOp.getVectorType();
        indices.assign(writeOp.getIndices().begin(),
                       writeOp.getIndices().end());
      } else {
        continue;
      }

      auto memrefType = dyn_cast<MemRefType>(base.getType());
      if (!memrefType)
        continue;

      // The rewrite accesses the vector as one contiguous, unmasked, in-bounds
      // run of the flattened buffer. Skip transfers that are not of that form.
      auto xfer = cast<VectorTransferOpInterface>(op);
      if (xfer.getMask() || !xfer.getPermutationMap().isMinorIdentity() ||
          !llvm::all_of(xfer.getInBoundsValues(), [](bool b) { return b; }))
        continue;
      if (!memref::isStaticShapeAndContiguousRowMajor(memrefType))
        continue;
      // A vector<4x16> row block is contiguous only if each vector dim inside
      // its outermost non-unit one spans its whole memref dim (so
      // vector<1x1x8x8> of memref<4x8x8x8> is, vector<4x16> of memref<64x64>
      // is not).
      int64_t vecRank = vectorType.getRank();
      int64_t memRank = memrefType.getRank();
      if (vecRank > memRank)
        continue;
      int64_t outer = 0;
      while (outer < vecRank - 1 && vectorType.getDimSize(outer) == 1)
        ++outer;
      bool contiguous = true;
      for (int64_t i = outer + 1; i < vecRank; ++i)
        if (vectorType.getDimSize(i) !=
            memrefType.getDimSize(memRank - vecRank + i))
          contiguous = false;
      if (!contiguous)
        continue;
      // The flattened view is created before the loop.
      if (!forOp.isDefinedOutsideOfLoop(base))
        continue;

      // Each index must be coef * iv + (loop-invariant); the pointer then
      // advances by sum(coef * dimStride) * step elements per iteration.
      std::optional<int64_t> step =
          forOp.getConstantStep()
              ? std::optional<int64_t>(forOp.getConstantStep()->getSExtValue())
              : std::nullopt;
      bool analyzable = true;
      int64_t elementsPerIV = 0;
      int64_t dimStride = 1;
      for (int64_t d = memRank - 1; d >= 0; --d) {
        std::optional<int64_t> coef =
            getIVCoefficient(indices[d], forOp, ivCoefs);
        if (!coef) {
          analyzable = false;
          break;
        }
        elementsPerIV += *coef * dimStride;
        dimStride *= memrefType.getDimSize(d);
      }
      if (!analyzable)
        continue;
      bool hasIVDependentIndices = elementsPerIV != 0;
      if (hasIVDependentIndices && !step)
        continue;
      int64_t constantStride =
          hasIVDependentIndices ? elementsPerIV * *step : 0;

      transferOps.push_back({&op, base, memrefType, vectorType, indices,
                             constantStride, hasIVDependentIndices});
    }

    // If there are no transfer ops, don't modify
    if (transferOps.empty())
      return failure();

    // Prepare to add iter_args for each transfer operation with IV-dependent
    // indices
    SmallVector<Value> newInitArgs;
    SmallVector<Value> flatMemrefs;

    for (const auto &info : transferOps) {
      if (!info.hasIVDependentIndices)
        continue;

      // Flatten the memref if needed
      rewriter.setInsertionPoint(forOp);
      Value flatMemref = info.base;
      if (info.memrefType.getRank() > 1) {
        int64_t totalSize = 1;
        for (int64_t dim : info.memrefType.getShape()) {
          if (dim == ShapedType::kDynamic)
            return failure(); // Dynamic memref shapes not supported
          totalSize *= dim;
        }

        // Preserve strided layout if present
        MemRefType flatMemrefType;
        if (auto stridedLayout = dyn_cast_or_null<StridedLayoutAttr>(
                info.memrefType.getLayout())) {
          // The collapsed stride is the innermost stride (last element)
          int64_t collapsedStride = stridedLayout.getStrides().back();
          int64_t offset = stridedLayout.getOffset();

          auto newLayout = StridedLayoutAttr::get(rewriter.getContext(), offset,
                                                  {collapsedStride});
          flatMemrefType =
              MemRefType::get({totalSize}, info.memrefType.getElementType(),
                              newLayout, info.memrefType.getMemorySpace());
        } else {
          flatMemrefType =
              MemRefType::get({totalSize}, info.memrefType.getElementType(),
                              AffineMap(), info.memrefType.getMemorySpace());
        }

        SmallVector<ReassociationIndices> reassociation;
        ReassociationIndices allDims;
        for (size_t i = 0; i < static_cast<size_t>(info.memrefType.getRank());
             ++i) {
          allDims.push_back(i);
        }
        reassociation.push_back(allDims);

        flatMemref = memref::CollapseShapeOp::create(
            rewriter, loc, flatMemrefType, info.base, reassociation);
      }
      flatMemrefs.push_back(flatMemref);

      // Compute base pointer (with zeros for IV-dependent parts)
      int64_t rank = info.memrefType.getRank();
      AffineExpr linearExpr = rewriter.getAffineConstantExpr(0);
      int64_t stride = 1;
      for (int64_t i = rank - 1; i >= 0; --i) {
        linearExpr = linearExpr + rewriter.getAffineDimExpr(i) * stride;
        if (i > 0)
          stride *= info.memrefType.getShape()[i];
      }
      auto linearMap = AffineMap::get(rank, 0, linearExpr);

      // Initial pointer: every index evaluated at the lower bound.
      SmallVector<Value> evaluatedIndices;
      IRMapping indexMapping;
      for (Value idx : info.indices)
        evaluatedIndices.push_back(
            cloneAtLowerBound(idx, forOp, rewriter, indexMapping));

      Value basePointer = affine::AffineApplyOp::create(
          rewriter, loc, linearMap, evaluatedIndices);

      newInitArgs.push_back(basePointer);
    }

    // If there are no IV-dependent transfers, just process them to flatten
    // vectors
    if (newInitArgs.empty()) {
      // Check if any transfer needs flattening (avoid infinite rewrites)
      bool needsFlattening = false;
      bool hasProcessableTransfers = false;
      for (const auto &info : transferOps) {
        // Skip if base is defined inside the loop (e.g., a subview)
        // We can't hoist these
        if (info.base.getDefiningOp() &&
            forOp->isProperAncestor(info.base.getDefiningOp()))
          continue;

        hasProcessableTransfers = true;

        // Check if this transfer has already been flattened
        // (flattened transfers use 1D identity map)
        if (auto readOp = dyn_cast<vector::TransferReadOp>(info.op)) {
          if (readOp.getPermutationMap().getNumDims() != 1)
            needsFlattening = true;
        } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(info.op)) {
          if (writeOp.getPermutationMap().getNumDims() != 1)
            needsFlattening = true;
        }
      }

      // If there are no processable transfers (all bases defined in loop)
      // or nothing needs flattening, bail out
      if (!hasProcessableTransfers || !needsFlattening)
        return failure();

      // First, create flattened memrefs outside the loop for bases not defined
      // inside
      DenseMap<Value, Value> baseFlatMemrefs;
      rewriter.setInsertionPoint(forOp);
      for (const auto &info : transferOps) {
        if (baseFlatMemrefs.count(info.base))
          continue;

        // Skip if base is defined inside the loop (e.g., a subview)
        if (info.base.getDefiningOp() &&
            forOp->isProperAncestor(info.base.getDefiningOp()))
          continue;

        Value flatMemref = info.base;
        if (info.memrefType.getRank() > 1) {
          int64_t totalSize = 1;
          for (int64_t dim : info.memrefType.getShape()) {
            totalSize *= dim;
          }

          // Preserve strided layout if present
          MemRefType flatMemrefType;
          if (auto stridedLayout = dyn_cast_or_null<StridedLayoutAttr>(
                  info.memrefType.getLayout())) {
            int64_t collapsedStride = stridedLayout.getStrides().back();
            int64_t offset = stridedLayout.getOffset();

            auto newLayout = StridedLayoutAttr::get(rewriter.getContext(),
                                                    offset, {collapsedStride});
            flatMemrefType =
                MemRefType::get({totalSize}, info.memrefType.getElementType(),
                                newLayout, info.memrefType.getMemorySpace());
          } else {
            flatMemrefType =
                MemRefType::get({totalSize}, info.memrefType.getElementType(),
                                AffineMap(), info.memrefType.getMemorySpace());
          }

          SmallVector<ReassociationIndices> reassociation;
          ReassociationIndices allDims;
          for (size_t i = 0; i < static_cast<size_t>(info.memrefType.getRank());
               ++i) {
            allDims.push_back(i);
          }
          reassociation.push_back(allDims);
          flatMemref = memref::CollapseShapeOp::create(
              rewriter, loc, flatMemrefType, info.base, reassociation);
        }
        baseFlatMemrefs[info.base] = flatMemref;
      }

      // Process all transfers without using iter_args
      bool madeChanges = false;
      for (const auto &info : transferOps) {
        // Skip if base is defined inside the loop
        if (info.base.getDefiningOp() &&
            forOp->isProperAncestor(info.base.getDefiningOp()))
          continue;

        // Skip if we don't have a flattened version
        if (!baseFlatMemrefs.count(info.base))
          continue;

        rewriter.setInsertionPoint(info.op);

        // Flatten vector type
        int64_t numElements = getVectorNumElements(info.vectorType);
        VectorType flatVectorType =
            VectorType::get({numElements}, info.vectorType.getElementType());

        // Get the flattened memref
        Value flatMemref = baseFlatMemrefs[info.base];

        // Compute pointer from indices
        int64_t rank = info.memrefType.getRank();
        AffineExpr linearExpr = rewriter.getAffineConstantExpr(0);
        int64_t stride = 1;
        for (int64_t i = rank - 1; i >= 0; --i) {
          linearExpr = linearExpr + rewriter.getAffineDimExpr(i) * stride;
          if (i > 0)
            stride *= info.memrefType.getShape()[i];
        }
        auto linearMap = AffineMap::get(rank, 0, linearExpr);

        Value currentPointer = affine::AffineApplyOp::create(
            rewriter, loc, linearMap, info.indices);

        // Transform the transfer operation
        AffineMap identityMap1D = AffineMap::get(
            1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext());
        auto inBoundsAttr = rewriter.getBoolArrayAttr({true});

        if (auto readOp = dyn_cast<vector::TransferReadOp>(info.op)) {
          Value flatRead = vector::TransferReadOp::create(
              rewriter, loc, flatVectorType, flatMemref,
              ValueRange{currentPointer}, AffineMapAttr::get(identityMap1D),
              readOp.getPadding(),
              /*mask=*/Value(), inBoundsAttr);
          Value shapedRead = vector::ShapeCastOp::create(
              rewriter, loc, info.vectorType, flatRead);
          rewriter.replaceOp(readOp, shapedRead);
          madeChanges = true;
        } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(info.op)) {
          Value flatValue = vector::ShapeCastOp::create(
              rewriter, loc, flatVectorType, writeOp.getVector());
          rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
              writeOp, flatValue, flatMemref, ValueRange{currentPointer},
              AffineMapAttr::get(identityMap1D), /*mask=*/Value(),
              inBoundsAttr);
          madeChanges = true;
        }
      }
      return madeChanges ? success() : failure();
    }

    // Use replaceWithAdditionalYields to add pointer iter_args
    auto yieldValuesFn =
        [&](OpBuilder &b, Location yieldLoc,
            ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
      SmallVector<Value> yieldValues;

      // Process each transfer operation with IV-dependent indices
      size_t iterArgIdx = 0;
      for (const auto &info : transferOps) {
        if (!info.hasIVDependentIndices)
          continue;

        BlockArgument ptrIterArg =
            newBbArgs[newBbArgs.size() - newInitArgs.size() + iterArgIdx];
        Value flatMemref = flatMemrefs[iterArgIdx];

        // Flatten vector type
        int64_t numElements = getVectorNumElements(info.vectorType);
        VectorType flatVectorType =
            VectorType::get({numElements}, info.vectorType.getElementType());

        // Transform the transfer operation to use the iter_arg pointer
        b.setInsertionPoint(info.op);

        AffineMap identityMap1D =
            AffineMap::get(1, 0, b.getAffineDimExpr(0), b.getContext());
        auto inBoundsAttr = b.getBoolArrayAttr({true});

        if (auto readOp = dyn_cast<vector::TransferReadOp>(info.op)) {
          Value flatRead = vector::TransferReadOp::create(
              b, loc, flatVectorType, flatMemref, ValueRange{ptrIterArg},
              AffineMapAttr::get(identityMap1D), readOp.getPadding(),
              /*mask=*/Value(), inBoundsAttr);
          Value shapedRead =
              vector::ShapeCastOp::create(b, loc, info.vectorType, flatRead);
          rewriter.replaceOp(readOp, shapedRead);
        } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(info.op)) {
          Value flatValue = vector::ShapeCastOp::create(b, loc, flatVectorType,
                                                        writeOp.getVector());
          rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
              writeOp, flatValue, flatMemref, ValueRange{ptrIterArg},
              AffineMapAttr::get(identityMap1D), /*mask=*/Value(),
              inBoundsAttr);
        }

        // Compute next pointer value: current_ptr + constant_stride
        Value strideConst =
            arith::ConstantIndexOp::create(b, yieldLoc, info.constantStride);
        Value nextPtr =
            arith::AddIOp::create(b, yieldLoc, ptrIterArg, strideConst);
        yieldValues.push_back(nextPtr);

        iterArgIdx++;
      }

      return yieldValues;
    };

    // Create new loop with additional iter_args for pointers
    FailureOr<LoopLikeOpInterface> newLoopResult =
        cast<LoopLikeOpInterface>(forOp.getOperation())
            .replaceWithAdditionalYields(
                rewriter, newInitArgs, // new init operands (base pointers)
                true,                  // replace uses in loop
                yieldValuesFn);

    if (failed(newLoopResult))
      return failure();

    return success();
  }
};

//===----------------------------------------------------------------------===//
// AIEHoistVectorTransferPointersPass
//===----------------------------------------------------------------------===//

struct AIEHoistVectorTransferPointersPass
    : xilinx::AIE::impl::AIEHoistVectorTransferPointersBase<
          AIEHoistVectorTransferPointersPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<HoistVectorTransferPointersPattern>(context);

    // Apply patterns to the entire module - the pattern will only match scf.for
    // ops within aie.core regions
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
AIE::createAIEHoistVectorTransferPointersPass() {
  return std::make_unique<AIEHoistVectorTransferPointersPass>();
}
