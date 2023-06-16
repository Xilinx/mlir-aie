#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h" // ???
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"         // ???
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"                   // ???
#include "mlir/Dialect/SCF/IR/SCF.h"                         // ???
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h" // ???
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "VectorToVectorConversions.h"

#define DEBUG_TYPE "aievec-canonicalization"

using namespace mlir;
using namespace arith;
using namespace vector;
using namespace xilinx;
using namespace xilinx::aievec;

//============================================================================//
//================== Common AIE canonicalization analysis ====================//
//============================================================================//

// Return the offset of a given transfer read operation with regards to the
// specified vector type. If the read is aligned size of the vector type, then
// the offset is 0. Otherwise, the offset is the number of elements past the
// immediately preceding aligned address.
template <
    typename TransferReadLikeOp,
    typename = std::enable_if_t<
        std::is_same_v<TransferReadLikeOp, vector::TransferReadOp> ||
        std::is_same_v<TransferReadLikeOp, vector::TransferReadOp::Adaptor>>>
static unsigned getTransferReadAlignmentOffset(TransferReadLikeOp op,
                                               VectorType vType) {
  // TODO: Add support for cases where the index is not comming from an
  // TODO: `affine.apply` op. E.g.: when the index is a constant.
  auto innerMostIndex = op.getIndices().back();
  auto vectorLength = vType.getShape().back();
  if (auto defOp = innerMostIndex.getDefiningOp())
    if (auto applyOp = dyn_cast<AffineApplyOp>(defOp))
      if (applyOp.getAffineMap().getNumDims() == 1)
        return applyOp.getAffineMap().compose(ArrayRef<int64_t>{0})[0] %
               vectorLength;
  return 0;
}

//============================================================================//
//============ Common AIE canonicalization conversion patterns ===============//
//============================================================================//

// This pattern converts a `vector.transfer_read` with an unaligned access
// into an aligned `vector.transfer_read` twice as long, followed by a
// `vector.extract_strided_slice` selecting the subvector matching the
// original `vector.transfer_read`.
struct SplitUnalignedTransferReadPattern
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check that it's not a splat transfer read.
    if (adaptor.getPermutationMap().isConstant())
      return failure();

    // Check if the transfer is unaligned.
    auto vType = readOp.getVectorType();
    unsigned offset = getTransferReadAlignmentOffset(adaptor, vType);
    if (offset == 0)
      return failure();

    // Create an aligned transfer read

    // Calculate the aligned indices for the lower and higher parts.
    // TODO: Add support for cases where the offset is greater than the
    // TODO: vector length.
    auto lowIdx =
        dyn_cast<AffineApplyOp>(adaptor.getIndices().back().getDefiningOp())
            .getMapOperands()[0];
    auto vLen = vType.getShape().back();
    auto longVecTy = VectorType::get(2 * vLen, vType.getElementType());
    SmallVector<Value, 8> alignedIdx;
    alignedIdx.append(adaptor.getIndices().begin(), adaptor.getIndices().end());
    alignedIdx[alignedIdx.size() - 1] = lowIdx;

    // Create the aligned transfer read for a vector 2x as long that covers the
    // elements of the unaligned vector.
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), longVecTy, adaptor.getSource(), alignedIdx,
        adaptor.getPadding());

    // Create a `vector.extract_strided_slice` to extract the unaligned vector.
    rewriter.replaceOpWithNewOp<vector::ExtractStridedSliceOp>(
        readOp, newReadOp.getResult(), offset, vLen, 1);

    return success();
  }
};

// This pattern converts a `vector.transfer_read` with a splat permutation map
// into a contiguous `vector.transfer_read` followed by a `vector.extract` to
// obtain the splat value and a `vector.broadcast` to broadcast it into a
// vector of the right size.
struct ConvertSplatTransferReadToBroadcastPattern
    : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AffineMap map = readOp.getPermutationMap();
    if (!map.isConstant())
      return failure();

    // If the innermost index comes from an `affine.apply` op, take the base
    // as the new innermost index for the new `vector.transfer_read`, and the
    // offset as the index for the `aievec.broadcast` op.
    SmallVector<Value, 8> indices;
    indices.append(adaptor.getIndices().begin(), adaptor.getIndices().end());
    Value innerMostIdx = indices[indices.size() - 1];
    Value newIdx = innerMostIdx;
    int64_t offset = 0;
    if (auto defOp = innerMostIdx.getDefiningOp())
      if (auto applyOp = dyn_cast<AffineApplyOp>(defOp))
        if (applyOp.getAffineMap().getNumDims() == 1) {
          newIdx = applyOp.getMapOperands()[0];
          offset = applyOp.getAffineMap().compose(ArrayRef<int64_t>{0})[0];
        }
    // XXX: We assume we are reading 1D vectors
    int64_t vlen = readOp.getVector().getType().getShape()[0];
    if (offset >= vlen) {
      // If the splat element is beyond the first vector, we calculate the
      // address of the vector containing the element.
      int64_t numElemsToSkip = vlen * (offset / vlen);
      offset = offset % vlen;
      auto newAddrMap = AffineMap::get(
          1, 0, getAffineDimExpr(0, readOp.getContext()) + numElemsToSkip);
      newIdx = rewriter
                   .create<AffineApplyOp>(readOp.getLoc(), newAddrMap,
                                          SmallVector<Value, 1>({newIdx}))
                   .getResult();
    }
    indices[indices.size() - 1] = newIdx;
    auto newReadOp = rewriter.create<vector::TransferReadOp>(
        readOp.getLoc(), readOp.getVector().getType(), adaptor.getSource(),
        indices, adaptor.getPadding());
    auto extractOp = rewriter.create<vector::ExtractOp>(
        readOp.getLoc(), newReadOp.getResult(), ArrayRef<int64_t>{offset});
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        readOp, newReadOp.getVector().getType(), extractOp.getResult());
    return success();
  }
};

//============================================================================//
//================ Common AIE canonicalization configuration =================//
//============================================================================//

static void
configureCommonAIECanonicalizeLegalizations(ConversionTarget &target) {
  target.addLegalDialect<vector::VectorDialect>();
  target.addLegalDialect<AffineDialect>();
  target.addLegalDialect<aievec::AIEVecDialect>();
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant();
      });
}

static void
populateCommonAIECanonicalizeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertSplatTransferReadToBroadcastPattern>(
      patterns.getContext());
}

//============================================================================//
//============== AIEv1-specific canonicalization configuration ===============//
//============================================================================//

static void configureAIEv1CanonicalizeLegalizations(ConversionTarget &target) {
  target.addLegalDialect<arith::ArithDialect>();
  target.addDynamicallyLegalOp<vector::TransferReadOp>(
      [](vector::TransferReadOp op) {
        return !op.getPermutationMap().isConstant() &&
               getTransferReadAlignmentOffset(op, op.getVectorType()) == 0;
      });
}

static void
populateAIEv1CanonicalizeConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<SplitUnalignedTransferReadPattern>(patterns.getContext());
}

//============================================================================//
//============== AIEML-specific canonicalization configuration ===============//
//============================================================================//

static void configureAIEMLCanonicalizeLegalizations(ConversionTarget &target) {}

static void
populateAIEMLCanonicalizeConversionPatterns(RewritePatternSet &patterns) {}

//============================================================================//
//=================== Common AIE Canonicalization Passes =====================//
//============================================================================//

// This pass converts standard vector ops into a subset of `Vector` ops more
// amenable to being converted to `AIEVec`. So far, this process consists of
// two steps:
//    1) Replace splat transfer reads with contiguous transfer reads followed
//       by `extract` + `broadcast` operations.
//    2) Split unaligned transfer reads into a wider aligned transfer read
//       followed by a `vector.extract_strided_slice` operation.
struct CanonicalizeVectorForAIEVecPass
    : public PassWrapper<CanonicalizeVectorForAIEVecPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeVectorForAIEVecPass)

  CanonicalizeVectorForAIEVecPass() = default;
  CanonicalizeVectorForAIEVecPass(const CanonicalizeVectorForAIEVecPass &pass)
      : PassWrapper(pass) {}

  CanonicalizeVectorForAIEVecPass(
      const CanonicalizeVectorForAIEVecOptions &options)
      : CanonicalizeVectorForAIEVecPass() {
    aieTarget = options.aieTarget;
  }

  // In case we want to register this pass as a standalone pass for test
  // purposes.
  StringRef getArgument() const final {
    return "test-canonicalize-vector-for-aievec";
  }
  StringRef getDescription() const final {
    return "Canonicalize vector operations for AIEVec conversion";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, vector::VectorDialect>();
  }

  Option<std::string> aieTarget{
      *this, "aie-target",
      llvm::cl::desc("Select AIE version: \"aie\" or \"aieml\". This will "
                     "determine the vector size and available operations."),
      llvm::cl::init("aie")};

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    AIEArch aieVersion = AIEArch::AIE;
    if (!aieTarget.empty()) {
      std::string target = aieTarget;
      if (target == "aieml") {
        aieVersion = AIEArch::AIE_ML;
      } else if (target != "aie") {
        funcOp.emitError() << "unknown AIE target '" << aieTarget << "'";
        signalPassFailure();
        return;
      }
    }

    populateCommonAIECanonicalizeConversionPatterns(patterns);
    configureCommonAIECanonicalizeLegalizations(target);
    if (aieVersion == AIEArch::AIE) {
      populateAIEv1CanonicalizeConversionPatterns(patterns);
      configureAIEv1CanonicalizeLegalizations(target);
    } else {
      populateAIEMLCanonicalizeConversionPatterns(patterns);
      configureAIEMLCanonicalizeLegalizations(target);
    }

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

static std::unique_ptr<::mlir::Pass> createCanonicalizeVectorForAIEVecPass(
    const CanonicalizeVectorForAIEVecOptions &options) {
  return std::make_unique<CanonicalizeVectorForAIEVecPass>(options);
}

//============================================================================//
//=============== Main Vector2Vector Pipeline Configuration ==================//
//============================================================================//

void xilinx::aievec::buildCanonicalizeVectorForAIEVec(
    OpPassManager &pm, const CanonicalizeVectorForAIEVecOptions &options) {
  // Add `Vector` code canonicalization passes
  // TODO: Add passes to unroll vector with unsupported types
  // TODO: Add passes to split vectors that won't fit in registers
  pm.addPass(createCanonicalizeVectorForAIEVecPass(options));
}
