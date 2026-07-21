//===- AIEXDialect.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/TypeSize.h"

#include <cstdint>
#include <numeric>

using namespace mlir;
using namespace xilinx;

#include "aie/Dialect/AIEX/IR/AIEXDialect.cpp.inc"

#include "aie/Dialect/AIEX/IR/AIEXEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/AIEX/IR/AIEXTypes.cpp.inc"

namespace xilinx::AIEX {

// FIXME: use Tablegen'd dialect class
void AIEXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/AIEX/IR/AIEX.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "aie/Dialect/AIEX/IR/AIEXTypes.cpp.inc"
      >();
}

} // namespace xilinx::AIEX

#define GET_OP_CLASSES
#include "aie/Dialect/AIEX/IR/AIEX.cpp.inc"

/* Return the correct values to write to the hardware registers to configure
  strides and wraps given the input user-facing strides and wraps.

  In the IR, we express strides in units of element data type, but the hardware
  requires it in units of address granularity. Address granularity currently is
  4 bytes for all hardware.


  User-facing strides/wraps relate to hardware as follows:

   - By default, stride 0 and size 1 is assumed if unspecified.
   - If only N strides/wraps are defined, those define the lowest N dimensions.

  inputStride[3]        == iteration_stride / elemSizeFac + 1
  inputWrap[3]          == iteration_size + 1
    Highest-dimension stride/wrap is iteration count / iteration stride.
  inputStride[2]        == d2_stride / elemSizeFac + 1
                           Note: d2_size is not specified in hardware as it is
                           implicit from the total buffer transfer length
  inputStride[1]        == d1_stride / elemSizeFac + 1
  inputSize[1]          == d1_size
  inputStride[0]        == d0_stride / elemSizeFac + 1
  inputSize[0]          == d0_size / elemSizeFac

  where elemSizeFac == bufferElementSize / addressGranularity
  where bufferElementSize == size in bytes of elements in buffer,
                             e.g. 4 for int32
  where addressGranularity == transfer granularity in hardware, which is
                              4 bytes for all current hardware

  Note: strides are expressed offset by one from user input strides, because the
  hardware does not support a 0 stride (repeat).
  */
void AIEX::getHardwareStridesWraps(const AIE::AIETargetModel &targetModel,
                                   mlir::Operation *op,
                                   mlir::BaseMemRefType referencedBufType,
                                   llvm::SmallVector<int64_t, 4> inputSizes,
                                   llvm::SmallVector<int64_t, 4> inputStrides,
                                   llvm::SmallVector<int64_t, 4> &sizes,
                                   llvm::SmallVector<int64_t, 4> &strides) {
  assert(inputSizes.size() == inputStrides.size());
  assert(sizes.size() == 4);
  assert(strides.size() == 4);

  DataLayout dataLayout = DataLayout::closest(op);
  auto elemWidth =
      dataLayout.getTypeSizeInBits(referencedBufType.getElementType());
  auto addressGranularity = targetModel.getAddressGenGranularity();

  // Output strides and sizes are default-initialized to 0
  std::fill(sizes.begin(), sizes.end(), 0);
  std::fill(strides.begin(), strides.end(), 0);

  if (inputSizes[0] == 0) {
    // Illegal input, this won't transfer anything at all.
    // Leave it to the verification functions to complain to the user.
    return;
  }

  ConstStridePolicy policy;
  int64_t inS[4] = {inputSizes[0], inputSizes[1], inputSizes[2], inputSizes[3]};
  int64_t inT[4] = {inputStrides[0], inputStrides[1], inputStrides[2],
                    inputStrides[3]};
  int64_t outS[4], outT[4];
  encodeHardwareStridesWraps(policy, elemWidth, addressGranularity, inS, inT,
                             outS, outT);
  for (int i = 0; i < 4; i++) {
    sizes[i] = outS[i];
    strides[i] = outT[i];
  }
}

mlir::LogicalResult
AIEX::verifyStridesWraps(mlir::Operation *forOp,
                         mlir::BaseMemRefType referencedBufType, int tileCol,
                         int tileRow, llvm::SmallVector<int64_t, 4> inputSizes,
                         llvm::SmallVector<int64_t, 4> inputStrides,
                         llvm::SmallVector<int64_t, 4> hardwareSizes,
                         llvm::SmallVector<int64_t, 4> hardwareStrides,
                         bool skipTransformationChecks) {
  const auto &targetModel = AIE::getTargetModel(forOp);
  auto addressGranularity = targetModel.getAddressGenGranularity();
  DataLayout dataLayout = DataLayout::closest(forOp);
  auto elemWidth =
      dataLayout.getTypeSizeInBits(referencedBufType.getElementType());

  uint32_t wrap_bits = 0;
  uint32_t step_bits = 0;
  uint32_t iter_bits = 6;
  if (targetModel.isShimNOCTile(tileCol, tileRow)) {
    step_bits = 20; // XAIEMLGBL_NOC_MODULE_DMA_BD0_3_D0_STEPSIZE_WIDTH
    wrap_bits = 10; // XAIEMLGBL_NOC_MODULE_DMA_BD0_3_D0_WRAP_WIDTH
  } else if (targetModel.isMemTile(tileCol, tileRow)) {
    step_bits = 17; // XAIEMLGBL_MEM_TILE_MODULE_DMA_BD0_2_D0_STEPSIZE_WIDTH
    wrap_bits = 10; // XAIEMLGBL_MEM_TILE_MODULE_DMA_BD0_2_D0_WRAP_WIDTH
  } else if (targetModel.isCoreTile(tileCol, tileRow)) {
    step_bits = 13; // XAIEMLGBL_MEMORY_MODULE_DMA_BD0_2_D0_STEPSIZE_WIDTH
    wrap_bits = 8;  // XAIEMLGBL_MEMORY_MODULE_DMA_BD0_3_D0_WRAP_WIDTH
  } else {
    return forOp->emitOpError(
        "Unsupported tile type at (" + std::to_string(tileCol) + ", " +
        std::to_string(tileRow) + ") Must be ShimNOC, Mem or Core.");
  }

  for (int i = 0; i < 4; i++) {
    if (inputSizes[i] <= 0) {
      return forOp->emitOpError("Size ") << i << " must be a positive integer.";
    }
  }

  if (!isConstMultipleOfGranule(inputSizes[0], elemWidth, addressGranularity)) {
    std::stringstream msg;
    msg << "Transfer sizes must be multiples of " << (addressGranularity / 8)
        << " bytes. " << inputSizes[0] << " elements at " << (elemWidth / 8)
        << " bytes each equal " << (inputSizes[0] * elemWidth / 8)
        << " bytes, which is not divisible by " << (addressGranularity / 8)
        << ". ";
    return forOp->emitOpError(msg.str());
  }

  for (int i = 0; i < 3; i++) {
    if (inputSizes[i] > 1 && inputStrides[i] < 1) {
      // If inputSize[i] == 1, anything is allowable in the stride, since that
      // stride will never be applied. For any larger size, we must verify that
      // the stride is positive.
      return forOp->emitOpError("Stride ")
             << i << " must be a positive integer.";
    }
  }
  // A value of zero is allowable for the fourth-dimension stride
  // (this indicates an interation stride for the repeat of 0)
  if (inputSizes[3] > 1 && inputStrides[3] < 0) {
    return forOp->emitOpError("Stride 3 must be a non-negative integer.");
  }

  for (int i = 0; i < 4; i++) {
    // strides[0] == 1 is ok iff the transfer size is a multiple of
    // addressGranularity, which is checked below
    if (i == 0 && inputStrides[i] == 1)
      continue;
    if (!isConstMultipleOfGranule(inputStrides[i], elemWidth,
                                  addressGranularity)) {
      std::stringstream msg;
      msg << "Stride " << i << " is " << inputStrides[i] << " elements * "
          << (elemWidth / 8) << " bytes = " << (inputStrides[i] * elemWidth / 8)
          << " bytes, which is not divisible by " << (addressGranularity / 8)
          << ". ";
      return forOp->emitOpError(msg.str());
    }
  }

  if (!skipTransformationChecks && hardwareSizes[0] > (1 << wrap_bits) - 1)
    return forOp->emitOpError(
        "Size 0 exceeds the [0:" + std::to_string((1 << wrap_bits) - 1) +
        "] range.");
  if (!skipTransformationChecks && hardwareSizes[1] > (1 << wrap_bits) - 1)
    return forOp->emitOpError(
        "Size 1 exceeds the [0:" + std::to_string((1 << wrap_bits) - 1) +
        "] range.");
  if (hardwareSizes[3] > (1 << iter_bits))
    return forOp->emitOpError(
        "Size 3 exceeds the [1:" + std::to_string(1 << iter_bits) + "] range.");
  if (hardwareStrides[0] > (1 << step_bits))
    return forOp->emitOpError("Stride 0 exceeds the [1:" +
                              std::to_string(1 << step_bits) + "] range.");
  if (hardwareStrides[1] > (1 << step_bits))
    return forOp->emitOpError("Stride 1 exceeds the [1:" +
                              std::to_string(1 << step_bits) + "] range.");
  if (hardwareStrides[2] > (1 << step_bits))
    return forOp->emitOpError("Stride 2 exceeds the [1:" +
                              std::to_string(1 << step_bits) + "] range.");
  // strides[3] exceeding the range is ok iff the sizes[3] is one, which is
  // checked below
  if (hardwareStrides[3] > (1 << step_bits) && hardwareSizes[3] > 0)
    return forOp->emitOpError("Stride 3 exceeds the [1:" +
                              std::to_string(1 << step_bits) + "] range.");

  return success();
}

//===----------------------------------------------------------------------===//
// UseTokenOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::UseTokenOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  if (isa<func::FuncOp>(parentOp) || isa<AIE::CoreOp>(parentOp) ||
      isa<AIE::MemOp>(parentOp) || isa<AIE::ShimDMAOp>(parentOp))
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// MulticastOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::MulticastOp::verify() {
  Region &body = getPorts();
  assert(getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front())
    if (!isa<MultiDestOp, AIE::EndOp>(ops))
      return ops.emitOpError("cannot be contained in a Multicast op");

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastPacketOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::BroadcastPacketOp::verify() {
  Region &body = getPorts();
  assert(getOperation()->getNumRegions());
  assert(!body.empty());
  for (auto &ops : body.front())
    if (!isa<BPIDOp, AIE::EndOp>(ops))
      return ops.emitOpError("cannot be contained in a BroadcastPacket op");

  return success();
}

//===----------------------------------------------------------------------===//
// NpuDmaMemcpyNdOp
//===----------------------------------------------------------------------===//

/* Calculates the offset value to be written to the
 */
int64_t AIEX::NpuDmaMemcpyNdOp::getOffsetInBytes() {
  llvm::SmallVector<int64_t, 4> offsets =
      llvm::map_to_vector(llvm::reverse(getMixedOffsets()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  auto strides = llvm::to_vector<4>(llvm::reverse(getMixedStrides()));
  size_t offset = 0;
  size_t R = offsets.size();
  size_t el_bit_width = getElementTypeBitwidth();
  assert(el_bit_width % 8 == 0 &&
         "Expected Memref element bitwidth to be multiple of 8.");
  size_t S = el_bit_width / 8;
  // A dimension only contributes to the byte offset when its (constant) offset
  // is non-zero; a runtime stride paired with a zero offset is fine and must
  // not be forced to a constant. The verifier requires any stride multiplied by
  // a non-zero offset to be constant.
  for (size_t i = 0; i < R; i++) {
    if (offsets[i] == 0)
      continue;
    offset += offsets[i] * getConstantIntValue(strides[i]).value() * S;
  }
  return offset;
}

// Returns true when sizes/strides describe a plain contiguous transfer with
// no data layout transformation (d1/d2 sizes == 1, d0 stride == 1).
// d3 (repeat) is intentionally excluded.
bool AIEX::isLinearTransfer(llvm::ArrayRef<int64_t> sizes,
                            llvm::ArrayRef<int64_t> strides) {
  return sizes[1] == 1 && sizes[2] == 1 && strides[0] == 1 && strides[1] == 0 &&
         strides[2] == 0;
}

// Returns true when sizes/strides (innermost-first) describe a contiguous
// row-major scan: innermost stride == 1 and each outer stride equals the
// product of all inner sizes.  The repeat dimension (index 3) is excluded.
// Size-1 dimensions are allowed to carry any stride value because that stride
// is never applied during the transfer (the loop runs only once).
// This is the vector-form counterpart of AIE::isContiguousBDTransfer.
bool AIEX::isContiguousTransfer(llvm::ArrayRef<int64_t> sizes,
                                llvm::ArrayRef<int64_t> strides) {
  if (strides[0] != 1)
    return false;
  if (sizes[1] > 1 && strides[1] != sizes[0])
    return false;
  if (sizes[2] > 1 && strides[2] != sizes[0] * sizes[1])
    return false;
  return true;
}

// dma_memcpy_nd transfers of the form [*, 1, 1, len][*, 0, 0, 1] do not
// specify any data layout transformation, but simply express a contiguous
// transfer of `len`. The 4th dimension is excluded because a repeat count
// is still compatible with a linear transfer.
bool AIEX::NpuDmaMemcpyNdOp::isLinearTransferWithoutTransformation() {
  llvm::SmallVector<int64_t, 4> inputSizes =
      llvm::map_to_vector(llvm::reverse(getMixedSizes()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  llvm::SmallVector<int64_t, 4> inputStrides =
      llvm::map_to_vector(llvm::reverse(getMixedStrides()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  return isLinearTransfer(inputSizes, inputStrides);
}

// Canonicalization pattern: rewrite a contiguous row-major access pattern to
// the canonical linear form [s3, 1, 1, N][st3, 0, 0, 1].
//
// Using outermost-first index notation (matching the IR syntax), a 4D access
// [s3, s2, s1, s0][st3, st2, st1, st0] is a contiguous linear scan when:
//   st0 == 1
//   s1 == 1  ||  st1 == s0          (stride irrelevant when size is 1)
//   s2 == 1  ||  st2 == s0 * s1
// yielding a total of N = s0 * s1 * s2 contiguous elements.  The repeat
// dimension s3 / stride st3 is unchanged by the fold.
//
// Note: this pattern applies only to NpuDmaMemcpyNdOp.  The analogous
// dimensionsToStream / dimensionsFromStreamPerConsumer attributes on
// ObjectFifoCreateOp are not canonicalized here; they are lowered separately
// by the ObjectFifo stateful transform pass.
//
// This fold is always semantically valid and never introduces new hardware
// limit violations: in the resulting linear form, isLinearTransferWithout-
// Transformation() returns true, so verifyStridesWraps() skips the 10-bit
// d0 wrap-size check.  The hardware uses a wider buffer_length register in
// linear mode (32-bit on shim tiles, 17-bit on mem tiles, 14-bit on core
// tiles), so N can be much larger than the 10-bit ND wrap limit.
namespace {
struct LinearizeContiguousTransfer
    : public mlir::OpRewritePattern<AIEX::NpuDmaMemcpyNdOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AIEX::NpuDmaMemcpyNdOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Only constant sizes/strides/offsets can be analysed statically.
    if (!llvm::all_of(op.getMixedSizes(), [](mlir::OpFoldResult s) {
          return mlir::getConstantIntValue(s).has_value();
        }))
      return mlir::failure();
    if (!llvm::all_of(op.getMixedStrides(), [](mlir::OpFoldResult s) {
          return mlir::getConstantIntValue(s).has_value();
        }))
      return mlir::failure();
    if (!llvm::all_of(op.getMixedOffsets(), [](mlir::OpFoldResult s) {
          return mlir::getConstantIntValue(s).has_value();
        }))
      return mlir::failure();

    // Skip ops that are already in canonical linear form.
    if (op.isLinearTransferWithoutTransformation())
      return mlir::failure();

    // getMixedSizes/Strides/Offsets return outermost-first; reverse to
    // innermost-first so index 0 = d0 (innermost) and index 3 = repeat.
    llvm::SmallVector<int64_t, 4> sizes = llvm::map_to_vector(
        llvm::reverse(op.getMixedSizes()), [](mlir::OpFoldResult s) {
          return mlir::getConstantIntValue(s).value();
        });
    llvm::SmallVector<int64_t, 4> strides = llvm::map_to_vector(
        llvm::reverse(op.getMixedStrides()), [](mlir::OpFoldResult s) {
          return mlir::getConstantIntValue(s).value();
        });
    llvm::SmallVector<int64_t, 4> offsets = llvm::map_to_vector(
        llvm::reverse(op.getMixedOffsets()), [](mlir::OpFoldResult s) {
          return mlir::getConstantIntValue(s).value();
        });

    // Require a contiguous row-major scan.
    if (!AIEX::isContiguousTransfer(sizes, strides))
      return mlir::failure();

    // Fold d0/d1/d2 into one linear count; keep the repeat dimension intact.
    // Build directly in outermost-first order for the replacement op.
    int64_t N = sizes[0] * sizes[1] * sizes[2];
    llvm::SmallVector<int64_t, 4> newSizesOuter = {sizes[3], 1, 1, N};
    llvm::SmallVector<int64_t, 4> newStridesOuter = {strides[3], 0, 0, 1};

    // getOffsetInBytes() computes: sum(offsets[i] * strides[i] * elemSize).
    // After folding, the intermediate strides become 0, so any non-zero offset
    // in those dimensions would silently contribute 0 bytes.  Preserve the
    // correct start address by collapsing the innermost three offset/stride
    // pairs into a single linear element index at d0.
    int64_t linearOffset = offsets[0] * strides[0] + offsets[1] * strides[1] +
                           offsets[2] * strides[2];
    llvm::SmallVector<int64_t, 4> newOffsetsOuter = {offsets[3], 0, 0,
                                                     linearOffset};

    rewriter.replaceOpWithNewOp<AIEX::NpuDmaMemcpyNdOp>(
        op, op.getMemref(),
        /*offsets=*/mlir::ValueRange{},
        /*sizes=*/mlir::ValueRange{},
        /*strides=*/mlir::ValueRange{},
        mlir::DenseI64ArrayAttr::get(op.getContext(), newOffsetsOuter),
        mlir::DenseI64ArrayAttr::get(op.getContext(), newSizesOuter),
        mlir::DenseI64ArrayAttr::get(op.getContext(), newStridesOuter),
        op.getPacketAttr(), op.getMetadata(), op.getIdAttr(),
        op.getIssueTokenAttr(), op.getD0ZeroBeforeAttr(),
        op.getD1ZeroBeforeAttr(), op.getD2ZeroBeforeAttr(),
        op.getD0ZeroAfterAttr(), op.getD1ZeroAfterAttr(),
        op.getD2ZeroAfterAttr(), op.getBurstLengthAttr(),
        op.getOffsetParameterAttr(), op.getOffsetStateTableIdxAttr());
    return mlir::success();
  }
};
} // namespace

void AIEX::NpuDmaMemcpyNdOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<LinearizeContiguousTransfer>(context);
}

// Helper method to check if a requested burst length is supported by the target
// model. Returns an error message if the burst length is not supported or an
// empty option otherwise.
static std::optional<std::string>
checkBurstLength(const xilinx::AIE::AIETargetModel &targetModel,
                 uint32_t requestedBurstLength) {
  if (requestedBurstLength != 0) {
    auto bel = targetModel.getShimBurstEncodingsAndLengths();
    auto pair = std::find_if(bel.begin(), bel.end(),
                             [=](const std::pair<uint32_t, uint32_t> &p) {
                               return p.second == requestedBurstLength;
                             });

    if (pair == bel.end()) {
      std::string errorMessage =
          "Requested burst length is not supported by the target. "
          "Supported burst lengths:";

      errorMessage =
          std::accumulate(bel.begin(), bel.end(), errorMessage,
                          [](const std::string &a, auto b) {
                            return a + " " + std::to_string(b.second);
                          });

      return errorMessage;
    }
  }

  return std::nullopt;
}

// Verify the supported scope for a dma_memcpy_nd carrying runtime (SSA)
// offsets/sizes/strides, hard-erroring on any statically-provable violation.
// The scope here is exactly what the dynamic BD encoder (AIEDmaToNpu.cpp
// lowerDynamic) can lower; runtime values are never silently masked.
LogicalResult AIEX::NpuDmaMemcpyNdOp::verifyDynamicSizesStrides(
    const AIE::AIETargetModel &targetModel, mlir::BaseMemRefType buffer) {
  // Shim NOC only.
  AIE::DeviceOp dev = getOperation()->getParentOfType<AIE::DeviceOp>();
  auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(
      dev, getMetadata().getRootReference());
  if (!allocOp)
    return emitOpError(
        "runtime sizes/strides require a shim_dma_allocation to resolve the "
        "tile; none found.");
  AIE::TileOp tile = allocOp.getTileOp();
  if (!tile)
    return emitOpError("shim DMA allocation must reference a valid TileOp");
  if (!targetModel.isShimNOCTile(tile.getCol(), tile.getRow()))
    return emitOpError(
        "runtime sizes/strides are only supported for shim NOC tile DMAs.");

  // No zero-padding with runtime sizes/strides.
  if (getD0ZeroBefore() || getD1ZeroBefore() || getD2ZeroBefore() ||
      getD0ZeroAfter() || getD1ZeroAfter() || getD2ZeroAfter())
    return emitOpError(
        "zero padding is not supported with runtime sizes/strides.");

  // Per-field runtime values are allowed: any individual size/stride may be an
  // SSA value while others stay constant. Only the runtime-dependent fields
  // need runtime handling; the encoder produces the same word for a constant
  // operand either way, so the static ≡ dynamic byte-identity holds field by
  // field.
  auto sizes = getMixedSizes();
  auto strides = getMixedStrides();

  // The innermost stride may be runtime like any other dimension: the encoder
  // resolves its collapse-to-zero case with a select, and its realizability
  // (unit stride, or granule-aligned) is enforced below for constants and by an
  // assert_bd_divisible guard for runtime values.

  // (The memref must also trace to a runtime-sequence block argument through
  // static subview/cast offsets; that structural check, with the same clean
  // diagnostic, is enforced by the lowering in AIEDmaToNpu.cpp, which already
  // owns the trace utility. It is not duplicated here to keep the dialect
  // verifier free of the analysis-layer dependency.)

  // A constant size must fit its hardware wrap field (an out-of-range constant
  // is a hard error, never a silent truncation); runtime sizes are guarded at
  // lowering. Bounds are checked on the ENCODED value, matching
  // verifyStridesWraps: d0 wrap scaled to granules (size * elemWidth / gran),
  // iteration wrap biased by -1, d1 a raw element count. Sizes outermost-first.
  DataLayout dataLayout = DataLayout::closest(getOperation());
  uint64_t elemWidth = dataLayout.getTypeSizeInBits(buffer.getElementType());
  uint32_t gran = targetModel.getAddressGenGranularity();
  llvm::SmallVector<mlir::OpFoldResult, 4> sizesRev(llvm::reverse(sizes));
  auto checkSize = [&](mlir::OpFoldResult ofr, int64_t hwVal, int64_t hi,
                       llvm::StringRef what) -> LogicalResult {
    if (!getConstantIntValue(ofr))
      return success();
    if (hwVal < 0 || hwVal > hi)
      return emitOpError(what) << " hardware value " << hwVal
                               << " exceeds hardware range [0:" << hi << "].";
    return success();
  };
  auto hwSize = [&](mlir::OpFoldResult ofr) {
    return getConstantIntValue(ofr).value_or(0);
  };
  int64_t d0Hw = (int64_t)(hwSize(sizesRev[0]) * elemWidth / gran);
  int64_t d1Hw = hwSize(sizesRev[1]);
  int64_t iterRaw = hwSize(sizesRev[3]);
  int64_t iterHw = iterRaw > 1 ? iterRaw - 1 : 0;
  if (failed(checkSize(sizesRev[0], d0Hw, ShimBdFieldWidths::d0WrapMax(),
                       "d0 size")) ||
      failed(checkSize(sizesRev[1], d1Hw, ShimBdFieldWidths::d1WrapMax(),
                       "d1 size")) ||
      failed(checkSize(sizesRev[3], iterHw, ShimBdFieldWidths::iterWrapMax(),
                       "iteration size")))
    return failure();

  // Realizability of the CONSTANT size/stride operands (divisibility +
  // positivity, innermost-first). Runtime operands get an assert_bd_divisible
  // guard at lowering time. Shared with the dma_task path.
  llvm::SmallVector<mlir::OpFoldResult, 4> stridesRev(llvm::reverse(strides));
  if (failed(verifyConstBdRealizability(getOperation(), sizesRev, stridesRev,
                                        elemWidth, gran)))
    return failure();

  // A runtime size landing in a narrow BD field (d0/d1 wrap 10-bit, iteration
  // 6-bit) could exceed the field and silently truncate on hardware. The TXN
  // stream has no on-device trap, so the dynamic lowering emits a host-side
  // bounds guard (npu.assert_bd_field -> generated-C++ early return of nullopt)
  // for exactly those fields. Nothing to reject here: wide fields
  // (buffer_length via linear mode, repeat_count) need no guard, and narrow
  // fields are guarded at lowering time.

  auto errorMessage = checkBurstLength(targetModel, getBurstLength());
  if (errorMessage.has_value())
    return emitOpError(errorMessage.value());

  return success();
}

LogicalResult AIEX::NpuDmaMemcpyNdOp::verify() {
  BaseMemRefType buffer = getMemref().getType();
  const auto &targetModel = AIE::getTargetModel(*this);
  auto addressGranularity = targetModel.getAddressGenGranularity();

  if (getElementTypeBitwidth() > addressGranularity) {
    return emitOpError("Maximum element bit width allowed is ")
           << addressGranularity << "bits. ";
  }
  if (buffer.hasStaticShape() &&
      (buffer.getNumElements() * getElementTypeBitwidth()) <
          addressGranularity) {
    return emitOpError("Minimum data transfer size required is ")
           << addressGranularity << "bits. ";
  }
  bool allStridesConstant = llvm::all_of(getMixedStrides(), [](OpFoldResult s) {
    return getConstantIntValue(s).has_value();
  });
  bool allSizesConstant = llvm::all_of(getMixedSizes(), [](OpFoldResult s) {
    return getConstantIntValue(s).has_value();
  });
  bool allOffsetsConstant = llvm::all_of(getMixedOffsets(), [](OpFoldResult s) {
    return getConstantIntValue(s).has_value();
  });

  // Dynamic path: any runtime size/stride/offset. A runtime offset flows into
  // the address-patch arg_plus as arith (see AIEDmaToNpu.cpp emitBufferAddress-
  // Patch); runtime sizes/strides use the dynamic BD-word encoder. The shared
  // scope check enforces what the dynamic lowering can represent.
  if (!allStridesConstant || !allSizesConstant || !allOffsetsConstant)
    return verifyDynamicSizesStrides(targetModel, buffer);

  llvm::SmallVector<int64_t, 4> inputSizes =
      llvm::map_to_vector(llvm::reverse(getMixedSizes()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  llvm::SmallVector<int64_t, 4> inputStrides =
      llvm::map_to_vector(llvm::reverse(getMixedStrides()), [](OpFoldResult s) {
        return getConstantIntValue(s).value();
      });
  llvm::SmallVector<int64_t, 4> hardwareSizes(4);
  llvm::SmallVector<int64_t, 4> hardwareStrides(4);
  getHardwareStridesWraps(targetModel, getOperation(), buffer, inputSizes,
                          inputStrides, hardwareSizes, hardwareStrides);
  int64_t offset = getOffsetInBytes();

  auto errorMessage = checkBurstLength(targetModel, getBurstLength());
  if (errorMessage.has_value()) {
    return emitOpError(errorMessage.value());
  }

  // The experimental HSA target uses this op on AIE1, skip all the AIE2
  // specific checks
  if (targetModel.getTargetArch() == AIE::AIEArch::AIE1)
    return success();

  if (offset % 4 != 0) {
    return emitOpError("Offset must be 4-byte-aligned.");
  }

  // dma_memcpy_nd transfers of the form [1, 1, 1, len][0, 0, 0, 1] do not
  // specify any data layout transformation, but simply express a contiguous
  // transfer of `len`. For backwards compatibility, we allow this to proceed
  // even if it exceeds the maximum stride/wrap size of any one dimension,
  // and simply do not lower any data layout transformations, since there is
  // no other way to express this at the dma_memcpy_nd interface otherwise.
  AIE::DeviceOp dev = getOperation()->getParentOfType<AIE::DeviceOp>();
  if (auto allocOp = AIE::ShimDMAAllocationOp::getForSymbol(
          dev, getMetadata().getRootReference())) {
    AIE::TileOp tile = allocOp.getTileOp();
    if (!tile) {
      return emitOpError("shim DMA allocation must reference a valid TileOp");
    }
    int col = tile.getCol();
    int row = tile.getRow();
    // A contiguous row-major ND access is also exempt from the ND wrap-size
    // limit: aie-dma-to-npu lowers it to linear mode (d0_size=d1_size=0),
    // and LinearizeContiguousTransfer canonicalizes it to explicit linear form.
    bool skipTransformationChecks =
        isLinearTransferWithoutTransformation() ||
        (targetModel.isShimNOCTile(col, row) &&
         AIEX::isContiguousTransfer(inputSizes, inputStrides));
    if (failed(verifyStridesWraps(*this, buffer, col, row, inputSizes,
                                  inputStrides, hardwareSizes, hardwareStrides,
                                  skipTransformationChecks))) {
      return failure();
    }
  }

  // packet header
  if (auto packetInfo = getPacket()) {
    if (packetInfo->getPktType() > 7)
      return emitOpError("Packet type field can only hold 3 bits.");
    if (packetInfo->getPktId() > 31)
      return emitOpError("Packet ID field can only hold 5 bits.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// NpuDmaWaitOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuDmaWaitOp::verify() {
  AIE::DeviceOp dev = (*this)->getParentOfType<AIE::DeviceOp>();
  // Some passes (e.g. aie-standard-lowering) use aiex ops outside a DeviceOp,
  // so we can't expect the device to always exist.
  if (dev && !dev.lookupSymbol(getSymbol()))
    return emitOpError("couldn't find symbol in parent device");
  return success();
}

//===----------------------------------------------------------------------===//
// NpuPushQueueOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuPushQueueOp::verify() {
  const auto &targetModel = AIE::getTargetModel(*this);
  auto numBds = targetModel.getNumBDs(getColumn(), getRow());
  // bd_id and repeat_count are SSA operands; range-check them only when they
  // are compile-time constants. A runtime (non-constant) value is left
  // unchecked here: bounds checking of runtime operands is not yet implemented
  // (it belongs to the dynamic lowering path added in a later patch).
  if (std::optional<uint32_t> bdId = getConstantIntOperand(getBdId());
      bdId && *bdId > numBds)
    return emitOpError("BD ID exceeds the maximum ID.");
  if (std::optional<uint32_t> repeatCount =
          getConstantIntOperand(getRepeatCount());
      repeatCount && *repeatCount > 255)
    return emitOpError("Repeat count exceeds the [0:255] range.");
  return success();
}

//===----------------------------------------------------------------------===//
// NpuWriteBdOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuWriteBdOp::verify() {
  const auto &targetModel = AIE::getTargetModel(*this);
  auto numBds = targetModel.getNumBDs(getColumn(), getRow());
  bool isLinearTransfer =
      (getD0Size() >= 1) && (getD1Size() == 1) && (getIterationSize() == 0);
  if (getBdId() > numBds)
    return emitOpError("BD ID exceeds the maximum ID.");
  if (getPacketId() > 31)
    return emitOpError("Packet ID exceeds the maximum supported by 5 bits.");
  if (getPacketType() > 7)
    return emitOpError("Packet Type exceeds the maximum supported by 3 bits.");
  if (!isLinearTransfer && getD0Size() > 0x3FF)
    return emitOpError("D0 Size exceeds the [0:1023] range.");
  if (getD0Stride() > 0xFFFFF)
    return emitOpError("D0 Stride exceeds the [0:1M-1] range.");
  if (getD1Size() > 0x3FF)
    return emitOpError("D1 Size exceeds the [0:1023] range.");
  if (getD1Stride() > 0xFFFFF)
    return emitOpError("D1 Stride exceeds the [0:1M-1] range.");
  if (getD2Stride() > 0xFFFFF)
    return emitOpError("D2 Stride exceeds the [0:1M-1] range.");
  if (getIterationSize() > 0x3F)
    return emitOpError("Iteration Size exceeds the [0:63] range.");
  if (getIterationStride() > 0xFFFFF)
    return emitOpError("Iteration Stride exceeds the [0:1M-1] range.");
  if (targetModel.isShimNOCTile(getColumn(), getRow()) && getD2Size() != 0)
    return emitOpError("ShimTile only supports 3 dimensions of sizes.");
  if (targetModel.isShimNOCTile(getColumn(), getRow()) &&
      (getD0ZeroBefore() != 0 || getD0ZeroAfter() != 0 ||
       getD1ZeroBefore() != 0 || getD1ZeroAfter() != 0 ||
       getD2ZeroBefore() != 0 || getD2ZeroAfter() != 0))
    return emitOpError("ShimTile doesn't support zero padding.");
  if (!targetModel.isShimNOCTile(getColumn(), getRow()) &&
      getBurstLength() != 0)
    return emitOpError("Only ShimTiles support burst length.");
  auto errorMessage = checkBurstLength(targetModel, getBurstLength());
  if (errorMessage.has_value()) {
    return emitOpError(errorMessage.value());
  }

  return success();
}

std::optional<uint32_t> AIEX::getConstantIntOperand(mlir::Value v) {
  mlir::APInt cst;
  if (!mlir::matchPattern(v, mlir::m_ConstantInt(&cst)))
    return std::nullopt;
  return static_cast<uint32_t>(cst.getZExtValue());
}

mlir::Value AIEX::createConstantI32(mlir::OpBuilder &builder,
                                    mlir::Location loc, uint32_t value) {
  return arith::ConstantOp::create(
      builder, loc, builder.getI32IntegerAttr(static_cast<int32_t>(value)));
}

//===----------------------------------------------------------------------===//
// NpuWrite32Op
//===----------------------------------------------------------------------===//

template <typename T>
static std::optional<uint32_t> getAbsoluteAddress(T *op,
                                                  uint32_t addressOffset) {
  AIE::DeviceOp device =
      op->getOperation()->template getParentOfType<AIE::DeviceOp>();
  if (!device) {
    op->emitError("Must be inside a device.");
    return std::nullopt;
  }
  const AIE::AIETargetModel &tm = device.getTargetModel();

  uint32_t address = 0;

  // If blockwrite references a buffer, the given address is understood to be
  // relative to the buffer's start address.
  if (op->getBuffer()) {
    AIE::BufferOp buffer = device.lookupSymbol<AIE::BufferOp>(*op->getBuffer());
    if (!buffer) {
      op->emitError() << "buffer '" << *op->getBuffer()
                      << "' not found in device";
      return std::nullopt;
    }

    if (!buffer.getAddress()) {
      mlir::InFlightDiagnostic err =
          op->emitError("referenced buffer must have address assigned");
      err.attachNote(buffer.getLoc()) << "This buffer must have an address.";
      return std::nullopt;
    }

    uint32_t col = buffer.getTileOp().getCol();
    uint32_t row = buffer.getTileOp().getRow();
    address = static_cast<uint32_t>(*buffer.getAddress()) +
              addressOffset * sizeof(uint32_t);
    address = ((col & 0xff) << tm.getColumnShift()) |
              ((row & 0xff) << tm.getRowShift()) | (address & 0xfffff);
  } else { // otherwise, the given address is absolute
    address = addressOffset;
    std::optional<uint32_t> col = op->getColumn();
    std::optional<uint32_t> row = op->getRow();
    if (col && row) {
      // If col and row are set, only the lower 20 bits of the address are
      // used, and col and row dictate the upper bits (ignored)
      address = ((*col & 0xff) << tm.getColumnShift()) |
                ((*row & 0xff) << tm.getRowShift()) | (address & 0xfffff);
    }
  }

  return address;
}

std::optional<uint32_t> AIEX::NpuWrite32Op::getAbsoluteAddress() {
  std::optional<uint32_t> addressOffset = getConstantIntOperand(getAddress());
  if (!addressOffset)
    return std::nullopt;
  return ::getAbsoluteAddress(this, *addressOffset);
}

//===----------------------------------------------------------------------===//
// NpuAssertBdFieldOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuAssertBdFieldOp::verify() {
  if (auto c = getConstantIntValue(getValue()))
    if (*c < 0 || *c > (int64_t)getMax())
      return emitOpError("constant value ")
             << *c << " exceeds the guarded field range [0:" << getMax()
             << "].";
  return success();
}

//===----------------------------------------------------------------------===//
// NpuAssertBdDivisibleOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuAssertBdDivisibleOp::verify() {
  if (getDivisor() == 0)
    return emitOpError("divisor must be non-zero.");
  if (auto c = getConstantIntValue(getValue())) {
    if (getAllowUnit() && *c == 1)
      return success();
    if (*c % (int64_t)getDivisor() != 0)
      return emitOpError("constant value ")
             << *c << " is not divisible by " << getDivisor()
             << " (transfer is not a whole number of address-gen granules).";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NpuAddressPatchOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuAddressPatchOp::verify() {
  // A runtime register address (addr_val) is meaningful only on the EmitC (C++
  // TXN) target; the constant `addr` still carries the fallback / static value.
  // The static binary target checks for the operand and diagnoses it there.
  if (getAddrVal() && !getAddrVal().getType().isInteger(32))
    return emitOpError("addr_val must be an i32 value");
  return success();
}

//===----------------------------------------------------------------------===//
// NpuUpdateFromScratchpadOp
//===----------------------------------------------------------------------===//

std::optional<uint32_t> AIEX::NpuUpdateFromScratchpadOp::getAbsoluteAddress() {
  return ::getAbsoluteAddress(this, getAddress());
}

LogicalResult AIEX::NpuUpdateFromScratchpadOp::verify() {
  // StateTable has at most 32 entries (32-bit words).
  constexpr uint32_t kMaxStateTableEntries = 32;
  if (getStateTableIdx() >= kMaxStateTableEntries)
    return emitOpError("state_table_idx ")
           << static_cast<uint32_t>(getStateTableIdx())
           << " exceeds maximum StateTable index ("
           << (kMaxStateTableEntries - 1) << ").";

  // Cross-check against any npu.create_scratchpad ops in the same block: the
  // index must fit within the allocated scratchpad (size in 32-bit words).
  Block *block = (*this)->getBlock();
  if (block) {
    for (auto createOp : block->getOps<AIEX::NpuCreateScratchpadOp>()) {
      uint32_t sizeBytes = createOp.getSize();
      uint32_t numEntries = sizeBytes / 4;
      if (getStateTableIdx() >= numEntries) {
        return emitOpError("state_table_idx ")
               << static_cast<uint32_t>(getStateTableIdx())
               << " is out of bounds for scratchpad of size " << sizeBytes
               << " bytes (" << numEntries << " entries) created by "
               << createOp->getName() << ".";
      }
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NpuCreateScratchpadOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuCreateScratchpadOp::verify() {
  // Only usage_type == 0 is currently supported by firmware.
  if (getUsageType() != 0) {
    return emitOpError("usage_type must be 0 (got ")
           << static_cast<uint32_t>(getUsageType())
           << "); other layouts are not supported.";
  }

  // The StateTable layout (usage_type 0) has a max of 32 32-bit-word entries,
  // i.e. a maximum total scratchpad size of 128 bytes.
  constexpr uint32_t kMaxScratchpadSizeBytes = 128;
  if (getSize() == 0) {
    return emitOpError("size must be greater than 0.");
  }
  if (getSize() % 4 != 0) {
    return emitOpError("size (")
           << getSize() << ") must be a multiple of 4 bytes.";
  }
  if (getSize() > kMaxScratchpadSizeBytes) {
    return emitOpError("size (")
           << getSize() << " bytes) exceeds maximum scratchpad size of "
           << kMaxScratchpadSizeBytes << " bytes.";
  }

  // At most one create_scratchpad may appear per runtime sequence. Walk the
  // parent RuntimeSequenceOp to check; only report from the duplicate (i.e.
  // the op that is NOT the first occurrence) to avoid emitting the same error
  // twice.
  auto runtimeSeq = getOperation()->getParentOfType<AIE::RuntimeSequenceOp>();
  if (!runtimeSeq) {
    return success();
  }

  NpuCreateScratchpadOp firstSeen;
  runtimeSeq.walk([&](NpuCreateScratchpadOp op) {
    if (!firstSeen) {
      firstSeen = op;
    }
  });
  if (firstSeen != *this) {
    InFlightDiagnostic diag =
        emitOpError("only one 'aiex.npu.create_scratchpad' is allowed per "
                    "runtime sequence");
    diag.attachNote(firstSeen.getLoc())
        << "previous 'aiex.npu.create_scratchpad' here";
    return diag;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// NpuMaskWrite32Op
//===----------------------------------------------------------------------===//

std::optional<uint32_t> AIEX::NpuMaskWrite32Op::getAbsoluteAddress() {
  std::optional<uint32_t> addressOffset = getConstantIntOperand(getAddress());
  if (!addressOffset)
    return std::nullopt;
  return ::getAbsoluteAddress(this, *addressOffset);
}

//===----------------------------------------------------------------------===//
// NpuBlockWriteOp
//===----------------------------------------------------------------------===//

std::optional<uint32_t> AIEX::NpuBlockWriteOp::getAbsoluteAddress() {
  return ::getAbsoluteAddress(this, getAddress());
}

DenseIntElementsAttr AIEX::NpuBlockWriteOp::getDataWords() {
  Value memref = this->getData();
  DataLayout dataLayout = DataLayout::closest(*this);
  int64_t width = dataLayout.getTypeSizeInBits(
      cast<MemRefType>(memref.getType()).getElementType());
  if (width != 32) {
    emitWarning("Only 32-bit data type is supported for now");
    return nullptr;
  }

  memref::GetGlobalOp getGlobal = memref.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal) {
    emitError("Only MemRefs from memref.get_global are supported");
    return nullptr;
  }

  auto global = dyn_cast_if_present<memref::GlobalOp>(
      (*this)->getParentOfType<AIE::DeviceOp>().lookupSymbol(
          getGlobal.getName()));
  if (!global) {
    emitError("Global symbol not found");
    return nullptr;
  }

  auto initVal = global.getInitialValue();
  if (!initVal) {
    emitError("Global symbol has no initial value");
    return nullptr;
  }

  auto data = dyn_cast<DenseIntElementsAttr>(*initVal);
  if (!data) {
    emitError("Global symbol initial value is not a dense int array");
    return nullptr;
  }

  return data;
}

//===----------------------------------------------------------------------===//
// DMAConfigureTaskOp
//===----------------------------------------------------------------------===//

std::optional<uint32_t> AIEX::DMAConfigureTaskOp::getFirstBdId() {
  Region &body = getBody();
  if (body.empty()) {
    return std::nullopt;
  }
  auto bd_ops = body.front().getOps<AIE::DMABDOp>();
  if (bd_ops.empty() && body.front().getNumSuccessors() == 1) {
    // Allow the first block to be empty and point to the entry point of the
    // chain. This allows for specifying cyclying BD chains (infinite loops)
    // within the constraints of MLIR syntax.
    Block &chain_entry = *body.front().getSuccessor(0);
    bd_ops = chain_entry.getOps<AIE::DMABDOp>();
  }
  if (bd_ops.empty()) {
    return std::nullopt;
  }
  AIE::DMABDOp bd = *bd_ops.begin();
  if (!bd.getBdId().has_value()) {
    return std::nullopt;
  }
  return bd.getBdId().value();
}

LogicalResult
AIEX::DMAConfigureTaskOp::canonicalize(AIEX::DMAConfigureTaskOp op,
                                       PatternRewriter &rewriter) {
  // Remove blocks that contain nothing but a terminator
  Region &body = op.getBody();
  bool did_rewrite = false;
  for (auto it = body.begin(); it != body.end(); ++it) {
    Block &block = *it;
    if (block.empty()) {
      continue;
    }
    auto ops_it = block.without_terminator();
    if (std::distance(ops_it.begin(), ops_it.end()) == 0) {
      rewriter.eraseOp(block.getTerminator());
      did_rewrite = true;
    }
  }
  if (did_rewrite) {
    return success();
  }
  return failure();
}

// Enforce the per-BD ND access-pattern limit for BDs nested inside a
// runtime-sequence DMA task. The AIE::DMABDOp verifier skips these BDs (their
// parent is a DMA task op, not a *DMAOp), so this is the only check of the BD
// dimension count on the runtime-sequence path.
//
// Every AIE2/AIE2P DMA BD register file carries getBDMaxDims ND address
// dimensions (D0..) plus one separate iteration/repeat dimension: a core/shim
// BD has D0..D2 + iteration, a MemTile BD has D0..D3 + iteration. On this path
// aiex.shim_dma_single_bd_task hoists the leading tap dimension into that
// iteration register, so a shim/core BD may carry one dimension beyond its ND
// access limit (3 + 1). A MemTile is not given the +1: AIEDMATasksToNPU maps
// the 4th task dimension onto the iteration register for every tile type and
// caps the total at 4, which the MemTile's 4 ND dimensions already reach. Both
// branches therefore land on the same uniform 4-dimension cap enforced later by
// AIEDMATasksToNPU.
static LogicalResult
verifyTaskBDDimensions(const AIE::AIETargetModel &targetModel, int col, int row,
                       Region &body) {
  size_t maxNDims = targetModel.getBDMaxDims(col, row);
  if (!targetModel.isMemTile(col, row))
    ++maxNDims; // leading dim is hoisted into the iteration/repeat register
  LogicalResult result = success();
  body.walk([&](AIE::DMABDOp bd) {
    size_t numDims = bd.getMixedSizes().size();
    if (numDims > maxNDims) {
      bd.emitOpError() << "Cannot give more than " << std::to_string(maxNDims)
                       << " dimensions for step sizes and wraps on this tile "
                          "(got "
                       << std::to_string(numDims) << " dimensions).";
      result = failure();
    }
  });
  return result;
}

LogicalResult AIEX::DMAConfigureTaskOp::verify() {
  const AIE::AIETargetModel &targetModel = AIE::getTargetModel(getOperation());
  // Skip the per-BD dimension check on an unplaced (logical) tile: the ND limit
  // is a function of the tile's placed coordinates, which are not yet known.
  // The verifier runs again on the concrete tile once placement resolves it.
  std::optional<int> col = getTileLike().tryGetCol();
  std::optional<int> row = getTileLike().tryGetRow();
  if (col && row &&
      failed(verifyTaskBDDimensions(targetModel, *col, *row, getBody())))
    return failure();
  Region &body = getBody();
  for (auto it = body.begin(); it != body.end(); ++it) {
    Block &block = *it;
    if (block.empty()) {
      continue;
    }
    if (block.hasNoPredecessors() && !block.isEntryBlock()) {
      auto error = block.getTerminator()->emitError(
          "Block ending in this terminator does not form a chain with "
          "entry block.");
      return failure();
    }

    const AIE::AIETargetModel &targetModel =
        AIE::getTargetModel(getOperation());

    // This is a layering violation on the DMABDOps, but they are never verified
    // otherwise Because DMAConfigureTaskOps are not yet merged into the AIE
    // dialect. The normal DMABDOp verify operation will skip over any BD inside
    // a DMAConfigureTaskOp
    LogicalResult result = success();
    block.walk([&](AIE::DMABDOp bd) {
      if (bd.getBurstLength() != 0 &&
          !targetModel.isShimNOCTile(getTileID().col, getTileID().row)) {
        bd.emitOpError("Burst length is only supported in Shim NOC tiles that "
                       "are connected to the memory-mapped NOC.");
        result = failure();
      }
    });
    if (failed(result)) {
      return result;
    }
  }
  return success();
}

LogicalResult AIEX::DMAConfigureTaskForOp::verify() {
  // Recover the shim tile through the referenced shim DMA allocation symbol so
  // the per-BD dimension limit can be enforced on the runtime-sequence path
  // before the allocation is substituted into a concrete DMAConfigureTaskOp.
  AIE::DeviceOp dev = getOperation()->getParentOfType<AIE::DeviceOp>();
  if (!dev)
    return success();
  AIE::ShimDMAAllocationOp allocOp = AIE::ShimDMAAllocationOp::getForSymbol(
      dev, getAlloc().getRootReference());
  if (!allocOp)
    return success(); // symbol resolved during a later pass; defer the check
  // Do not call allocOp.getTileOp(): it hard-asserts when the allocation is
  // still bound to an unplaced (logical) tile. Resolve the concrete tile
  // defensively and defer the check until placement substitutes a real tile.
  auto tile =
      llvm::dyn_cast_or_null<AIE::TileOp>(allocOp.getTile().getDefiningOp());
  if (!tile)
    return success();
  const AIE::AIETargetModel &targetModel = AIE::getTargetModel(getOperation());
  return verifyTaskBDDimensions(targetModel, tile.getCol(), tile.getRow(),
                                getBody());
}

//===----------------------------------------------------------------------===//
// DMAStartBdChainOp
//===----------------------------------------------------------------------===//

AIE::BDChainOp AIEX::DMAStartBdChainOp::getBDChainOp() {
  AIE::DeviceOp device = (*this)->getParentOfType<AIE::DeviceOp>();
  AIE::BDChainOp chain = device.lookupSymbol<AIE::BDChainOp>(getSymbol());
  return chain;
}

LogicalResult AIEX::DMAStartBdChainOp::verify() {
  AIE::BDChainOp chain = getBDChainOp();
  if (!chain) {
    return emitOpError("symbol does not reference valid BD chain");
  }

  auto actualArgTypes = getArgs().getTypes();
  auto expectedArgTypes = chain.getRegion().getArgumentTypes();
  if (actualArgTypes.size() != expectedArgTypes.size()) {
    return emitOpError("Number of arguments mismatches.");
  }
  for (unsigned i = 0, n = expectedArgTypes.size(); i < n; i++) {
    if (actualArgTypes[i] != expectedArgTypes[i]) {
      return emitOpError("Argument ") << (i + 1) << " types mismatch: "
                                      << "expected " << expectedArgTypes[i]
                                      << " but got " << actualArgTypes[i];
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// NpuControlPacketOp
//===----------------------------------------------------------------------===//

uint32_t AIEX::NpuControlPacketOp::getRowFromAddr() {
  const auto &targetModel = AIE::getTargetModel(*this);
  uint32_t addr = getAddress();
  uint32_t rowInt = (addr >> targetModel.getRowShift()) & 0x1f;
  return rowInt;
}

uint32_t AIEX::NpuControlPacketOp::getColumnFromAddr() {
  const auto &targetModel = AIE::getTargetModel(*this);
  uint32_t addr = getAddress();
  uint32_t colInt = (addr >> targetModel.getColumnShift()) & 0x1f;
  return colInt;
}

//===----------------------------------------------------------------------===//
// SetLockOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::SetLockOp::verify() {
  const auto &targetModel = AIE::getTargetModel(*this);

  if (targetModel.getTargetArch() == AIE::AIEArch::AIE1)
    return emitOpError("SetLockOp is not supported on AIE1.");

  if (getValue() > targetModel.getMaxLockValue())
    return emitOpError("Lock value exceeds the maximum value of " +
                       std::to_string(targetModel.getMaxLockValue()));

  auto lockOp = getLockOp();
  auto lockIDOpt = getLockOp().getLockID();
  // Note that the lockID may not be assigned initially, so lets wait until it
  // is to verify the lockID dependent conditions
  if (!lockIDOpt) {
    return success();
  }

  auto col = lockOp.colIndex();
  auto row = lockOp.rowIndex();
  uint32_t lockID = lockOp.getLockIDValue();

  if (lockID >= targetModel.getNumLocks(col, row)) {
    return emitOpError("Lock ID out of range for given tile. Max ID: " +
                       std::to_string(targetModel.getNumLocks(col, row) - 1));
  }

  if (!targetModel.getLocalLockAddress(lockID, lockOp.getTileID())) {
    return emitOpError("Invalid lock ID and tile combination when trying to "
                       "retrieve the local lock address.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BlockFloatingPointType
//===----------------------------------------------------------------------===//
uint64_t AIEX::BlockFloatType::getTotalSizeInBits() const {
  return getBlockSize() * getMantissaBits() + getExponentBits() +
         getSubtileShiftBits();
}

llvm::TypeSize AIEX::BlockFloatType::getTypeSizeInBits(
    const mlir::DataLayout &dataLayout,
    mlir::DataLayoutEntryListRef params) const {
  return llvm::TypeSize::getFixed(getTotalSizeInBits());
}

uint64_t AIEX::BlockFloatType::getABIAlignment(
    const mlir::DataLayout &dataLayout,
    mlir::DataLayoutEntryListRef params) const {
  // For the purposes of the data movement operations, we want all types to be
  // packed <=> ABI alignment is 1.
  return 1;
}

std::optional<AIEX::BlockFloatType::BlockFormat>
AIEX::BlockFloatType::getBlockFormat(StringRef blockType) {
  static const llvm::StringMap<AIEX::BlockFloatType::BlockFormat>
      blockFormatsMap = {
          {"v8bfp16ebs8", {8, 8, 8, 0}},
          {"v16bfp16ebs16", {16, 8, 8, 0}},
      };

  auto it = blockFormatsMap.find(blockType);
  if (it != blockFormatsMap.end()) {
    return it->second;
  }

  return std::nullopt;
}

LogicalResult
AIEX::BlockFloatType::verify(function_ref<InFlightDiagnostic()> emitError,
                             StringRef block_type) {
  if (!getBlockFormat(block_type))
    return emitError() << "Invalid block type: " << block_type
                       << ". Known types are: v8bfp16ebs8, v16bfp16ebs16.";

  return success();
}

//===----------------------------------------------------------------------===//
// ConfigureOp
//===----------------------------------------------------------------------===//

AIE::DeviceOp AIEX::ConfigureOp::getReferencedDeviceOp() {
  ModuleOp moduleOp = this->getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    emitError("aiex.configure must be inside of a module");
    return nullptr;
  }
  Operation *maybeReferencedDevice =
      SymbolTable::lookupSymbolIn(moduleOp.getOperation(), getSymbolAttr());
  if (!maybeReferencedDevice) {
    emitError("No such device: '") << getSymbolAttr() << "'";
    return nullptr;
  }
  AIE::DeviceOp referencedDevice =
      llvm::dyn_cast<AIE::DeviceOp>(maybeReferencedDevice);
  if (!referencedDevice) {
    emitError("Not a device: '") << getSymbolAttr() << "'";
    return nullptr;
  }
  return referencedDevice;
}

//===----------------------------------------------------------------------===//
// ReadScratchpadParameterOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::ReadScratchpadParameterOp::verify() {
  auto device = (*this)->getParentOfType<AIE::DeviceOp>();
  if (!device) {
    return emitOpError("must be inside an aie.device");
  }
  if (!(*this)->getParentOfType<AIE::CoreOp>()) {
    return emitOpError("must be inside an aie.core");
  }
  auto moduleOp = (*this)->getParentOfType<ModuleOp>();
  if (!moduleOp ||
      !moduleOp.lookupSymbol<AIEX::ScratchpadParameterOp>(getParameter())) {
    return emitOpError("references unknown parameter '")
           << getParameter()
           << "' (aiex.scratchpad_parameter ops are declared at module scope)";
  }
  if (getResult().getType().isF32()) {
    return emitOpError(
        "f32 parameters are not supported: the scratchpad encoding zeroes "
        "the top 2 bits, which clobbers the sign bit and top exponent bit "
        "of an f32. Use bf16 or an integer type up to i32 instead.");
  }
  return success();
}

LogicalResult AIEX::ConfigureOp::verify() {
  AIE::DeviceOp parentDev = getOperation()->getParentOfType<AIE::DeviceOp>();
  AIE::DeviceOp referencedDev = getReferencedDeviceOp();
  if (!referencedDev) {
    return failure();
  }
  if (parentDev.getDevice() != referencedDev.getDevice()) {
    emitError("Device types do not match: '")
        << AIE::stringifyAIEDevice(parentDev.getDevice()) << "' vs. '"
        << AIE::stringifyAIEDevice(referencedDev.getDevice()) << "'";
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RunOp
//===----------------------------------------------------------------------===//

AIE::DeviceOp AIEX::RunOp::getCalleeDeviceOp() {
  AIEX::ConfigureOp configureOp =
      getOperation()->getParentOfType<AIEX::ConfigureOp>();
  if (!configureOp) {
    return nullptr;
  }
  AIE::DeviceOp referencedDevice = configureOp.getReferencedDeviceOp();
  return referencedDevice;
}

AIE::RuntimeSequenceOp AIEX::RunOp::getCalleeRuntimeSequenceOp() {
  AIEX::ConfigureOp configureOp =
      getOperation()->getParentOfType<AIEX::ConfigureOp>();
  if (!configureOp) {
    return nullptr;
  }
  AIE::DeviceOp referencedDevice = configureOp.getReferencedDeviceOp();
  if (!referencedDevice) {
    return nullptr;
  }

  Operation *maybeRuntimeSequence =
      SymbolTable::lookupSymbolIn(referencedDevice, getRuntimeSequenceSymbol());

  if (!maybeRuntimeSequence) {
    return nullptr;
  }
  AIE::RuntimeSequenceOp runtimeSequence =
      llvm::dyn_cast<AIE::RuntimeSequenceOp>(maybeRuntimeSequence);
  if (!runtimeSequence) {
    return nullptr;
  }

  return runtimeSequence;
}

//===----------------------------------------------------------------------===//
// NpuLoadPdiOp
//===----------------------------------------------------------------------===//

LogicalResult AIEX::NpuLoadPdiOp::canonicalize(AIEX::NpuLoadPdiOp op,
                                               PatternRewriter &rewriter) {
  // Check for back-to-back identical load_pdi ops and remove duplicates
  Operation *nextOp = op->getNextNode();
  if (!nextOp)
    return failure();

  // Check if next op is also a NpuLoadPdiOp
  auto nextLoadPdi = dyn_cast<AIEX::NpuLoadPdiOp>(nextOp);
  if (!nextLoadPdi)
    return failure();

  // Check if they are identical (all attributes match)
  if (op.getDeviceRefAttr() == nextLoadPdi.getDeviceRefAttr() &&
      op.getId() == nextLoadPdi.getId() &&
      op.getSize() == nextLoadPdi.getSize() &&
      op.getAddress() == nextLoadPdi.getAddress()) {
    // Erase the first one, keeping the second
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}
