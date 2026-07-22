//===- DmaDecomposition.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEX/Utils/DmaDecomposition.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Utils/BdLowering.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

std::pair<uint32_t, uint32_t> getWrapStepBits(const AIE::AIETargetModel &tm,
                                              int col, int row) {
  if (tm.isShimNOCTile(col, row))
    return {10, 20};
  if (tm.isMemTile(col, row))
    return {10, 17};
  if (tm.isCoreTile(col, row))
    return {8, 13};
  return {0, 0};
}

int64_t maxLegalInputSizeForDim(const AIE::AIETargetModel &tm, int col,
                                int row, unsigned dim, uint64_t elemWidth,
                                uint32_t gran) {
  uint32_t wrapBits = getWrapStepBits(tm, col, row).first;
  if (wrapBits == 0)
    return 0;

  if (dim == 0) {
    int64_t maxHw = (1LL << wrapBits) - 1;
    int64_t maxInput = maxHw * (int64_t)gran / (int64_t)elemWidth;
    int64_t divisor = bdGranuleDivisor(elemWidth, gran);
    if (divisor > 1)
      maxInput = (maxInput / divisor) * divisor;
    return maxInput;
  }
  if (dim == 3)
    return 1LL << 6; // iteration wrap is 6 bits
  return (1LL << wrapBits) - 1;
}

bool patternPassesVerification(Operation *forOp, BaseMemRefType bufType,
                               const AIE::AIETargetModel &tm, int col, int row,
                               const NdDmaPattern &pattern) {
  SmallVector<int64_t, 4> hwSizes(4);
  SmallVector<int64_t, 4> hwStrides(4);
  getHardwareStridesWraps(tm, forOp, bufType, pattern.sizes, pattern.strides,
                          hwSizes, hwStrides);

  ScopedDiagnosticHandler handler(forOp->getContext(),
                                  [](Diagnostic &) { return success(); });
  return succeeded(verifyStridesWraps(forOp, bufType, col, row, pattern.sizes,
                                    pattern.strides, hwSizes, hwStrides,
                                    /*skipTransformationChecks=*/false));
}

/// Enumerate divisors b of n in descending order (largest inner factor first).
void divisorsDescending(int64_t n, SmallVectorImpl<int64_t> &out) {
  out.clear();
  if (n <= 1)
    return;
  SmallVector<int64_t, 32> small;
  for (int64_t b = 2; b * b <= n; ++b) {
    if (n % b != 0)
      continue;
    out.push_back(n / b);
    if (b != n / b)
      small.push_back(b);
  }
  llvm::sort(out);
  llvm::reverse(out);
  out.append(small.rbegin(), small.rend());
}

FailureOr<SmallVector<NdDmaPattern>>
decomposeRecursive(Operation *forOp, BaseMemRefType bufType,
                   const AIE::AIETargetModel &tm, int col, int row,
                   const NdDmaPattern &pattern) {
  if (patternPassesVerification(forOp, bufType, tm, col, row, pattern))
    return SmallVector<NdDmaPattern>{pattern};

  DataLayout dataLayout = DataLayout::closest(forOp);
  uint64_t elemWidth = dataLayout.getTypeSizeInBits(bufType.getElementType());
  uint32_t gran = tm.getAddressGenGranularity();

  // The hardware BD emits elements in lexicographic order of the loop indices
  // with d0 the innermost (fastest) and d3 the outermost (slowest) dimension.
  // A decomposition is only correct if the concatenation of the sub-transfers'
  // emitted element sequences is IDENTICAL (same order, not just same set) to
  // the original. Both transformations below are order-preserving by
  // construction.

  // Outermost active dimension (highest index with size > 1).
  int outermost = -1;
  for (int i = 3; i >= 0; --i)
    if (pattern.sizes[i] > 1) {
      outermost = i;
      break;
    }
  if (outermost < 0)
    return failure();

  // (1) Order-preserving dimension factoring: split dim d (size N = a*b,
  // stride s) into an inner dim (b, s) kept at position d and an outer dim
  // (a, b*s) inserted at position d+1, shifting the higher dims outward. The
  // factored pair stays adjacent so the sub-traversal of dim d is contiguous
  // and its place in the overall nesting is unchanged => element order is
  // preserved. Requires the outermost slot to be free so no dim is dropped.
  if (pattern.sizes[3] == 1) {
    for (unsigned d = 0; d < 3; ++d) {
      int64_t n = pattern.sizes[d];
      if (n <= 1)
        continue;
      int64_t s = pattern.strides[d];

      SmallVector<int64_t, 32> divisors;
      divisorsDescending(n, divisors);
      for (int64_t b : divisors) {
        int64_t a = n / b;
        if (a <= 1 || b <= 1)
          continue;

        // Both factors must remain granule-realizable on the innermost dim.
        if (d == 0 && (!isConstMultipleOfGranule(b, elemWidth, gran)))
          continue;

        NdDmaPattern factored = pattern;
        // Shift dims (d+1 .. 2) outward to (d+2 .. 3).
        for (unsigned i = 3; i > d + 1; --i) {
          factored.sizes[i] = pattern.sizes[i - 1];
          factored.strides[i] = pattern.strides[i - 1];
          factored.offsets[i] = pattern.offsets[i - 1];
        }
        factored.sizes[d] = b;           // inner factor
        factored.strides[d] = s;         // inner keeps original stride/offset
        factored.sizes[d + 1] = a;       // outer factor
        factored.strides[d + 1] = b * s; // outer stride
        factored.offsets[d + 1] = 0;

        auto sub =
            decomposeRecursive(forOp, bufType, tm, col, row, factored);
        if (succeeded(sub))
          return sub;
      }
    }
  }

  // (2) Order-preserving slicing: only the OUTERMOST active dimension may be
  // split into contiguous index ranges emitted in order. Slicing an inner
  // dimension would interleave the outer iterations and reorder the emitted
  // element stream, so it is not allowed.
  {
    unsigned d = static_cast<unsigned>(outermost);
    int64_t n = pattern.sizes[d];
    int64_t chunkSize =
        maxLegalInputSizeForDim(tm, col, row, d, elemWidth, gran);
    if (chunkSize > 0 && chunkSize < n) {
      int64_t numChunks = (n + chunkSize - 1) / chunkSize;
      SmallVector<NdDmaPattern> combined;
      for (int64_t i = 0; i < numChunks; ++i) {
        NdDmaPattern slice = pattern;
        slice.sizes[d] = std::min(chunkSize, n - i * chunkSize);
        slice.offsets[d] = pattern.offsets[d] + i * chunkSize;

        auto sub = decomposeRecursive(forOp, bufType, tm, col, row, slice);
        if (failed(sub))
          return failure();
        combined.append(sub->begin(), sub->end());
      }
      return combined;
    }
  }

  return failure();
}

} // namespace

bool AIEX::isNdDmaPatternLegal(Operation *forOp, BaseMemRefType referencedBufType,
                               const AIE::AIETargetModel &targetModel,
                               int tileCol, int tileRow,
                               const NdDmaPattern &pattern) {
  return patternPassesVerification(forOp, referencedBufType, targetModel,
                                   tileCol, tileRow, pattern);
}

bool AIEX::isDecomposableNdDmaPattern(
    Operation *forOp, BaseMemRefType referencedBufType,
    const AIE::AIETargetModel &targetModel, int tileCol, int tileRow,
    ArrayRef<int64_t> offsetsInnermostFirst,
    ArrayRef<int64_t> sizesInnermostFirst,
    ArrayRef<int64_t> stridesInnermostFirst) {
  if (offsetsInnermostFirst.size() != 4 || sizesInnermostFirst.size() != 4 ||
      stridesInnermostFirst.size() != 4)
    return false;

  if (isContiguousTransfer(sizesInnermostFirst, stridesInnermostFirst))
    return false;

  NdDmaPattern pattern;
  pattern.offsets.assign(offsetsInnermostFirst.begin(),
                       offsetsInnermostFirst.end());
  pattern.sizes.assign(sizesInnermostFirst.begin(), sizesInnermostFirst.end());
  pattern.strides.assign(stridesInnermostFirst.begin(),
                         stridesInnermostFirst.end());

  if (isNdDmaPatternLegal(forOp, referencedBufType, targetModel, tileCol,
                          tileRow, pattern))
    return false;

  auto decomposed = decomposeNdDmaPattern(forOp, referencedBufType, pattern,
                                            targetModel, tileCol, tileRow);
  return succeeded(decomposed) && !decomposed->empty();
}

FailureOr<SmallVector<NdDmaPattern>>
AIEX::decomposeNdDmaPattern(Operation *forOp, BaseMemRefType referencedBufType,
                            const NdDmaPattern &pattern,
                            const AIE::AIETargetModel &targetModel, int tileCol,
                            int tileRow) {
  if (pattern.offsets.size() != 4 || pattern.sizes.size() != 4 ||
      pattern.strides.size() != 4)
    return failure();

  if (isContiguousTransfer(pattern.sizes, pattern.strides))
    return failure();

  if (isNdDmaPatternLegal(forOp, referencedBufType, targetModel, tileCol,
                          tileRow, pattern))
    return failure();

  return decomposeRecursive(forOp, referencedBufType, targetModel, tileCol,
                            tileRow, pattern);
}
