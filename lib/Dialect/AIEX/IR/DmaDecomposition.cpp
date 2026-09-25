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

/// The longest iteration dimension (d3) a BD holds: its wrap field is biased
/// by one.
int64_t maxIterations(const AIE::AIETargetModel &tm, int col, int row) {
  return 1LL << tm.getDmaBdIterBits(col, row);
}

int64_t maxLegalInputSizeForDim(const AIE::AIETargetModel &tm, int col, int row,
                                unsigned dim, uint64_t elemWidth,
                                uint32_t gran) {
  uint32_t wrapBits = tm.getDmaBdWrapBits(col, row);
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
    return maxIterations(tm, col, row);
  return (1LL << wrapBits) - 1;
}

/// Whether dimension `d`'s stride fits the BD's step field. The hardware
/// stride is in address granules, biased by one as the field encodes it
/// (getHardwareStridesWraps); the iteration slot (d == 3) shares the field
/// width and additionally admits a zero stride, a re-read.
bool strideFitsStepField(const AIE::AIETargetModel &tm, Operation *forOp,
                         BaseMemRefType bufType, int col, int row,
                         const NdDmaPattern &pattern, unsigned d) {
  SmallVector<int64_t, kNdDmaDims> hwSizes(kNdDmaDims);
  SmallVector<int64_t, kNdDmaDims> hwStrides(kNdDmaDims);
  getHardwareStridesWraps(tm, forOp, bufType, pattern.sizes, pattern.strides,
                          hwSizes, hwStrides);
  uint32_t stepBits = tm.getDmaBdStepBits(col, row);
  return hwStrides[d] <= (1LL << stepBits) - 1;
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
  std::reverse(out.begin(), out.end());
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
  // preserved. The outermost slot must be free and carry no base offset.
  bool outerSlotHasOffset = pattern.offsets[3] != 0 && pattern.strides[3] != 0;
  if (pattern.sizes[3] == 1 && !outerSlotHasOffset) {
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

        auto sub = decomposeRecursive(forOp, bufType, tm, col, row, factored);
        if (succeeded(sub))
          return sub;
      }
    }
  }

  // (2) Order-preserving slicing: only the OUTERMOST active dimension may be
  // split into contiguous index ranges emitted in order. Slicing an inner
  // dimension would interleave the outer iterations and reorder the emitted
  // element stream, so it is not allowed.
  //
  // The chunk is the largest wrap the slot holds, or a single index when the
  // dimension's stride is past the step field: a stride the BD cannot
  // encode is carried in each slice's offset instead, which has no such
  // limit (a column-major weight whose column-block stride is the whole
  // matrix's height, for one). Same bytes, same order, one descriptor per
  // index of that dimension.
  {
    unsigned d = static_cast<unsigned>(outermost);
    int64_t n = pattern.sizes[d];
    int64_t chunkSize =
        maxLegalInputSizeForDim(tm, col, row, d, elemWidth, gran);
    // An offset-bearing singleton cannot be reused by factoring; slice instead.
    if (pattern.sizes[3] == 1 && outerSlotHasOffset && n <= chunkSize)
      chunkSize = 1;
    bool oversizedStride =
        !strideFitsStepField(tm, forOp, bufType, col, row, pattern, d);
    // Peel outer dimensions until an oversized inner stride can be folded.
    for (unsigned i = 0; i <= d; ++i)
      if (pattern.sizes[i] > 1 && pattern.strides[i] > 0 &&
          !strideFitsStepField(tm, forOp, bufType, col, row, pattern, i)) {
        chunkSize = 1;
        break;
      }
    // It fits, so what is illegal is inside it and could not be factored
    // away: slice it into single indices, each slice's outermost dimension
    // then being the next one in.
    if (chunkSize >= n)
      chunkSize = 1;
    if (chunkSize > 0 && chunkSize < n) {
      int64_t numChunks = (n + chunkSize - 1) / chunkSize;
      SmallVector<NdDmaPattern> combined;
      for (int64_t i = 0; i < numChunks; ++i) {
        NdDmaPattern slice = pattern;
        slice.sizes[d] = std::min(chunkSize, n - i * chunkSize);
        slice.offsets[d] = pattern.offsets[d] + i * chunkSize;
        if (slice.sizes[d] == 1 && oversizedStride) {
          int64_t stride = bdGranuleDivisor(elemWidth, gran);
          if (pattern.strides[d] % stride != 0)
            return failure();
          slice.offsets[d] *= pattern.strides[d] / stride;
          slice.strides[d] = stride; // keep the singleton granule-aligned
        }

        auto sub = decomposeRecursive(forOp, bufType, tm, col, row, slice);
        // failed() above already guards this deref; the checker just doesn't
        // associate FailureOr's failed()/succeeded() with its optional base.
        if (failed(sub))
          return failure();
        SmallVector<NdDmaPattern> &subPatterns =
            *sub; // NOLINT(bugprone-unchecked-optional-access)
        combined.append(subPatterns.begin(), subPatterns.end());
      }
      return combined;
    }
  }

  return failure();
}

// The kNdDmaDims-dimension patterns a longer pattern reduces to, in order (see
// decomposeNdDmaPattern). Every dimension past d2 is an iteration dimension:
// one execution per index of them all, outermost slowest.
SmallVector<NdDmaPattern> reduceIterationDims(const NdDmaPattern &pattern) {
  NdDmaPattern piece;
  piece.baseOffset = pattern.baseOffset;
  for (unsigned d = 0; d < 3; ++d) {
    piece.offsets.push_back(pattern.offsets[d]);
    piece.sizes.push_back(pattern.sizes[d]);
    piece.strides.push_back(pattern.strides[d]);
  }
  // The iteration dimensions, innermost first, with a dimension that continues
  // the one inside it merged into it and unit ones dropped. Their offsets move
  // into the base, since merging and peeling both move the dimensions.
  SmallVector<int64_t> sizes, strides;
  for (unsigned d = 3; d < pattern.sizes.size(); ++d) {
    piece.baseOffset += pattern.offsets[d] * pattern.strides[d];
    if (pattern.sizes[d] == 1)
      continue;
    if (!sizes.empty() && pattern.strides[d] == sizes.back() * strides.back()) {
      sizes.back() *= pattern.sizes[d];
      continue;
    }
    sizes.push_back(pattern.sizes[d]);
    strides.push_back(pattern.strides[d]);
  }
  piece.offsets.push_back(0);
  piece.sizes.push_back(sizes.empty() ? 1 : sizes.front());
  piece.strides.push_back(strides.empty() ? 0 : strides.front());

  // Peel the rest, one piece per index, working outward so that each outer
  // dimension repeats all the pieces inside it.
  SmallVector<NdDmaPattern> pieces{piece};
  for (unsigned i = 1; i < sizes.size(); ++i) {
    SmallVector<NdDmaPattern> outer;
    outer.reserve(pieces.size() * sizes[i]);
    for (int64_t index = 0; index < sizes[i]; ++index)
      for (NdDmaPattern inner : pieces) {
        inner.baseOffset += index * strides[i];
        outer.push_back(std::move(inner));
      }
    pieces = std::move(outer);
  }
  return pieces;
}

// A contiguous pattern is lowered as a plain length, so only its iteration
// dimension has a limit left to exceed.
bool contiguousAndFits(const AIE::AIETargetModel &tm, int col, int row,
                       const NdDmaPattern &pattern) {
  return isContiguousTransfer(pattern.sizes, pattern.strides) &&
         pattern.sizes[3] <= maxIterations(tm, col, row);
}

// A contiguous pattern whose iteration dimension is too long, as consecutive
// runs of it, each at most `chunk` long.
SmallVector<NdDmaPattern> sliceIterations(const NdDmaPattern &pattern,
                                          int64_t chunk) {
  SmallVector<NdDmaPattern> slices;
  for (int64_t first = 0; first < pattern.sizes[3]; first += chunk) {
    NdDmaPattern slice = pattern;
    slice.sizes[3] = std::min(chunk, pattern.sizes[3] - first);
    slice.offsets[3] = pattern.offsets[3] + first;
    slices.push_back(std::move(slice));
  }
  return slices;
}

// The 4-dimensional patterns a 4-dimensional one that is neither legal nor
// contiguous and short enough lowers to.
FailureOr<SmallVector<NdDmaPattern>> decompose4d(Operation *forOp,
                                                 BaseMemRefType bufType,
                                                 const AIE::AIETargetModel &tm,
                                                 int col, int row,
                                                 const NdDmaPattern &pattern) {
  if (isContiguousTransfer(pattern.sizes, pattern.strides))
    return sliceIterations(pattern, maxIterations(tm, col, row));
  return decomposeRecursive(forOp, bufType, tm, col, row, pattern);
}

// Drops the dimensions of size one past d0 while there are more than a BD
// holds, innermost first. Such a dimension moves nothing, but where it sits
// between d0 and d2 it keeps an iteration dimension out of the BD, which then
// has to be peeled.
void squeezeUnitDims(NdDmaPattern &pattern) {
  for (unsigned d = 1;
       d < pattern.sizes.size() && pattern.sizes.size() > kNdDmaDims;) {
    if (pattern.sizes[d] != 1) {
      ++d;
      continue;
    }
    pattern.baseOffset += pattern.offsets[d] * pattern.strides[d];
    pattern.offsets.erase(pattern.offsets.begin() + d);
    pattern.sizes.erase(pattern.sizes.begin() + d);
    pattern.strides.erase(pattern.strides.begin() + d);
  }
}

} // namespace

bool AIEX::patternPassesVerification(Operation *forOp,
                                     BaseMemRefType referencedBufType,
                                     const AIE::AIETargetModel &tm, int tileCol,
                                     int tileRow, const NdDmaPattern &pattern) {
  if (pattern.sizes.size() != kNdDmaDims)
    return false;
  SmallVector<int64_t, kNdDmaDims> hwSizes(kNdDmaDims);
  SmallVector<int64_t, kNdDmaDims> hwStrides(kNdDmaDims);
  getHardwareStridesWraps(tm, forOp, referencedBufType, pattern.sizes,
                          pattern.strides, hwSizes, hwStrides);

  ScopedDiagnosticHandler handler(forOp->getContext(),
                                  [](Diagnostic &) { return success(); });
  return succeeded(verifyStridesWraps(forOp, referencedBufType, tileCol,
                                      tileRow, pattern.sizes, pattern.strides,
                                      hwSizes, hwStrides,
                                      /*skipTransformationChecks=*/false));
}

bool AIEX::isDecomposableNdDmaPattern(Operation *forOp,
                                      BaseMemRefType referencedBufType,
                                      const AIE::AIETargetModel &targetModel,
                                      int tileCol, int tileRow,
                                      ArrayRef<int64_t> offsetsInnermostFirst,
                                      ArrayRef<int64_t> sizesInnermostFirst,
                                      ArrayRef<int64_t> stridesInnermostFirst) {
  if (offsetsInnermostFirst.size() != kNdDmaDims ||
      sizesInnermostFirst.size() != kNdDmaDims ||
      stridesInnermostFirst.size() != kNdDmaDims)
    return false;

  if (isContiguousTransfer(sizesInnermostFirst, stridesInnermostFirst))
    return false;

  NdDmaPattern pattern;
  pattern.offsets.assign(offsetsInnermostFirst.begin(),
                         offsetsInnermostFirst.end());
  pattern.sizes.assign(sizesInnermostFirst.begin(), sizesInnermostFirst.end());
  pattern.strides.assign(stridesInnermostFirst.begin(),
                         stridesInnermostFirst.end());

  if (patternPassesVerification(forOp, referencedBufType, targetModel, tileCol,
                                tileRow, pattern))
    return false;

  auto decomposed = decomposeNdDmaPattern(forOp, referencedBufType, pattern,
                                          targetModel, tileCol, tileRow);
  // failed() above already guards this deref; the checker just doesn't
  // associate FailureOr's failed()/succeeded() with its optional base.
  if (failed(decomposed))
    return false;
  SmallVector<NdDmaPattern> &bds =
      *decomposed; // NOLINT(bugprone-unchecked-optional-access)
  return !bds.empty();
}

FailureOr<SmallVector<NdDmaPattern>>
AIEX::decomposeNdDmaPattern(Operation *forOp, BaseMemRefType referencedBufType,
                            const NdDmaPattern &pattern,
                            const AIE::AIETargetModel &targetModel, int tileCol,
                            int tileRow) {
  if (pattern.offsets.size() != pattern.sizes.size() ||
      pattern.strides.size() != pattern.sizes.size() ||
      pattern.sizes.size() < kNdDmaDims)
    return failure();

  if (pattern.sizes.size() > kNdDmaDims) {
    NdDmaPattern squeezed = pattern;
    squeezeUnitDims(squeezed);
    SmallVector<NdDmaPattern> pieces =
        squeezed.sizes.size() > kNdDmaDims
            ? reduceIterationDims(squeezed)
            : SmallVector<NdDmaPattern>{squeezed};
    SmallVector<NdDmaPattern> result;
    for (const NdDmaPattern &piece : pieces) {
      // A piece already legal, or contiguous and so left to the lowering as a
      // plain length, is kept as it is.
      if (contiguousAndFits(targetModel, tileCol, tileRow, piece) ||
          patternPassesVerification(forOp, referencedBufType, targetModel,
                                    tileCol, tileRow, piece)) {
        result.push_back(piece);
        continue;
      }
      auto sub = decompose4d(forOp, referencedBufType, targetModel, tileCol,
                             tileRow, piece);
      if (failed(sub))
        return failure();
      SmallVector<NdDmaPattern> &subPatterns =
          *sub; // NOLINT(bugprone-unchecked-optional-access)
      result.append(subPatterns.begin(), subPatterns.end());
    }
    return result;
  }

  if (contiguousAndFits(targetModel, tileCol, tileRow, pattern))
    return failure();

  if (patternPassesVerification(forOp, referencedBufType, targetModel, tileCol,
                                tileRow, pattern))
    return failure();

  return decompose4d(forOp, referencedBufType, targetModel, tileCol, tileRow,
                     pattern);
}
