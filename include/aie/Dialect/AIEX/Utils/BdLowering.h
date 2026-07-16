//===- BdLowering.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared BD (buffer-descriptor) size/stride encoding used by both the static
// (constant-folded) and dynamic (runtime SSA) shim-NOC DMA lowering paths.
//
// The hardware size/stride computation lives here ONCE as a policy-templated
// algorithm (encodeHardwareStridesWraps) so the constant path and the dynamic
// arith-emitting path cannot drift -- a divergence would silently miscompile a
// descriptor. The constant path instantiates it with ConstStridePolicy (plain
// integer math); the dynamic path uses SsaStridePolicy (emits arith ops).
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEX_UTILS_BDLOWERING_H
#define AIE_DIALECT_AIEX_UTILS_BDLOWERING_H

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <numeric>
#include <tuple>

namespace xilinx::AIE {
class AIETargetModel;
} // namespace xilinx::AIE

namespace xilinx::AIEX {

// Shim-NOC BD hardware field widths (bits), shared by the dynamic verifier and
// the runtime-value guard so both reject the same field overflows.
// TODO: fold into a getDmaBdWrapBits(tileType) accessor on AIETargetModel (also
// covers the widths hardcoded per tile type in verifyStridesWraps).
struct ShimBdFieldWidths {
  static constexpr int64_t kD0WrapBits = 10;
  static constexpr int64_t kD1WrapBits = 10;
  static constexpr int64_t kIterWrapBits = 6;
  static constexpr int64_t d0WrapMax() { return (1 << kD0WrapBits) - 1; }
  static constexpr int64_t d1WrapMax() { return (1 << kD1WrapBits) - 1; }
  static constexpr int64_t iterWrapMax() { return (1 << kIterWrapBits) - 1; }
};

// The address generator transfers whole granules only, so a size or stride is
// realizable iff its byte extent (value * elemWidth) is a granule multiple.
// Dividing through by gcd(elemWidth, gran), that is `value % divisor == 0` with
// divisor = gran / gcd(elemWidth, gran) -- the element-count multiple a size or
// stride must land on (e.g. int8 against a 32-bit granule => divisor 4). Shared
// so the static verifier, the dynamic verifier, and the runtime guard apply one
// definition.
//
// These realizability predicates are header-inline (no AIEX-op dependencies) so
// the AIEX dialect verifier can call them without the AIEX IR library depending
// on AIEXUtils -- which would close a link cycle (AIEXUtils already uses AIEX
// ops).
inline int64_t bdGranuleDivisor(uint64_t elemWidth,
                                uint32_t addressGranularity) {
  return addressGranularity / std::gcd(elemWidth, (uint64_t)addressGranularity);
}

// Whether a constant element count is realizable: value * elemWidth is a whole
// number of granules. Equivalent to value % bdGranuleDivisor(...) == 0.
inline bool isConstMultipleOfGranule(int64_t value, uint64_t elemWidth,
                                     uint32_t addressGranularity) {
  return value * (int64_t)elemWidth % (int64_t)addressGranularity == 0;
}

// Check the constant size/stride operands (innermost-first) of a shim-NOC BD
// for realizability: d0 size and every non-unit stride must be a whole number
// of granules (a unit innermost stride is the exempt contiguous case), and a
// stride must be positive where its size > 1. Runtime operands are skipped
// (guarded at lowering by assert_bd_divisible). Shared by both dynamic paths;
// emits a diagnostic on `op` and fails on the first violation.
inline mlir::LogicalResult
verifyConstBdRealizability(mlir::Operation *op,
                           llvm::ArrayRef<mlir::OpFoldResult> sizes,
                           llvm::ArrayRef<mlir::OpFoldResult> strides,
                           uint64_t elemWidth, uint32_t gran) {
  if (!sizes.empty())
    if (auto d0 = mlir::getConstantIntValue(sizes[0]))
      if (!isConstMultipleOfGranule(*d0, elemWidth, gran))
        return op->emitOpError("d0 size ")
               << *d0 << " elements at " << (elemWidth / 8)
               << " bytes each is not a multiple of the " << (gran / 8)
               << "-byte address-gen granule.";
  for (int i = 0; i < (int)strides.size(); i++) {
    auto s = mlir::getConstantIntValue(strides[i]);
    if (!s)
      continue;
    // A unit innermost stride is the contiguous case: successive elements are
    // packed with no gap, so the transfer is dense and the stride need not land
    // on a granule boundary. Every other stride addresses a strided access and
    // must be granule-aligned.
    if (i == 0 && *s == 1)
      continue;
    if (!isConstMultipleOfGranule(*s, elemWidth, gran))
      return op->emitOpError("stride ")
             << i << " is " << *s << " elements at " << (elemWidth / 8)
             << " bytes each, not a multiple of the " << (gran / 8)
             << "-byte address-gen granule.";
  }
  // A stride must be positive where its size > 1 (it is never applied when
  // size == 1). The d3 iteration dimension is the exception: a zero stride
  // there is the pure-repeat case (the BD wraps every iteration, repeat carried
  // by the queue push), matching verifyStridesWraps' dim-3 `< 0` rule. Lists
  // are innermost-first, so d3 is index 3 (present only for a full 4D
  // descriptor). Runtime strides are trusted (the caller controls them).
  constexpr int kIterDim = 3;
  for (int i = 0; i < (int)sizes.size() && i < (int)strides.size(); i++) {
    auto sz = mlir::getConstantIntValue(sizes[i]);
    auto st = mlir::getConstantIntValue(strides[i]);
    if (!sz || !st || *sz <= 1)
      continue;
    if (i == kIterDim ? *st < 0 : *st < 1)
      return op->emitOpError("stride ")
             << i
             << (i == kIterDim ? " must be non-negative when size > 1."
                               : " must be positive when size > 1.");
  }
  return mlir::success();
}

// Shared hardware size/stride encoder. The BD encodes each dimension's wrap
// ("size") scaled to address-gen granules and step ("stride") biased by -1 (a
// stored 0 means one granule). One algorithm drives both lowerings via a
// Policy: ConstStridePolicy (int64 math) for the static path, SsaStridePolicy
// (arith ops) for runtime operands -- keeping them bit-identical. The Policy
// supplies cst/mul/div/sub and selectGT1/selectGT0/selectLt. Inputs/outputs are
// 4-element arrays in innermost-first order [d0, d1, d2, d3/iter].
template <typename Policy>
void encodeHardwareStridesWraps(Policy &p, uint64_t elemWidth,
                                uint32_t addressGranularity,
                                typename Policy::V inputSizes[4],
                                typename Policy::V inputStrides[4],
                                typename Policy::V sizes[4],
                                typename Policy::V strides[4]) {
  using V = typename Policy::V;
  // Scale an element-count stride into hardware granules and apply the -1 bias:
  //   stride * elemWidth / addressGranularity - 1
  auto biasedStride = [&](V inStride) -> V {
    return p.sub(p.div(p.mul(inStride, elemWidth), addressGranularity), 1);
  };

  // d0_size, d0_stride
  sizes[0] = p.div(p.mul(inputSizes[0], elemWidth), addressGranularity);
  // d0_stride collapses to hardware 0 for a sub-granule stride (byte extent <
  // one granule, i.e. the contiguous unit-stride case) or a wide element; else
  // it is the biased stride. The wide-element test is compile-time; the
  // sub-granule test is a policy select, so the stride may be runtime. A
  // non-unit sub-granule stride is unrealizable and rejected/guarded elsewhere.
  if (elemWidth > addressGranularity) {
    strides[0] = p.cst(0);
  } else {
    strides[0] = p.selectLt(p.mul(inputStrides[0], elemWidth),
                            p.cst((int64_t)addressGranularity), p.cst(0),
                            biasedStride(inputStrides[0]));
  }

  // d1_size, d1_stride / d2_size, d2_stride: stride only matters when size > 1.
  sizes[1] = inputSizes[1];
  strides[1] =
      p.selectGT1(inputSizes[1], biasedStride(inputStrides[1]), p.cst(0));
  sizes[2] = inputSizes[2];
  strides[2] =
      p.selectGT1(inputSizes[2], biasedStride(inputStrides[2]), p.cst(0));

  // iteration_size, iteration_stride. Size is stored biased by -1. The stride
  // must be positive like the others, but a zero-stride "repeat" is encoded by
  // leaving size at 1 (via a positive repeat_count on the queue push) so the BD
  // wraps every iteration and never adds the stride. Hence stride is gated on
  // BOTH size > 1 and inStride > 0.
  sizes[3] = p.selectGT1(inputSizes[3], p.sub(inputSizes[3], 1), p.cst(0));
  strides[3] = p.selectGT1(
      inputSizes[3],
      p.selectGT0(inputStrides[3], biasedStride(inputStrides[3]), p.cst(0)),
      p.cst(0));
}

// Constant (compile-time int64) policy: plain integer arithmetic.
struct ConstStridePolicy {
  using V = int64_t;
  static V cst(int64_t c) { return c; }
  static V mul(V v, int64_t c) { return v * c; }
  static V div(V v, int64_t c) { return v / c; }
  static V sub(V v, int64_t c) { return v - c; }
  static V selectGT1(V cond, V t, V e) { return cond > 1 ? t : e; }
  static V selectGT0(V cond, V t, V e) { return cond > 0 ? t : e; }
  static V selectLt(V a, V b, V t, V e) { return a < b ? t : e; }
};

// SSA (runtime arith) policy: emits i32 arith ops mirroring ConstStridePolicy.
// Every primitive builds an arith op at the policy's insertion point, so the
// innermost stride may be a runtime value like any other dimension.
struct SsaStridePolicy {
  using V = mlir::Value;
  mlir::OpBuilder &builder;
  mlir::Location loc;

  SsaStridePolicy(mlir::OpBuilder &b, mlir::Location l) : builder(b), loc(l) {}

  V cst(int64_t c) const;
  V mul(V v, int64_t c) const;
  V div(V v, int64_t c) const;
  V sub(V v, int64_t c) const;
  V selectGT1(V cond, V t, V e) const;
  V selectGT0(V cond, V t, V e) const;
  V selectLt(V a, V b, V t, V e) const;
};

// Coerce an OpFoldResult (constant attr or SSA value) to an SSA Value of the
// given integer type, materializing an arith.constant / trunc / extui as
// needed.
mlir::Value getAsValue(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::OpFoldResult ofr, mlir::Type intType);

// Build the address-patch `arg_plus` (buffer BYTE offset) as an i32 Value.
// `elementOffsets` (innermost-first) are the per-dim element offsets and
// `strides` their matching strides; the byte offset is
// sum(elementOffsets[i] * strides[i]) * elemWidthBytes + baseByteOffset.
// Any entry may be a runtime Value; a fully-constant set folds to a single
// arith.constant. Shared by the memcpy_nd and dma_task dynamic lowerings so a
// runtime DMA offset flows into the patch instead of being rejected.
mlir::Value buildArgPlusValue(mlir::OpBuilder &builder, mlir::Location loc,
                              llvm::ArrayRef<mlir::OpFoldResult> elementOffsets,
                              llvm::ArrayRef<mlir::OpFoldResult> strides,
                              int64_t elemWidthBytes, int64_t baseByteOffset);

// Pack a set of (value, mask, shift) fields into a single i32 BD word via
// arith and/shl/or. mask == 0xFFFFFFFF skips the AND; shift == 0 skips the SHL.
mlir::Value
buildBdWord(mlir::OpBuilder &builder, mlir::Location loc,
            llvm::ArrayRef<std::tuple<mlir::Value, uint32_t, uint32_t>> fields);

// Emit the per-word `npu.write32` overrides carrying a shim-NOC BD's runtime
// sizes/strides on top of a zero-template blockwrite, plus assert_bd_field /
// assert_bd_divisible guards for runtime values in narrow or sub-granule
// fields. Shared by the dma_memcpy_nd and dma_task paths.
// `mixedSizes`/`mixedStrides` are outermost-first (d3..d0). `bufLenOverride`,
// if non-null, sets buffer_length (dma_task's runtime len); else it is the
// d0*d1*d2 size-product. `repeatCountOut` receives the biased hardware
// outer-dim (d3) value for the caller's queue push.
mlir::LogicalResult emitDynamicShimBdWordOverrides(
    mlir::OpBuilder &builder, mlir::Location loc,
    const xilinx::AIE::AIETargetModel &targetModel, int tileCol, int tileRow,
    uint32_t bdId, llvm::ArrayRef<mlir::OpFoldResult> mixedSizes,
    llvm::ArrayRef<mlir::OpFoldResult> mixedStrides, uint64_t elemWidth,
    uint32_t burstLength, mlir::Value bufLenOverride,
    mlir::Value &repeatCountOut);

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_BDLOWERING_H
