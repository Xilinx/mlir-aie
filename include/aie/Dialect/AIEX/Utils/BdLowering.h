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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <tuple>

namespace xilinx::AIE {
class AIETargetModel;
} // namespace xilinx::AIE

namespace xilinx::AIEX {

// Shim-NOC BD hardware field widths (bits). The d0/d1 wrap and iteration wrap
// bound the encoded sizes that can be programmed without silent truncation; the
// dynamic verifier and the runtime-value guard share these so the constant and
// runtime paths reject exactly the same overflows.
//
// TODO: these mirror the XAIEMLGBL_NOC_MODULE_* register widths hardcoded in
// verifyStridesWraps for every tile type; a getDmaBdWrapBits(tileType) accessor
// on AIETargetModel would let all sites (and non-shim tiles) share one source.
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
// definition (see issue #2566).
int64_t bdGranuleDivisor(uint64_t elemWidth, uint32_t addressGranularity);

// Whether a constant element count is realizable: value * elemWidth is a whole
// number of granules. Equivalent to value % bdGranuleDivisor(...) == 0.
bool isConstMultipleOfGranule(int64_t value, uint64_t elemWidth,
                              uint32_t addressGranularity);

// Check the CONSTANT size/stride operands of a shim-NOC BD for hardware
// realizability, shared by the dma_memcpy_nd verifier and the dma_task lowering
// so both dynamic paths reject the same unrealizable constants. Runtime
// operands are skipped here (guarded at lowering by assert_bd_divisible). Sizes
// and strides are innermost-first (d0 first) and of equal length. Checks: d0
// size and every non-unit stride must be a whole number of granules; the
// innermost stride may be 1 (contiguous); a stride must be positive where its
// size > 1. Emits a diagnostic on `op` and returns failure on the first
// violation.
mlir::LogicalResult verifyConstBdRealizability(
    mlir::Operation *op, llvm::ArrayRef<mlir::OpFoldResult> sizesInnermostFirst,
    llvm::ArrayRef<mlir::OpFoldResult> stridesInnermostFirst,
    uint64_t elemWidth, uint32_t addressGranularity);

// Shared hardware size/stride encoder.
//
// The AIE DMA buffer descriptor encodes each dimension's wrap ("size") and step
// ("stride") in hardware units: sizes count elements scaled to the address-gen
// granularity, and strides are stored biased by -1 (so a stored 0 means a step
// of one granule). The exact same arithmetic must produce the constant-folded
// values used by the static lowering AND, in the dynamic path, an equivalent
// chain of `arith` ops over runtime SSA operands.
//
// A Policy provides a value type `V` and: cst(int64), mul/div/sub over a V and
// an int64 constant, and the selects selectGT1/selectGT0/selectLt(a, b, then,
// els). For the constant policy these are plain integer ops and C++ ternaries;
// for the SSA policy they emit arith ops, so every dimension (including the
// innermost) may be a runtime value. Inputs/outputs are 4-element arrays in
// innermost-first order [d0, d1, d2, d3/iter].
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
  // The innermost stride has a collapse-to-zero case: a sub-granule stride (its
  // byte extent < one granule -- the contiguous unit-stride case) or an element
  // wider than a granule encodes as hardware stride 0. Otherwise it is the
  // normal biased stride. The wide-element test is purely compile-time; the
  // sub-granule test is a policy select over the (possibly runtime) stride, so
  // d0 is handled exactly like d1/d2/d3 below -- no compile-time constraint on
  // the innermost stride. (Realizability of a non-unit sub-granule stride, e.g.
  // int8 stride 2, is enforced separately: constants by the verifier, runtime
  // by an assert_bd_divisible guard with the unit-stride exemption.)
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

// Constant (compile-time int64) policy: the original static arithmetic.
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

// Pack a set of (value, mask, shift) fields into a single i32 BD word via
// arith and/shl/or. mask == 0xFFFFFFFF skips the AND; shift == 0 skips the SHL.
mlir::Value
buildBdWord(mlir::OpBuilder &builder, mlir::Location loc,
            llvm::ArrayRef<std::tuple<mlir::Value, uint32_t, uint32_t>> fields);

// Emit the per-word `npu.write32` overrides that carry a shim-NOC BD's runtime
// sizes/strides, on top of a zero-template blockwrite whose size/stride words
// were left at 0. Shared by the dma_memcpy_nd and dma_task dynamic lowering
// paths, which converge on the identical shim BD-word layout (words 0/3/4/5/6);
// only the descriptor template (locks, next_bd, packet) and the queue push
// differ, and those stay with each caller.
//
// `mixedSizes`/`mixedStrides` are outermost-first (d3..d0), matching
// NpuDmaMemcpyNdOp::getMixedSizes and AIE::DMABDOp::getMixedSizes.
// `bufLenOverride`, if non-null, is written verbatim into buffer_length
// (word 0) -- dma_task passes its runtime `len`; dma_memcpy_nd passes null, so
// buffer_length is computed as the d0*d1*d2 hardware-unit size-product. Emits
// `npu.assert_bd_field` guards for runtime values landing in narrow fields
// (d0/d1 wrap 10-bit in ND mode, iteration wrap 6-bit always). The op verifier
// is expected to have enforced the supported scope (shim NOC, innermost stride
// == 1) already.
//
// On success `repeatCountOut` receives the hardware iteration/repeat value for
// the outer (d3) dimension, which the caller feeds to its queue push (this is
// the biased hw value, matching the static path's `repeat_count = sizes[3]`).
mlir::LogicalResult emitDynamicShimBdWordOverrides(
    mlir::OpBuilder &builder, mlir::Location loc,
    const xilinx::AIE::AIETargetModel &targetModel, int tileCol, int tileRow,
    uint32_t bdId, llvm::ArrayRef<mlir::OpFoldResult> mixedSizes,
    llvm::ArrayRef<mlir::OpFoldResult> mixedStrides, uint64_t elemWidth,
    uint32_t burstLength, mlir::Value bufLenOverride,
    mlir::Value &repeatCountOut);

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_BDLOWERING_H
