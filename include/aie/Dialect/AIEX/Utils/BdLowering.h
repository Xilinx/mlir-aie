//===- BdLowering.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
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
// an int64 constant, selectGT1/selectGT0(cond, then, els), and
// d0StrideIsZero(inStride0, elemWidth, gran) deciding the innermost special
// case. For the constant policy these are plain integer ops and a C++ ternary;
// for the SSA policy they emit arith ops. Inputs/outputs are 4-element arrays
// in innermost-first order [d0, d1, d2, d3/iter].
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
  // The innermost stride is the sub-granularity / wide-element special case.
  // The static path folds this from constants; the dynamic path requires
  // inputStrides[0] == 1 (verified), so in both domains d0_stride resolves to a
  // compile-time constant and the branch is decided by the policy on constants.
  if (p.d0StrideIsZero(inputStrides[0], elemWidth, addressGranularity)) {
    // First reason: the hardware cannot transfer less than addressGranularity
    // bits at a time, but the user may express a contiguous transfer of
    // multiple elements with a stride smaller than addressGranularity. Setting
    // the stride to 1 (encoded in hardware as 0) allows such transfers.
    //   verify: inStride0*elemWidth < gran  iff  inSize0*elemWidth > gran.
    // Second reason: when elemWidth > gran, all bytes must be copied, so the
    // stride must be 1 (encoded as 0).
    //   verify: inStride0*elemWidth % gran == 0
    //           && inStride0 == 1 if elemWidth > gran.
    // This forbids a stride > 1 for elemWidths bigger than gran even when a
    // multiple of it; such transfers must use an additional dimension.
    strides[0] = p.cst(0);
  } else {
    strides[0] = biasedStride(inputStrides[0]);
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
  static bool d0StrideIsZero(V inStride0, uint64_t elemWidth, uint32_t gran) {
    return inStride0 * (int64_t)elemWidth < (int64_t)gran || elemWidth > gran;
  }
};

// SSA (runtime arith) policy: emits i32 arith ops mirroring ConstStridePolicy.
// The innermost-stride special case (d0StrideIsZero) is still resolved on
// compile-time constants: the dynamic path requires inputStrides[0] == 1 and
// elemWidth/gran are always constants, so no runtime branch is ever needed for
// it. Every other primitive builds an arith op at the policy's insertion point.
struct SsaStridePolicy {
  using V = mlir::Value;
  mlir::OpBuilder &builder;
  mlir::Location loc;
  // The compile-time innermost stride (== 1 on the dynamic path) so the d0
  // special case matches the constant policy exactly.
  int64_t constInStride0;

  SsaStridePolicy(mlir::OpBuilder &b, mlir::Location l, int64_t inStride0)
      : builder(b), loc(l), constInStride0(inStride0) {}

  V cst(int64_t c) const;
  V mul(V v, int64_t c) const;
  V div(V v, int64_t c) const;
  V sub(V v, int64_t c) const;
  V selectGT1(V cond, V t, V e) const;
  V selectGT0(V cond, V t, V e) const;
  bool d0StrideIsZero(V /*inStride0*/, uint64_t elemWidth,
                      uint32_t gran) const {
    return constInStride0 * (int64_t)elemWidth < (int64_t)gran ||
           elemWidth > gran;
  }
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
// `mixedSizes`/`mixedStrides` are innermost-first (d0..d3). `bufLenOverride`,
// if non-null, is written verbatim into buffer_length (word 0) -- dma_task
// passes its runtime `len`; dma_memcpy_nd passes null, so buffer_length is
// computed as the d0*d1*d2 hardware-unit size-product. Emits
// `npu.assert_bd_field` guards for runtime values landing in narrow fields
// (d0/d1 wrap 10-bit in ND mode, iteration wrap 6-bit always). The op verifier
// is expected to have enforced the supported scope (shim NOC, innermost stride
// == 1) already.
mlir::LogicalResult emitDynamicShimBdWordOverrides(
    mlir::OpBuilder &builder, mlir::Location loc,
    const xilinx::AIE::AIETargetModel &targetModel, int tileCol, int tileRow,
    uint32_t bdId, llvm::ArrayRef<mlir::OpFoldResult> mixedSizes,
    llvm::ArrayRef<mlir::OpFoldResult> mixedStrides, uint64_t elemWidth,
    uint32_t burstLength, mlir::Value bufLenOverride);

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_BDLOWERING_H
