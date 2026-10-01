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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <numeric>
#include <tuple>

namespace xilinx::AIE {
class AIETargetModel;
} // namespace xilinx::AIE

namespace xilinx::AIEX {

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
// (buildShimBdWords guards them on the host). Shared by both dynamic paths;
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
  // descriptor).
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

// SSA (runtime arith) policy: emits i64 arith ops mirroring ConstStridePolicy.
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

// Require `cond` to hold when the runtime sequence runs. A condition that
// folds to true emits nothing and one that folds to false is a compile-time
// error at `loc`; otherwise a cf.assert carries `message` to the TXN builder,
// which refuses the dispatch when it fails.
mlir::LogicalResult emitRuntimeCheck(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value cond,
                                     const llvm::Twine &message);

// Erase the arith ops under `root` left unused once the runtime BD words and
// guards built with these helpers have folded.
void eraseDeadArith(mlir::Operation *root);

// Coerce an OpFoldResult (constant attr or SSA value) to an SSA Value of the
// given integer type, materializing an arith.constant / trunc / extui as
// needed.
mlir::Value getAsValue(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::OpFoldResult ofr, mlir::Type intType);

// An integer or index OpFoldResult as an i64 Value, zero-extended (a negative
// value reads as a huge one the guards reject). An operand wider than 64 bits
// gets a host-side check that it fits.
mlir::FailureOr<mlir::Value> getAsI64(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      mlir::OpFoldResult ofr);

// Build the address-patch `arg_plus` (buffer BYTE offset):
// sum(elementOffsets[i] * strides[i]) * elemWidthBytes + baseByteOffset, where
// any entry may be runtime. A fully-constant set folds to one i32 constant (i64
// when it does not fit), byte-identical to the static path; a runtime one is
// i64 arith with a host-side check that it is `granuleBytes`-aligned.
mlir::FailureOr<mlir::Value>
buildArgPlusValue(mlir::OpBuilder &builder, mlir::Location loc,
                  llvm::ArrayRef<mlir::OpFoldResult> elementOffsets,
                  llvm::ArrayRef<mlir::OpFoldResult> strides,
                  int64_t elemWidthBytes, int64_t baseByteOffset,
                  uint32_t granuleBytes);

// Require a runtime BD walk to stay inside its host buffer: the furthest
// element it touches, sum(offsets[k] * offsetStrides[k]) +
// sum((sizes[i] - 1) * strides[i]), must lie within `hostBufferType` past
// `baseByteOffset`. An unknown (dynamic) shape bounds the walk at 2^32
// elements, which still rejects a negative offset.
mlir::LogicalResult guardWithinHostBuffer(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::BaseMemRefType hostBufferType, int64_t baseByteOffset,
    int64_t elemWidthBytes, llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> offsetStrides,
    llvm::ArrayRef<mlir::OpFoldResult> sizes,
    llvm::ArrayRef<mlir::OpFoldResult> strides);

// Pack a set of (value, mask, shift) fields into a single i32 BD word via
// arith and/shl/or. mask == 0xFFFFFFFF skips the AND; shift == 0 skips the SHL.
mlir::Value
buildBdWord(mlir::OpBuilder &builder, mlir::Location loc,
            llvm::ArrayRef<std::tuple<mlir::Value, uint32_t, uint32_t>> fields);

// The base register address of a BD block, `getDmaBdAddress(col,row,bd_id)`, as
// an i32 Value. The address is linear in bd_id (base + bd_id*bdStride), so a
// constant bdId folds to the exact literal the static path uses, while a
// runtime bdId (dynamic free-list pool) is emitted as arith. Shared by the
// BD-word encoder and the descriptor / address-patch lowering so every register
// touched for one BD is computed from the same base.
mlir::Value getBdRegisterBase(mlir::OpBuilder &builder, mlir::Location loc,
                              const xilinx::AIE::AIETargetModel &targetModel,
                              int tileCol, int tileRow,
                              mlir::OpFoldResult bdId);

// Always-constant BD fields (locks, packet, next_bd), gathered by the caller
// from the DMA structure and baked into the template words.
struct BdTemplateFields {
  uint32_t use_next_bd = 0, next_bd_id = 0;
  int32_t enable_packet = 0, packet_id = 0, packet_type = 0;
  int32_t out_of_order_id = 0;
  int32_t lock_rel_val = 0, lock_rel_id = 0;
  int32_t lock_acq_enable = 0, lock_acq_val = 0, lock_acq_id = 0;
};

// Build a shim-NOC BD's full 8-word register block as i32 SSA values, for the
// dynamic lowering shared by dma_memcpy_nd and dma_task. Every word is written
// (unset slots are zero), so a BD slot reused from the runtime pool cannot
// inherit a stale field.
//
// `mixedSizes`/`mixedStrides` are outermost-first (d3..d0), matching
// NpuDmaMemcpyNdOp::getMixedSizes and AIE::DMABDOp::getMixedSizes.
// buffer_length (word 0) is the d0*d1*d2 extent in granules. `lenElems`, if
// set, is a dma_task's explicit transfer length, which must agree with it.
// Every size and stride gets a host-side check (emitRuntimeCheck) that it fits
// its BD field and is granule-realizable, so a value the field would truncate
// refuses the dispatch instead. `repeatCountOut` receives the outer-dim
// hardware repeat for the caller's queue push.
//
// The words do not depend on the BD id -- only the register ADDRESS does, and
// that is the caller's `getBdRegisterBase` + `npu.blockwrite_values` -- so one
// routine serves both a pinned bd_id and one drawn from the runtime pool.
mlir::LogicalResult
buildShimBdWords(mlir::OpBuilder &builder, mlir::Location loc,
                 const xilinx::AIE::AIETargetModel &targetModel,
                 const BdTemplateFields &fields,
                 llvm::ArrayRef<mlir::OpFoldResult> mixedSizes,
                 llvm::ArrayRef<mlir::OpFoldResult> mixedStrides,
                 uint64_t elemWidth, uint32_t burstLength, uint32_t axcache,
                 mlir::OpFoldResult lenElems, mlir::Value &repeatCountOut,
                 llvm::SmallVectorImpl<mlir::Value> &wordsOut);

} // namespace xilinx::AIEX

#endif // AIE_DIALECT_AIEX_UTILS_BDLOWERING_H
