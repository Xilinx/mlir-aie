//===- BdLowering.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for emitting NPU buffer-descriptor (BD) words from a mix of
// SSA Values and compile-time constants. These are used by the dynamic
// (SSA-operand) lowering paths of both `aiex.npu.dma_memcpy_nd` and the
// lower-level `aie.dma_bd` (within `aiex.dma_configure_task`).
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEX_UTILS_BDLOWERING_H
#define AIE_DIALECT_AIEX_UTILS_BDLOWERING_H

#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <tuple>

namespace xilinx {
namespace AIEX {

/// Get an OpFoldResult as an SSA Value of type intType, creating a constant
/// if needed. If the SSA value has a different width, truncate or extend it.
mlir::Value getAsValue(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::OpFoldResult ofr, mlir::Type intType);

/// Build a BD word from a list of (value, mask, shift) tuples using arith ops.
/// word = (field1 & mask1) << shift1 | (field2 & mask2) << shift2 | ...
mlir::Value
buildBdWord(mlir::OpBuilder &builder, mlir::Location loc,
            llvm::ArrayRef<std::tuple<mlir::Value, uint32_t, uint32_t>> fields);

/// Hardware-encoded BD fields, computed dynamically (as SSA arith chains)
/// from the user-visible `mixed{Sizes,Strides}Rev` lists. All values are i32.
///
/// `mixedSizesRev` and `mixedStridesRev` are 4-element lists ordered
/// innermost-first (`[d0, d1, d2, d3/iter]`), matching the convention used
/// inside `npu.dma_memcpy_nd` and `aie.dma_bd` after reversal.
struct HwBdEncoding {
  mlir::Value d0Size;
  mlir::Value d0Stride;
  mlir::Value d1Size;
  mlir::Value d1Stride;
  mlir::Value d2Size; // valid only on memtile; on shim NOC always 0
  mlir::Value d2Stride;
  mlir::Value iterSize;
  mlir::Value iterStride;
  mlir::Value bufLen;      // d0Size * d1Size * d2Size, in addr-gen units
  mlir::Value repeatCount; // for queue push (max(inSize3 - 1, 0))
};

/// Emit the arith chain that converts the user-supplied sizes/strides into
/// hardware-encoded BD fields. `bufType` provides the element bit-width and
/// is used together with the target model's address-gen granularity to
/// rescale d0 and stride fields. Currently dimensioned for shim-NOC tiles
/// (matches the existing `npu.dma_memcpy_nd` dynamic path's restriction).
HwBdEncoding
emitDynamicHwBdEncoding(mlir::OpBuilder &builder, mlir::Location loc,
                        const AIE::AIETargetModel &targetModel,
                        mlir::BaseMemRefType bufType,
                        llvm::ArrayRef<mlir::OpFoldResult> mixedSizesRev,
                        llvm::ArrayRef<mlir::OpFoldResult> mixedStridesRev);

/// Compile-time-constant counterpart of HwBdEncoding: the hardware BD field
/// values for the constant subset of a (possibly partially-dynamic) transfer.
/// Each field is the actual hardware value when its inputs are constant, or 0
/// as a placeholder when any contributing input is an SSA value (the dynamic
/// lowering then overrides that BD word via npu.write32). All values are in
/// hardware (address-gen) units, matching emitDynamicHwBdEncoding.
struct StaticBdPlaceholders {
  int64_t d0Size = 0;
  int64_t d0Stride = 0;
  int64_t d1Size = 0;
  int64_t d1Stride = 0;
  int64_t d2Stride = 0;
  int64_t iterSize = 0;
  int64_t iterStride = 0;
  int64_t bufLen = 0; // d0Size * d1Size * d2Size when all three are constant
};

/// Compute the static placeholder BD field values for the constant subset of a
/// transfer. `mixedSizesRev`/`mixedStridesRev` are innermost-first
/// (`[d0, d1, d2, iter]`). Shared by both dynamic lowering paths
/// (`npu.dma_memcpy_nd` and `aie.dma_bd`).
StaticBdPlaceholders
computeStaticBdPlaceholders(llvm::ArrayRef<mlir::OpFoldResult> mixedSizesRev,
                            llvm::ArrayRef<mlir::OpFoldResult> mixedStridesRev,
                            uint64_t elemWidth, uint32_t addrGran);

} // namespace AIEX
} // namespace xilinx

#endif // AIE_DIALECT_AIEX_UTILS_BDLOWERING_H
