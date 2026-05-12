//===- AIEUtils.h -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"

using namespace mlir;

namespace xilinx {
namespace AIEX {

memref::GlobalOp getOrCreateDataMemref(OpBuilder &builder, AIE::DeviceOp dev,
                                       mlir::Location loc,
                                       ArrayRef<uint32_t> words);

// Result of tracing through subview/cast operations to a block argument for
// traceSubviewToBlockArgument function.
struct SubviewTraceResult {
  BlockArgument rootArg;
  int64_t offsetInBytes;
};

// Trace through memref.subview, memref.cast, and memref.reinterpret_cast
// operations until the referenced SSA value is a block argument.
//
// Returns the root block argument and cumulative byte offset, or std::nullopt
// if the chain doesn't lead to a block argument or contains unsupported ops.
//
// This function checks that all subviews remain static and contiguous.
std::optional<SubviewTraceResult> traceSubviewToBlockArgument(Value value);

// Emit an `aiex.npu.update_from_scratchpad` op that adds the runtime offset
// (held in the scratchpad slot referenced by `bdOp`'s `offset_parameter` /
// `offset_state_table_idx` attributes, multiplied by the element size of
// `bufType`) into the BD address register at `registerAddr`.
//
// `bdOp` must carry both the `offset_parameter` (FlatSymbolRefAttr pointing at
// an `aiex.parameter`) and `offset_state_table_idx` (IntegerAttr, set by
// `--aie-lower-parameters`) attributes. The referenced parameter must have
// type `i32`.
LogicalResult emitUpdateBdAddressFromOffsetParameter(OpBuilder &builder,
                                                     Operation *bdOp,
                                                     BaseMemRefType bufType,
                                                     uint64_t registerAddr);
}
} // namespace xilinx