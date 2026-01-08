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
// This function supports checks that all subviews remain static and contiguous.
std::optional<SubviewTraceResult> traceSubviewToBlockArgument(Value value);
}
} // namespace xilinx