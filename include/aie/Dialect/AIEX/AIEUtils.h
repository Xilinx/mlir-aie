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

using namespace mlir;

namespace xilinx {
namespace AIEX {

memref::GlobalOp getOrCreateDataMemref(OpBuilder &builder, AIE::DeviceOp dev,
                                       mlir::Location loc,
                                       ArrayRef<uint32_t> words);

}
} // namespace xilinx