// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef XILINX_INITALLDIALECTS_H_
#define XILINX_INITALLDIALECTS_H_

#include "aie/Dialect/ADF/ADFDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "mlir/IR/Dialect.h"

namespace xilinx {

// Add all the MLIR dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    ADF::ADFDialect,
    aievec::AIEVecDialect,
    // todo: initial AIE dialect here
    // AIE::AIEDialect
  >();
  // clang-format on
}

} // namespace xilinx

#endif // XILINX_INITALLDIALECTS_H_
