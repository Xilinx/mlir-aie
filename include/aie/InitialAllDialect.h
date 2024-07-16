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
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Dialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/XLLVM/XLLVMDialect.h"

#include "mlir/IR/Dialect.h"

namespace xilinx {

// Add all the MLIR dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    ADF::ADFDialect,
    AIE::AIEDialect,
    aievec::AIEVecDialect,
    aievec_aie1::AIEVecAIE1Dialect,
    AIEX::AIEXDialect,
    xllvm::XLLVMDialect
  >();
  // clang-format on
}

} // namespace xilinx

#endif // XILINX_INITALLDIALECTS_H_
