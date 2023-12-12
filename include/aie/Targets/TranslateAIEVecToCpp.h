//===- TranslateAIEVecToCpp.h - C++ emitter for AIE vector code -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// This file defines helpers to emit C++ code for AIE vector dialect.
//===----------------------------------------------------------------------===//

#ifndef TARGET_TRANSLATEAIEVECTOCPP_H
#define TARGET_TRANSLATEAIEVECTOCPP_H

#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"

namespace xilinx {
namespace aievec {

/// Translates the AIE vector dialect MLIR to C++ code.
mlir::LogicalResult translateAIEVecToCpp(mlir::Operation *op, bool aieml,
                                         mlir::raw_ostream &os);

} // namespace aievec
} // namespace xilinx

#endif // TARGET_TRANSLATEAIEVECTOCPP_H
