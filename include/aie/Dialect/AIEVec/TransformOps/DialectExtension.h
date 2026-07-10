//===- DialectExtension.h - AIEVec transform dialect extension --*- C++ -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_TRANSFORMOPS_DIALECTEXTENSION_H
#define AIE_DIALECT_AIEVEC_TRANSFORMOPS_DIALECTEXTENSION_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace xilinx::aievec {

void registerTransformDialectExtension(mlir::DialectRegistry &registry);

} // namespace xilinx::aievec

#endif
