//===- DialectExtension.h - AIEVec transform dialect extension --*- C++ -*-===//
//
// Copyright (c) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
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
