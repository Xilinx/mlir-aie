//===- DialectExtension.h - AIEVec transform dialect extension --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
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
