//===- AIEXToEmitC.h - AIEX to EmitC conversion ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_AIEXTO_EMITC_H
#define AIE_CONVERSION_AIEXTO_EMITC_H

#include <memory>

namespace mlir {
class Pass;
template <typename T> class OperationPass;
} // namespace mlir

namespace xilinx {
namespace AIE {
class RuntimeSequenceOp;
} // namespace AIE

namespace AIEX {

std::unique_ptr<mlir::OperationPass<AIE::RuntimeSequenceOp>>
createConvertAIEXToEmitCPass();

} // namespace AIEX
} // namespace xilinx

#endif // AIE_CONVERSION_AIEXTO_EMITC_H
