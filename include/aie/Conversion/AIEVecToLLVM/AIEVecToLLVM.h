//===- AIEVecToLLVM.h - AIEVec to LLVM dialect conversion -------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_AIEVECTOLLVM_AIEVECTOLLVM_H
#define AIE_CONVERSION_AIEVECTOLLVM_AIEVECTOLLVM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class ModuleOp;
} // namespace mlir

namespace xilinx {
namespace aievec {

enum class Aie2Fp32Emulation : uint32_t;

void populateAIEVecToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    Aie2Fp32Emulation aie2Fp32EmulationOption, llvm::StringRef aieTarget);

void populateAIEVecToLLVMCommonConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

void populateAIEVecToLLVMAIE2ConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

void populateAIEVecToLLVMAIE2pConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEVecToLLVMPass();
} // namespace aievec

// Forward declare options struct from generated code (in xilinx:: namespace)
struct ConvertAIEVecToLLVMOptions;

namespace aievec {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertAIEVecToLLVMPass(
    const xilinx::ConvertAIEVecToLLVMOptions &options);
} // namespace aievec
} // namespace xilinx

#endif // AIE_CONVERSION_AIEVECTOLLVM_AIEVECTOLLVM_H
