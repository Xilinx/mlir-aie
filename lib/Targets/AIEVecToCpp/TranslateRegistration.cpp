//===- TranslateRegistration.cpp - Register translation ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Targets/AIETargets.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace xilinx {
namespace aievec {

//===----------------------------------------------------------------------===//
// AIEVec to Cpp translation registration
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool> AIEML("aieml", llvm::cl::desc("AI Engine-ML"),
                                 llvm::cl::init(false));

void registerAIEVecToCppTranslation() {
  TranslateFromMLIRRegistration reg(
      "aievec-to-cpp", "Translate AIEVecDialect dialect to C++",
      [](ModuleOp module, raw_ostream &output) {
        return aievec::translateAIEVecToCpp(module, AIEML.getValue(), output);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithDialect,
                        emitc::EmitCDialect,
                        LLVM::LLVMDialect,
                        math::MathDialect,
                        memref::MemRefDialect,
                        func::FuncDialect,
                        cf::ControlFlowDialect,
                        DLTIDialect,
                        scf::SCFDialect,
                        vector::VectorDialect,
                        xilinx::aievec::AIEVecDialect>();
        // clang-format on
      });
}

} // namespace aievec
} // namespace xilinx
