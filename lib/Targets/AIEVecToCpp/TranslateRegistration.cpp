//===- TranslateRegistration.cpp - Register translation ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "TranslateAIEVecToCpp.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace xilinx {
namespace aievec {

//===----------------------------------------------------------------------===//
// AIEVec to Cpp translation registration
//===----------------------------------------------------------------------===//

void registerAIEVecToCppTranslation() {
  TranslateFromMLIRRegistration reg(
      "aievec-to-cpp",
      [](ModuleOp module, raw_ostream &output) {
        return aievec::translateAIEVecToCpp(module, output);
      },
      [](DialectRegistry &registry) {
        // clang-format off
        registry.insert<arith::ArithmeticDialect,
                        emitc::EmitCDialect,
                        math::MathDialect,
                        memref::MemRefDialect,
                        func::FuncDialect,
                        cf::ControlFlowDialect,
                        scf::SCFDialect,
                        vector::VectorDialect,
                        xilinx::aievec::AIEVecDialect>();
        // clang-format on
      });
}

} // namespace aievec
} // namespace xilinx
