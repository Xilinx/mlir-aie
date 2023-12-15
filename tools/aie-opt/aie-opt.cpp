//===- aie-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "aie/Conversion/Passes.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEVec/Analysis/Passes.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEVec/TransformOps/DialectExtension.h"
#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/InitialAllDialect.h"

using namespace llvm;
using namespace mlir;

namespace mlir::test {
void registerTestTransformDialectEraseSchedulePass();
void registerTestTransformDialectInterpreterPass();
} // namespace mlir::test

namespace test {
void registerTestTransformDialectExtension(DialectRegistry &);
} // namespace test

int main(int argc, char **argv) {

  registerAllPasses();
  xilinx::registerConversionPasses();
  xilinx::AIE::registerAIEPasses();
  xilinx::AIEX::registerAIEXPasses();
  xilinx::aievec::registerAIEVecAnalysisPasses();
  xilinx::aievec::registerAIEVecPasses();
  xilinx::aievec::registerAIEVecPipelines();

  mlir::test::registerTestTransformDialectEraseSchedulePass();
  mlir::test::registerTestTransformDialectInterpreterPass();

  DialectRegistry registry;
  registerAllDialects(registry);
  xilinx::registerAllDialects(registry);

  registerAllExtensions(registry);

  xilinx::aievec::registerTransformDialectExtension(registry);
  ::test::registerTestTransformDialectExtension(registry);

  return failed(
      MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry));
}
