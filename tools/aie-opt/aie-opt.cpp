//===- aie-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "aie/CIR/CIRToAIEPasses.h"
#include "aie/Conversion/Passes.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEVec/Analysis/Passes.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEVec/TransformOps/DialectExtension.h"
#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/InitialAllDialect.h"
#include "aie/version.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Passes.h"

void version_printer(llvm::raw_ostream &os) {
  os << "aie-opt " << AIE_GIT_COMMIT << "\n";
}

int main(int argc, char **argv) {

  mlir::registerAllPasses();
  xilinx::registerConversionPasses();
  xilinx::AIE::registerAIEPasses();
  xilinx::AIEX::registerAIEXPasses();
  xilinx::aievec::registerAIEVecAnalysisPasses();
  xilinx::aievec::registerAIEVecPasses();
  xilinx::aievec::registerAIEVecPipelines();
  xilinx::AIE::CIR::registerCIRToAIEPasses();

  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  xilinx::registerAllDialects(registry);

  registerAllExtensions(registry);

  xilinx::aievec::registerTransformDialectExtension(registry);

  llvm::cl::AddExtraVersionPrinter(version_printer);

  // ClangIR dialect
  registry.insert<mlir::cir::CIRDialect>();

  // ClangIR-specific passes
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return ::cir::createConvertMLIRToLLVMPass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRSimplifyPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createSCFPreparePass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return ::cir::createConvertCIRToMLIRPass();
  });

  mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions> pipeline(
      "cir-to-llvm", "", [](mlir::OpPassManager &pm) {
        ::cir::direct::populateCIRToLLVMPasses(pm);
      });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createFlattenCFGPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createReconcileUnrealizedCastsPass();
  });

  mlir::registerTransformsPasses();

  return failed(
      MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry));
}
