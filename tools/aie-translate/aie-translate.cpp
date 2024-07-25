//===- aie-translate.cpp ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Target/LLVMIR/Dialect/All.h"
#include "aie/version.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace aie {
// We redefine the MLIR -> LLVM IR translation to include our AIE intrinsics
// translations.
void registerToLLVMIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-llvmir", "Translate MLIR to LLVMIR",
      [](Operation *op, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule)
          return failure();

        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<DLTIDialect, func::FuncDialect>();
        xilinx::registerAllAIEToLLVMIRTranslations(registry);
        registerAllToLLVMIRTranslations(registry);
      });
}
} // namespace aie

void version_printer(raw_ostream &os) {
  os << "aie-translate " << AIE_GIT_COMMIT << "\n";
}

int main(int argc, char **argv) {
  // NOTE: these are the contents of registerAllTranslations();
  registerFromLLVMIRTranslation();
  registerFromSPIRVTranslation();
  registerToCppTranslation();
  aie::registerToLLVMIRTranslation();
  registerToSPIRVTranslation();

  xilinx::AIE::registerAIETranslations();
  xilinx::aievec::registerAIEVecToCppTranslation();

  llvm::cl::AddExtraVersionPrinter(version_printer);

  return failed(mlirTranslateMain(argc, argv, "AIE Translation Tool"));
}
