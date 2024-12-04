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
#ifdef CLANGIR_MLIR_FRONTEND
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Dialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/LowerToMLIR.h"

namespace cir::direct {
extern void registerCIRDialectTranslation(mlir::DialectRegistry &registry);
} // namespace cir::direct
#endif

namespace {
#ifdef CLANGIR_MLIR_FRONTEND
// TODO refactor clang/tools/cir-translate/cir-translate.cpp to avoid the
// following copy-paste
void registerToLLVMTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "cir-to-llvmir", "Translate CIR to LLVMIR",
      [](mlir::Operation *op, mlir::raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = cir::direct::lowerDirectlyFromCIRToLLVMIR(
            llvm::dyn_cast<mlir::ModuleOp>(op), llvmContext);
        if (!llvmModule)
          return mlir::failure();
        llvmModule->print(output, nullptr);
        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::DLTIDialect, mlir::func::FuncDialect>();
        mlir::registerAllToLLVMIRTranslations(registry);
        cir::direct::registerCIRDialectTranslation(registry);
      });
}
#endif

// We redefine the MLIR -> LLVM IR translation to include CIR & AIE intrinsics
// translations.
// The problem by picking the same "mlir-to-llvmir" name is that it is no longer
// possible to combine this with other standard MLIR translations which
// already define "mlir-to-llvmir"
void registerToLLVMIRTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "mlir-to-llvmir", "Translate MLIR to LLVMIR",
      [](mlir::Operation *op, mlir::raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule)
          return mlir::failure();

        llvmModule->print(output, nullptr);
        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::DLTIDialect, mlir::func::FuncDialect>();
#ifdef CLANGIR_MLIR_FRONTEND
        mlir::registerAllToLLVMIRTranslations(registry);
        cir::direct::registerCIRDialectTranslation(registry);
#endif
        xilinx::registerAllAIEToLLVMIRTranslations(registry);
        registerAllToLLVMIRTranslations(registry);
      });
}

// Mainly copy-paste of registerAllTranslations() to handle "mlir-to-llvmir"
// option duplicated by aie-translate
void registerAllTranslationsWithoutToLLVMIR() {
  static bool initOnce = [] {
    mlir::registerFromLLVMIRTranslation();
    mlir::registerFromSPIRVTranslation();
    mlir::registerToCppTranslation();
    // "mlir-to-llvmir" is now handled by aie::registerToLLVMIRTranslation();
    // registerToLLVMIRTranslation();
    mlir::registerToSPIRVTranslation();
    return true;
  }();
  static_cast<void>(initOnce);
}

void versionPrinter(llvm::raw_ostream &os) {
  os << "aie-translate " << AIE_GIT_COMMIT << "\n";
}
} // namespace

int main(int argc, char **argv) {
  registerAllTranslationsWithoutToLLVMIR();
#ifdef CLANGIR_MLIR_FRONTEND
  registerToLLVMTranslation();
  cir::runAtStartOfConvertCIRToMLIRPass([](mlir::ConversionTarget ct) {
    ct.addLegalDialect<xilinx::AIE::AIEDialect, xilinx::AIEX::AIEXDialect,
                       xilinx::aievec::aie1::AIEVecAIE1Dialect,
                       xilinx::aievec::AIEVecDialect>();
  });
#endif
  registerToLLVMIRTranslation();
  xilinx::AIE::registerAIETranslations();
  xilinx::aievec::registerAIEVecToCppTranslation();

  llvm::cl::AddExtraVersionPrinter(versionPrinter);

  return failed(mlir::mlirTranslateMain(argc, argv, "AIE Translation Tool"));
}
