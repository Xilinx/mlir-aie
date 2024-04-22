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

// TODO refactor clang/tools/cir-translate/cir-translate.cpp to avoid the
// following copy-paste
namespace cir {
namespace direct {
extern void registerCIRDialectTranslation(mlir::DialectRegistry &registry);
extern std::unique_ptr<llvm::Module>
lowerDirectlyFromCIRToLLVMIR(mlir::ModuleOp theModule,
                             llvm::LLVMContext &llvmCtx,
                             bool disableVerifier = false);
} // namespace direct

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
} // namespace cir

namespace aie {
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
        mlir::registerAllToLLVMIRTranslations(registry);
        cir::direct::registerCIRDialectTranslation(registry);
        xilinx::registerAllAIEToLLVMIRTranslations(registry);
        registerAllToLLVMIRTranslations(registry);
      });
}
} // namespace aie

namespace mlir {

void registerFromLLVMIRTranslation();
void registerFromSPIRVTranslation();
void registerToCppTranslation();
//void registerToLLVMIRTranslation();
void registerToSPIRVTranslation();

// Mainly copy-paste registerAllTranslations() to handle "mlir-to-llvmir" option
// duplicated by aie-translate
inline void registerAllTranslationsWithoutToLLVMIR() {
  static bool initOnce = []() {
    registerFromLLVMIRTranslation();
    registerFromSPIRVTranslation();
    registerToCppTranslation();
    // "mlir-to-llvmir" is now handle by aie::registerToLLVMIRTranslation();
    //registerToLLVMIRTranslation();
    registerToSPIRVTranslation();
    return true;
  }();
  (void)initOnce;
}
} // namespace mlir

void version_printer(raw_ostream &os) {
  os << "aie-translate " << AIE_GIT_COMMIT << "\n";
}

int main(int argc, char **argv) {
  mlir::registerAllTranslationsWithoutToLLVMIR();
  cir::registerToLLVMTranslation();
  aie::registerToLLVMIRTranslation();
  xilinx::AIE::registerAIETranslations();
  xilinx::aievec::registerAIEVecToCppTranslation();

  llvm::cl::AddExtraVersionPrinter(version_printer);

  return failed(mlir::mlirTranslateMain(argc, argv, "AIE Translation Tool"));
}
