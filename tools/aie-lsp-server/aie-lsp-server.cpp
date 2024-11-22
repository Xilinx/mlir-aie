//===- aie-lsp-server -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.
//
// A server exposing AIE dialects to be used by IDE & editors supporting LSP
// https://microsoft.github.io/language-server-protocol like Emacs, VScode, etc.
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "aie/InitialAllDialect.h"
#ifdef CLANGIR_MLIR_FRONTEND
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  xilinx::registerAllDialects(registry);
#ifdef CLANGIR_MLIR_FRONTEND
  registry.insert<cir::CIRDialect>();
#endif
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
