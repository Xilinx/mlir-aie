//===- aie-lsp-server -------------------------------------------*- C++ -*-===//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A server exposing AIE dialects to be used by IDE & editors supporting LSP
// https://microsoft.github.io/language-server-protocol like Emacs, VScode, etc.
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "aie/InitialAllDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  xilinx::registerAllDialects(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
