//===- aie-visualize.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Xilinx Inc.
//
//===---------------------------------------------------------------------===//

// This tool generates a simple visualization of a design, showing the
// device layout and highlighting which device tiles are being used.

#include "aie/InitialAllDialect.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

#include <iostream>

using namespace llvm;
using namespace mlir;
using namespace xilinx;

static cl::opt<std::string> fileName(cl::Positional, cl::desc("<input mlir>"),
                                     cl::Required);

const std::string bold("\033[0;1m");
const std::string dim("\033[0;2m");
const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string blue("\033[1;34m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string bwhite("\033[0;47m");
const std::string reset("\033[0m");
const std::string bgray("\033[48;5;239m");

int main(int argc, char *argv[]) {
  cl::ParseCommandLineOptions(argc, argv);

  MLIRContext ctx;
  ParserConfig pcfg(&ctx);
  SourceMgr srcMgr;

  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<vector::VectorDialect>();
  xilinx::registerAllDialects(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  xilinx::xllvm::registerXLLVMDialectTranslation(registry);
  ctx.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> owning =
      parseSourceFile<ModuleOp>(fileName, srcMgr, pcfg);

  if (!owning)
    return 1;

  auto deviceOps = owning->getOps<AIE::DeviceOp>();
  if (!llvm::hasSingleElement(deviceOps))
    return 2;

  AIE::DeviceOp deviceOp = *deviceOps.begin();

  const xilinx::AIE::AIETargetModel &model = deviceOp.getTargetModel();

  model.validate();

  std::vector<bool> used(model.columns() * model.rows());
  for (int col = 0; col < model.columns(); col++) {
    for (int row = 0; row < model.rows(); row++) {
      used[col + model.columns() * row] = false;
    }
  }
  for (auto tile : deviceOp.getOps<AIE::TileOp>()) {
    used[tile.getCol() + model.columns() * tile.getRow()] = true;
  }

  std::cout << model.columns() << " Columns and " << model.rows() << " Rows\n";
  for (int row = model.rows() - 1; row >= 0; row--) {
    std::cout << reset << row % 10 << " ";
    for (int col = 0; col < model.columns(); col++) {
      if (used[col + model.columns() * row])
        std::cout << bgray;
      else
        std::cout << dim;
      std::string v = reset + ".";
      if (model.isCoreTile(col, row))
        v = green + 'C';
      else if (model.isMemTile(col, row))
        v = red + 'M';
      else if (model.isShimNOCTile(col, row))
        v = blue + 'D';
      else if (model.isShimPLTile(col, row))
        v = magenta + 'P';
      std::cout << v << reset;
    }
    std::cout << "\n";
  }

  std::cout << reset << "  ";
  for (int col = 0; col < model.columns(); col++)
    std::cout << col % 10;
  std::cout << "\n";

  std::cout << "  ";
  for (int col = 0; col < model.columns(); col++) {
    int coltens = col / 10;
    if (coltens > 0)
      std::cout << coltens;
    else
      std::cout << " ";
  }
  std::cout << "\n";

  return 0;
}
