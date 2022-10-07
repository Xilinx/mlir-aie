//===- phy-opt.cpp - The phy-opt driver -----------------------------------===//
//
// This file implements the 'phy-opt' tool, which is the phy analog of
// mlir-opt, used to drive compiler passes.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "phy/Conversion/Passes.h"
#include "phy/Dialect/Layout/LayoutDialect.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithmeticDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  // Register MLIR-PHY Dialects
  registry.insert<xilinx::phy::layout::LayoutDialect>();
  registry.insert<xilinx::phy::physical::PhysicalDialect>();
  registry.insert<xilinx::phy::spatial::SpatialDialect>();

  // Register the conversion passes.
  xilinx::phy::registerConversionPasses();

  // Register the standard passes we want.
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerInlinerPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerSCCPPass();
  mlir::registerSymbolDCEPass();

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "MLIR-PHY optimizer driver", registry));
}
