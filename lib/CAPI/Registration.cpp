//===- Registration.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include "aie-c/Registration.h"

#include "aie/Conversion/Passes.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEVec/Analysis/Passes.h"
#include "aie/Dialect/AIEVec/Pipelines/Passes.h"
#include "aie/Dialect/AIEVec/Transforms/Passes.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "aie/InitialAllDialect.h"
#include "aie/Target/LLVMIR/Dialect/All.h"

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

using namespace llvm;
using namespace mlir;

void aieRegisterAllPasses() {
  xilinx::registerConversionPasses();
  xilinx::AIE::registerAIEPasses();
  xilinx::AIEX::registerAIEXPasses();
  xilinx::aievec::registerAIEVecAnalysisPasses();
  xilinx::aievec::registerAIEVecPasses();
  xilinx::aievec::registerAIEVecPipelines();

  DialectRegistry registry;
  registerAllDialects(registry);
  xilinx::registerAllDialects(registry);

  registerAllExtensions(registry);

  registry.insert<DLTIDialect>();
  xilinx::registerAllAIEToLLVMIRTranslations(registry);
  registerAllToLLVMIRTranslations(registry);
}
