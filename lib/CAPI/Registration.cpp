//===- Registration.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "aie-c/Registration.h"
#include "aie/InitialAllDialect.h"

#include "mlir/CAPI/IR.h"
// #include "mlir/InitAllPasses.h"

void aieRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  xilinx::registerAllDialects(registry);
}

void aieRegisterAllPasses() {
  // xilinx::AIE::registerAllPasses();
}
