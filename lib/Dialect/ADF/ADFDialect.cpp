//===- ADFDialect.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/ADF/ADFDialect.h"
#include "aie/Dialect/ADF/ADFOps.h"

#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::ADF;

//===----------------------------------------------------------------------===//
// ADF Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/ADF/ADFTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// ADF Dialect
//===----------------------------------------------------------------------===//
void ADFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/ADF/ADF.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "aie/Dialect/ADF/ADFTypes.cpp.inc"
      >();
}

#include "aie/Dialect/ADF/ADFDialect.cpp.inc"
