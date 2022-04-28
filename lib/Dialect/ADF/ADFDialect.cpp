//===- ADFDialect.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
//

#include "aie/Dialect/ADF/ADFDialect.h"
#include "aie/Dialect/ADF/ADFOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace xilinx;
using namespace ADF;

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
