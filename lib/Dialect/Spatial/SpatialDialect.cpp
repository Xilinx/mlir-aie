//===- SpatialDialect.cpp - Implement the Spatial dialect -----------------===//
//
// This file implements the Spatial dialect.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Spatial/SpatialDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir;
using namespace ::xilinx::phy::spatial;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void SpatialDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "phy/Dialect/Spatial/Spatial.cpp.inc"
      >();
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "phy/Dialect/Spatial/SpatialTypes.cpp.inc"
      >();
}

// Provide implementations for the enums, attributes and interfaces that we use.
#include "phy/Dialect/Spatial/SpatialDialect.cpp.inc"
#include "phy/Dialect/Spatial/SpatialEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "phy/Dialect/Spatial/SpatialTypes.cpp.inc"

// TableGen'd op method definitions
#define GET_OP_CLASSES
#include "phy/Dialect/Spatial/Spatial.cpp.inc"
