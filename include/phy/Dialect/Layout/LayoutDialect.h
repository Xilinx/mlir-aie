//===- LayoutDialect.h ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_LAYOUT_H
#define MLIR_PHY_DIALECT_LAYOUT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "phy/Dialect/Spatial/SpatialDialect.h"

#include "phy/Dialect/Layout/LayoutDialect.h.inc"
#include "phy/Dialect/Layout/LayoutEnums.h.inc"
#include "phy/Dialect/Layout/LayoutPasses.h.inc"

#define GET_OP_CLASSES
#include "phy/Dialect/Layout/Layout.h.inc"

#endif // MLIR_PHY_DIALECT_LAYOUT_H
