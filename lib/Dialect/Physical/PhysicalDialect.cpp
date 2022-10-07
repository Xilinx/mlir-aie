//===- PhysicalDialect.cpp - Implement the Physical dialect ---------------===//
//
// This file implements the Physical dialect.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Physical/PhysicalDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir;
using namespace ::xilinx::phy::physical;

class PhysicalInlinerInterface : public DialectInlinerInterface {
public:
  using DialectInlinerInterface::DialectInlinerInterface;

  // The physical operations are always inlinable.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void PhysicalDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "phy/Dialect/Physical/Physical.cpp.inc"
      >();
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "phy/Dialect/Physical/PhysicalTypes.cpp.inc"
      >();
  // Register interfaces.
  addInterfaces<PhysicalInlinerInterface>();
}

// Provide implementations for the enums, attributes and interfaces that we use.
#include "phy/Dialect/Physical/PhysicalDialect.cpp.inc"
#include "phy/Dialect/Physical/PhysicalEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "phy/Dialect/Physical/PhysicalTypes.cpp.inc"

// TableGen'd op method definitions
#define GET_OP_CLASSES
#include "phy/Dialect/Physical/Physical.cpp.inc"
