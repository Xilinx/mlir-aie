//===- PhysicalOps.cpp - Implement the Phy operations ---------------------===//
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
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir;
using namespace ::xilinx::phy::physical;

LogicalResult CoreOp::verifySymbolUses(SymbolTableCollection &symbol_table) {
  auto fn_attr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fn_attr)
    return emitOpError("requires a 'callee' symbol reference attribute");

  func::FuncOp fn =
      symbol_table.lookupNearestSymbolFrom<func::FuncOp>(*this, fn_attr);
  if (!fn) {
    return emitOpError() << "expected symbol reference " << getCallee()
                         << " to point to a function";
  }

  // Verify that the operand and result types match the callee.
  auto fn_type = fn.getFunctionType();
  if (fn_type.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fn_type.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fn_type.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fn_type.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fn_type.getNumResults() != 0)
    return emitOpError("callee cannot have a return value");

  return success();
}
