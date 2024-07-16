//===-- AIEVecAIE1Ops.cpp - MLIR AIE Vector Dialect Operations --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// This file implements AIE1 vector op printing, pasing, and verification.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Ops.h"
#include "aie/Dialect/AIEVec/AIEVecUtils.h"

using namespace llvm;
using namespace mlir;
using namespace xilinx;
using namespace xilinx::aievec;

// #include "aie/Dialect/AIEVec/IR/AIEVecEnums.cpp.inc"
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// AIEVecAIE1Dialect
//===----------------------------------------------------------------------===//

void AIEVecAIE1Dialect::initialize() {
  // registerTypes();
  //   addAttributes<
  // #define GET_ATTRDEF_LIST
  // #include "aie/Dialect/AIEVec/IR/AIEVecAttributes.cpp.inc"
  //       >();
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AIE1_AddOp and AIE1_SubOp
//===----------------------------------------------------------------------===//

// Print out Add and Sub op.
template <typename T>
void printAddAIE1_SubOp(OpAsmPrinter &p, T op) {
  // Print the lhs operand
  p << " " << op.getLhs();
  // Print the rhs operand
  p << ", " << op.getRhs();

  // Print the attributes, but don't print attributes that are empty strings
  SmallVector<StringRef, 10> elidedAttrs;
  for (int idx = 0; idx < 2; ++idx) {
    if (op.getStart(idx).empty())
      elidedAttrs.push_back(op.getStartAttrName(idx));
    if (op.getOffset(idx).empty())
      elidedAttrs.push_back(op.getOffsetAttrName(idx));
    if (op.getOffsetHi(idx).empty())
      elidedAttrs.push_back(op.getOffsetHiAttrName(idx));
    if (op.getSquare(idx).empty())
      elidedAttrs.push_back(op.getSquareAttrName(idx));
  }
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << op.getLhs().getType() << ", " << op.getRhs().getType();
  p << ", " << op.getResult().getType();
}

void aievec::AIE1_AddOp::print(OpAsmPrinter &p) {
  printAddAIE1_SubOp<aievec::AIE1_AddOp>(p, *this);
}

void aievec::AIE1_SubOp::print(OpAsmPrinter &p) {
  printAddAIE1_SubOp<aievec::AIE1_SubOp>(p, *this);
}

// Verify Add and Sub op.
template <typename T>
LogicalResult verifyAddAIE1_SubOp(T op) {
  // Verify the types
  auto resultType = llvm::dyn_cast<VectorType>(op.getResult().getType());
  auto lhsType = llvm::dyn_cast<VectorType>(op.getLhs().getType());
  auto rhsType = llvm::dyn_cast<VectorType>(op.getRhs().getType());

  if (!lhsType || !rhsType || !resultType)
    return op.emitError("requires vector type");

  // All the vector types must match
  if (lhsType != rhsType || rhsType != resultType)
    return op.emitError("all vectors must be of same type");

  return success();
}

LogicalResult aievec::AIE1_AddOp::verify() {
  return verifyAddAIE1_SubOp<aievec::AIE1_AddOp>(*this);
}

LogicalResult aievec::AIE1_SubOp::verify() {
  return verifyAddAIE1_SubOp<aievec::AIE1_SubOp>(*this);
}

// Parse Add and Sub op.
ParseResult parseAddAIE1_SubOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::UnresolvedOperand lhs, rhs;

  // Parse the lhs and rhs
  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  // Assert that there are three types: lhs, rhs, and result
  if (types.size() != 3)
    return parser.emitError(typesLoc, "requires three types");

  // Some verification
  VectorType lhsType = llvm::dyn_cast<VectorType>(types[0]);
  if (!lhsType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType rhsType = llvm::dyn_cast<VectorType>(types[1]);
  if (!rhsType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType resultType = llvm::dyn_cast<VectorType>(types[2]);
  if (!resultType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the lhs, rhs, and accumulator in the result
  if (parser.resolveOperand(lhs, lhsType, result.operands) ||
      parser.resolveOperand(rhs, rhsType, result.operands))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

ParseResult AIE1_AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAddAIE1_SubOp(parser, result);
}

ParseResult AIE1_SubOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAddAIE1_SubOp(parser, result);
}

// #define GET_ATTRDEF_CLASSES
// #include "aie/Dialect/AIEVec/IR/AIEVecAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Ops.cpp.inc"
