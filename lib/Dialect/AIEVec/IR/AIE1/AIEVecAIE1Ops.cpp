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

// #include "aie/Dialect/AIEVec/IR/AIEVecEnums.cpp.inc"
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1OpsDialect.cpp.inc"

namespace xilinx::aievec::aie1 {

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
// AddOp and SubOp
//===----------------------------------------------------------------------===//

// Print out Add and Sub op.
template <typename T>
void printAddSubOp(OpAsmPrinter &p, T op) {
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

void AddOp::print(OpAsmPrinter &p) { printAddSubOp<AddOp>(p, *this); }

void SubOp::print(OpAsmPrinter &p) { printAddSubOp<SubOp>(p, *this); }

// Verify Add and Sub op.
template <typename T>
LogicalResult verifyAddSubOp(T op) {
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

LogicalResult AddOp::verify() { return verifyAddSubOp<AddOp>(*this); }

LogicalResult SubOp::verify() { return verifyAddSubOp<SubOp>(*this); }

// Parse Add and Sub op.
ParseResult parseAddSubOp(OpAsmParser &parser, OperationState &result) {
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

ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAddSubOp(parser, result);
}

ParseResult SubOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseAddSubOp(parser, result);
}

//===----------------------------------------------------------------------===//
// MulOp and FMAOp
//===----------------------------------------------------------------------===//

// MulOp and FMAOp are structurally similar, except that FMA op has few extra
// fields (accumulator, bool flag to indicate if it is fmsub, etc.). We create
// some specializations to print those fields specifically for FMA op.

// Print the accumulator
template <typename T>
void printAccumulator(OpAsmPrinter &p, T op);
template <>
inline void printAccumulator(OpAsmPrinter &p, FMAOp op) {
  p << ", " << op.getAcc();
}
template <>
inline void printAccumulator(OpAsmPrinter &p, MulOp op) {}

// Mark fmsub indicator as elided if the FMA op is not fmsub
template <typename T>
void elideFMSubAttr(T op, SmallVector<StringRef, 10> &elidedAttrs);
template <>
inline void elideFMSubAttr(FMAOp op, SmallVector<StringRef, 10> &elidedAttrs) {
  if (!op.getFmsub())
    elidedAttrs.push_back(op.getSubAttrName());
}
template <>
inline void elideFMSubAttr(MulOp, SmallVector<StringRef, 10> &elidedAttrs) {}

// Print out Mul and FMA op.
template <typename T>
static void printMulFMAOp(OpAsmPrinter &p, T op) {
  // Print the left operand
  p << " " << op.getLhs();
  // Print the right operand
  p << ", " << op.getRhs();
  // For fma op, print the accumulator
  printAccumulator(p, op);

  // Print the attributes, but don't print attributes that are empty strings
  SmallVector<StringRef, 10> elidedAttrs;
  for (int idx = 0; idx < 2; ++idx) {
    if (op.getStart(idx).empty())
      elidedAttrs.push_back(op.getStartAttrName(idx));
    if (op.getOffset(idx).empty())
      elidedAttrs.push_back(op.getOffsetAttrName(idx));
    if (op.getOffsetHi(idx).empty())
      elidedAttrs.push_back(op.getOffsetHiAttrName(idx));
    if (op.getStep(idx).empty())
      elidedAttrs.push_back(op.getStepAttrName(idx));
    if (op.getSquare(idx).empty())
      elidedAttrs.push_back(op.getSquareAttrName(idx));
    elideFMSubAttr(op, elidedAttrs);
  }
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << op.getLhs().getType() << ", " << op.getRhs().getType();
  p << ", " << op.getResult().getType();
}

void MulOp::print(OpAsmPrinter &p) { printMulFMAOp<MulOp>(p, *this); }

void FMAOp::print(OpAsmPrinter &p) { printMulFMAOp<FMAOp>(p, *this); }

// Verify Mul and FMA op.
template <typename T>
LogicalResult verifyMulFMAOp(T op) {
  // Verify the types
  auto lhsType = llvm::dyn_cast<VectorType>(op.getLhs().getType());
  auto rhsType = llvm::dyn_cast<VectorType>(op.getRhs().getType());

  if (!lhsType || !rhsType)
    return op.emitError("requires vector type");

  auto resultType = llvm::dyn_cast<VectorType>(op.getResult().getType());
  if (!resultType)
    return op.emitError("requires vector type");

  // Additional checks for FMA op
  // Get the width of the underlying scalars of all the vectors
  Type ltype = lhsType.getElementType();
  Type rtype = rhsType.getElementType();
  Type atype = resultType.getElementType();
  unsigned ltypeWidth = ltype.getIntOrFloatBitWidth();
  unsigned rtypeWidth = rtype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  // Checks on the number of lanes
  unsigned accLanes = getVectorLaneSize(resultType);
  unsigned rhsLanes = getVectorLaneSize(rhsType);
  unsigned lhsLanes = getVectorLaneSize(lhsType);

  // If this is not a simple scheme, perform complex checks
  if (accLanes != rhsLanes || accLanes != lhsLanes) {
    if (rhsLanes != 256 / rtypeWidth)
      return op.emitError("incorrect rhs operand vector lanes");
    if (lhsLanes < 2 * rhsLanes)
      return op.emitError("The number of lanes in lhs operand "
                          "must be at least twice that of rhs operand");
    if (accLanes > rhsLanes)
      return op.emitError("The number of lanes in accumulator "
                          "must be less than that of rhs operand");
  }

  // lhs and rhs vector's element type must match
  if (ltype != rtype)
    return op.emitError("The element type of lhs and rhs "
                        "operand vectors must match");

  // The datatype of accumulator must always be greater width
  if (isa<IntegerType>(atype)) {
    if (!isa<IntegerType>(ltype))
      return op.emitError("Integer result must have integer operands");

    if (ltypeWidth >= atypeWidth || rtypeWidth >= atypeWidth)
      return op.emitError("the element type of accumulator must have "
                          "wider width than that of the operand vectors");
  } else if (isa<FloatType>(atype)) {
    if (!isa<FloatType>(ltype))
      return op.emitError("Floating point result must have "
                          "floating point operands");

    if (ltypeWidth != atypeWidth || rtypeWidth != atypeWidth)
      return op.emitError("the element type of accumulator must be "
                          "same width as the operand vectors");
  }

  return success();
}

LogicalResult MulOp::verify() { return verifyMulFMAOp<MulOp>(*this); }

LogicalResult FMAOp::verify() { return verifyMulFMAOp<FMAOp>(*this); }

// Parse Mul and FMA op.
ParseResult parseMulFMAOp(OpAsmParser &parser, OperationState &result,
                          bool isFMAOp = true) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::UnresolvedOperand lhs, rhs, acc;

  // Parse the lhs and rhs
  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs))
    return failure();

  // Parse the acc for FMA op
  if (isFMAOp) {
    if (parser.parseComma() || parser.parseOperand(acc))
      return failure();
  }

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  // Assert that there are three types: lhs, rhs, and acc
  if (types.size() != 3)
    return parser.emitError(typesLoc, "requires three types");

  // Some verification
  VectorType lhsType = llvm::dyn_cast<VectorType>(types[0]);
  if (!lhsType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType rhsType = llvm::dyn_cast<VectorType>(types[1]);
  if (!rhsType)
    return parser.emitError(typesLoc, "requires vector type");

  // Int ops use the accumulator while float ops use normal vector registers
  VectorType accType = llvm::dyn_cast<VectorType>(types[2]);
  if (!accType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the lhs and rhs operands, and result
  if (parser.resolveOperand(lhs, lhsType, result.operands) ||
      parser.resolveOperand(rhs, rhsType, result.operands))
    return failure();

  // Populate acc operand for FMA op
  if (isFMAOp) {
    if (parser.resolveOperand(acc, accType, result.operands))
      return failure();
  }

  return parser.addTypeToList(accType, result.types);
}

ParseResult MulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMulFMAOp(parser, result, false);
}

ParseResult FMAOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMulFMAOp(parser, result, true);
}

} // namespace xilinx::aievec::aie1

// #define GET_ATTRDEF_CLASSES
// #include "aie/Dialect/AIEVec/IR/AIEVecAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/AIE1/IR/AIEVecAIE1Ops.cpp.inc"
