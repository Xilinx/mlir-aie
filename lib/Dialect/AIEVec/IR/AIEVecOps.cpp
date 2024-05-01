//===---- AIEVecOps.cpp - MLIR AIE Vector Dialect Operations ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file implements AIE vector op printing, pasing, and verification.
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/AIEVecUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/FoldUtils.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::aievec;

#include "aie/Dialect/AIEVec/IR/AIEVecOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// AIEVecDialect
//===----------------------------------------------------------------------===//

void AIEVecDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/AIEVec/IR/AIEVecOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// UPDOp
//===----------------------------------------------------------------------===//

// Print out UPD op.
void UPDOp::print(OpAsmPrinter &p) {
  // Print the source memref
  p << " " << getSource() << "[" << getIndices() << "]";
  // Now print the optional vector that links upd idx=1 with idx=0
  if (getVector())
    p << ", " << getVector();

  // Print the attributes, but don't print the operand segment sizes
  SmallVector<StringRef, 3> elidedAttrs;
  elidedAttrs.push_back(UPDOp::getOperandSegmentSizeAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify UPD op.
LogicalResult UPDOp::verify() {
  // Verify the types: source is memref, and result is vector
  MemRefType sourceType = llvm::dyn_cast<MemRefType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType)
    return emitError("requires memref type");
  if (!resultType)
    return emitError("requires vector type");
  if (getIndices().empty())
    return emitError("upd source cannot come from scalar value");

  // If this UPD op is linked to another UPD op, then verify that the linked
  // vector and the result vector match.
  if (getVector()) {
    Type vecType = llvm::dyn_cast<VectorType>(getVector().getType());
    if (vecType != resultType)
      return emitError("result types of linked UPD ops do not match");
  }
  return success();
}

// Parse UPD op.
ParseResult UPDOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source, vector;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> indices;

  // Parse the source, indices, and optional vector
  if (parser.parseOperand(source) ||
      parser.parseOperandList(indices, OpAsmParser::Delimiter::Square))
    return failure();
  ParseResult hasVector = parser.parseOptionalComma();
  if (hasVector.succeeded() && parser.parseOperand(vector))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 2)
    return parser.emitError(typesLoc, "requires two attributes");

  // Assert that there are two types (memref source and vector result)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification
  auto memrefType = llvm::dyn_cast<MemRefType>(types[0]);
  if (!memrefType)
    return parser.emitError(typesLoc, "requires memref type");
  VectorType vectorType = llvm::dyn_cast<VectorType>(types[1]);
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");
  auto indicesType = builder.getIndexType();

  // Populate the source and indices in result
  if (parser.resolveOperand(source, memrefType, result.operands) ||
      parser.resolveOperands(indices, indicesType, result.operands))
    return failure();
  // Populate optional vector in result
  if (hasVector.succeeded())
    if (parser.resolveOperand(vector, vectorType, result.operands))
      return failure();

  // Populate operand size attribute in result
  result.addAttribute(UPDOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(indices.size()),
                           static_cast<int32_t>(hasVector.succeeded())}));

  return parser.addTypeToList(vectorType, result.types);
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

// Print out Cast op.
void CastOp::print(OpAsmPrinter &p) {
  // Print the source accumulator
  p << " " << getSource();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify Cast op.
LogicalResult CastOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType)
    return emitError("requires source vector type");
  if (!resultType)
    return emitError("requires result vector type");

  if (sourceType.getElementType().getIntOrFloatBitWidth() !=
      resultType.getElementType().getIntOrFloatBitWidth()) {
    return emitError("the bitwidth of resource and result should be equal");
  }

  return success();
}

// Parse Cast op.
ParseResult CastOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source vector
  if (parser.parseOperand(source))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "requires one attribute");

  // Assert that there are two types (source and result)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification of types
  VectorType sourceType = llvm::dyn_cast<VectorType>(types[0]);
  if (!sourceType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType vectorType = llvm::dyn_cast<VectorType>(types[1]);
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, sourceType, result.operands))
    return failure();

  return parser.addTypeToList(vectorType, result.types);
}

// Cast fold method. It will fold with a preceding Cast operation.
OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  auto srcCastOp = getSource().getDefiningOp<aievec::CastOp>();
  if (!srcCastOp)
    return nullptr;

  if (srcCastOp.getIsResAcc() == getIsResAcc())
    return srcCastOp.getResult();

  return srcCastOp.getSource();
}

//===----------------------------------------------------------------------===//
// SRSOp
//===----------------------------------------------------------------------===//

// SRS fold method. It will fold with a preceding UPS operation.
OpFoldResult SRSOp::fold(FoldAdaptor adaptor) {
  auto srcDefOp = getSource().getDefiningOp();
  if (!srcDefOp)
    return nullptr;

  auto upsOp = dyn_cast<UPSOp>(srcDefOp);
  if (!upsOp)
    return nullptr;

  auto shiftDefOp = getShift().getDefiningOp();
  if (!shiftDefOp)
    return nullptr;

  auto constOp = dyn_cast<arith::ConstantOp>(shiftDefOp);
  if (!constOp)
    return nullptr;

  if (upsOp.getSource().getType() != getResult().getType())
    return nullptr;

  return upsOp.getSource();
}

// Print out SRS op.
void SRSOp::print(OpAsmPrinter &p) {
  // Print the source accumulator
  p << " " << getSource() << ", ";

  // Print the shift
  p << getShift();

  // And now print the types
  p << " : " << getSource().getType() << ", " << getShift().getType() << ", "
    << getResult().getType();
}

// Verify SRS op.
LogicalResult SRSOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType)
    return emitError("requires accumulator type");
  if (!resultType)
    return emitError("requires vector type");

  // The number of lanes of source accumulator and result vector must match
  unsigned accLanes = getVectorLaneSize(sourceType);
  unsigned vecLanes = getVectorLaneSize(resultType);
  if (accLanes != vecLanes)
    return emitError("The number of lanes in result vector "
                     "and source accumulator must match");

  // The datatype of accumulator must have greater width
  Type stype = resultType.getElementType();
  Type atype = sourceType.getElementType();
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  if (atype.isa<IntegerType>() && stypeWidth >= atypeWidth)
    return emitError("the element type of source accumulator must be "
                     "wider than that of the result vector");
  else if (atype.isa<FloatType>() && stypeWidth != 16 &&
           stypeWidth != atypeWidth)
    return emitError("the element type of source accumulator must be "
                     "same as the result vector");

  return success();
}

// Parse SRS op.
ParseResult SRSOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::UnresolvedOperand source, shift;

  // Parse the source accumulator
  if (parser.parseOperand(source) || parser.parseComma() ||
      parser.parseOperand(shift))
    return failure();

  // Parse types
  if (parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  // Assert that there are two types (accumulator source and vector result)
  if (types.size() != 3)
    return parser.emitError(typesLoc, "requires three types");

  // Some verification of types
  VectorType accType = llvm::dyn_cast<VectorType>(types[0]);
  if (!accType)
    return parser.emitError(typesLoc, "requires vector type");

  IntegerType shiftType = llvm::dyn_cast<IntegerType>(types[1]);
  if (!shiftType)
    return parser.emitError(typesLoc, "requires integer type");

  VectorType vectorType = llvm::dyn_cast<VectorType>(types[2]);
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, accType, result.operands) ||
      parser.resolveOperand(shift, shiftType, result.operands))
    return failure();

  return parser.addTypeToList(vectorType, result.types);
}

//===----------------------------------------------------------------------===//
// UPSOp
//===----------------------------------------------------------------------===//

// UPS fold method. It will fold with a preceding SRS operation.
OpFoldResult UPSOp::fold(FoldAdaptor adaptor) {
  // TODO: Both UPS and SRS have an aditional parameter (shift) that's being
  // TODO: ignored here. Somebody should take a careful look at it.
  // TODO: In next llvm version: auto srsDefOp =
  // adaptor.getSource().getDefiningOp();
  auto srcDefOp = getSource().getDefiningOp();
  if (!srcDefOp)
    return nullptr;
  auto srsOp = dyn_cast<SRSOp>(srcDefOp);
  if (!srsOp)
    return nullptr;
  return srsOp.getSource();
}

// Print out UPS op.
void UPSOp::print(OpAsmPrinter &p) {
  // Print the source vector
  p << " " << getSource();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify UPS op.
LogicalResult UPSOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType)
    return emitError("requires vector type");
  if (!resultType)
    return emitError("requires vector type");

  // The number of lanes must match
  unsigned vecLanes = getVectorLaneSize(sourceType);
  unsigned accLanes = getVectorLaneSize(resultType);
  if (vecLanes != accLanes)
    return emitError("The number of lanes in source vector "
                     "and result accumulator must match");

  // The datatype of accumulator must always be greater width
  Type stype = sourceType.getElementType();
  Type atype = resultType.getElementType();
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  if (stypeWidth >= atypeWidth)
    return emitError("the element type of result accumulator "
                     "must be wider than that of the source vector");

  return success();
}

// Parse UPS op.
ParseResult UPSOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source vector
  if (parser.parseOperand(source))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "requires one attribute");

  // Assert that there are two types (source vector and accumulator result)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification
  VectorType vectorType = llvm::dyn_cast<VectorType>(types[0]);
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType accType = llvm::dyn_cast<VectorType>(types[1]);
  if (!accType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, vectorType, result.operands))
    return failure();

  return parser.addTypeToList(accType, result.types);
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// Print out Broadcast op.
void BroadcastOp::print(OpAsmPrinter &p) {
  // Print the source vector
  p << " " << getSource();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify Broadcast op.
LogicalResult BroadcastOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());

  if (!sourceType)
    return emitError("requires vector type");
  if (!resultType)
    return emitError("requires vector type");

  if (sourceType != resultType) {
    return emitError("The vector type of source vector "
                     "and result vector must match");
  }
  // The number of lanes must match
  unsigned sourceLanes = getVectorLaneSize(sourceType);
  unsigned resultLanes = getVectorLaneSize(resultType);
  if (sourceLanes != resultLanes)
    return emitError("The number of lanes in source vector "
                     "and result vector must match");

  // The element type of vectors must always be the same
  Type stype = sourceType.getElementType();
  Type rtype = resultType.getElementType();

  if (stype != rtype) {
    return emitError("the element type of result vector "
                     "must be the same as source vector");
  }

  return success();
}

// Parse Broadcast op.
ParseResult BroadcastOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source vector
  if (parser.parseOperand(source))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "requires one attribute");

  // Assert that there are two types (source vector and result vector)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification
  VectorType vecType = llvm::dyn_cast<VectorType>(types[0]);
  if (!vecType)
    return parser.emitError(typesLoc, "requires vector type");

  VectorType resType = llvm::dyn_cast<VectorType>(types[1]);
  if (!resType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, vecType, result.operands))
    return failure();

  return parser.addTypeToList(resType, result.types);
}

//===----------------------------------------------------------------------===//
// BroadcastScalarOp
//===----------------------------------------------------------------------===//

// Print out BroadcastScalar op.
void BroadcastScalarOp::print(OpAsmPrinter &p) {
  // Print the source vector
  p << " " << getSource();

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify BroadcastScalar op.
LogicalResult BroadcastScalarOp::verify() {
  // Verify the types
  Type sourceType = getSource().getType();
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());

  if (!resultType)
    return emitError("requires vector type");

  if (!sourceType.isa<IntegerType, FloatType>())
    return emitError("requires source type to be integer or float");

  Type resultElemType = resultType.getElementType();
  if (sourceType != resultElemType) {
    return emitError("the element type of result vector must be the same as "
                     "the source type");
  }

  return success();
}

// Parse BroadcastScalar op.
ParseResult BroadcastScalarOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source vector
  if (parser.parseOperand(source))
    return failure();

  // Parse all the attributes and types
  if (parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (!result.attributes.getAttrs().empty())
    return parser.emitError(typesLoc, "do not require attributes");

  // Assert that there is two type (source and result vector)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification
  VectorType resType = llvm::dyn_cast<VectorType>(types[1]);
  if (!resType)
    return parser.emitError(typesLoc, "requires vector type");

  if (parser.resolveOperand(source, types[0], result.operands))
    return failure();

  return parser.addTypeToList(resType, result.types);
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
inline void printAccumulator(OpAsmPrinter &p, aievec::FMAOp op) {
  p << ", " << op.getAcc();
}
template <>
inline void printAccumulator(OpAsmPrinter &p, aievec::MulOp op) {}

// Mark fmsub indicator as elided if the FMA op is not fmsub
template <typename T>
void elideFMSubAttr(T op, SmallVector<StringRef, 10> &elidedAttrs);
template <>
inline void elideFMSubAttr(aievec::FMAOp op,
                           SmallVector<StringRef, 10> &elidedAttrs) {
  if (!op.getFmsub())
    elidedAttrs.push_back(op.getSubAttrName());
}
template <>
inline void elideFMSubAttr(aievec::MulOp,
                           SmallVector<StringRef, 10> &elidedAttrs) {}

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

void MulOp::print(OpAsmPrinter &p) { printMulFMAOp<aievec::MulOp>(p, *this); }

void aievec::FMAOp::print(OpAsmPrinter &p) {
  printMulFMAOp<aievec::FMAOp>(p, *this);
}

// Verify Mul and FMA op.
template <typename T>
LogicalResult verifyMulFMAOp(T op) {
  // Verify the types
  auto lhsType = op.getLhs().getType().template dyn_cast<VectorType>();
  auto rhsType = op.getRhs().getType().template dyn_cast<VectorType>();

  if (!lhsType || !rhsType)
    return op.emitError("requires vector type");

  auto resultType = op.getResult().getType().template dyn_cast<VectorType>();
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
  if (atype.isa<IntegerType>()) {
    if (!ltype.isa<IntegerType>())
      return op.emitError("Integer result must have integer operands");

    if (ltypeWidth >= atypeWidth || rtypeWidth >= atypeWidth)
      return op.emitError("the element type of accumulator must have "
                          "wider width than that of the operand vectors");
  } else if (atype.isa<FloatType>()) {
    if (!ltype.isa<FloatType>())
      return op.emitError("Floating point result must have "
                          "floating point operands");

    if (ltypeWidth != atypeWidth || rtypeWidth != atypeWidth)
      return op.emitError("the element type of accumulator must be "
                          "same width as the operand vectors");
  }

  return success();
}

LogicalResult aievec::MulOp::verify() {
  return verifyMulFMAOp<aievec::MulOp>(*this);
}

LogicalResult aievec::FMAOp::verify() {
  return verifyMulFMAOp<aievec::FMAOp>(*this);
}

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

//===----------------------------------------------------------------------===//
// MulElemOp and FMAElemOp
//===----------------------------------------------------------------------===//

// MulElemOp and FMAElemOp are structurally similar, except that FMAElem op
// has few extra fields (accumulator, bool flag to indicate if it is fmsub,
// etc.). We create some specializations to print those fields specifically for
// FMAElemOp and MULElemOp.

// Print the accumulator
template <typename T>
void printAccumulator(OpAsmPrinter &p, T op);
template <>
inline void printAccumulator(OpAsmPrinter &p, aievec::FMAElemOp op) {
  p << ", " << op.getAcc();
}
template <>
inline void printAccumulator(OpAsmPrinter &p, aievec::MulElemOp op) {}

// Mark fmsub indicator as elided if the FMAElem op is not fmsub
template <typename T>
void elideFMSubAttr(T op, SmallVector<StringRef, 4> &elidedAttrs);
template <>
inline void elideFMSubAttr(aievec::FMAElemOp op,
                           SmallVector<StringRef, 4> &elidedAttrs) {
  if (!op.getFmsub())
    elidedAttrs.push_back(op.getSubAttrName());
}

template <>
inline void elideFMSubAttr(aievec::MulElemOp op,
                           SmallVector<StringRef, 4> &elidedAttrs) {}

// Print out MulElem and FMAElem op.
template <typename T>
static void printMulFMAElemOp(OpAsmPrinter &p, T op) {
  // Print the left operand
  p << " " << op.getLhs();
  // Print the right operand
  p << ", " << op.getRhs();
  // For fma op, print the accumulator
  printAccumulator(p, op);

  // Print the attributes, but don't print attributes that are empty strings
  SmallVector<StringRef, 4> elidedAttrs;
  for (int idx = 0; idx < 2; ++idx) {
    elideFMSubAttr(op, elidedAttrs);
  }
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << op.getLhs().getType() << ", " << op.getRhs().getType();
  p << ", " << op.getResult().getType();
}

void MulElemOp::print(OpAsmPrinter &p) {
  printMulFMAElemOp<aievec::MulElemOp>(p, *this);
}

void aievec::FMAElemOp::print(OpAsmPrinter &p) {
  printMulFMAElemOp<aievec::FMAElemOp>(p, *this);
}

// Verify MulElem and FMAElem op.
template <typename T>
LogicalResult verifyMulFMAElemOp(T op) {
  // Verify the types
  auto lhsType = op.getLhs().getType().template dyn_cast<VectorType>();
  auto rhsType = op.getRhs().getType().template dyn_cast<VectorType>();

  if (!lhsType || !rhsType)
    return op.emitError("requires vector type");

  auto resultType = op.getResult().getType().template dyn_cast<VectorType>();

  if (!resultType)
    return op.emitError("requires vector type");

  // Additional checks for FMAElem op
  // Get the width of the underlying scalars of all the vectors
  Type ltype = lhsType.getElementType();
  Type rtype = rhsType.getElementType();
  Type atype = resultType.getElementType();
  unsigned ltypeWidth = ltype.getIntOrFloatBitWidth();
  unsigned rtypeWidth = rtype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  // Checks on the number of lanes
  unsigned rhsLanes = getVectorLaneSize(rhsType);
  unsigned lhsLanes = getVectorLaneSize(lhsType);

  // lane size must match
  if (lhsLanes != rhsLanes) {
    return op.emitError("The number of lanes in lhs operand "
                        "must be the same as rhs operand");
  }

  // lhs and rhs vector's element type must match
  if (ltype != rtype)
    return op.emitError("The element type of lhs and rhs "
                        "operand vectors must match");

  // The integer datatype of accumulator must always be greater width
  if (atype.isa<IntegerType>()) {
    if (!ltype.isa<IntegerType>())
      return op.emitError("Integer result must have integer operands");

    if (ltypeWidth >= atypeWidth || rtypeWidth >= atypeWidth)
      return op.emitError("the element type of accumulator must have "
                          "wider width than that of the operand vectors");
  } else if (atype.isa<FloatType>()) {
    if (!ltype.isa<FloatType>())
      return op.emitError("Floating point result must have "
                          "floating point operands");
  }

  return success();
}

LogicalResult aievec::MulElemOp::verify() {
  return verifyMulFMAElemOp<aievec::MulElemOp>(*this);
}

LogicalResult aievec::FMAElemOp::verify() {
  return verifyMulFMAElemOp<aievec::FMAElemOp>(*this);
}

// Parse MulElem and FMAElem op.
ParseResult parseMulFMAElemOp(OpAsmParser &parser, OperationState &result,
                              bool isFMAElemOp = true) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::UnresolvedOperand lhs, rhs, acc;

  // Parse the lhs and rhs
  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs))
    return failure();

  // Parse the acc for FMA op
  if (isFMAElemOp) {
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
  if (isFMAElemOp) {
    if (parser.resolveOperand(acc, accType, result.operands))
      return failure();
  }

  return parser.addTypeToList(accType, result.types);
}

ParseResult MulElemOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMulFMAElemOp(parser, result, false);
}

ParseResult FMAElemOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMulFMAElemOp(parser, result, true);
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

void aievec::AddOp::print(OpAsmPrinter &p) {
  printAddSubOp<aievec::AddOp>(p, *this);
}

void aievec::SubOp::print(OpAsmPrinter &p) {
  printAddSubOp<aievec::SubOp>(p, *this);
}

// Verify Add and Sub op.
template <typename T>
LogicalResult verifyAddSubOp(T op) {
  // Verify the types
  auto resultType = op.getResult().getType().template dyn_cast<VectorType>();
  auto lhsType = op.getLhs().getType().template dyn_cast<VectorType>();
  auto rhsType = op.getRhs().getType().template dyn_cast<VectorType>();

  if (!lhsType || !rhsType || !resultType)
    return op.emitError("requires vector type");

  // All the vector types must match
  if (lhsType != rhsType || rhsType != resultType)
    return op.emitError("all vectors must be of same type");

  return success();
}

LogicalResult aievec::AddOp::verify() {
  return verifyAddSubOp<aievec::AddOp>(*this);
}

LogicalResult aievec::SubOp::verify() {
  return verifyAddSubOp<aievec::SubOp>(*this);
}

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
// ConcatOp
//===----------------------------------------------------------------------===//

// Print out Concat op.
void ConcatOp::print(OpAsmPrinter &p) {
  // Print the source vectors
  assert(!getSources().empty() && "concat source empty");
  p << " " << getSources();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getSources().getTypes().front() << ", "
    << getResult().getType();
}

// Verify Concat op.
LogicalResult ConcatOp::verify() {
  // Must be concatenating at least two sources
  if (getSources().size() < 2)
    return emitError("Must concatenate at least two vectors");

  // Verify the types
  VectorType sourceType =
      llvm::dyn_cast<VectorType>(getSources().getTypes().front());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitError("requires vector type");

  SmallVector<Value, 8> srcs(getSources().begin(), getSources().end());
  // All the sources must have the same type
  for (auto source : srcs) {
    VectorType type = llvm::dyn_cast<VectorType>(source.getType());
    if (!type)
      return emitError("requires vector type");
    if (type != sourceType)
      return emitError("All sources must have same type");
  }

  // The lanes in concatenated type must be the sum of lanes of source vector
  unsigned totalLanes = 0;
  for (auto source : srcs) {
    VectorType type = llvm::dyn_cast<VectorType>(source.getType());
    totalLanes += getVectorLaneSize(type);
  }

  if (totalLanes != getVectorLaneSize(resultType))
    return emitError("mismatch between vector lanes "
                     "and sum of source lanes");

  return success();
}

// Parse Concat op.
ParseResult ConcatOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> sources;

  // Parse the source vectors
  if (parser.parseOperandList(sources))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  // Currently there are no attributes in concat op
  if (!result.attributes.getAttrs().empty())
    return parser.emitError(typesLoc, "expects no attribute");

  // Assert that there are two types (type of all sources, and result)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification
  VectorType sourceType = llvm::dyn_cast<VectorType>(types[0]);
  VectorType resultType = llvm::dyn_cast<VectorType>(types[1]);
  if (!sourceType || !resultType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source vectors in result
  if (parser.resolveOperands(sources, sourceType, result.operands))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

LogicalResult
ConcatOp::inferReturnTypes(MLIRContext *, std::optional<Location>,
                           ConcatOp::Adaptor adaptor,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
  SmallVector<Value, 8> srcs(adaptor.getSources().begin(),
                             adaptor.getSources().end());
  unsigned totalLength = 0;
  for (auto source : srcs) {
    VectorType type = llvm::dyn_cast<VectorType>(source.getType());
    assert(type.getRank() == 1 &&
           "only rank 1 vectors currently supported by concat");
    totalLength += type.getDimSize(0);
  }
  inferredReturnTypes.push_back(VectorType::get(
      {totalLength},
      srcs[0].getType().dyn_cast<VectorType>().getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ExtOp
//===----------------------------------------------------------------------===//

// Print out Ext op.
void ExtOp::print(OpAsmPrinter &p) {
  // Print the source vector
  p << " " << getSource();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify Ext op.
LogicalResult ExtOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitError("requires vector type");

  // Check the number of lanes
  unsigned sourceLanes = getVectorLaneSize(sourceType);
  unsigned resultLanes = getVectorLaneSize(resultType);
  // Source lanes must be greater than result lanes
  if (sourceLanes / resultLanes <= 1)
    return emitError("lanes in source vector must be at least "
                     "twice that of result vector");
  // Source lanes must be a multiple of result lanes
  if (sourceLanes % resultLanes != 0)
    return emitError("lanes in result vector must be a multiple "
                     "of source vector lanes");

  // Verify validity of index
  unsigned factor = sourceLanes / resultLanes;
  if (static_cast<unsigned>(getIndex()) >= factor)
    return emitError("index out of bounds");

  // The datatype of source and result must match
  Type stype = sourceType.getElementType();
  Type rtype = resultType.getElementType();
  if (stype != rtype)
    return emitError("source and result element type must be same");

  return success();
}

// Parse Ext op.
ParseResult ExtOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source vector
  if (parser.parseOperand(source))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "requires one attribute");

  // Assert that there are two types (source and result)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification
  VectorType sourceType = llvm::dyn_cast<VectorType>(types[0]);
  VectorType resultType = llvm::dyn_cast<VectorType>(types[1]);
  if (!sourceType || !resultType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, sourceType, result.operands))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// Print out select op.
void aievec::SelectOp::print(OpAsmPrinter &p) {
  // Print the xbuff
  p << " " << getXbuff();
  // Print the start, offsets, etc. for xbuff
  if (getYbuff())
    p << ", " << getYbuff();

  // Print the attributes, but don't print attributes that are empty strings
  SmallVector<StringRef, 10> elidedAttrs;
  for (int idx = 0; idx < 2; ++idx) {
    if (getStart(idx).empty())
      elidedAttrs.push_back(getStartAttrName(idx));
    if (getOffset(idx).empty())
      elidedAttrs.push_back(getOffsetAttrName(idx));
    if (getOffsetHi(idx).empty())
      elidedAttrs.push_back(getOffsetHiAttrName(idx));
    if (getSquare(idx).empty())
      elidedAttrs.push_back(getSquareAttrName(idx));
  }
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << getXbuff().getType();
  if (getYbuff())
    p << ", " << getYbuff().getType();
  p << ", " << getResult().getType();
}

// Verify select op.
LogicalResult aievec::SelectOp::verify() {
  // Verify the types
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  VectorType xbuffType = llvm::dyn_cast<VectorType>(getXbuff().getType());

  if (!resultType || !xbuffType)
    return emitError("requires vector type");

  // The underlying scalar element type of all vectors must match
  Type rtype = resultType.getElementType();
  Type xtype = xbuffType.getElementType();
  if (rtype != xtype)
    return emitError("types of result and xbuff must match");

  // If yuff is present, its vector type should be same as xbuff
  if (getYbuff()) {
    VectorType ybuffType = llvm::dyn_cast<VectorType>(getYbuff().getType());
    if (xbuffType != ybuffType)
      return emitError("types of xbuff and ybuff must match");
  }

  // Compare the lanes. xtype should have more lanes
  unsigned sourceLanes = getVectorLaneSize(xbuffType);
  unsigned resultLanes = getVectorLaneSize(resultType);
  if (sourceLanes < resultLanes)
    return emitError("xbuff cannot be smaller than result");

  return success();
}

// Parse select op.
ParseResult SelectOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::UnresolvedOperand xbuff, ybuff;

  // Parse xbuff
  if (parser.parseOperand(xbuff))
    return failure();

  // Parse optional ybuff
  ParseResult hasYbuff = parser.parseOptionalComma();
  if (hasYbuff.succeeded() && parser.parseOperand(ybuff))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  // Assert that there is at least two types
  if (types.size() < 2)
    return parser.emitError(typesLoc, "requires at least two type");

  // Some verification
  VectorType xbuffType = llvm::dyn_cast<VectorType>(types[0]);
  if (!xbuffType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType ybuffType;
  if (hasYbuff.succeeded()) {
    ybuffType = llvm::dyn_cast<VectorType>(types[1]);
    if (!ybuffType)
      return parser.emitError(typesLoc, "requires vector type");
  }
  VectorType resultType = llvm::dyn_cast<VectorType>(types.back());
  if (!resultType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the xbuff
  if (parser.resolveOperand(xbuff, xbuffType, result.operands))
    return failure();
  // Populate optional ybuff in result
  if (hasYbuff.succeeded())
    if (parser.resolveOperand(ybuff, ybuffType, result.operands))
      return failure();

  return parser.addTypeToList(resultType, result.types);
}

//===----------------------------------------------------------------------===//
// PackOp and UnpackOp
//===----------------------------------------------------------------------===//

// Print out Pack and Unpack op.
template <typename T>
static void printPackUnpackOp(OpAsmPrinter &p, T op) {
  // Print the source vector
  p << " " << op.getSource();

  // Print the attributes
  p.printOptionalAttrDict(op->getAttrs());

  // And now print the types
  p << " : " << op.getSource().getType() << ", " << op.getResult().getType();
}

void PackOp::print(OpAsmPrinter &p) { printPackUnpackOp<PackOp>(p, *this); }

void UnpackOp::print(OpAsmPrinter &p) { printPackUnpackOp<UnpackOp>(p, *this); }

// Verify Pack and Unpack op.
template <typename T>
LogicalResult verifyPackUnpackOp(T op) {
  // Verify the types
  auto sourceType = op.getSource().getType().template dyn_cast<VectorType>();
  auto resultType = op.getResult().getType().template dyn_cast<VectorType>();
  if (!sourceType || !resultType)
    return op.emitError("requires vector type");

  // The number of lanes must match
  unsigned sourceLanes = getVectorLaneSize(sourceType);
  unsigned resultLanes = getVectorLaneSize(resultType);
  if (sourceLanes != resultLanes)
    return op.emitError("The number of lanes in input and "
                        "output vector must match");

  Type stype = sourceType.getElementType();
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  Type rtype = resultType.getElementType();
  unsigned rtypeWidth = rtype.getIntOrFloatBitWidth();

  if (isa<PackOp>(op)) {
    // The datatype of source must be i16, and datatype of result must be i8
    if (stypeWidth != 16)
      return op.emitError("input must be an int16 vector");
    if (rtypeWidth != 8)
      return op.emitError("output must be an int8 vector");
  } else {
    if (stypeWidth != 8)
      return op.emitError("input must be an int8 vector");
    if (rtypeWidth != 16)
      return op.emitError("output must be an int16 vector");
  }

  return success();
}

LogicalResult PackOp::verify() { return verifyPackUnpackOp<PackOp>(*this); }

LogicalResult UnpackOp::verify() { return verifyPackUnpackOp<UnpackOp>(*this); }

// Parse Pack and Unpack op.
ParseResult parsePackUnpackOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source vector
  if (parser.parseOperand(source))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  // Currently there are no attributes in pack/unpack op
  if (!result.attributes.getAttrs().empty())
    return parser.emitError(typesLoc, "expects no attributes");

  // Assert that there are two types (source and result vectors)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification
  VectorType sourceType = llvm::dyn_cast<VectorType>(types[0]);
  VectorType resultType = llvm::dyn_cast<VectorType>(types[1]);
  if (!sourceType || !resultType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, sourceType, result.operands))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

ParseResult PackOp::parse(OpAsmParser &parser, OperationState &result) {
  return parsePackUnpackOp(parser, result);
}

ParseResult UnpackOp::parse(OpAsmParser &parser, OperationState &result) {
  return parsePackUnpackOp(parser, result);
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//

// Print out Shift op.
void ShiftOp::print(OpAsmPrinter &p) {
  // Print the lhs and rhs vectors
  p << " " << getLhs() << ", " << getRhs();

  // Print shift
  p << ", " << getShift();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getLhs().getType() << ", " << getLhs().getType() << ", "
    << getShift().getType() << ", " << getResult().getType();
}

// Verify Shift op.
LogicalResult ShiftOp::verify() {
  // Verify the types
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!resultType)
    return emitError("requires vector type");

  // lhs, rhs and result must have the same type
  VectorType lhsType = llvm::dyn_cast<VectorType>(getLhs().getType());
  VectorType rhsType = llvm::dyn_cast<VectorType>(getRhs().getType());

  if (!lhsType || !rhsType)
    return emitError("requires vector type");
  if (lhsType != resultType || rhsType != resultType)
    return emitError("All vectors must have same type");

  if (!isa<IntegerType>(getShift().getType()))
    return emitError("requires integer type");

  return success();
}

// Parse Shift op.
ParseResult ShiftOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 4> types;
  OpAsmParser::UnresolvedOperand lhs, rhs, shift;

  // Parse the source vectors
  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) || parser.parseComma() ||
      parser.parseOperand(shift))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "expects one attribute");

  // Assert that there are two types (source and result vectors)
  if (types.size() != 4)
    return parser.emitError(typesLoc, "requires four types");

  // Some verification
  VectorType lhsType = llvm::dyn_cast<VectorType>(types[0]);
  VectorType rhsType = llvm::dyn_cast<VectorType>(types[1]);
  IntegerType shiftType = llvm::dyn_cast<IntegerType>(types[2]);
  VectorType resultType = llvm::dyn_cast<VectorType>(types[3]);
  if (!lhsType || !rhsType || !resultType)
    return parser.emitError(typesLoc, "requires vector type");

  if (!shiftType)
    return parser.emitError(typesLoc, "requires integer type");

  // Populate the lhs vector, rhs vectors and shift in result
  if (parser.resolveOperand(lhs, lhsType, result.operands) ||
      parser.resolveOperand(rhs, rhsType, result.operands) ||
      parser.resolveOperand(shift, shiftType, result.operands))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

// Print out Shuffle op.
void ShuffleOp::print(OpAsmPrinter &p) {
  // Print the source vector
  p << " " << getSource();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify Shuffle op.
LogicalResult ShuffleOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType || !resultType)
    return emitError("requires vector type");

  // The number of lanes must match
  unsigned sourceLanes = getVectorLaneSize(sourceType);
  unsigned resultLanes = getVectorLaneSize(resultType);
  if (sourceLanes != resultLanes)
    return emitError("The number of lanes in input and "
                     "output vector must match");

  Type stype = sourceType.getElementType();
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  Type rtype = resultType.getElementType();
  unsigned rtypeWidth = rtype.getIntOrFloatBitWidth();

  if (stypeWidth != rtypeWidth)
    return emitError("The type width in input and "
                     "output must match");

  return success();
}

// Parse Shuffle op.
ParseResult ShuffleOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source vector
  if (parser.parseOperand(source))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  // Currently there is one attribute in shuffle op
  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "expects one attribute");

  // Some verification
  VectorType sourceType = llvm::dyn_cast<VectorType>(types[0]);
  VectorType resultType = llvm::dyn_cast<VectorType>(types[1]);
  if (!sourceType || !resultType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source vectors in result
  if (parser.resolveOperand(source, sourceType, result.operands))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

//===----------------------------------------------------------------------===//
// MulConvOp and FMAConvOp
//===----------------------------------------------------------------------===//

// MulConvOp and FMAConvOp are structurally similar, except that FMAConv op
// has few extra fields (accumulator, bool flag to indicate if it is fmsub,
// etc.). We create some specializations to print those fields specifically for
// FMAConvOp and MULConvOp.

// Print the accumulator
template <typename T>
void printAccumulator(OpAsmPrinter &p, T op);
template <>
inline void printAccumulator(OpAsmPrinter &p, aievec::FMAConvOp op) {
  p << ", " << op.getAcc();
}
template <>
inline void printAccumulator(OpAsmPrinter &p, aievec::MulConvOp op) {}

// Mark fmsub indicator as elided if the FMAElem op is not fmsub
template <typename T>
void elideFMSubAttr(T op, SmallVector<StringRef, 4> &elidedAttrs);
template <>
inline void elideFMSubAttr(FMAConvOp op,
                           SmallVector<StringRef, 4> &elidedAttrs) {
  if (!op.getFmsub())
    elidedAttrs.push_back(op.getSubAttrName());
}

template <>
inline void elideFMSubAttr(MulConvOp op,
                           SmallVector<StringRef, 4> &elidedAttrs) {}

// Print out MulConv and FMAConv op.
template <typename T>
static void printMulFMAConvOp(OpAsmPrinter &p, T op) {
  // Print the left operand
  p << " " << op.getLhs();
  // Print the right operand
  p << ", " << op.getRhs();
  // For fma op, print the accumulator
  printAccumulator(p, op);

  // Print the attributes, but don't print attributes that are empty strings
  SmallVector<StringRef, 4> elidedAttrs;
  for (int idx = 0; idx < 2; ++idx) {
    elideFMSubAttr(op, elidedAttrs);
  }
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << op.getLhs().getType() << ", " << op.getRhs().getType();
  p << ", " << op.getResult().getType();
}

void MulConvOp::print(OpAsmPrinter &p) {
  printMulFMAConvOp<aievec::MulConvOp>(p, *this);
}

void aievec::FMAConvOp::print(OpAsmPrinter &p) {
  printMulFMAConvOp<aievec::FMAConvOp>(p, *this);
}

// Verify MulConv and FMAConv op.
template <typename T>
LogicalResult verifyMulFMAConvOp(T op) {
  // Verify the types
  auto lhsType = op.getLhs().getType().template dyn_cast<VectorType>();
  auto rhsType = op.getRhs().getType().template dyn_cast<VectorType>();

  if (!lhsType || !rhsType)
    return op.emitError("requires vector type");

  unsigned M = op.getM();
  unsigned N = op.getN();

  if (M <= 0 || N <= 0 || 2 * M < M + N - 1)
    return op.emitError(
        "M and N should be larger than 0 and 2*M should be no less than M+N-1");

  auto resultType = op.getResult().getType().template dyn_cast<VectorType>();

  if (!resultType)
    return op.emitError("requires vector type");

  // Additional checks for FMAElem op
  // Get the width of the underlying scalars of all the vectors
  Type ltype = lhsType.getElementType();
  Type rtype = rhsType.getElementType();
  Type atype = resultType.getElementType();

  // lhs and rhs vector's element type must match
  if (ltype != rtype)
    return op.emitError("The element type of lhs and rhs "
                        "operand vectors must match");

  if (!ltype.isa<IntegerType>() || !rtype.isa<IntegerType>() ||
      !atype.isa<IntegerType>()) {
    return op.emitError("requires integer type");
  }

  unsigned ltypeWidth = ltype.getIntOrFloatBitWidth();
  unsigned rtypeWidth = rtype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  // Checks on the number of lanes
  unsigned accLanes = getVectorLaneSize(resultType);
  unsigned rhsLanes = getVectorLaneSize(rhsType);
  unsigned lhsLanes = getVectorLaneSize(lhsType);

  // lane size must match
  if (accLanes != M || accLanes != (rhsLanes / 2) || lhsLanes != rhsLanes) {
    return op.emitError(
        "The number of lanes in accumulator "
        "must be the same as M and the half as lhs and rhs operand");
  }

  // The integer datatype of accumulator must always be greater width
  if (ltypeWidth >= atypeWidth || rtypeWidth >= atypeWidth)
    return op.emitError("the element type of accumulator must have "
                        "wider width than that of the operand vectors");

  return success();
}

LogicalResult aievec::MulConvOp::verify() {
  return verifyMulFMAConvOp<aievec::MulConvOp>(*this);
}

LogicalResult aievec::FMAConvOp::verify() {
  return verifyMulFMAConvOp<aievec::FMAConvOp>(*this);
}

// Parse MulConv and FMAConv op.
ParseResult parseMulFMAConvOp(OpAsmParser &parser, OperationState &result,
                              bool isFMAConvOp = true) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::UnresolvedOperand lhs, rhs, acc;

  // Parse the lhs and rhs
  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs))
    return failure();

  // Parse the acc for FMA op
  if (isFMAConvOp) {
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

  // Int ops use the accumulator
  VectorType accType = llvm::dyn_cast<VectorType>(types[2]);
  if (!accType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the lhs and rhs operands, and result
  if (parser.resolveOperand(lhs, lhsType, result.operands) ||
      parser.resolveOperand(rhs, rhsType, result.operands))
    return failure();

  // Populate acc operand for FMA op
  if (isFMAConvOp) {
    if (parser.resolveOperand(acc, accType, result.operands))
      return failure();
  }

  return parser.addTypeToList(accType, result.types);
}

ParseResult MulConvOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMulFMAConvOp(parser, result, false);
}

ParseResult FMAConvOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMulFMAConvOp(parser, result, true);
}

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOps.cpp.inc"
