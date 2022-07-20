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
#include "mlir/IR/AffineMap.h"

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
  p << " " << source() << "[" << indices() << "]";
  // Now print the optional vector that links upd idx=1 with idx=0
  if (vector())
    p << ", " << vector();

  // Print the attributes, but don't print the operand segment sizes
  SmallVector<StringRef, 3> elidedAttrs;
  elidedAttrs.push_back(UPDOp::getOperandSegmentSizeAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << source().getType() << ", " << result().getType();
}

// Verify UPD op.
LogicalResult UPDOp::verify() {
  // Verify the types: source is memref, and result is vector
  MemRefType sourceType = source().getType().dyn_cast<MemRefType>();
  VectorType resultType = result().getType().dyn_cast<VectorType>();
  if (!sourceType)
    return emitError("requires memref type");
  if (!resultType)
    return emitError("requires vector type");
  if (indices().empty())
    return emitError("upd source cannot come from scalar value");

  // If this UPD op is linked to another UPD op, then verify that the linked
  // vector and the result vector match.
  if (vector()) {
    Type vecType = vector().getType().dyn_cast<VectorType>();
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
  auto memrefType = types[0].dyn_cast<MemRefType>();
  if (!memrefType)
    return parser.emitError(typesLoc, "requires memref type");
  VectorType vectorType = types[1].dyn_cast<VectorType>();
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
  result.addAttribute(
      UPDOp::getOperandSegmentSizeAttr(),
      builder.getI32VectorAttr({1, static_cast<int32_t>(indices.size()),
                                static_cast<int32_t>(hasVector.succeeded())}));

  return parser.addTypeToList(vectorType, result.types);
}

//===----------------------------------------------------------------------===//
// SRSOp
//===----------------------------------------------------------------------===//

// Print out SRS op.
void SRSOp::print(OpAsmPrinter &p) {
  // Print the source accumulator
  p << " " << source();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << source().getType() << ", " << result().getType();
}

// Verify SRS op.
LogicalResult SRSOp::verify() {
  // Verify the types
  VectorType sourceType = source().getType().dyn_cast<VectorType>();
  VectorType resultType = result().getType().dyn_cast<VectorType>();
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
  else if (atype.isa<FloatType>() && stypeWidth != atypeWidth)
    return emitError("the element type of source accumulator must be "
                     "same as the result vector");

  return success();
}

// Parse SRS op.
ParseResult SRSOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source accumulator
  if (parser.parseOperand(source))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "requires one attribute");

  // Assert that there are two types (accumulator source and vector result)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification of types
  VectorType accType = types[0].dyn_cast<VectorType>();
  if (!accType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType vectorType = types[1].dyn_cast<VectorType>();
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, accType, result.operands))
    return failure();

  return parser.addTypeToList(vectorType, result.types);
}

//===----------------------------------------------------------------------===//
// UPSOp
//===----------------------------------------------------------------------===//

// Print out UPS op.
void UPSOp::print(OpAsmPrinter &p) {
  // Print the source vector
  p << " " << source();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << source().getType() << ", " << result().getType();
}

// Verify UPS op.
LogicalResult UPSOp::verify() {
  // Verify the types
  VectorType sourceType = source().getType().dyn_cast<VectorType>();
  VectorType resultType = result().getType().dyn_cast<VectorType>();
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

  if (atype.isa<IntegerType>() && stypeWidth >= atypeWidth)
    return emitError("the element type of result accumulator "
                     "must be wider than that of the source vector");
  else if (atype.isa<FloatType>() && stypeWidth != atypeWidth)
    return emitError("the element type of result accumulator must "
                     "be same as the source vector");

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
  VectorType vectorType = types[0].dyn_cast<VectorType>();
  if (!vectorType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType accType = types[1].dyn_cast<VectorType>();
  if (!accType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, vectorType, result.operands))
    return failure();

  return parser.addTypeToList(accType, result.types);
}

//===----------------------------------------------------------------------===//
// MulOp and FMAOp
//===----------------------------------------------------------------------===//

// MulOp and FMAOp are structurally similar, except that FMA op has few extra
// fields (accumulator, bool flag to indicate if it is fmsub, etc.). We create
// some specializations to print those fields specifically for FMA op.

// Print the accumulator
template <typename T> inline void printAccumulator(OpAsmPrinter &p, T op);
template <> inline void printAccumulator(OpAsmPrinter &p, aievec::FMAOp op) {
  p << ", " << op.acc();
}
template <> inline void printAccumulator(OpAsmPrinter &p, aievec::MulOp op) {}

// Mark fmsub indicator as elided if the FMA op is not fmsub
template <typename T>
inline void elideFMSubAttr(T op, SmallVector<StringRef, 10> &elidedAttrs);
template <>
inline void elideFMSubAttr(aievec::FMAOp op,
                           SmallVector<StringRef, 10> &elidedAttrs) {
  if (!op.fmsub())
    elidedAttrs.push_back(op.getSubAttrName());
}
template <>
inline void elideFMSubAttr(aievec::MulOp,
                           SmallVector<StringRef, 10> &elidedAttrs) {}

// Print out Mul and FMA op.
template <typename T> static void printMulFMAOp(OpAsmPrinter &p, T op) {
  // Print the left operand
  p << " " << op.lhs();
  // Print the right operand
  p << ", " << op.rhs();
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
  p << " : " << op.lhs().getType() << ", " << op.rhs().getType();
  p << ", " << op.result().getType();
}

void MulOp::print(OpAsmPrinter &p) { printMulFMAOp<aievec::MulOp>(p, *this); }

void aievec::FMAOp::print(OpAsmPrinter &p) {
  printMulFMAOp<aievec::FMAOp>(p, *this);
}

// Verify Mul and FMA op.
template <typename T> LogicalResult verifyMulFMAOp(T op) {
  // Verify the types
  VectorType lhsType = op.lhs().getType().template dyn_cast<VectorType>();
  VectorType rhsType = op.rhs().getType().template dyn_cast<VectorType>();

  if (!lhsType || !rhsType)
    return op.emitError("requires vector type");

  VectorType resultType = op.result().getType().template dyn_cast<VectorType>();
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
  VectorType lhsType = types[0].dyn_cast<VectorType>();
  if (!lhsType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType rhsType = types[1].dyn_cast<VectorType>();
  if (!rhsType)
    return parser.emitError(typesLoc, "requires vector type");

  // Int ops use the accumulator while float ops use normal vector registers
  VectorType accType = types[2].dyn_cast<VectorType>();
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
// AddOp and SubOp
//===----------------------------------------------------------------------===//

// Print out Add and Sub op.
template <typename T> void printAddSubOp(OpAsmPrinter &p, T op) {
  // Print the lhs operand
  p << " " << op.lhs();
  // Print the rhs operand
  p << ", " << op.rhs();

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
  p << " : " << op.lhs().getType() << ", " << op.rhs().getType();
  p << ", " << op.result().getType();
}

void aievec::AddOp::print(OpAsmPrinter &p) {
  printAddSubOp<aievec::AddOp>(p, *this);
}

void aievec::SubOp::print(OpAsmPrinter &p) {
  printAddSubOp<aievec::SubOp>(p, *this);
}

// Verify Add and Sub op.
template <typename T> LogicalResult verifyAddSubOp(T op) {
  // Verify the types
  VectorType resultType = op.result().getType().template dyn_cast<VectorType>();
  VectorType lhsType = op.lhs().getType().template dyn_cast<VectorType>();
  VectorType rhsType = op.rhs().getType().template dyn_cast<VectorType>();

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
  VectorType lhsType = types[0].dyn_cast<VectorType>();
  if (!lhsType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType rhsType = types[1].dyn_cast<VectorType>();
  if (!rhsType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType resultType = types[2].dyn_cast<VectorType>();
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
  assert(!sources().empty() && "concat source empty");
  p << " " << sources();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << sources().getTypes().front() << ", " << result().getType();
}

// Verify Concat op.
LogicalResult ConcatOp::verify() {
  // Must be concatenating at least two sources
  if (sources().size() < 2)
    return emitError("Must concatenate at least two vectors");

  // Verify the types
  VectorType sourceType = sources().getTypes().front().dyn_cast<VectorType>();
  VectorType resultType = result().getType().dyn_cast<VectorType>();
  if (!sourceType || !resultType)
    return emitError("requires vector type");

  SmallVector<Value, 8> srcs(sources().begin(), sources().end());
  // All the sources must have the same type
  for (auto source : srcs) {
    VectorType type = source.getType().dyn_cast<VectorType>();
    if (!type)
      return emitError("requires vector type");
    if (type != sourceType)
      return emitError("All sources must have same type");
  }

  // The lanes in concatenated type must be the sum of lanes of source vector
  unsigned totalLanes = 0;
  for (auto source : srcs) {
    VectorType type = source.getType().dyn_cast<VectorType>();
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
  VectorType sourceType = types[0].dyn_cast<VectorType>();
  VectorType resultType = types[1].dyn_cast<VectorType>();
  if (!sourceType || !resultType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source vectors in result
  if (parser.resolveOperands(sources, sourceType, result.operands))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

//===----------------------------------------------------------------------===//
// ExtOp
//===----------------------------------------------------------------------===//

// Print out Ext op.
void ExtOp::print(OpAsmPrinter &p) {
  // Print the source vector
  p << " " << source();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << source().getType() << ", " << result().getType();
}

// Verify Ext op.
LogicalResult ExtOp::verify() {
  // Verify the types
  VectorType sourceType = source().getType().dyn_cast<VectorType>();
  VectorType resultType = result().getType().dyn_cast<VectorType>();
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
  if (index() >= factor)
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
  VectorType sourceType = types[0].dyn_cast<VectorType>();
  VectorType resultType = types[1].dyn_cast<VectorType>();
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
  p << " " << xbuff();
  // Print the start, offsets, etc. for xbuff
  if (ybuff())
    p << ", " << ybuff();

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
  p << " : " << xbuff().getType();
  if (ybuff())
    p << ", " << ybuff().getType();
  p << ", " << result().getType();
}

// Verify select op.
LogicalResult aievec::SelectOp::verify() {
  // Verify the types
  VectorType resultType = result().getType().dyn_cast<VectorType>();
  VectorType xbuffType = xbuff().getType().dyn_cast<VectorType>();

  if (!resultType || !xbuffType)
    return emitError("requires vector type");

  // The underlying scalar element type of all vectors must match
  Type rtype = resultType.getElementType();
  Type xtype = xbuffType.getElementType();
  if (rtype != xtype)
    return emitError("types of result and xbuff must match");

  // If yuff is present, its vector type should be same as xbuff
  if (ybuff()) {
    VectorType ybuffType = ybuff().getType().dyn_cast<VectorType>();
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
  VectorType xbuffType = types[0].dyn_cast<VectorType>();
  if (!xbuffType)
    return parser.emitError(typesLoc, "requires vector type");
  VectorType ybuffType;
  if (hasYbuff.succeeded()) {
    ybuffType = types[1].dyn_cast<VectorType>();
    if (!ybuffType)
      return parser.emitError(typesLoc, "requires vector type");
  }
  VectorType resultType = types.back().dyn_cast<VectorType>();
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
template <typename T> static void printPackUnpackOp(OpAsmPrinter &p, T op) {
  // Print the source vector
  p << " " << op.source();

  // Print the attributes
  p.printOptionalAttrDict(op->getAttrs());

  // And now print the types
  p << " : " << op.source().getType() << ", " << op.result().getType();
}

void PackOp::print(OpAsmPrinter &p) { printPackUnpackOp<PackOp>(p, *this); }

void UnpackOp::print(OpAsmPrinter &p) { printPackUnpackOp<UnpackOp>(p, *this); }

// Verify Pack and Unpack op.
template <typename T> LogicalResult verifyPackUnpackOp(T op) {
  // Verify the types
  VectorType sourceType = op.source().getType().template dyn_cast<VectorType>();
  VectorType resultType = op.result().getType().template dyn_cast<VectorType>();
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
  VectorType sourceType = types[0].dyn_cast<VectorType>();
  VectorType resultType = types[1].dyn_cast<VectorType>();
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

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOps.cpp.inc"
