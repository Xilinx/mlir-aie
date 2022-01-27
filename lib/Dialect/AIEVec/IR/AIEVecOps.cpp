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
static void print(OpAsmPrinter &p, UPDOp upd) {
  // Print the source memref
  p << " " << upd.source() << "[" << upd.indices() << "]";
  // Now print the optional vector that links upd idx=1 with idx=0
  if (upd.vector())
    p << ", " << upd.vector();

  // Print the attributes, but don't print the operand segment sizes
  SmallVector<StringRef, 3> elidedAttrs;
  elidedAttrs.push_back(UPDOp::getOperandSegmentSizeAttr());
  p.printOptionalAttrDict(upd->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << upd.source().getType() << ", " << upd.result().getType();
}

// Verify UPD op.
static LogicalResult verify(UPDOp upd) {
  // Verify the types: source is memref, and result is vector
  MemRefType sourceType = upd.source().getType().dyn_cast<MemRefType>();
  VectorType resultType = upd.result().getType().dyn_cast<VectorType>();
  if (!sourceType)
    return upd.emitError("requires memref type");
  if (!resultType)
    return upd.emitError("requires vector type");
  if (upd.indices().empty())
    return upd.emitError("upd source cannot come from scalar value");

  // If this UPD op is linked to another UPD op, then verify that the linked
  // vector and the result vector match.
  if (upd.vector()) {
    Type vecType = upd.vector().getType().dyn_cast<VectorType>();
    if (vecType != resultType)
      return upd.emitError("result types of linked UPD ops do not match");
  }
  return success();
}

// Parse UPD op.
static ParseResult parseUPDOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType source, vector;
  SmallVector<OpAsmParser::OperandType, 8> indices;

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
static void print(OpAsmPrinter &p, SRSOp srs) {
  // Print the source accumulator
  p << " " << srs.source();

  // Print the attributes
  p.printOptionalAttrDict(srs->getAttrs());

  // And now print the types
  p << " : " << srs.source().getType() << ", " << srs.result().getType();
}

// Verify SRS op.
static LogicalResult verify(SRSOp srs) {
  // Verify the types
  aievec::AccType sourceType =
      srs.source().getType().dyn_cast<aievec::AccType>();
  VectorType resultType = srs.result().getType().dyn_cast<VectorType>();
  if (!sourceType)
    return srs.emitError("requires accumulator type");
  if (!resultType)
    return srs.emitError("requires vector type");

  // The number of lanes of source accumulator and result vector must match
  unsigned accLanes = sourceType.getLanes();
  unsigned vecLanes = getVectorLaneSize(resultType);
  if (accLanes != vecLanes)
    return srs.emitError("The number of lanes in result vector "
                         "and source accumulator must match");

  // The datatype of accumulator must have greater width
  Type stype = resultType.getElementType();
  Type atype = sourceType.getValueType();
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  if (atype.isa<IntegerType>() && stypeWidth >= atypeWidth)
    return srs.emitError("the element type of source accumulator must be "
                         "wider than that of the result vector");
  else if (atype.isa<FloatType>() && stypeWidth != atypeWidth)
    return srs.emitError("the element type of source accumulator must be "
                         "same as the result vector");

  return success();
}

// Parse SRS op.
static ParseResult parseSRSOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType source;

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
  aievec::AccType accType = types[0].dyn_cast<aievec::AccType>();
  if (!accType)
    return parser.emitError(typesLoc, "requires accumulator type");
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
static void print(OpAsmPrinter &p, UPSOp ups) {
  // Print the source vector
  p << " " << ups.source();

  // Print the attributes
  p.printOptionalAttrDict(ups->getAttrs());

  // And now print the types
  p << " : " << ups.source().getType() << ", " << ups.result().getType();
}

// Verify UPS op.
static LogicalResult verify(UPSOp ups) {
  // Verify the types
  VectorType sourceType = ups.source().getType().dyn_cast<VectorType>();
  aievec::AccType resultType =
      ups.result().getType().dyn_cast<aievec::AccType>();
  if (!sourceType)
    return ups.emitError("requires vector type");
  if (!resultType)
    return ups.emitError("requires accumulator type");

  // The number of lanes must match
  unsigned vecLanes = getVectorLaneSize(sourceType);
  unsigned accLanes = resultType.getLanes();
  if (vecLanes != accLanes)
    return ups.emitError("The number of lanes in source vector "
                         "and result accumulator must match");

  // The datatype of accumulator must always be greater width
  Type stype = sourceType.getElementType();
  Type atype = resultType.getValueType();
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  if (atype.isa<IntegerType>() && stypeWidth >= atypeWidth)
    return ups.emitError("the element type of result accumulator "
                         "must be wider than that of the source vector");
  else if (atype.isa<FloatType>() && stypeWidth != atypeWidth)
    return ups.emitError("the element type of result accumulator must "
                         "be same as the source vector");

  return success();
}

// Parse UPS op.
static ParseResult parseUPSOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType source;

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
  aievec::AccType accType = types[1].dyn_cast<aievec::AccType>();
  if (!accType)
    return parser.emitError(typesLoc, "requires accumulator type");

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
template <> inline void elideFMSubAttr(aievec::FMAOp op,
                                SmallVector<StringRef, 10> &elidedAttrs) {
  if (!op.fmsub())
    elidedAttrs.push_back(op.getSubAttrName());
}
template <> inline void elideFMSubAttr(aievec::MulOp, 
                                SmallVector<StringRef, 10> &elidedAttrs) {}

// Verification checks for accumulator field of FMA op
template <typename T>
inline LogicalResult verifyAccType(T op, aievec::AccType resultType);
template <> inline LogicalResult verifyAccType(aievec::FMAOp op, 
                                               aievec::AccType resultType) {
  aievec::AccType accType =
        op.acc().getType().dyn_cast<aievec::AccType>();
  if (!accType)
    return op.emitError("requires accumulator type");
  if (resultType != accType)
    return op.emitError("the result type and accumulator type must match");
  return success();
}
template <> inline LogicalResult verifyAccType(aievec::MulOp op, 
                                               aievec::AccType resultType) {
  return success();
}

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

static void print(OpAsmPrinter &p, aievec::MulOp mul) {
  printMulFMAOp<aievec::MulOp>(p, mul);
}

static void print(OpAsmPrinter &p, aievec::FMAOp fma) {
  printMulFMAOp<aievec::FMAOp>(p, fma);
}

// Verify Mul and FMA op.
template <typename T> static LogicalResult verifyMulFMAOp(T op) {
  // Verify the types
  aievec::AccType resultType =
      op.result().getType().template dyn_cast<aievec::AccType>();
  VectorType lhsType = op.lhs().getType().template dyn_cast<VectorType>();
  VectorType rhsType = op.rhs().getType().template dyn_cast<VectorType>();

  if (!lhsType || !rhsType)
    return op.emitError("requires vector type");
  if (!resultType)
    return op.emitError("requires accumulator type");

  // Additional checks for FMA op
  if (failed(verifyAccType(op, resultType)))
    return failure();

  // Get the width of the underlying scalars of all the vectors
  Type ltype = lhsType.getElementType();
  Type rtype = rhsType.getElementType();
  Type atype = resultType.getValueType();
  unsigned ltypeWidth = ltype.getIntOrFloatBitWidth();
  unsigned rtypeWidth = rtype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  // Checks on the number of lanes
  unsigned accLanes = resultType.getLanes();
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
    if (ltypeWidth >= atypeWidth || rtypeWidth >= atypeWidth)
      return op.emitError("the element type of accumulator must have "
                           "wider width than that of the operand vectors");
  } else if (atype.isa<FloatType>()) {
    if (ltypeWidth != atypeWidth || rtypeWidth != atypeWidth)
      return op.emitError("the element type of accumulator must be "
                           "same width as the operand vectors");
  }

  return success();
}

static LogicalResult verify(aievec::MulOp op) {
  return verifyMulFMAOp<aievec::MulOp>(op);
}

static LogicalResult verify(aievec::FMAOp op) {
  return verifyMulFMAOp<aievec::FMAOp>(op);
}

// Parse Mul and FMA op.
static ParseResult parseMulFMAOp(OpAsmParser &parser, OperationState &result, 
                                 bool isFMAOp = true) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::OperandType lhs, rhs, acc;

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
  aievec::AccType accType = types[2].dyn_cast<aievec::AccType>();
  if (!accType)
    return parser.emitError(typesLoc, "requires accumulator type");

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

static ParseResult parseMulOp(OpAsmParser &parser, OperationState &result) {
  return parseMulFMAOp(parser, result, false);
}

static ParseResult parseFMAOp(OpAsmParser &parser, OperationState &result) {
  return parseMulFMAOp(parser, result, true);
}

//===----------------------------------------------------------------------===//
// AddOp and SubOp
//===----------------------------------------------------------------------===//

// Print out Add and Sub op.
template <typename T> static void printAddSubOp(OpAsmPrinter &p, T op) {
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

static void print(OpAsmPrinter &p, aievec::AddOp add) {
  printAddSubOp<aievec::AddOp>(p, add);  
}

static void print(OpAsmPrinter &p, aievec::SubOp sub) {
  printAddSubOp<aievec::SubOp>(p, sub);  
}

// Verify Add and Sub op.
template <typename T> static LogicalResult verifyAddSubOp(T op) {
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

static LogicalResult verify(aievec::AddOp add) {
  return verifyAddSubOp<aievec::AddOp>(add);
}

static LogicalResult verify(aievec::SubOp sub) {
  return verifyAddSubOp<aievec::SubOp>(sub);
}

// Parse Add and Sub op.
static ParseResult parseAddSubOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::OperandType lhs, rhs;

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

static ParseResult parseAddOp(OpAsmParser &parser, OperationState &result) {
  return parseAddSubOp(parser, result);
}

static ParseResult parseSubOp(OpAsmParser &parser, OperationState &result) {
  return parseAddSubOp(parser, result);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// Print out Concat op.
static void print(OpAsmPrinter &p, ConcatOp concat) {
  // Print the source vectors
  assert(!concat.sources().empty() && "concat source empty");
  p << " " << concat.sources();

  // Print the attributes
  p.printOptionalAttrDict(concat->getAttrs());

  // And now print the types
  p << " : " << concat.sources().getTypes().front() << ", "
    << concat.result().getType();
}

// Verify Concat op.
static LogicalResult verify(ConcatOp concat) {
  // Must be concatenating at least two sources
  if (concat.sources().size() < 2)
    return concat.emitError("Must concatenate at least two vectors");

  // Verify the types
  VectorType sourceType =
      concat.sources().getTypes().front().dyn_cast<VectorType>();
  VectorType resultType = concat.result().getType().dyn_cast<VectorType>();
  if (!sourceType || !resultType)
    return concat.emitError("requires vector type");

  SmallVector<Value, 8> sources(concat.sources().begin(),
                                concat.sources().end());
  // All the sources must have the same type
  for (auto source : sources) {
    VectorType type = source.getType().dyn_cast<VectorType>();
    if (!type)
      return concat.emitError("requires vector type");
    if (type != sourceType)
      return concat.emitError("All sources must have same type");
  }

  // The lanes in concatenated type must be the sum of lanes of source vector
  unsigned totalLanes = 0;
  for (auto source : sources) {
    VectorType type = source.getType().dyn_cast<VectorType>();
    totalLanes += getVectorLaneSize(type);
  }
  if (totalLanes != getVectorLaneSize(resultType))
    return concat.emitError("mismatch between vector lanes "
                            "and sum of source lanes");

  return success();
}

// Parse Concat op.
static ParseResult parseConcatOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  SmallVector<OpAsmParser::OperandType, 8> sources;

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
static void print(OpAsmPrinter &p, ExtOp ext) {
  // Print the source vector
  p << " " << ext.source();

  // Print the attributes
  p.printOptionalAttrDict(ext->getAttrs());

  // And now print the types
  p << " : " << ext.source().getType() << ", " << ext.result().getType();
}

// Verify Ext op.
static LogicalResult verify(ExtOp ext) {
  // Verify the types
  VectorType sourceType = ext.source().getType().dyn_cast<VectorType>();
  VectorType resultType = ext.result().getType().dyn_cast<VectorType>();
  if (!sourceType || !resultType)
    return ext.emitError("requires vector type");

  // Check the number of lanes
  unsigned sourceLanes = getVectorLaneSize(sourceType);
  unsigned resultLanes = getVectorLaneSize(resultType);
  // Source lanes must be greater than result lanes
  if (sourceLanes / resultLanes <= 1)
    return ext.emitError("lanes in source vector must be at least "
                         "twice that of result vector");
  // Source lanes must be a multiple of result lanes
  if (sourceLanes % resultLanes != 0)
    return ext.emitError("lanes in result vector must be a multiple "
                         "of source vector lanes");

  // Verify validity of index
  unsigned factor = sourceLanes / resultLanes;
  if (ext.index() >= factor)
    return ext.emitError("index out of bounds");

  // The datatype of source and result must match
  Type stype = sourceType.getElementType();
  Type rtype = resultType.getElementType();
  if (stype != rtype)
    return ext.emitError("source and result element type must be same");

  return success();
}

// Parse Ext op.
static ParseResult parseExtOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType source;

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
static void print(OpAsmPrinter &p, aievec::SelectOp sop) {
  // Print the xbuff
  p << " " << sop.xbuff();
  // Print the start, offsets, etc. for xbuff
  if (sop.ybuff())
    p << ", " << sop.ybuff();

  // Print the attributes, but don't print attributes that are empty strings
  SmallVector<StringRef, 10> elidedAttrs;
  for (int idx = 0; idx < 2; ++idx) {
    if (sop.getStart(idx).empty())
      elidedAttrs.push_back(sop.getStartAttrName(idx));
    if (sop.getOffset(idx).empty())
      elidedAttrs.push_back(sop.getOffsetAttrName(idx));
    if (sop.getOffsetHi(idx).empty())
      elidedAttrs.push_back(sop.getOffsetHiAttrName(idx));
    if (sop.getSquare(idx).empty())
      elidedAttrs.push_back(sop.getSquareAttrName(idx));
  }
  p.printOptionalAttrDict(sop->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << sop.xbuff().getType();
  if (sop.ybuff())
    p << ", " << sop.ybuff().getType();
  p << ", " << sop.result().getType();
}

// Verify select op.
static LogicalResult verify(aievec::SelectOp sop) {
  // Verify the types
  VectorType resultType = sop.result().getType().dyn_cast<VectorType>();
  VectorType xbuffType = sop.xbuff().getType().dyn_cast<VectorType>();

  if (!resultType || !xbuffType)
    return sop.emitError("requires vector type");

  // The underlying scalar element type of all vectors must match
  Type rtype = resultType.getElementType();
  Type xtype = xbuffType.getElementType();
  if (rtype != xtype)
    return sop.emitError("types of result and xbuff must match");

  // If yuff is present, its vector type should be same as xbuff
  if (sop.ybuff()) {
    VectorType ybuffType = sop.ybuff().getType().dyn_cast<VectorType>();
    if (xbuffType != ybuffType)
      return sop.emitError("types of xbuff and ybuff must match");
  }

  // Compare the lanes. xtype should have more lanes
  unsigned sourceLanes = getVectorLaneSize(xbuffType);
  unsigned resultLanes = getVectorLaneSize(resultType);
  if (sourceLanes < resultLanes)
    return sop.emitError("xbuff cannot be smaller than result");

  return success();
}

// Parse select op.
static ParseResult parseSelectOp(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::OperandType xbuff, ybuff;

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
template <typename T> static void printPackUnpackOp(OpAsmPrinter &p,T op) {
  // Print the source vector 
  p << " " << op.source();

  // Print the attributes
  p.printOptionalAttrDict(op->getAttrs());

  // And now print the types
  p << " : " << op.source().getType() << ", " << op.result().getType();
}

static void print(OpAsmPrinter &p, PackOp pack) {
  printPackUnpackOp<PackOp>(p, pack);
}

static void print(OpAsmPrinter &p, UnpackOp unpack) {
  printPackUnpackOp<UnpackOp>(p, unpack);
}

// Verify Pack and Unpack op.
template <typename T> static LogicalResult verifyPackUnpackOp(T op) {
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
  }
  else {
    if (stypeWidth != 8)
      return op.emitError("input must be an int8 vector");
    if (rtypeWidth != 16)
      return op.emitError("output must be an int16 vector");
  }

  return success();
}

static LogicalResult verify(PackOp pack) {
  return verifyPackUnpackOp<PackOp>(pack);
}

static LogicalResult verify(UnpackOp unpack) {
  return verifyPackUnpackOp<UnpackOp>(unpack);
}

// Parse Pack and Unpack op.
static ParseResult parsePackUnpackOp(OpAsmParser &parser, 
                                     OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType source;

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

static ParseResult parsePackOp(OpAsmParser &parser, OperationState &result) {
  return parsePackUnpackOp(parser, result);
}

static ParseResult parseUnpackOp(OpAsmParser &parser, OperationState &result) {
  return parsePackUnpackOp(parser, result);
}

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOps.cpp.inc"
