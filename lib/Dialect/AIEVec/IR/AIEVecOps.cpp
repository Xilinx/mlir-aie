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

#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
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
static ParseResult parseUPDOp(OpAsmParser &parser,
                              OperationState &result) {
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
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
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
  aievec::AccType sourceType = srs.source().getType().dyn_cast<aievec::AccType>();
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
static ParseResult parseSRSOp(OpAsmParser &parser,
                              OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType source;

  // Parse the source accumulator
  if (parser.parseOperand(source)) 
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
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
  aievec::AccType resultType = ups.result().getType().dyn_cast<aievec::AccType>();
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
static ParseResult parseUPSOp(OpAsmParser &parser,
                               OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType source;

  // Parse the source vector 
  if (parser.parseOperand(source)) 
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
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
// FMAOp
//===----------------------------------------------------------------------===//

// Print out FMA op.
static void print(OpAsmPrinter &p, aievec::FMAOp fma) {
  // Print the left operand
  p << " " << fma.lhs();
  // Print the right operand
  p << ", " << fma.rhs();
  // Print the accumulator
  p << ", " << fma.acc();

  // Print the attributes, but don't print attributes that are empty strings
  SmallVector<StringRef, 10> elidedAttrs;
  for (int idx=0; idx<2; ++idx) {
    if (fma.getStart(idx).empty())
      elidedAttrs.push_back(fma.getStartAttrName(idx));
    if (fma.getOffset(idx).empty())
      elidedAttrs.push_back(fma.getOffsetAttrName(idx));
    if (fma.getOffsetHi(idx).empty())
      elidedAttrs.push_back(fma.getOffsetHiAttrName(idx));
    if (fma.getStep(idx).empty())
      elidedAttrs.push_back(fma.getStepAttrName(idx));
    if (fma.getSquare(idx).empty())
      elidedAttrs.push_back(fma.getSquareAttrName(idx));
    if (!fma.fmsub())
      elidedAttrs.push_back(fma.getSubAttrName());
  }
  p.printOptionalAttrDict(fma->getAttrs(), elidedAttrs);

  assert(fma.acc().getType() == fma.result().getType());
  // And now print the types
  p << " : " << fma.lhs().getType() << ", " << fma.rhs().getType();
  p << ", " << fma.result().getType();
}

// Verify FMA op.
static LogicalResult verify(aievec::FMAOp fma) {
  // Verify the types
  aievec::AccType accType = fma.acc().getType().dyn_cast<aievec::AccType>();
  aievec::AccType resultType = fma.result().getType().dyn_cast<aievec::AccType>();
  VectorType lhsType = fma.lhs().getType().dyn_cast<VectorType>();
  VectorType rhsType = fma.rhs().getType().dyn_cast<VectorType>();

  if (!lhsType || !rhsType)
    return fma.emitError("requires vector type");
  if (!resultType || !accType)
    return fma.emitError("requires accumulator type");
  if (resultType != accType)
    return fma.emitError("the result type and accumulator type must match");

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
    if (rhsLanes != 256/rtypeWidth)
      return fma.emitError("incorrect rhs operand vector lanes");
    if (lhsLanes < 2*rhsLanes)
      return fma.emitError("The number of lanes in lhs operand "
                        "must be at least twice that of rhs operand");
    if (accLanes > rhsLanes)
      return fma.emitError("The number of lanes in accumulator "
                           "must be less than that of rhs operand");
  }
  // lhs and rhs vector's element type must match
  if (ltype != rtype)
    return fma.emitError("The element type of lhs and rhs "
                         "operand vectors must match");

  // The datatype of accumulator must always be greater width
  if (atype.isa<IntegerType>()) {
    if (ltypeWidth >= atypeWidth || rtypeWidth >= atypeWidth)
      return fma.emitError("the element type of accumulator must have "
                          "wider width than that of the operand vectors");
  }
  else if (atype.isa<FloatType>()) {
    if (ltypeWidth != atypeWidth || rtypeWidth != atypeWidth)
      return fma.emitError("the element type of accumulator must be "
                          "same width as the operand vectors");
  }
  return success();
}

// Parse FMA op.
static ParseResult parseFMAOp(OpAsmParser &parser,
                              OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::OperandType lhs, rhs, acc;

  // Parse the lhs, rhs, and accumulator
  if (parser.parseOperand(lhs) ||
      parser.parseComma() ||
      parser.parseOperand(rhs) ||
      parser.parseComma() ||
      parser.parseOperand(acc))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
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

  // Populate the lhs, rhs, and accumulator in the result
  if (parser.resolveOperand(lhs, lhsType, result.operands) ||
      parser.resolveOperand(rhs, rhsType, result.operands) || 
      parser.resolveOperand(acc, accType, result.operands))
    return failure();

  return parser.addTypeToList(accType, result.types);
}

//===----------------------------------------------------------------------===//
// MulOp 
//===----------------------------------------------------------------------===//

// Print out Mul op.
static void print(OpAsmPrinter &p, aievec::MulOp mul) {
  // Print the lhs operand
  p << " " << mul.lhs();
  // Print the rhs operand
  p << ", " << mul.rhs();

  // Print the attributes, but don't print attributes that are empty strings
  SmallVector<StringRef, 10> elidedAttrs;
  for (int idx=0; idx<2; ++idx) {
    if (mul.getStart(idx).empty())
      elidedAttrs.push_back(mul.getStartAttrName(idx));
    if (mul.getOffset(idx).empty())
      elidedAttrs.push_back(mul.getOffsetAttrName(idx));
    if (mul.getOffsetHi(idx).empty())
      elidedAttrs.push_back(mul.getOffsetHiAttrName(idx));
    if (mul.getStep(idx).empty())
      elidedAttrs.push_back(mul.getStepAttrName(idx));
    if (mul.getSquare(idx).empty())
      elidedAttrs.push_back(mul.getSquareAttrName(idx));
  }
  p.printOptionalAttrDict(mul->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << mul.lhs().getType() << ", " << mul.rhs().getType();
  p << ", " << mul.result().getType();
}

// Verify Mul op.
static LogicalResult verify(aievec::MulOp mul) {
  // Verify the types
  aievec::AccType resultType = mul.result().getType().dyn_cast<aievec::AccType>();
  VectorType lhsType = mul.lhs().getType().dyn_cast<VectorType>();
  VectorType rhsType = mul.rhs().getType().dyn_cast<VectorType>();

  if (!lhsType || !rhsType)
    return mul.emitError("requires vector type");
  if (!resultType)
    return mul.emitError("requires accumulator type");

  // Get the width of the underlying scalars of all the vectors
  Type ltype = lhsType.getElementType();
  Type rtype = rhsType.getElementType();
  Type atype = resultType.getValueType();
  unsigned ltypeWidth = ltype.getIntOrFloatBitWidth();
  unsigned rtypeWidth = rtype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  // Checks on number of lanes
  unsigned accLanes = resultType.getLanes();
  unsigned rhsLanes = getVectorLaneSize(rhsType);
  unsigned lhsLanes = getVectorLaneSize(lhsType);

  // If this is not a simple scheme, perform complex checks
  if (accLanes != rhsLanes || accLanes != lhsLanes) {
    if (rhsLanes != 256/rtypeWidth)
      return mul.emitError("incorrect rhs operand vector lanes");
    if (lhsLanes < 2*rhsLanes)
      return mul.emitError("The number of lanes in lhs operand "
                        "must be at least twice that of rhs operand");
    if (accLanes > rhsLanes)
      return mul.emitError("The number of lanes in accumulator "
                           "must be less than that of rhs operand");
  }

  // lhs and rhs vector's element type must match
  if (ltype != rtype)
    return mul.emitError("The element type of lhs and rhs "
                         "operand vectors must match");

  // The datatype of accumulator must always be greater width
  if (atype.isa<IntegerType>()) {
    if (ltypeWidth >= atypeWidth || rtypeWidth >= atypeWidth)
      return mul.emitError("the element type of accumulator must have "
                          "wider width than that of the operand vectors");
  }
  else if (atype.isa<FloatType>()) {
    if (ltypeWidth != atypeWidth || rtypeWidth != atypeWidth)
      return mul.emitError("the element type of accumulator must be "
                          "same width as the operand vectors");
  }

  return success();
}

// Parse Mul op.
static ParseResult parseMulOp(OpAsmParser &parser,
                              OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::OperandType lhs, rhs;

  // Parse the lhs and rhs
  if (parser.parseOperand(lhs) ||
      parser.parseComma() ||
      parser.parseOperand(rhs)) 
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
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

  // Populate the lhs, rhs, and accumulator in the result
  if (parser.resolveOperand(lhs, lhsType, result.operands) ||
      parser.resolveOperand(rhs, rhsType, result.operands)) 
    return failure();

  return parser.addTypeToList(accType, result.types);
}

//===----------------------------------------------------------------------===//
// AddOp 
//===----------------------------------------------------------------------===//

// Print out Add op.
static void print(OpAsmPrinter &p, aievec::AddOp add) {
  // Print the lhs operand
  p << " " << add.lhs();
  // Print the rhs operand
  p << ", " << add.rhs();

  // Print the attributes, but don't print attributes that are empty strings
  SmallVector<StringRef, 10> elidedAttrs;
  for (int idx=0; idx<2; ++idx) {
    if (add.getStart(idx).empty())
      elidedAttrs.push_back(add.getStartAttrName(idx));
    if (add.getOffset(idx).empty())
      elidedAttrs.push_back(add.getOffsetAttrName(idx));
    if (add.getOffsetHi(idx).empty())
      elidedAttrs.push_back(add.getOffsetHiAttrName(idx));
    if (add.getSquare(idx).empty())
      elidedAttrs.push_back(add.getSquareAttrName(idx));
  }
  p.printOptionalAttrDict(add->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << add.lhs().getType() << ", " << add.rhs().getType();
  p << ", " << add.result().getType();
}

// Verify Add op.
static LogicalResult verify(aievec::AddOp add) {
  // Verify the types
  VectorType resultType = add.result().getType().dyn_cast<VectorType>();
  VectorType lhsType = add.lhs().getType().dyn_cast<VectorType>();
  VectorType rhsType = add.rhs().getType().dyn_cast<VectorType>();

  if (!lhsType || !rhsType || !resultType)
    return add.emitError("requires vector type");

  // All the vector types must match
  if (lhsType != rhsType || 
      rhsType != resultType)
    return add.emitError("all vectors must be of same type");

  return success();
}

// Parse Add op.
static ParseResult parseAddOp(OpAsmParser &parser,
                              OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::OperandType lhs, rhs;

  // Parse the lhs and rhs
  if (parser.parseOperand(lhs) ||
      parser.parseComma() ||
      parser.parseOperand(rhs)) 
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
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

//===----------------------------------------------------------------------===//
// SubOp 
//===----------------------------------------------------------------------===//

// Print out Sub op.
static void print(OpAsmPrinter &p, aievec::SubOp sub) {
  // Print the lhs operand
  p << " " << sub.lhs();
  // Print the rhs operand
  p << ", " << sub.rhs();

  // Print the attributes, but don't print attributes that are empty strings
  SmallVector<StringRef, 10> elidedAttrs;
  for (int idx=0; idx<2; ++idx) {
    if (sub.getStart(idx).empty())
      elidedAttrs.push_back(sub.getStartAttrName(idx));
    if (sub.getOffset(idx).empty())
      elidedAttrs.push_back(sub.getOffsetAttrName(idx));
    if (sub.getOffsetHi(idx).empty())
      elidedAttrs.push_back(sub.getOffsetHiAttrName(idx));
    if (sub.getSquare(idx).empty())
      elidedAttrs.push_back(sub.getSquareAttrName(idx));
  }
  p.printOptionalAttrDict(sub->getAttrs(), elidedAttrs);

  // And now print the types
  p << " : " << sub.lhs().getType() << ", " << sub.rhs().getType();
  p << ", " << sub.result().getType();
}

// Verify Sub op.
static LogicalResult verify(aievec::SubOp sub) {
  // Verify the types
  VectorType resultType = sub.result().getType().dyn_cast<VectorType>();
  VectorType lhsType = sub.lhs().getType().dyn_cast<VectorType>();
  VectorType rhsType = sub.rhs().getType().dyn_cast<VectorType>();

  if (!lhsType || !rhsType || !resultType)
    return sub.emitError("requires vector type");

  // All the vector types must match
  if (lhsType != rhsType || 
      rhsType != resultType)
    return sub.emitError("all vectors must be of same type");

  return success();
}

// Parse Sub op.
static ParseResult parseSubOp(OpAsmParser &parser,
                              OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::OperandType lhs, rhs;

  // Parse the lhs and rhs
  if (parser.parseOperand(lhs) ||
      parser.parseComma() ||
      parser.parseOperand(rhs)) 
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
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
  p << " : " << concat.sources().getTypes().front() << ", " << concat.result().getType();
}

// Verify Concat op.
static LogicalResult verify(ConcatOp concat) {
  // Must be concatenating at least two sources
  if (concat.sources().size() < 2)
    return concat.emitError("Must concatenate at least two vectors");

  // Verify the types
  VectorType sourceType = concat.sources().getTypes().front().dyn_cast<VectorType>();
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
static ParseResult parseConcatOp(OpAsmParser &parser,
                              OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  SmallVector<OpAsmParser::OperandType, 8> sources;

  // Parse the source vectors
  if (parser.parseOperandList(sources))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
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
  if (sourceLanes/resultLanes <= 1) 
    return ext.emitError("lanes in source vector must be at least "
                         "twice that of result vector");
  // Source lanes must be a multiple of result lanes
  if (sourceLanes%resultLanes != 0) 
    return ext.emitError("lanes in result vector must be a multiple "
                         "of source vector lanes");

  // Verify validity of index
  unsigned factor = sourceLanes/resultLanes;
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
static ParseResult parseExtOp(OpAsmParser &parser,
                              OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType source;

  // Parse the source vector 
  if (parser.parseOperand(source)) 
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
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
  for (int idx=0; idx<2; ++idx) {
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
static ParseResult parseSelectOp(OpAsmParser &parser,
                                 OperationState &result) {
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
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
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
// PackOp 
//===----------------------------------------------------------------------===//

// Print out Pack op.
static void print(OpAsmPrinter &p, PackOp pack) {
  // Print the source accumulator 
  p << " " << pack.source();

  // Print the attributes
  p.printOptionalAttrDict(pack->getAttrs());

  // And now print the types
  p << " : " << pack.source().getType() << ", " << pack.result().getType(); 
}

// Verify Pack op.
static LogicalResult verify(PackOp pack) {
  // Verify the types
  VectorType sourceType = pack.source().getType().dyn_cast<VectorType>();
  VectorType resultType = pack.result().getType().dyn_cast<VectorType>();
  if (!sourceType || !resultType)
    return pack.emitError("requires vector type");

  // The number of lanes must match
  unsigned sourceLanes = getVectorLaneSize(sourceType);
  unsigned resultLanes = getVectorLaneSize(resultType);
  if (sourceLanes != resultLanes)
    return pack.emitError("The number of lanes in input and "
                         "output vector must match");

  // The datatype of source must be i16
  Type stype = sourceType.getElementType();
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  if (stypeWidth != 16)
    return pack.emitError("input must be an int16 vector");

  // The datatype of result must be i8
  Type rtype = resultType.getElementType();
  unsigned rtypeWidth = rtype.getIntOrFloatBitWidth();
  if (rtypeWidth != 8)
    return pack.emitError("output must be an int8 vector");

  return success();
}

// Parse Pack op.
static ParseResult parsePackOp(OpAsmParser &parser,
                              OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType source;

  // Parse the source vector 
  if (parser.parseOperand(source)) 
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
    return failure();

  // Currently there are no attributes in pack op
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

//===----------------------------------------------------------------------===//
// UnpackOp 
//===----------------------------------------------------------------------===//

// Print out Unpack op.
static void print(OpAsmPrinter &p, UnpackOp unpack) {
  // Print the source accumulator 
  p << " " << unpack.source();

  // Print the attributes
  p.printOptionalAttrDict(unpack->getAttrs());

  // And now print the types
  p << " : " << unpack.source().getType() << ", " << unpack.result().getType();
}

// Verify Unpack op.
static LogicalResult verify(UnpackOp unpack) {
  // Verify the types
  VectorType sourceType = unpack.source().getType().dyn_cast<VectorType>();
  VectorType resultType = unpack.result().getType().dyn_cast<VectorType>();
  if (!sourceType || !resultType)
    return unpack.emitError("requires vector type");

  // The number of lanes must match
  unsigned sourceLanes = getVectorLaneSize(sourceType);
  unsigned resultLanes = getVectorLaneSize(resultType);
  if (sourceLanes != resultLanes)
    return unpack.emitError("The number of lanes in input and "
                         "output vector must match");

  // The datatype of source must be i8
  Type stype = sourceType.getElementType();
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  if (stypeWidth != 8)
    return unpack.emitError("input must be an int8 vector");

  // The datatype of result must be i16
  Type rtype = resultType.getElementType();
  unsigned rtypeWidth = rtype.getIntOrFloatBitWidth();
  if (rtypeWidth != 16)
    return unpack.emitError("output must be an int16 vector");

  return success();
}

// Parse Unpack op.
static ParseResult parseUnpackOp(OpAsmParser &parser,
                              OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::OperandType source;

  // Parse the source vector 
  if (parser.parseOperand(source)) 
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || 
      parser.parseColonTypeList(types))
    return failure();

  // Currently there are no attributes in unpack op
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

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOps.cpp.inc"
