//===---- AIEVecOps.cpp - MLIR AIE Vector Dialect Operations ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// This file implements AIE vector op printing, pasing, and verification.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"

using namespace llvm;
using namespace mlir;
using namespace xilinx;
using namespace xilinx::aievec;

#include "aie/Dialect/AIEVec/IR/AIEVecEnums.cpp.inc"
#include "aie/Dialect/AIEVec/IR/AIEVecOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// AIEVecDialect
//===----------------------------------------------------------------------===//

void AIEVecDialect::initialize() {
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aie/Dialect/AIEVec/IR/AIEVecAttributes.cpp.inc"
      >();
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

  if (isa<IntegerType>(atype) && stypeWidth >= atypeWidth)
    return emitError("the element type of source accumulator must be "
                     "wider than that of the result vector");
  else if (isa<FloatType>(atype) && stypeWidth != 16 &&
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
  auto srsOp = llvm::dyn_cast<SRSOp>(srcDefOp);
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

  if (!isa<IntegerType, FloatType>(sourceType))
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
  auto lhsType = llvm::dyn_cast<VectorType>(op.getLhs().getType());
  auto rhsType = llvm::dyn_cast<VectorType>(op.getRhs().getType());

  if (!lhsType || !rhsType)
    return op.emitError("requires vector type");

  auto resultType = llvm::dyn_cast<VectorType>(op.getResult().getType());

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
      llvm::dyn_cast<VectorType>(srcs[0].getType()).getElementType()));
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
  auto sourceType = llvm::dyn_cast<VectorType>(op.getSource().getType());
  auto resultType = llvm::dyn_cast<VectorType>(op.getResult().getType());
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
// ExtElemOp
//===----------------------------------------------------------------------===//

// Verify Extract Element op.
LogicalResult ExtElemOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());

  if (!sourceType)
    return emitError("source requires vector type");

  // The element type of vectors must always be the same
  Type stype = sourceType.getElementType();
  Type rtype = getResult().getType();

  if (stype != rtype) {
    return emitError("the type of result must be the same as the element "
                     "type of source vector");
  }

  return success();
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

// This verification function makes sure that the shuffle mode supports the
// number and type of operands provided.
LogicalResult ShuffleOp::verify() {
  unsigned modeBitWidth;
  bool requireRhs = true;
  auto mode = getMode();
  switch (mode) {
  case ShuffleMode::T8_8X8:  // 35
  case ShuffleMode::T8_16X4: // 36
  case ShuffleMode::T8_4X16: // 37
  case ShuffleMode::T8_8X4:  // 46
  case ShuffleMode::T8_4X8:  // 47
    requireRhs = false;
    LLVM_FALLTHROUGH;
  case ShuffleMode::T8_64X2_LO: //  0
  case ShuffleMode::T8_64X2_HI: //  1
  case ShuffleMode::T8_2X64_LO: // 20
  case ShuffleMode::T8_2X64_HI: // 21
    modeBitWidth = 8u;
    break;
  case ShuffleMode::T16_8X4:      // 28
  case ShuffleMode::T16_4X8:      // 29
  case ShuffleMode::T16_1X2_flip: // 38
  case ShuffleMode::T16_4X4:      // 39
  case ShuffleMode::T16_4X2:      // 40
  case ShuffleMode::T16_2X4:      // 41
  case ShuffleMode::T16_8X2:      // 42
  case ShuffleMode::T16_2X8:      // 43
  case ShuffleMode::T16_16X2:     // 44
  case ShuffleMode::T16_2X16:     // 45
    requireRhs = false;
    LLVM_FALLTHROUGH;
  case ShuffleMode::T16_32X2_LO: //  2
  case ShuffleMode::T16_32X2_HI: //  3
  case ShuffleMode::T16_2X32_LO: // 18
  case ShuffleMode::T16_2X32_HI: // 19
  case ShuffleMode::T16_16X4_LO: // 24
  case ShuffleMode::T16_16X4_HI: // 25
  case ShuffleMode::T16_4X16_LO: // 26
  case ShuffleMode::T16_4X16_HI: // 27
    modeBitWidth = 16u;
    break;
  case ShuffleMode::T32_4X4: // 34
    requireRhs = false;
    LLVM_FALLTHROUGH;
  case ShuffleMode::T32_16X2_LO: //  4
  case ShuffleMode::T32_16X2_HI: //  5
  case ShuffleMode::T32_2X16_LO: // 16
  case ShuffleMode::T32_2X16_HI: // 17
  case ShuffleMode::T32_8X4_LO:  // 30
  case ShuffleMode::T32_8X4_HI:  // 31
  case ShuffleMode::T32_4X8_LO:  // 32
  case ShuffleMode::T32_4X8_HI:  // 33
    modeBitWidth = 32u;
    break;
  case ShuffleMode::T64_8X2_LO: //  6
  case ShuffleMode::T64_8X2_HI: //  7
  case ShuffleMode::T64_2X8_LO: // 14
  case ShuffleMode::T64_2X8_HI: // 15
    modeBitWidth = 64u;
    break;
  case ShuffleMode::T128_4X2_LO: //  8
  case ShuffleMode::T128_4X2_HI: //  9
  case ShuffleMode::T128_2X4_LO: // 12
  case ShuffleMode::T128_2X4_HI: // 13
    modeBitWidth = 128u;
    break;
  case ShuffleMode::T256_2X2_LO: // 10
  case ShuffleMode::T256_2X2_HI: // 11
    modeBitWidth = 256u;
    break;
  case ShuffleMode::T512_1X2_LO: // 22
  case ShuffleMode::T512_1X2_HI: // 23
    modeBitWidth = 512u;
    break;
  }

  // Verify number of operands
  if (requireRhs && !getRhs())
    return emitError() << "shuffle mode '" << stringifyEnum(mode)
                       << "' requires a second operand";

  if (!requireRhs && getRhs())
    return emitError() << "shuffle mode '" << stringifyEnum(mode)
                       << "' does not admit a second operand";

  // Verify vector element type
  auto elemBitWidth =
      cast<VectorType>(getLhs().getType()).getElementTypeBitWidth();
  if (modeBitWidth != elemBitWidth)
    return emitError() << "shuffle mode '" << stringifyEnum(mode)
                       << "' requires vectors of " << modeBitWidth
                       << "-bit elements";

  return success();
}

// Print out Shuffle op.
void LegacyShuffleOp::print(OpAsmPrinter &p) {
  // Print the source vector
  p << " " << getSource();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify Shuffle op.
LogicalResult LegacyShuffleOp::verify() {
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
ParseResult LegacyShuffleOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
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
  auto lhsType = llvm::dyn_cast<VectorType>(op.getLhs().getType());
  auto rhsType = llvm::dyn_cast<VectorType>(op.getRhs().getType());

  if (!lhsType || !rhsType)
    return op.emitError("requires vector type");

  unsigned M = op.getM();
  unsigned N = op.getN();

  if (M <= 0 || N <= 0 || 2 * M < M + N - 1)
    return op.emitError(
        "M and N should be larger than 0 and 2*M should be no less than M+N-1");

  auto resultType = llvm::dyn_cast<VectorType>(op.getResult().getType());

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

  if (!isa<IntegerType>(ltype) || !isa<IntegerType>(rtype) ||
      !isa<IntegerType>(atype)) {
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

#define GET_ATTRDEF_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "aie/Dialect/AIEVec/IR/AIEVecOps.cpp.inc"
