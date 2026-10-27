//===---- AIEVecOps.cpp - MLIR AIE Vector Dialect Operations ----*- C++ -*-===//
//
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements AIE vector op printing, pasing, and verification.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/AIEVecUtils.h"
#include "aie/Dialect/AIEVec/IR/AIEVecMatMul.h"
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#include "aie/Dialect/AIEVec/IR/AIEVecShuffleModes.h"

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
  auto *srcDefOp = getSource().getDefiningOp();
  if (!srcDefOp)
    return nullptr;

  auto upsOp = dyn_cast<UPSOp>(srcDefOp);
  if (!upsOp)
    return nullptr;

  auto *shiftDefOp = getShift().getDefiningOp();
  if (!shiftDefOp)
    return nullptr;

  auto constOp = dyn_cast<arith::ConstantOp>(shiftDefOp);
  if (!constOp)
    return nullptr;

  // Only fold srs(ups(x), 0) → x. Non-zero shift performs actual
  // shift-round-saturate and must not be folded away.
  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr || intAttr.getInt() != 0)
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

  // Always print the attribute dict, but elide the sign attribute when it
  // has the default value (1 = signed) to avoid breaking existing tests.
  if (getSign() == 1) {
    p.printOptionalAttrDict((*this)->getAttrs(), {"sign"});
  } else {
    p.printOptionalAttrDict((*this)->getAttrs());
  }

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
  if (isa<FloatType>(atype) && stypeWidth != 16 && stypeWidth != atypeWidth)
    return emitError("the element type of         source accumulator must be "
                     "same as the result         vector");

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

  // Parse optional attributes (e.g., {sign = 0 : i32})
  if (parser.parseOptionalAttrDict(result.attributes))
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
  auto *srcDefOp = getSource().getDefiningOp();
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
static LogicalResult verifyMulFMAElemOp(T op) {
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
  if (getVectorLaneSize(cast<VectorType>(getLhs().getType())) == 64 &&
      (!getElementTypeOrSelf(getLhs().getType()).isBF16() ||
       !getElementTypeOrSelf(getResult().getType()).isF32()))
    return emitError("64 lanes are supported only for bf16 operands with an "
                     "f32 result");
  return verifyMulFMAElemOp<aievec::MulElemOp>(*this);
}

LogicalResult aievec::FMAElemOp::verify() {
  return verifyMulFMAElemOp<aievec::FMAElemOp>(*this);
}

// Parse MulElem and FMAElem op.
static ParseResult parseMulFMAElemOp(OpAsmParser &parser,
                                     OperationState &result,
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
    VectorType type = llvm::cast<VectorType>(source.getType());
    assert(type.getRank() == 1 &&
           "only rank 1 vectors currently supported by concat");
    totalLength += type.getDimSize(0);
  }
  inferredReturnTypes.push_back(VectorType::get(
      {totalLength},
      llvm::cast<VectorType>(srcs[0].getType()).getElementType()));
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
static LogicalResult verifyPackUnpackOp(T op) {
  // Verify the types
  auto sourceType = llvm::dyn_cast<VectorType>(op.getSource().getType());
  auto resultType = llvm::dyn_cast<VectorType>(op.getResult().getType());
  if (!sourceType || !resultType)
    return op.emitError("requires vector type");

  unsigned sourceLanes = getVectorLaneSize(sourceType);
  unsigned resultLanes = getVectorLaneSize(resultType);

  Type stype = sourceType.getElementType();
  Type rtype = resultType.getElementType();
  // Pack/unpack are integer packing ops; reject non-integer element types
  // (e.g. f16/bf16) even though they share bitwidths with the legal forms.
  if (!isa<IntegerType>(stype) || !isa<IntegerType>(rtype))
    return op.emitError("requires integer element types");
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  unsigned rtypeWidth = rtype.getIntOrFloatBitWidth();

  if (isa<PackOp>(op)) {
    // Pack: i16 -> i8 with matching lane count (existing AIE2 path) OR
    // i8 -> i8 with input lanes = 2 * output lanes (AIE2P int4 path; the
    // result carries 2 packed nibbles per byte).
    bool legalI16I8 =
        stypeWidth == 16 && rtypeWidth == 8 && sourceLanes == resultLanes;
    bool legalI8I4Packed =
        stypeWidth == 8 && rtypeWidth == 8 && sourceLanes == 2 * resultLanes;
    if (!legalI16I8 && !legalI8I4Packed)
      return op.emitError(
          "pack must narrow i16->i8 (same lanes) or i8->i8 (input lanes = "
          "2 * output lanes, int4-packed)");
  } else {
    // Unpack: i8 -> i16 with matching lane count (existing AIE2 path) OR
    // i8 -> i8 with output lanes = 2 * input lanes (AIE2P int4 path; the
    // input carries 2 packed nibbles per byte and the output expands each
    // nibble into a full byte).
    bool legalI8I16 =
        stypeWidth == 8 && rtypeWidth == 16 && sourceLanes == resultLanes;
    bool legalI4PackedI8 =
        stypeWidth == 8 && rtypeWidth == 8 && resultLanes == 2 * sourceLanes;
    if (!legalI8I16 && !legalI4PackedI8)
      return op.emitError(
          "unpack must widen i8->i16 (same lanes) or i8->i8 (output lanes = "
          "2 * input lanes, int4-packed)");
  }

  return success();
}

LogicalResult PackOp::verify() { return verifyPackUnpackOp<PackOp>(*this); }

LogicalResult UnpackOp::verify() { return verifyPackUnpackOp<UnpackOp>(*this); }

// Parse Pack and Unpack op.
static ParseResult parsePackUnpackOp(OpAsmParser &parser,
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
// MatMulOp and MatMulOp_AIE2P
//===----------------------------------------------------------------------===//

namespace {
// One row of the recipe table. Element types are given by kind and width,
// with 0 standing for bf16 operands or an f32 accumulator.
struct MatMulRecipeEntry {
  bool aie2p;
  unsigned lhsWidth, rhsWidth, accWidth;
  int64_t m, k, n;
  MatMulRecipe recipe;
};
} // namespace

// The lhs and rhs layouts in each row are those the multiply reads: the MAC
// modes read lhs [m, k] and rhs [k, n]; bfp16 `8x8_8x8T` reads rhs [n, k],
// since both of its operands share an exponent along k; the elementwise bf16
// recipes broadcast the lhs a column at a time and so read it [k, m], except
// that the 4-row one pads lhs [m, k] to 8 rows before transposing it itself.
// A 1x8 rhs or 8x1 lhs reads the same either way.
static ArrayRef<MatMulRecipeEntry> getMatMulRecipeTable() {
  using K = MatMulRecipeKind;
  // clang-format off
  static const MatMulRecipeEntry table[] = {
    // aie2p, lhs, rhs and acc widths, m, k, n, {kind, lhsKM, rhsNK}
    {false,  8,  4, 32, 4, 16, 8, {K::AIE2_I8xI4_4x16x8, false, false}},
    {false,  8,  8, 32, 4,  8, 8, {K::AIE2_I8xI8_4x8x8, false, false}},
    {false, 16,  8, 32, 4,  4, 8, {K::AIE2_I16xI8_4x4x8, false, false}},
    {false, 16, 16, 32, 4,  2, 8, {K::AIE2_I16xI16_4x2x8, false, false}},
    {false, 16,  8, 64, 2,  8, 8, {K::AIE2_I16xI8_2x8x8_Acc64, false, false}},
    {false, 16,  8, 64, 4,  8, 4, {K::AIE2_I16xI8_4x8x4_Acc64, false, false}},
    {false, 16, 16, 64, 2,  4, 8, {K::AIE2_I16xI16_2x4x8_Acc64, false, false}},
    {false, 16, 16, 64, 4,  4, 4, {K::AIE2_I16xI16_4x4x4_Acc64, false, false}},
    {false, 32, 16, 64, 4,  2, 4, {K::AIE2_I32xI16_4x2x4_Acc64, false, false}},
    {false,  0,  0,  0, 4,  8, 4, {K::AIE2_BF16_4x8x4, false, false}},
    {true,   8,  8, 32, 8,  8, 8, {K::AIE2P_I8_8x8x8, false, false}},
    {true,  16, 16, 32, 8,  2, 8, {K::AIE2P_I16_8x2x8, false, false}},
    {true,   0,  0,  0, 8,  8, 8, {K::AIE2P_BFP16_8x8x8, false, true}},
    {true,   0,  0,  0, 4,  8, 8, {K::AIE2P_BFP16_4x8x8, false, true}},
    {true,   0,  0,  0, 8,  8, 4, {K::AIE2P_BF16Elem_8x8x4, true, false}},
    {true,   0,  0,  0, 4,  8, 4, {K::AIE2P_BF16Elem_4x8x4, false, false}},
    {true,   0,  0,  0, 8,  1, 8, {K::AIE2P_BF16Outer_8x1x8, false, false}},
  };
  // clang-format on
  return table;
}

// The table width of an operand element type: its width for an integer, 0
// for bf16, and nullopt for anything else.
static std::optional<unsigned> getRecipeOperandWidth(Type t) {
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth();
  if (t.isBF16())
    return 0u;
  return std::nullopt;
}

static std::optional<unsigned> getRecipeAccWidth(Type t) {
  if (auto intTy = dyn_cast<IntegerType>(t))
    return intTy.getWidth();
  if (t.isF32())
    return 0u;
  return std::nullopt;
}

std::optional<MatMulRecipe>
xilinx::aievec::getMatMulRecipe(bool aie2p, Type lhsElemTy, Type rhsElemTy,
                                Type accElemTy, int64_t m, int64_t k,
                                int64_t n) {
  auto lhsWidth = getRecipeOperandWidth(lhsElemTy);
  auto rhsWidth = getRecipeOperandWidth(rhsElemTy);
  auto accWidth = getRecipeAccWidth(accElemTy);
  if (!lhsWidth || !rhsWidth || !accWidth)
    return std::nullopt;
  // bf16 multiplies accumulate in f32, and integers in integers.
  if ((*lhsWidth == 0 || *rhsWidth == 0 || *accWidth == 0) &&
      (*lhsWidth | *rhsWidth | *accWidth) != 0)
    return std::nullopt;
  for (const MatMulRecipeEntry &e : getMatMulRecipeTable())
    if (e.aie2p == aie2p && e.lhsWidth == *lhsWidth &&
        e.rhsWidth == *rhsWidth && e.accWidth == *accWidth && e.m == m &&
        e.k == k && e.n == n)
      return e.recipe;
  return std::nullopt;
}

std::optional<SmallVector<ShuffleMode, 2>>
xilinx::aievec::getTileTransposeModes(bool aie2p, Type elemTy, int64_t rows,
                                      int64_t cols) {
  if (rows == 1 || cols == 1)
    return SmallVector<ShuffleMode, 2>{};
  if (!elemTy.isIntOrFloat())
    return std::nullopt;
  unsigned width = elemTy.getIntOrFloatBitWidth();
  int64_t bits = rows * cols * width;
  // Full 512-bit transposes, as AIE2 and AIE2P both provide them. Modes are
  // named after the shape they read.
  static const struct {
    unsigned width;
    int64_t rows, cols;
    ShuffleMode mode;
  } modes512[] = {
      {8, 4, 16, ShuffleMode::T8_4X16},   {8, 8, 8, ShuffleMode::T8_8X8},
      {8, 16, 4, ShuffleMode::T8_16X4},   {16, 2, 16, ShuffleMode::T16_2X16},
      {16, 4, 8, ShuffleMode::T16_4X8},   {16, 8, 4, ShuffleMode::T16_8X4},
      {16, 16, 2, ShuffleMode::T16_16X2}, {32, 4, 4, ShuffleMode::T32_4X4},
  };
  // AIE2P modes that transpose each 256-bit half of a register on its own.
  static const struct {
    unsigned width;
    int64_t rows, cols;
    ShuffleMode mode;
  } modes256[] = {
      {8, 4, 8, ShuffleMode::T8_4X8},   {8, 8, 4, ShuffleMode::T8_8X4},
      {16, 2, 8, ShuffleMode::T16_2X8}, {16, 4, 4, ShuffleMode::T16_4X4},
      {16, 8, 2, ShuffleMode::T16_8X2},
  };
  if (bits == 512) {
    for (const auto &e : modes512)
      if (e.width == width && e.rows == rows && e.cols == cols)
        return SmallVector<ShuffleMode, 2>{e.mode};
    return std::nullopt;
  }
  if (!aie2p)
    return std::nullopt;
  if (bits == 256) {
    for (const auto &e : modes256)
      if (e.width == width && e.rows == rows && e.cols == cols)
        return SmallVector<ShuffleMode, 2>{e.mode};
    return std::nullopt;
  }
  if (width == 16 && rows == 8 && cols == 8)
    return SmallVector<ShuffleMode, 2>{ShuffleMode::T16_8X8_LO,
                                       ShuffleMode::T16_8X8_HI};
  return std::nullopt;
}

// checkMatMulTile, with the reason for a failure written to `os`.
static std::optional<MatMulRecipe>
checkMatMulTileImpl(bool aie2p, VectorType lhsTy, VectorType rhsTy,
                    VectorType accTy, bool lhsTransposed, bool rhsTransposed,
                    llvm::raw_ostream &os) {
  if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || accTy.getRank() != 2) {
    os << "expects rank-2 operands";
    return std::nullopt;
  }
  ArrayRef<int64_t> lhsShape = lhsTy.getShape(), rhsShape = rhsTy.getShape(),
                    accShape = accTy.getShape();
  int64_t m = lhsShape[lhsTransposed ? 1 : 0];
  int64_t k = lhsShape[lhsTransposed ? 0 : 1];
  int64_t rhsK = rhsShape[rhsTransposed ? 1 : 0];
  int64_t n = rhsShape[rhsTransposed ? 0 : 1];
  if (k != rhsK || accShape[0] != m || accShape[1] != n) {
    os << "lhs " << lhsTy << (lhsTransposed ? " [k, m]" : " [m, k]")
       << " times rhs " << rhsTy << (rhsTransposed ? " [n, k]" : " [k, n]")
       << " is not a contraction into " << accTy;
    return std::nullopt;
  }
  StringRef target = aie2p ? "AIE2P" : "AIE2";
  std::optional<MatMulRecipe> recipe =
      getMatMulRecipe(aie2p, lhsTy.getElementType(), rhsTy.getElementType(),
                      accTy.getElementType(), m, k, n);
  if (!recipe) {
    os << "no " << target << " matrix multiply takes " << m << "x" << k << " "
       << lhsTy.getElementType() << " times " << k << "x" << n << " "
       << rhsTy.getElementType() << " into " << accTy.getElementType();
    return std::nullopt;
  }
  auto layoutReachable = [&](VectorType ty, bool stored, bool native) {
    return stored == native ||
           getTileTransposeModes(aie2p, ty.getElementType(), ty.getDimSize(0),
                                 ty.getDimSize(1));
  };
  if (!layoutReachable(lhsTy, lhsTransposed, recipe->lhsKM)) {
    os << "the " << target << " " << m << "x" << k << "x" << n
       << " multiply reads lhs as " << (recipe->lhsKM ? "[k, m]" : "[m, k]")
       << ", and no shuffle transposes " << lhsTy;
    return std::nullopt;
  }
  if (!layoutReachable(rhsTy, rhsTransposed, recipe->rhsNK)) {
    os << "the " << target << " " << m << "x" << k << "x" << n
       << " multiply reads rhs as " << (recipe->rhsNK ? "[n, k]" : "[k, n]")
       << ", and no shuffle transposes " << rhsTy;
    return std::nullopt;
  }
  return recipe;
}

std::optional<MatMulRecipe> xilinx::aievec::checkMatMulTile(
    bool aie2p, VectorType lhsTy, VectorType rhsTy, VectorType accTy,
    bool lhsTransposed, bool rhsTransposed,
    llvm::function_ref<InFlightDiagnostic()> emitError) {
  std::string message;
  llvm::raw_string_ostream os(message);
  std::optional<MatMulRecipe> recipe = checkMatMulTileImpl(
      aie2p, lhsTy, rhsTy, accTy, lhsTransposed, rhsTransposed, os);
  if (!recipe && emitError)
    emitError() << message;
  return recipe;
}

template <typename OpTy>
static LogicalResult verifyMatMulOp(OpTy op, bool aie2p) {
  auto lhsTy = cast<VectorType>(op.getLhs().getType());
  auto rhsTy = cast<VectorType>(op.getRhs().getType());
  auto accTy = cast<VectorType>(op.getAcc().getType());
  if (!checkMatMulTile(aie2p, lhsTy, rhsTy, accTy, op.getLhsTransposed(),
                       op.getRhsTransposed(),
                       [&]() { return op.emitOpError(); }))
    return failure();
  return success();
}

LogicalResult MatMulOp::verify() { return verifyMatMulOp(*this, false); }

LogicalResult MatMulOp_AIE2P::verify() { return verifyMatMulOp(*this, true); }

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

ShuffleModeShape xilinx::aievec::getShuffleModeShape(ShuffleMode mode) {
  // Every mode is named t<w>_<r>x<c> with an optional _lo, _hi or _flip.
  StringRef name = stringifyShuffleMode(mode);
  ShuffleModeShape shape{};
  bool parsed =
      name.consume_front("t") && !name.consumeInteger(10, shape.elemBits) &&
      name.consume_front("_") && !name.consumeInteger(10, shape.rows) &&
      name.consume_front("x") && !name.consumeInteger(10, shape.cols);
  shape.flip = name == "_flip";
  shape.hi = name == "_hi";
  assert(parsed && (name.empty() || name == "_lo" || shape.hi || shape.flip) &&
         "unexpected shuffle mode name");
  (void)parsed;
  return shape;
}

bool xilinx::aievec::isAIE2POnlyShuffleMode(ShuffleMode mode) {
  return static_cast<uint32_t>(mode) >= 48;
}

std::array<uint8_t, 64>
xilinx::aievec::getShuffleModeBytePermutation(ShuffleMode mode) {
  ShuffleModeShape shape = getShuffleModeShape(mode);
  unsigned elemBytes = shape.elemBits / 8;
  unsigned blockBytes = shape.blockBits() / 8;
  unsigned blockElems = shape.rows * shape.cols;
  std::array<uint8_t, 64> perm;
  for (unsigned out = 0; out < 64; ++out) {
    // Position in the 1024-bit result of a two-input mode, or in the 512-bit
    // result otherwise.
    unsigned pos = shape.isTwoInput() && shape.hi ? out + 64 : out;
    unsigned block = pos / blockBytes;
    unsigned elem = pos % blockBytes / elemBytes;
    unsigned src;
    if (shape.flip) {
      src = blockElems - 1 - elem;
    } else {
      // elem = j * rows + i reads in[i * cols + j].
      unsigned i = elem % shape.rows, j = elem / shape.rows;
      src = i * shape.cols + j;
    }
    perm[out] = block * blockBytes + src * elemBytes + pos % elemBytes;
  }
  return perm;
}

// The mode fixes the element width and whether rhs is needed.
LogicalResult ShuffleOp::verify() {
  ShuffleMode mode = getMode();
  ShuffleModeShape shape = getShuffleModeShape(mode);
  bool requireRhs = shape.isTwoInput();

  if (requireRhs && !getRhs())
    return emitError() << "shuffle mode '" << stringifyEnum(mode)
                       << "' requires a second operand";

  if (!requireRhs && getRhs())
    return emitError() << "shuffle mode '" << stringifyEnum(mode)
                       << "' does not admit a second operand";

  auto elemBitWidth =
      cast<VectorType>(getLhs().getType()).getElementTypeBitWidth();
  if (shape.elemBits != elemBitWidth)
    return emitError() << "shuffle mode '" << stringifyEnum(mode)
                       << "' requires vectors of " << shape.elemBits
                       << "-bit elements";

  // When the op sits in a device, the device fixes the architecture.
  if (isAIE2POnlyShuffleMode(mode))
    if (auto target = (*this)->getParentOfType<AIE::AIETarget>())
      if (target.getTargetModel().getTargetArch() != AIE::AIEArch::AIE2p)
        return emitError() << "shuffle mode '" << stringifyEnum(mode)
                           << "' is only available on AIE2P";

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
static LogicalResult verifyMulFMAConvOp(T op) {
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
static ParseResult parseMulFMAConvOp(OpAsmParser &parser,
                                     OperationState &result,
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
