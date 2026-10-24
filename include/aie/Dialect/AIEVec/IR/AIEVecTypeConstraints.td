//===- AIEVecTypeConstraints.td - AIEVec type constraints--*- tablegen -*-====//
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

//===----------------------------------------------------------------------===//
// Extra type constraint definitions for AIEVec operations.
//===----------------------------------------------------------------------===//
#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECTYPECONSTRAINTS_TD
#define AIE_DIALECT_AIEVEC_IR_AIEVECTYPECONSTRAINTS_TD

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def I4  : I<4>;
def I256 : I<256>;
def I512 : I<512>;

class TypeShape<string name> :
  StrFunc<"cast<::mlir::ShapedType>($" # name # ").getShape()">;

// Notice: This predicate class assumes that the type has been verified to be a
//         ShapedType
class VectorOfShape<list<int> shape> :
  CPred<TypeShape<"_self">.result # " == ArrayRef<int64_t>({" #
        !interleave(shape, ", ") # "})">;

// Notice: This predicate class assumes that the type has been verified to be a
//         ShapedType
class VectorOfElementType<Type type> :
  SubstLeaves<"$_self", ElementType<"_self">.result, type.predicate>;

// Notice: This predicate class assumes that the type has been verified to be a
//         ShapedType
class VectorOfShapeAndType<list<int> shape, Type type> :
  Type<And<[VectorOfShape<shape>,
            VectorOfElementType<type>]>,
        "vector of shape <" # !interleave(shape, "x") # "> and",
        "::mlir::VectorType">;

class ShapedTypeBitWidth<string name> :
  StrFunc<"(cast<::mlir::ShapedType>($" # name # ").getElementTypeBitWidth() * "
          # !subst(".getType()", "", ElementCount<name>.result) # ")">;

class VectorOfBitWidthAndElementTypes<int bitwidth, list<Type> allowedTypes> :
  Type<And<[VectorOfNonZeroRankOf<allowedTypes>.predicate,
            CPred<ShapedTypeBitWidth<"_self">.result # " == " # bitwidth>]>,
       bitwidth # "-bit wide vector, of " # AnyTypeOf<allowedTypes>.summary>;

// Matmul integer operands carry signedness in the element type (e.g. si8/ui8);
// a signless type keeps the historical default. AnyI<w> (isInteger(w)) matches
// signless, signed, and unsigned integers of that width. Upstream defines
// AnyI8/16/32/64; AnyI4 is not upstream, so define it here.
def AnyI4 : AnyI<4>;

// The matmul operand types below only filter element types. Which shapes,
// types and layouts form a tile is checked by the op verifiers against the
// recipe table in AIEVecMatMul.h.
class MatMulOperand<list<Type> types, string what>
    : Type<VectorOfRankAndType<[2], types>.predicate,
           "a vector compatible with "#what#" and accumulate",
           "::mlir::VectorType">;

def AIE2MatMulLHS : MatMulOperand<[AnyI8, AnyI16, AnyI32, BF16],
                                  "a lhs operand of matrix-multiply">;
def AIE2MatMulRHS : MatMulOperand<[AnyI4, AnyI8, AnyI16, BF16],
                                  "a rhs operand of matrix-multiply">;
def AIE2MatMulACC
    : MatMulOperand<[AnyI32, AnyI64, F32], "an accumulator of matrix-multiply">;
def AIE2PMatMulLHS : MatMulOperand<[AnyI8, AnyI16, BF16],
                                   "a lhs operand of AIE2P matrix-multiply">;
def AIE2PMatMulRHS : MatMulOperand<[AnyI8, AnyI16, BF16],
                                   "a rhs operand of AIE2P matrix-multiply">;
def AIE2PMatMulACC
    : MatMulOperand<[AnyI32, F32], "an accumulator of AIE2P matrix-multiply">;

class isOperandResultTypePairValidForAIE2MulElem<string lhs, string rhs, string acc> :
  PredOpTrait<acc # " type is a valid accumulator type given the type of the" #
              " operands.",
    Or<[  
      And <[ElementTypeIsPred<acc,I32>, ElementTypeIsPred<lhs,I8>]>, 
      And <[ElementTypeIsPred<acc,I32>, ElementTypeIsPred<lhs,I16>]>, 
      And <[ElementTypeIsPred<acc,I64>, ElementTypeIsPred<lhs,I32>]>, 
      And <[ElementTypeIsPred<acc,F32>, ElementTypeIsPred<lhs,BF16>]>, 
      And <[ElementTypeIsPred<acc,F32>, ElementTypeIsPred<lhs,F32>]>
    ]>>;

 
#endif
