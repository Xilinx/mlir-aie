//===- AIEVecTypeConstraints.td - AIEVec type constraints--*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 AMD Inc.
//

//===----------------------------------------------------------------------===//
// Extra type constraint definitions for AIEVec operations.
//===----------------------------------------------------------------------===//
#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECTYPECONSTRAINTS_TD
#define AIE_DIALECT_AIEVEC_IR_AIEVECTYPECONSTRAINTS_TD

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"

def I4  : I<4>;

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
  Type<And<[VectorOf<allowedTypes>.predicate,
            CPred<ShapedTypeBitWidth<"_self">.result # " == " # bitwidth>]>,
       bitwidth # "-bit wide vector, of " # AnyTypeOf<allowedTypes>.summary>;

def AIE2MatMulLHS :
  AnyTypeOf<[VectorOfShapeAndType<[4, 16], I8>,
             VectorOfShapeAndType<[4, 8], I8>,
             VectorOfShapeAndType<[4, 4], I16>,
             VectorOfShapeAndType<[4, 2], I16>,
             VectorOfShapeAndType<[2, 8], I16>,
             VectorOfShapeAndType<[4, 8], I16>,
             VectorOfShapeAndType<[2, 4], I16>,
             VectorOfShapeAndType<[4, 2], I32>,
             VectorOfShapeAndType<[4, 8], BF16>],
            "a vector compatible with a lhs operand of matrix-multiply and "
            # "accumulate",
            "::mlir::VectorType">;

def AIE2MatMulRHS :
  AnyTypeOf<[VectorOfShapeAndType<[16, 8], I4>,
             VectorOfShapeAndType<[8, 8], I8>,
             VectorOfShapeAndType<[4, 8], I8>,
             VectorOfShapeAndType<[2, 8], I16>,
             VectorOfShapeAndType<[8, 4], I8>,
             VectorOfShapeAndType<[4, 8], I16>,
             VectorOfShapeAndType<[4, 4], I16>,
             VectorOfShapeAndType<[2, 4], I16>,
             VectorOfShapeAndType<[8, 4], BF16>],
            "a vector compatible with a rhs operand of matrix-multiply and "
            # "accumulate",
            "::mlir::VectorType">;

def AIE2MatMulACC :
  AnyTypeOf<[VectorOfShapeAndType<[4, 8], I32>,
             VectorOfShapeAndType<[4, 4], I32>,
             VectorOfShapeAndType<[2, 8], I64>,
             VectorOfShapeAndType<[4, 4], I64>,
             VectorOfShapeAndType<[4, 4], F32>],
            "a vector compatible with an accumulator of matrix-multiply and "
            # "accumulate",
            "::mlir::VectorType">;

class ShapeDimsMatch<string lhs, int ld, string rhs, int rd> :
  CPred<Shape<lhs>.result # "[" # ld # "] == " #
        Shape<rhs>.result # "[" # rd # "]">;

class ShapesCompatibleWithContraction<string lhs, string rhs, string acc> :
  PredOpTrait<"[" # lhs # " x " # rhs # " = " # acc #
              "] is a valid contraction",
              And<[ShapeDimsMatch<lhs, 1, rhs, 0>,
                   ShapeDimsMatch<lhs, 0, acc, 0>,
                   ShapeDimsMatch<rhs, 1, acc, 1>]>>;

class VectorType<string name> : StrFunc<"cast<VectorType>($" # name #
                                        ".getType())">;

class VectorTypesMatch<string op1, Type t1,
                       string op2, Type t2,
                       string op3, Type t3> :
  And<[SubstLeaves<"$_self", VectorType<op1>.result, t1.predicate>,
       SubstLeaves<"$_self", VectorType<op2>.result, t2.predicate>,
       SubstLeaves<"$_self", VectorType<op3>.result, t3.predicate>]>;

class IsValidAIE2MatMulShapeAndType<string lhs, string rhs, string acc> :
  PredOpTrait<lhs # " x " # rhs # " = " # acc # " is a valid AIE2 " #
              "matrix-multiply and accumulate op",
              Or<[VectorTypesMatch<lhs, VectorOfShapeAndType<[4, 16], I8>,
                                   rhs, VectorOfShapeAndType<[16, 8], I4>,
                                   acc, VectorOfShapeAndType<[4, 8], I32>>,
                  VectorTypesMatch<lhs, VectorOfShapeAndType<[4, 8], I8>,
                                   rhs, VectorOfShapeAndType<[8, 8], I8>,
                                   acc, VectorOfShapeAndType<[4, 8], I32>>,
                  VectorTypesMatch<lhs, VectorOfShapeAndType<[4, 4], I16>,
                                   rhs, VectorOfShapeAndType<[4, 8], I8>,
                                   acc, VectorOfShapeAndType<[4, 8], I32>>,
                  VectorTypesMatch<lhs, VectorOfShapeAndType<[4, 2], I16>,
                                   rhs, VectorOfShapeAndType<[2, 8], I16>,
                                   acc, VectorOfShapeAndType<[4, 8], I32>>,

                  VectorTypesMatch<lhs, VectorOfShapeAndType<[2, 8], I16>,
                                   rhs, VectorOfShapeAndType<[8, 8], I8>,
                                   acc, VectorOfShapeAndType<[2, 8], I64>>,
                  VectorTypesMatch<lhs, VectorOfShapeAndType<[4, 8], I16>,
                                   rhs, VectorOfShapeAndType<[8, 4], I8>,
                                   acc, VectorOfShapeAndType<[4, 4], I64>>,
                  VectorTypesMatch<lhs, VectorOfShapeAndType<[2, 4], I16>,
                                   rhs, VectorOfShapeAndType<[4, 8], I16>,
                                   acc, VectorOfShapeAndType<[2, 8], I64>>,
                  VectorTypesMatch<lhs, VectorOfShapeAndType<[4, 4], I16>,
                                   rhs, VectorOfShapeAndType<[4, 4], I16>,
                                   acc, VectorOfShapeAndType<[4, 4], I64>>,
                  VectorTypesMatch<lhs, VectorOfShapeAndType<[4, 2], I32>,
                                   rhs, VectorOfShapeAndType<[2, 4], I16>,
                                   acc, VectorOfShapeAndType<[4, 4], I64>>,

                  VectorTypesMatch<lhs, VectorOfShapeAndType<[4, 8], BF16>,
                                   rhs, VectorOfShapeAndType<[8, 4], BF16>,
                                   acc, VectorOfShapeAndType<[4, 4], F32>>]>>;

class hasAnyTypeOf<string name, list<Type> vaildTypes> :
  SubstLeaves<"$_self", ElementType<name>.result, AnyTypeOf<vaildTypes>.predicate>;

class IsValidAccTypeForAIE2MulElem<string lhs, string rhs, string acc> :
  PredOpTrait<lhs # " x " # rhs # " = " # acc # " is a valid AIE2 " #
    "element-wise multiply and accumulate op",
    Or<[  
      And <[hasAnyTypeOf<lhs, [I8 ]>, hasAnyTypeOf<acc, [I8, I32]>]>,
      And <[hasAnyTypeOf<lhs, [I16]>, hasAnyTypeOf<acc, [I16, I32, I64]>]>, 
      And <[hasAnyTypeOf<lhs, [I32]>, hasAnyTypeOf<acc, [I32, I64]>]>, 
      And <[hasAnyTypeOf<lhs, [BF16]>, hasAnyTypeOf<acc, [BF16, F32]>]>, 
      And <[hasAnyTypeOf<lhs, [F32]>,  hasAnyTypeOf<acc, [F32]>]>
    ]>>;
#endif
