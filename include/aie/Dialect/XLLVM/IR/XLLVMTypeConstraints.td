//===- XLLVMTypeConstraints.td - XLLVM type constraints. --*- tablegen -*-====//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines type constraints for LLVM types used in XLLVM intrinsic op
// definitions.
//===----------------------------------------------------------------------===//


#ifndef AIE_DIALECT_XLLVM_IR_XLLVMTYPECONSTRAINTS_TD
#define AIE_DIALECT_XLLVM_IR_XLLVMTYPECONSTRAINTS_TD

class EnumeratedType<Type ty, int idx> {
    Type type = ty;
    int index = idx;
}

class EnumerateTypeListFrom<list<Type> tlist, int from = 0> {
    list<EnumeratedType> sequence =
        !if(!empty(tlist), [],
            !listconcat(
                    [EnumeratedType<!head(tlist), from>],
                    EnumerateTypeListFrom<!tail(tlist), !add(from, 1)>.sequence
                    ));
}

class LLVM_StructOf<list<Type> structTypes> :
    Type<
        And<[LLVM_AnyStruct.predicate,
            CPred<"cast<::mlir::LLVM::LLVMStructType>($_self).getBody().size() == " # !size(structTypes)>,
            And<!foreach(enumTy, EnumerateTypeListFrom<structTypes>.sequence,
                SubstLeaves<"$_self",
                    "cast<::mlir::LLVM::LLVMStructType>($_self).getBody()[" # enumTy.index # "]",
                    enumTy.type.predicate>
                )
            >
        ]>,
        "an LLVM struct of {" # !interleave(!foreach(ty, structTypes, ty.summary), "; ") # "}"
    >;

#endif // AIE_DIALECT_XLLVM_IR_XLLVMTYPECONSTRAINTS_TD
