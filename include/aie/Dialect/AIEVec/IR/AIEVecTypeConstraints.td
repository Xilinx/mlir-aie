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

include "mlir/IR/BuiltinTypes.td"

class VectorOfShape<list<int> shape> :
    And<[IsVectorTypePred,
         CPred<"::llvm::cast<::mlir::VectorType>($_self).getShape() == ArrayRef<int64_t>({" # !interleave(shape, ", ") # "})">
    ]>;

class VectorOfShapeAndType<list<int> shape, Type type> :
                ShapedContainerType<[type], VectorOfShape<shape>,
                "vector of shape <" # !interleave(shape, "x") # "> and",
                "::mlir::VectorType">;
