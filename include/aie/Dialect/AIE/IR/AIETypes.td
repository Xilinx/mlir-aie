//===- AIETypes.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TYPES
#define AIE_TYPES

include "aie/Dialect/AIE/IR/AIE.td"
include "aie/Dialect/AIE/IR/AIEAttrs.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"

def AIE_ObjectFifoType : TypeDef<AIE_Dialect, "AIEObjectFifo", []> {
  let mnemonic = "objectfifo";
  let description = [{
    An object FIFO represents the data movement connection between a source and one or multiple destinations.
  }];

  let parameters = (ins AnyMemRef:$memref);

  let assemblyFormat = "`<` $memref `>`";

  // This is for compatibility with code prior to tablegenerated types in the project
  let extraClassDeclaration = [{
    static AIEObjectFifoType get(mlir::MemRefType elementType)
    {
      mlir::MLIRContext *ctx = elementType.getContext();
      return get(ctx, elementType);
    }

    mlir::MemRefType getElementType() {
      return getMemref();
    }
  }];
}

def AIE_ObjectFifoSubviewType : TypeDef<AIE_Dialect, "AIEObjectFifoSubview", []> {
  let mnemonic = "objectfifosubview";
  let description = [{
    An Object FIFO subview represents a subview of an object FIFO.
  }];

  let parameters = (ins AnyMemRef:$memref);

  let assemblyFormat = "`<` $memref `>`";

  // This is for compatibility with code prior to tablegenerated types in the project
  let extraClassDeclaration = [{
    static AIEObjectFifoSubviewType get(mlir::MemRefType elementType)
    {
      mlir::MLIRContext *ctx = elementType.getContext();
      return get(ctx, elementType);
    }

    mlir::MemRefType getElementType() {
      return getMemref();
    }
  }];
}

def AIE_ObjectFifo_Depth : AnyAttrOf<[ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>, ArrayAttr]>;

def AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                          "scalar-or-tensor">;

def AnyScalar : TypeConstraint<Or<[Index.predicate,
                                   AnySignlessInteger.predicate,
                                   AnyFloat.predicate]>,
                                  "scalar">;

#endif // AIE_TYPES
