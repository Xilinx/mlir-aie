//===- FlowAccessOps.td ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_FLOW_ACCESS_OPS_TD
#define MLIR_PHY_DIALECT_SPATIAL_FLOW_ACCESS_OPS_TD

include "phy/Dialect/Spatial/SpatialBase.td"

//===----------------------------------------------------------------------===//
// StartLoadOp
//===----------------------------------------------------------------------===//

def StartLoadOp : Spatial_Op<"start_load",
    [TypesMatchWith<"result's type is the promise of element type of 'memref'",
                    "memref", "result",
                    "PromiseType::get($_self.getContext(), "
                      "$_self.cast<MemRefType>().getElementType())">,
     MemRefsNormalizable]> {

  let summary = "non-blocking load of the data";
  let description = [{
    A non-blocking bus access that reads the data from the memory in a queue
    as specified in 'memref'.  This operation returns a promise, which may be
    waited using `spatial.wait` to get the access result.

    Example:

    ```mlir
    %load_promise = spatial.start_load %mem[%idx] : memref<1024xi32>
    %0 = spatial.wait(%load_promise) : i32
    ```
  }];

  let arguments = (ins AnyStaticShapeMemRef:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// StartStoreOp
//===----------------------------------------------------------------------===//

def StartStoreOp : Spatial_Op<"start_store",
    [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                    "memref", "value",
                    "$_self.cast<MemRefType>().getElementType()">,
     TypesMatchWith<"result's type is a promise of NoneType",
                    "memref", "promise",
                    "PromiseType::get($_self.getContext(), "
                      "NoneType::get($_self.getContext()))">,
     MemRefsNormalizable]> {

  let summary = "non-blocking store of the data";
  let description = [{
    A non-blocking bus access that stores the data to a bufferin a queue
    as specified in 'memref'.  This operation returns a promise, which may
    be waited using `spatial.wait`.

    Example:

    ```mlir
    %store_promise = spatial.start_store %0, %mem[%idx] : memref<1024xi32>
    spatial.wait(%store_promise) : none
    ```
  }];

  let arguments = (ins AnyType:$value,
                       AnyStaticShapeMemRef:$memref,
                       Variadic<Index>:$indices);
  let results = (outs PromiseTypeOf<NoneType>:$promise);

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

def WaitOp : Spatial_Op<"wait",
    [TypesMatchWith<"promise's data type matches result's type",
                    "result", "promise",
                    "PromiseType::get($_self.getContext(), $_self)">]> {

  let summary = "blocking wait until a promise is ready";
  let description = [{
    A non-blocking bus access returns a promise, which may be waited using this
    operation to get the access result.  For store access, none is returned.

    Example:

    ```mlir
    %0 = spatial.wait(%load_promise) : i32
    spatial.wait(%store_promise) : none
    ```
  }];

  let arguments = (ins AnyPromiseType:$promise);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $promise `)` attr-dict `:` type($result) 
  }];
}

#endif  // MLIR_PHY_DIALECT_SPATIAL_FLOW_ACCESS_OPS_TD
