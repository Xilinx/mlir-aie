//===- AsyncOps.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_ASYNC_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_ASYNC_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// StartLoadOp
//===----------------------------------------------------------------------===//

def StartLoadOp : Physical_Op<"start_load",
    [TypesMatchWith<"result's type is the handle of element type of 'memref'",
                    "memref", "result",
                    "AsyncHandleType::get($_self.getContext(), "
                      "($_self.isa<MemRefType>() ? ("
                        "$_self.cast<MemRefType>().getElementType()"
                      ") : ("
                        "$_self.cast<BusType>().getDatatype()"
                      ")))">,
     MemRefsNormalizable]> {

  let summary = "non-blocking load of the data";
  let description = [{
    A non-blocking bus access that reads the data from a buffer, or an addressed
    bus, as specified in 'memref'.  This operation returns a handle, which may be
    waited using `physical.wait` to get the access result.

    Example:

    ```mlir
    %load_handle = physical.start_load %mem[%idx] : memref<1024xi32>
    %0 = physical.wait(%load_handle) : i32
    ```
  }];

  let arguments = (ins AnyLoadStoreType:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// StartStoreOp
//===----------------------------------------------------------------------===//

def StartStoreOp : Physical_Op<"start_store",
    [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                    "memref", "value",
                    "$_self.isa<MemRefType>() ? ("
                        "$_self.cast<MemRefType>().getElementType()"
                    ") : ("
                        "$_self.cast<BusType>().getDatatype()"
                    ")">,
     TypesMatchWith<"result's type is a handle of NoneType",
                    "memref", "handle",
                    "AsyncHandleType::get($_self.getContext(), "
                      "NoneType::get($_self.getContext()))">,
     MemRefsNormalizable]> {

  let summary = "non-blocking store of the data";
  let description = [{
    A non-blocking bus access that stores the data to a buffer, or an addressed
    bus, as specified in 'memref'.  This operation returns a handle, which may
    be waited using `physical.wait`.

    Example:

    ```mlir
    %store_handle = physical.start_store %0, %mem[%idx] : memref<1024xi32>
    physical.wait(%store_handle) : none
    ```
  }];

  let arguments = (ins AnyType:$value,
                       AnyLoadStoreType:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AsyncHandleTypeOf<NoneType>:$handle);

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

def WaitOp : Physical_Op<"wait",
    [TypesMatchWith<"handle's data type matches result's type",
                    "result", "handle",
                    "AsyncHandleType::get($_self.getContext(), $_self)">]> {

  let summary = "blocking wait until a handle is ready";
  let description = [{
    A non-blocking bus access returns a handle, which may be waited using this
    operation to get the access result.  For store access, none is returned.

    Example:

    ```mlir
    %0 = physical.wait(%load_handle) : i32
    physical.wait(%store_handle) : none
    ```
  }];

  let arguments = (ins AnyAsyncHandleType:$handle);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $handle `)` attr-dict `:` type($result) 
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_ASYNC_OPS_TD
