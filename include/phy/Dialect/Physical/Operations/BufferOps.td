//===- BufferOps.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_BUFFER_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_BUFFER_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// BufferOp
//===----------------------------------------------------------------------===//

def BufferOp : Physical_Op<"buffer"> {
  let summary = "buffer creation operation";
  let description = [{
    The `physical.buffer` operation represents a creation of a buffer that has
    the type argument as its datatype.  A buffer is a memory space that stores
    data.  A buffer can be randomly accessed.  It can have a device-specific
    attribute of location.

    Example:

    ```mlir
    %buffer = physical.buffer() : memref<1024xi32>
    ```
  }];

  let results = (outs AnyStaticShapeMemRef:$buffer);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($buffer)
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_BUFFER_OPS_TD
