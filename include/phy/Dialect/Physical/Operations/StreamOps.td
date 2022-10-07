//===- StreamOps.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_STREAM_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_STREAM_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// StreamOp
//===----------------------------------------------------------------------===//

def StreamOp : Physical_Op<"stream"> {
  let summary = "streaming connection creation operation";
  let description = [{
    The `physical.stream` operation represents a creation of a stream that
    connects two endpoints and provides a streaming connection.  The created
    stream can be connected as an operand in `physical.core` operations for the
    software function to access and communicate, or by a `physical.stream_dma`
    operation.  Streams can be connected using `physical.stream_hub`s.  A
    stream can optionally observe tagged data, and the observed tags shall be
    specified as the `tags` attribute.  Data's tag will be preseved when
    passing from the ostream endpoint to the istream endpoint.

    Example:

    ```mlir
    %stream:2 = physical.stream<[0,1]>()
              : (!physical.ostream<i32>, !physical.istream<i32>)
    // %stream#0 is the ostream, and %stream#1 is the istream
    ```
  }];

  let arguments = (ins OptionalAttr<I64ArrayAttr>:$tags);
  let results = (outs AnyOStreamType:$ostream,
                      AnyIStreamType:$istream);

  let assemblyFormat = [{
    (`<` $tags^ `>`)? `(` `)` attr-dict
    `:` `(` type($ostream) `,` type($istream) `)`
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_STREAM_OPS_TD
