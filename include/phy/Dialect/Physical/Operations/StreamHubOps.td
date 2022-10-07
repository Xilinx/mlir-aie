//===- StreamHubOps.td -------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_ROUTER_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_ROUTER_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// StreamHubOp
//===----------------------------------------------------------------------===//

def StreamHubOp : Physical_Op<"stream_hub"> {
  let summary = "creation of a stream hub connecting multiple streams";
  let description = [{
    An operation creating a stream hub to connect multiple streams.  A stream
    reads data from all `phy.istream` endpoints, and broadcast data to all
    `phy.ostream` endpoints.  Depending on the target, one or multiple input
    or output endpoints are supported.  Depending on the target, multicasting
    using data tags may be supported and only streams observing the tag of a
    piece of data will receive the data.

    Example:

    ```mlir
    %stream_hub = physical.stream_hub(%src1, %src2, %dest1, %dest2)
            : (!physical.istream<i32>,  !physical.istream<i32>,
               !physical.ostream<i32>, !physical.ostream<i32>)
            -> !physical.stream_hub<i32>
    ```
  }];

  let arguments = (ins Variadic<AnyStreamEndpointType>:$endpoints);
  let results = (outs AnyStreamHubType:$stream_hub);

  let assemblyFormat = [{
    `(` $endpoints `)` attr-dict `:` functional-type($endpoints, $stream_hub)
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_ROUTER_OPS_TD
