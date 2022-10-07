//===- StreamDmaOps.td -------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_STREAM_DMA_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_STREAM_DMA_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// StreamDmaOp
//===----------------------------------------------------------------------===//

def StreamDmaOp : Physical_Op<"stream_dma", [
    SingleBlockImplicitTerminator<"EndOp">, RegionKindInterface]> {

  let summary = "creation of a dma engine that connects buffers with a stream";
  let description = [{
    An operation creating a stream dma engines that is connected to a stream
    endpoint.  If the endpoint is an istream, then the stream's data is written
    to the buffers sequentially according to the order as specified in the
    `physical.stream_dma_connect` operation.  Otherwise, if the endpoint is an
    ostream, the buffers are read from instead.  Depending on the target, one
    or multiple `physical.stream_dma_connect` operations are supported in a
    single `physical.stream_dma` region.  The first connection in the region is
    established first, and the next connection as specified in the operation is
    established next, so on and so forth.

    Example:

    ```mlir
    physical.stream_dma(%endpoint: !physical.istream<i32>) {
      %0 = physical.stream_dma_connect(
        %lock1[0->1], %buffer1[0:1024]: memref<1024xi32>, %1)
      %1 = physical.stream_dma_connect(
        %lock2[0->1], %buffer2[0:1024]: memref<1024xi32>, %0)
    }
    ```
  }];

  let arguments = (ins AnyStreamEndpointType:$endpoint);
  let regions = (region AnyRegion:$connections);

  let extraClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }
  }];

  let assemblyFormat = [{
    `(` $endpoint `:` type($endpoint) `)` regions attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// StreamDmaConnectOp
//===----------------------------------------------------------------------===//

def StreamDmaConnectOp : Physical_Op<"stream_dma_connect",
                                     [HasParent<"StreamDmaOp">]> {
      
  let summary = "creation of a dma connection that connects to a buffer";
  let description = [{
    `physical.stream_dma_connect` connects a buffer to/from the stream endpoint
    and when the data transfer in this connection is completed, the next
    connection as specified in `next` will be established.  If the `next` is
    not specified, the DMA engine will be terminated when the current
    connection is completed.
    
    In each connection, the `buffer` is specified with the `start` point,
    inclusively, and the `end` point, exclusively.  The data from the
    buffer/stream will be transferred to the stream/buffer, and the lock will
    be acquired from the `acquire` state before the transfer is performed, and
    released to the `release` state when the transfer is done.

    A connection can have an optional `tag` attribute.  When the `tag` is
    specified, an output connection will have the stream data tagged with
    `tag`.  For input connections, the `tag` will be ignored and all data
    received from the stream will be part of the connection.

    Example:

    ```mlir
    %0 = physical.stream_dma_connect<1>(
      %lock[0->1], %buffer[0:1024]: memref<1024xi32>, %0)
    }
    ```
  }];

  let arguments = (ins OptionalAttr<I64Attr>:$tag,
                       LockType:$lock,
                       I64Attr:$acquire, I64Attr:$release,
                       AnyStaticShapeMemRef:$buffer,
                       I64Attr:$start, I64Attr:$end,
                       Optional<StreamDmaConnectType>:$next);
  let results = (outs StreamDmaConnectType:$connection);

  let assemblyFormat = [{
    (`<` $tag^ `>`)?
    `(` $lock `[` $acquire `->` $release `]` `,`
        $buffer `[` $start `:` $end `]` `:` type($buffer)
        ( `,` $next^ )?
    `)` regions attr-dict
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_STREAM_DMA_OPS_TD
