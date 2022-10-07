//===- BusOps.td -------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_BUS_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_BUS_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// BusOp
//===----------------------------------------------------------------------===//

def BusOp : Physical_Op<"bus"> {
  let summary = "addressed bus creation operation";
  let description = [{
    The `physical.bus` operation represents a creation of an addressed bus
    that can have buffers mapped to its memory space using the
    'physical.bus_mmap' operation.

    Example:

    ```mlir
    %buf = physical.buffer() : memref<1024xi32>
    %bus = physical.bus() : !physical.bus<i32>
    physical.bus_mmap(%bus[10:15], %buf[20:] : memref<1024xi32>)
    %pe = physical.core @func(%bus) : (!physical.bus<i32>) -> !physical.core
    ```
  }];

  let arguments = (ins);
  let results = (outs BusTypeOf<AnyType>:$bus);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($bus)
  }];
}

//===----------------------------------------------------------------------===//
// BusMmapOp
//===----------------------------------------------------------------------===//

def BusMmapOp : Physical_Op<"bus_mmap",
    [TypesMatchWith<"type of 'bus' matches element type of the 'buffer'",
                    "buffer", "bus",
                    "BusType::get($_self.getContext(), "
                      "$_self.cast<MemRefType>().getElementType())">]> {

  let summary = "creation of a buffer into an addressed bus memory space";
  let description = [{
    An operation that maps the 'buffer' starting the 'offset'-th element,
    into the 'bus'.  The mapped address is from 'begin'-th element
    (inclusive), to the 'end'-th element (exclusive) on the bus.

    Example:

    ```mlir
    physical.bus_mmap(%bus[10:15], %buf[20:] : memref<1024xi32>)
    // bus[10] will be buf[20], bus[11] will be buf[21], ...
    ```
  }];

  let arguments = (ins BusTypeOf<AnyType>:$bus,
                       I64Attr:$begin,
                       I64Attr:$end,
                       AnyStaticShapeMemRef:$buffer,
                       I64Attr:$offset);

  let assemblyFormat = [{
    `(` $bus `[` $begin `:` $end `]` `,` $buffer `[` $offset `:` `]`
    `:` type($buffer) `)` attr-dict 
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_BUS_OPS_TD
