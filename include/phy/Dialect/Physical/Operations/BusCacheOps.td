//===- BusCacheOps.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_BUS_CACHE_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_BUS_CACHE_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// BusCacheOp
//===----------------------------------------------------------------------===//

def BusCacheOp : Physical_Op<"bus_cache",
    [TypesMatchWith<"type of 'upstream' matches element type of the cache",
                    "bus_cache", "upstream",
                    "BusType::get($_self.getContext(), "
                      "$_self.cast<BusCacheType>().getDatatype())">,
     TypesMatchWith<"type of 'downstream' matches element type of the cache",
                    "bus_cache", "downstream",
                    "BusType::get($_self.getContext(), "
                      "$_self.cast<BusCacheType>().getDatatype())">]> {

  let summary = "creation of a cache layer between two addressed buses";
  let description = [{
    An operation creating a cache to connect two buses.  With the bus cache,
    the memory access on the 'downstream' bus from the buffers on the 'upstream'
    bus will be cached, and the 'downstream' bus is able to receive data from
    the buffers on the 'upstream' bus.  The 'physical.bus_cache' type specifies
    the number of elements that can be cached.

    Example:

    ```mlir
    %bus1 = physical.bus() : !physical.bus<i32>
    %bus2 = physical.bus() : !physical.bus<i32>
    %cache = physical.bus_cache(%bus1, %bus2) : !physical.bus_cache<i32, 1024>
    ```
  }];

  let arguments = (ins BusTypeOf<AnyType>:$upstream,
                       BusTypeOf<AnyType>:$downstream);
  let results = (outs BusCacheTypeOf<AnyType>:$bus_cache);

  let assemblyFormat = [{
    `(` $upstream `,` $downstream `)` attr-dict `:` type($bus_cache)
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_BUS_CACHE_OPS_TD
