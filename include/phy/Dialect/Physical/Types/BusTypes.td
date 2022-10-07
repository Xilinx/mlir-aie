//===- BusTypes.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_BUS_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_BUS_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// BusType
//===----------------------------------------------------------------------===//

def BusType : Physical_Type<"Bus", "bus"> {
  let summary = "A type specifiying a bus with address mapping specification";
  let description = [{
    A bus with its address space used by buffers.  In the defining operation, a
    buffer can specify how its memory address space is mapped to the bus.  A
    bus can be used in the same way as a 'memref' in a PE with the
    'physical.start_load' and 'physical.start_store' operations.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class BusTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<xilinx::phy::physical::BusType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<xilinx::phy::physical::BusType>().getDatatype()",
      element.predicate>
  ]>,
  "a bus of " # element.summary
>;

def AnyBusType : Type<
  Or<[
    BusTypeOf<AnyType>.predicate,
  ]>,
  "a bus"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_BUS_TYPES_TD
