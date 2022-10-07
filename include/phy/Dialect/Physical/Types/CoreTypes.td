//===- CoreTypes.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_CORE_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_CORE_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"
include "phy/Dialect/Physical/Types/BusTypes.td"
include "phy/Dialect/Physical/Types/LockTypes.td"
include "phy/Dialect/Physical/Types/StreamTypes.td"

//===----------------------------------------------------------------------===//
// CoreType
//===----------------------------------------------------------------------===//

def CoreType : Physical_Type<"Core", "core"> {
  let summary = "A type specifiying a processing core";
  let description = [{
    A core is a logical function that computes.  It can have a device-specific
    attribute of location, e.g. a core in CPU or a tile core in AIE.
  }];
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def AnyCoreArgsType : Type<
  Or<[
    AnyStaticShapeMemRef.predicate,
    AnyBusType.predicate,
    AnyStreamEndpointType.predicate,
    LockType.predicate,
  ]>,
  "a memref, a bus, a stream endpoint or a lock"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_CORE_TYPES_TD
