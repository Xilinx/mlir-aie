//===- PhysicalBase.td -------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_BASE_TD
#define MLIR_PHY_DIALECT_PHYSICAL_BASE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def Physical_Dialect : Dialect {
  let name = "physical";
  let cppNamespace = "::xilinx::phy::physical";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
  let description = [{
    This dialect describes the actual physical implementation of the spatial
    architecture. It abstracted out the most common low-level features provided
    by the devices and platforms.  The layout information is embedded into the
    node as device-specific attributes.

    The physical operations used in a valid code must be non-overlapping with
    others and has a specific physical location.  For sharing of a resource,
    a shared version is implemented depending on device support, and only one
    shared operation is specified in the code.

    The dialect acts as the assembly for physical designs.  The mapping from
    this dialect to either code generator (mlir-translate), or the lower level
    dialects must be one-to-one.  Depending on the device and platform, some
    operations defined might not be supported.
  }];
  let useDefaultTypePrinterParser = 1;
}

class Physical_Op<string mnemonic, list<Trait> traits = []> :
    Op<Physical_Dialect, mnemonic, traits>;

class Physical_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Physical_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_BASE_TD
