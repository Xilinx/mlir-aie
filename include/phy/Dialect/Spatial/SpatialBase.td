//===- SpatialBase.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_BASE_TD
#define MLIR_PHY_DIALECT_SPATIAL_BASE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def Spatial_Dialect : Dialect {
  let name = "spatial";
  let cppNamespace = "::xilinx::phy::spatial";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
  let description = [{
    This dialect defines a graph of stateful vertices of free-running compute
    nodes and message queues.  The vertices are connected using flow control 
    edges called flows.  This dialect is intended to be target-independent
    and express the logical concurrency semantics.  It does not express
    anything that is target-dependent.
  }];
  let useDefaultTypePrinterParser = 1;
}

class Spatial_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Spatial_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Spatial_VertexType<string name, string typeMnemonic, list<Trait> traits = []>
    : Spatial_Type<name, typeMnemonic, traits> {
  let mnemonic = typeMnemonic;
}

class Spatial_EdgeType<string name, string typeMnemonic, list<Trait> traits = []>
    : Spatial_Type<name, typeMnemonic, traits> {
  let mnemonic = typeMnemonic;
}

class Spatial_Op<string mnemonic, list<Trait> traits = []> :
    Op<Spatial_Dialect, mnemonic, traits>;

#endif  // MLIR_PHY_DIALECT_SPATIAL_BASE_TD
