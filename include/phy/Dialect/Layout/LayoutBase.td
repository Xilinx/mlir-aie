//===- LayoutBase.td ---------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_LAYOUT_BASE_TD
#define MLIR_PHY_DIALECT_LAYOUT_BASE_TD

include "mlir/IR/OpBase.td"

def Layout_Dialect : Dialect {
  let name = "layout";
  let cppNamespace = "::xilinx::phy::layout";
  let dependentDialects = [
    "xilinx::phy::spatial::SpatialDialect",
  ];
  let description = [{
    This dialect defines the placement and routing of the graph defined in the
    'spatial' dialect.  In this dialect, Each node or queue vertex can be
    assigned a target-dependent slot, e.g., a core in the CPU, a coroutine in
    a host, a core or a memory bank in an AIE tile, or a reconfiguration slot
    in FPGA.  Each flow edge can be assigned an array of target-dependent
    wires, e.g., the switch boxes for a stream to flow through AIE tiles or a
    set of anchor registers between FPGA slots.

    This dialect is intended to be target-dependent, and forms an unambiguous
    description of the physical design when combined with the spatial dialect
    and when all operations in the spatial dialect has their wire or slot
    specified.  It separates hardware design from logic.
  }];
}

class Layout_Op<string mnemonic, list<Trait> traits = []> :
    Op<Layout_Dialect, mnemonic, traits>;

#endif  // MLIR_PHY_DIALECT_LAYOUT_BASE_TD
