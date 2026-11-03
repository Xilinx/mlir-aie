//===- AIE.td ----------------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_BASE
#define AIE_BASE

include "mlir/IR/OpBase.td"

def AIE_Dialect : Dialect {
  let name = "aie";
  let cppNamespace = "::xilinx::AIE";
  let description = [{

    This is a dialect for describing netlists of AIE components in a
    Versal device.  It focuses on representing logical stream connections
    between cores and DMAs, along with the implementation of those logical
    connections in the various switch components.  In the dialect, a
    switch is referred to as `switchbox` to avoid confusion with the
    `switch` keyword in C/C++.

  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  // Python wrappers in dialects/aie.py and AIEX lowerings emit arith.constant
  // and other operations, so this dialect depends on arith being available.
  let dependentDialects = ["::mlir::arith::ArithDialect"];
  // The op assembly formats still print inherent attributes in `attr-dict`;
  // opt out of upstream's strict-properties default until they migrate.
  let useStrictPropertiesInAssemblyFormat = 0;
  // Overlay-owned settings survive application configuration and teardown.
  // On route endpoints and shim allocations, shareable_with lists same-device
  // endpoint symbols permitted to share their channel. An empty list forbids
  // sharing; an explicit list overrides lifetime analysis, but not hardware
  // constraints. Omitted attributes allow analysis to infer eligibility.
  // On devices, reserve_controller_packet_ids excludes controller IDs from allocation.
  let discardableAttrs = (ins "::mlir::UnitAttr":$is_ctrl_overlay,
      "::mlir::UnitAttr":$priority_route,
      "::mlir::ArrayAttr":$shareable_with,
      "::mlir::UnitAttr":$reserve_controller_packet_ids);
}

#endif // AIE_BASE
