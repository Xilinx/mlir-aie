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
  // Several AIE constructs materialize arith ops via the builder API (e.g. the
  // Python wrappers in dialects/aie.py build arith.constant for runtime scalar
  // operands, and AIEX lowerings emit arith into runtime_sequence bodies). MLIR
  // requires dependent dialects to be pre-loaded before ops referencing them may
  // be built, so declaring the dependency at dialect level is the safe placement.
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

#endif // AIE_BASE