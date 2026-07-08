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
  // aie.dma_bd materializes arith.constant operands for its offset/len fields
  // (and its builders/canonicalizer create them), so arith must be loaded.
  let dependentDialects = ["::mlir::arith::ArithDialect"];
}

#endif // AIE_BASE