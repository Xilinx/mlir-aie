//===- AIEFindFlows.td -------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_FIND_FLOWS
#define AIE_FIND_FLOWS

include "mlir/IR/PatternBase.td"
include "aie/Dialect/AIE/IR/AIE.td"

def : Pat<(AIE_WireOp (AIE_CoreOp:$core $x, $y), $port, (AIE_SwitchboxOp:$switch $x, $y), $b), (AIE_FlowOp $core, $port)>;

#endif // AIE_FIND_FLOWS
