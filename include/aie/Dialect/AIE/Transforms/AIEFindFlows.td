//===- AIEFindFlows.td -------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2019 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_FIND_FLOWS
#define AIE_FIND_FLOWS

include "mlir/IR/PatternBase.td"
include "aie/Dialect/AIE/IR/AIE.td"

def : Pat<(AIE_WireOp (AIE_CoreOp:$core $x, $y), $port, (AIE_SwitchboxOp:$switch $x, $y), $b), (AIE_FlowOp $core, $port)>;

#endif // AIE_FIND_FLOWS
