//===- AIEFindFlows.td -------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_FIND_FLOWS
#define AIE_FIND_FLOWS

include "AIE.td"

def : Pat<(aie_WireOp (aie_CoreOp:$core $x, $y), $port, (aie_SwitchboxOp:$switch $x, $y), $b),
					(aie_FlowOp $core, $port)>;

#endif // AIE_FIND_FLOWS
