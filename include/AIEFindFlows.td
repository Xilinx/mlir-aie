// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#ifndef AIE_FIND_FLOWS
#define AIE_FIND_FLOWS

include "AIE.td"

def : Pat<(aie_WireOp (aie_CoreOp:$core $x, $y), $port, (aie_SwitchboxOp:$switch $x, $y), $b),
					(aie_FlowOp $core, $port)>;

#endif // AIE_FIND_FLOWS
