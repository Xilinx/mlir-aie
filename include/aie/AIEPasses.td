//===- AIEPasses.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_PASSES
#define AIE_PASSES

include "mlir/Pass/PassBase.td"

def AIEAssignBufferAddresses : Pass<"aie-assign-buffer-addresses", "ModuleOp"> {
  let summary = "Assign memory locations for buffers in each tile";
  let description = [{
    Buffers in a tile generally have an address that does not significantly
    matter in the design.  Hence, most of the time we can instantiate
    aie.buffer operations without an address.  This pass determines
    updates each aie.buffer operation without an address to have a
    well-defined address.  This enables later passes to have a
    consistent view of the memory map of a system.
  }];

  let constructor = "xilinx::AIE::createAIEAssignBufferAddressesPass()";
}

def AIECoreToStandard : Pass<"aie-standard-lowering", "ModuleOp"> {
  let summary = "Lowering operations in AIE cores' regions to Standard";
  let description = [{

    Outline code inside a particular AIE.core operation into the llvm dialect.
    BufferOp operations are converted to a GlobalMemrefOp and references to
    those buffers are converted to GetGlobalMemrefOp.  Other AIE operations
    inside the cores are generally lowered to appropriate function intrinsics.
    Other AIE operations (e.g. CoreOp, TileOp, LockOp) outside the core are removed.

  }];
  let options = [
    Option<"tileCol", "tilecol", "unsigned",
           /*default=*/"-1", "X coordinate of tile to generate code for">,
    Option<"tileRow", "tilerow", "unsigned",
           /*default=*/"-1", "Y coordinate of tile to generate code for">
  ];

  let constructor = "xilinx::AIE::createAIECoreToStandardPass()";
  let dependentDialects = [
    "StandardOpsDialect",
    "memref::MemRefDialect",
    "xilinx::AIE::AIEDialect",
  ];
}
def AIECreateCores : Pass<"aie-create-cores", "ModuleOp"> {
  let summary = "Create CoreOp, MemOp, and FlowOp of AIE dialect";
  let description = [{
    Lower toplevel 'call' operations with implicit data movement into
    aie.core, aie.mem, and aie.flow operations with explicit data
    movement.  The tile tile associated with the operations is inferred
    based on the aie.x and aie.y attributes of the call operation. e.g.:
    ```
    func @aie_task(%arg0: memref<256xi32>, %arg1: i32) -> () {
      ...
      return
    }
    %a = constant 0 : i32
    call @aie_task(%buf, %a) { aie.x = 1, aie.y = 1 } : (memref<256xi32>, i32) -> ()

    ```
    will result in:
    ```
    %0 = AIE.tile(1, 1)
    %1 = AIE.buffer(%0) : memref<256xi32>
    %2 = AIE.buffer(%0) : memref<1xi32>
    %c0_i32 = constant 0 : i32
    %5 = AIE.core(%0) {
      ...
    }
    ```
    In this case, each argument to aie_task results in an explicit buffer accessed by the core.

  }];

  let constructor = "xilinx::AIE::createAIECreateCoresPass()";
  let dependentDialects = [
    "StandardOpsDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

def AIECreateLocks : Pass<"aie-create-locks", "ModuleOp"> {
  let summary = "Generate physical lock ops from logical lock (token) ops";
  let description = [{
    Tokens represent high-level buffer synchronization through a sequence of
    pipeline stages.  This pass lowers token operations into physical aie.lock
    operations.
  }];

  let constructor = "xilinx::AIE::createAIECreateLocksPass()";
}
def AIEHerdRouting : Pass<"aie-herd-routing", "ModuleOp"> {
  let summary =       "Lowering herds with place and route ops to AIE cores, mems, and switchboxes";
  let description = [{
    An experimental pass which elaborates herd operations (e.g. aie.herd, aie.iter, aie.select)
    into an explicit representation (e.g. aie.core, aie.mem, etc.).
  }];

  let constructor = "xilinx::AIE::createAIEHerdRoutingPass()";
}

def AIENormalizeAddressSpaces : Pass<"aie-normalize-address-spaces", "ModuleOp"> {
  let summary = "Remove non-default address spaces";
  let description = [{
    Early in the flow, it is convenient to represent multiple memories using different
    address spaces.  However, after outlining code for AIE engines, the core itself only
    has access to a single address space.  To avoid confusion, this pass normalizes
    any address spaces remaining in the code, converting them to the default address
    space.
  }];

  let constructor = "xilinx::AIE::createAIENormalizeAddressSpacesPass()";
  let dependentDialects = [
    "StandardOpsDialect",
    "memref::MemRefDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

def AIERouteFlows : Pass<"aie-create-flows", "ModuleOp"> {
  let summary = "Route aie.flow operations through switchboxes";
  let description = [{
    Replace each aie.flow operation with an equivalent set of aie.switchbox and aie.wire
    operations.  
  }];

  let constructor = "xilinx::AIE::createAIERouteFlowsPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
  ];
}

def AIERoutePathfinderFlows : Pass<"aie-create-pathfinder-flows", "ModuleOp"> {
  let summary = "Route aie.flow operations through switchboxes with Pathfinder algorithm";
  let description = [{
    Replace each aie.flow operation with an equivalent set of aie.switchbox and aie.wire
    operations. Uses Pathfinder congestion-aware algorithm. 
  }];

  let constructor = "xilinx::AIE::createAIEPathfinderPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
  ];
}

def AIERoutePacketFlows : Pass<"aie-create-packet-flows", "ModuleOp"> {
  let summary = "Route aie.packetflow operations through switchboxes";
  let description = [{
    Replace each aie.packetflow operation with an equivalent set of aie.switchbox and aie.wire
    operations.  
  }];

  let constructor = "xilinx::AIE::createAIERoutePacketFlowsPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
  ];
}

def AIEFindFlows : Pass<"aie-find-flows", "ModuleOp"> {
  let summary = "Recover flows from switchbox configuration";
  let description = [{
    Under normal circumstances, every configured aie.switchbox operation
    should contribute to describing an end-to-end flow from one point
    to another.  These flows may be circuit-switched flows (represented
    by aie.flow) or a packet-switched connection (represensted by
    aie.packetflow).  This pass is primarily used for testing automatic
    routing.
  }];

  let constructor = "xilinx::AIE::createAIEFindFlowsPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
  ];
}

def AIELowerMemcpy : Pass<"aie-lower-memcpy", "ModuleOp"> {
  let summary = "Lower aie.memcpy operations to Flows and DMA programs";
  let description = [{
    aie.memcpy operations are an experimental high-level abstraction which
    move data from one buffer to another.
    This pass lowers them into appropriate aie.flow and aie.mem operations.
  }];

  let constructor = "xilinx::AIE::createAIELowerMemcpyPass()";
}

def AIEVectorOpt : Pass<"aie-vector-opt", "FuncOp"> {
  let summary = "optimize vector instructions for AIE";
  let description = [{
    After super-vectorization, some additional optimizations are important
    for improving QOR and enabling lowering to LLVM.
  }];

  let constructor = "xilinx::AIE::createAIEVectorOptPass()";
  let dependentDialects = [
  ];
}
#endif
