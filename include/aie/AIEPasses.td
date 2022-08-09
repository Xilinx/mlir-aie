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
    "func::FuncDialect",
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
    func.func @aie_task(%arg0: memref<256xi32>, %arg1: i32) -> () {
      ...
      return
    }
    %a = constant 0 : i32
    func.call @aie_task(%buf, %a) { aie.x = 1, aie.y = 1 } : (memref<256xi32>, i32) -> ()

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
    "func::FuncDialect",
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
def AIELocalizeLocks : Pass<"aie-localize-locks", "ModuleOp"> {
  let summary = "Convert global locks to a core-relative index";
  let description = [{
    An individual lock can be referenced by 4 different AIE cores.  However, each individual core
    accesses the lock through a different 'lock address space'.  This pass converts a lock in the
    conceptual global address space into a local index.  e.g.:
    ```
    %lock = AIE.lock(%tile, 2)
    AIE.core(%tile) {
      AIE.useLock(%lock, "Acquire", 1)
    }
    ```
    becomes
    ```
    AIE.core(%tile) {
      %lockindex = arith.constant ? : index
      AIE.useLock(%lockindex, "Acquire", 1)
    }
    ```
  }];

  let constructor = "xilinx::AIE::createAIELocalizeLocksPass()";
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
    "func::FuncDialect",
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

def AIEBroadcastPacket : Pass<"aie-lower-broadcast-packet", "ModuleOp"> {
  let summary = "Replace combination of broadcast and packet-switch by packet-flow";
  let description = [{
    Replace combination of broadcast and packet-switch by packet-flow  
  }];

  let constructor = "xilinx::AIE::createAIEBroadcastPacketPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
  ];
}

def AIEMulticast : Pass<"aie-lower-multicast", "ModuleOp"> {
  let summary = "Lower AIE.multicast operation to AIE.flow operations";
  let description = [{
    This pass replaces AIE.multicast operation with the equivalent number of AIE.flow 
    operations. The lowered AIE.flow operations have the same source port but different
    destinations.
  }];

  let constructor = "xilinx::AIE::createAIELowerMulticastPass()";
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
  let dependentDialects = [
    "func::FuncDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

def AIEVectorOpt : Pass<"aie-vector-opt", "func::FuncOp"> {
  let summary = "optimize vector instructions for AIE";
  let description = [{
    After super-vectorization, some additional optimizations are important
    for improving QOR and enabling lowering to LLVM.
  }];

  let constructor = "xilinx::AIE::createAIEVectorOptPass()";
  let dependentDialects = [
  ];
}

def AIEObjectFifoStatefulTransform : Pass<"aie-objectFifo-stateful-transform", "ModuleOp"> {
  let summary = "Instantiate the buffers and locks of aie.objectFifo.createObjectFifo operations";
  let description = [{
    Replace each aie.objectFifo.createObjectFifo operation with aie.buffer and aie.lock operations in the 
    producer tile. Convert aie.objectFifo.acquire, aie.objectFifo.release and aie.objectFifo.subviewAccess
    operations into useLock operations by keeping track of acquire/release operations on each objectFifo by
    each process.

    If the producer and consumer tiles of an aie.objectFifo.createObjectFifo operation are not adjacent, the 
    pass also establised aie.flow and aie.dma operations to enable communication between the tiles.
  }];

  let constructor = "xilinx::AIE::createAIEObjectFifoStatefulTransformPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

def AIEObjectFifoLoopUnroll : Pass<"aie-unroll-objectFifos", "ModuleOp"> {
  let summary = "Identify and unroll loops that contain operations on objectFifos based on their number of elements";
  let description = [{
    Extend the body of each loop that contains operations on objectFifos such that it is unrolled
    based on the number of elements in the objectFifos. If the number of iterations of the loop 
    cannot be divided pefectly by the unrolling factor, the pass duplicates the loop body after 
    the original loop.
  }];

  let constructor = "xilinx::AIE::createAIEObjectFifoLoopUnrollPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "func::FuncDialect",
    "arith::ArithmeticDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

def AIEObjectFifoRegisterProcess : Pass<"aie-register-objectFifos", "ModuleOp"> {
  let summary = "Generate acquire/release patterns for producer/consumer processes registered to an objectFifo";
  let description = [{
    Generate acquire/release patterns in the CoreOps of associated cores for each 
    aie.objectFifo.registerProcess operation. Patterns are generated as for loops
    of different sizes depending on input patterns.
  }];

  let constructor = "xilinx::AIE::createAIEObjectFifoRegisterProcessPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "func::FuncDialect",
    "arith::ArithmeticDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

#endif
