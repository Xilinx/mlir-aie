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

def AIEAssignBufferAddresses : Pass<"aie-assign-buffer-addresses", "DeviceOp"> {
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

def AIEAssignLockIDs : Pass<"aie-assign-lock-ids", "DeviceOp"> {
  let summary = "Assigns the lockIDs of locks that do not have IDs.";
  let description = [{
    Assigns the lockIDs of locks that do not have IDs.
  }];

  let constructor = "xilinx::AIE::createAIEAssignLockIDsPass()";
}

def AIECanonicalizeDevice : Pass<"aie-canonicalize-device", "ModuleOp"> {
  let summary = "Canonicalize Designs to include a toplevel device";
  let description = [{
    This pass inserts a toplevel device operation in designs that do not have one.
    This allows us to support backwards compatability for older models targetting the VC1902
    device without explicit device operations. 
  }];

  let constructor = "xilinx::AIE::createAIECanonicalizeDevicePass()";
}

def AIECoreToStandard : Pass<"aie-standard-lowering", "ModuleOp"> {
  let summary = "Lowering operations in AIE cores' regions to Standard";
  let description = [{

    Outline code inside AIE.core operations into the llvm dialect.
    BufferOp operations are converted to a GlobalMemrefOp and references to
    those buffers are converted to GetGlobalMemrefOp.  Other AIE operations
    inside the cores are generally lowered to appropriate function intrinsics.
    Other AIE operations (e.g. CoreOp, TileOp, LockOp) outside the core are removed.

    Optionally, tileCol and tileRow can specify a single core to export

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
def AIELocalizeLocks : Pass<"aie-localize-locks", "DeviceOp"> {
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
def AIENormalizeAddressSpaces : Pass<"aie-normalize-address-spaces", "DeviceOp"> {
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

def AIERoutePacketFlows : Pass<"aie-create-packet-flows", "DeviceOp"> {
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

def AIEObjectFifoStatefulTransform : Pass<"aie-objectFifo-stateful-transform", "DeviceOp"> {
  let summary = "Instantiate the buffers and locks of aie.objectFifo.createObjectFifo operations";
  let description = [{
    Replace each aie.objectFifo.createObjectFifo operation with aie.buffer and aie.lock operations in the 
    producer tile. Convert aie.objectFifo.acquire, aie.objectFifo.release and aie.objectFifo.subviewAccess
    operations into useLock operations by keeping track of acquire/release operations on each objectFifo by
    each process.

    If the producer and consumer tiles of an aie.objectFifo.createObjectFifo operation are not adjacent, the 
    pass also establised aie.flow and aie.dma operations to enable communication between the tiles.
    Extend the body of each loop that contains operations on objectFifos such that it is unrolled
    based on the number of elements in the objectFifos. If the number of iterations of the loop 
    cannot be divided pefectly by the unrolling factor, the pass duplicates the loop body after 
    the original loop.
  }];

  let constructor = "xilinx::AIE::createAIEObjectFifoStatefulTransformPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

def AIEObjectFifoRegisterProcess : Pass<"aie-register-objectFifos", "DeviceOp"> {
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
    "arith::ArithDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

#endif
