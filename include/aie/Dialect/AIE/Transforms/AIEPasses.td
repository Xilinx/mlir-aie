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

  let options = [
    Option<"clAllocScheme", "alloc-scheme", "std::string", /*default=*/"",
           "Select allocation scheme: basic-sequential or bank-aware. Default is bank-aware, falling back to basic-sequential if it fails.">,
  ];
}

def AIEAssignLockIDs : Pass<"aie-assign-lock-ids", "DeviceOp"> {
  let summary = "Assigns the lockIDs of locks that do not have IDs.";
  let description = [{
    Assigns the lockIDs of locks that do not have IDs.
  }];

  let constructor = "xilinx::AIE::createAIEAssignLockIDsPass()";
}

def AIECanonicalizeDevice : Pass<"aie-canonicalize-device", "mlir::ModuleOp"> {
  let summary = "Canonicalize Designs to include a toplevel device";
  let description = [{
    This pass inserts a toplevel device operation in designs that do not have one.
    This allows us to support backwards compatability for older models targetting the VC1902
    device without explicit device operations. 
  }];

  let constructor = "xilinx::AIE::createAIECanonicalizeDevicePass()";
}

def AIECoreToStandard : Pass<"aie-standard-lowering", "mlir::ModuleOp"> {
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
    Option<"deviceName", "device", "std::string", /*default=*/"\"\"", "Device to generate code for">,
    Option<"tileCol", "tilecol", "unsigned",
           /*default=*/"-1", "X coordinate of tile to generate code for">,
    Option<"tileRow", "tilerow", "unsigned",
           /*default=*/"-1", "Y coordinate of tile to generate code for">
  ];

  let constructor = "xilinx::AIE::createAIECoreToStandardPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
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

def AIEAssignBufferDescriptorIDs : Pass<"aie-assign-bd-ids", "DeviceOp"> {
  let summary = "Assign bd ids to aie.dma_bd ops.";
  let constructor = "xilinx::AIE::createAIEAssignBufferDescriptorIDsPass()";
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
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

def AIERoutePathfinderFlows : Pass<"aie-create-pathfinder-flows", "DeviceOp"> {
  let summary = "Route aie.flow and aie.packetflow operations through switchboxes";
  let description = [{    
    Uses Pathfinder congestion-aware algorithm. 
    Each aie.flow is replaced with aie.connect operation.
    Each aie.packetflow is replace with the set of aie.amsel, aie.masterset 
    and aie.packet_rules operations.
  }];

  let constructor = "xilinx::AIE::createAIEPathfinderPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
  ];

  let options = [
    Option<"clRouteCircuit", "route-circuit", "bool", /*default=*/"true",
            "Flag to enable aie.flow lowering.">,      
    Option<"clRoutePacket", "route-packet", "bool", /*default=*/"true",
            "Flag to enable aie.packetflow lowering.">,     
  ];
}

def AIEFindFlows : Pass<"aie-find-flows", "DeviceOp"> {
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

def AIEVectorOpt : Pass<"aie-vector-opt", "mlir::func::FuncOp"> {
  let summary = "optimize vector instructions for AIE";
  let description = [{
    After super-vectorization, some additional optimizations are important
    for improving QOR and enabling lowering to LLVM.
  }];

  let constructor = "xilinx::AIE::createAIEVectorOptPass()";
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
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "xilinx::AIE::AIEDialect",
  ];

  let options = [
    Option<"clDynamicObjectFifos", "dynamic-objFifos", "bool", /*default=*/"false", 
           "Flag to enable dynamic object fifo lowering in cores instead of loop unrolling.">,      
    Option<"clPacketSwObjectFifos", "packet-sw-objFifos", "bool", /*default=*/"false",
           "Flag to enable aie.packetflow lowering from objectfifos.">, 
  ];
}

def AIEObjectFifoRegisterProcess : Pass<"aie-register-objectFifos", "DeviceOp"> {
  let summary = "Generate acquire/release patterns for producer/consumer processes registered to an objectFifo";
  let description = [{
    Generate acquire/release patterns in the CoreOps of associated cores for each 
    aie.objectfifo.register_process operation. Patterns are generated as for loops
    of different sizes depending on input patterns.
  }];

  let constructor = "xilinx::AIE::createAIEObjectFifoRegisterProcessPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

def AIELowerCascadeFlows : Pass<"aie-lower-cascade-flows", "DeviceOp"> {
  let summary = "Lower aie.cascade_flow operations through `aie.configure_cascade` operations";
  let description = [{
    Replace each aie.cascade_flow operation with an equivalent set of `aie.configure_cascade` 
    operations. 
  }];

  let constructor = "xilinx::AIE::createAIELowerCascadeFlowsPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
  ];
}

def AIEAssignTileCtrlIDs : Pass<"aie-assign-tile-controller-ids", "DeviceOp"> {
  let summary = "Assign controller id per aie.tile_op";
  let description = [{
    For each aie.tile_op used in the design, assign a unique controller ID.
  }];

  let constructor = "xilinx::AIE::createAIEAssignTileCtrlIDsPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
  ];
  let options = [
    Option<"clColumnWiseUniqueIDs", "column-wise-unique-ids", "bool", /*default=*/"true",
            "Flag to generate controller ids only unique within each column. Otherwise globally unique.">,
  ];
}

def AIEGenerateColumnControlOverlay : Pass<"aie-generate-column-control-overlay", "DeviceOp"> {
  let summary = "Spawns streaming interconnect network for column control";
  let description = [{
    For each column of AIE tiles being employed in the design, spawn a network of control packet
    streaming interconnects which overlay on top of the design.
  }];

  let constructor = "xilinx::AIE::createAIEGenerateColumnControlOverlayPass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
  ];
  let options = [
    Option<"clRouteShimCTRLToTCT", "route-shim-to-tct", "std::string", /*default=*/"\"shim-only\"",
            "Flag to generate TCT routing between tile CTRL and shim SOUTH ports. Available options: ['shim-only', 'all-tiles', 'disable'].">,      
    Option<"clRouteShimDmaToTileCTRL", "route-shim-to-tile-ctrl", "bool", /*default=*/"false",
            "Flag to generate routing between shim dma DMA and tile CTRL ports, for configuration.">,  
  ];
}

def AIESetELFforCore : Pass<"aie-set-elf-for-core", "DeviceOp"> {
  let summary = "Set the ELF file for a core";
  let description = [{
    This pass sets the `elf_file` attribute for the given core in the design,
    erasing the core's MLIR body.
  }];

  let constructor = "xilinx::AIE::createAIESetELFforCorePass()";
  let dependentDialects = [
    "xilinx::AIE::AIEDialect",
  ];

  let options = [
    Option<"clDevice", "device", "std::string", /*default=*/"\"\"", "Device for whose cores ELF file should be set.">,
    Option<"clTileCol", "col", "unsigned", /*default=*/"0", "Column index of tile to set the ELF file for.">,      
    Option<"clTileRow", "row", "unsigned", /*default=*/"0", "Row index of tile to set the ELF file for.">,  
    Option<"clElfFile", "elf-file", "std::string", /*default=*/"", "Path to the ELF file to set for the core.">,
  ];
}

#endif
