//===- AIE.td ----------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_OPS
#define AIE_OPS

include "aie/Dialect/AIE/IR/AIE.td"
include "aie/Dialect/AIE/IR/AIEAttrs.td"
include "aie/Dialect/AIE/IR/AIEInterfaces.td"
include "aie/Dialect/AIE/IR/AIETypes.td"

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class AIE_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIE_Dialect, mnemonic, traits>;


def AIE_DeviceOp: AIE_Op<"device", [
    Symbol,
    AIETarget, 
    HasParent<"mlir::ModuleOp">,
    SymbolTable, 
    SingleBlockImplicitTerminator<"EndOp">, 
    IsolatedFromAbove
  ]> {
  let summary = "Define an AIE design targetting a complete device";
  let description = [{
    This operation describes a design that executes on a particular AIEngine device.
    It exists at the toplevel of a design; although currently it does not replace the
    default toplevel module in MLIR, the intention is that this could be the case
    in the future.

    When using this operation, all resources in a physical device are available and
    the design does not need to be concerned with other potential users of a physical
    device.  In addition, within an `aie.device` operation, tile addresses are absolute
    coordinates and are not intended to describe a relocatable design.  To describe
    a portion of a device which may be relocatable, the intention would be to provide another
    operation, for instance maybe `aie.segment`.
    The design itself is described using a region of code contained by the device
    operation.

    Example:
    ```
    aie.device(xcvc1902) {
      %tile = aie.tile(1, 1)
      %CORE = aie.core(%tile) { ... }
    }
    ```
  }];

  let arguments = (
    ins AIEDevice:$device,
        DefaultValuedAttr<SymbolNameAttr, "\"main\"">:$sym_name
  );
  let regions = (region AnyRegion:$body_region);
  let assemblyFormat = [{
    `(` $device `)` ($sym_name^)? regions attr-dict
  }];
  let extraClassDeclaration = [{
    const xilinx::AIE::AIETargetModel &getTargetModel();
    static xilinx::AIE::DeviceOp getForSymbolInModule(mlir::ModuleOp module, llvm::StringRef symbol);
    static xilinx::AIE::DeviceOp getForSymbolInModuleOrError(mlir::ModuleOp module, llvm::StringRef symbol);
  }];
}

def AIE_TileOp: AIE_Op<"tile", [
    Pure,
    FlowEndPoint,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index:$result)> {
  let arguments = (
    ins ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$col,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$row,
        OptionalAttr<StrAttr>:$allocation_scheme
  );

  let summary = "Declare an AIE tile";
  let description = [{
    This operation creates an AIE tile in the AIE array. We specify the column and the row of the tile. 
    Optionally, we can specify the memory allocation scheme for the tile (basic-sequential or bank-aware).

    A tile encompasses core module (CoreOp), memory module (MemOp), stream switch (SwitchboxOp),
    memory buffer (BufferOp), and lock (LockOp).

    A tile is a logical abstraction. We use a tile to establish ownership of a hardware entity.

    Note that row 0 of the Tile array is different from other rows, since it models the shim interface between
    the AIE array proper and the PL. The South-West/Lower Right most core exists in Tile(0,1).
  }];

  let extraClassDeclaration = [{
    size_t getNumSourceConnections(WireBundle bundle);
    size_t getNumDestConnections(WireBundle bundle);
    int colIndex() { return getCol(); }
    int rowIndex() { return getRow(); }
    TileID getTileID() { return {getCol(), getRow()}; }
    bool isShimTile() { return getRow() == 0; }
    bool isMemTile();
    bool isShimNOCTile();
    bool isShimPLTile();
    bool isShimNOCorPLTile();
    bool isInternalMemWest() { return ((rowIndex() % 2) == 0); };

    MemOp getMemOp() {
      auto users = getResult().getUsers();
      for(auto user : users)
        if(auto memOp = llvm::dyn_cast<MemOp>(*user))
          return memOp;
      return nullptr;
    }

    CoreOp getCoreOp() {
      auto users = getResult().getUsers();
      for(auto user : users)
        if(auto coreOp = llvm::dyn_cast<CoreOp>(*user))
          return coreOp;
      return nullptr;
    }

    static AIE::TileOp getOrCreate(mlir::OpBuilder builder, AIE::DeviceOp device, int col, int row);
  }];

  let assemblyFormat = [{
    `(` $col `,` $row `)` attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDefinition = [{
    void $cppClass::getAsmResultNames(
        function_ref<void(::mlir::Value, ::llvm::StringRef)> setNameFn) {
       std::string tileName =
             getOperationName().str().substr(getOperationName().find('.') + 1);
       // Specialize the SSA value name according to the tile kind
       if (isMemTile())
         tileName = "mem_" + tileName;
       else if (isShimNOCTile())
         tileName = "shim_noc_" + tileName;
       else if (isShimPLTile())
         tileName = "shim_pl_" + tileName;
       else if (isShimTile())
         tileName = "shim_" + tileName;
       setNameFn(getResult(), tileName + "_" +
                                  std::to_string(getCol()) + "_" +
                                  std::to_string(getRow()));
    }
  }];

  let builders = [
    OpBuilder<(ins "int32_t":$col, "int32_t":$row), [{
      build($_builder, $_state, col, row, nullptr);
    }]>,
    OpBuilder<(ins "mlir::Type":$type, "int32_t":$col, "int32_t":$row), [{
      build($_builder, $_state, type, col, row, nullptr);
    }]>
  ];
}

def AIE_EndOp: AIE_Op<"end", [Terminator]> {
  let summary = "end op";
  let description = [{
    A generic terminator operation for AIE ops' regions.
  }];
  let assemblyFormat = [{ attr-dict }];
}

def AIE_SwitchboxOp: AIE_Op<"switchbox", [
    Interconnect, TileElement,
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index:$result)> {
  let arguments = (ins Index:$tile);

  let summary = "Declare a switch";
  let description = [{
    This operation represents the switchbox that is part of a tile.  A switchbox is configured
    by code in its region, representing various connections

    Example:
    ```
    %tile = aie.tile(1, 1)
    aie.switchbox(%tile) {
      aie.connect<"West" : 0, "Core" : 1>
    }
    ```
  }];

  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    size_t getNumSourceConnections(WireBundle bundle);
    size_t getNumDestConnections(WireBundle bundle);
    int colIndex();
    int rowIndex();
    TileOp getTileOp();
    using ::xilinx::AIE::TileElement::Trait<SwitchboxOp>::getAsmResultNames;
  }];
}

def AIE_ShimSwitchboxOp: AIE_Op<"shim_switchbox", [
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (ins AIEI32Attr:$col);
  let summary = "Declare a switch in the PL shim";
  let description = [{

    A switch in the Shim.
    AXI-Stream Master Ports AXI-Stream Slave Ports
    6 Ports to North (Core Tile) 4 Ports from North (Core Tile)
    4 Ports to West 4 Ports from West
    4 Ports to East 4 Ports from East
    6 Ports to South (DMA, NoC I/F, PL I/F) 8 Ports from South (DMA, NoC I/F, PL I/F)
    2 Ports to FIFOs 2 Ports from FIFOs
    1 Port for control packet for Shim register access
    1 Port for response to access for Shim registers
    1 Port for trace packet from Shim

  }];
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $col `)` regions attr-dict }];
  let hasVerifier = 1;
}

def AIE_ShimMuxOp: AIE_Op<"shim_mux", [
    Interconnect, TileElement, IsShimTile,
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );
  let summary = "Declare a switch in the PL shim";
  let description = [{
    This operation represents the additional interconnect that is part of a shim interface tile.
    Like the `aie.switchbox` operation, `aie.shim_mux` is configured
    by code in its region, but can only contain connect operations

    Example:
    ```
    %tile = aie.tile(1, 1)
    aie.shim_mux(%tile) {
      aie.connect<"North" : 0, "DMA" : 1>
    }
    ```
  }];
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    size_t getNumSourceConnections(WireBundle bundle);
    size_t getNumDestConnections(WireBundle bundle);
    TileOp getTileOp();
    using ::xilinx::AIE::TileElement::Trait<ShimMuxOp>::getAsmResultNames;
  }];
}

def AIE_ShimDMAOp: AIE_Op<"shim_dma", [
    FlowEndPoint, TileElement, HasValidBDs,
    HasValidDMAChannels, IsShimNOCTile,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );
  let summary = "Declare a DMA in the PL shim";
  let description = [{
    This operation creates a DMA that belongs to a shim tile.
    The region of a ShimDMAOp is used to setup the DMAs and Block Descriptors.

    Example:
    ```
      %buf = aie.external_buffer : memref<256xi64>
      %lock1 = aie.lock(%t70, 1)

      %dma = aie.shim_dma(%t70) {
          aie.dma_start(MM2S, 0, ^bd0, ^end)
        ^bd0:
          aie.use_lock(%lock1, Acquire, 1)
          aie.dma_bd(%buf : memref<512 x i16>, 0, 512)
          aie.use_lock(%lock1, Release, 0)
          aie.next_bd ^bd0
        ^end:
          aie.end
      }
    ```
    Create the shim_dma for tile `%t70` and setup one DMA channel and one Buffer Descriptor.
  }];
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    TileOp getTileOp();
    using ::xilinx::AIE::TileElement::Trait<ShimDMAOp>::getAsmResultNames;
  }];
}

def AIE_CoreOp: AIE_Op<"core", [
    FlowEndPoint, IsCoreTile, TileElement,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins Index:$tile,
    DefaultValuedAttr<AIEI32Attr, "0x400">:$stack_size,
    OptionalAttr<StrAttr>:$link_with,
    OptionalAttr<StrAttr>:$elf_file,
    OptionalAttr<BoolAttr>:$dynamic_objfifo_lowering
  );
  let summary = "Declare a core module";
  let description = [{
    This operation represents an AIEngine processor core belonging to a tile.
    The region of a CoreOp contains code that gets run on the AIE core.  This code will
    typically be outlined into the LLVM dialect, eventually resulting in a binary file
    for each core.  The name of this file can be be specified using the `elf_file`
    attribute.

    If the `elf_file` attribute is present, no MLIR besides a terminator may be
    present in the core; in that case, the binary file linked dictates what 
    will run in the core. The path specified should is relative to the MLIR
    file.

    This op has an optional `stackSize` attribute, to control the amount of memory (in bytes)
    reserved for the stack.  The default value is 1024.  The stack (and other data allocations)
    are always stored in the local core memory, to avoid conflicts with static data allocations
    in other cores.

    This op has an optional `dynamic_objfifo_lowering` attribute, to finely control whether the
    objectfifos in this core should be lowered using the dynamic runtime lowering.

    Examples:
    ```
    %tile = aie.tile(1, 1)
    %lock11_8 = aie.lock(%tile, 8)
    aie.core(%tile) {
      aie.use_lock(%lock11_8, "Acquire", 1)
      aie.use_lock(%lock11_8, "Release", 0)
      aie.end
    }
    ```
    ```
    %tile = aie.tile(3, 3)
    aie.core(%tile) {
      aie.end
    } { stackSize = 2048 : i32, elf_file = "core_33.elf" }
    ```
  }];

  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    bool isMemWest() { return ((rowIndex() % 2) == 0); };
    bool isEmpty();
    TileOp getTileOp();
    using ::xilinx::AIE::TileElement::Trait<CoreOp>::getAsmResultNames;
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$tile), [{
      build($_builder, $_state, $_builder.getIndexType(), tile);
    }]>
  ];
}

def AIE_DebugOp: AIE_Op<"debug", []> {
  let arguments = (
    ins AnyType:$arg
  );
  let summary = "Capture a value for debugging";
  let description = [{
    Output the given value for debugging.  This is primarily used for simulation.
  }];
  let assemblyFormat = [{ `(` $arg `:` type($arg) `)` attr-dict }];
}

def AIE_PLIOOp: AIE_Op<"plio", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (ins AIEI32Attr:$col);
  let summary = "Declare an interface to the PL";
  let description = [{
    An interface to the PL.
  }];
  let assemblyFormat = [{ `(` $col `)` attr-dict }];

  let extraClassDeclaration = [{
    int colIndex() { return getCol(); }
  }];
}

def AIE_ConnectOp: AIE_Op<"connect", [ParentOneOf<["SwitchboxOp", "ShimMuxOp"]> ]> {
  let arguments = (
    ins WireBundle:$source_bundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$source_channel,
        WireBundle:$dest_bundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$dest_channel
  );
  let summary = "A circuit-switched connection inside a switchbox";
  let description = [{
    This operation represents a programmed circuit-switched connection in a stream switch.
    It associates a source bundle and source channel with a destination bundle and a destination channel.
    This operation must exist within an `aie.switchbox` or `aie.shim_switchbox` operation.
    All of the `aie.connect` operations in a switchbox must have different destinations.
    All of the `aie.connect` operations must also have a destination which is different from all
    of the `aie.masterset` operations in the same switchbox.

    Example:
    ```
    %tile = aie.tile(1, 1)
    aie.switchbox(%tile) {
      aie.connect<"West" : 0, "Core" : 1>
    }
    ```
  }];

  let assemblyFormat = [{
    `<` $source_bundle `:` $source_channel `,` $dest_bundle `:` $dest_channel `>` attr-dict
  }];

  let extraClassDeclaration = [{
    int sourceIndex() { return getSourceChannel(); }
    int destIndex() { return getDestChannel(); }
    Port sourcePort() { return {getSourceBundle(), sourceIndex()}; }
    Port destPort() { return {getDestBundle(), destIndex()}; }
  }];
}

def AIE_FlowOp: AIE_Op<"flow", []> {
  let arguments = (
    ins Index:$source,
        WireBundle:$source_bundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$source_channel,
        Index:$dest,
        WireBundle:$dest_bundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$dest_channel
  );
  let summary = "A logical circuit-switched connection between cores";
  let description = [{
    The `aie.flow` operation represents a circuit switched connection between two endpoints, usually
    `aie.tile` operations.  During routing, this is replaced by `aie.connect` operations which represent
    the programmed connections inside a switchbox, along with `aie.wire` operations which represent
    physical connections between switchboxes and other components.

    Example:
    ```
      %00 = aie.tile(0, 0)
      %11 = aie.tile(1, 1)
      %01 = aie.tile(0, 1)
      aie.flow(%00, "DMA" : 0, %11, "Core" : 1)
    ```
  }];

  let assemblyFormat = [{
    `(` $source `,` $source_bundle `:` $source_channel `,` $dest `,` $dest_bundle `:` $dest_channel `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int sourceIndex() { return getSourceChannel(); }
    int destIndex() { return getDestChannel(); }
  }];
}

def AIE_AMSelOp: AIE_Op<"amsel", [
    HasParent<"SwitchboxOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins ConfinedAttr<AIEI8Attr, [IntMinValue<0>, IntMaxValue<5>]>:$arbiterID,
        ConfinedAttr<AIEI8Attr, [IntMinValue<0>, IntMaxValue<3>]>:$msel
  );
  let summary = "Declare an arbiter of a switchbox with a master select value (arbiter + msel)";
  let description = [{
    A combination of arbiter ID and master select (msel) value.
    This op is used as a pointer to select the arbiter for routing a packet-switched flow

    Example:
    ```
        %a0_0 = aie.amsel<5>(3)
        %m1 = aie.masterset("East" : 0, %a0_0 )
        aie.packet_rules("South" : 0) {
          aie.rule(0x1F, 0x10, %a0_0)
        }
    ```
    This code associates arbiter 5 with msel=3.  A packet-switched connection is made routing
    traffic from the South:0 port to the East:0 port using this arbiter.
    There are 6 arbiters per switchbox and 4 possible master select values.
    See also [MasterSetOp](#aiemasterset-aiemastersetop),
    [PacketRulesOp](#aiepacketrules-aiepacketrulesop), and
    [PacketRuleOp](#aierule-aiepacketruleop) for more information.
  }];

  let assemblyFormat = [{
    `<` $arbiterID `>` `(` $msel `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int arbiterIndex() { return getArbiterID(); }
    int getMselValue() { return getMsel(); }
  }];

  let builders = [
    OpBuilder<(ins "int":$arbiterID, "int":$msel),
    [{
      build($_builder, $_state, $_builder.getIndexType(),
            $_builder.getI8IntegerAttr(arbiterID),
            $_builder.getI8IntegerAttr(msel));
    }]>
  ];
}

def AIE_MasterSetOp: AIE_Op<"masterset", [
    HasParent<"SwitchboxOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins WireBundle:$dest_bundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$dest_channel,
        Variadic<Index>:$amsels,
        OptionalAttr<BoolAttr>:$keep_pkt_header
  );
  let summary = "Packet switched input connection";
  let description = [{
    A packet switched connection inside a switchbox.
    This operation specifies the configuration for a master port.

    Example:
    ```
      %a0_m2 = aie.amsel<0>(2)
      aie.masterset("Core" : 0, %a0_m2)
    ```

    The code will configure the master port <"Core" : 0> to use arbiter 0 with msel 2
    (see AMSelOp for more details regarding AMSel)

    In the current architecture, a master port can only be associated with one arbiter. However,
    a master port can be activated by different msels from one arbiter

    Example:
    ```
      %a1_0 = aie.amsel<1>(0)
      %a1_1 = aie.amsel<1>(1)
      %a2_3 = aie.amsel<2>(3)

      aie.masterset("West" : 2, %a1_0, %a2_3) // this is illegal, please don't do this
      aie.masterset("West" : 3, %a1_0, %a1_1) // this is OK
    ```
  }];

  let assemblyFormat = [{
    `(` $dest_bundle `:` $dest_channel `,` $amsels `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int destIndex() { return getDestChannel(); }
    Port destPort() { return {getDestBundle(), destIndex()}; }
  }];
}

def AIE_WireOp: AIE_Op<"wire", []> {
  let arguments = (
    ins Index:$source,
        WireBundle:$source_bundle,
        Index:$dest,
        WireBundle:$dest_bundle
  );
  let summary = "A bundle of physical wires between components";
  let description = [{
    The `aie.wire` operation represents a physical set of connections between components in a Versal device.
    Typically, these components are switches, represented by an `aie.switchbox` operation, and tiles,
    represented by an [aie.tile](#aietile-aietileop) operation.
  }];

  let assemblyFormat = [{
    `(` $source `:` $source_bundle `,` $dest `:` $dest_bundle `)` attr-dict
  }];
}

def AIE_PacketRulesOp: AIE_Op<"packet_rules", [SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins WireBundle:$source_bundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$source_channel
  );
  let regions = (region AnyRegion:$rules);
  let summary = "Packet switched routing rules";
  let description = [{
    This operation defines packet-switched routing configuration for packets entering a switchbox.
    It references a port of the containing switchbox, which must be unique among other packetRules
    operations and [aie.connect]($aieconnect-aieconnectop) operations in the containing switchbox.
    It contains a region of up to 4 [aie.rule](#aierule-aiepacketruleop) operations.

    See [aie.rule](#aierule-aiepacketruleop) for an example.
  }];

  let assemblyFormat = [{ `(` $source_bundle `:` $source_channel `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    int sourceIndex() { return getSourceChannel(); }
    Port sourcePort() { return {getSourceBundle(), sourceIndex()}; }
  }];
}

def AIE_PacketRuleOp: AIE_Op<"rule", [HasParent<"PacketRulesOp">]> {
  let arguments = (
    ins AIEI8Attr:$mask,
        AIEI8Attr:$value,
        Index:$amsel
  );
  let summary = "Packet switched routing rule";
  let description = [{
    This operation defines a matching rule and a destination for packet-switched
    connections in a switchbox.  Routing is based on the ID field of packet arriving on the
    matching port of the containing [aie.packetRules](#aiepacketrules-aiepacketrulesop).
    The ID is first bitwise-AND'd with the mask and then checked for equality with the given ID.
    It is routed to arbiter and master set associated with the first matching entry.

    Example:
      LUT ID  |  Mask     | ID          | Arbiter | Msel
      ---     | ---       | ---         | ---     | ---
      0       |  5'b11111 | 5'b00010    | 4       | 1
      1       |  5'b11011 | 5'b00001    | 3       | 2
      2       |           |             |         |
      3       |           |             |         |

    If a packet flow that has an ID of 2, it will be directed to the arbiter 4 with msel 1,
    If a packet flow that has an ID of 1 or 5, it will be directed to the arbiter 3 with msel 2,

    We encapsulate the configuration table as follows:
    Example:
    ```
      %a4_1 = aie.amsel<4>(1)
      %a3_2 = aie.amsel<3>(2)

      aie.packet_rules("Core" : 0) {
        aie.rule(0x1F, 0x2, %a4_1)
        aie.rule(0x1B, 0x1, %a3_2)
      }
    ```
  }];

  let extraClassDeclaration = [{
    int maskInt() { return getMask(); }
    int valueInt() { return getValue(); }
  }];

  let assemblyFormat = [{
    `(` $mask `,` $value `,` $amsel `)` attr-dict
  }];
}


def AIE_PacketFlowOp: AIE_Op<"packet_flow", [SingleBlockImplicitTerminator<"EndOp">]> {
  let summary = "Packet switched flow";
  let description = [{
    A logical packet-switched flow between tiles.  During place and
    route, this is replaced by MasterSets and PacketRules inside
    switchboxes.

    The optional attribute keep_pkt_header indicates whether each
    data packet's packet header gets preserved at the flow's
    destination. The optional attribute priority_route indicates
    whether the packet flow is routed in priority over other flows,
    so that they always get allocated with the same master, slave
    ports, arbiters and master selects (msel).

    Example:
    ```
      %01 = aie.tile(0, 1)
      aie.packet_flow(0x10) {
        aie.packet_source<%01, "Core" : 0>
        aie.packet_dest<%01, "Core" : 0>
      }
    ```
  }];

  let arguments = (
    ins AIEI8Attr:$ID,
        OptionalAttr<BoolAttr>:$keep_pkt_header,
        OptionalAttr<BoolAttr>:$priority_route
  );
  let regions = (region AnyRegion:$ports);

  let assemblyFormat = [{ `(` $ID `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    int IDInt() { return getID(); }
  }];
}

def AIE_PacketSourceOp: AIE_Op<"packet_source", [HasParent<"PacketFlowOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$channel
  );
  let summary = "A sourceport";
  let description = [{
    An object representing the destination of a packet-switched flow. This must exist
    within an [aie.packet_flow](#aiepacketflow-aiepacketflowop) operation.

    See [aie.packet_flow](#aiepacketflow-aiepacketflowop) for an example.
  }];

  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];

  let extraClassDeclaration = [{
    int channelIndex() { return getChannel(); }
    Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_PacketDestOp: AIE_Op<"packet_dest", [HasParent<"PacketFlowOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$channel
  );
  let summary = "A destination port";
  let description = [{
    A object representing the destination of a packet-switched flow. This must exist
    within an [aie.packet_flow](#aiepacketflow-aiepacketflowop) operation. The destination
    Must be unique within a design.

    See [aie.packet_flow](#aiepacketflow-aiepacketflowop) for an example.
  }];

  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];

  let extraClassDeclaration = [{
    int channelIndex() { return getChannel(); }
    Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_DMABDPACKETOp: AIE_Op<"dma_bd_packet", []> {
  let summary = "Enable packet headers for a dma block descriptor";
  let description = [{
    This operation enables packet headers for a block descriptor for DMA operations. In particular, it specifies
    the packet type (3-bits) and packet ID (5-bits).

    This operation must be used in an MLIR block that lives inside a MemOp's region, and before aie.dma_bd.
    The block descriptor specifies what lock to use and the buffer configuration.

    Example:
    ```
      // this defines a BD that uses lock %lck0 and buffer %buf0
      ^bd5:
        aie.use_lock(%lck, "Acquire", 0)
        aie.dma_bd_packet(0x4, 0xD)
        aie.dma_bd(<$buf0 : memref<512xi32>, 0, 512>, 1)
        aie.use_lock(%lck, "Release", 1)
        br ^bd6 // point to the next Block, which is also a different Block Descriptor
    ```
  }];

  let arguments = (
    ins AIEI32Attr:$packet_type,
        AIEI32Attr:$packet_id
  );

  let assemblyFormat = [{
    `(` $packet_type `,` $packet_id `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int getPacketID() { return getPacketId(); }
  }];
}

def AIE_DMABDOp: AIE_Op<"dma_bd", []> {
  let summary = "Declare a dma buffer descriptor op";
  let description = [{
    This operation describes a buffer descriptor for DMA operations. In particular, it specifies
    what buffer to use, and optionally:

        1. the offset into the buffer;
        2. the transfer length;
        3. the sizes and strides for n-d tensor addressing (described below);
        4. the "bd_id" with which to associate the buffer descriptor (most often left empty);
        5. the number of zeros to pad before and after every dimension of an n-d tensor (described below);
        6. the burst length to use for the DMA operation. This option is only available for data movement between the host memory and Shim NOC tiles.
          The default value of 0 is interpreted as the maximum available burst length in the architecture.

    `offset`, `len`, `size`s and `stride`s are all denominated in element width; e.g., transferring the whole of
    `memref<512xi32>` means `len == 512`, and also while transferring the whole of `memref<512xi16>`, `len == 512`.

    The only caveat to this "everything-is-in-terms-of-element-width" rule is regarding the inner-most dimension's stride
    (see [Important gotcha regarding strides](#important-gotcha-regarding-strides) below).

    `dma_bd` ops must appear in their own BBs (basic blocks) and such BBs can (optionally) include `use_lock`
    operations (specifying an "acquire" and a "release" lock; see the `use_lock` operation) and subsequent BDs in
    a "chain" of BDs (using `next_bd` as a "jump" to the next BB which contains a `dma_bd`).

    Example:
    ```
      // this defines a BD that uses lock %lck0 and buffer %buf0
      ^bd5:
        aie.use_lock(%lck, "Acquire", 0)
        // transfer the first 32 elements of the memref
        aie.dma_bd(<$buf0 : memref<128xi32>, 0, 32)
        aie.use_lock(%lck, "Release", 1)
        aie.next_bd ^bd6 // point to the next bb, which describes the next buffer descriptor
      ^bd6:
        aie.use_lock(%lck, "Acquire", 1)
        // transfer the last 32 elements of the memref
        aie.dma_bd(<$buf1 : memref<128xi32>, 96, 32)
        aie.use_lock(%lck, "Release", 0)
        aie.next_bd ^end

      ...

      // this defines a BD that does not use any lock
      ^bd8:
        aie.dma_bd(<$buf2 : memref<64xi32>, 0, 64)
    ```

    #### Background/context

    A DMA channel in a Memory Module can process one buffer descriptor after another by chaining them.
    There are 16 buffer descriptors per Core memory module and 48 buffer descriptors per Memtile memory module.
    They are shared by four DMA channels (or 12).

    #### DMA Data Layout Transformations on AIE-ML Devices

    AIE-ML devices can apply data layout transformations at the buffer
    descriptor level. These transformation are described by strides and sizes in up to three dimensions (four
    dimensions on memtiles). Strides and sizes can be supplied to the `dma_bd`
    through an optional argument, an array of "tuple-like" attributes `bd_dim_layout<size, stride>`.

    The first element of this array gives the outer-most dimension's stride and
    size, the last element of the array gives the inner-most dimension's stride and size.
    We can model the access pattern strides and sizes generate by a series of
    nested loops. In general, a set of strides and sizes like this...

    ```
    [<size_2, stride_2>, <size_1, stride_1>, <size_0, stride_0>]
    ```

    ...translates to an access pattern that can be expressed like this:

    ```
    int *buffer;  // i32
    for(int i = 0; i < size_2; i++)
      for(int j = 0; j < size_1; j++)
        for(int k = 0; k < size_0; k++)
          // access/store element at/to buffer[  i * stride_2
          //                                   + j * stride_1
          //                                   + k * stride_0]
    ```

    The following example shows an access pattern that corresponds to
    alternating between even and odd elements of the buffer/stream every 8
    elements:

    ```
    aie.dma_bd(%buf : memref<128xi32>, 0, 128, [<8, 16>, <2, 1>, <8, 2>])
    ```

    implies

    ```
    for(int i = 0; i < 8 /*size_2*/; i++)
      for(int j = 0; j < 2 /*size_1*/; j++)
        for(int k = 0; k < 8 /*size_0*/; k++)
          // access/store element at/to index (i * 16 /*stride_2*/ + j * 1 /*stride_1*/ + k * 2 /*stride_0*/)
    ```

    #### Important gotcha regarding strides

    All strides are expressed in multiples of the element width (just like `len` and `offset`)
    **with the caveat that the inner-most dimension's stride must be 1**.

    ## DMA constant padding on AIE-ML Devices

    AIE-ML devices can apply constant padding at the buffer descriptor level, described with pairs of padding
    counts before and after a dimension, to all dimensions in the data layout transformations. The padding
    counts can be supplied to the `dma_bd` through an optional argument, an array of "tuple-like" attributes
    `bd_pad_layout<const_pad_before, const_pad_after>`, followed by an optional argument `const_val` (default
    is 0). All counts are expressed in multiples of the element width.
  }];

  let arguments = (
    ins AnyMemRef:$buffer,
        // in multiples of element width (not bytes)
        DefaultValuedAttr<AIEI32Attr, "0">:$offset,
        // in multiples of element width (not bytes)
        OptionalAttr<AIEI32Attr>:$len,
        OptionalAttr<BDDimLayoutArrayAttr>:$dimensions,
        OptionalAttr<BDPadLayoutArrayAttr>:$pad_dimensions,
        DefaultValuedOptionalAttr<AIEI32Attr, "0">:$pad_value,
        OptionalAttr<AIEI32Attr>:$bd_id,
        OptionalAttr<PacketInfoAttr>:$packet,
        DefaultValuedOptionalAttr<AIEI32Attr, "0">:$burst_length,
        // should never be assigned by user...
        OptionalAttr<AIEI32Attr>:$next_bd_id
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    BufferOp getBufferOp();
    int32_t getBufferElementTypeWidthInBytes() {
      mlir::DataLayout dataLayout = mlir::DataLayout::closest(*this);
      return dataLayout.getTypeSize(getBuffer().getType().getElementType());
    }
    uint32_t getLenInBytes() {
      if (std::optional<uint32_t> len = getLen(); len.has_value())
        return len.value() * getBufferElementTypeWidthInBytes();
      else
        return getBuffer().getType().getNumElements() * getBufferElementTypeWidthInBytes();
    }
    int32_t getOffsetInBytes() { return getOffset() * getBufferElementTypeWidthInBytes(); }
  }];

  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
    }]>,
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "BDDimLayoutArrayAttr":$dims), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
      $_state.addAttribute("dimensions", dims);
    }]>,
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "BDPadLayoutArrayAttr":$paddims), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
      $_state.addAttribute("pad_dimensions", paddims);
    }]>,
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "BDDimLayoutArrayAttr":$dims, "BDPadLayoutArrayAttr":$paddims), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
      $_state.addAttribute("dimensions", dims);
      $_state.addAttribute("pad_dimensions", paddims);
    }]>,
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "PacketInfoAttr":$pkt), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
      $_state.addAttribute("packet", pkt);
    }]>,
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "BDDimLayoutArrayAttr":$dims, "PacketInfoAttr":$pkt), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
      $_state.addAttribute("dimensions", dims);
      $_state.addAttribute("packet", pkt);
    }]>,
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "BDPadLayoutArrayAttr":$paddims, "PacketInfoAttr":$pkt), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
      $_state.addAttribute("pad_dimensions", paddims);
      $_state.addAttribute("packet", pkt);
    }]>,
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "BDDimLayoutArrayAttr":$dims, "BDPadLayoutArrayAttr":$paddims, "PacketInfoAttr":$pkt), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
      $_state.addAttribute("dimensions", dims);
      $_state.addAttribute("pad_dimensions", paddims);
      $_state.addAttribute("packet", pkt);
    }]>
  ];
}

def AIE_DMAStartOp: AIE_Op<"dma_start", [
    ParentOneOf<["MemOp", "MemTileDMAOp", "mlir::func::FuncOp", "ShimDMAOp"]>,
    Terminator,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs I1:$valid)> {

  let summary = "An op to start DMA";
  let description = [{
    This operation declares a DMA channel to be used for data transfer. It usually exists inside
    either a MemOp (representing a TileDMA), a MemTileDMAOp (representing a DMA in a MemTile),
    or in a ShimDMAOp (representing a ShimDMA).
    A channel is defined by a direction (i.e., MM2S or S2MM) and an index.

    Example:
    ```
        aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        aie.use_lock(%lock0, "Acquire", 0)
        aie.dma_bd(%buffer : memref<16 x f32>, 0, 16)
        aie.use_lock(%lock0, "Release", 1)
        br ^bd0
      ^end:
        aie.end
    ```

    Conceptually, the aie.dma_start operation is a terminator that either passes
    control to a basic block containing DMA operations (through its first successor)
    or to a basic block for another dma_start, to an aie.end operation.
  }];

  let arguments = (
    ins DMAChannelDir:$channel_dir,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$channel_index,
        // repeat_count==0 means "do it once" and don't repeat
        DefaultValuedAttr<AIEI32Attr, "0">:$repeat_count
  );
  let successors = (successor AnySuccessor:$dest, AnySuccessor:$chain);
  let assemblyFormat = [{
    `(` $channel_dir `,` $channel_index `,` $dest `,` $chain (`,` `repeat_count` `=` $repeat_count^)? `)` attr-dict
  }];

  let extraClassDeclaration = [{
    bool isSend() { return getChannelDir() == DMAChannelDir::MM2S; }
    bool isRecv() { return getChannelDir() == DMAChannelDir::S2MM; }
  }];

  let hasCanonicalizer = 1;
}

def AIE_DMAOp: AIE_Op<"dma", [
    ParentOneOf<["MemOp", "MemTileDMAOp", "ShimDMAOp"]>,
    NoTerminator,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]>, Results<(outs I1:$valid)> {

  let summary = "An op to describe a set of DMA operations.";

  let arguments = (
    ins DMAChannelDir:$channel_dir,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$channel_index,
        DefaultValuedAttr<BoolAttr, "true">:$loop,
        // repeat_count==0 means "do it once" and don't repeat
        DefaultValuedAttr<AIEI32Attr, "0">:$repeat_count,
        OptionalAttr<StrAttr>:$sym_name
  );
  let regions = (region VariadicRegion<SizedRegion<1>>:$bds);
  let assemblyFormat = [{
    `(` $channel_dir `,` $channel_index `)`
    attr-dict ` `
    `[`regions`]`
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      if (auto name = getOperation()->getAttrOfType<mlir::StringAttr>(
              mlir::SymbolTable::getSymbolAttrName()))
        setNameFn(getResult(), name.str());
    }
  }];
}

def AIE_MemOp: AIE_Op<"mem", [
    TileElement, FlowEndPoint,
    IsCoreTile, HasValidBDs, HasValidDMAChannels,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let summary = "Declare a memory op";
  let description = [{
    This operation creates a Memory module that belongs to a tile.
    The region of a MemOp is used to setup the DMAs and Block Descriptors.
    See DMAStartOp and DMABdOp for more concrete examples on DMAs and Block Descriptors.

    Example:
    ```
      m73 = aie.mem(%t73) {
          %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^end)
        ^bd0:
          aie.use_lock(%lock, "Acquire", 0)
          aie.dma_bd(%buf : memref<64xi16>, 0, 64)
          aie.use_lock(%lock, "Release", 1)
          aie.next_bd ^bd0
        ^end:
          aie.end
      }
    ```
    Create the memory module for tile %t73 and setup one DMA channel and one Buffer Descriptor.
  }];

  let arguments = (ins Index:$tile);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    TileOp getTileOp();
    using ::xilinx::AIE::TileElement::Trait<MemOp>::getAsmResultNames;
  }];
}

def AIE_MemTileDMAOp: AIE_Op<"memtile_dma", [
    TileElement, FlowEndPoint,
    IsMemTile, HasValidBDs, HasValidDMAChannels,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let summary = "Declare a memtile_dma op";
  let description = [{
    This operation describes a DMA inside an AIE2 MemTile.
    The region of the op is used to setup the DMAs and Block Descriptors.
    See DMAStartOp and DMABdOp for more concrete examples on DMAs and Block Descriptors.

    This operation is restricted to certain compatible tiles in AIE2 devices:
    xcve2302: row 1
    xcve2802: row 1 and 2

    Example:
    ```
      m73 = aie.memtile_dma(%t71) {
          %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^end)
        ^bd0:
          aie.use_lock(%lock, "Acquire", 0)
          aie.dma_bd(%buf : memref<64xi16>, 0, 64>, 0)
          aie.use_lock(%lock, "Release", 1)
          aie.next_bd ^bd0
        ^end:
          aie.end
      }
    ```
    Create a description for tile `%t73` and setup one DMA channel and one Buffer Descriptor.
  }];

  let arguments = (ins Index:$tile);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    TileOp getTileOp();
    using ::xilinx::AIE::TileElement::Trait<MemTileDMAOp>::getAsmResultNames;
  }];
}

def AIE_NextBDOp: AIE_Op<"next_bd", [Terminator]> {
  let summary = "The next buffer descriptor";
  let description = [{
    This operation terminates the basic block describing a buffer descriptor inside
    a tile or shim DMA operation.  It references a single following buffer descriptor.
    Note that unlike other terminators (like cf.br), canonicalization should not remove
    the `next_bd` terminator, since it would result in invalid buffer descriptors.

    Example:
    ```
      m73 = aie.mem(%t73) {
          %srcDma = aie.dma_start("S2MM", 0, ^bd0, ^end)
        ^bd0:
          aie.use_lock(%lock, "Acquire", 0)
          aie.dma_bd(%buf : memref<64xi16>, 0, 64)
          aie.use_lock(%lock, "Release", 1)
          aie.next_bd ^bd0
        ^end:
          aie.end
      }
    ```
  }];

  let successors = (successor AnySuccessor:$dest);

  let assemblyFormat = [{
    $dest attr-dict
  }];
}

def AIE_LockOp: AIE_Op<"lock", [
    TileElement,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let summary = "Declare a physical lock";
  let description = [{
    This operation creates a physical lock. For this operation the lockID variable is optional.
    However, if that is the case then the lockID must be assigned using the AIEAssignLockIDs pass.

    Example:
    ```
      %tile33 = aie.tile(3, 3)
      %lck = aie.lock(%tile33, 7)
    ```
    This operation represents a lock that lives in the Memory module of Tile(3, 3) with a lockID of 7

    Case when LockID is not assigned:
      Before AIEAssignLockIDs: `%tile33 = aie.tile(3)`
      After AIEAssignLockIDs: `%tile33 = aie.tile(3, $assigned_value)`
  }];

  let arguments = (
    ins Index:$tile,
      OptionalAttr<ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>>:$lockID,
      OptionalAttr<AIEI32Attr>:$init,
      OptionalAttr<StrAttr>:$sym_name
  );

  let assemblyFormat = [{ `(` $tile (`,` $lockID^ )? `)` attr-dict }];

  let extraClassDeclaration = [{
    bool hasName() {
      return bool(getOperation()->getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName()));
    }

    mlir::StringAttr name() {
      if (auto attr = getOperation()->getAttrOfType<mlir::StringAttr>(
              mlir::SymbolTable::getSymbolAttrName()))
        return attr;
      emitOpError("does not have '")
          << mlir::SymbolTable::getSymbolAttrName() << "' attribute specified";
      llvm::report_fatal_error("couldn't get name");
    }

    int getLockIDValue() {
      assert(getLockID().has_value() && "Lock has no ID value");
      return getLockID().value();
    }

    int colIndex();
    int rowIndex();
    TileOp getTileOp();
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      if (hasName())
        setNameFn(getResult(), name().str());
      else {
        std::string nameWithoutDialect =
            getOperationName().str().substr(getOperationName().find('.') + 1);
        setNameFn(getResult(), nameWithoutDialect + "_" +
              std::to_string(getTileID().col) + "_" +
              std::to_string(getTileID().row));
      }
    }
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$tile, "int":$lockID, "int":$init), [{
      build($_builder, $_state,
        $_builder.getIndexType(),
        tile,
        $_builder.getI32IntegerAttr(lockID),
        $_builder.getI32IntegerAttr(init),
        nullptr
      );
    }]>
  ];
  let hasVerifier = 1;
}

def AIE_UseLockOp: AIE_Op<"use_lock", []> {
  let summary = "acquire/release lock op";
  let description = [{
    This operation uses a lock.
    AIE1 locks are binary locks while AIE2 locks are counting semaphores
    without an inherent acquire/release characteristic.
    This operation must appear in a parent op where the tile can be determined
    (A CoreOp, a ShimDMAOp, a MemOp, or a MemTileDMAOp).
    The behavior of this operation is dependent on `action`:
    - Acquire: blocks execution until the lock is set to `value`.
    - AcquireGreaterEqual (only AIE2): blocks execution until the lock is set to `value` or greater.
      Then, the value of the lock is decremented by `value`.
    - Release: In AIE1, set the lock to `value`.
      In AIE2, increment the lock by `value`.
  }];

  let arguments = (
    ins Index:$lock,
        LockAction:$action,
        OptionalAttr<AIEI32Attr>:$value,
        OptionalAttr<LockBlocking>:$blocking,
        DefaultValuedOptionalAttr<BoolAttr, "true">:$acq_en
  );

  let assemblyFormat = [{
    `(` $lock `,` $action (`,` $value^)? (`,` $blocking^)? `)` attr-dict
  }];

  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$lock,
                   "xilinx::AIE::LockAction":$action,
                   "int32_t":$value), [{
      build($_builder, $_state, lock, action, $_builder.getI32IntegerAttr(value), nullptr);
    }]>
  ];

  let extraClassDeclaration = [{
    bool acquire() { return (getAction() == LockAction::Acquire); }
    bool acquireGE() { return (getAction() == LockAction::AcquireGreaterEqual); }
    bool release() { return (getAction() == LockAction::Release); }
    int getLockValue() { return getValue().value_or(1); }
    int getTimeout() {
      // LockBlocking is an EnumAttr.
      if (auto val = getBlocking())
        return (int)*val;
      return 1;
    }
    LockOp getLockOp() {
      return llvm::cast<xilinx::AIE::LockOp>(getLock().getDefiningOp());
    }
  }];
}

def AIE_BufferOp: AIE_Op<"buffer", [
    TileElement, IsTileWithMemory
  ]>, Results<(outs AnyMemRef)> {
  let summary = "Declare a buffer";
  let description = [{
    This operation instantiates a buffer that belongs to a Memory Module of a tile.

    Example:
    ```
      %tile33 = aie.tile(3, 3)
      %buf = aie.buffer(%tile33) : memref<256xi64>
    ```
    This operation represents a buffer in tile (3, 3) of 256 elements, each a 64-bit integer.
  }];

  let arguments = (
    ins Index:$tile,
        OptionalAttr<StrAttr>:$sym_name,
        OptionalAttr<AIEI32Attr>:$address,
        OptionalAttr<ElementsAttr>:$initial_value,
        OptionalAttr<AIEI32Attr>:$mem_bank
  );

  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = [{
    `(` $tile `)`
    attr-dict `:` type($buffer)
    custom<BufferInitialValue>(ref(type($buffer)), $initial_value)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool hasName() {
      return bool(getOperation()->getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName()));
    }

    mlir::StringAttr name() {
      if (auto attr = getOperation()->getAttrOfType<mlir::StringAttr>(
              mlir::SymbolTable::getSymbolAttrName()))
        return attr;
      emitOpError("does not have '")
          << mlir::SymbolTable::getSymbolAttrName() << "' attribute specified";
      llvm::report_fatal_error("couldn't get name");
    }

    // Return the number of bytes that need to be allocated for this buffer.
    int64_t getAllocationSize();
    TileOp getTileOp();
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      if (hasName())
        setNameFn(getResult(), name().str());
      else {
        std::string nameWithoutDialect =
            getOperationName().str().substr(getOperationName().find('.') + 1);
        setNameFn(getResult(), nameWithoutDialect + "_" +
              std::to_string(getTileID().col) + "_" +
              std::to_string(getTileID().row));
      }
    }
  }];
}

def AIE_ExternalBufferOp: AIE_Op<"external_buffer", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]>, Results<(outs AnyMemRef)> {
  let summary = "Declare a buffer in external memory";
  let description = [{
    This operation represents a buffer that exists in some physical
    location in a device, most likely external memory. The exact address
    of the external buffer is passed by the mlir_aie_external_set_addr()
    and mlir_aie_external_set_addr_myBuffer_ functions in the associated
    .cpp test file.

    These external buffers are used within the buffer descriptors of a
    shim_dma, i.e., within AIE_DMABdOp operations of a AIE_ShimDMAOp.

    Example:
    ```
      %buf = aie.external_buffer : memref<256xi64>
    ```
    This operation represents an external buffer.
  }];

  let arguments = (
    ins OptionalAttr<StrAttr>:$sym_name,
        OptionalAttr<AIEI64Attr>:$address
  );

  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = [{ attr-dict `:` type($buffer) }];

  let extraClassDeclaration = [{
    bool hasName() {
      return bool(getOperation()->getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName()));
    }

    mlir::StringAttr name() {
      if (auto attr = getOperation()->getAttrOfType<mlir::StringAttr>(
              mlir::SymbolTable::getSymbolAttrName()))
        return attr;
      emitOpError("does not have '")
          << mlir::SymbolTable::getSymbolAttrName() << "' attribute specified";
      llvm::report_fatal_error("couldn't get name");
    }
  }];

  let extraClassDefinition = [{
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      if (hasName())
        setNameFn(getResult(), name().str());
    }
  }];
}

def AIE_EventOp: AIE_Op<"event", []> {
  let summary = "Event instruction";
  let description = [{
    Event instruction.
  }];
  let arguments = (ins ConfinedAttr<AIEI32Attr, [IntMinValue<0>, IntMaxValue<1>]>:$val);
  let results = (outs);
  let assemblyFormat = [{
    `(` $val `)` attr-dict
  }];
}

def AIE_GetStreamOp: AIE_Op<"get_stream", [
    HasParent<"CoreOp">
  ]>, Results<(outs AnyTypeOf<[F32, I32, I<128>]>)> {
  let summary = "An op to read from a stream channel/port of a switchbox";
  let description = [{
    An op to read from a stream channel/port of a switchbox.
  }];

  let arguments = (ins AnyInteger:$channel);
  let results = (outs AnyTypeOf<[F32, I32, I<128>]>:$stream_value);

  let assemblyFormat = [{
    `(` $channel `:` type($channel) `)` attr-dict `:` type($stream_value)
  }];

  let extraClassDeclaration = [{
    bool isWideStream() { return getStreamValue().getType().isInteger(128); }
    bool isFloatStream() {
      return llvm::isa<mlir::FloatType>(getStreamValue().getType());
    }
  }];
}

def AIE_PutStreamOp: AIE_Op<"put_stream", [HasParent<"CoreOp">]> {
  let summary = "An op to write to a stream channel/port of a switchbox";
  let description = [{
    An op to write to a stream channel/port of a switchbox.
  }];

  let arguments = (
    ins AnyInteger:$channel,
        AnyTypeOf<[F32, I32, I<128>]>:$stream_value
  );

  let assemblyFormat = [{
    `(` $channel `:` type($channel) `,` $stream_value `:` type($stream_value) `)` attr-dict
  }];

  let extraClassDeclaration = [{
    bool isWideStream() { return getStreamValue().getType().isInteger(128); }
    bool isFloatStream() {
      return llvm::isa<mlir::FloatType>(getStreamValue().getType());
    }
  }];
}

def AIE_CascadeFlowOp: AIE_Op<"cascade_flow", []> {
  let arguments = (
    ins Index:$source_tile,
        Index:$dest_tile
  );
  let summary = "A cascade connection between tiles";
  let description = [{
    The `aie.cascade_flow` operation represents a cascade connection between two `aie.tile` operations.
    During lowering, this is replaced by `aie.configure_cascade` operations for each `aie.tile` based on
    their relative placement to one another.

    Example:
    ```
      %tile03 = aie.tile(0, 3)
      %tile13 = aie.tile(1, 3)
      aie.cascade_flow(%tile03, %tile13)
    ```
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $source_tile `,` $dest_tile `)` attr-dict
  }];
  let extraClassDeclaration = [{
    TileOp getSourceTileOp();
    TileOp getDestTileOp();
  }];
}

def AIE_ConfigureCascadeOp: AIE_Op<"configure_cascade", [HasParent<"DeviceOp">]> {
  let summary = "An op to configure the input and output directions of the cascade for a single AIE tile";
  let description = [{
    An operation to configure the cascade on a single tile in both the input and the output
    directions.

    Example:
    ```
      %tile00 = aie.tile(1, 3)
      aie.configure_cascade(%tile00, West, East)
    ```
    Configures the input cascade port of %tile00 to the West direction, and the output port to the East direction.
  }];
  let arguments = (
    ins Index:$tile,
        CascadeDir:$inputDir,
        CascadeDir:$outputDir
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{ `(` $tile `,` $inputDir `,` $outputDir `)` attr-dict }];
}

def AIE_GetCascadeOp: AIE_Op<"get_cascade", []>, Results<(outs AnyType:$cascade_value)> {
  let summary = "An op to read from a cascading stream from a neighboring core";
  let description = [{
    An op to read from a cascading stream from a neighboring core.
    The result type of this operation must have a size that matches the cascade size,
    which is architecture-dependent. e.g. AIE1: i384 or vector<8xi48>  AIE2: i512 or vector<16xi32>
  }];
  let hasVerifier = 1;
  let assemblyFormat = [{ `(` `)` attr-dict `:` type($cascade_value) }];
}

def AIE_PutCascadeOp: AIE_Op<"put_cascade", []> {
  let summary = "An op to write to a cascading stream from a neighboring core";
  let description = [{
    An op to write to a cascading stream from a neighboring core.
    The argument type of this operation must have a size that matches the cascade size,
    which is architecture-dependent. e.g. AIE1: i384 or vector<8xi48>  AIE2: i512 or vector<16xi32>
  }];

  let arguments = (ins AnyType:$cascade_value);
  let hasVerifier = 1;
  let assemblyFormat = [{ `(` $cascade_value `:` type($cascade_value) `)` attr-dict }];
}

def AIE_ShimDMAAllocationOp : AIE_Op<"shim_dma_allocation", [HasParent<"DeviceOp">]> {
  let summary = "Runtime allocation information for a single shim DMA";
  let description = [{
    This op exists for cases where shim_dma configuration is performed outside of MLIR-AIE
    and hence there is no appropriate dma_start operation to indicate which channel is being
    used and on which column the shim_dma is.

    It contains attributes for the sym_name of an operation which generated the shim DMA,
    for the DMAChannelDir and channel index, and for the column of the shim tile to which
    the originating operation was mapped.

    Example:
    ```
      %tile00 = aie.tile(0, 0)
      %tile02 = aie.tile(0, 2)
      aie.objectfifo @of_in_0 (%tile00, { %tile02 }, 2) : !aie.objectfifo<memref<64xi16>>
    ```
    could produce the following allocation info (channel direction MM2S, channel index 1, and shim column 0):
    ```
      aie.shim_dma_allocation @of_in_0 (MM2S, 1, 0)
    ```
  }];

  let arguments = (
    ins FlatSymbolRefAttr:$sym_name,
        DMAChannelDir:$channel_dir,
        AIEI64Attr:$channel_index,
        AIEI64Attr:$col,
        // If this is set we are using the PLIO in this ShimTile
        DefaultValuedAttr<BoolAttr, "false">:$plio,
        OptionalAttr<PacketInfoAttr>:$packet
  );

  let results = (outs);

  let assemblyFormat = [{
    $sym_name `(` $channel_dir `,` $channel_index `,` $col (`,` $packet^)? `)` attr-dict
  }];

  let extraClassDeclaration = [{
    static ::xilinx::AIE::ShimDMAAllocationOp getForSymbol(::xilinx::AIE::DeviceOp device, ::llvm::StringRef symbol);
  }];

  let builders = [
    OpBuilder<(ins "mlir::StringRef":$sym_name, "DMAChannelDir":$dir, "int":$channel_index,
               "int":$col, "bool":$plio), [{
      build($_builder, $_state, ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), sym_name), 
            DMAChannelDirAttr::get(odsBuilder.getContext(), dir),
            $_builder.getI64IntegerAttr(channel_index), 
            $_builder.getI64IntegerAttr(col), $_builder.getBoolAttr(plio), nullptr);
    }]>
  ];
}

def AIE_ObjectFifoCreateOp: AIE_Op<"objectfifo", [HasParent<"DeviceOp">, Symbol]> {
  let summary = "Create a circular buffer or channel between two tiles";
  let description = [{
    The `aie.objectFifo` operation creates a circular buffer established between a producer and one or
    more consumers, which are `aie.tile` operations. The `aie.objectFifo` instantiates the given number of
    buffers (of given output type) and their locks in the Memory Module of the appropriate tile(s) after
    lowering, based on tile-adjacency. These elements represent the conceptual depth of the `objectFifo` or,
    more specifically, of its object pool.

    For the producer and for each consumer, a different size (i.e., element number) can be specified as an
    array of integer values. This will take effect in the case of consumers placed on tiles non-adjacent to
    the producer. Otherwise, the producer size will be applied. If a single size is specified, it will be
    applied to both producer and consumers.

    This operation is then converted by the `AIEObjectFifoStatefulTransformPass` into `aie.buffers` and their associated
    `aie.locks`. The pass also establishes Flow and DMA operations between the producer and consumer tiles if they are
    not adjacent.

    1-to-1 tile example:
    ```
      aie.objectfifo @of1 (%tile12, { %tile23 }, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    ```
    This operation creates an `objectFifo` between `%tile12` and `%tile23` of 4 elements, each a buffer of 16 32-bit integers.
    Note: If there are no `ObjectFifoAcquireOps` corresponding to this `objectFifo` on the cores of `%tile12` and `%tile23`,
    then the depths of the object pools on each tile will be 4, as specified. Otherwise, the cores are scanned and the
    highest number of acquired elements (+1 for prefetching) will be used instead, to ensure minimal resource usage.

    1-to-2 tiles broadcast example:
    ```
      aie.objectfifo @of2 (%tile12, { %tile13, %tile23 }, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    ```
    This operation creates an `objectFifo` between `%tile12` and tiles `%tile13`, `%tile23` of 4 elements, each a buffer of x16
    32-bit integers.

    1-to-2 tiles broadcast with explicit sizes example:
    ```
      aie.objectfifo @of3 (%tile12, { %tile13, %tile23 }, [2, 3, 4]) : !aie.objectfifo<memref<16xi32>>
    ```
    This operation creates an `objectFifo` between `%tile12`, `%tile13` and `%tile23`. The depths of the `objectFifo` object pool
    at each tile are respectively 2, 3 and 4 for tiles `%tile12`, `%tile13` and `%tile23`. This overrides the depth analysis
    specified in the first example.

    #### Data Layout Transformations on AIE-ML devices

    On AIE-ML devices, objectFifos can also apply data layout transformations by
    using the DMAs n-dimensional address generation scheme. Two transformations
    can be applied for an objectFifo: one on the producer side, given by a
    `dimensionsToStream` attribute, and one transformation on the consumer side, given by
    a `dimensionsFromStream` attribute. See the `DMABDOp` documentation for a description
    of strides and sizes. The `dimensionsToStream` and `dimensionsFromStream` optional attributes
    are given directly following the producer or consumer tile declaration.
    Different transformations can be specified for each consumer. See example
    below.

    Note that using data layout transformations will cause DMAs to be used even
    between adjacent tiles whose objectFifos would otherwise use shared memory.

    Further note that data layout transforms always apply at a granularity of
    `i32`s, irrespective of the used `memref` data type. This is an
    architectural requirement. Hence, a stride of 4 always expresses 4 `i32`s,
    i.e. 16 bytes.

    The following example shows an objectFifo which transposes a 16x16 matrix of
    `i32`s on the producer side using strides and sizes. No transformation is
    applied on the consumer side of `%tile13` (in this case the `fromStream`
    attribute may also be left off), and a transformation on `%tile23` first gives
    all even indices from the stream, followed by all odd indices:

    ```
      aie.objectfifo @of4 (%tile12 dimensionsToStream [<16, 1>, <16, 16>, <1,1>],
                           {
                            %tile13 dimensionsFromStream [],
                            %tile23 dimensionsFromStream [<2, 1>, <128, 2>]
                           }, 2 : i32
                          ) : !aie.objectfifo<memref<256xi32>>
    ```
  }];

  let arguments = (
    ins SymbolNameAttr:$sym_name,
        Index:$producerTile,
        Variadic<Index>:$consumerTiles,
        AIE_ObjectFifo_Depth:$elemNumber,
        TypeAttrOf<AIE_ObjectFifoType>:$elemType,
        BDDimLayoutArrayAttr:$dimensionsToStream,
        BDDimLayoutArrayArrayAttr:$dimensionsFromStreamPerConsumer,
        DefaultValuedAttr<BoolAttr, "false">:$via_DMA,
        DefaultValuedAttr<BoolAttr, "false">:$plio,
        // disable_synchronization==true will skip lock generation for
        // objectfifo synchronous accesses
        DefaultValuedAttr<BoolAttr, "false">:$disable_synchronization,
        // repeat_count==1 means "do it once"
        OptionalAttr<ConfinedAttr<AIEI32Attr, [IntMinValue<1>]>>:$repeat_count,
        InitValuesArrayAttr:$initValues,
        OptionalAttr<BDPadLayoutArrayAttr>:$padDimensions
  );

  let assemblyFormat = [{
    $sym_name
    `(`
        custom<ObjectFifoProducerTile>($producerTile, $dimensionsToStream) `,`
        `{`
          custom<ObjectFifoConsumerTiles>($consumerTiles, $dimensionsFromStreamPerConsumer)
        `}`
        `,`
        $elemNumber
    `)` attr-dict `:` $elemType
    custom<ObjectFifoInitValues>(ref($elemNumber), ref($elemType), $initValues)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    int size(int index = 0) {
      if (llvm::isa<mlir::ArrayAttr>(getElemNumber()))
        return llvm::dyn_cast<mlir::IntegerAttr>(
                   llvm::dyn_cast<mlir::ArrayAttr>(getElemNumber())[index])
            .getInt();
      else
        return llvm::dyn_cast<mlir::IntegerAttr>(getElemNumber()).getInt();
    }

    TileOp getProducerTileOp();

    mlir::StringAttr name() {
      return getOperation()->getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName());
    }
  }];

  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$sym_name, "mlir::Value":$producerTile,
                   "mlir::ValueRange":$consumerTiles, "mlir::Attribute":$elemNumber, "mlir::Type":$elem_type,
                   CArg<"llvm::ArrayRef<AIE::BDDimLayoutAttr>", "{}">:$dimensionsToStream,
                   CArg<"llvm::ArrayRef<AIE::BDDimLayoutArrayAttr>", "{}">:$dimensionsFromStreamPerConsumer,
                   CArg<"llvm::ArrayRef<AIE::BDPadLayoutArrayAttr>", "{}">:$padDimensions), [{
      odsState.addOperands(producerTile);
      odsState.addOperands(consumerTiles);
      odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
      odsState.addAttribute(getElemNumberAttrName(odsState.name), elemNumber);
      odsState.addAttribute(getElemTypeAttrName(odsState.name), mlir::TypeAttr::get(elem_type));
      odsState.addAttribute(getDimensionsToStreamAttrName(odsState.name),
                            odsBuilder.getAttr<BDDimLayoutArrayAttr>(dimensionsToStream));
      odsState.addAttribute(getDimensionsFromStreamPerConsumerAttrName(odsState.name),
                            odsBuilder.getAttr<BDDimLayoutArrayArrayAttr>(dimensionsFromStreamPerConsumer));
    }]>
  ];
}

def AIE_ObjectFifoAllocateOp: AIE_Op<"objectfifo.allocate", [HasParent<"DeviceOp">]> {
  let summary = "Sets a tile's memory module as the target for an objectfifo's memory allocation";
  let description = [{
    Specify a tile's memory module to be used as the allocation target for this objectfifo.
    Only works if all the objectfifo's tiles, producer and consumers, have shared memory access to
    the specified tile's memory module.

    In the following example the objects of @of4 will be allocated on %tile14:
    ```
      aie.objectfifo @of4 (%tile13, {%tile14}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
      aie.objectfifo.allocate @of4 (%tile14)
    ```
  }];

  let arguments = (
    ins FlatSymbolRefAttr:$objFifo_name,
        Index:$delegateTile
  );

  let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{
    $objFifo_name `(` $delegateTile `)` attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ObjectFifoCreateOp getObjectFifo();
    TileOp getDelegateTileOp();
  }];
}

def AIE_ObjectFifoLinkOp: AIE_Op<"objectfifo.link", [HasParent<"DeviceOp">]> {
  let summary = "Links two objectFifos through an intermediary tile's DMA";
  let description = [{
    The `aie.objectfifo.link` operation marks two or more `objectFifos` as linked. This implies that the `objectFifos` form
    one dataflow movement which is split accross multiple `objectFifos`. Specifically, during the `objectFifo` lowering there will
    be less memory elements generated at the link point (i.e., the shared tile of all `objectFifos` in the link) as the `objectFifos` can share.

    The `objectFifos` which are linked must have a link point (i.e., a shared AIE tile).

    Example:
    ```
      aie.objectfifo @of1 (%t70, { %t72 }, 2) : !aie.objectfifo<memref<64xi16>>
      aie.objectfifo @of2 (%t72, { %t74 }, 2) : !aie.objectfifo<memref<64xi16>>
      aie.objectfifo.link [@of1] -> [@of2] ([] [])
    ```
    This operation links two `objectFifos` which have tile `%t72` as a link point. The offset input arrays are not required as the full size of
    the objects are transferred from @of1 to @of2.

    To achieve a broadcast pattern through the link tile, the output `objectFifo` should have a list of all the consumers tiles.
    To achieve a distribute pattern from the link tile, there should be multiple output `objectFifos` in the OjbectFifoLinkOp. In this case,
    parts will be taken out of the input `objectFifo`'s buffers based on dst_offsets input array.
    The join pattern is the exact inverse of the distribute one and uses the src_offsets input array instead.
  }];

  let arguments = (
    ins SymbolRefArrayAttr:$fifoIns,
        SymbolRefArrayAttr:$fifoOuts,
        I64ArrayAttr:$src_offsets,
        I64ArrayAttr:$dst_offsets
  );

  let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{
    $fifoIns `->` $fifoOuts `(` $src_offsets $dst_offsets `)` attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    std::vector<ObjectFifoCreateOp> getInputObjectFifos();
    std::vector<ObjectFifoCreateOp> getOutputObjectFifos();
    bool isJoin() { return getFifoIns().size() > 1; }
    bool isDistribute() { return getFifoOuts().size() > 1; }
    std::optional<mlir::Value> getOptionalSharedTile();
    std::optional<int> getRepeatCount();
    std::vector<int> getJoinTransferLengths();
    std::vector<int> getDistributeTransferLengths();
  }];
}

def AIE_ObjectFifoRegisterExternalBuffersOp: AIE_Op<"objectfifo.register_external_buffers", [
    HasParent<"DeviceOp">, TileElement, IsShimNOCTile
  ]> {
  let summary = "Registers external buffers to an objectFifo's shim tile(s) for use in the associated shim DMA(s)";
  let description = [{
    The `aie.objectfifo.register_external_buffers` operation is used to register one or multiple external buffers
    to the shim tile(s) used in an `objectFifo` creation. During the `objectFifo` lowering pass, shim DMAs that are
    generated for those shim tiles will use the registered external buffers. This is currently done because
    external buffers typically have a different size than the AIE buffers which are used in the AIE tiles of the
    same `objectFifos`.

    Example:
    ```
      aie.objectfifo @of1 (%t70, %t73, 2) : !aie.objectfifo<memref<64xi16>>
      %buffer_in_0  = aie.external_buffer : memref<512 x i16>
      %buffer_in_1  = aie.external_buffer : memref<512 x i16>
      aie.objectfifo.register_external_buffers @of1 (%t70, {buffer_in_0, buffer_in_1}) : (memref<512 x i16>, memref<512 x i16>)
    ```
    This operation registers external buffers `%buffer_in_0` and `%buffer_in_1` to use in the shim_dma of shimTile `%t70`.
  }];

  let arguments = (
    ins FlatSymbolRefAttr:$objFifo_name,
        Index:$tile,
        Variadic<AnyMemRef>:$externalBuffers
  );

  let assemblyFormat = [{
    attr-dict $objFifo_name `(` $tile `,` `{` $externalBuffers `}` `)` `:` `(` type($externalBuffers) `)`
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    TileOp getTileOp();
    ObjectFifoCreateOp getObjectFifo();
    // No results so just use default impl.
    using ::mlir::OpAsmOpInterface::Trait<ObjectFifoRegisterExternalBuffersOp>::getAsmResultNames;
  }];
}

def AIE_ObjectFifoAcquireOp: AIE_Op<"objectfifo.acquire", []> {
  let summary = "Acquire operation to lock and return objects of an ObjectFifo";
  let description = [{
    The `aie.objectFifo.acquire` operation first acquires the locks of the next given number
    of objects in the `objectFifo`. The mode it acquires the locks in is chosen based on the port
    (producer: acquire for write, consumer: acquire for read). Then, it returns a subview of
    the acquired objects which can be used to access them.

    This operation is then converted by the `AIEObjectFifoStatefulTransformPass` into `aie.use_lock` operations on
    the locks of the `objectFifo` objects that will be acquired. Under the hood, the operation only performs
    new acquires if necessary. For example, if two objects have been acquired in the past and none have yet
    to be released by the same process, then performing another acquire operation on the same `objectFifo`
    within the same process of size two or less will not result in any new use_lock operations (and for size
    greater than two, only (size - 2) use_lock operations will be performed).

    Example:
    ```
      %subview = aie.objectfifo.acquire @of1 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
    ```
    This operation acquires the locks of the next two objects in the `objectFifo` named `@of1` from its consumer
    port and returns a subview of the acquired objects.
  }];

  let arguments = (
    ins ObjectFifoPort:$port,
        FlatSymbolRefAttr:$objFifo_name,
        ConfinedAttr<AIEI32Attr, [IntMinValue<1>]>:$size
  );

  let results = (outs AIE_ObjectFifoSubviewType:$subview);

  let assemblyFormat = [{
    attr-dict $objFifo_name `(` $port `,` $size `)` `:` qualified(type($subview))
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ObjectFifoCreateOp getObjectFifo();
    int acqNumber() { return getSize(); }
  }];
}

def AIE_ObjectFifoReleaseOp: AIE_Op<"objectfifo.release", []> {
  let summary = "Release operation for object locks in an ObjectFifo";
  let description = [{
    The `aie.objectFifo.release` operation releases the locks of the given number of objects
    in the `objectFifo`. The mode it releases the locks in is chosen based on the `port`
    (producer: release for read, consumer: release for write).

    This operation is then converted by the `AIEObjectFifoStatefulTransformPass` into `aie.use_lock` operations.

    Example:
    ```
      aie.objectfifo.release @of1 (Produce, 1)
    ```
    This operation releases the lock of the next object in the `objectFifo` named `@of1` from producer port.
  }];

  let arguments = (
    ins ObjectFifoPort:$port,
        FlatSymbolRefAttr:$objFifo_name,
        ConfinedAttr<AIEI32Attr, [IntMinValue<1>]>:$size
  );

  let assemblyFormat = [{
    attr-dict $objFifo_name `(` $port `,` $size `)`
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ObjectFifoCreateOp getObjectFifo();
    int relNumber() { return getSize(); }
  }];
}

def AIE_ObjectFifoSubviewAccessOp : AIE_Op<"objectfifo.subview.access", []> {
  let summary = "ObjectFifoSubview type accessor method";
  let description = [{
    Access the Nth element of a value of `ObjectFifoSubview` type.

    Example:
    ```
      %subview = aie.objectfifo.acquire @of1 (Produce, 3) : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
    ```
    In this example, `%elem` is the first object of the subview. Note that this may not correspond to the first element of
    the `objectFifo` if other acquire operations took place beforehand.

  }];

  let arguments = (
    ins AIE_ObjectFifoSubviewType:$subview,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$index
  );

  let hasVerifier = 1;

  let results = (outs AnyMemRef:$output);

  let assemblyFormat = [{
    $subview `[` $index `]` attr-dict `:` qualified(type($subview)) `->` qualified(type($output))
  }];
}

def AIE_ObjectFifoRegisterProcessOp: AIE_Op<"objectfifo.register_process", []> {
  let summary = "Operation that produces the acquire/release patterns for a process registered to an objectFifo";
  let description = [{
    The `aie.registerProcess` operation allows the user to register a function to an `objectFifo` along with its
    acquire and release patterns. These patterns will be used to generate a sequence of acquires and releases
    on the `objectFifo` elements. This generated sequence is often in the form of a for loop, however, in the case
    of cyclo-static patterns only the repetition of same number accesses and releases will generate a for loop.
    This may result in multiple for loops of different sizes being generated. If there is no repetition, then no
    loops will be generated.

    Example:
    ```
      aie.objectfifo @of1 (%t72, %t73, 2) : !aie.objectfifo<memref<16xi32>>
      %length = arith.constant 10 : index
      %acquirePatternProducer = arith.constant dense<[1, 2, 2, 0]> : tensor<4xi32>
      %releasePatternProducer = arith.constant dense<[0, 1, 1, 2]> : tensor<4xi32>
      func @producer_work(%input : !aie.objectfifosubview<memref<16xi32>>) -> () { ... }

      aie.objectfifo.register_process @of1 (Produce, %acquirePatternProducer : tensor<4xi32>, %releasePatternProducer : tensor<4xi32>, @producer_work, %length)
    ```
    This operation registers function @producer_work and associated patterns to the produce end of @of1.
    @producer_work will be called with the subviews produced when acquiring elements from @of1 following the acquire pattern.

    If the input patterns are static (only one element) then the length of the produced for loop will be that of the input `%length`.
    If the input patterns are cyclo-static then they must be of the same size.
  }];

  let arguments = (
    ins ObjectFifoPort:$port,
        FlatSymbolRefAttr:$objFifo_name,
        I32Tensor:$acquirePatternTensor,
        I32Tensor:$releasePatternTensor,
        FlatSymbolRefAttr:$callee,
        Index:$length
  );

  let assemblyFormat = [{
    attr-dict $objFifo_name `(`
        $port `,`
        $acquirePatternTensor `:` type($acquirePatternTensor) `,`
        $releasePatternTensor `:` type($releasePatternTensor) `,`
        $callee `,` $length
    `)`
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ObjectFifoCreateOp getObjectFifo();

    mlir::DenseIntElementsAttr getAcquirePattern() {
      return llvm::cast<mlir::DenseIntElementsAttr>(getAcquirePatternTensor()
          .getDefiningOp<mlir::arith::ConstantOp>()
          .getValue());
    }

    mlir::DenseIntElementsAttr getReleasePattern() {
      return llvm::cast<mlir::DenseIntElementsAttr>(getReleasePatternTensor()
          .getDefiningOp<mlir::arith::ConstantOp>()
          .getValue());
    }

    int getProcessLength() {
      return llvm::cast<mlir::IntegerAttr>(getLength()
          .getDefiningOp<mlir::arith::ConstantOp>()
          .getValue())
          .getInt();
    }
  }];
}

def AIE_BDChainOp: AIE_Op<"bd_chain", [Symbol, SkipAccessibilityCheckTrait]> {
  let summary = "Definition of a Parametrizable Chain of Buffer Descriptors";
  let description = [{
    This operation allows you to define buffer descriptor chains with parametrizable inputs.
    This is useful for common patterns such as double buffering (ping-pong) that may look identical but use different input/output buffers and locks.
    Currently, only buffers and locks are parametrizable.

    Once defined, an abstract BD chain can be used elsewhere using AIEX ops in the runtime sequence.
    In the future, abstract BD chains will also be usable elsewhere, inside the static configuration.
    At its usage sites, the abstract BD chain will be concretized with the given input arguments.
  }];

  let arguments = (
    ins SymbolNameAttr:$sym_name
  );
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
  }];
  let hasCustomAssemblyFormat = 1;

}

#endif // AIE_OPS
