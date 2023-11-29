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

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class AIE_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIE_Dialect, mnemonic, traits>;


def AIE_DeviceOp: AIE_Op<"device", [
    AIETarget, HasParent<"mlir::ModuleOp">,
    SymbolTable, SingleBlock, NoTerminator, IsolatedFromAbove
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

  let arguments = (ins AIEDevice:$device);
  let regions = (region AnyRegion:$bodyRegion);
  let assemblyFormat = [{
    `(` $device `)` regions attr-dict
  }];
  let extraClassDeclaration = [{
    const xilinx::AIE::AIETargetModel &getTargetModel();
  }];
  let hasVerifier = 1;
}

def AIE_TileOp: AIE_Op<"tile", [
    FlowEndPoint,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index:$result)> {
  let arguments = (
    ins ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$col,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$row
  );

  let summary = "Declare an AIE tile";
  let description = [{
    This operation creates an AIE tile in the AIE array. We specify what the column and the row of the tile.

    A tile encompasses core module (CoreOp), memory module (MemOp), stream switch (SwitchboxOp),
    memory buffer (BufferOp), and lock (LockOp).

    A tile is a logical abstraction. We use a tile to establish an ownership of a hardware entity
    to it.
    Note that row 0 of the Tile array is different from other rows, since it models the shim interface between
    the AIE array proper and the PL.  The South-West/Lower Right most core exists in Tile(0,1)
  }];

  let extraClassDeclaration = [{
    int getNumSourceConnections(WireBundle bundle);
    int getNumDestConnections(WireBundle bundle);
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
  }];

  let assemblyFormat = [{
    `(` $col `,` $row `)` attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDefinition = [{
    void $cppClass::getAsmResultNames(
        function_ref<void(::mlir::Value, ::llvm::StringRef)> setNameFn) {
       std::string nameWithoutDialect =
             getOperationName().str().substr(getOperationName().find('.') + 1);
       setNameFn(getResult(), nameWithoutDialect + "_" +
                                  std::to_string(getCol()) + "_" +
                                  std::to_string(getRow()));
    }
  }];
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
    int getNumSourceConnections(WireBundle bundle);
    int getNumDestConnections(WireBundle bundle);
    int colIndex();
    int rowIndex();
    TileOp getTileOp();
    using ::xilinx::AIE::TileElement::Trait<SwitchboxOp>::getAsmResultNames;
  }];
}

def AIE_ShimSwitchboxOp: AIE_Op<"shimswitchbox", [
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
    6 Ports to South(DMA, NoC I/F, PL I/F) 8 Ports from South (DMA, NoC I/F, PL I/F)
    2 Ports to FIFOs 2 Ports from FIFOs
    1 Port for control packet for Shim register access
    1 Port for response to access for Shim registers
    1 Port for trace packet from Shim

  }];
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $col `)` regions attr-dict }];
  let hasVerifier = 1;
}

def AIE_ShimMuxOp: AIE_Op<"shimmux", [
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
    Like the `AIE.switchbox` operation, `AIE.shimMux` is configured
    by code in its region, but can only contain connect operations

    Example:
    ```
    %tile = aie.tile(1, 1)
    aie.shimmux(%tile) {
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
    int getNumSourceConnections(WireBundle bundle);
    int getNumDestConnections(WireBundle bundle);
    TileOp getTileOp();
    using ::xilinx::AIE::TileElement::Trait<ShimMuxOp>::getAsmResultNames;
  }];
}

def AIE_ShimDMAOp: AIE_Op<"shimDMA", [
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
      %buf = AIE.external_buffer : memref<256xi64>
      %lock1 = AIE.lock(%t70, 1)

      %dma = AIE.shimDMA(%t70) {
          AIE.dmaStart(MM2S, 0, ^bd0, ^end)
        ^bd0:
          AIE.useLock(%lock1, Acquire, 1)
          AIE.dmaBd(<%buf : memref<512 x i16>, 0, 512>, 0)
          AIE.useLock(%lock1, Release, 0)
          AIE.nextBd ^bd0
        ^end:
          AIE.end
      }
    ```
    Create the shimDMA for tile %t70 and setup one DMA channel and one Buffer Descriptor.
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
    DefaultValuedAttr<AIEI32Attr, "0x400">:$stackSize,
    OptionalAttr<StrAttr>:$link_with
  );
  let summary = "Declare a core module";
  let description = [{
    This operation represents an AIEngine processor core belonging to a tile.
    The region of a CoreOp contains code that gets run on the AIE core.  This code will
    typically be outlined into the LLVM dialect, eventually resulting in a binary file
    for each core.  The name of this file can be be specified using the `elf_file`
    attribute.

    This op has an optional `stackSize` attribute, to control the amount of memory (in bytes)
    reserved for the stack.  The default value is 1024.  The stack (and other data allocations)
    are always stored in the local core memory, to avoid conflicts with static data allocations
    in other cores.

    Examples:
    ```
    %tile = aie.tile(1, 1)
    %lock11_8 = AIE.lock(%tile, 8)
    aie.core(%tile) {
      AIE.useLock(%lock11_8, "Acquire", 1)
      AIE.useLock(%lock11_8, "Release", 0)
      AIE.end
    }
    ```
    ```
    %tile = AIE.tile(3, 3)
    AIE.core(%tile) {
      AIE.end
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
    ins WireBundle:$sourceBundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$sourceChannel,
        WireBundle:$destBundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$destChannel
  );
  let summary = "A circuit-switched connection inside a switchbox";
  let description = [{
    This operation represents a programmed circuit-switched connection in a stream switch.
    It associates a source bundle and source channel with a destination bundle and a destination channel.
    This operation must exist within an `aie.switchbox` or `aie.shimswitchbox` operation.
    All of the `aie.connect` operations in a switchbox must have a different destinations.
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
    `<` $sourceBundle `:` $sourceChannel `,` $destBundle `:` $destChannel `>` attr-dict
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
        WireBundle:$sourceBundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$sourceChannel,
        Index:$dest,
        WireBundle:$destBundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$destChannel
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
    `(` $source `,` $sourceBundle `:` $sourceChannel `,` $dest `,` $destBundle `:` $destChannel `)` attr-dict
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
        %a0_0 = AIE.amsel<5>(3)
        %m1 = AIE.masterset("East" : 0, %a0_0 )
        AIE.packetrules("South" : 0) {
          AIE.rule(0x1F, 0x10, %a0_0)
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
    ins WireBundle:$destBundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$destChannel,
        Variadic<Index>:$amsels
  );
  let summary = "Packet switched input connection";
  let description = [{
    A Packet switched connection inside a switchbox.
    This operation specifies the configuration for a master port.

    Example:
      %a0_m2 = AIE.amsel<0>(2)
      AIE.masterset("Core" : 0, %a0_m2)

    The code will configure the master port <"Core" : 0> to use arbiter 0 with msel 2
    (see AMSelOp for more details regarding AMSel)

    In the current architecture, a master port can only be associated with one arbiter. However,
    a master port can be activated by different msels from one arbiter

    Example:
      %a1_0 = AIE.amsel<1>(0)
      %a1_1 = AIE.amsel<1>(1)
      %a2_3 = AIE.amsel<2>(3)

      AIE.masterset("West" : 2, %a1_0, %a2_3) // this is illegal, please don't do this
      AIE.masterset("West" : 3, %a1_0, %a1_1) // this is OK
  }];

  let assemblyFormat = [{
    `(` $destBundle `:` $destChannel `,` $amsels `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int destIndex() { return getDestChannel(); }
    Port destPort() { return {getDestBundle(), destIndex()}; }
  }];
}

def AIE_WireOp: AIE_Op<"wire", []> {
  let arguments = (
    ins Index:$source,
        WireBundle:$sourceBundle,
        Index:$dest,
        WireBundle:$destBundle
  );
  let summary = "A bundle of physical wires between components";
  let description = [{
    The `aie.wire` operation represents a physical set of connections between components in a Versal device.
    Typically, these components are switches, represented by an `aie.switchbox` operation, and tiles,
    represented by an [aie.tile](#aietile-aietileop) operation.
  }];

  let assemblyFormat = [{
    `(` $source `:` $sourceBundle `,` $dest `:` $destBundle `)` attr-dict
  }];
}

def AIE_PacketRulesOp: AIE_Op<"packetrules", [SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins WireBundle:$sourceBundle,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$sourceChannel
  );
  let regions = (region AnyRegion:$rules);
  let summary = "Packet switched routing rules";
  let description = [{
    This operation defines packet-switched routing configuration for packets entering a switchbox.
    It references a port of the containing swithcbox, which be unique among other packetRules
    operations and [AIE.connect]($aieconnect-aieconnectop) operations in the containing switchbox.
    It contains a region of up to 4 [AIE.rule](#aierule-aiepacketruleop) operations.

    See [AIE.rule](#aierule-aiepacketruleop) for an example.
  }];

  let assemblyFormat = [{ `(` $sourceBundle `:` $sourceChannel `)` regions attr-dict }];
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
    matching port of the containing [AIE.packetRules](#aiepacketrules-aiepacketrulesop).
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
      %a4_1 = AIE.amsel<4>(1)
      %a3_2 = AIE.amsel<3>(2)

      AIE.packetRules("Core" : 0) {
        AIE.rule(0x1F, 0x2, %a4_1)
        AIE.rule(0x1B, 0x1, %a3_2)
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
  let arguments = (ins AIEI8Attr:$ID);
  let regions = (region AnyRegion:$ports);
  let summary = "Packet switched flow";
  let description = [{
    A logical packet-switched flow between tiles.  During place and
    route, this is replaced by MasterSets and PacketRules inside
    switchboxes.

    Example:
    ```
      %01 = AIE.tile(0, 1)
      AIE.packet_flow(0x10) {
        AIE.packet_source<%01, "Core" : 0>
        AIE.packet_dest<%01, "Core" : 0>
      }
    ```
  }];

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
    A object representing the destination of a packet-switched flow. This must exist
    within an [AIE.packet_flow](#aiepacketflow-aiepacketflowop) operation.

    See [AIE.packet_flow](#aiepacketflow-aiepacketflowop) for an example.
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
    within an [AIE.packet_flow](#aiepacketflow-aiepacketflowop) operation. The destination
    Must be unique within a design.

    See [AIE.packet_flow](#aiepacketflow-aiepacketflowop) for an example.
  }];

  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];

  let extraClassDeclaration = [{
    int channelIndex() { return getChannel(); }
    Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_DMABDPACKETOp: AIE_Op<"dmaBdPacket", []> {
  let summary = "Enable packet headers for a dma block descriptor";
  let description = [{
    This operation enables packet headers for a block descriptor for DMA operations. In particular, it specifies
    the packet type (3-bits) and packet ID (5-bits).
    
    This operation must be used in an MLIR block that lives inside a MemOp's region, and before AIE.dmaBd.
    The block descriptor specifies what lock to use and the buffer configuration.

    Example:
    ```
      // this defines a BD that uses lock %lck0 and buffer %buf0
      ^bd5:
        AIE.useLock(%lck, "Acquire", 0)
        AIE.dmaBdPacket(0x4, 0xD)
        AIE.dmaBd(<$buf0 : memref<512xi32>, 0, 512>, 1)
        AIE.useLock(%lck, "Release", 1)
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

def AIE_DMABDOp: AIE_Op<"dmaBd", []> {
  let summary = "Declare a dma block descriptor op";
  let description = [{
    This operation describes a block descriptor for DMA operations. In particular, it specifies
    what buffer addresss to use, the transfer length, and the buffer type (A or B).
    
    This operation must be used in an MLIR block that lives inside a MemOp's region.
    The block descriptor specifies what lock to use and the buffer configuration.

    Example:
    ```
      // this defines a BD that uses lock %lck0 and buffer %buf0
      ^bd5:
        AIE.useLock(%lck, "Acquire", 0)
        AIE.dmaBd(<$buf0 : memref<512xi32>, 0, 512>, 1)
        AIE.useLock(%lck, "Release", 1)
        br ^bd6 // point to the next Block, which is also a different Block Descriptor

      ...

      // this defines a BD that does not use any lock
      ^bd8:
        AIE.dmaBd(<$buf1 : memref<64xi32>, 0, 64>, 0)
    ```
    A DMA channel in a Memory Module can process one block descriptor after another by chaining them.
    There are 16 block descriptors per Memory Module. They are shared by four DMA channels.

    ## DMA Data Layout Transformations on AIE-ML Devices

    AIE-ML devices can apply data layout transformations at the buffer
    descriptor level. These transformation are described by strides (also called
    stepsizes) and wraps (also called lengths) in up to three dimensions (four
    dimensions on memtiles). Strides and wraps can be supplied to the `dmaBdOp`
    through an optional argument, an array of tuples `<wrap, stride>`.

    The first element of this array gives the _highest-dimension_ stride and
    wrap, the last element of the array gives the lowest-dimension.

    Strides are always expressed in units of `i32`s; this is an architectural
    requirement, as data is moved by the DMA at this fundamental size. 

    We can model the access pattern strides and wraps generate by a series of
    nested loops. In general, a set of strides and wraps like this...

    ```
    [<wrap_2, stride_2>, <wrap_1, stride_1>, <wrap_0, stride_0>]
    ```

    ...translates to an access pattern that can be epxressed like this:

    ```
    int *buffer;  // i32
    for(int i = 0; i < wrap_2; i++)
      for(int j = 0; j < wrap_1; j++)
        for(int k = 0; k < wrap_0; k++)
          // access/store element at/to buffer[  i * stride_2 
          //                                   + j * stride_1 
          //                                   + k * stride_0]
    ```

    The following example shows an access pattern that corresponds to 
    alternating between even and odd elements of the buffer/stream every 8 
    elements: 

    ```
    AIE.dmaBd(<%buf : memref<128xi32>, 0, 128>, 0, [<8, 16>, <2, 1>, <8, 2>])
    ```
  }];

  let arguments = (
    ins AnyMemRef:$buffer,
        AIEI32Attr:$offset,
        AIEI32Attr:$len,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>, IntMaxValue<1>]>:$AB, // 0: A, 1: B
        OptionalAttr<AIE_DimTupleArrayAttr>:$dimensions
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` `<` $buffer  `:` type($buffer) `,` $offset `,` $len `>` `,` $AB (`,` $dimensions^ )? `)` attr-dict
  }];

  let extraClassDeclaration = [{
    BufferOp getBufferOp();
    int getOffsetValue() { return getOffset(); }
    int getLenValue() { return getLen(); }
    bool isA() { return (getAB() == 0); }
    bool isB() { return (getAB() == 1); }
  }];

  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "int":$AB), [{
      odsState.addOperands(buffer);
      odsState.addAttribute(getOffsetAttrName(odsState.name), $_builder.getI32IntegerAttr(offset));
      odsState.addAttribute(getLenAttrName(odsState.name), $_builder.getI32IntegerAttr(len));
      odsState.addAttribute(getABAttrName(odsState.name), $_builder.getI32IntegerAttr(AB));
      }]>
  ];
}

def AIE_DMAStartOp: AIE_Op<"dmaStart", [
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
        AIE.dmaStart("MM2S", 0, ^bd0, ^end)
      ^bd0:
        AIE.useLock(%lock0, "Acquire", 0)
        AIE.dmaBd(<%buffer : memref<16 x f32>, 0, 16>, 0)
        AIE.useLock(%lock0, "Release", 1)
        br ^bd0
      ^end:
        AIE.end
    ```

    Conceptually, the AIE.dmaStart operation is a terminator that either passes
    control to a basic block containing DMA operations (through its first successor)
    or to a basic block for another dmaStart, to an AIE.end operation.
  }];

  let arguments = (
    ins DMAChannelDir:$channelDir,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$channelIndex
  );
  let successors = (successor AnySuccessor:$dest, AnySuccessor:$chain);
  let assemblyFormat = [{
    `(` $channelDir `,` $channelIndex `,` $dest `,` $chain `)` attr-dict
  }];

  let extraClassDeclaration = [{
    bool isSend() { return (static_cast<int32_t>(getChannelDir()) == 1); }
    bool isRecv() { return (static_cast<int32_t>(getChannelDir()) == 0); }
  }];
}

// MemOps are not actually Callable, but we want to inline code into them, so we have to 
// implement CallableOpInterface
def AIE_MemOp: AIE_Op<"mem", [
    TileElement, FlowEndPoint, CallableOpInterface,
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
      m73 = AIE.mem(%t73) {
          %srcDma = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
        ^bd0:
          AIE.useLock(%lock, "Acquire", 0)
          AIE.dmaBd(<%buf : memref<64xi16>, 0, 64>, 0)
          AIE.useLock(%lock, "Release", 1)
          AIE.nextBd ^bd0
        ^end:
          AIE.end
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
    int maxSizeInBytes() { return 32768; }
    // CallableOpInterface
    mlir::Region *getCallableRegion();
    llvm::ArrayRef<mlir::Type> getArgumentTypes() { return getOperand().getType(); }
    llvm::ArrayRef<mlir::Type> getResultTypes() { return getType(); }
    using ::xilinx::AIE::TileElement::Trait<MemOp>::getAsmResultNames;
  }];
}

// This op is not actually Callable, but we want to inline code into them, so we have to
// implement CallableOpInterface
def AIE_MemTileDMAOp: AIE_Op<"memTileDMA", [
    TileElement, FlowEndPoint, CallableOpInterface,
    IsMemTile, HasValidBDs, HasValidDMAChannels,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let summary = "Declare a memTileDMA op";
  let description = [{
    This operation describes a DMA inside an AIE2 MemTile.
    The region of the op is used to setup the DMAs and Block Descriptors.
    See DMAStartOp and DMABdOp for more concrete examples on DMAs and Block Descriptors.

    This operation is restricted to certain compatible tiles in AIE2 devices:
    xcve2302: row 1
    xcve2802: row 1 and 2

    Example:
    ```
      m73 = AIE.memTileDMA(%t71) {
          %srcDma = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
        ^bd0:
          AIE.useLock(%lock, "Acquire", 0)
          AIE.dmaBd(<%buf : memref<64xi16>, 0, 64>, 0)
          AIE.useLock(%lock, "Release", 1)
          AIE.nextBd ^bd0
        ^end:
          AIE.end
      }
    ```
    Create a description for tile %t73 and setup one DMA channel and one Buffer Descriptor.
  }];

  let arguments = (ins Index:$tile);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    TileOp getTileOp();
    // CallableOpInterface
    mlir::Region *getCallableRegion();
    llvm::ArrayRef<mlir::Type> getArgumentTypes() { return getOperand().getType(); }
    llvm::ArrayRef<mlir::Type> getResultTypes() { return getType(); }
    using ::xilinx::AIE::TileElement::Trait<MemTileDMAOp>::getAsmResultNames;
  }];
}

def AIE_NextBDOp: AIE_Op<"nextBd", [
    Terminator, ParentOneOf<["MemOp", "MemTileDMAOp", "mlir::func::FuncOp", "ShimDMAOp"]>
  ]> {
  let summary = "The next buffer descriptor";
  let description = [{
    This operation terminates the basic block describing a buffer descriptor inside
    a tile or shim DMA operation.  It references a single following buffer descriptor.
    Note that unlike other terminators (like cf.br), canonicalization should not remove
    the `nextBd` terminator, since it would result in invalid buffer descriptors.

    Example:
    ```
      m73 = AIE.mem(%t73) {
          %srcDma = AIE.dmaStart("S2MM", 0, ^bd0, ^end)
        ^bd0:
          AIE.useLock(%lock, "Acquire", 0)
          AIE.dmaBd(<%buf : memref<64xi16>, 0, 64>, 0)
          AIE.useLock(%lock, "Release", 1)
          AIE.nextBd ^bd0
        ^end:
          AIE.end
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
      %tile33 = AIE.tile(3, 3)
      %lck = AIE.lock(%tile33, 7)
    ```
    This operation represents a lock that lives in the Memory module of Tile(3, 3) with a lockID of 7

    Case when LockID is not assigned:
      Before AIEAssignLockIDs: %tile33 = AIE.tile(3)
      After AIEAssignLockIDs: %tile33 = AIE.tile(3, $assigned_value)
  }];

  let arguments = (
    ins Index:$tile,
        OptionalAttr<ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>>:$lockID,
        OptionalAttr<AIEI32Attr>:$init
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
    using ::xilinx::AIE::TileElement::Trait<LockOp>::getAsmResultNames;
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$tile, "int":$lockID, "int":$init), [{
      build($_builder, $_state,
        $_builder.getIndexType(),
        tile,
        $_builder.getI32IntegerAttr(lockID),
        $_builder.getI32IntegerAttr(init));
    }]>
  ];
  let hasVerifier = 1;
}

def AIE_UseLockOp: AIE_Op<"useLock", []> {
  let summary = "acquire/release lock op";
  let description = [{
    This operation uses a lock. In AIE1, a lock can be acquired with a value,
    or released with a value. This should be understood as a "blocking"
    operation. In AIE2, locks are counting semaphores without inherent
    acquired/release characteristic. This lock must appear in a parent op where
    the tile can be determined (A CoreOp, a ShimDMAOp, a MemOp, or a
    MemTileDMAOp).
  }];

  let arguments = (
    ins Index:$lock,
        AIEI32Attr:$value,
        LockAction:$action,
        OptionalAttr<LockBlocking>:$blocking
  );

  let assemblyFormat = [{
    `(` $lock `,` $action `,` $value ( `,` $blocking^ )? `)` attr-dict
  }];

  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$lock,
                   "int32_t":$value,
                   "xilinx::AIE::LockAction":$action), [{
      build($_builder, $_state, lock, value, action, nullptr);
    }]>
  ];

  let extraClassDeclaration = [{
    bool acquire() { return (getAction() == LockAction::Acquire); }
    bool acquireGE() { return (getAction() == LockAction::AcquireGreaterEqual); }
    bool release() { return (getAction() == LockAction::Release); }
    int getLockValue() { return getValue(); }
    int getTimeout() {
      // LockBlocking is an EnumAttr.
      if (auto val = getBlocking())
        return (int)*val;
      return 1;
    }
    LockOp getLockOp() {
      return llvm::dyn_cast<xilinx::AIE::LockOp>(getLock().getDefiningOp());
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
      %tile33 = AIE.tile(3, 3)
      %buf = AIE.buffer(%tile33) : memref<256xi64>
    ```
    This operation represents a buffer in tile (3, 3) of 256 elements, each a 64-bit integer.
  }];

  let arguments = (
    ins Index:$tile,
    OptionalAttr<StrAttr>:$sym_name
  );

  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = [{ `(` $tile `)` attr-dict `:` type($buffer) }];
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

    // Return the address of this buffer
    int64_t address() {
      if (auto attr = getOperation()->getAttrOfType<mlir::IntegerAttr>("address"))
        return attr.getInt();
      emitOpError("does not have 'address' attribute specified");
      llvm::report_fatal_error("couldn't address");
    }

    // Return the number of bytes that need to be allocated for this buffer.
    int64_t getAllocationSize();
    TileOp getTileOp();
    using ::xilinx::AIE::TileElement::Trait<BufferOp>::getAsmResultNames;
  }];
}

def AIE_ExternalBufferOp: AIE_Op<"external_buffer", []>, Results<(outs AnyMemRef)> {
  let summary = "Declare a buffer in external memory";
  let description = [{
    This operation represents a buffer that exists in some physical
    location in a device, most likely external memory. The exact address
    of the external buffer is passed by the mlir_aie_external_set_addr()
    and mlir_aie_external_set_addr_myBuffer_ functions in the associated 
    .cpp test file. 

    These external buffers are used within the buffer descriptors of a 
    shimDMA, i.e., within AIE_DMABdOp operations of a AIE_ShimDMAOp.

    Example:
    ```
      %buf = AIE.external_buffer : memref<256xi64>
    ```
    This operation represents an external buffer.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$sym_name);

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

def AIE_GetStreamOp: AIE_Op<"getStream", [
    HasParent<"CoreOp">
  ]>, Results<(outs AnyTypeOf<[F32, I32, I<128>]>)> {
  let summary = "An op to read from a stream channel/port of a switchbox";
  let description = [{
    An op to read from a stream channel/port of a switchbox.
  }];

  let arguments = (ins AnyInteger:$channel);
  let results = (outs AnyTypeOf<[F32, I32, I<128>]>:$streamValue);

  let assemblyFormat = [{
    `(` $channel `:` type($channel) `)` attr-dict `:` type($streamValue)
  }];

  let extraClassDeclaration = [{
    bool isWideStream() { return getStreamValue().getType().isInteger(128); }
    bool isFloatStream() {
      return getStreamValue().getType().isa<mlir::FloatType>();
    }
  }];
}

def AIE_PutStreamOp: AIE_Op<"putStream", [HasParent<"CoreOp">]> {
  let summary = "An op to write to a stream channel/port of a switchbox";
  let description = [{
    An op to write to a stream channel/port of a switchbox.
  }];

  let arguments = (
    ins AnyInteger:$channel,
        AnyTypeOf<[F32, I32, I<128>]>:$streamValue
  );

  let assemblyFormat = [{
    `(` $channel `:` type($channel) `,` $streamValue `:` type($streamValue) `)` attr-dict
  }];

  let extraClassDeclaration = [{
    bool isWideStream() { return getStreamValue().getType().isInteger(128); }
    bool isFloatStream() {
      return getStreamValue().getType().isa<mlir::FloatType>();
    }
  }];
}

def AIE_GetCascadeOp: AIE_Op<"getCascade", [HasParent<"CoreOp">]>, Results<(outs AnyI<384>)> {
  let summary = "An op to read from a cascading stream from a neighboring core";
  let description = [{
    An op to read from a cascading stream from a neighboring core.
  }];
  let results = (outs AnyI<384>:$cascadeValue);
  let assemblyFormat = [{ `(` `)` attr-dict `:` type($cascadeValue) }];
}

def AIE_PutCascadeOp: AIE_Op<"putCascade", [HasParent<"CoreOp">]> {
  let summary = "An op to write to a cascading stream from a neighboring core";
  let description = [{
    An op to write to a cascading stream from a neighboring core.
  }];

  let arguments = (ins AnyI<384>:$cascadeValue);

  let assemblyFormat = [{ `(` $cascadeValue `:` type($cascadeValue) `)` attr-dict }];
}

def AIE_ShimDMAAllocationOp : AIE_Op<"shimDMAAllocation", [HasParent<"DeviceOp">]> {
  let summary = "Runtime allocation information for a single shim DMA";
  let description = [{
    This op exists for cases where shimDMA configuration is performed outside of MLIR-AIE 
    and hence there is no appropriate dmaStart operation to indicate which channel is being
    used and on which column the shimDMA is. 

    It contains attributes for the sym_name of an operation which generated the shim DMA, 
    for the DMAChannelDir and channel index, and for the column of the shim tile to which 
    the originating operation was mapped.

    Example:
    ```
      %tile00 = AIE.tile(0, 0)
      %tile02 = AIE.tile(0, 2)
      AIE.objectfifo @of_in_0 (%tile00, { %tile02 }, 2) : !AIE.objectfifo<memref<64xi16>>
    ```
    could produce the following allocation info (channel direction MM2S, channel index 1, and shim column 0):
    ```
      AIE.shimDMAAllocation @of_in_0 (MM2S, 1, 0)
    ```
  }];

  let arguments = (
    ins FlatSymbolRefAttr:$sym_name,
        DMAChannelDir:$channelDir,
        AIEI64Attr:$channelIndex,
        AIEI64Attr:$col
  );

  let results = (outs);

  let assemblyFormat = [{
    $sym_name `(` $channelDir `,` $channelIndex `,` $col `)` attr-dict
  }];
}

def AIE_ObjectFifoCreateOp: AIE_Op<"objectfifo", [HasParent<"DeviceOp">, Symbol]> {
  let summary = "Create a circular buffer or channel between two tiles";
  let description = [{
    The `aie.objectFifo` operation creates a circular buffer established between a producer and one or
    more consumers, which are `aie.tile` operations. The`aie.objectFifo` instantiates the given number of
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
      AIE.objectfifo @of1 (%tile12, { %tile23 }, 4 : i32) : !AIE.objectfifo<memref<16xi32>>
    ```
    This operation creates an `objectFifo` between `%tile12` and `%tile23` of 4 elements, each a buffer of 16 32-bit integers.
    Note: If there are no `ObjectFifoAcquireOps` corresponding to this `objectFifo` on the cores of `%tile12` and `%tile23`,
    then the depths of the object pools on each tile will be 4, as specified. Otherwise, the cores are scanned and the
    highest number of acquired elements (+1 for prefetching) will be used instead, to ensure minimal resource usage.

    1-to-2 tiles broadcast example:
    ```
      AIE.objectfifo @of2 (%tile12, { %tile13, %tile23 }, 4 : i32) : !AIE.objectfifo<memref<16xi32>>
    ```
    This operation creates an `objectFifo` between `%tile12` and tiles `%tile13`, `%tile23` of 4 elements, each a buffer of x16
    32-bit integers.

    1-to-2 tiles broadcast with explicit sizes example:
    ```
      AIE.objectfifo @of3 (%tile12, { %tile13, %tile23 }, [2, 3, 4]) : !AIE.objectfifo<memref<16xi32>>
    ```
    This operation creates an `objectFifo` between `%tile12`, `%tile13` and `%tile23`. The depths of the `objectFifo` object pool
    at each tile are respectively 2, 3 and 4 for tiles `%tile12`, `%tile13` and `%tile23`. This overrides the depth analysis
    specified in the first example.

    ## Data Layout Transformations on AIE-ML devices
    
    On AIE-ML devices, objectFifos can also apply data layout transformations by
    using the DMAs n-dimensional address generation scheme. Two transformations
    can be applied for an objectFifo: one on the producer side, given by a
    `toStream` attribute, and one transformation on the consumer side, given by
    a `fromStream` attribute. See the `DMABDOp` documentation for a description
    of strides and wraps. The `toStream` and `fromStream` optional attributes
    are given directly following the producer or consumer tile declaration. 
    Different transformations can be specified for each consumer. See example
    below.

    Note that using data layout transformations will cause the DMA be used even
    between adjacent tiles whose objectFifos would otherwise use shared memory.

    Further note that data layout transforms always apply at a granularity of 
    `i32`s, irrespective of the used `memref` data type. This is an 
    architectural requirement. Hence, a stride of 4 always expresses 4 `i32`s, 
    i.e. 16 bytes.

    The following example shows an objectFifo which transposes a 16x16 matrix of 
    `i32`s on the producer side using strides and wraps. No transformation is
    applied on the consumer side of %tile13 (in this case the `fromStream` 
    attribute may also be left off), and a transformation on %tile23 first gives
    all even indices from the stream, followed by all odd indices:

    ```
      AIE.objectfifo @of4 (%tile12 toStream [<16, 1>, <16, 16>, <1,1>],
                           {%tile13 fromStream [],
                            %tile23 fromStream [<2, 1>, <128, 2>]}, 
                           2 : i32
                          ) : !AIE.objectfifo<memref<256xi32>>
    ```
  }];

  let arguments = (
    ins SymbolNameAttr:$sym_name,
        Index:$producerTile,
        Variadic<Index>:$consumerTiles,
        AIE_ObjectFifo_Depth:$elemNumber,
        TypeAttrOf<AIE_ObjectFifoType>:$elem_type,
        AIE_DimTupleArrayAttr:$dimensionsToStream,
        AIE_DimTupleArrayArrayAttr:$dimensionsFromStreamPerConsumer
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
    `)` attr-dict `:` $elem_type
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
                   CArg<"llvm::ArrayRef<AIE::DimTupleAttr>", "{}">:$dimensionsToStream,
                   CArg<"llvm::ArrayRef<AIE::DimTupleArrayAttr>", "{}">:$dimensionsFromStreamPerConsumer), [{
      odsState.addOperands(producerTile);
      odsState.addOperands(consumerTiles);
      odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
      odsState.addAttribute(getElemNumberAttrName(odsState.name), elemNumber);
      odsState.addAttribute(getElemTypeAttrName(odsState.name), mlir::TypeAttr::get(elem_type));
      odsState.addAttribute(getDimensionsToStreamAttrName(odsState.name),
                            odsBuilder.getAttr<DimTupleArrayAttr>(dimensionsToStream));
      odsState.addAttribute(getDimensionsFromStreamPerConsumerAttrName(odsState.name),
                            odsBuilder.getAttr<DimTupleArrayArrayAttr>(dimensionsFromStreamPerConsumer));
    }]>
  ];
}

def AIE_ObjectFifoLinkOp: AIE_Op<"objectfifo.link", [HasParent<"DeviceOp">]> {
  let summary = "Links two objectFifos through an intermediary tile's DMA";
  let description = [{
    The `aie.objectFifo.link` operation allows to mark two `objectFifos` as linked. This implies that the two `objectFifos` form
    one dataflow movement which is split accross multiple `objectFifos`. Specifically, during the `objectFifo` lowering there will
    be less memory elements generated at the link point as the two `objectFifos` can share.

    The two `objectFifos` which are linked must have a link point (i.e., a shared AIE tile).
    In L1, only `objectFifos` of same size may be linked. In L2, different sized objectFifos can be linked.

    Example:
    ```
      AIE.objectfifo @of1 (%t70, { %t72 }, 2) : !AIE.objectfifo<memref<64xi16>>
      AIE.objectfifo @of2 (%t72, { %t74 }, 2) : !AIE.objectfifo<memref<64xi16>>
      AIE.objectfifo.link [@of1] -> [@of2] ()
    ```
    This operation links two `objectFifos` which have tile `%t72` as a link point.

    To achieve a broadcast pattern through the link tile, the output `objectFifo` should have a list of all the consumers tiles.
    To achieve a distribute pattern from the link tile, there should be multiple output `objectFifos` in the LinkOp. In this case,
    parts will be taken out of the input `objectFifo`'s buffers based on the sizes of the output `objectFifos`, in the order they
    were given in the LinkOp.
    The join pattern is the exact inverse of the distribute one.
  }];

  let arguments = (
    ins SymbolRefArrayAttr:$fifoIns,
        SymbolRefArrayAttr:$fifoOuts
  );

  let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{
    $fifoIns `->` $fifoOuts `(` `)` attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    std::vector<ObjectFifoCreateOp> getInputObjectFifos();
    std::vector<ObjectFifoCreateOp> getOutputObjectFifos();

    bool isJoin() {
      return getFifoIns().size() > 1;
    }

    bool isDistribute() {
      return getFifoOuts().size() > 1;
    }

    std::optional<mlir::Value> getOptionalSharedTile();
  }];
}

def AIE_ObjectFifoRegisterExternalBuffersOp: AIE_Op<"objectfifo.register_external_buffers", [
    HasParent<"DeviceOp">, TileElement, IsShimNOCTile
  ]> {
  let summary = "Registers external buffers to given object fifo shim tile(s) to use in the associated shim DMA(s)";
  let description = [{
    The `aie.objectfifo.register_external_buffers` operation is used to register one or multiple external buffers
    to the shim tile(s) used in an `objectFifo` creation. During the `objectFifo` lowering pass, shim DMAs that are
    generated for those shim tiles will use the registered external buffers. This is currently done because 
    external buffers typically have a different size than the AIE buffers which are used in the AIE tiles of the
    same `objectFifos`.

    Example:
    ```
      AIE.objectfifo @of1 (%t70, %t73, 2) : !AIE.objectfifo<memref<64xi16>>
      %buffer_in_0  = AIE.external_buffer : memref<512 x i16>
      %buffer_in_1  = AIE.external_buffer : memref<512 x i16>
      AIE.objectfifo.register_external_buffers @of1 (%t70, {buffer_in_0, buffer_in_1}) : (memref<512 x i16>, memref<512 x i16>)
    ```
    This operation registers external buffers %buffer_in_0 and %buffer_in_1 to use in the shimDMA of shimTile %t70.
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

    This operation is then converted by the `AIEObjectFifoStatefulTransformPass` into `aie.useLock` operations on
    the locks of the `objectFifo` objects that will be acquired. Under the hood, the operation only performs
    new acquires if necessary. For example, if two objects have been acquired in the past and none have yet
    to be released by the same process, then performing another acquire operation on the same `objectFifo`
    within the same process of size two or less will not result in any new useLock operations (and for size
    greater than two, only (size - 2) useLock operations will be performed).

    Example:
    ```
      %subview = AIE.objectfifo.acquire @of1 (Consume, 2) : !AIE.objectfifosubivew<memref<16xi32>>
    ```
    This operation acquires the locks of the next two objects in the `objectFifo` named `@of1` from its consumer
    port and returns a subview of the acquired objects.
  }];

  let arguments = (
    ins ObjectFifoPort:$port,
        FlatSymbolRefAttr:$objFifo_name,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$size
  );

  let results = (outs AIE_ObjectFifoSubviewType:$subview);

  let assemblyFormat = [{
    attr-dict $objFifo_name `(` $port `,` $size `)` `:` type($subview)
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

    This operation is then converted by the `AIEObjectFifoStatefulTransformPass` into `aie.useLock` operations.

    Example:
    ```
      AIE.objectfifo.release @of1 (Produce, 1)
    ```
    This operation releases the lock of the next object in the `objectFifo` named `@of1` from producer port.
  }];

  let arguments = (
    ins ObjectFifoPort:$port,
        FlatSymbolRefAttr:$objFifo_name,
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$size
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
      %subview = AIE.objectfifo.acquire @of1 (Produce, 3) : !AIE.objectfifosubivew<memref<16xi32>>
      %elem = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubivew<memref<16xi32>> -> memref<16xi32>
    ```
    In this example, %elem is the first object of the subview. Note that this may not correspond to the first element of
    the `objectFifo` if other acquire operations took place beforehand.

  }];

  let arguments = (
    ins AIE_ObjectFifoSubviewType:$subview, 
        ConfinedAttr<AIEI32Attr, [IntMinValue<0>]>:$index
  );

  let hasVerifier = 1;

  let results = (outs AnyMemRef:$output);

  let assemblyFormat = [{
    $subview `[` $index `]` attr-dict `:` type($subview) `->` type($output)
  }];

  // Allow building an AIE_ObjectFifoSubviewAccessOp with just a subview value and an index.
  let builders = [
    OpBuilder<(ins "mlir::Value":$subview, "size_t":$index)>
  ];
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
      AIE.objectfifo @of1 (%t72, %t73, 2) : !AIE.objectfifo<memref<16xi32>>
      %length = arith.constant 10 : index
      %acquirePatternProducer = arith.constant dense<[1, 2, 2, 0]> : tensor<4xi32>
      %releasePatternProducer = arith.constant dense<[0, 1, 1, 2]> : tensor<4xi32>
      func @producer_work(%input : !AIE.objectfifosubivew<memref<16xi32>>) -> () { ... }

      AIE.objectfifo.register_process @of1 (Produce, %acquirePatternProducer : tensor<4xi32>, %releasePatternProducer : tensor<4xi32>, @producer_work, %length)
    ```
    This operation registers function @producer_work and associated patterns to the produce end of @of1.
    @producer_work will be called with the subviews produced when acquiring elements from @of1 following the acquire pattern.

    If the input patterns are static (only one element) then the length of the produced for loop will be that of the input %length.
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
      return getAcquirePatternTensor()
          .getDefiningOp<mlir::arith::ConstantOp>()
          .getValue()
          .cast<mlir::DenseIntElementsAttr>();
    }

    mlir::DenseIntElementsAttr getReleasePattern() {
      return getReleasePatternTensor()
          .getDefiningOp<mlir::arith::ConstantOp>()
          .getValue()
          .cast<mlir::DenseIntElementsAttr>();
    }

    int getProcessLength() {
      return getLength()
          .getDefiningOp<mlir::arith::ConstantOp>()
          .getValue()
          .cast<mlir::IntegerAttr>()
          .getInt();
    }
  }];
}

#endif // AIE_OPS