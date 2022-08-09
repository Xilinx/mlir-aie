//===- AIE.td ----------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

#ifdef AIE_OPS
#else
#define AIE_OPS
#endif

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "AIEInterfaces.td"

def AIE_Dialect : Dialect {
  let name = "AIE";
  let cppNamespace = "::xilinx::AIE";
  let description = [{

This is a dialect for describing netlists of AIE components in a
Versal device.  It focuses on representing logical stream connections
between cores and DMAs, along with the implementation of those logical
connections in the various switch components.  In the dialect, a
switch is referred to as 'switchbox' to avoid confusion with the
'switch' keyword in C/C++.

  }];
  let useDefaultTypePrinterParser = 1;
}

def Core: I32EnumAttrCase<"Core", 0>;
def DMA: I32EnumAttrCase<"DMA", 1>;
def FIFO: I32EnumAttrCase<"FIFO", 2>;
def South: I32EnumAttrCase<"South", 3>;
def West: I32EnumAttrCase<"West", 4>;
def North: I32EnumAttrCase<"North", 5>;
def East: I32EnumAttrCase<"East", 6>;
def PLIO: I32EnumAttrCase<"PLIO", 7>;
def NOC: I32EnumAttrCase<"NOC", 8>;
def Trace: I32EnumAttrCase<"Trace", 9>;

def WireBundle: I32EnumAttr<"WireBundle", "Bundle of wires",
  [Core, DMA, FIFO, South, West, North, East, PLIO, NOC, Trace]> {

  let cppNamespace = "xilinx::AIE";
}

def Produce: I32EnumAttrCase<"Produce", 0>;
def Consume: I32EnumAttrCase<"Consume", 1>;

def ObjectFifoPort: I32EnumAttr<"ObjectFifoPort", "Ports of an object FIFO",
  [Produce, Consume]> {

  let cppNamespace = "xilinx::AIE";
}


class AIE_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIE_Dialect, mnemonic, traits>;

def AIE_ObjectFifoType :
    DialectType<AIE_Dialect, CPred<"$_self.isa<AIEObjectFifoType>()">,
                "AIE objectFifo type">;

def AIE_ObjectFifoSubviewType :
    DialectType<AIE_Dialect, CPred<"$_self.isa<AIEObjectFifoSubviewType>()">,
                "AIE ObjectFifoSubview type">;

def AIE_Type : AnyTypeOf<[AIE_ObjectFifoType, AIE_ObjectFifoSubviewType]>;

def AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                          "scalar-or-tensor">;

def AnyScalar : TypeConstraint<Or<[Index.predicate,
                                   AnySignlessInteger.predicate,
                                   AnyFloat.predicate]>,
                                  "scalar">;

def AIE_TileOp: AIE_Op<"tile", [FlowEndPoint]>, Results<(outs Index:$result)> {
  let arguments = (
    ins Confined<I32Attr, [IntMinValue<0>]>:$col,
        Confined<I32Attr, [IntMinValue<0>]>:$row
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
    int colIndex() { return col(); }
    int rowIndex() { return row(); }
    bool isShimTile() { return row() == 0; }
    bool isShimNOCTile();
    bool isShimPLTile();
    bool isShimNOCorPLTile();
    bool isInternalMemWest() { return ((rowIndex() % 2) == 0); };
    MemOp getMemOp() {
      auto users = result().getUsers();
      for(auto user : users)
        if(auto memOp = llvm::dyn_cast<MemOp>(*user))
          return memOp;
      return nullptr;
    }
    CoreOp getCoreOp() {
      auto users = result().getUsers();
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

  let builders = [
    OpBuilder<(ins "int":$col, "int":$row),
    [{
              build($_builder, $_state, $_builder.getIndexType(),
                    $_builder.getI32IntegerAttr(col),
                    $_builder.getI32IntegerAttr(row));
              }]>
  ];
}

def AIE_GetTileOp: AIE_Op<"getTile", []>, Results<(outs Index:$result)> {
  let arguments = (
    ins Index:$col,
        Index:$row
  );

  let summary = "Get a reference to an AIE tile";
  let description = [{
    Return a reference to an AIE tile, given the column and the row of the tile.
  }];
  let assemblyFormat = [{ `(` $col `,` $row `)` attr-dict }];
}

def AIE_EndOp: AIE_Op<"end", [Terminator]> {
  let summary = "end op";
  let description = [{
    A generic terminator operation for AIE ops' regions.
  }];
  let assemblyFormat = [{ attr-dict }];
}

def AIE_SwitchboxOp: AIE_Op<"switchbox", [Interconnect, SingleBlockImplicitTerminator<"EndOp">]>,
                     Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );

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
  }];
  let builders = [
    OpBuilder<(ins "Value":$tile),
    [{
              build($_builder, $_state, $_builder.getIndexType(), tile);
              }]>
  ];
}

def AIE_ShimSwitchboxOp: AIE_Op<"shimswitchbox", [SingleBlockImplicitTerminator<"EndOp">]>,
                         Results<(outs Index)> {
  let arguments = (
    ins I32Attr:$col
  );
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
  let builders = [
    OpBuilder<(ins "Type":$resultType,
                      "int":$col), [{
        build($_builder, $_state, resultType, $_builder.getI32IntegerAttr(col));
    }]>
  ];
}

def AIE_ShimMuxOp: AIE_Op<"shimmux", [Interconnect,
                                      SingleBlockImplicitTerminator<"EndOp">]>,
                         Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );
  let summary = "Declare a switch in the PL shim";
  let description = [{
    This operation represents the additional interconnect that is part of a shim interface tile.
    Like the "AIE.switchbox" operation, "AIE.shimMux" is configured
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
  }];
 let builders = [
    OpBuilder<(ins "Value":$tile), [{
      build($_builder, $_state, $_builder.getIndexType(), tile);
    }]>
 ];
}

def AIE_ShimDMAOp: AIE_Op<"shimDMA", [FlowEndPoint/*, CallableOpInterface*/]>,
                         Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );
  let summary = "Declare a DMA in the PL shim";
  let description = [{
  }];
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    TileOp getTileOp();
  }];
}

def AIE_CoreOp: AIE_Op<"core", [FlowEndPoint]>, Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );
  let summary = "Declare a core module";
  let description = [{
    This operation represents an AIEngine processor core belonging to a tile.
    The region of a CoreOp contains code that gets run on the AIE core.  This code will
    typically be outlined into the LLVM dialect, eventually resulting in a binary file
    for each core.  The name of this file can be be specified using the 'elf_file'
    attribute.

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
    } { elf_file = "core_33.elf" }
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
  }];
  let builders = [
    OpBuilder<(ins "Value":$tile), [{
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

def AIE_PLIOOp: AIE_Op<"plio", []>, Results<(outs Index)> {
  let arguments = (
    ins I32Attr:$col
  );
  let summary = "Declare an interface to the PL";
  let description = [{
    An interface to the PL.
  }];
  let assemblyFormat = [{ `(` $col `)` attr-dict }];
  let extraClassDeclaration = [{
    int colIndex() { return col(); }
  }];
}

def AIE_ConnectOp: AIE_Op<"connect", [ParentOneOf<["SwitchboxOp", "ShimMuxOp"]> ]> {
  let arguments = (
    ins WireBundle:$sourceBundle,
        I32Attr:$sourceChannel,
        WireBundle:$destBundle,
        I32Attr:$destChannel
  );
  let summary = "A circuit-switched connection inside a switchbox";
  let description = [{
    This operation represents a programmed circuit-switched connection in a stream switch.
    It associates a source bundle and source channel with a destination bundle and a destination channel.
    This operation must exist within an "aie.switchbox" or "aie.shimswitchbox" operation.
    All of the "aie.connect" operations in a switchbox must have a different destinations.
    All of the "aie.connect" operations must also have a destination which is different from all
    of the "aie.masterset" operations in the same switchbox.

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
    int sourceIndex() { return sourceChannel(); }
    int destIndex() { return destChannel(); }
    Port sourcePort() { return std::make_pair(sourceBundle(), sourceIndex()); }
    Port destPort() { return std::make_pair(destBundle(), destIndex()); }
  }];
}

def AIE_FlowOp: AIE_Op<"flow", []> {
  let arguments = (
    ins Index:$source,
        WireBundle:$sourceBundle,
        I32Attr:$sourceChannel,
        Index:$dest,
        WireBundle:$destBundle,
        I32Attr:$destChannel
  );
  let summary = "A logical circuit-switched connection between cores";
  let description = [{
    The "aie.flow" operation represents a circuit switched connection between two endpoints, usually
    "aie.tile" operations.  During routing, this is replaced by "aie.connect" operations which represent
    the programmed connections inside a switchbox, along with "aie.wire" operations which represent
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
    int sourceIndex() { return sourceChannel(); }
    int destIndex() { return destChannel(); }
  }];
  // let builders = [
  //   OpBuilder<(ins "Value":$source, "int":$sourceBundle,
  //     "int":$sourceChannel, "Value":$dest, "int":$destBundle, "int":$destChannel), [{
  //     build($_builder, $_state,
  //           source,
  //           $_builder.getI32IntegerAttr(sourceBundle),
  //           $_builder.getI32IntegerAttr(sourceChannel),
  //           dest,
  //           $_builder.getI32IntegerAttr(destBundle),
  //           $_builder.getI32IntegerAttr(destChannel));
  //     }]>
  // ];
}

def AIE_ConnectionOp: AIE_Op<"connection", []> {
  let arguments = (
    ins Index:$source,
        WireBundle:$sourceBundle,
        I32Attr:$sourceChannel,
        Index:$dest,
        WireBundle:$destBundle,
        I32Attr:$destChannel
  );
  let summary = "A logical circuit-switched connection between cores";
  let description = [{
    The "aie.connection" operation represents a circuit switched connection between two endpoints, usually
    "aie.core" operations.  During routing, this is replaced by "aie.connect" operations which represent
    the programmed connections inside a switchbox, along with "aie.wire" operations which represent
    physical connections between switchboxes and other components.  Note that while "aie.flow" operations
    can express partial routes between tiles, this is not possible with "aie.connection" operations.

    Example:
      %22 = aie.tile(2, 2)
      %c22 = aie.core(%22)
      %11 = aie.tile(1, 1)
      %c11 = aie.core(%11)
      aie.flow(%c22, "Core" : 0, %c11, "Core" : 1)

  }];
  let assemblyFormat = [{
    `(` $source `,` $sourceBundle `:` $sourceChannel `,` $dest `,` $destBundle `:` $destChannel `)` attr-dict
  }];
  let extraClassDeclaration = [{
    int sourceIndex() { return sourceChannel(); }
    int destIndex() { return destChannel(); }
  }];
  // let builders = [
  //   OpBuilder<(ins "Value":$source, "int":$sourceBundle,
  //     "int":$sourceChannel, "Value":$dest, "int":$destBundle, "int":$destChannel), [{
  //     build($_builder, $_state,
  //           source,
  //           $_builder.getI32IntegerAttr(sourceBundle),
  //           $_builder.getI32IntegerAttr(sourceChannel),
  //           dest,
  //           $_builder.getI32IntegerAttr(destBundle),
  //           $_builder.getI32IntegerAttr(destChannel));
  //     }]>
  // ];
}

def AIE_AMSelOp: AIE_Op<"amsel", [HasParent<"SwitchboxOp">]>, Results<(outs Index)> {
  let arguments = (
    ins Confined<I8Attr, [IntMinValue<0>, IntMaxValue<5>]>:$arbiterID,
        Confined<I8Attr, [IntMinValue<0>, IntMaxValue<3>]>:$msel
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
    int arbiterIndex() { return arbiterID(); }
    int getMselValue() { return msel(); }
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

def AIE_MasterSetOp: AIE_Op<"masterset", [HasParent<"SwitchboxOp">]>, Results<(outs Index)> {
  let arguments = (
    ins WireBundle:$destBundle,
        I32Attr:$destChannel,
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
    int destIndex() { return destChannel(); }
    Port destPort() { return std::make_pair(destBundle(), destIndex()); }
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
    The "aie.wire" operation represents a physical set of connections between components in a Versal device.
    Typically, these components are switches, represented by an "aie.switchbox" operation, and tiles,
    represented by an [aie.tile](#aietile-aietileop) operation.
  }];
  let assemblyFormat = [{
    `(` $source `:` $sourceBundle `,` $dest `:` $destBundle `)` attr-dict
  }];
  // let builders = [
  //   OpBuilder<(ins "Value":$source, "int":$sourceBundle, "Value":$dest,
  //     "int":$destBundle), [{
  //     build($_builder, $_state,
  //           source,
  //           $_builder.getI32IntegerAttr((int)sourceBundle),
  //           dest,
  //           $_builder.getI32IntegerAttr((int)destBundle));
  //     }]>
  // ];
}

def AIE_PacketRulesOp: AIE_Op<"packetrules", [/*HasParent<"SwitchboxOp">*/
                                              SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins WireBundle:$sourceBundle,
        I32Attr:$sourceChannel
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
  let extraClassDeclaration = [{
    int sourceIndex() { return sourceChannel(); }
    Port sourcePort() { return std::make_pair(sourceBundle(), sourceIndex()); }
  }];
}

def AIE_PacketRuleOp: AIE_Op<"rule", [HasParent<"PacketRulesOp">]> {
  let arguments = (
    ins I8Attr:$mask,
        I8Attr:$value,
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
    int maskInt() { return mask(); }
    int valueInt() { return value(); }
  }];
  let assemblyFormat = [{
    `(` $mask `,` $value `,` $amsel `)` attr-dict
  }];
  // let builders = [
  //   OpBuilder<(ins "int":$mask, "int":$value, "Value":$amsel), [{
  //     build($_builder, $_state,
  //           $_builder.getI8IntegerAttr(mask),
  //           $_builder.getI8IntegerAttr(value),
  //           amsel);
  //     }]>
  // ];
}

def AIE_BroadcastPacketOp: AIE_Op<"broadcast_packet", [SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        I32Attr:$channel
  );
  let regions = (region AnyRegion:$ports);
  let summary = "Combination of broadcast and packet-switch";
  let description = [{
    An abstraction of broadcast and packet-switched flow. During place and
    route, it will be replaced by packet-switched flow and further replaced 
    by MasterSets and PacketRules inside switchboxes.

    Example:
    ```
      %70 = AIE.tile(7, 0)
      %73 = AIE.tile(7, 3)
      %74 = AIE.tile(7, 4)
      %63 = AIE.tile(6, 3)
      %64 = AIE.tile(6, 4)
      AIE.broadcast_packet(%70, "DMA" : 0){
        AIE.bp_id(0x0){
          AIE.bp_dest<%73, "DMA" : 0>
          AIE.bp_dest<%63, "DMA" : 0>
        }
        AIE.bp_id(0x1){
          AIE.bp_dest<%74, "DMA" : 0>
          AIE.bp_dest<%64, "DMA" : 0>
        }
      }
    ```
  }];
  let assemblyFormat = [{ `(` $tile `,` $bundle `:` $channel `)` regions attr-dict }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    int channelIndex() { return channel(); }
    Port port() { return std::make_pair(bundle(), channelIndex()); }
  }];
}

def AIE_BPIDOp: AIE_Op<"bp_id", [SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins I8Attr:$ID
  );
  let regions = (region AnyRegion:$ports);
  let summary = "A set of packets that share the same ID";
  let description = [{
    A set of destination packets that share the same source and ID. This must exist
    within an [AIE.broadcast_packet] operation.
    See [AIE.broadcast_packet]for an example.
  }];
  let assemblyFormat = [{ `(` $ID `)` regions attr-dict }];
  let extraClassDeclaration = [{
    int IDInt() { return ID(); }
  }];
}

def AIE_BPDestOp: AIE_Op<"bp_dest", [HasParent<"BPIDOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        I32Attr:$channel
  );
  let summary = "A destination port";
  let description = [{
    An object representing the destination of a  Broad Packet. This must exist
    within an [AIE.bp_id] operation.
    See [AIE.broadcast_packet] for an example.
  }];
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int channelIndex() { return channel(); }
    Port port() { return std::make_pair(bundle(), channelIndex()); }
  }];
}

def AIE_MulticastOp: AIE_Op<"multicast", [SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        I32Attr:$channel
  );
  let regions = (region AnyRegion:$ports);
  let summary = "An abstraction of multicast";
  let description = [{
    An abstraction of broadcast. During place and
    route, it will be replaced by multiple flows.

    Example:
    ```
      %70 = AIE.tile(7, 0)
      %73 = AIE.tile(7, 3)
      %74 = AIE.tile(7, 4)
      %63 = AIE.tile(6, 3)
      %64 = AIE.tile(6, 4)
      AIE.multicast(%70, "DMA" : 0){
        AIE.multi_dest<%73, "DMA" : 0>
        AIE.multi_dest<%74, "DMA" : 0>
        AIE.multi_dest<%63, "DMA" : 0>
        AIE.multi_dest<%64, "DMA" : 0>
      }
    ```
  }];
  let assemblyFormat = [{ `(` $tile `,` $bundle `:` $channel `)` regions attr-dict }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    int channelIndex() { return channel(); }
    Port port() { return std::make_pair(bundle(), channelIndex()); }
  }];
}

def AIE_MultiDestOp: AIE_Op<"multi_dest", [HasParent<"MulticastOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        I32Attr:$channel
  );
  let summary = "A destination port of multicast flow";
  let description = [{
    An object representing the destination of a multicast flow. This must exist
    within an [AIE.multicast] operation. There can be multiple destinations within an
    AIE.multicast Op.

    See [AIE.multicast]for an example.
  }];
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int channelIndex() { return channel(); }
    Port port() { return std::make_pair(bundle(), channelIndex()); }
  }];
}

def AIE_PacketFlowOp: AIE_Op<"packet_flow", [SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins I8Attr:$ID
  );
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
    int IDInt() { return ID(); }
  }];
}

def AIE_PacketSourceOp: AIE_Op<"packet_source", [HasParent<"PacketFlowOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        I32Attr:$channel
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
    int channelIndex() { return channel(); }
    Port port() { return std::make_pair(bundle(), channelIndex()); }
  }];
}

def AIE_PacketDestOp: AIE_Op<"packet_dest", [HasParent<"PacketFlowOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        I32Attr:$channel
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
    int channelIndex() { return channel(); }
    Port port() { return std::make_pair(bundle(), channelIndex()); }
  }];
}

def S2MM0:  I32EnumAttrCase<"S2MM0", 0>;
def S2MM1:  I32EnumAttrCase<"S2MM1", 1>;
def MM2S0:  I32EnumAttrCase<"MM2S0", 2>;
def MM2S1:  I32EnumAttrCase<"MM2S1", 3>;

def DMAChan: I32EnumAttr<"DMAChan", "DMA Channel number",
  [S2MM0, S2MM1, MM2S0, MM2S1]> {

  let cppNamespace = "xilinx::AIE";
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
    ins I32Attr:$packet_type,
        I32Attr:$packet_id
  );

  let assemblyFormat = [{
    `(` $packet_type `,` $packet_id `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int getPacketType() { return packet_type(); }
    int getPacketID() { return packet_id(); }
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
  }];

  let arguments = (
    ins AnyMemRef:$buffer,
        I32Attr:$offset,
        I32Attr:$len,
        Confined<I32Attr, [IntMinValue<0>, IntMaxValue<1>]>:$AB // 0: A, 1: B
  );

  let assemblyFormat = [{
    `(` `<` $buffer  `:` type($buffer) `,` $offset `,` $len `>` `,` $AB `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int getOffsetValue() { return offset(); }
    int getLenValue() { return len(); }
    bool isA() { return (AB() == 0); }
    bool isB() { return (AB() == 1); }
  }];

  // let builders = [
  //   OpBuilder<(ins "Value":$buffer, "int":$offset, "int":$len, "int":$AB), [{
  //     build($_builder, $_state,
  //           buffer,
  //           $_builder.getI32IntegerAttr(offset),
  //           $_builder.getI32IntegerAttr(len),
  //           $_builder.getI32IntegerAttr(AB));
  //     }]>
  // ];
}

def AIE_DMAStartOp: AIE_Op<"dmaStart", [ParentOneOf<["MemOp", "func::FuncOp", "ShimDMAOp"]>, Terminator]>,
                     Results<(outs I1:$valid)> {

  let summary = "An op to start DMA";
  let description = [{
    This operation declares a DMA channel to be used for data transfer.  It usually exists inside
    either a MemOp (representing a TileDMA channel), or in a ShimDMAOp (representing a ShimDMA channel).

    Example:
    ```
        AIE.dmaStart("MM2S0", ^bd0, ^end)
      ^bd0:
        AIE.useLock(%lock0, "Acquire", 0)
        AIE.dmaBd(<%buffer : memref<16 x f32>, 0, 16>, 0)
        AIE.useLock(%lock0, "Release", 1)
        br ^bd0
      ^end:
        AIE.end
    ```

    Comceptually, the AIE.dmaStart operation is a terminator that either passes
    control to a basic block containing DMA operations (through its first successor)
    or to a basic block for another dmaStart, to an AIE.end operation.
  }];

  let arguments = (ins DMAChan:$dmaChan);
  let successors = (successor AnySuccessor:$dest, AnySuccessor:$chain);
  let assemblyFormat = [{
    `(` $dmaChan `,` $dest `,` $chain `)` attr-dict
  }];

  let extraClassDeclaration = [{
    bool isSend() { return ((static_cast<int32_t>(dmaChan()) == 2) ||
                            (static_cast<int32_t>(dmaChan()) == 3)); }
    bool isRecv() { return ((static_cast<int32_t>(dmaChan()) == 0) ||
                            (static_cast<int32_t>(dmaChan()) == 1)); }
    int getSendChannelIndex() {
      return static_cast<int32_t>(dmaChan()) - 2;
    }
    int getRecvChannelIndex() {
      return static_cast<int32_t>(dmaChan()) - 0;
    }
    int getChannelNum() {
      return static_cast<int32_t>(dmaChan());
    }
  }];

  let builders = [
    OpBuilder<(ins "DMAChan":$channel, "Block *":$dest, "Block *":$chain),
    [{
              build($_builder, $_state, $_builder.getIntegerType(1), channel, dest, chain);
              }]>
  ];
}

// MemOps are not actually Callable, but we want to inline code into them, so we have to 
// implement CallableOpInterface
def AIE_MemOp: AIE_Op<"mem", [FlowEndPoint, CallableOpInterface]>,
                     Results<(outs Index)> {
  let summary = "Declare a memory op";
  let description = [{
    This operation creates a Memory module that belongs to a tile.
    The region of a MemOp is used to setup the DMAs and Block Descriptors.
    See DMAOp and DMABdOp for more concrete examples.
  }];
  let arguments = (
    ins Index:$tile
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    int maxSizeInBytes() { return 32768; }
    // CallableOpInterface
    Region *getCallableRegion();
    ArrayRef<Type> getCallableResults();
  }];
  let builders = [
    OpBuilder<(ins "Value":$tile), [{
      build($_builder, $_state, $_builder.getIndexType(), tile);
    }]>
  ];
}

def AIE_LockOp: AIE_Op<"lock", []>, Results<(outs Index)> {
  let summary = "Declare a physical lock";
  let description = [{
    This operation creates a physical lock.

    Example:
    ```
      %tile33 = AIE.tile(3, 3)
      %lck = AIE.lock(%tile33, 7)
    ```
    This operation represents a lock that lives in the Memory module of Tile(3, 3) with a lockID of 7
  }];
  let arguments = (
    ins Index:$tile,
        Confined<I32Attr, [IntMinValue<0>, IntMaxValue<15>]>:$lockID
  );
  let assemblyFormat = [{ `(` $tile `,` $lockID `)` attr-dict }];
  let extraClassDeclaration = [{
    int getLockID() { return lockID(); }
  }];
  let builders = [
    OpBuilder<(ins "Value":$tile, "int":$lockID), [{
      build($_builder, $_state,
        $_builder.getIndexType(),
        tile,
        $_builder.getI32IntegerAttr(lockID));
      }]>
  ];
}

def Acquire: I32EnumAttrCase<"Acquire", 0>;
def Release: I32EnumAttrCase<"Release", 1>;
def NonBlocking: I32EnumAttrCase<"NonBlocking", 0>;
def Blocking: I32EnumAttrCase<"Blocking", 1>;

def LockAction: I32EnumAttr<"LockAction", "lock acquire/release",
  [Acquire, Release]> {

  let cppNamespace = "xilinx::AIE";
}
def LockBlocking: I32EnumAttr<"LockBlocking", "lock operation is blocking",
  [NonBlocking, Blocking]> {

  let cppNamespace = "xilinx::AIE";
}

def AIE_UseLockOp: AIE_Op<"useLock", []> {
  let summary = "acquire/release lock op";
  let description = [{
    This operation uses a lock. A lock can be acquired with a value, or release with a value.
    This should be understood as a "blocking" operation.  This lock must appear in a parent op
    where the tile can be determined (A CoreOp, a ShimDMAOp, or a MemOp).  If the useLock
    operation appears in a module directly, an initialization to the lock will be generated in
    the host implementation.
  }];
  let arguments = (
    ins Index:$lock,
        Confined<I32Attr, [IntMinValue<0>, IntMaxValue<2>]>:$value,
        LockAction:$action,
        OptionalAttr<LockBlocking>:$blocking
  );
  let assemblyFormat = [{
    `(` $lock `,` $action `,` $value ( `,` $blocking^ )? `)` attr-dict
  }];
  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Value":$lock, "uint32_t":$value, "xilinx::AIE::LockAction":$action),
    [{
      build($_builder, $_state, lock, value, action, nullptr);
    }]>
  ];

  let extraClassDeclaration = [{
    bool acquire() { return (action() == LockAction::Acquire); }
    bool release() { return (action() == LockAction::Release); }
    int getLockValue() { return value(); }
    int getTimeout() { if(auto val = blocking()) return (int)*val; else return 1;}
  }];
}

def AIE_BufferOp: AIE_Op<"buffer", []>, Results<(outs AnyMemRef)> {
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
    ins Index:$tile
  );
  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = [{ `(` $tile `)` attr-dict `:` type($buffer) }];
  let extraClassDeclaration = [{
    bool hasName() {
      if(auto attr = getOperation()->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
        return true;
      } else {
        return false;
      }
    }
    StringAttr name() {
      if(auto attr = getOperation()->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
        return attr;
      } else {
        emitOpError("does not have '") << SymbolTable::getSymbolAttrName() <<
          "' attribute specified";
      }
      llvm_unreachable("unreachable");
    }
    // Return the address of this buffer
    int64_t address() {
      if(auto attr = getOperation()->getAttrOfType<IntegerAttr>("address")) {
        return attr.getInt();
      } else {
        emitOpError("does not have 'address' attribute specified");
      }
      llvm_unreachable("unreachable");
    }
    // Return the number of bytes that need to be allocated for this buffer.
    int64_t getAllocationSize();
    TileOp getTileOp();
  }];
  }

def AIE_ExternalBufferOp: AIE_Op<"external_buffer", []>, Results<(outs AnyMemRef)> {
  let summary = "Declare a buffer in external memory";
  let description = [{
    This operation represents a buffer that exists in some physical
    location in a device, most likely external memory.

    Example:
    ```
      %buf = AIE.external_buffer 0x200000 : memref<256xi64>
    ```
    This operation represents a buffer living at physical address 0x200000.
  }];
  let arguments = (ins I64Attr:$address);
  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = [{ $address attr-dict `:` type($buffer) }];
}

def AIE_TokenOp: AIE_Op<"token", [Symbol]> {
  let summary = "Declare a token (a logical lock)";
  let description = [{
    This operation creates a logical lock. We use Symbol so that it can be referenced globally.
    Unlike phsical locks, logical locks are unlimited, and we can specify any integer value
    associated with a lock. The logical lock is used to manually specify the dependence of tasks, or
    core executions.

    The operation can also be generated automatically if the Dependence Analysis can be leveraged.

    Example:
      AIE.token(0) {sym_name = "token0"} // Declare token0 with initial value of 0

      ...

      AIE.useToken @token0("Acquire", 0) // acquire token0 if its value is 0

      ...

      AIE.useToken @token0("Release", 5) // release token0 and set its value to 5

  }];
  let arguments = (ins I32Attr:$value);
  let assemblyFormat = [{ `(` $value `)` attr-dict }];
  let extraClassDeclaration = [{
    int getTokenValue() { return value(); }
  }];
}

def AIE_UseTokenOp: AIE_Op<"useToken", []> {
  let summary = "acquire/release a logical lock";
  let description = [{
    This operation uses token (logical lock). A logical lock can be acquired or released with a value.
    Similar to UseLockOp, this operation can be understood as "blocking" op.
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$tokenName,
        I32Attr:$value,
        LockAction:$action
  );
  let assemblyFormat = [{ $tokenName `(` $action `,` $value `)` attr-dict }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    bool acquire() { return (action() == LockAction::Acquire); }
    bool release() { return (action() == LockAction::Release); }
    int getTokenValue() { return value(); }
  }];
}

def AIE_MemcpyOp: AIE_Op<"memcpy", []> {
  let summary = "A memcpy op";
  let description = [{
    This operation defines a logical data transfer of a buffer from a source tile to another buffer
    from a destination tile.

    This operation should be lowered to Mem ops with DMA setup and Flow ops for routing data from
    the source tile to the dest. tile.
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$tokenName,
        I32Attr:$acqValue,
        I32Attr:$relValue,
        Index:$srcTile,
        AnyMemRef:$srcBuf,
        I32Attr:$srcOffset,
        I32Attr:$srcLen,
        Index:$dstTile,
        AnyMemRef:$dstBuf,
        I32Attr:$dstOffset,
        I32Attr:$dstLen
  );
  let assemblyFormat = [{
    $tokenName `(` $acqValue `,` $relValue `)` `(`
      $srcTile `:` `<` $srcBuf `,` $srcOffset `,` $srcLen `>` `,`
      $dstTile `:` `<` $dstBuf `,` $dstOffset `,` $dstLen `>` `)`
        attr-dict `:` `(` type($srcBuf) `,` type($dstBuf) `)`
  }];
  let extraClassDeclaration = [{
    int getAcquireTokenValue() { return acqValue(); }
    int getReleaseTokenValue() { return relValue(); }
    int getSrcOffsetValue() { return srcOffset(); }
    int getDstOffsetValue() { return dstOffset(); }
    int getSrcLenValue() { return srcLen(); }
    int getDstLenValue() { return dstLen(); }
  }];
}

def AIE_GetStreamOp: AIE_Op<"getStream", [HasParent<"CoreOp">]>,
                 Results<(outs AnyTypeOf<[F32, I32, I<128>]>)> {
  let summary = "An op to read from a stream channel/port of a switchbox";
  let description = [{
    An op to read from a stream channel/port of a switchbox.
  }];
  let arguments = (
    ins AnyInteger:$channel
  );
  let results = (outs AnyTypeOf<[F32, I32, I<128>]>:$streamValue);
  let assemblyFormat = [{
    `(` $channel `:` type($channel) `)` attr-dict `:` type($streamValue)
  }];
  let extraClassDeclaration = [{
    bool isWideStream() { return streamValue().getType().isInteger(128); }
    bool isFloatStream() { return streamValue().getType().isa<FloatType>(); }
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
    `(` $streamValue `:` type($streamValue) `,` $channel `:` type($channel) `)` attr-dict
  }];
  let extraClassDeclaration = [{
    bool isWideStream() { return streamValue().getType().isInteger(128); }
    bool isFloatStream() { return streamValue().getType().isa<FloatType>(); }
  }];
}

def AIE_GetCascadeOp: AIE_Op<"getCascade", [HasParent<"CoreOp">]>,
                      Results<(outs AnyI<384>)> {
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
  let arguments = (
    ins AnyI<384>:$cascadeValue
  );
  let assemblyFormat = [{ `(` $cascadeValue `:` type($cascadeValue) `)` attr-dict }];
}


/// Experimental Herd operations
def AIE_HerdOp: AIE_Op<"herd", []>, Results<(outs Index)> {
  let summary = "Declare a herd which is a bundle of core organized in a rectangular shape";
  let description = [{
    This operation creates a group of AIE tiles in 2D shape.

    Example:
      %herd0 = AIE.herd[1][1] // a single AIE tile. location unknown
      %herd1 = AIE.herd[4][1] // a row of four-AIE tile

    The operation can be used in replacement of a TileOp -- in case we want to select a group of
    hardware entities (cores, mems, switchboxes) instead of individual entity, and we don't want to
    specify their locations just yet. This can be useful if we want to generate parameterizable
    code (the column and row values are parameterized).

    Example:

      %herd = AIE.herd[2][2] // a herd of 2x2 AIE tiles

      AIE.core(%herd) {
        // all the cores belong to this herd runs the same code
      }
  }];
  let arguments = (
    ins I32Attr:$width,
        I32Attr:$height
  );
  let extraClassDeclaration = [{
    int getHerdWidth()   { return width(); }
    int getHerdHeight()  { return height(); }
    int getNumAIETiles() { return getHerdWidth() * getHerdHeight(); }
    StringAttr name() {
      if(auto attr = getOperation()->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
        return attr;
      } else {
        emitOpError("does not have '") << SymbolTable::getSymbolAttrName() <<
          "' attribute specified";
      }
      llvm_unreachable("unreachable");
    }
  }];
  let assemblyFormat = [{ `[` $width `]` `[` $height `]` attr-dict }];
  let builders = [
    OpBuilder<(ins "int":$width, "int":$height),
    [{
              build($_builder, $_state, $_builder.getIndexType(),
                    $_builder.getI32IntegerAttr(width),
                    $_builder.getI32IntegerAttr(height));
              }]>
  ];
}

def AIE_PlaceOp: AIE_Op<"place", []> {
  let summary = "A place operation that specifies the relative placement (XY) of one herd to another";
  let description = [{
    A place operation that specifies the relative placement (XY) of one herd to another.
  }];
  let arguments = (
    ins Index:$sourceHerd,
        Index:$destHerd,
        I32Attr:$distX,
        I32Attr:$distY
  );
  let assemblyFormat = [{ `(` $sourceHerd `,` $destHerd `,` $distX `,` $distY `)` attr-dict }];
  let extraClassDeclaration = [{
    int getDistXValue() { return distX(); }
    int getDistYValue() { return distY(); }
  }];
}

def AIE_RouteOp: AIE_Op<"route", []> {
  let summary = "A route operation that routes one herd to another";
  let description = [{
    A route operation that routes one herd to another.
  }];
  let arguments = (
    ins Index:$sourceHerds,
        WireBundle:$sourceBundle,
        I32Attr:$sourceChannel,
        Index:$destHerds,
        WireBundle:$destBundle,
        I32Attr:$destChannel
  );
  let assemblyFormat = [{
    `(` `<` $sourceHerds `,` $sourceBundle `:` $sourceChannel `>` `,`
        `<` $destHerds   `,` $destBundle   `:` $destChannel   `>` `)` attr-dict
  }];
  let extraClassDeclaration = [{
    int getSourceChannelValue()  { return sourceChannel(); }
    int getDestChannelValue()  { return destChannel(); }
  }];
}

def AIE_IterOp: AIE_Op<"iter", []>, Results<(outs Index)> {
  let summary = "An iter operation";
  let description = [{
    This operation generates index values that can be used with the SelectOp to select a group of tiles
    from a herd.

    Example:
      %iter0 = AIE.iter(0, 15, 1) // 0, 1, 2, ... , 15
      %iter1 = AIE.iter(2, 8, 2)  // 2, 4, 6
  }];
  let arguments = (
    ins I32Attr:$start,
        I32Attr:$end,
        I32Attr:$stride
  );
  let assemblyFormat = [{ `(` $start `,` $end `,` $stride `)` attr-dict }];
  let extraClassDeclaration = [{
    int getStartValue()  { return start(); }
    int getEndValue()    { return end(); }
    int getStrideValue() { return stride(); }
  }];
  let builders = [
    OpBuilder<(ins "int":$start, "int":$end, "int":$stride), [{
      build($_builder, $_state, $_builder.getIndexType(),
            $_builder.getI32IntegerAttr(start),
            $_builder.getI32IntegerAttr(end),
            $_builder.getI32IntegerAttr(stride));
      }]>
  ];
}

def AIE_SelectOp: AIE_Op<"select", []>, Results<(outs Index)> {
  let summary = "A select operation";
  let description = [{
    This operation selects a group of tiles based on the selected indices.

    Example:

      %herd = AIE.herd[4][4] // a herd of 4x4 tiles

      %ix = AIE.iter(0, 4, 1) // 0, 1, 2, 3
      %iy = AIE.iter(0, 1, 1) // 0

      %sub_herd = AIE.select(%herd, %ix, %iy)

    The SelectOp in the above example will select the tiles %herd[0][0], %herd[1][0],
    %herd[2][0], %herd[3][0] (the first column of the herd).
  }];
  let arguments = (
    ins Index:$startHerd,
        Index:$iterX,
        Index:$iterY
  );
  let assemblyFormat = [{ `(` $startHerd `,` $iterX `,` $iterY `)` attr-dict }];
  let builders = [
    OpBuilder<(ins "Value":$startHerd, "Value":$iterX, "Value":$iterY), [{
      build($_builder, $_state, $_builder.getIndexType(),
            startHerd, iterX, iterY);
      }]>
  ];
}

def AIE_ObjectFifoCreateOp: AIE_Op<"objectFifo.createObjectFifo", []> {
  let summary = "Create a circular buffer or channel between two tiles";
  let description = [{
    The "aie.createObjectFifo" operation creates a circular buffer established between a producer and
    a consumer, which are "aie.tile" operations. The aie.createObjectFifo instantiates the given number of 
    buffers (of given output type) and their locks in the Memory Module of the producer tile after lowering. 
    These elements represent the conceptual depth of the objectFifo.

    This operation is then converted by the AIEObjectFifoStatefulTransformPass into buffers and their associated 
    locks. The pass also establishes Flow and DMA operations between the producer and consumer tiles if they are
    not adjacent.

    Example:
    ```
      %objFifo = AIE.objectFifo.createObjectFifo(%tile12, %tile13, 4) : !AIE.objectFifo<memref<16xi32>> 
    ```
    This operation creates an objectFifo between tiles 12 and 13 of 4 elements, each a buffer of 16 32-bit integers.
  }];

  let arguments = (
    ins Index:$producerTile,
        Index:$consumerTile,
        Confined<I32Attr, [IntMinValue<0>]>:$elemNumber
  );

  let results = (outs AIE_ObjectFifoType:$fifo);

  let assemblyFormat = [{
    `(` $producerTile `,` $consumerTile `,` $elemNumber`)` attr-dict `:` type($fifo)
  }];

  let extraClassDeclaration = [{
    int size() { return elemNumber(); }
    TileOp getProducerTileOp();
    TileOp getConsumerTileOp();
  }];
}

def AIE_ObjectFifoAcquireOp: AIE_Op<"objectFifo.acquire", []> {
  let summary = "Acquire operation to lock and return objects of an ObjectFifo";
  let description = [{
    The "aie.objectFifo.acquire" operation first acquires the locks of the next given number 
    of objects in the objectFifo. The mode it acquires the locks in is chosen based on the port 
    (producer: acquire for write, consumer: acquire for read). Then, it returns a subview of 
    the acquired objects which can be used to access them.

    This operation is then converted by the AIEObjectFifoStatefulTransformPass into useLock operations on 
    the locks of the objectFifo objects that will be acquired. Under the hood, the operation only performs
    new acquires if necessary. For example, if two objects have been acquired in the past and none have yet
    to be released by the same process, then performing another acquire operation on the same objectFifo 
    within the same process of size two or less will not result in any new useLock operations (and for size 
    greater than two, only (size - 2) useLock operations will be performed).

    Example:
    ```
      %subview = AIE.objectFifo.acquire<Consume>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
    ```
    This operation acquires the locks of the next two objects in objFifo from its consumer port and returns a subview of the acquired objects.
  }];

  let arguments = (
    ins ObjectFifoPort:$port,
        AIE_ObjectFifoType:$fifo,
        Confined<I32Attr, [IntMinValue<0>]>:$size
  );

  let results = (outs AIE_ObjectFifoSubviewType:$subview);

  let assemblyFormat = [{
    attr-dict `<` $port `>` `(` $fifo `:` type($fifo) `,` $size `)` `:` type($subview)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Value getAIEObjectFifo() { return fifo(); }
    AIEObjectFifoType getAIEObjectFifoType() {
      return getAIEObjectFifo().getType().cast<AIEObjectFifoType>();
    }
    int acqNumber() { return size(); }
  }];
}

def AIE_ObjectFifoReleaseOp: AIE_Op<"objectFifo.release", []> {
  let summary = "Release operation for object locks in an ObjectFifo";
  let description = [{
    The "aie.objectFifo.release" operation releases the locks of the given number of objects 
    in the objectFifo. The mode it releases the locks in is chosen based on the "port" 
    (producer: release for read, consumer: release for write). 

    This operation is then converted by the AIEObjectFifoStatefulTransformPass into useLock operations.

    Example:
    ```
      AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
    ```
    This operation releases the lock of the next object in objFifo from producer port.
  }];

  let arguments = (
    ins ObjectFifoPort:$port,
        AIE_ObjectFifoType:$fifo,
        Confined<I32Attr, [IntMinValue<0>]>:$size
  );

  let assemblyFormat = [{
    attr-dict `<` $port `>` `(` $fifo `:` type($fifo) `,` $size `)` 
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Value getAIEObjectFifo() { return fifo(); }
    AIEObjectFifoType getAIEObjectFifoType() {
      return getAIEObjectFifo().getType().cast<AIEObjectFifoType>();
    }
    int relNumber() { return size(); }
  }];
}

def AIE_ObjectFifoSubviewAccessOp : AIE_Op<"objectFifo.subview.access", []> {
  let summary = "ObjectFifoSubview type accessor method";
  let description = [{
    Access the Nth element of a value of ObjectFifoSubview type.

  Example:
    ```
      %subview = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 3) : !AIE.objectFifoSubview<memref<16xi32>>
      %elem = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
    ```
    In this example, elem is the first object of the subview. Note that this may not correspond to the first element of 
    the objectFifo if other acquire operations took place beforehand.

  }];

  let arguments = (
    ins AIE_ObjectFifoSubviewType:$subview, 
        Confined<I32Attr, [IntMinValue<0>]>:$index
  );

  let results = (outs AnyMemRef:$output);

  let assemblyFormat = [{
    $subview `[` $index `]` attr-dict `:` type($subview) `->` type($output)
  }];

  // Allow building an AIE_ObjectFifoSubviewAccessOp with just a subview value and an index.
  let builders = [
    OpBuilder<(ins "Value":$subview, "size_t":$index)>
  ];

  let extraClassDeclaration = [{
    int getIndex() { return index(); }
  }];
}

def AIE_ObjectFifoRegisterProcessOp: AIE_Op<"objectFifo.registerProcess", []> {
  let summary = "Operation that produces the acquire/release patterns for a process registered to an objectFifo";
  let description = [{
    The "aie.registerProcess" operation allows the user to register a function to an objectFifo along with its 
    acquire and release patterns. These patterns will be used to generate a sequence of acquires and releases
    on the objectFifo elements. This generated sequence is often in the form of a for loop, however, in the case 
    of cyclo-static patterns only the repetition of same number accesses and releases will generate a for loop. 
    This may result in multiple for loops of different sizes being generated. If there is no repetition, then no 
    loops will be generated.

    Example:
    ```
      %length = arith.constant 10 : index
      %acquirePatternProducer = arith.constant dense<[1, 2, 2, 0]> : tensor<4xi32>
      %releasePatternProducer = arith.constant dense<[0, 1, 1, 2]> : tensor<4xi32>
      func @producer_work(%input : !AIE.objectFifoSubview<memref<16xi32>>) -> () { ... }

      AIE.objectFifo.registerProcess<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, %acquirePatternProducer : tensor<4xi32>, %releasePatternProducer : tensor<4xi32>, @producer_work, %length)
    ```
    This operation registers function @producer_work and associated patterns to the produce end of %objFifo. 
    @producer_work will be called with the subviews produced when acquiring elements from %objFifo following the acquire pattern.

    If the input patterns are static (only one element) then the length of the produced for loop will be that of the input %length.
    If the input patterns are cyclo-static then they must be of the same size.
  }];

  let arguments = (
    ins ObjectFifoPort:$port,
        AIE_ObjectFifoType:$fifo,
        I32Tensor:$acquirePattern,
        I32Tensor:$releasePattern,
        FlatSymbolRefAttr:$callee,
        Index:$length
  );

  let assemblyFormat = [{
    attr-dict `<` $port `>` `(` $fifo `:` type($fifo) `,` $acquirePattern `:` type($acquirePattern) `,` $releasePattern `:` type($releasePattern) `,` $callee `,` $length`)` 
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ObjectFifoPortAttr getPortAttr() { return portAttr(); }
    DenseIntElementsAttr getAcquirePattern() { return acquirePattern().getDefiningOp<arith::ConstantOp>().getValue().cast<DenseIntElementsAttr>(); }
    DenseIntElementsAttr getReleasePattern() { return releasePattern().getDefiningOp<arith::ConstantOp>().getValue().cast<DenseIntElementsAttr>(); }
    int getProcessLength() { return length().getDefiningOp<arith::ConstantOp>().getValue().cast<IntegerAttr>().getInt(); }
  }];
}