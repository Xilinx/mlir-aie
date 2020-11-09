// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef AIE_OPS
#else
#define AIE_OPS
#endif

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AIE_Dialect : Dialect {
  let name = "AIE";
  let description = [{

This is a dialect for describing netlists of AIE components in a
Versal device.  It focuses on representing logical stream connections
between cores and DMAs, along with the implementation of those logical
connections in the various switch components.  In the dialect, a
switch is referred to as 'switchbox' to avoid confusion with the
'switch' keyword in C/C++.

  }];
}

def ME: I32EnumAttrCase<"ME", 0>;
def DMA: I32EnumAttrCase<"DMA", 1>;
def FIFO: I32EnumAttrCase<"FIFO", 2>;
def South: I32EnumAttrCase<"South", 3>;
def West: I32EnumAttrCase<"West", 4>;
def North: I32EnumAttrCase<"North", 5>;
def East: I32EnumAttrCase<"East", 6>;

def WireBundle: I32EnumAttr<"WireBundle", "Bundle of wires",
  [ME, DMA, FIFO, South, West, North, East]> {

  let cppNamespace = "xilinx::AIE";
}

// 0 ME0
// 1 ME1
// 2 DMA0
// 3 DMA1
// 4 Tile Ctrl
// 5 FIFO0
// 6 FIFO1
// 7 South0
// 8 South1
// 9 South2
// 10 South3
// 11 West0
// 12 West1
// 13 West2
// 14 West3
// 15 North0
// 16 North1
// 17 North2
// 18 North3
// 19 North4
// 20 North5
// 21 East0
// 22 East1
// 23 East2
// 24 East3
def ME0: I32EnumAttrCase<"ME0", 0>;
def ME1: I32EnumAttrCase<"ME1", 1>;
def DMA0: I32EnumAttrCase<"DMA0", 2>;
def DMA1: I32EnumAttrCase<"DMA1", 3>;
def TileCtrl: I32EnumAttrCase<"TileCtrl", 4>;
def FIFO0: I32EnumAttrCase<"FIFO0", 5>;
def FIFO1: I32EnumAttrCase<"FIFO1", 6>;
def South0: I32EnumAttrCase<"South0", 7>;
def South1: I32EnumAttrCase<"South1", 8>;
def South2: I32EnumAttrCase<"South2", 9>;
def South3: I32EnumAttrCase<"South3", 10>;
def West0: I32EnumAttrCase<"West0", 11>;
def West1: I32EnumAttrCase<"West1", 12>;
def West2: I32EnumAttrCase<"West2", 13>;
def West3: I32EnumAttrCase<"West3", 14>;
def North0: I32EnumAttrCase<"North0", 15>;
def North1: I32EnumAttrCase<"North1", 16>;
def North2: I32EnumAttrCase<"North2", 17>;
def North3: I32EnumAttrCase<"North3", 18>;
def North4: I32EnumAttrCase<"North4", 19>;
def North5: I32EnumAttrCase<"North5", 20>;
def East0: I32EnumAttrCase<"East0", 21>;
def East1: I32EnumAttrCase<"East1", 22>;
def East2: I32EnumAttrCase<"East2", 23>;
def East3: I32EnumAttrCase<"East3", 24>;
def CoreTrace: I32EnumAttrCase<"CoreTrace", 25>;
def MemTrace: I32EnumAttrCase<"MemTrace", 26>;

def SSouth4: I32EnumAttrCase<"South4", 11>;
def SSouth5: I32EnumAttrCase<"South5", 12>;
def SWest0: I32EnumAttrCase<"West0", 13>;
def SWest1: I32EnumAttrCase<"West1", 14>;
def SWest2: I32EnumAttrCase<"West2", 15>;
def SWest3: I32EnumAttrCase<"West3", 16>;
def SNorth0: I32EnumAttrCase<"North0", 17>;
def SNorth1: I32EnumAttrCase<"North1", 18>;
def SNorth2: I32EnumAttrCase<"North2", 19>;
def SNorth3: I32EnumAttrCase<"North3", 20>;

def MasterPortEnum: I32EnumAttr<"MasterPortEnum", "AIE Switchbox Master Port",
                                [ME0, ME1, DMA0, DMA1, TileCtrl, FIFO0, FIFO1,
                                 South0, South1, South2, South3,
                                 West0, West1, West2, West3,
                                 North0, North1, North2, North3, North4, North5,
                                 East0, East1, East2, East3]> {
  let cppNamespace = "xilinx::AIE";
}
def SlavePortEnum: I32EnumAttr<"SlavePortEnum", "AIE Switchbox Slave Port",
                               [ME0, ME1, DMA0, DMA1, TileCtrl, FIFO0, FIFO1,
                                South0, South1, South2, South3, SSouth4, SSouth5,
                                SWest0, SWest1, SWest2, SWest3,
                                SNorth0, SNorth1, SNorth2, SNorth3,
                                East0, East1, East2, East3, CoreTrace, MemTrace]> {
  let cppNamespace = "xilinx::AIE";
}

// def StatisticsOpInterface : OpInterface<"StatisticsOpInterface"> {
//   let description = [{
//      This interface allows ops to expose a static operation profile,
// 	  describing the computational behavior of their function.
//   }];

//   let methods = [
//     InterfaceMethod<
//       "Return statistics about the compute requirements of an op",
//       "std::map<std::string, uint64_t>", "getStatistics"
//     >,
//   ];
// }

class AIE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<AIE_Dialect, mnemonic, traits>;

def AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                          "scalar-or-tensor">;

def AnyScalar : TypeConstraint<Or<[Index.predicate,
                                   AnySignlessInteger.predicate,
                                   AnyFloat.predicate]>,
                                  "scalar">;

def AIE_TileOp: AIE_Op<"tile", []>, Results<(outs Index:$result)> {
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

  let verifier = [{ return ::verify(*this); }];
  let extraClassDeclaration = [{
    int getNumSourceConnections(WireBundle bundle);
    int getNumDestConnections(WireBundle bundle);
    int colIndex() { return col(); }
    int rowIndex() { return row(); }
    bool isInternalMemWest() { return ((rowIndex() % 2) == 0); };
    MemOp getMemOp() {
      auto users = result().getUsers();
      bool found = false;
      for(auto user : users) {
        if(auto memOp = llvm::dyn_cast<MemOp>(*user)) {
          return memOp;
        }
      }
      return nullptr;
    }
    CoreOp getCoreOp() {
      auto users = result().getUsers();
      bool found = false;
      for(auto user : users) {
        if(auto coreOp = llvm::dyn_cast<CoreOp>(*user)) {
          return coreOp;
        }
      }
      return nullptr;
    }
  }];

  let assemblyFormat = [{
    `(` $col `,` $row `)` attr-dict
  }];

  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "int col, int row", [{
              build(builder, result, builder.getIndexType(),
                    builder.getI32IntegerAttr(col),
                    builder.getI32IntegerAttr(row));
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
  let assemblyFormat = [{
    `(` $col `,` $row `)` attr-dict
  }];
}

def AIE_EndOp: AIE_Op<"end", [Terminator]> {
  let summary = "end op";
  let description = [{
    A generic terminator operation for AIE ops' regions.
  }];
  let assemblyFormat = [{
    attr-dict
  }];
}

def AIE_SwitchboxOp: AIE_Op<"switchbox", [SingleBlockImplicitTerminator<"EndOp">]>,
                     Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );

  let summary = "Declare a switch";
  let description = [{
    This operation creates a stream switch that belongs to a tile.
  }];
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let verifier = [{ return ::verify(*this); }];
  let extraClassDeclaration = [{
    int getNumSourceConnections(WireBundle bundle);
    int getNumDestConnections(WireBundle bundle);
    int colIndex();
    int rowIndex();
  }];
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value tile", [{
              build(builder, result, builder.getIndexType(), tile);
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
    6 Ports to North (ME Tile) 4 Ports from North (ME Tile)
    4 Ports to West 4 Ports from West
    4 Ports to East 4 Ports from East
    6 Ports to South(DMA, NoC I/F, PL I/F) 8 Ports from South (DMA, NoC I/F, PL I/F)
    2 Ports to FIFOs 2 Ports from FIFOs
    1 Port for control packet for ME Shim register access
    1 Port for response to access for ME Shim registers
    1 Port for trace packet from ME Shim

  }];
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $col `)` regions attr-dict }];
  let verifier = [{ return ::verify(*this); }];
  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Type resultType, int col", [{
              build(builder, result, resultType,
                    builder.getI32IntegerAttr(col));
              }]>
  ];
}

def AIE_CoreOp: AIE_Op<"core", []>, Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );

  let summary = "Declare a core module";
  let description = [{
    This operation creates a Core module that belongs to a tile.
    The region of a CoreOp contains code that gets run on the AIE core.
    Therefore, all the operations inside this region will get lowered into LLVM Dialect.
  }];

  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    bool isMemWest() { return ((rowIndex() % 2) == 0); };
  }];

  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, Value tile", [{
              build(builder, result, builder.getIndexType(), tile);
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
  let assemblyFormat = [{
    `(` $arg `:` type($arg) `)` attr-dict
  }];
}

def AIE_PLIOOp: AIE_Op<"plio", []>, Results<(outs Index)> {
  let arguments = (
    ins I32Attr:$col
  );

  let summary = "Declare an interface to the PL";
  let description = [{
    An interface to the PL.
  }];
  let assemblyFormat = [{
    `(` $col `)` attr-dict
  }];
  let extraClassDeclaration = [{
    int colIndex() { return col(); }
  }];
  // let builders = [
  //   OpBuilder<"OpBuilder & builder, OperationState &result, "
  //             "Type resultType, int col, int row">
  // ];
}

def AIE_ShimDMAOp: AIE_Op<"shimDMA", []>, Results<(outs Index)> {
  let arguments = (
    ins I32Attr:$col
  );

  let summary = "Declare a ShimDMA";
  let description = [{
    A DMA in the PL-AIE Shim.
  }];
  let assemblyFormat = [{
    `(` $col `)` attr-dict
  }];
  let extraClassDeclaration = [{
    int colIndex() { return col(); }
  }];
  // let builders = [
  //   OpBuilder<"OpBuilder & builder, OperationState &result, "
  //             "Type resultType, int col, int row">
  // ];
}

def AIE_ConnectOp: AIE_Op<"connect", [/*HasParent<"SwitchboxOp">*/]> {
  let arguments = (
    ins WireBundle:$sourceBundle,
        I32Attr:$sourceChannel,
        WireBundle:$destBundle,
        I32Attr:$destChannel
  );
  let summary = "A circuit-switched connection inside a switchbox";
  let description = [{
    The "aie.connect" operation represents a programmed circuit-switched connection in a stream switch.
    It associates a source bundle and source channel with a destination bundle and a destination channel.
    The "aie.connect" operation must exist within an "aie.switchbox" or "aie.shimswitchbox" operation.
    All of the "aie.connect" operations in a switchbox must have a different destination.
    All of the "aie.connect" operations must also have a destination which is different from all
    of the "aie.masterset" operations in the same switchbox.
    Example:

    %7 = aie.switchbox(1, 1) {
      aie.connect<"West" : 0, "ME" : 1>
    }
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
  // let builders = [
  //   OpBuilder<"OpBuilder & builder, OperationState &result, "
  //             "WireBundle sourceBundle, int sourceChannel, "
  //             "WireBundle destBundle, int destChannel", [{
  //             build(builder, result,
  //                   sourceBundle,
  //                   APInt(32, sourceChannel),
  //                   destBundle,
  //                   APInt(32, destChannel));
  //             }]
  //   >
  // ];
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
      %00 = aie.tile(0, 0)
      %11 = aie.tile(1, 1)
      %01 = aie.tile(0, 1)
      aie.flow(%00, "DMA" : 0, %11, "ME" : 1)

  }];
  let assemblyFormat = [{
    `(` $source `,` $sourceBundle `:` $sourceChannel `,` $dest `,` $destBundle `:` $destChannel `)` attr-dict
  }];
  let extraClassDeclaration = [{
    int sourceIndex() { return sourceChannel(); }
    int destIndex() { return destChannel(); }
  }];
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "Value source, int sourceBundle, int sourceChannel,"
              "Value dest, int destBundle, int destChannel", [{
              build(builder, result,
                    source,
                    builder.getI32IntegerAttr(sourceBundle),
                    builder.getI32IntegerAttr(sourceChannel),
                    dest,
                    builder.getI32IntegerAttr(destBundle),
                    builder.getI32IntegerAttr(destChannel));
              }]>
  ];
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
      %a5_m3 = AIE.msel<5>(3)

    This code declares a pointer that represents an  arbiter 5 with msel=3

    From the spec, the number of arbiter per switchbox is 6, and the number of msel values per
    arbiter is 4

    If a slave port uses this pointer in one of its configuration (via PacketRulesOp and PacketRuleOp),
    and so does a master port (via MastersetOp), the compiler will establish a packet-switched
    route path from the slave port to the master port.

    Check out MastersetOp and PacketRuleOp for how to use amsel pointer.
  }];

  let assemblyFormat = [{
    `<` $arbiterID `>` `(` $msel `)` attr-dict
  }];
  let extraClassDeclaration = [{
    int arbiterIndex() { return arbiterID(); }
    int getMselValue() { return msel(); }
  }];

  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "int arbiterID, int msel", [{
              build(builder, result, builder.getIndexType(),
                    builder.getI8IntegerAttr(arbiterID),
                    builder.getI8IntegerAttr(msel));
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
      AIE.masterset("ME" : 0, %a0_m2)

    The code will configure the master port <"ME" : 0> to use arbiter 0 with msel 2
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
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "Value source, int sourceBundle, "
              "Value dest, int destBundle", [{
              build(builder, result,
                    source,
                    builder.getI32IntegerAttr((int)sourceBundle),
                    dest,
                    builder.getI32IntegerAttr((int)destBundle));
              }]>
  ];
}

def AIE_PacketRulesOp: AIE_Op<"packetrules", [/*HasParent<"SwitchboxOp">*/
                                              SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins WireBundle:$sourceBundle,
        I32Attr:$sourceChannel
  );
  let regions = (region AnyRegion:$rules);
  let summary = "Packet switched input connection";
  let description = [{
    This operation defines the slave port configuration (Lookup Table) for packet-switched routing,
    and it only accepts PacketRule operation inside its region (rules).
    The maximum number of entries in a Lookup Table is 4. Therefore, there can only be 4 PacketRule
    operations in the region.

    See the description of PacketRuleOp for a more concrete example.
  }];
  let assemblyFormat = [{ `(` $sourceBundle `:` $sourceChannel `)` regions attr-dict }];
  let extraClassDeclaration = [{
    int sourceIndex() { return sourceChannel(); }
    Port sourcePort() { return std::make_pair(sourceBundle(), sourceIndex()); }
  }];

  // let builders = [
  //   OpBuilder<"OpBuilder & builder, OperationState &result, "
  //             "WireBundle sourceBundle, int sourceChannel", [{
  //             build(builder, result,
  //                   sourceBundle,
  //                   APInt(32, sourceChannel));
  //             }]>
  // ];
}

def AIE_PacketRuleOp: AIE_Op<"rule", [HasParent<"PacketRulesOp">]> {
  let arguments = (
    ins I8Attr:$mask,
        I8Attr:$value,
        Index:$amsel
  );
  let summary = "Packet switched input connection";
  let description = [{
    This operation defines the rule (or configuration) of what arbiter and msel to use when a packet
    stream flows through the slave port that matches the ID after masking.

    Example:
      LUT ID  |  Mask     | ID          | Arbiter | Msel
       0      |  5'b11111 | 5'b00010    | 4       | 1
       1      |  5'b11011 | 5'b00001    | 3       | 2
       2      |           |             |         |
       3      |           |             |         |

    If a packet flow that has an ID of 2, it will be directed to the arbiter 4 with msel 1,
    If a packet flow that has an ID of 1 or 5, it will be directed to the arbiter 3 with msel 2,

    We encapsulate the configuration table as follows:
    Example:
      %a4_1 = AIE.amsel<4>(1)
      %a3_2 = AIE.amsel<3>(2)

      AIE.packetRules("ME" : 0) {
        AIE.rule(0x1F, 0x2, %a4_1)
        AIE.rule(0x1B, 0x1, %a3_2)
      }
  }];
  let extraClassDeclaration = [{
    int maskInt() { return mask(); }
    int valueInt() { return value(); }
  }];
  let assemblyFormat = [{
    `(` $mask `,` $value `,` $amsel `)` attr-dict
  }];
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "int mask, int value, Value amsel", [{
              build(builder, result,
                    builder.getI8IntegerAttr(mask),
                    builder.getI8IntegerAttr(value),
                    amsel);
              }]>
  ];
}

def AIE_PacketFlowOp: AIE_Op<"packet_flow", [/*HasParent<"SwitchboxOp">*/
                                             SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins I8Attr:$ID
  );
  let regions = (region AnyRegion:$ports);
  let summary = "Packet switched flow";
  let description = [{
    A logical packet switched flow between tiles.  During place and
    route, this is replaced by MasterSets and PacketRules inside
    switchboxes.
  }];
  let assemblyFormat = [{ `(` $ID `)` regions attr-dict }];
  let extraClassDeclaration = [{
    int IDInt() { return ID(); }
  }];
  // let builders = [
  //   OpBuilder<"OpBuilder & builder, OperationState &result, "
  //             "int ID", [{
  //             build(builder, result,
  //                   APInt(8, ID));
  //             }]>
  // ];
}

def AIE_PacketSourceOp: AIE_Op<"packet_source", [HasParent<"PacketFlowOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        I32Attr:$channel
  );
  let summary = "A sourceport";
  let description = [{
     A port which is the source of a packet-switched flow.
  }];
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int channelIndex() { return channel(); }
    Port port() { return std::make_pair(bundle(), channelIndex()); }
  }];
  // let builders = [
  //   OpBuilder<"OpBuilder & builder, OperationState &result, "
  //             "Value tile, WireBundle bundle, int channel", [{
  //             build(builder, result,
  //                   tile,
  //                   bundle,
  //                   APInt(32, channel));
  //             }]>
  // ];
}

def AIE_PacketDestOp: AIE_Op<"packet_dest", [HasParent<"PacketFlowOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        I32Attr:$channel
  );
  let summary = "A destination port";
  let description = [{
     A port which is the destination of a packet-switched flow.
  }];
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int channelIndex() { return channel(); }
    Port port() { return std::make_pair(bundle(), channelIndex()); }
  }];
  // let builders = [
  //   OpBuilder<"OpBuilder & builder, OperationState &result, "
  //             "Value tile, WireBundle bundle, int channel", [{
  //             build(builder, result,
  //                   tile,
  //                   bundle,
  //                   APInt(32, channel));
  //             }]>
  // ];
}

def S2MM0:  I32EnumAttrCase<"S2MM0", 0>;
def S2MM1:  I32EnumAttrCase<"S2MM1", 1>;
def MM2S0:  I32EnumAttrCase<"MM2S0", 2>;
def MM2S1:  I32EnumAttrCase<"MM2S1", 3>;

def DMAChan: I32EnumAttr<"DMAChan", "DMA Channel number",
  [S2MM0, S2MM1, MM2S0, MM2S1]> {

  let cppNamespace = "xilinx::AIE";
}

def AIE_DMABDOp: AIE_Op<"dmaBd", []> {
  let summary = "Declare a dma block descriptor op";
  let description = [{
    This operation describes the block descriptor buffer configuration. In particular, it specifies
    what buffer addresss to use, the transfer length, and the buffer type (A or B).
    
    This operation must be used in an MLIR block that lives inside a MemOp's region.
    The block descriptor specifies what lock to use and the buffer configuration.

    Example:

      // this defines a BD that uses lock %lck0 and buffer %buf0
      ^bd5:
        AIE.useLock(%lck, "Acquire", 0, 0)
        AIE.dmaBd(<$buf0 : memref<512xi32>, 0, 512>, 1)
        AIE.useLock(%lck, "Release", 1, 0)
        br ^bd6 // point to the next Block, which is also a different Block Descriptor

      ...

      // this defines a BD that does not use any lock
      ^bd8:
        AIE.dmaBd(<$buf1 : memref<64xi32>, 0, 64>, 0)

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

  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "Value buffer, int offset, int len, int AB", [{
              build(builder, result,
                    buffer,
                    builder.getI32IntegerAttr(offset),
                    builder.getI32IntegerAttr(len),
                    builder.getI32IntegerAttr(AB));
              }]>
  ];
}

def AIE_DMAStartOp: AIE_Op<"dmaStart", [ParentOneOf<["MemOp", "FuncOp"]>, Terminator]>,
                     Results<(outs I1:$valid)> {

  let summary = "An op to start DMA";
  let description = [{
    This operation declares a DMA channel to be used for data transfer. The operation can only be
    used inside a region of a MemOp.

    Example:
      %dma0 = AIE.dma("MM2S0")
      %dma1 = AIE.dma("S2MM1")

    The returned SSA values are used to specify the starting Block Descriptor.

    Example:
      AIE.terminator(^dma_block0, ^dma_block1, ^end)

      ^dma_block0:
        cond_br %dma0, ^bd5, ^end // want DMA MM2S0 to start BD5

      ^dma_block1:
        cond_br %dma1, ^bd1, ^end // want DMA S2MM1 to start BD1
      ...

      ^bd1:
        ...

      ^bd5:
        ...

      ...

      ^end:
        AIE.end


    Note that this piece code does not make any sense in terms of execution flow. We want to model
    the control flow of the BDs and the relationship with the DMA channels by using MLIR Block semantics.
    Will this be useful in other cases?
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
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "DMAChan channel, Block *dest, Block *chain", [{
              build(builder, result, builder.getIntegerType(1), channel, dest, chain);
              }]>
  ];
}

// MemOps are not actually Callable, but we want to inline code into them, so we have to 
// implement CallableOpInterface
def AIE_MemOp: AIE_Op<"mem", [CallableOpInterface]>,
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
  let verifier = [{ return ::verify(*this); }];
  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    int maxSizeInBytes() { return 32768; }
    // CallableOpInterface
    Region *getCallableRegion();
    ArrayRef<Type> getCallableResults();
  }];
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "Value tile", [{
              build(builder, result, builder.getIndexType(), tile);
              }]>
  ];
}

def AIE_LockOp: AIE_Op<"lock", []>, Results<(outs Index)> {

  let summary = "Declare a physical lock";
  let description = [{
    This operation creates a physical lock.

    Example:
      %lck = AIE.lock(%tile33, 7)

    This SSA value represents a lock that lives in the Memory module of Tile(3, 3) with a lockID of 7

  }];

  let arguments = (
    ins Index:$tile,
        Confined<I32Attr, [IntMinValue<0>, IntMaxValue<15>]>:$lockID
  );

  let assemblyFormat = [{
    `(` $tile `,` $lockID `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int getLockID() { return lockID(); }
  }];
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "Value tile, int lockID", [{
              build(builder, result,
                    builder.getIndexType(),
                    tile,
                    builder.getI32IntegerAttr(lockID));
              }]>
  ];
}

def Acquire: I32EnumAttrCase<"Acquire", 0>;
def Release: I32EnumAttrCase<"Release", 1>;

def LockAction: I32EnumAttr<"LockAction", "lock acquire/release",
  [Acquire, Release]> {

  let cppNamespace = "xilinx::AIE";
}

def AIE_UseLockOp: AIE_Op<"useLock", []> {
  let summary = "acquire/release lock op";
  let description = [{
    This operation uses a lock. A lock can be acquired with a value, or release with a value.
    This should be understood as a "blocking" operation.
  }];

  let arguments = (
    ins Index:$lock,
        Confined<I32Attr, [IntMinValue<0>, IntMaxValue<2>]>:$value,
        LockAction:$action,
        I32Attr:$timeout
  );

  let assemblyFormat = [{
    `(` $lock `,` $action `,` $value `,` $timeout `)` attr-dict
  }];

  let verifier = [{ return ::verify(*this); }];

  let extraClassDeclaration = [{
    bool acquire() { return (action() == LockAction::Acquire); }
    bool release() { return (action() == LockAction::Release); }
    int getLockValue() { return value(); }
    int getTimeout() { return timeout(); }
  }];
  // let builders = [
  //   OpBuilder<"OpBuilder & builder, OperationState &result, "
  //             "Value lock, int value, LockAction action, int timeout", [{
  //             build(builder, result, lock, APInt(32, value), action, APInt(32, timeout));
  //             }]>
  // ];
}

def AIE_BufferOp: AIE_Op<"buffer", []>, Results<(outs AnyMemRef)> {
  let summary = "Declare a buffer";
  let description = [{
    This operation instantiates a buffer that belongs to a Memory Module of a tile.

    Example:
      %tile33 = AIE.tile(3, 3)
      %buf = AIE.buffer(%tile33) : memref<256xi64>

    This SSA value represents a 64-bit buffer of 256 elements. The buffer is created in tile (3, 3).
  }];

  let arguments = (
    ins Index:$tile
  );

  let results = (outs AnyMemRef:$buffer);

  let assemblyFormat = [{
    `(` $tile `)` attr-dict `:`  type($buffer)
  }];
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

  let arguments = (
    ins I32Attr:$value
  );

  let assemblyFormat = [{
    `(` $value `)` attr-dict
  }];

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

  let assemblyFormat = [{
    $tokenName `(` $action `,` $value `)` attr-dict
  }];

  let extraClassDeclaration = [{
    bool acquire() { return (action() == LockAction::Acquire); }
    bool release() { return (action() == LockAction::Release); }
    int getTokenValue() { return value(); }
  }];

  // let builders = [
  //   OpBuilder<"OpBuilder & builder, OperationState &result, "
  //             "StringRef tokenName, int value, LockAction action", [{
  //             build(builder, result, tokenName, APInt(32, value), action);
  //             }]>
  // ];
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

def AIE_GetStreamOp: AIE_Op<"getStream", []>,
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

def AIE_PutStreamOp: AIE_Op<"putStream", []> {
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

def AIE_GetCascadeOp: AIE_Op<"getCascade", []>,
                      Results<(outs AnyI<384>)> {
  let summary = "An op to read from a cascading stream from a neighboring core";
  let description = [{
    An op to read from a cascading stream from a neighboring core.
  }];

  let results = (outs AnyI<384>:$cascadeValue);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($cascadeValue)
  }];
}

def AIE_PutCascadeOp: AIE_Op<"putCascade", []> {
  let summary = "An op to write to a cascading stream from a neighboring core";
  let description = [{
    An op to write to a cascading stream from a neighboring core.
  }];

  let arguments = (
    ins AnyI<384>:$cascadeValue
  );

  let assemblyFormat = [{
    `(` $cascadeValue `:` type($cascadeValue) `)` attr-dict
  }];
}

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
  }];

  let assemblyFormat = [{
    `[` $width `]` `[` $height `]` attr-dict
  }];

  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "int width, int height", [{
              build(builder, result, builder.getIndexType(),
                    builder.getI32IntegerAttr(width),
                    builder.getI32IntegerAttr(height));
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

  let assemblyFormat = [{
    `(` $sourceHerd `,` $destHerd `,` $distX `,` $distY `)` attr-dict
  }];

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

  let assemblyFormat = [{
    `(` $start `,` $end `,` $stride `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int getStartValue()  { return start(); }
    int getEndValue()    { return end(); }
    int getStrideValue() { return stride(); }
  }];

  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "int start, int end, int stride", [{
              build(builder, result, builder.getIndexType(),
                    builder.getI32IntegerAttr(start),
                    builder.getI32IntegerAttr(end),
                    builder.getI32IntegerAttr(stride));
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

  let assemblyFormat = [{
    `(` $startHerd `,` $iterX `,` $iterY `)` attr-dict
  }];

  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "Value startHerd, Value iterX, Value iterY", [{
              build(builder, result, builder.getIndexType(),
                    startHerd, iterX, iterY);
              }]>
  ];
}

