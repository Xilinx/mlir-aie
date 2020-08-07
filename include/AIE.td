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


def Port : TypeConstraint<Index.predicate, "port">;

def AIE_SwitchboxOp: AIE_Op<"switchbox", [SingleBlockImplicitTerminator<"EndswitchOp">]>,
                     Results<(outs AnyType)> {
  let arguments = (
    ins Index:$tile
  );

  let summary = "Declare a switch";
  let description = [{
    A switch
  }];
  let regions = (region AnyRegion:$connections);
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseSwitchboxOp(parser, result); }];
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

def AIE_ShimSwitchboxOp: AIE_Op<"shimswitchbox", [SingleBlockImplicitTerminator<"EndswitchOp">]>,
                         Results<(outs AnyType)> {
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
1 Port for control packet for ME Shim
register access
1 Port for response to access for ME Shim
registers
1 Port for trace packet from ME Shim

  }];
  let regions = (region AnyRegion:$connections);
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseShimSwitchboxOp(parser, result); }];
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

  let summary = "Declare a core";
  let description = [{
    A core
  }];

  let regions = (region AnyRegion:$body);
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseCoreOp(parser, result); }];
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
    int colIndex() { return col().getZExtValue(); }
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
    int colIndex() { return col().getZExtValue(); }
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
    The "aie.connect" operation represents a programmed circuit-switched connection in a stream switch.  It associates a source bundle and source channel with a destination bundle and a destination channel.  The "aie.connect" operation must exist within an "aie.switchbox" or "aie.shimswitchbox" operation.  All of the "aie.connect" operations in a switchbox must have a different destination.  All of the "aie.connect" operations must also have a destination which is different from all of the "aie.masterset" operations in the same switchbox.
    Example:

    %7 = aie.switchbox(1, 1) {
      aie.connect<"West" : 0, "ME" : 1>
    }
  }];
  let assemblyFormat = [{
    `<` $sourceBundle `:` $sourceChannel `,` $destBundle `:` $destChannel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int sourceIndex() { return sourceChannel().getZExtValue(); }
    int destIndex() { return destChannel().getZExtValue(); }
    Port sourcePort() { return std::make_pair(sourceBundle(), sourceIndex()); }
    Port destPort() { return std::make_pair(destBundle(), destIndex()); }
  }];
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "WireBundle sourceBundle, int sourceChannel, "
              "WireBundle destBundle, int destChannel", [{
              build(builder, result,
                    sourceBundle,
                    APInt(32, sourceChannel),
                    destBundle,
                    APInt(32, destChannel));
              }]
    >
  ];
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
  The "aie.flow" operation represents a circuit switched connection between two endpoints, usually "aie.core" operations.  During routing, this is
  replaced by "aie.connect" operations which represent the programmed connections inside a switchbox, along with "aie.wire" operations which represent physical connections between switchboxes and other components.

Example:
    %00 = aie.core(0, 0)
    %11 = aie.core(1, 1)
    %01 = aie.core(0, 1)
    aie.flow(%00, "DMA" : 0, %11, "ME" : 1)

  }];
  let assemblyFormat = [{
    `(` $source `,` $sourceBundle `:` $sourceChannel `,` $dest `,` $destBundle `:` $destChannel `)` attr-dict
  }];
  let extraClassDeclaration = [{
    int sourceIndex() { return sourceChannel().getZExtValue(); }
    int destIndex() { return destChannel().getZExtValue(); }
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

def AIE_MasterSetOp: AIE_Op<"masterset", [HasParent<"SwitchboxOp">]>, Results<(outs Index)> {
  let arguments = (
    ins I32Attr:$arbiter,
        WireBundle:$destBundle,
        I32Attr:$destChannel
  );
  let summary = "Packet switched input connection";
  let description = [{
    A Packet switched connection inside a switchbox.
  }];
  let assemblyFormat = [{
    `(` $arbiter `,` $destBundle `:` $destChannel `)` attr-dict
  }];
  let extraClassDeclaration = [{
    int arbiterIndex() { return arbiter().getZExtValue(); }
    int destIndex() { return destChannel().getZExtValue(); }
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
    The "aie.wire" operation represents a physical set of connections between components in a Versal device.  Typically, these components are switches, represented by an "aie.switchbox" operation, and cores, represented by an [aie.core](#aiecore-aiecoreop) operation.
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
                                              SingleBlockImplicitTerminator<"EndswitchOp">]> {
  let arguments = (
    ins WireBundle:$sourceBundle,
        I32Attr:$sourceChannel
  );
  let regions = (region AnyRegion:$rules);
  let summary = "Packet switched input connection";
  let description = [{
    A Packet switched connection inside a switchbox.
  }];
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parsePacketRulesOp(parser, result); }];
  // let assemblyFormat = [{
  // 	`(` $sourceBundle `:` $sourceChannel `)` attr-dict
  // }];
  let extraClassDeclaration = [{
    int sourceIndex() { return sourceChannel().getZExtValue(); }
    Port sourcePort() { return std::make_pair(sourceBundle(), sourceIndex()); }
  }];
}
def AIE_PacketRuleOp: AIE_Op<"rule", [HasParent<"PacketRulesOp">]> {
  let arguments = (
    ins I8Attr:$mask,
        I8Attr:$value,
        Index:$masterset
  );
  let summary = "Packet switched input connection";
  let description = [{
    A Packet switched connection inside a switchbox.
  }];
  let extraClassDeclaration = [{
    int maskInt() { return mask().getZExtValue(); }
    int valueInt() { return value().getZExtValue(); }
    MasterSetOp &masterSetOp() {
      // FIXME: Verify this..
      Operation *op = masterset().getDefiningOp();
      assert(op);
      auto masterSetOp = dyn_cast_or_null<MasterSetOp>(op);
      //assert(masterSetOp);
    }
  }];
  let assemblyFormat = [{
    `(` $mask `,` $value `,` $masterset `)` attr-dict
  }];
}
def AIE_PacketFlowOp: AIE_Op<"packet_flow", [/*HasParent<"SwitchboxOp">*/
                                             SingleBlockImplicitTerminator<"EndswitchOp">]> {
  let arguments = (
    ins I8Attr:$ID
  );
  let regions = (region AnyRegion:$ports);
  let summary = "Packet switched flow";
  let description = [{
    A logical packet switched flow between cores.  During place and
    route, this is replaced by MasterSets and PacketRules inside
    switchboxes.
  }];
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parsePacketFlowOp(parser, result); }];
  // let assemblyFormat = [{
  // 	`(` $sourceBundle `:` $sourceChannel `)` attr-dict
  // }];
  let extraClassDeclaration = [{
    int IDInt() { return ID().getZExtValue(); }
  }];
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "int ID", [{
              build(builder, result,
                    APInt(8, ID));
              }]>
  ];
}

def AIE_PacketSourceOp: AIE_Op<"packet_source", [HasParent<"PacketFlowOp">]> {
  let arguments = (
    ins Index:$core,
        WireBundle:$bundle,
        I32Attr:$channel
  );
  let summary = "A sourceport";
  let description = [{
     A port which is the source of a flow.
  }];
  let assemblyFormat = [{
    `<` $core `,` $bundle `:` $channel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int channelIndex() { return channel().getZExtValue(); }
  }];
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "Value core, WireBundle bundle, int channel", [{
              build(builder, result,
                    core,
                    bundle,
                    APInt(32, channel));
              }]>
  ];
}

def AIE_PacketDestOp: AIE_Op<"packet_dest", [HasParent<"PacketFlowOp">]> {
  let arguments = (
    ins Index:$core,
        WireBundle:$bundle,
        I32Attr:$channel
  );
  let summary = "A destination port";
  let description = [{
     A port which is the destination  of a flow.
  }];
  let assemblyFormat = [{
    `<` $core `,` $bundle `:` $channel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int channelIndex() { return channel().getZExtValue(); }
  }];
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "Value core, WireBundle bundle, int channel", [{
              build(builder, result,
                    core,
                    bundle,
                    APInt(32, channel));
              }]>
  ];
}

def AIE_EndswitchOp: AIE_Op<"endswitch", [Terminator
                                          /*,Or<[HasParent<"SwitchboxOp">,
                                                 HasParent<"ShimSwitchboxOp">]>*/
                                         ]> {
  let summary = "end of a switch";
  let description = [{
    Terminator for switchbox
  }];
}

def AIE_EndarbiterOp: AIE_Op<"endarbiter", [Terminator
                                            /*,Or<[HasParent<"SwitchboxOp">,
                                                   HasParent<"ShimSwitchboxOp">]>*/
                                           ]> {
  let summary = "end of a arbiter";
  let description = [{
    Terminator for arbiterbox
  }];
}

def AIE_EndOp: AIE_Op<"end", [Terminator]> {
  let summary = "end op";
  let description = [{
    Terminator for AIE Ops that have multi-block region
  }];
  let assemblyFormat = [{
    attr-dict
  }];
}

def AIE_TileOp: AIE_Op<"tile", []>, Results<(outs Index)> {
  let arguments = (
    ins I32Attr:$col,
        I32Attr:$row
  );

  let summary = "Declare an AIE tile";
  let description = [{
    An AIE tile
  }];

  let extraClassDeclaration = [{
    int getNumSourceConnections(WireBundle bundle);
    int getNumDestConnections(WireBundle bundle);
    int colIndex() { return col().getZExtValue(); }
    int rowIndex() { return row().getZExtValue(); }
    bool isMemWest() { return ((rowIndex() % 2) == 0); };
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

def S2MM0:  I32EnumAttrCase<"S2MM0", 0>;
def S2MM1:  I32EnumAttrCase<"S2MM1", 1>;
def MM2S0:  I32EnumAttrCase<"MM2S0", 2>;
def MM2S1:  I32EnumAttrCase<"MM2S1", 3>;

def DMAChan: I32EnumAttr<"DMAChan", "DMA Channel number",
  [S2MM0, S2MM1, MM2S0, MM2S1]> {

  let cppNamespace = "xilinx::AIE";
}

def AIE_DMABDOp: AIE_Op<"dmaBd", []> {
  let summary = "Declare a dma op";
  let description = [{
    TODO
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
    int getOffsetValue() { return offset().getZExtValue(); }
    int getLenValue() { return len().getZExtValue(); }
    bool isA() { return (AB().getZExtValue() == 0); }
    bool isB() { return (AB().getZExtValue() == 1); }
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

def AIE_DMAStartOp: AIE_Op<"dmaStart", [HasParent<"MemOp">]>,
                     Results<(outs I1:$valid)> {

  let summary = "An op to start DMA";
  let description = [{
  }];

  let arguments = (
    ins DMAChan:$dmaChan
  );

  let assemblyFormat = [{
    `(` $dmaChan `)` attr-dict
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
              "DMAChan channel", [{
              build(builder, result, builder.getIntegerType(1), channel);
              }]>
  ];
}

def AIE_MemOp: AIE_Op<"mem", []>,
                     Results<(outs Index)> {

  let summary = "Declare a memory op";
  let description = [{
    TODO
  }];

  let arguments = (
    ins Index:$tile
  );

  let regions = (region AnyRegion:$body);
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseMemOp(parser, result); }];
  let verifier = [{ return ::verify(*this); }];
  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    int maxSizeInBytes() { return 32768; }
  }];
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "Value tile", [{
              build(builder, result, builder.getIndexType(), tile);
              }]>
  ];
}

def AIE_TerminatorOp: AIE_Op<"terminator", [Terminator]> {
  let summary = "A multi-successor terminator op";
  let description = [{
    TODO
  }];

  let successors = (successor VariadicSuccessor<AnySuccessor>:$dests);
  let assemblyFormat = [{
    `(` $dests `)` attr-dict
  }];
}

def AIE_LockOp: AIE_Op<"lock", []>, Results<(outs Index)> {

  let summary = "Declare a physical lock";
  let description = [{
    TODO
  }];

  let arguments = (
    ins Index:$tile,
        Confined<I32Attr, [IntMinValue<0>, IntMaxValue<15>]>:$lockID
  );

  let assemblyFormat = [{
    `(` $tile `,` $lockID `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int getLockID() { return lockID().getZExtValue(); }
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
    TODO
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
    int getLockValue() { return value().getZExtValue(); }
    int getTimeout() { return timeout().getZExtValue(); }
  }];
  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "Value lock, int value, LockAction action, int timeout", [{
              build(builder, result, lock, APInt(32, value), action, APInt(32, timeout));
              }]>
  ];
}

def AIE_BufferOp: AIE_Op<"buffer", []>, Results<(outs AnyMemRef)> {
  let summary = "Declare a buffer";
  let description = [{
    TODO
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
    TODO
  }];

  let arguments = (
    ins I32Attr:$value
  );

  let assemblyFormat = [{
    `(` $value `)` attr-dict
  }];

  let extraClassDeclaration = [{
    int getTokenValue() { return value().getZExtValue(); }
  }];
}

def AIE_UseTokenOp: AIE_Op<"useToken", []> {
  let summary = "acquire/release a logical lock";
  let description = [{
    TODO
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
    int getTokenValue() { return value().getZExtValue(); }
  }];

  let builders = [
    OpBuilder<"OpBuilder & builder, OperationState &result, "
              "StringRef tokenName, int value, LockAction action", [{
              build(builder, result, tokenName, APInt(32, value), action);
              }]>
  ];
}

def AIE_MemcpyOp: AIE_Op<"memcpy", []> {
  let summary = "A memcpy op";
  let description = [{
    TODO
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
    int getAcquireTokenValue() { return acqValue().getZExtValue(); }
    int getReleaseTokenValue() { return relValue().getZExtValue(); }
    int getSrcOffsetValue() { return srcOffset().getZExtValue(); }
    int getDstOffsetValue() { return dstOffset().getZExtValue(); }
    int getSrcLenValue() { return srcLen().getZExtValue(); }
    int getDstLenValue() { return dstLen().getZExtValue(); }
  }];
}

def AIE_GetStreamOp: AIE_Op<"getStream", []>,
                 Results<(outs AnyTypeOf<[F32, I32, I<128>]>)> {
  let summary = "An op to read from a stream channel/port of a switchbox";
  let description = [{
    TODO
  }];

  let arguments = (
    ins Confined<I32Attr, [IntMinValue<0>, IntMaxValue<1>]>:$channel
  );
  let results = (outs AnyTypeOf<[F32, I32, I<128>]>:$streamValue);

  let assemblyFormat = [{
    `(` $channel `)` attr-dict `:` type($streamValue)
  }];
  let extraClassDeclaration = [{
    bool isWideStream() { return streamValue().getType().isInteger(128); }
    bool isFloatStream() { return streamValue().getType().isa<FloatType>(); }
    int getChannelValue() { return channel().getZExtValue(); }
  }];
}

def AIE_PutStreamOp: AIE_Op<"putStream", []> {
  let summary = "An op to write to a stream channel/port of a switchbox";
  let description = [{
    TODO
  }];

  let arguments = (
    ins Confined<I32Attr, [IntMinValue<0>, IntMaxValue<1>]>:$channel,
        AnyTypeOf<[F32, I32, I<128>]>:$streamValue
  );

  let assemblyFormat = [{
    `(` $channel `,` $streamValue `:` type($streamValue) `)` attr-dict
  }];
  let extraClassDeclaration = [{
    bool isWideStream() { return streamValue().getType().isInteger(128); }
    bool isFloatStream() { return streamValue().getType().isa<FloatType>(); }
    int getChannelValue() { return channel().getZExtValue(); }
  }];
}

def AIE_GetCascadeOp: AIE_Op<"getCascade", []>,
                      Results<(outs AnyI<384>)> {
  let summary = "An op to read from a cascading stream from a neighboring core";
  let description = [{
    TODO
  }];

  let results = (outs AnyI<384>:$cascadeValue);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($cascadeValue)
  }];
}

def AIE_PutCascadeOp: AIE_Op<"putCascade", []> {
  let summary = "An op to write to a cascading stream from a neighboring core";
  let description = [{
    TODO
  }];

  let arguments = (
    ins AnyI<384>:$cascadeValue
  );

  let assemblyFormat = [{
    `(` $cascadeValue `:` type($cascadeValue) `)` attr-dict
  }];
}

// Fairly high-level logical ops: ComputeOp, MemoryOp, NetOp

def AIE_ComputeOp: AIE_Op<"compute", []>, Results<(outs Index)> {
  let summary = "A logical representation of a compute task";
  let description = [{
    TODO
  }];

  let assemblyFormat = [{
    `(` `)` attr-dict
  }];
}

def AIE_MemoryOp: AIE_Op<"memory", []>, Results<(outs Index)> {
  let summary = "A logical representation of a memory buffer accessed by compute tasks";
  let description = [{
    TODO
  }];

  let arguments = (
    ins Optional<AnyMemRef>:$buffer
  );

  let assemblyFormat = [{
    `(` $buffer type($buffer) `)` attr-dict
  }];
}

def AIE_NetOp: AIE_Op<"net", []> {
  let summary = "A logical directional connection of logical blocks "
                "(compute->compute, memory->compute, or compute->memory)";
  let description = [{
    TODO
  }];

  let arguments = (
    ins Index:$source,
        Index:$dest
  );

  let assemblyFormat = [{
    `(` $source `,` $dest `)` attr-dict
  }];

}
