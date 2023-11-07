//===- AIE.td ----------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIEX_OPS
#define AIEX_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "aie/Dialect/AIE/IR/AIEInterfaces.td"

def AIEX_Dialect : Dialect {
  let name = "AIEX";
  let cppNamespace = "::xilinx::AIEX";
  let description = [{

    This is a dialect for experimental work related to AIEngine processors.
    The expectation is that new ideas can be developed here before migration
    to the more mature AIE dialect.

  }];
  let useDefaultTypePrinterParser = 1;
}


class AIEX_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIEX_Dialect, mnemonic, traits>;


def AIE_GetTileOp: AIEX_Op<"getTile", []>, Results<(outs Index:$result)> {
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

def AIE_ConnectionOp: AIEX_Op<"connection", []> {
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
    int sourceIndex() { return getSourceChannel(); }
    int destIndex() { return getDestChannel(); }
  }];
}

def AIE_MulticastOp: AIEX_Op<"multicast", [SingleBlockImplicitTerminator<"AIE::EndOp">]> {
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
      AIEX.multicast(%70, "DMA" : 0){
        AIEX.multi_dest<%73, "DMA" : 0>
        AIEX.multi_dest<%74, "DMA" : 0>
        AIEX.multi_dest<%63, "DMA" : 0>
        AIEX.multi_dest<%64, "DMA" : 0>
      }
    ```
  }];
  let assemblyFormat = [{ `(` $tile `,` $bundle `:` $channel `)` regions attr-dict }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    int channelIndex() { return getChannel(); }
    AIE::Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_MultiDestOp: AIEX_Op<"multi_dest", [HasParent<"MulticastOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        I32Attr:$channel
  );
  let summary = "A destination port of multicast flow";
  let description = [{
    An object representing the destination of a multicast flow. This must exist
    within an [AIEX.multicast] operation. There can be multiple destinations within an
    AIEX.multicast Op.

    See [AIEX.multicast]for an example.
  }];
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int channelIndex() { return getChannel(); }
    AIE::Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_BroadcastPacketOp: AIEX_Op<"broadcast_packet", [SingleBlockImplicitTerminator<"AIE::EndOp">]> {
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
    int channelIndex() { return getChannel(); }
    AIE::Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_BPIDOp: AIEX_Op<"bp_id", [SingleBlockImplicitTerminator<"AIE::EndOp">]> {
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
    int IDInt() { return getID(); }
  }];
}

def AIE_BPDestOp: AIEX_Op<"bp_dest", [HasParent<"BPIDOp">]> {
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
    int channelIndex() { return getChannel(); }
    AIE::Port port() { return {getBundle(), channelIndex()}; }
  }];
}

def AIE_TokenOp: AIEX_Op<"token", [Symbol]> {
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
    int getTokenValue() { return getValue(); }
  }];
}

def AIE_UseTokenOp: AIEX_Op<"useToken", []> {
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
    bool acquire() { return (getAction() == AIE::LockAction::Acquire); }
    bool release() { return (getAction() == AIE::LockAction::Release); }
    int getTokenValue() { return getValue(); }
  }];
}

def AIE_MemcpyOp: AIEX_Op<"memcpy", []> {
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
    int getAcquireTokenValue() { return getAcqValue(); }
    int getReleaseTokenValue() { return getRelValue(); }
    int getSrcOffsetValue() { return getSrcOffset(); }
    int getDstOffsetValue() { return getDstOffset(); }
    int getSrcLenValue() { return getSrcLen(); }
    int getDstLenValue() { return getDstLen(); }
  }];
}




/// Experimental Herd operations
def AIE_HerdOp: AIEX_Op<"herd", []>, Results<(outs Index)> {
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
    int getHerdWidth()   { return getWidth(); }
    int getHerdHeight()  { return getHeight(); }
    int getNumAIETiles() { return getHerdWidth() * getHerdHeight(); }
    mlir::StringAttr name() {
      if(auto attr = getOperation()->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName()))
        return attr;
      emitOpError("does not have '") << mlir::SymbolTable::getSymbolAttrName() << "' attribute specified";
      llvm::report_fatal_error("couldn't get name");
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

def AIE_PlaceOp: AIEX_Op<"place", []> {
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
    int getDistXValue() { return getDistX(); }
    int getDistYValue() { return getDistY(); }
  }];
}

def AIE_RouteOp: AIEX_Op<"route", []> {
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
    int getSourceChannelValue()  { return getSourceChannel(); }
    int getDestChannelValue()  { return getDestChannel(); }
  }];
}

def AIE_IterOp: AIEX_Op<"iter", []>, Results<(outs Index)> {
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
    int getStartValue()  { return getStart(); }
    int getEndValue()    { return getEnd(); }
    int getStrideValue() { return getStride(); }
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

def AIE_SelectOp: AIEX_Op<"select", []>, Results<(outs Index)> {
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
    OpBuilder<(ins "mlir::Value":$startHerd, "mlir::Value":$iterX, "mlir::Value":$iterY), [{
      build($_builder, $_state, $_builder.getIndexType(),
            startHerd, iterX, iterY);
    }]>
  ];
}

#endif // AIEX_OPS