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

include "aie/Dialect/AIE/IR/AIEAttrs.td"
include "aie/Dialect/AIE/IR/AIEInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AIEX_Dialect : Dialect {
  let name = "aiex";
  let cppNamespace = "::xilinx::AIEX";
  let description = [{

    This is a dialect for experimental work related to AIEngine processors.
    The expectation is that new ideas can be developed here before migration
    to the more mature AIE dialect.

  }];
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
        AIEI32Attr:$sourceChannel,
        Index:$dest,
        WireBundle:$destBundle,
        AIEI32Attr:$destChannel
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
        AIEI32Attr:$channel
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
      aiex.multicast(%70, "DMA" : 0){
        aiex.multi_dest<%73, "DMA" : 0>
        aiex.multi_dest<%74, "DMA" : 0>
        aiex.multi_dest<%63, "DMA" : 0>
        aiex.multi_dest<%64, "DMA" : 0>
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
        AIEI32Attr:$channel
  );
  let summary = "A destination port of multicast flow";
  let description = [{
    An object representing the destination of a multicast flow. This must exist
    within an [aiex.multicast] operation. There can be multiple destinations within an
    aiex.multicast Op.

    See [aiex.multicast]for an example.
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
        AIEI32Attr:$channel
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
  let arguments = (ins AIEI8Attr:$ID);
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
        AIEI32Attr:$channel
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
  let arguments = (ins AIEI32Attr:$value);
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
        AIEI32Attr:$value,
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
        AIEI32Attr:$acqValue,
        AIEI32Attr:$relValue,
        Index:$srcTile,
        AnyMemRef:$srcBuf,
        AIEI32Attr:$srcOffset,
        AIEI32Attr:$srcLen,
        Index:$dstTile,
        AnyMemRef:$dstBuf,
        AIEI32Attr:$dstOffset,
        AIEI32Attr:$dstLen
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
    ins AIEI32Attr:$width,
        AIEI32Attr:$height
  );
  let extraClassDeclaration = [{
    int getHerdWidth() { return getWidth(); }
    int getHerdHeight() { return getHeight(); }
    int getNumAIETiles() { return getHerdWidth() * getHerdHeight(); }
    mlir::StringAttr name() {
      if (auto attr = getOperation()->getAttrOfType<mlir::StringAttr>(
              mlir::SymbolTable::getSymbolAttrName()))
        return attr;
      emitOpError("does not have '")
          << mlir::SymbolTable::getSymbolAttrName() << "' attribute specified";
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
        AIEI32Attr:$distX,
        AIEI32Attr:$distY
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
        AIEI32Attr:$sourceChannel,
        Index:$destHerds,
        WireBundle:$destBundle,
        AIEI32Attr:$destChannel
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
    ins AIEI32Attr:$start,
        AIEI32Attr:$end,
        AIEI32Attr:$stride
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

def AIE_NpuDmaMemcpyNdOp: AIEX_Op<"npu.dma_memcpy_nd", [
    AttrSizedOperandSegments,
    MyOffsetSizeAndStrideOpInterface
  ]> {
  let summary = "half dma operator";

  let description = [{
    An nd half dma operator.

    Programs a DMA on coordinates (`x`, `y`) to access a memory `memref` with an access
    pattern specified by `offsets`, `sizes` and `strides` or `static_offsets`, `static_sizes`
    and `static_strides`. The operator references the target channel through the `metadata`
    symbol and specifies a descriptor `id` to be used, which will become the `bd_id` to be used
    when lowered further. The `issue_token` attribute specifies whether the execution of this
    operation should issue a token which can be received and read for synchronization purposes.
    This `issue_token` attribute is set to `false` by default for `MM2S` for backward compatibility
    and **is always set to true for** `S2MM` channels.
  }];

  let arguments = (
    ins I64Attr:$x,
        I64Attr:$y,
        AnyMemRef:$memref,
        // NOTE: these are in reverse order: offset3, offset2, ...
        Variadic<I64>:$offsets,
        Variadic<I64>:$sizes,
        Variadic<I64>:$strides,
        ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_offsets,
        ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_sizes,
        ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<3>]>:$static_strides,
        FlatSymbolRefAttr:$metadata,
        I64Attr:$id,
        DefaultValuedOptionalAttr<BoolAttr, "false">:$issue_token
  );

  let assemblyFormat = [{
    `(` $x `,` $y `,` $memref ``
    custom<DynamicIndexList>($offsets, $static_offsets) ``
    custom<DynamicIndexList>($sizes, $static_sizes) ``
    custom<DynamicIndexList>($strides, $static_strides) `)`
    attr-dict `:` type($memref)
  }];

  let extraClassDeclaration = [{
    static unsigned getOffsetSizeAndStrideStartOperandIndex();
    static std::array<unsigned, 3> getArrayAttrMaxRanks();
  }];

  let extraClassDefinition = [{
    unsigned $cppClass::getOffsetSizeAndStrideStartOperandIndex() { return 1; }
    std::array<unsigned, 3> $cppClass::getArrayAttrMaxRanks() { return {4, 4, 3}; }
  }];

  let hasVerifier = 1;
}

def AIE_NpuDmaWaitOp: AIEX_Op<"npu.dma_wait", []> {
  let summary = "Blocking operation to wait for a DMA to complete execution.";
  let description = [{
    The NpuDmaWaitOp blocks until the DMA referenced through `symbol` completes execution
    and issues a task-complete-token.

    Example:
    ```mlir
      ...
      aie.objectfifo @out0(%tile_0_1, {%tile_0_0}, 4 : i32) : !aie.objectfifo<memref<32x32xi32>>
      ...
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[1, 1, 0, 0][1, 1, 32, 32][1, 1, 64]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<32x64xi32>
      ...
      aiex.npu.dma_wait { symbol = @out0 }
    ```
    Here, we have an objectfifo with symbol name `out0`, which is then referenced in the
    `npu.dma_memcpy_nd` operation as the target for the respective DMA operation. Afterwards,
    an `npu.dma_wait` operation references the same symbol to block until the respective DMA
    has executed all of its tasks.
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$symbol
  );
  let assemblyFormat = [{
    attr-dict
  }];
  let hasVerifier = 1;
}

// Write RTP
def AIE_NpuWriteRTPOp: AIEX_Op<"npu.rtp_write", []> {
  let summary = "rtp write operator";
  let arguments = (
    ins StrAttr:$buffer_sym_name,
        UI32Attr:$col,
        UI32Attr:$row,
        UI32Attr:$index,
        I32Attr:$value
  );
  let results = (outs );
  let assemblyFormat = [{ `(` $col `,` $row `,` $index `,` $value `)` attr-dict 
  }];
  let description = [{
    rtp write operator
  }];
}

// Push BD to Queue
def AIE_NpuShimTilePushQueueOp: AIEX_Op<"npu.shimtile_push_queue", []> {
  let summary = "bd queue push operator";
  let arguments = (
    ins FlatSymbolRefAttr:$metadata,
        BoolAttr:$issue_token,
        I32Attr:$repeat_count,
        I32Attr:$bd_id
  );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
  let hasVerifier = 1;
  let description = [{
    bd queue push operator
  }];
}

// WRITE32
def AIE_NpuWrite32Op: AIEX_Op<"npu.write32", []> {
  let summary = "write32 operator";
  let arguments = (
    ins I32Attr:$column,
        I32Attr:$row,
        UI32Attr:$address,
        UI32Attr:$value
  );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
  let description = [{
    write32 operator
  }];
}

// OP_SYNC
def AIE_NpuSyncOp: AIEX_Op<"npu.sync", []> {
  let summary = "sync operator";
  let arguments = (
    ins I32Attr:$column,
        I32Attr:$row,
        I32Attr:$direction,
        I32Attr:$channel,
        I32Attr:$column_num,
        I32Attr:$row_num
  );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
  let description = [{
    tct sync operator
  }];
}

// WRITEBD_EXTEND_SHIMTILE
def AIE_NpuWriteBdExShimTileOp: AIEX_Op<"npu.writebd_shimtile", []> {
  let summary = "dma operator";
  let arguments = (
    ins I32Attr:$column,
        I32Attr:$column_num,
        I32Attr:$ddr_id,
        I32Attr:$bd_id,
        I32Attr:$buffer_length,
        I32Attr:$buffer_offset,
        I32Attr:$enable_packet,
        I32Attr:$out_of_order_id,
        I32Attr:$packet_id,
        I32Attr:$packet_type,
        I32Attr:$d0_size,
        I32Attr:$d0_stride,
        I32Attr:$d1_size,
        I32Attr:$d1_stride,
        I32Attr:$d2_stride,
        I32Attr:$iteration_current,
        I32Attr:$iteration_size,
        I32Attr:$iteration_stride,
        I32Attr:$next_bd,
        I32Attr:$use_next_bd,
        I32Attr:$valid_bd,
        I32Attr:$lock_rel_val,
        I32Attr:$lock_rel_id,
        I32Attr:$lock_acq_enable,
        I32Attr:$lock_acq_val,
        I32Attr:$lock_acq_id
  );
  let results = (outs );
  let assemblyFormat = [{ attr-dict }];
  let hasVerifier = 1;
  let description = [{
    writebd_shimtile operator
  }];
}

#endif // AIEX_OPS