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
include "mlir/IR/CommonAttrConstraints.td"

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

def AIE_RuntimeSequenceOp : AIEX_Op<"runtime_sequence", [NoTerminator, HasParent<"AIE::DeviceOp">]> {
  let summary = "Program the configuration co-processor of the AI Engine array";
  let description = [{
    Instructions in this operation allow for runtime (re-)configuration of the AI Engine array, such as configuring data movement buffer descriptors.
    These instructions will execute on the configuration co-processor of the AI Engine array.

    Typically, these instructions include configuring the data transfers between host and AIE array on the shims.
    The input arguments are arguments passed in from the host at kernel invocation time. This may include buffers on the host.
  }];
  let arguments = (
    ins OptionalAttr<SymbolNameAttr>:$sym_name
  );
  let regions = (region
    AnyRegion:$body
  );
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def AIE_NpuDmaMemcpyNdOp: AIEX_Op<"npu.dma_memcpy_nd", [
    AttrSizedOperandSegments,
    MyOffsetSizeAndStrideOpInterface
  ]> {
  let summary = "half DMA operator";

  let description = [{
    An n-dimensional half DMA operator.

    Programs a DMA on coordinates (`x`, `y`) to access a memory `memref` with an access
    pattern specified by `offsets`, `sizes` and `strides` or `static_offsets`, `static_sizes`
    and `static_strides`. The operator references the target channel through the `metadata`
    symbol and specifies a descriptor `id` to be used, which will become the `bd_id` to be used
    when lowered further. The `issue_token` attribute specifies whether the execution of this
    operation should issue a token which can be received and read for synchronization purposes.
    This `issue_token` attribute is set to `false` by default for `MM2S` for backward compatibility
    and **is always set to true for** `S2MM` channels.

    #### `metadata` -- Specifying Tile, Channel, Direction and Linking a `dma_memcpy_nd` to its Other Half

    The `metadata` attribute must point to a symbol referencing a 
    [`aie.shim_dma_allocation` operation](AIE.html#aiedma_bd-xilinxaiedmabdop).
    The tile coordinates of the DMA to configure, the channel number and the direction (`MM2S` or `S2MM`) are taken from this operation.

    To connect the DMA to its other half (i.e. a `MM2S` DMA to its receiving end and a `S2MM` to the sending end), 
    the user must configure a flow (`aie.flow`) between the tile and channel referenced in the `aie.shim_dma_allocation` and the corresponding other end.

    When using ObjectFIFOs, the `aie.shim_dma_allocation` operations and the `aie.flows` are generated automatically.
    The symbol of the `aie.objectfifo` (create) operation can be used directly in `metadata` in this case.

    #### Notes on Synchronization and Reusing Buffer Descriptor IDs

    When the `dma_memcpy_nd` operation executes, it immediately reprograms the buffer descriptor with ID `bd_id` on tile (`x`, `y`), even if that buffer descriptor is currently executing.
    Without proper synchronization, this inevitably leads to nondeterministic results.

    Programming a buffer descriptor that is not currently executing is harmless. 
    Thus, the first `dma_memcpy_nd` call for each `bd_id` requires no synchronization.

    However, if you wish to later re-use a `bd_id` on the same tile, you must wait for the previous buffer descriptor to complete. 
    The `sync` or `dma_wait` operations can be used for this.

    `sync` blocks until it receives a _task completion token_ (TCT). 
    To properly synchronize, you must thus configure your BD to issue a TCT using the `issue_token` attribute, then wait on that token before reusing the BD.

    `dma_wait` is a convenience operation that lowers to the corresponding `sync` operation for the refrenced symbol.

    Note that if you have multiple concurrently running BDs and you can reason one BD will always complete after all others, it is not strictly necessary to issue and wait on the TC token for every BD.
    For example, if you have input and output BDs on the shim, and you know the cores will only push output onto the output BD after the input BDs have completed, it may be sufficient to synchronize only on the output BD before reusing input BDs.

    #### Data Layout Transformations

    The `sizes` and `strides` attributes describe a data layout transformation to be performed by the DMA.
    These transformations are described in more depth in the documentation for the 
    [`aie.dma_bd` operation](AIE.html#aiedma_bd-xilinxaiedmabdop).
    Note that the syntax here differs from that of the `dma_bd` operation: 
    offsets and strides are given as separate arrays instead of tuples.

    The `offsets` array is used to calculate a static offset into the memref.
    Each offset in the array is understood in relation to the shape of the memref; 
    the lowest-dimension `offset` is a direct offset in units of memref element type, and the higher dimensions are multiplied by the size of the memref in those dimensions. 
    Note that this is for convenience of the user only. 
    The hardware only supports a single static offset, and this offset is calculated at compile time.
    Thus, all offsets can be equivalently expressed with the lowest dimension only.

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
        ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_strides,
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

    /* Returns the provided multi-dimensional data transfer strides in units of
       address granularity. In the IR, we express strides in units of element
       data type, but the hardware requires it in units of address granularity.
       Address granularity currently is 4 bytes for all hardware.

       The returned stride[0] is the second-lowest dimension stride, i.e.
       stride 1. The lowest stride is currently implicitly one, but this is not
       a hardware requirement and could be changed in the future.  */
    llvm::SmallVector<int64_t, 4> getStridesInAddressGranularity();

    /* Returns the multi-dimensional data transfer sizes in units of address
       granularity. These sizes are expressed in units of element data type in
       the IR, but the hardware requires them to be in units of address
       granularity. Address granularity currently is 4 bytes for all hardware. 

       The returned size[0] is the lowest dimension size. In the IR, the sizes
       are given in reverse order. For example, specifying sizes in IR as
       [1, 2, 3, 4] would result in this function returning [4, 3, 2, 1].
       */
    llvm::SmallVector<int64_t, 4> getSizesInAddressGranularity();

    /* Returns the data transfer offset in bytes, i.e. the first N bytes of the
       target buffer will be skipped. In the IR, offsets are expressed in units
       of memref element data type size. */
    int64_t getOffsetInBytes(); 
  }];

  let extraClassDefinition = [{
    unsigned $cppClass::getOffsetSizeAndStrideStartOperandIndex() { return 1; }
    std::array<unsigned, 3> $cppClass::getArrayAttrMaxRanks() { return {4, 4, 4}; }
  }];

  let hasVerifier = 1;
}

def AIE_NpuDmaWaitOp: AIEX_Op<"npu.dma_wait", []> {
  let summary = "Blocking operation to wait for a DMA to complete execution.";
  let description = [{
    The NpuDmaWaitOp blocks until the DMA referenced through `symbol` completes execution
    and issues a task-complete-token (TCT).

    `symbol` is a reference to a `aie.shim_dma_allocation`, which contains information about the column, channel and channel direction on which to wait for a TCT. 
    The `aie.shim_dma_allocation` may be generated from an ObjectFIFO, in which case you can directly pass the ObjectFIFO symbol refrence.
    `npu.dma_wait` will be lowered to the corresponding `npu.sync` operation using the information from `symbol`.

    Example:
    ```mlir
      ...
      aie.objectfifo @out0(%tile_0_1, {% raw %}{%tile_0_0}{% endraw %}, 4 : i32) : !aie.objectfifo<memref<32x32xi32>>
      ...
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[1, 1, 0, 0][1, 1, 32, 32][1, 1, 64, 1]) {id = 0 : i64, issue_token = true, metadata = @out0} : memref<32x64xi32>
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
    ins FlatSymbolRefAttr:$buffer,
        UI32Attr:$index,
        I32Attr:$value
  );
  let results = (outs );
  let assemblyFormat = [{ `(` $buffer `,` $index `,` $value `)` attr-dict 
  }];
  let description = [{
    rtp write operator
  }];
}

// Push BD to Queue
def AIE_NpuPushQueueOp: AIEX_Op<"npu.push_queue", []> {
  let summary = "bd queue push operator";
  let arguments = (
    ins I32Attr:$column,
        I32Attr:$row,
        DMAChannelDir:$direction,
        I32Attr:$channel,
        BoolAttr:$issue_token,
        I32Attr:$repeat_count,
        I32Attr:$bd_id
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $column `,` $row `,` $direction `:` $channel `)` attr-dict
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
    ins UI32Attr:$address,
        UI32Attr:$value,
        OptionalAttr<FlatSymbolRefAttr>:$buffer,
        OptionalAttr<I32Attr>:$column,
        OptionalAttr<I32Attr>:$row
  );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
  let description = [{
    NPU write32 operator writes a 32bit value to the AIE array.
    If 'buffer' is present then 'address' is interpreted as an offset into the
    aie.buffer with symbol name 'buffer'.
    If 'column' and 'row' are present then 'address' is interpreted as an offset
    into the memory space of aie.tile(column, row).
    If 'buffer' is not present and 'column' and 'row' are not present then
    'address' is interpreted as a full 32-bit address in the AIE array.
  }];
}

// BLOCKWRITE
def AIE_NpuBlockWriteOp: AIEX_Op<"npu.blockwrite", []> {
  let summary = "blockwrite operator";
  let arguments = (
    ins UI32Attr:$address,
        AnyMemRef:$data,
        OptionalAttr<FlatSymbolRefAttr>:$buffer,
        OptionalAttr<I32Attr>:$column,
        OptionalAttr<I32Attr>:$row
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $data `)` attr-dict `:` type($data)
  }];
  let description = [{
    blockwrite operator writes the data from the memref 'data' to the AIE array.
    If 'buffer' is present then 'address' is interpreted as an offset into the
    aie.buffer with symbol name 'buffer'.
    If 'column' and 'row' are present then 'address' is interpreted as an offset
    into the memory space of aie.tile(column, row).
    If 'buffer' is not present and 'column' and 'row' are not present then
    'address' is interpreted as a full 32-bit address in the AIE array.
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
    The sync operation blocks execution of the instruction stream until a task-complete token (TCT) is received on `column`, `row`, channel `channel`, direction `direction` (where `0` is `S2MM` and `1` is `MM2S`).

    #### Troubleshooting

    If this operation appears to deadlock, ensure that at least one buffer descriptor is configured to issue a TCT on the channel you expect.
    By default, `dma_memcpy_nd` operations only issue tokens for `S2MM` channels, and `issue_token` must be set to `true` to issue tokens for `MM2S` channels.
  }];
}

// XAIE_IO_CUSTOM_OP_BEGIN + 1 (address patch)
def AIE_NpuAddressPatchOp: AIEX_Op<"npu.address_patch", []> {
  let summary = "address patch operator";
  let arguments = (
    ins UI32Attr:$addr,
        I32Attr:$arg_idx,
        I32Attr:$arg_plus
  );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
  let description = [{
    address patch operator
  }];
}

// NPU Bd Write operation
def AIE_NpuWriteBdOp: AIEX_Op<"npu.writebd", []> {
  let summary = "dma operator";
  let arguments = (
    ins I32Attr:$column,
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
        I32Attr:$row,
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
    writebd operator
  }];
}

def AIE_DMAConfigureBDs : AIEX_Op<"dma_configure_bds", []>, Results<(outs Index:$result)> {
  let summary = "Concrete Instantiation of a Buffer Descriptor Chain";
  let description = [{
    Describes a chain of buffer descriptors. 
    Unlike aie.bd_chain, this chain cannot be parametrized, and it is associated with one tile on which it may be submitted to a channel's task queue for execution.
  }];

  let arguments = (
    ins Index:$tile
  );

  let regions = (
    region AnyRegion:$body
  );

  let assemblyFormat = [{
    `(` $tile `)` regions attr-dict
  }];

  let extraClassDeclaration = [{
    AIE::TileOp getTileOp();
  }];

  let extraClassDefinition = [{
    AIE::TileOp DMAConfigureBDs::getTileOp() { return cast<AIE::TileOp>(getTile().getDefiningOp()); }
  }];
}

def AIE_DMAFreeBDs : AIEX_Op<"dma_free_bds", []> {
  let arguments = (
    ins Index:$bds
  );

  let assemblyFormat = [{
    `(` $bds `)` attr-dict
  }];

  let extraClassDeclaration = [{
    DMAConfigureBDs getBdsOp();
  }];

  let extraClassDefinition = [{
    DMAConfigureBDs DMAFreeBDs::getBdsOp() { return cast<DMAConfigureBDs>(getBds().getDefiningOp()); }
  }];
}

def AIE_DMAStartBDs : AIEX_Op<"dma_start_bds", []> {
  let arguments = (
    ins Index:$bds,
        Index:$tile,
        DMAChannelDir:$direction,
        I32Attr:$channel
  );

  let assemblyFormat = [{
    $bds `on` ` ` `(` $tile `,` $direction `,` $channel `)` attr-dict
  }];

  let extraClassDeclaration = [{
    AIE::TileOp getTileOp();
  }];

  let extraClassDefinition = [{
    AIE::TileOp DMAStartBDs::getTileOp() { return cast<AIE::TileOp>(getTile().getDefiningOp()); }
  }];
}

def AIE_DMAStartTask: AIEX_Op<"dma_start_task", [
    DeclareOpInterfaceMethods<CallOpInterface>
  ]> {

  let arguments = (
    ins FlatSymbolRefAttr:$symbol,
        Variadic<AnyType>:$concrete_args,
        Index:$tile,
        DMAChannelDir:$direction,
        I32Attr:$channel
  );

  let assemblyFormat = [{ 
    $symbol `(` $concrete_args `)` `:` `(` type($concrete_args) `)` `on` ` ` `(` $tile `,` $direction `,` $channel `)` attr-dict 
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    AIE::TileOp getTileOp();
  }];

  let extraClassDefinition = [{
    AIE::TileOp DMAStartTask::getTileOp() { return cast<AIE::TileOp>(getTile().getDefiningOp()); }
  }];

}

#endif // AIEX_OPS