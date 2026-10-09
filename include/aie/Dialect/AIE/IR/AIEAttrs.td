//===- AIEAttrs.td -----------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2019-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_ATTRS
#define AIE_ATTRS

include "aie/Dialect/AIE/IR/AIE.td"
include "aie/Dialect/AIE/IR/AIETraceAttrs.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Overridden upstream attributes that return int instead of uint
//===----------------------------------------------------------------------===//

class AIETypedSignlessIntegerAttrBase<I attrValType, string retType,
                                      string descr>
    : SignlessIntegerAttrBase<attrValType, descr> {
  let returnType = retType;
  let convertFromStorage = "$_self.getValue().getSExtValue()";
}

def AIEI8Attr
    : AIETypedSignlessIntegerAttrBase<I8, "int8_t",
                                      "8-bit signless integer attribute">;
def AIEI16Attr
    : AIETypedSignlessIntegerAttrBase<I16, "int16_t",
                                      "16-bit signless integer attribute">;
def AIEI32Attr
    : AIETypedSignlessIntegerAttrBase<I32, "int32_t",
                                      "32-bit signless integer attribute">;
def AIEI64Attr
    : AIETypedSignlessIntegerAttrBase<I64, "int64_t",
                                      "64-bit signless integer attribute">;

//===----------------------------------------------------------------------===//
// AIE attributes.
//===----------------------------------------------------------------------===//

def CoreWire : I32EnumAttrCase<"Core", 0>;
def DMAWire : I32EnumAttrCase<"DMA", 1>;
def FIFOWire : I32EnumAttrCase<"FIFO", 2>;
def SouthWire : I32EnumAttrCase<"South", 3>;
def WestWire : I32EnumAttrCase<"West", 4>;
def NorthWire : I32EnumAttrCase<"North", 5>;
def EastWire : I32EnumAttrCase<"East", 6>;
def PLIOWire : I32EnumAttrCase<"PLIO", 7>;
def NOCWire : I32EnumAttrCase<"NOC", 8>;
def TraceWire : I32EnumAttrCase<"Trace", 9>;
def TileControlWire : I32EnumAttrCase<"TileControl", 10>;
def Controller32Wire : I32EnumAttrCase<"Controller32", 11>;

def WireBundle : I32EnumAttr<"WireBundle", "Bundle of wires",
                             [CoreWire, DMAWire, FIFOWire, SouthWire, WestWire,
                              NorthWire, EastWire, PLIOWire, NOCWire, TraceWire,
                              TileControlWire, Controller32Wire]> {

  let cppNamespace = "xilinx::AIE";
}

def CascadeDir : I32EnumAttr<"CascadeDir", "Directions for cascade",
                             [SouthWire, WestWire, NorthWire, EastWire]> {

  let cppNamespace = "xilinx::AIE";
}

def LockAction : I32EnumAttr<"LockAction", "lock acquire/release",
                             [I32EnumAttrCase<"Acquire", 0>,
                              I32EnumAttrCase<"AcquireGreaterEqual", 2>,
                              I32EnumAttrCase<"Release", 1>,
]> {

  let cppNamespace = "xilinx::AIE";
}

def LockBlocking : I32EnumAttr<"LockBlocking", "lock operation is blocking",
                               [I32EnumAttrCase<"NonBlocking", 0>,
                                I32EnumAttrCase<"Blocking", 1>]> {

  let cppNamespace = "xilinx::AIE";
}

def AIEArch
    : I32EnumAttr<"AIEArch", "AIE Architecture",
                  [I32EnumAttrCase<"AIE1", 1>, I32EnumAttrCase<"AIE2", 2>,
                   I32EnumAttrCase<"AIE2p", 3>, I32EnumAttrCase<"AIE2ps", 4>,
]> {

  let cppNamespace = "xilinx::AIE";
}

def AIETileType
    : I32EnumAttr<
          "AIETileType", "Type of AIE Tile",
          [I32EnumAttrCase<"CoreTile", 0>, I32EnumAttrCase<"MemTile", 1>,
           I32EnumAttrCase<"ShimNOCTile", 2>, I32EnumAttrCase<"ShimPLTile", 3>,
]> {

  let cppNamespace = "xilinx::AIE";
}

def AIEDevice
    : I32EnumAttr<
          "AIEDevice", "AIE Device",
          [I32EnumAttrCase<"xcvc1902", 1>, I32EnumAttrCase<"xcve2302", 2>,
           I32EnumAttrCase<"xcve2802", 3>, I32EnumAttrCase<"npu1", 4>,
           I32EnumAttrCase<"npu1_1col", 5>, I32EnumAttrCase<"npu1_2col", 6>,
           I32EnumAttrCase<"npu1_3col", 7>, I32EnumAttrCase<"npu2", 8>,
           I32EnumAttrCase<"npu2_1col", 9>, I32EnumAttrCase<"npu2_2col", 10>,
           I32EnumAttrCase<"npu2_3col", 11>, I32EnumAttrCase<"npu2_4col", 12>,
           I32EnumAttrCase<"npu2_5col", 13>, I32EnumAttrCase<"npu2_6col", 14>,
           I32EnumAttrCase<"npu2_7col", 15>, I32EnumAttrCase<"xcve3858", 16>,
]> {

  let cppNamespace = "xilinx::AIE";
}

def ObjectFifoPort : I32EnumAttr<"ObjectFifoPort", "Ports of an object FIFO",
                                 [I32EnumAttrCase<"Produce", 0>,
                                  I32EnumAttrCase<"Consume", 1>]> {
  let cppNamespace = "xilinx::AIE";
}

def DMAChannelDir
    : I32EnumAttr<"DMAChannelDir", "DMA Channel direction",
                  [I32EnumAttrCase<"S2MM", 0>, I32EnumAttrCase<"MM2S", 1>]> {
  let cppNamespace = "xilinx::AIE";
}

def BDDimLayoutAttr : AttrDef<AIE_Dialect, "BDDimLayout", []> {
  let mnemonic = "bd_dim_layout";
  let summary = [{
    Tuple encoding the stride and size of one dimension in an AIE2 n-dimensional
    buffer descriptor;
  }];

  let parameters = (ins
    "uint32_t" : $size,
    "uint32_t" : $stride
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def BDDimLayoutArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"BDDimLayoutArray",
    /*attrMnemonic*/"bd_dim_layout_array",
    /*eltName*/BDDimLayoutAttr.cppClassName
>;

def BDDimLayoutArrayArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"BDDimLayoutArrayArray",
    /*attrMnemonic*/"bd_dim_layout_array_array",
    /*eltName*/BDDimLayoutArrayAttr.cppClassName
>;

def BDPadLayoutAttr : AttrDef<AIE_Dialect, "BDPadLayout", []> {
  let mnemonic = "bd_pad_layout";
  let summary = [{
    Tuple encoding number of zeros before and after on that dimension in an AIE2
    n-dimensional buffer descriptor;
  }];

  let parameters = (ins
    "uint16_t" : $const_pad_before,
    "uint16_t" : $const_pad_after
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def BDPadLayoutArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"BDPadLayoutArray",
    /*attrMnemonic*/"bd_pad_layout_array",
    /*eltName*/BDPadLayoutAttr.cppClassName
>;

// Per-segment, per-dimension padding for an objectFifo DMA endpoint.
def BDPadLayoutArrayArrayAttr
    : ArrayOfAttr<
          /*dialect*/ AIE_Dialect,
          /*attrName*/ "BDPadLayoutArrayArray",
          /*attrMnemonic*/ "bd_pad_layout_array_array",
          /*eltName*/ BDPadLayoutArrayAttr.cppClassName>;

def PacketInfoAttr : AttrDef<AIE_Dialect, "PacketInfo", []> {
  let mnemonic = "packet_info";
  let summary = "The header of a packet-switched stream";
  let description = [{
    One spelling of a packet header, from the request down to the hardware.

    `pkt_id` is optional so the attribute can describe a packet-switched
    connection before allocation has picked its header: `#aie.packet_info<>`
    asks for packet switching with any free id, `#aie.packet_info<pkt_id = 3>`
    pins the id. `--aie-objectfifo-allocate` fills the id in, so on an endpoint,
    a buffer descriptor or a runtime op the id is always assigned; those ops
    verify that.

    `pkt_type` defaults to 0 and is omitted when it is.

    ```
      aie.route from @src to [@dst] {packet = #aie.packet_info<pkt_id = 3>}
      aie.dma_bd(%buf : memref<16xi32>) {packet = #aie.packet_info<pkt_id = 3>}
    ```
  }];

  let parameters = (ins DefaultValuedParameter<"uint16_t", "0">:$pkt_type,
      OptionalParameter<"std::optional<uint16_t>">:$pkt_id);

  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    /// Whether allocation has assigned the header id yet.
    bool isAssigned() const { return getPktId().has_value(); }

    /// The id of a header allocation has resolved. Every op below `aie.route`
    /// verifies its header is assigned, so readers at that level use this
    /// rather than re-checking.
    uint16_t assignedId() const {
      assert(isAssigned() && "packet header has no pkt_id");
      return *getPktId();
    }
  }];
}

def PacketInfoArrayAttr
    : TypedArrayAttrBase<PacketInfoAttr, "array of packet headers">;

def BDIterationAttr : AttrDef<AIE_Dialect, "BDIteration", []> {
  let mnemonic = "bd_iteration";
  let summary = [{
    Tuple encoding the iteration state of a buffer descriptor in
    true (element) values;
  }];

  let parameters = (ins
    "uint32_t" : $size,
    "uint32_t" : $stride,
    "uint32_t" : $current
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def TraceBufferAttr : AttrDef<AIE_Dialect, "TraceBuffer", []> {
  let mnemonic = "trace_buffer";
  let summary = [{
    The runtime sequence argument that receives trace data;
  }];

  let description = [{
    `-aie-insert-trace-flows` sets this on every traced `aie.runtime_sequence`.
    `-aie-fuse-trace-buffers` reads it to find the argument. That pass grows the
    argument to cover the sequences this one calls, then writes the attribute
    back.

    `arg_index` indexes the block arguments of the sequence body. `offset` and
    `size` bound the bytes that the trace DMAs write inside that argument.
    `dedicated` is false when the trace shares an output buffer, which
    `aie.trace.host_config` requests with `reuse_output_buffer`.

    Write this attribute by hand to point tracing at an argument of your choice.
    The passes and the Python trace parser both read it.
  }];

  let parameters = (ins "uint32_t":$arg_index, "uint32_t":$offset,
      "uint32_t":$size, "bool":$dedicated);

  let assemblyFormat = "`<` struct(params) `>`";
}

def TraceSliceAttr : AttrDef<AIE_Dialect, "TraceSlice", []> {
  let mnemonic = "trace_slice";
  let summary = [{
    One design's byte range inside a shared trace buffer;
  }];

  let description = [{
    `-aie-fuse-trace-buffers` records one of these per `aiex.run` call site, in
    buffer order, on the calling `aie.runtime_sequence`. A host reads them to
    attribute each byte range of the trace buffer to the design that wrote it.

    `device` and `sequence` name the `aie.device` and the
    `aie.runtime_sequence` that the call site ran.
  }];

  let parameters = (ins StringRefParameter<"device symbol name">:$device,
      StringRefParameter<"runtime sequence symbol name">:$sequence,
      "uint32_t":$offset, "uint32_t":$size);

  let assemblyFormat = "`<` struct(params) `>`";
}

def TraceSliceArrayAttr
    : TypedArrayAttrBase<TraceSliceAttr, "array of trace buffer slices">;

def InitValuesArrayAttr
  : OptionalAttr<TypedArrayAttrBase<ElementsAttr, "array of ElementsAttr">>;

//===----------------------------------------------------------------------===//
// ObjectFifo pools and endpoints
//===----------------------------------------------------------------------===//

def ObjectFifoRole
    : I32EnumAttr<"ObjectFifoRole",
                  "What an objectFifo endpoint does to its pool's buffers",
                  [I32EnumAttrCase<"Fill", 0, "fills">,
                   I32EnumAttrCase<"Drain", 1, "drains">]> {
  let cppNamespace = "xilinx::AIE";
}

//===----------------------------------------------------------------------===//
// ObjectFifo transport
//===----------------------------------------------------------------------===//

def ObjectFifoTransportMode
    : I32EnumAttr<"ObjectFifoTransportMode",
                  "The hardware path an objectFifo's objects travel over",
                  [I32EnumAttrCase<"Auto", 0, "auto">,
                   I32EnumAttrCase<"SharedMem", 1, "shared_mem">,
                   I32EnumAttrCase<"DMA", 2, "dma">]> {
  let cppNamespace = "xilinx::AIE";
}

// Spelled in full, `#aie.packet_info<...>`, as on `aie.route`.
def PacketInfoField : OptionalParameter<"PacketInfoAttr"> {
  let parser = [{ [&]() -> ::mlir::FailureOr<PacketInfoAttr> {
    PacketInfoAttr packet;
    if (::mlir::failed($_parser.parseAttribute(packet)))
      return ::mlir::failure();
    return packet;
  }() }];
  let printer = "$_printer.printAttribute($_self)";
}

def ObjectFifoTransportAttr : AttrDef<AIE_Dialect, "ObjectFifoTransport", []> {
  let mnemonic = "transport";
  let summary = "Which hardware path carries an objectFifo's objects";
  let description = [{
    `auto` leaves the choice to `--aie-objectfifo-split`, which uses shared
    memory when both ends reach one memory module and nothing about the fifo
    forces DMAs. `shared_mem` and `dma` name a path, and split fails when it
    cannot be had.

    `packet` asks for the DMA path's stream connection to be packet-switched,
    sharing the stream with other packet flows rather than reserving a circuit.
    It is the same `#aie.packet_info` header the route and, after allocation,
    the endpoints and buffer descriptors carry: `#aie.packet_info<>` lets
    allocation pick the id, `#aie.packet_info<pkt_id = 3>` pins it. It belongs
    to `auto` and `dma`; on `auto` it only matters when the DMAs are used.

    What drives each end (a DMA channel or a core stream port) is not part of
    the path; see `#aie.end_port`.

    Example:
    ```
      transport = #aie.transport<dma>
      transport = #aie.transport<dma, packet = #aie.packet_info<pkt_id = 3>>
    ```
  }];

  let parameters = (ins EnumParameter<ObjectFifoTransportMode>:$mode,
      PacketInfoField:$packet);

  let assemblyFormat = "`<` $mode (`,` struct($packet)^)? `>`";

  let genVerifyDecl = 1;
}

def ObjectFifoEndPortAttr : AttrDef<AIE_Dialect, "ObjectFifoEndPort", []> {
  let mnemonic = "end_port";
  let summary = "What drives one end of an objectFifo";
  let description = [{
    The port on the end's tile that the end uses, in the vocabulary of
    `aie.route_endpoint`: a `DMA` channel, or a `Core` stream port the core
    reads or writes directly.

    `DMA` without a channel leaves the channel to `--aie-objectfifo-allocate`;
    with one it pins it. A `Core` end names its stream port (0 or 1), holds no
    objects, and is only possible on a compute tile.

    Example:
    ```
      prod_port = #aie.end_port<DMA : 1>
      cons_ports = [#aie.end_port<Core : 0>, #aie.end_port<DMA>]
    ```
  }];

  let parameters = (ins EnumParameter<WireBundle>:$bundle,
      OptionalParameter<"std::optional<uint32_t>">:$channel);

  let assemblyFormat = "`<` $bundle (`:` $channel^)? `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    bool isCore() const { return getBundle() == WireBundle::Core; }

    /// The pinned DMA channel or the stream port, as a channel index.
    std::optional<int> channelIndex() const {
      if (std::optional<uint32_t> channel = getChannel())
        return static_cast<int>(*channel);
      return std::nullopt;
    }
  }];
}

def ObjectFifoEndPortArrayAttr
    : TypedArrayAttrBase<ObjectFifoEndPortAttr,
                         "one #aie.end_port per consumer">;

//===----------------------------------------------------------------------===//
// Generated AIE event enumerations
//===----------------------------------------------------------------------===//

include "AIEEventsAIE.td.inc"
include "AIEEventsAIE2.td.inc"
include "AIEEventsAIE2P.td.inc"

#endif // AIE_ATTRS
