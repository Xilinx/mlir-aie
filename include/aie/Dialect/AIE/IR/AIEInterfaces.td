//===- AIEInterfaces.td ------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2020-2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef AIE_INTERFACES
#define AIE_INTERFACES

include "aie/Dialect/AIE/IR/AIE.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

// Op is a DMA-like operation with BD contraints
def HasValidBDs : NativeOpTrait<"HasValidBDs"> {
  string cppNamespace = "::xilinx::AIE";
}

// Op is a DMA-like operation with valid channels
def HasValidDMAChannels : NativeOpTrait<"HasValidDMAChannels"> {
  string cppNamespace = "::xilinx::AIE";
}

def PredIsCoreTile : CPred<
  "llvm::cast<xilinx::AIE::TileElement>($_op).getTileLike() && "
  "llvm::cast<xilinx::AIE::TileElement>($_op).getTileLike().isCoreTile()"
>;
def PredIsMemTile : CPred<
  "llvm::cast<xilinx::AIE::TileElement>($_op).getTileLike() && "
  "llvm::cast<xilinx::AIE::TileElement>($_op).getTileLike().isMemTile()"
>;
def PredIsShimNOCTile : CPred<
  "llvm::cast<xilinx::AIE::TileElement>($_op).getTileLike() && "
  "llvm::cast<xilinx::AIE::TileElement>($_op).getTileLike().isShimNOCTile()"
>;
def PredIsShimPLTile : CPred<
  "llvm::cast<xilinx::AIE::TileElement>($_op).getTileLike() && "
  "llvm::cast<xilinx::AIE::TileElement>($_op).getTileLike().isShimPLTile()"
>;

def IsCoreTile : PredOpTrait<"op exists in a core tile", PredIsCoreTile>;
def IsMemTile : PredOpTrait<"op exists in a MemTile", PredIsMemTile>;
def IsTileWithMemory : PredOpTrait<"op exists in a tile with local memory", Or<[PredIsCoreTile, PredIsMemTile]>>;
def IsShimTile : PredOpTrait<"op exists in a shim tile", Or<[PredIsShimNOCTile, PredIsShimPLTile]>>;
def IsShimNOCTile : PredOpTrait<"op exists in a shim tile with NOC connection", PredIsShimNOCTile>;
def IsShimPLTile : PredOpTrait<"op exists in a shim tile with PL interface", PredIsShimPLTile>;

def IsFlowEndPoint : NativeOpTrait<"IsFlowEndPoint"> {
  string cppNamespace = "::xilinx::AIE";
}

def TileLike : OpInterface<"TileLike"> {
  let description = [{
    Interface for operations that represent an AIE tile (logical or physical).
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [
    InterfaceMethod<[{Return the true hardware tile type.}],
      "xilinx::AIE::AIETileType", "getTileType", (ins)
    >,
    InterfaceMethod<[{Return the optional column index.}],
      "std::optional<int>", "tryGetCol", (ins)
    >,
    InterfaceMethod<[{Return the optional row index.}],
      "std::optional<int>", "tryGetRow", (ins)
    >,
    InterfaceMethod<[{}], "bool", "isCoreTile", (ins), [{}], [{
      return $_op.getTileType() == xilinx::AIE::AIETileType::CoreTile;
    }]>,
    InterfaceMethod<[{}], "bool", "isMemTile", (ins), [{}], [{
      return $_op.getTileType() == xilinx::AIE::AIETileType::MemTile;
    }]>,
    InterfaceMethod<[{}], "bool", "isShimNOCTile", (ins), [{}], [{
      return $_op.getTileType() == xilinx::AIE::AIETileType::ShimNOCTile;
    }]>,
    InterfaceMethod<[{}], "bool", "isShimPLTile", (ins), [{}], [{
      return $_op.getTileType() == xilinx::AIE::AIETileType::ShimPLTile;
    }]>,
    InterfaceMethod<[{}], "bool", "isShimNOCorPLTile", (ins), [{}], [{
      return $_op.isShimNOCTile() || $_op.isShimPLTile();
    }]>,
    InterfaceMethod<[{}], "bool", "isShimTile", (ins), [{}], [{
      return $_op.isShimNOCTile() || $_op.isShimPLTile();
    }]>,
    InterfaceMethod<[{Return the number of source connections for a wire bundle.}],
      "size_t", "getNumSourceConnections", (ins "xilinx::AIE::WireBundle":$bundle)
    >,
    InterfaceMethod<[{Return the number of dest connections for a wire bundle.}],
      "size_t", "getNumDestConnections", (ins "xilinx::AIE::WireBundle":$bundle)
    >,
  ];
}

def TileElement : OpInterface<"TileElement", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let description = [{
    Interface for operations that exist in a TileOp.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [
    InterfaceMethod<[{Return the tile operand value.}],
      "mlir::Value", "getTile", (ins)
    >,
    InterfaceMethod<[{Return the TileLike interface for this element's tile.}],
      "TileLike", "getTileLike", (ins), [{}], [{
        return llvm::dyn_cast<TileLike>($_op.getTile().getDefiningOp());
      }]
    >,
    InterfaceMethod<[{Return the column index of this element's tile. Asserts if unplaced.}],
      "int", "colIndex", (ins), [{}], [{
        std::optional<int> col = $_op.getTileLike().tryGetCol();
        assert(col.has_value() && "TileElement called colIndex() on an unplaced tile.");
        return *col;
      }]
    >,
    InterfaceMethod<[{Return the row index of this element's tile. Asserts if unplaced.}],
      "int", "rowIndex", (ins), [{}], [{
        std::optional<int> row = $_op.getTileLike().tryGetRow();
        assert(row.has_value() && "TileElement called rowIndex() on an unplaced tile.");
        return *row;
      }]
    >,
    InterfaceMethod<[{Return the TileID of this element's tile.}],
      "TileID", "getTileID", (ins), [{}], [{
        return TileID{$_op.colIndex(), $_op.rowIndex()};
      }]
    >
  ];

  let extraClassDeclaration = [{
    // Requires a placed tile: fatally errors if the tile is still logical.
    TileOp getTileOp();
    // Null when the tile is still logical, so callers that run before
    // placement can defer instead of aborting.
    TileOp tryGetTileOp();
  }];

  let extraTraitClassDeclaration = [{
    void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
      std::optional<int> col = $_op.getTileLike().tryGetCol();
      std::optional<int> row = $_op.getTileLike().tryGetRow();
      if (col && row) {
        llvm::StringRef opName = $_op.getOperation()->getName().stripDialect();
        std::string name = (opName + "_" + llvm::Twine(*col) + "_" + llvm::Twine(*row)).str();
        setNameFn($_op.getResult(), name);
      }
    }
  }];
}

def BufferLike : OpInterface<"BufferLike"> {
  let description = [{
    Interface for the ops that declare storage a DMA or a core can address:
    `aie.buffer` in a tile's memory module and `aie.external_buffer` in DDR.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [InterfaceMethod<[{The memref this op declares.}],
                                 "mlir::Value", "getBuffer", (ins), [{}],
                                 [{ return $_op.getBuffer(); }]>,
                 InterfaceMethod<
                     [{The tile whose memory module holds this buffer, null
      where it lives outside the array.}],
                     "mlir::Value", "getBufferTile", (ins), [{}],
                     [{ return nullptr; }]>,
  ];
}

def ObjectFifoFlowEndpoint : OpInterface<"ObjectFifoFlowEndpoint"> {
  let description = [{
    Interface for the ops an `aie.objectfifo.flow` connects. Gives
    `--aie-objectfifo-allocate` the switchbox port each end occupies, whether
    that end drives a DMA channel or hands the stream straight to a core.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods =
      [InterfaceMethod<[{Which way this end moves data.}], "DMAChannelDir",
                       "getFlowDirection", (ins)>,
       InterfaceMethod<[{The switchbox bundle this end occupies.}],
                       "WireBundle", "getFlowBundle", (ins)>,
       InterfaceMethod<[{The port within that bundle, once assigned.}],
                       "std::optional<int>", "getFlowChannel", (ins)>,
       InterfaceMethod<[{Record the port this end was given.}], "void",
                       "setFlowChannel", (ins "int":$channel)>,
       InterfaceMethod<[{The tile this end sits on.}], "mlir::Value", "getTile",
                       (ins), [{}], [{ return $_op.getTile(); }]>,
       InterfaceMethod<
           [{The objectFifo this end was split from, if it came from
      one; the name runtime records address it by.}],
           "std::optional<llvm::StringRef>", "getFifoName", (ins), [{}],
           [{ return $_op.getFifoName(); }]>,
       InterfaceMethod<[{Stamp a packet header on this end.}], "void",
                       "setFlowPacket", (ins "PacketInfoAttr":$packet), [{}],
                       [{ $_op.setPacketAttr(packet); }]>,
  ];
}

def DmaBody : OpInterface<"DmaBody", [TileElement]> {
  let description = [{
    Interface for the ops whose region holds one tile's DMA program: the
    `aie.dma_start` chains and the buffer descriptors they run.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [InterfaceMethod<[{Return the region holding the DMA program.}],
                                 "mlir::Region &", "getDmaBody", (ins), [{}],
                                 [{
        return $_op.getBody();
      }]>,
  ];
}

def Interconnect : OpInterface<"Interconnect", [TileElement]> {
  let description = [{
    Interface for operations that have interconnect-like properties,
    enabling them to host flows for routing.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [
    InterfaceMethod<[{}],
      "mlir::Region &", "getConnections", (ins )
    >,
    InterfaceMethod<[{}],
      "size_t", "getNumSourceConnections", (ins "WireBundle":$bundle)
    >,
    InterfaceMethod<[{}],
      "size_t", "getNumDestConnections", (ins "WireBundle":$bundle)
    >
  ];
}

def AIETarget : OpInterface<"AIETarget"> {
  let description = [{
    Interface for operations that model an array of AIEngine cores.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [
    InterfaceMethod<[{
        Return the target model describing the characteristics of how this operation will be implemented.
      }],
      "const ::xilinx::AIE::AIETargetModel&", "getTargetModel", (ins )
    >
  ];
}

// Don't delete - see AIEDialect::myVerifyOffsetSizeAndStrideOp
def MyOffsetSizeAndStrideOpInterface : OpInterfaceTrait<"::xilinx::AIE::MyOffsetSizeAndStrideOpInterface"> {}

// Ops may specify that the accessibility checks of their child ops are not verified.
// For example, by default, ops will verify that all contained aie.buffers are
// accessible from within the tile from which they are used. However, in abstract
// definitions not tied to any tile, it may still be useful to be able to refer
// to buffers and skip those checks.
def SkipAccessibilityCheckTrait : NativeOpTrait<"SkipAccessibilityCheckTrait"> {
  string cppNamespace = "::xilinx::AIE";
}

#endif // AIE_INTERFACES
