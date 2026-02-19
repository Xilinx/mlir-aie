//===- AIEInterfaces.td ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
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

def PredIsCoreTile : CPred<"xilinx::AIE::getTargetModel(&$_op).isCoreTile(llvm::cast<xilinx::AIE::TileElement>($_op).getTileID().col,"
                                                                       "llvm::cast<xilinx::AIE::TileElement>($_op).getTileID().row)">;
def PredIsMemTile : CPred<"xilinx::AIE::getTargetModel(&$_op).isMemTile(llvm::cast<xilinx::AIE::TileElement>($_op).getTileID().col,"
                                                                       "llvm::cast<xilinx::AIE::TileElement>($_op).getTileID().row)">;
def PredIsShimNOCTile : CPred<"xilinx::AIE::getTargetModel(&$_op).isShimNOCTile(llvm::cast<xilinx::AIE::TileElement>($_op).getTileID().col,"
                                                                        "llvm::cast<xilinx::AIE::TileElement>($_op).getTileID().row)">;
def PredIsShimPLTile : CPred<"xilinx::AIE::getTargetModel(&$_op).isShimPLTile(llvm::cast<xilinx::AIE::TileElement>($_op).getTileID().col,"
                                                                        "llvm::cast<xilinx::AIE::TileElement>($_op).getTileID().row)">;

def IsCoreTile : PredOpTrait<"op exists in a core tile", PredIsCoreTile>;
def IsMemTile : PredOpTrait<"op exists in a MemTile", PredIsMemTile>;
def IsTileWithMemory : PredOpTrait<"op exists in a tile with local memory", Or<[PredIsCoreTile, PredIsMemTile]>>;
def IsShimTile : PredOpTrait<"op exists in a shim tile", Or<[PredIsShimNOCTile, PredIsShimPLTile]>>;
def IsShimNOCTile : PredOpTrait<"op exists in a shim tile with NOC connection", PredIsShimNOCTile>;
def IsShimPLTile : PredOpTrait<"op exists in a shim tile with PL interface", PredIsShimPLTile>;

def FlowEndPoint : OpInterface<"FlowEndPoint"> {
  let description = [{
    Interface for operations that have interconnect-like properties,
    enabling them to host flows for routing.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [
    InterfaceMethod<[{}],
      "int", "colIndex", (ins )
    >,
    InterfaceMethod<[{}],
      "int", "rowIndex", (ins )
    >
  ];
}

def Interconnect : OpInterface<"Interconnect"> {
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
      "int", "colIndex", (ins )
    >,
    InterfaceMethod<[{}],
      "int", "rowIndex", (ins )
    >,
    InterfaceMethod<[{}],
      "size_t", "getNumSourceConnections", (ins "WireBundle":$bundle)
    >,
    InterfaceMethod<[{}],
      "size_t", "getNumDestConnections", (ins "WireBundle":$bundle)
    >
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
    InterfaceMethod<[{
        Return the location of the Tile where the element is located.
      }],
      "xilinx::AIE::TileID", "getTileID", (ins ),
      /*methodBody=*/[{}],
      /*defaultImpl=*/[{
        ConcreteOp op = llvm::cast<ConcreteOp>(this->getOperation());
        return op.getTileOp().getTileID();
      }]
    >,
  ];
  let extraTraitClassDeclaration = [{
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      ConcreteOp op = llvm::cast<ConcreteOp>(this->getOperation());
      std::string nameWithoutDialect =
          op.getOperationName().str().substr(op.getOperationName().find('.') + 1);
      setNameFn(op.getResult(), nameWithoutDialect + "_" +
                                    std::to_string(getTileID().col) + "_" +
                                    std::to_string(getTileID().row));
    }
  }];
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