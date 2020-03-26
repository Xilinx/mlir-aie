// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef AIE_OPS
#else
#define AIE_OPS
#endif


def aie_Dialect : Dialect {
  let name = "aie";
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
  let cppNamespace = "xilinx::aie";
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
  let cppNamespace = "xilinx::aie";
}
def SlavePortEnum: I32EnumAttr<"SlavePortEnum", "AIE Switchbox Slave Port",
								   [ME0, ME1, DMA0, DMA1, TileCtrl, FIFO0, FIFO1,
									South0, South1, South2, South3, SSouth4, SSouth5,
									SWest0, SWest1, SWest2, SWest3,
									SNorth0, SNorth1, SNorth2, SNorth3,
									East0, East1, East2, East3, CoreTrace, MemTrace]> {
  let cppNamespace = "xilinx::aie";
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

class aie_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<aie_Dialect, mnemonic, traits>;

def AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                       "scalar-or-tensor">;

def AnyScalar : TypeConstraint<Or<[Index.predicate,
											  AnySignlessInteger.predicate,
                                   AnyFloat.predicate]>,
                                 "scalar">;


def Port : TypeConstraint<Index.predicate, "port">;

def aie_SwitchboxOp: aie_Op<"switchbox", [SingleBlockImplicitTerminator<"EndswitchOp">]>,
                              Results<(outs AnyType)> {
  let arguments = (
    ins I32Attr:$col,
        I32Attr:$row
  );

  let summary = "Declare a switch";
  let description = [{
		A switch
  }];
  let regions = (region AnyRegion:$connections);
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parseSwitchboxOp(parser, result); }];
  let verifier = [{ return ::verify(*this); }];
}

def aie_CoreOp: aie_Op<"core", []>,
                              Results<(outs Index)> {
  let arguments = (
    ins I32Attr:$col,
        I32Attr:$row
  );

  let summary = "Declare a core";
  let description = [{
		A core
  }];
  let assemblyFormat = [{
	`(` $col `,` $row `)` attr-dict
  }];
  // let builders = [
  //   OpBuilder<"Builder *builder, OperationState &result, "
  //             "Type resultType, int col, int row">
  // ];
}

def aie_ConnectOp: aie_Op<"connect", [HasParent<"SwitchboxOp">]> {
	 let arguments = (
	 ins WireBundle:$sourceBundle,
		  I32Attr:$sourceChannel,
		  WireBundle:$destBundle,
		  I32Attr:$destChannel
	 );
  let summary = "connection inside a switch";
  let description = [{
		A connection inside a switchbox.
  }];
  let assemblyFormat = [{
	`<` $sourceBundle `:` $sourceChannel `,` $destBundle `:` $destChannel `>` attr-dict
  }];
  let extraClassDeclaration = [{
    int sourceIndex() { return sourceChannel().getZExtValue(); }
	 int destIndex() { return destChannel().getZExtValue(); }
  }];
}

def aie_WireOp: aie_Op<"wire", []> {
	 let arguments = (
	 ins Index:$source,
		  WireBundle:$sourceBundle,
		  Index:$dest,
		  WireBundle:$destBundle
	 );
  let summary = "wire between switches";
  let description = [{
		A wire between switches.
  }];
  let assemblyFormat = [{
	`(` $source `:` $sourceBundle `,` $dest `:` $destBundle `)` attr-dict
  }];
}

def aie_FlowOp: aie_Op<"flow", []> {
	 let arguments = (
	 ins Index:$source,
		  WireBundle:$sourceBundle,
		  I32Attr:$sourceChannel,
		  Index:$dest,
		  WireBundle:$destBundle,
		  I32Attr:$destChannel
	 );
  let summary = "logical flow between cores";
  let description = [{
  A logical flow between cores.  During place and route, this is
  replaced by SwitchboxOp and WireOp.
  }];
  let assemblyFormat = [{
	`(` $source `,` $sourceBundle `:` $sourceChannel `,` $dest `,` $destBundle `:` $destChannel `)` attr-dict
  }];
  let extraClassDeclaration = [{
    int sourceIndex() { return sourceChannel().getZExtValue(); }
	 int destIndex() { return destChannel().getZExtValue(); }
  }];
}

def aie_EndswitchOp: aie_Op<"endswitch", [Terminator, HasParent<"SwitchboxOp">]> {
  let summary = "end of a switch";
  let description = [{
		Terminator for switchbox
  }];
}
