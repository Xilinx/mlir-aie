#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECLLVMINTROP_TD
#define AIE_DIALECT_AIEVEC_IR_AIEVECLLVMINTROP_TD

include "aie/Dialect/AIE/IR/AIEAttrs.td"
include "aie/Dialect/AIEVec/IR/AIEVecTypes.td"
include "aie/Dialect/AIEVec/IR/AIEVecTypeConstraints.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

class ExtIntrOpBase<Dialect dialect,
                    string opName,
                    string enumName,
                    list<Trait> traits,
                    int numResults
> : LLVM_OpBase<dialect, opName, traits>,
    Results<!if(!gt(numResults, 0), (outs LLVM_Type:$res), (outs))> {
    string llvmBuilder = [{
      auto *inst = ::xilinx::aievec::createExternalIntrinsicCall(builder, moduleTranslation, &opInst, "llvm.}] #
      enumName # [{");
      (void) inst;
      $res = inst;
    }];
    string mlirBuilder = "";
    list<int> llvmArgIndices = [];
}

// For AIE2 only
class AIEVec2_IntrOp<string mnemonic,
                     list<Trait> traits = [],
                     int numResults = 1> :
    ExtIntrOpBase<AIEVec_Dialect,
                  /*opName =*/"intr." # mnemonic,
                  /*enumName =*/"aie2." # mnemonic,
                  traits,
                  numResults>;


// TODO: Find better names for these
def MacConfAcc32IntrOp :
    AIEVec2_IntrOp<"I512.I512.ACC1024.acc32.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   VectorOfLengthAndType<[16], [I64]>:$acc,
                   I32:$conf)>;

def MacConfBF16IntrOp :
    AIEVec2_IntrOp<"bf.mac16.conf",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$lhs,
                   VectorOfLengthAndType<[32], [BF16]>:$rhs,
                   VectorOfLengthAndType<[8], [I64]>:$acc,
                   I32:$conf)>;

def VectorSetI512I128IntrOp :
    AIEVec2_IntrOp<"set.I512.I128",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[4], [I32]>:$src)>;

def VectorSetI512I256IntrOp :
    AIEVec2_IntrOp<"set.I512.I256",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I32]>:$src,
                   I32:$pos)>;

#endif // AIE_DIALECT_AIEVEC_IR_AIEVECLLVMINTROP_TD
