// RUN: aie-opt %s -split-input-file -verify-diagnostics

func.func @invalidStructType(%A : vector<32xbf16>, %B : vector<32xbf16>)
                            -> vector<16xbf16> {
  // expected-error @+1 {{'res' is an LLVM struct of {vector of bfloat16 type values of length 32; 32-bit signless integer}}
  %rs = "xllvm.intr.aie2.vmax.ltbf16"(%A, %B) :
          (vector<32xbf16>, vector<32xbf16>) -> !llvm.struct<(vector<16xbf16>, i32)>
  %rv = llvm.extractvalue %rs[0] : !llvm.struct<(vector<16xbf16>, i32)>
  return %rv : vector<16xbf16>
}