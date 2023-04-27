// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" --convert-vector-to-aievec="aie-target=aieml" -lower-affine | aie-translate -aieml=true --aievec-to-cpp -o convert_aie-ml.cc
// RUN: xchesscc %S/kernel.cc -f -g +s -p me -P %aietools/data/aie_ml/lib/ +w work +o work -I%S -I. %S/testbench.cc
// RUN: xme_ca_udm_dbg -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xme_ca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xme_ca_udm_dbg.stdout %s

// CHECK: PASSED

module {
  func.func @mul_elem(%arg0: memref<1024xi16>, %arg1: memref<1024xi16>, %arg2: memref<1024xi16>) {
    affine.for %arg3 = 0 to 1024 {
      %0 = affine.load %arg0[%arg3] : memref<1024xi16>
      %1 = affine.load %arg1[%arg3] : memref<1024xi16>
      %2 = arith.muli %0, %1 : i16
      affine.store %2, %arg2[%arg3] : memref<1024xi16>
    }
    return
  }
}
