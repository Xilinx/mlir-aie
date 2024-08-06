// REQUIRES: valid_xchess_license
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | aie-translate -aie2=true -aievec-to-cpp -o kernel.tmp.cc
// RUN: echo "#include <cstdint>" > kernel.cc && cat kernel.tmp.cc >> kernel.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -I%aietools/include -D__AIENGINE__ -D__AIEARCH__=20 kernel.cc %S/helplib.cc %S/main.cc
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../../profiling.tcl ./work/a.out" | FileCheck %s

func.func private @printv16xi32(%v : vector<16xi32>)
func.func private @loadA64xi32() -> memref<64xi32>

#map6 = affine_map<(d0) -> (d0 + 6)>

func.func @entry() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index

  %buffi32 = func.call @loadA64xi32() : () -> (memref<64xi32>)

  %v0 = vector.transfer_read %buffi32[%c0], %c0_i32 : memref<64xi32>, vector<16xi32>
  func.call @printv16xi32(%v0) : (vector<16xi32>) -> ()

  %v5 = vector.transfer_read %buffi32[%c5], %c0_i32 : memref<64xi32>, vector<16xi32>
  func.call @printv16xi32(%v5) : (vector<16xi32>) -> ()

  %idx6 = affine.apply #map6(%c0)
  %v6 = vector.transfer_read %buffi32[%idx6], %c0_i32 : memref<64xi32>, vector<16xi32>
  func.call @printv16xi32(%v6) : (vector<16xi32>) -> ()

  return %c0_i32 : i32
}

// CHECK-LABEL: vector<16xi32>[ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ]
// CHECK-LABEL: vector<16xi32>[ 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ]
// CHECK-LABEL: vector<16xi32>[ 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 ]
// CHECK-LABEL: SUCCESS
