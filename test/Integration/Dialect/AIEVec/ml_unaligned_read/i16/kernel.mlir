// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -convert-vector-to-aievec="aie-target=aieml" | aie-translate -aieml=true -aievec-to-cpp -o kernel.tmp.cc
// RUN: echo "#include <cstdint>" > kernel.cc && cat kernel.tmp.cc >> kernel.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -I%aietools/include -D__AIENGINE__ -D__AIEARCH__=20 -c kernel.cc -o kernel.cc.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -I%aietools/include -D__AIENGINE__ -D__AIEARCH__=20 -c %S/helplib.cc -o helplib.cc.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -I%aietools/include -D__AIENGINE__ -D__AIEARCH__=20 ./work/kernel.cc.o ./work/helplib.cc.o %S/main.cc
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../../profiling.tcl ./work/a.out" | FileCheck %s

func.func private @printv32xi16(%v : vector<32xi16>)
func.func private @loadA64xi16() -> memref<64xi16>

#map6 = affine_map<(d0) -> (d0 + 6)>
#map7 = affine_map<(d0) -> (d0 + 7)>
#map8 = affine_map<(d0) -> (d0 + 8)>
#map9 = affine_map<(d0) -> (d0 + 9)>
#map10 = affine_map<(d0) -> (d0 + 10)>

func.func @entry() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c0_i16 = arith.constant 0 : i16
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c7 = arith.constant 7 : index
  %c8 = arith.constant 8 : index
  %c9 = arith.constant 9 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c12 = arith.constant 12 : index
  %c13 = arith.constant 13 : index
  %c14 = arith.constant 14 : index
  %c15 = arith.constant 15 : index

  %buffi16 = func.call @loadA64xi16() : () -> (memref<64xi16>)
  %v16 = vector.transfer_read %buffi16[%c0], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%v16) : (vector<32xi16>) -> ()

  %1 = vector.transfer_read %buffi16[%c1], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%1) : (vector<32xi16>) -> ()
  %2 = vector.transfer_read %buffi16[%c2], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%2) : (vector<32xi16>) -> ()
  %3 = vector.transfer_read %buffi16[%c3], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%3) : (vector<32xi16>) -> ()
  %4 = vector.transfer_read %buffi16[%c4], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%4) : (vector<32xi16>) -> ()
  %5 = vector.transfer_read %buffi16[%c5], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%5) : (vector<32xi16>) -> ()

  %i6 = affine.apply #map6(%c0)
  %6 = vector.transfer_read %buffi16[%i6], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%6) : (vector<32xi16>) -> ()
  %i7 = affine.apply #map7(%c0)
  %7 = vector.transfer_read %buffi16[%i7], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%7) : (vector<32xi16>) -> ()
  %i8 = affine.apply #map8(%c0)
  %8 = vector.transfer_read %buffi16[%i8], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%8) : (vector<32xi16>) -> ()
  %i9 = affine.apply #map9(%c0)
  %9 = vector.transfer_read %buffi16[%i9], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%9) : (vector<32xi16>) -> ()
  %i10 = affine.apply #map10(%c0)
  %10 = vector.transfer_read %buffi16[%i10], %c0_i16 : memref<64xi16>, vector<32xi16>
  func.call @printv32xi16(%10) : (vector<32xi16>) -> ()

  return %c0_i32 : i32
}

// CHECK-LABEL: vector<32xi16>[ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 ]
// CHECK-LABEL: vector<32xi16>[ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 ]
// CHECK-LABEL: vector<32xi16>[ 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 ]
// CHECK-LABEL: vector<32xi16>[ 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 ]
// CHECK-LABEL: vector<32xi16>[ 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 ]
// CHECK-LABEL: vector<32xi16>[ 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 ]
// CHECK-LABEL: vector<32xi16>[ 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 ]
// CHECK-LABEL: vector<32xi16>[ 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 ]
// CHECK-LABEL: vector<32xi16>[ 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 ]
// CHECK-LABEL: vector<32xi16>[ 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 ]
// CHECK-LABEL: vector<32xi16>[ 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 ]
// CHECK-LABEL: SUCCESS
