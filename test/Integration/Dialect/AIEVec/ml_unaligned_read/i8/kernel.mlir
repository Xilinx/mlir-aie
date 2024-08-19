// REQUIRES: valid_xchess_license
// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aie2" | aie-translate -aie2=true -aievec-to-cpp -o kernel.tmp.cc
// RUN: echo "#include <cstdint>" > kernel.cc && cat kernel.tmp.cc >> kernel.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -I%aietools/include -D__AIENGINE__ -D__AIEARCH__=20 kernel.cc %S/helplib.cc %S/main.cc
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../../profiling.tcl ./work/a.out" | FileCheck %s

func.func private @printv64xi8(%v : vector<64xi8>)
func.func private @loadA128xi8() -> memref<128xi8>

#map6 = affine_map<(d0) -> (d0 + 6)>
#map7 = affine_map<(d0) -> (d0 + 7)>
#map8 = affine_map<(d0) -> (d0 + 8)>
#map9 = affine_map<(d0) -> (d0 + 9)>
#map10 = affine_map<(d0) -> (d0 + 10)>

func.func @entry() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c0_i8 = arith.constant 0 : i8
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

  %buffi8 = func.call @loadA128xi8() : () -> (memref<128xi8>)
  %v16 = vector.transfer_read %buffi8[%c0], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%v16) : (vector<64xi8>) -> ()

  %1 = vector.transfer_read %buffi8[%c1], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%1) : (vector<64xi8>) -> ()
  %2 = vector.transfer_read %buffi8[%c2], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%2) : (vector<64xi8>) -> ()
  %3 = vector.transfer_read %buffi8[%c3], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%3) : (vector<64xi8>) -> ()
  %4 = vector.transfer_read %buffi8[%c4], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%4) : (vector<64xi8>) -> ()
  %5 = vector.transfer_read %buffi8[%c5], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%5) : (vector<64xi8>) -> ()

  %i6 = affine.apply #map6(%c0)
  %6 = vector.transfer_read %buffi8[%i6], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%6) : (vector<64xi8>) -> ()
  %i7 = affine.apply #map7(%c0)
  %7 = vector.transfer_read %buffi8[%i7], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%7) : (vector<64xi8>) -> ()
  %i8 = affine.apply #map8(%c0)
  %8 = vector.transfer_read %buffi8[%i8], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%8) : (vector<64xi8>) -> ()
  %i9 = affine.apply #map9(%c0)
  %9 = vector.transfer_read %buffi8[%i9], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%9) : (vector<64xi8>) -> ()
  %i10 = affine.apply #map10(%c0)
  %10 = vector.transfer_read %buffi8[%i10], %c0_i8 : memref<128xi8>, vector<64xi8>
  func.call @printv64xi8(%10) : (vector<64xi8>) -> ()

  return %c0_i32 : i32
}

// CHECK-LABEL: vector<64xi8>[ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 ]
// CHECK-LABEL: vector<64xi8>[ 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 ]
// CHECK-LABEL: vector<64xi8>[ 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 ]
// CHECK-LABEL: vector<64xi8>[ 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 ]
// CHECK-LABEL: vector<64xi8>[ 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 ]
// CHECK-LABEL: vector<64xi8>[ 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 ]
// CHECK-LABEL: vector<64xi8>[ 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 ]
// CHECK-LABEL: vector<64xi8>[ 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 ]
// CHECK-LABEL: vector<64xi8>[ 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 ]
// CHECK-LABEL: vector<64xi8>[ 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 ]
// CHECK-LABEL: vector<64xi8>[ 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 ]
// CHECK-LABEL: SUCCESS
