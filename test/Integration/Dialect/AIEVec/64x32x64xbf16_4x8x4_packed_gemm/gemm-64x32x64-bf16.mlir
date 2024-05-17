// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -convert-vector-to-aievec="aie-target=aieml" -lower-affine -canonicalize -cse | aie-translate --aieml -aievec-to-cpp -o gen_aie.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c %S/kernel.cc -o kernel.cc.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc ./work/kernel.cc.o
// RUN: cp -r %S/data . && xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @gemm_64x32x64_bf16_packed_4x8x4(%A: memref<16x4x4x8xbf16>,
                                             %B: memref<4x16x8x4xbf16>,
                                             %C: memref<16x16x4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c0_bf16 = arith.constant 0.000000e+00 : bf16
    %c0_f32 = arith.constant 0.000000e+00 : f32
    affine.for %i = 0 to 16 {
      affine.for %j = 0 to 16 {
        affine.for %k = 0 to 4 {
          %va = vector.transfer_read %A[%i, %k, %c0, %c0], %c0_bf16 :
                                        memref<16x4x4x8xbf16>, vector<4x8xbf16>
          %vb = vector.transfer_read %B[%k, %j, %c0, %c0], %c0_bf16 :
                                        memref<4x16x8x4xbf16>, vector<8x4xbf16>
          %vc = vector.transfer_read %C[%i, %j, %c0, %c0], %c0_f32 :
                                        memref<16x16x4x4xf32>, vector<4x4xf32>
          %vaf32 = arith.extf %va : vector<4x8xbf16> to vector<4x8xf32>
          %vbf32 = arith.extf %vb : vector<8x4xbf16> to vector<8x4xf32>
          %vr = vector.contract {
                        indexing_maps = [#map, #map1, #map2],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
                      %vaf32, %vbf32, %vc :
                      vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
          vector.transfer_write %vr, %C[%i, %j, %c0, %c0] :
                                        vector<4x4xf32>, memref<16x16x4x4xf32>
        }
      }
    }
    return
  }
}

// CHECK-LABEL: N: 64, M: 64, K: 32
// CHECK-LABEL: Running MATMUL...
// CHECK: Cycle count: [[CC:[0-9]+]]
// CHECK-LABEL: Finish MATMUL!
// CHECK-LABEL: Compare the results
// CHECK: PASSED, Max delta: [[MD:-?[0-9]+.[0-9]+]], pixel intensity
