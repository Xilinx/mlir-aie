// This test case will directly return the result generated from -affine-super-vectorize
// because in transfer_read %arg1[%arg5, %arg7], lower dim iv %arg7's corresponding loop
// (affine.for %arg7 = #map0(%arg4) to #map1(%arg4))'s upper bound’s affine_map(<(d0) -> (d0 + 4)>)
// result's offset(4) is not divisible by the vector lane size(8).

// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-affine-vectorize 2>&1 | FileCheck %s

// CHECK-LABEL: Loop upper bound's affine map offset of inner index of vector.transfer_read is not divisible by number of vector lanes.
// CHECK-LABEL: Cannot apply aie-vectorize to func.func because alignment check has failed.

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 4)>

module {
  func.func @matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
    affine.for %arg3 = 0 to 64 step 4 {
      affine.for %arg4 = 0 to 64 step 4 {
        affine.for %arg5 = 0 to 64 step 8 {
          affine.for %arg6 = #map0(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map0(%arg4) to #map1(%arg4) {
              %0 = affine.load %arg0[%arg6, %arg5] : memref<64x64xf32>
              %1 = affine.load %arg1[%arg5, %arg7] : memref<64x64xf32>
              %2 = arith.mulf %0, %1 : f32
              %3 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %4 = arith.addf %3, %2 : f32
              affine.store %4, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %6 = affine.load %arg0[%arg6, %arg5+1] : memref<64x64xf32>
              %7 = affine.load %arg1[%arg5+1, %arg7] : memref<64x64xf32>
              %8 = arith.mulf %6, %7 : f32
              %9 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %10 = arith.addf %9, %8 : f32
              affine.store %10, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %12 = affine.load %arg0[%arg6, %arg5+2] : memref<64x64xf32>
              %13 = affine.load %arg1[%arg5+2, %arg7] : memref<64x64xf32>
              %14 = arith.mulf %12, %13 : f32
              %15 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %16 = arith.addf %15, %14 : f32
              affine.store %16, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %18 = affine.load %arg0[%arg6, %arg5+3] : memref<64x64xf32>
              %19 = affine.load %arg1[%arg5+3, %arg7] : memref<64x64xf32>
              %20 = arith.mulf %18, %19 : f32
              %21 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %22 = arith.addf %21, %20 : f32
              affine.store %22, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %24 = affine.load %arg0[%arg6, %arg5+4] : memref<64x64xf32>
              %25 = affine.load %arg1[%arg5+4, %arg7] : memref<64x64xf32>
              %26 = arith.mulf %24, %25 : f32
              %27 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %28 = arith.addf %27, %26 : f32
              affine.store %28, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %30 = affine.load %arg0[%arg6, %arg5+5] : memref<64x64xf32>
              %31 = affine.load %arg1[%arg5+5, %arg7] : memref<64x64xf32>
              %32 = arith.mulf %30, %31 : f32
              %33 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %34 = arith.addf %33, %32 : f32
              affine.store %34, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %36 = affine.load %arg0[%arg6, %arg5+6] : memref<64x64xf32>
              %37 = affine.load %arg1[%arg5+6, %arg7] : memref<64x64xf32>
              %38 = arith.mulf %36, %37 : f32
              %39 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %40 = arith.addf %39, %38 : f32
              affine.store %40, %arg2[%arg6, %arg7] : memref<64x64xf32>
              %42 = affine.load %arg0[%arg6, %arg5+7] : memref<64x64xf32>
%43 = affine.load %arg1[%arg5+7, %arg7] : memref<64x64xf32>
              %44 = arith.mulf %42, %43 : f32
              %45 = affine.load %arg2[%arg6, %arg7] : memref<64x64xf32>
              %46 = arith.addf %45, %44 : f32
              affine.store %46, %arg2[%arg6, %arg7] : memref<64x64xf32>
            }
          }
        }
      }
    }
    return
  }
}

//CHECK-LABEL:#map = affine_map<(d0, d1) -> (0)>
//CHECK-NEXT: module {
//CHECK-NEXT:  func.func @matmul(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
//CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %c64 = arith.constant 64 : index
//CHECK-NEXT:    %c4 = arith.constant 4 : index
//CHECK-NEXT:    scf.for %arg3 = %c0 to %c64 step %c4 {
//CHECK-NEXT:      %c0_0 = arith.constant 0 : index
//CHECK-NEXT:      %c64_1 = arith.constant 64 : index
//CHECK-NEXT:      %c4_2 = arith.constant 4 : index
//CHECK-NEXT:      scf.for %arg4 = %c0_0 to %c64_1 step %c4_2 {
//CHECK-NEXT:        %c0_3 = arith.constant 0 : index
//CHECK-NEXT:        %c64_4 = arith.constant 64 : index
//CHECK-NEXT:        %c8 = arith.constant 8 : index
//CHECK-NEXT:        scf.for %arg5 = %c0_3 to %c64_4 step %c8 {
//CHECK-NEXT:          %c1 = arith.constant 1 : index
//CHECK-NEXT:          %0 = arith.addi %arg5, %c1 : index
//CHECK-NEXT:          %c2 = arith.constant 2 : index
//CHECK-NEXT:          %1 = arith.addi %arg5, %c2 : index
//CHECK-NEXT:          %c3 = arith.constant 3 : index
//CHECK-NEXT:          %2 = arith.addi %arg5, %c3 : index
//CHECK-NEXT:          %c4_5 = arith.constant 4 : index
//CHECK-NEXT:          %3 = arith.addi %arg5, %c4_5 : index
//CHECK-NEXT:          %c5 = arith.constant 5 : index
//CHECK-NEXT:          %4 = arith.addi %arg5, %c5 : index
//CHECK-NEXT:          %c6 = arith.constant 6 : index
//CHECK-NEXT:          %5 = arith.addi %arg5, %c6 : index
//CHECK-NEXT:          %c7 = arith.constant 7 : index
//CHECK-NEXT:          %6 = arith.addi %arg5, %c7 : index
//CHECK-NEXT:          %c4_6 = arith.constant 4 : index
//CHECK-NEXT:          %7 = arith.addi %arg3, %c4_6 : index
//CHECK-NEXT:          %c1_7 = arith.constant 1 : index
//CHECK-NEXT:          scf.for %arg6 = %arg3 to %7 step %c1_7 {
//CHECK-NEXT:            %c4_8 = arith.constant 4 : index
//CHECK-NEXT:            %8 = arith.addi %arg4, %c4_8 : index
//CHECK-NEXT:            %c8_9 = arith.constant 8 : index
//CHECK-NEXT:            scf.for %arg7 = %arg4 to %8 step %c8_9 {
//CHECK-NEXT:              %9 = vector.transfer_read %arg0[%arg6, %arg5], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %10 = vector.transfer_read %arg1[%arg5, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %11 = arith.mulf %9, %10 : vector<8xf32>
//CHECK-NEXT:              %12 = vector.transfer_read %arg2[%arg6, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %13 = arith.addf %12, %11 : vector<8xf32>
//CHECK-NEXT:              %14 = vector.transfer_read %arg0[%arg6, %0], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %15 = vector.transfer_read %arg1[%0, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %16 = arith.mulf %14, %15 : vector<8xf32>
//CHECK-NEXT:              %17 = arith.addf %13, %16 : vector<8xf32>
//CHECK-NEXT:              %18 = vector.transfer_read %arg0[%arg6, %1], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %19 = vector.transfer_read %arg1[%1, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %20 = arith.mulf %18, %19 : vector<8xf32>
//CHECK-NEXT:              %21 = arith.addf %17, %20 : vector<8xf32>
//CHECK-NEXT:              %22 = vector.transfer_read %arg0[%arg6, %2], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %23 = vector.transfer_read %arg1[%2, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %24 = arith.mulf %22, %23 : vector<8xf32>
//CHECK-NEXT:              %25 = arith.addf %21, %24 : vector<8xf32>
//CHECK-NEXT:              %26 = vector.transfer_read %arg0[%arg6, %3], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %27 = vector.transfer_read %arg1[%3, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %28 = arith.mulf %26, %27 : vector<8xf32>
//CHECK-NEXT:              %29 = arith.addf %25, %28 : vector<8xf32>
//CHECK-NEXT:              %30 = vector.transfer_read %arg0[%arg6, %4], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %31 = vector.transfer_read %arg1[%4, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %32 = arith.mulf %30, %31 : vector<8xf32>
//CHECK-NEXT:              %33 = arith.addf %29, %32 : vector<8xf32>
//CHECK-NEXT:              %34 = vector.transfer_read %arg0[%arg6, %5], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %35 = vector.transfer_read %arg1[%5, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %36 = arith.mulf %34, %35 : vector<8xf32>
//CHECK-NEXT:              %37 = arith.addf %33, %36 : vector<8xf32>
//CHECK-NEXT:              %38 = vector.transfer_read %arg0[%arg6, %6], %cst {in_bounds = [true], permutation_map = #map} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %39 = vector.transfer_read %arg1[%6, %arg7], %cst {in_bounds = [true]} : memref<64x64xf32>, vector<8xf32>
//CHECK-NEXT:              %40 = arith.mulf %38, %39 : vector<8xf32>
//CHECK-NEXT:              %41 = arith.addf %37, %40 : vector<8xf32>
//CHECK-NEXT:              vector.transfer_write %41, %arg2[%arg6, %arg7] {in_bounds = [true]} : vector<8xf32>, memref<64x64xf32>
//CHECK-NEXT:            }
//CHECK-NEXT:          }
//CHECK-NEXT:        }
//CHECK-NEXT:      }
//CHECK-NEXT:    }
//CHECK-NEXT:    return
//CHECK-NEXT:  }
//CHECK-NEXT:}
