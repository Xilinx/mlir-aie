// This test case will directly return the result generated from -affine-super-vectorize
// because in transfer_read %arg2[%arg3, %arg4], dim 1's memref shape size(2046) cannot be
// divisible by the vector lanes(8).

// RUN: aie-opt %s --affine-loop-unroll="unroll-full unroll-full-threshold=3" --canonicalize -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize -split-input-file -unaligned-loads-check=true --debug-only=aie-vect 2>&1 | FileCheck %s

func.func @conv2d (%A: memref<2048x2048xf32>, %B: memref<9xf32>, %C: memref<2046x2046xf32>) {
    affine.for %arg3 = 0 to 2046 {
        affine.for %arg4 = 0 to 2046 {
            //3x3 stencil
            affine.for %arg5 = 0 to 3 {
                affine.for %arg6 = 0 to 3 {
                    //Load the output point
                    %ci = affine.load %C[%arg3, %arg4] : memref<2046x2046xf32>
                     %a11 = affine.load %A[%arg3+%arg5, %arg4+%arg6] : memref<2048x2048xf32>
                     %b11 = affine.load %B[3*%arg5+%arg6] : memref<9xf32>
                     %p11 = arith.mulf %a11, %b11 : f32
                     %c11 = arith.addf %ci, %p11 : f32
                     //Store accumulated sum
                     affine.store %c11, %C[%arg3, %arg4] : memref<2046x2046xf32>
                }
            }
        }
    }
    return
}
//CHECK: %0 = vector.transfer_read %arg2[%arg3, %arg4], %cst {in_bounds = [true]} : memref<2046x2046xf32>, vector<8xf32>'s dim 1's shape size cannot be divisible by the vector lanes.

//CHECK-LABEL:#map0 = affine_map<(d0) -> (0)>
//CHECK-NEXT: #map1 = affine_map<(d0) -> (d0 + 1)>
//CHECK-NEXT: #map2 = affine_map<(d0) -> (d0 + 2)>
//CHECK-NEXT: module {
//CHECK-NEXT:   func.func @conv2d(%arg0: memref<2048x2048xf32>, %arg1: memref<9xf32>, %arg2: memref<2046x2046xf32>) {
//CHECK-NEXT:     %c8 = arith.constant 8 : index
//CHECK-NEXT:     %c7 = arith.constant 7 : index
//CHECK-NEXT:     %c6 = arith.constant 6 : index
//CHECK-NEXT:     %c5 = arith.constant 5 : index
//CHECK-NEXT:     %c4 = arith.constant 4 : index
//CHECK-NEXT:     %c3 = arith.constant 3 : index
//CHECK-NEXT:     %c2 = arith.constant 2 : index
//CHECK-NEXT:     %c1 = arith.constant 1 : index
//CHECK-NEXT:     %c0 = arith.constant 0 : index
//CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
//CHECK-NEXT:     affine.for %arg3 = 0 to 2046 {
//CHECK-NEXT:       affine.for %arg4 = 0 to 2046 step 8 {
//CHECK-NEXT:         %0 = vector.transfer_read %arg2[%arg3, %arg4], %cst {in_bounds = [true]} : memref<2046x2046xf32>, vector<8xf32>
//CHECK-NEXT:         %1 = vector.transfer_read %arg0[%arg3, %arg4], %cst {in_bounds = [true]} : memref<2048x2048xf32>, vector<8xf32>
//CHECK-NEXT:         %2 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true], permutation_map = #map0} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT:         %3 = arith.mulf %1, %2 : vector<8xf32>
//CHECK-NEXT:         %4 = arith.addf %0, %3 : vector<8xf32>
//CHECK-NEXT:         %5 = affine.apply #map1(%arg4)
//CHECK-NEXT:         %6 = vector.transfer_read %arg0[%arg3, %5], %cst {in_bounds = [true]} : memref<2048x2048xf32>, vector<8xf32>
//CHECK-NEXT:         %7 = vector.transfer_read %arg1[%c1], %cst {in_bounds = [true], permutation_map = #map0} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT:         %8 = arith.mulf %6, %7 : vector<8xf32>
//CHECK-NEXT:         %9 = arith.addf %4, %8 : vector<8xf32>
//CHECK-NEXT:         %10 = affine.apply #map2(%arg4)
//CHECK-NEXT:         %11 = vector.transfer_read %arg0[%arg3, %10], %cst {in_bounds = [true]} : memref<2048x2048xf32>, vector<8xf32>
//CHECK-NEXT:         %12 = vector.transfer_read %arg1[%c2], %cst {in_bounds = [true], permutation_map = #map0} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT:         %13 = arith.mulf %11, %12 : vector<8xf32>
//CHECK-NEXT:         %14 = arith.addf %9, %13 : vector<8xf32>
//CHECK-NEXT:         %15 = affine.apply #map1(%arg3)
//CHECK-NEXT:         %16 = vector.transfer_read %arg0[%15, %arg4], %cst {in_bounds = [true]} : memref<2048x2048xf32>, vector<8xf32>
//CHECK-NEXT:         %17 = vector.transfer_read %arg1[%c3], %cst {in_bounds = [true], permutation_map = #map0} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT:         %18 = arith.mulf %16, %17 : vector<8xf32>
//CHECK-NEXT:         %19 = arith.addf %14, %18 : vector<8xf32>
//CHECK-NEXT:         %20 = affine.apply #map1(%arg3)
//CHECK-NEXT:         %21 = affine.apply #map1(%arg4)
//CHECK-NEXT:         %22 = vector.transfer_read %arg0[%20, %21], %cst {in_bounds = [true]} : memref<2048x2048xf32>, vector<8xf32>
//CHECK-NEXT:         %23 = vector.transfer_read %arg1[%c4], %cst {in_bounds = [true], permutation_map = #map0} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT:         %24 = arith.mulf %22, %23 : vector<8xf32>
//CHECK-NEXT:         %25 = arith.addf %19, %24 : vector<8xf32>
//CHECK-NEXT:         %26 = affine.apply #map1(%arg3)
//CHECK-NEXT:         %27 = affine.apply #map2(%arg4)
//CHECK-NEXT:         %28 = vector.transfer_read %arg0[%26, %27], %cst {in_bounds = [true]} : memref<2048x2048xf32>, vector<8xf32>
//CHECK-NEXT:         %29 = vector.transfer_read %arg1[%c5], %cst {in_bounds = [true], permutation_map = #map0} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT:         %30 = arith.mulf %28, %29 : vector<8xf32>
//CHECK-NEXT:         %31 = arith.addf %25, %30 : vector<8xf32>
//CHECK-NEXT:         %32 = affine.apply #map2(%arg3)
//CHECK-NEXT:         %33 = vector.transfer_read %arg0[%32, %arg4], %cst {in_bounds = [true]} : memref<2048x2048xf32>, vector<8xf32>
//CHECK-NEXT:         %34 = vector.transfer_read %arg1[%c6], %cst {in_bounds = [true], permutation_map = #map0} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT:         %35 = arith.mulf %33, %34 : vector<8xf32>
//CHECK-NEXT:         %36 = arith.addf %31, %35 : vector<8xf32>
//CHECK-NEXT:         %37 = affine.apply #map2(%arg3)
//CHECK-NEXT:         %38 = affine.apply #map1(%arg4)
//CHECK-NEXT:         %39 = vector.transfer_read %arg0[%37, %38], %cst {in_bounds = [true]} : memref<2048x2048xf32>, vector<8xf32>
//CHECK-NEXT:         %40 = vector.transfer_read %arg1[%c7], %cst {in_bounds = [true], permutation_map = #map0} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT:         %41 = arith.mulf %39, %40 : vector<8xf32>
//CHECK-NEXT:         %42 = arith.addf %36, %41 : vector<8xf32>
//CHECK-NEXT:         %43 = affine.apply #map2(%arg3)
//CHECK-NEXT:         %44 = affine.apply #map2(%arg4)
//CHECK-NEXT:         %45 = vector.transfer_read %arg0[%43, %44], %cst {in_bounds = [true]} : memref<2048x2048xf32>, vector<8xf32>
//CHECK-NEXT:         %46 = vector.transfer_read %arg1[%c8], %cst {in_bounds = [true], permutation_map = #map0} : memref<9xf32>, vector<8xf32>
//CHECK-NEXT:         %47 = arith.mulf %45, %46 : vector<8xf32>
//CHECK-NEXT:         %48 = arith.addf %42, %47 : vector<8xf32>
//CHECK-NEXT:         vector.transfer_write %48, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<8xf32>, memref<2046x2046xf32>
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:     return
//CHECK-NEXT:   }
//CHECK-NEXT: }
