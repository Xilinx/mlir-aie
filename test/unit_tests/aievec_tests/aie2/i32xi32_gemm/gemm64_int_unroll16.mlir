// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" -aieml=true --aie-vectorize | aie-translate -aie2=true --aievec-to-cpp -o gen_aie-ml.cc
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" --convert-vector-to-aievec="aie-target=aie2" -lower-affine | aie-translate -aie2=true --aievec-to-cpp -o gen_aie-ml.cc
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. -c %S/kernel.cc -o kernel.o
// RUN: xchesscc_wrapper aie2 -f -g +s +w work +o work -I%S -I. %S/testbench.cc work/kernel.o
// RUN: cp -r %S/data . && xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../../profiling.tcl ./work/a.out"
module {
  func.func @matmul(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        affine.for %arg5 = 0 to 64 step 16 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<64x64xi32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<64x64xi32>
          %2 = arith.muli %0, %1 : i32
          %3 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %4 = arith.addi %3, %2 : i32
          affine.store %4, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %5 = affine.load %arg0[%arg3, %arg5 + 1] : memref<64x64xi32>
          %6 = affine.load %arg1[%arg5 + 1, %arg4] : memref<64x64xi32>
          %7 = arith.muli %5, %6 : i32
          %8 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %9 = arith.addi %8, %7 : i32
          affine.store %9, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %10 = affine.load %arg0[%arg3, %arg5 + 2] : memref<64x64xi32>
          %11 = affine.load %arg1[%arg5 + 2, %arg4] : memref<64x64xi32>
          %12 = arith.muli %10, %11 : i32
          %13 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %14 = arith.addi %13, %12 : i32
          affine.store %14, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %15 = affine.load %arg0[%arg3, %arg5 + 3] : memref<64x64xi32>
          %16 = affine.load %arg1[%arg5 + 3, %arg4] : memref<64x64xi32>
          %17 = arith.muli %15, %16 : i32
          %18 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %19 = arith.addi %18, %17 : i32
          affine.store %19, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %20 = affine.load %arg0[%arg3, %arg5 + 4] : memref<64x64xi32>
          %21 = affine.load %arg1[%arg5 + 4, %arg4] : memref<64x64xi32>
          %22 = arith.muli %20, %21 : i32
          %23 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %24 = arith.addi %23, %22 : i32
          affine.store %24, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %25 = affine.load %arg0[%arg3, %arg5 + 5] : memref<64x64xi32>
          %26 = affine.load %arg1[%arg5 + 5, %arg4] : memref<64x64xi32>
          %27 = arith.muli %25, %26 : i32
          %28 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %29 = arith.addi %28, %27 : i32
          affine.store %29, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %30 = affine.load %arg0[%arg3, %arg5 + 6] : memref<64x64xi32>
          %31 = affine.load %arg1[%arg5 + 6, %arg4] : memref<64x64xi32>
          %32 = arith.muli %30, %31 : i32
          %33 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %34 = arith.addi %33, %32 : i32
          affine.store %34, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %35 = affine.load %arg0[%arg3, %arg5 + 7] : memref<64x64xi32>
          %36 = affine.load %arg1[%arg5 + 7, %arg4] : memref<64x64xi32>
          %37 = arith.muli %35, %36 : i32
          %38 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %39 = arith.addi %38, %37 : i32
          affine.store %39, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %40 = affine.load %arg0[%arg3, %arg5 + 8] : memref<64x64xi32>
          %41 = affine.load %arg1[%arg5 + 8, %arg4] : memref<64x64xi32>
          %42 = arith.muli %40, %41 : i32
          %43 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %44 = arith.addi %43, %42 : i32
          affine.store %44, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %45 = affine.load %arg0[%arg3, %arg5 + 9] : memref<64x64xi32>
          %46 = affine.load %arg1[%arg5 + 9, %arg4] : memref<64x64xi32>
          %47 = arith.muli %45, %46 : i32
          %48 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %49 = arith.addi %48, %47 : i32
          affine.store %49, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %50 = affine.load %arg0[%arg3, %arg5 + 10] : memref<64x64xi32>
          %51 = affine.load %arg1[%arg5 + 10, %arg4] : memref<64x64xi32>
          %52 = arith.muli %50, %51 : i32
          %53 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %54 = arith.addi %53, %52 : i32
          affine.store %54, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %55 = affine.load %arg0[%arg3, %arg5 + 11] : memref<64x64xi32>
          %56 = affine.load %arg1[%arg5 + 11, %arg4] : memref<64x64xi32>
          %57 = arith.muli %55, %56 : i32
          %58 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %59 = arith.addi %58, %57 : i32
          affine.store %59, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %60 = affine.load %arg0[%arg3, %arg5 + 12] : memref<64x64xi32>
          %61 = affine.load %arg1[%arg5 + 12, %arg4] : memref<64x64xi32>
          %62 = arith.muli %60, %61 : i32
          %63 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %64 = arith.addi %63, %62 : i32
          affine.store %64, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %65 = affine.load %arg0[%arg3, %arg5 + 13] : memref<64x64xi32>
          %66 = affine.load %arg1[%arg5 + 13, %arg4] : memref<64x64xi32>
          %67 = arith.muli %65, %66 : i32
          %68 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %69 = arith.addi %68, %67 : i32
          affine.store %69, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %70 = affine.load %arg0[%arg3, %arg5 + 14] : memref<64x64xi32>
          %71 = affine.load %arg1[%arg5 + 14, %arg4] : memref<64x64xi32>
          %72 = arith.muli %70, %71 : i32
          %73 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %74 = arith.addi %73, %72 : i32
          affine.store %74, %arg2[%arg3, %arg4] : memref<64x64xi32>
          %75 = affine.load %arg0[%arg3, %arg5 + 15] : memref<64x64xi32>
          %76 = affine.load %arg1[%arg5 + 15, %arg4] : memref<64x64xi32>
          %77 = arith.muli %75, %76 : i32
          %78 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi32>
          %79 = arith.addi %78, %77 : i32
          affine.store %79, %arg2[%arg3, %arg4] : memref<64x64xi32>
        }
      }
    }
    return
  }
}
