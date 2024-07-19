// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=8" --aie-vectorize -unaligned-loads-check=false -split-input-file | FileCheck %s

//CHECK-LABEL: func.func @conv_2d(%arg0: memref<10x3x256x256xf32>, %arg1: memref<10x3x3x3xf32>, %arg2: memref<10x10x254x254xf32>) {
func.func @conv_2d(%arg0: memref<10x3x256x256xf32>, %arg1: memref<10x3x3x3xf32>, %arg2: memref<10x10x254x254xf32>) {
  %c0 = arith.constant 0 : index
  %c0_0 = arith.constant 0 : index
  %c0_1 = arith.constant 0 : index
  affine.for %arg3 = 0 to 10 {
    affine.for %arg4 = 0 to 10 {
      affine.for %arg5 = 0 to 254 {
        affine.for %arg6 = 0 to 254 {
          %2 = affine.load %arg0[%arg3, %c0, %arg5+%c0_0, %arg6+%c0_1] : memref<10x3x256x256xf32>
          %3 = affine.load %arg1[%arg4, %c0, %c0_0, %c0_1] : memref<10x3x3x3xf32>
          %4 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %5 = arith.mulf %2, %3 : f32
          %6 = arith.addf %4, %5 : f32
          affine.store %6, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %10 = affine.load %arg0[%arg3, %c0, %arg5+%c0_0, %arg6+%c0_1+1] : memref<10x3x256x256xf32>
          %11 = affine.load %arg1[%arg4, %c0, %c0_0, %c0_1+1] : memref<10x3x3x3xf32>
          %12 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %13 = arith.mulf %10, %11 : f32
          %14 = arith.addf %12, %13 : f32
          affine.store %14, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %18 = affine.load %arg0[%arg3, %c0, %arg5+%c0_0, %arg6+%c0_1+2] : memref<10x3x256x256xf32>
          %19 = affine.load %arg1[%arg4, %c0, %c0_0, %c0_1+2] : memref<10x3x3x3xf32>
          %20 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %21 = arith.mulf %18, %19 : f32
          %22 = arith.addf %20, %21 : f32
          affine.store %22, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %26 = affine.load %arg0[%arg3, %c0, %arg5+%c0_0+1, %arg6+%c0_1] : memref<10x3x256x256xf32>
          %27 = affine.load %arg1[%arg4, %c0, %c0_0+1, %c0_1] : memref<10x3x3x3xf32>
          %28 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %29 = arith.mulf %26, %27 : f32
          %30 = arith.addf %28, %29 : f32
          affine.store %30, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %34 = affine.load %arg0[%arg3, %c0, %arg5+%c0_0+1, %arg6+%c0_1+1] : memref<10x3x256x256xf32>
          %35 = affine.load %arg1[%arg4, %c0, %c0_0+1, %c0_1+1] : memref<10x3x3x3xf32>
          %36 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %37 = arith.mulf %34, %35 : f32
          %38 = arith.addf %36, %37 : f32
          affine.store %38, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %42 = affine.load %arg0[%arg3, %c0, %arg5+%c0_0+1, %arg6+%c0_1+2] : memref<10x3x256x256xf32>
          %43 = affine.load %arg1[%arg4, %c0, %c0_0+1, %c0_1+2] : memref<10x3x3x3xf32>
          %44 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %45 = arith.mulf %42, %43 : f32
          %46 = arith.addf %44, %45 : f32
          affine.store %46, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %50 = affine.load %arg0[%arg3, %c0, %arg5+%c0_0+2, %arg6+%c0_1] : memref<10x3x256x256xf32>
          %51 = affine.load %arg1[%arg4, %c0, %c0_0+2, %c0_1] : memref<10x3x3x3xf32>
          %52 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %53 = arith.mulf %50, %51 : f32
          %54 = arith.addf %52, %53 : f32
          affine.store %54, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %58 = affine.load %arg0[%arg3, %c0, %arg5+%c0_0+2, %arg6+%c0_1+1] : memref<10x3x256x256xf32>
          %59 = affine.load %arg1[%arg4, %c0, %c0_0+2, %c0_1+1] : memref<10x3x3x3xf32>
          %60 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %61 = arith.mulf %58, %59 : f32
          %62 = arith.addf %60, %61 : f32
          affine.store %62, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %66 = affine.load %arg0[%arg3, %c0, %arg5+%c0_0+2, %arg6+%c0_1+2] : memref<10x3x256x256xf32>
          %67 = affine.load %arg1[%arg4, %c0, %c0_0+2, %c0_1+2] : memref<10x3x3x3xf32>
          %68 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %69 = arith.mulf %66, %67 : f32
          %70 = arith.addf %68, %69 : f32
          affine.store %70, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %74 = affine.load %arg0[%arg3, %c0+1, %arg5+%c0_0, %arg6+%c0_1] : memref<10x3x256x256xf32>
          %75 = affine.load %arg1[%arg4, %c0+1, %c0_0, %c0_1] : memref<10x3x3x3xf32>
          %76 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %77 = arith.mulf %74, %75 : f32
          %78 = arith.addf %76, %77 : f32
          affine.store %78, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %82 = affine.load %arg0[%arg3, %c0+1, %arg5+%c0_0, %arg6+%c0_1+1] : memref<10x3x256x256xf32>
          %83 = affine.load %arg1[%arg4, %c0+1, %c0_0, %c0_1+1] : memref<10x3x3x3xf32>
          %84 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %85 = arith.mulf %82, %83 : f32
          %86 = arith.addf %84, %85 : f32
          affine.store %86, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %90 = affine.load %arg0[%arg3, %c0+1, %arg5+%c0_0, %arg6+%c0_1+2] : memref<10x3x256x256xf32>
          %91 = affine.load %arg1[%arg4, %c0+1, %c0_0, %c0_1+2] : memref<10x3x3x3xf32>
          %92 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %93 = arith.mulf %90, %91 : f32
          %94 = arith.addf %92, %93 : f32
          affine.store %94, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %98 = affine.load %arg0[%arg3, %c0+1, %arg5+%c0_0+1, %arg6+%c0_1] : memref<10x3x256x256xf32>
          %99 = affine.load %arg1[%arg4, %c0+1, %c0_0+1, %c0_1] : memref<10x3x3x3xf32>
          %100 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %101 = arith.mulf %98, %99 : f32
          %102 = arith.addf %100, %101 : f32
          affine.store %102, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %106 = affine.load %arg0[%arg3, %c0+1, %arg5+%c0_0+1, %arg6+%c0_1+1] : memref<10x3x256x256xf32>
          %107 = affine.load %arg1[%arg4, %c0+1, %c0_0+1, %c0_1+1] : memref<10x3x3x3xf32>
          %108 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %109 = arith.mulf %106, %107 : f32
          %110 = arith.addf %108, %109 : f32
          affine.store %110, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %114 = affine.load %arg0[%arg3, %c0+1, %arg5+%c0_0+1, %arg6+%c0_1+2] : memref<10x3x256x256xf32>
          %115 = affine.load %arg1[%arg4, %c0+1, %c0_0+1, %c0_1+2] : memref<10x3x3x3xf32>
          %116 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %117 = arith.mulf %114, %115 : f32
          %118 = arith.addf %116, %117 : f32
          affine.store %118, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %122 = affine.load %arg0[%arg3, %c0+1, %arg5+%c0_0+2, %arg6+%c0_1] : memref<10x3x256x256xf32>
          %123 = affine.load %arg1[%arg4, %c0+1, %c0_0+2, %c0_1] : memref<10x3x3x3xf32>
          %124 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %125 = arith.mulf %122, %123 : f32
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %130 = affine.load %arg0[%arg3, %c0+1, %arg5+%c0_0+2, %arg6+%c0_1+1] : memref<10x3x256x256xf32>
          %131 = affine.load %arg1[%arg4, %c0+1, %c0_0+2, %c0_1+1] : memref<10x3x3x3xf32>
          %132 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %133 = arith.mulf %130, %131 : f32
          %134 = arith.addf %132, %133 : f32
          affine.store %134, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %138 = affine.load %arg0[%arg3, %c0+1, %arg5+%c0_0+2, %arg6+%c0_1+2] : memref<10x3x256x256xf32>
          %139 = affine.load %arg1[%arg4, %c0+1, %c0_0+2, %c0_1+2] : memref<10x3x3x3xf32>
          %140 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %141 = arith.mulf %138, %139 : f32
          %142 = arith.addf %140, %141 : f32
          affine.store %142, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %146 = affine.load %arg0[%arg3, %c0+2, %arg5+%c0_0, %arg6+%c0_1] : memref<10x3x256x256xf32>
          %147 = affine.load %arg1[%arg4, %c0+2, %c0_0, %c0_1] : memref<10x3x3x3xf32>
          %148 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %149 = arith.mulf %146, %147 : f32
          %150 = arith.addf %148, %149 : f32
          affine.store %150, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %154 = affine.load %arg0[%arg3, %c0+2, %arg5+%c0_0, %arg6+%c0_1+1] : memref<10x3x256x256xf32>
          %155 = affine.load %arg1[%arg4, %c0+2, %c0_0, %c0_1+1] : memref<10x3x3x3xf32>
          %156 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %157 = arith.mulf %154, %155 : f32
          %158 = arith.addf %156, %157 : f32
          affine.store %158, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %162 = affine.load %arg0[%arg3, %c0+2, %arg5+%c0_0, %arg6+%c0_1+2] : memref<10x3x256x256xf32>
          %163 = affine.load %arg1[%arg4, %c0+2, %c0_0, %c0_1+2] : memref<10x3x3x3xf32>
          %164 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %165 = arith.mulf %162, %163 : f32
          %166 = arith.addf %164, %165 : f32
          affine.store %166, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %170 = affine.load %arg0[%arg3, %c0+2, %arg5+%c0_0+1, %arg6+%c0_1] : memref<10x3x256x256xf32>
          %171 = affine.load %arg1[%arg4, %c0+2, %c0_0+1, %c0_1] : memref<10x3x3x3xf32>
          %172 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %173 = arith.mulf %170, %171 : f32
          %174 = arith.addf %172, %173 : f32
          affine.store %174, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %178 = affine.load %arg0[%arg3, %c0+2, %arg5+%c0_0+1, %arg6+%c0_1+1] : memref<10x3x256x256xf32>
          %179 = affine.load %arg1[%arg4, %c0+2, %c0_0+1, %c0_1+1] : memref<10x3x3x3xf32>
          %180 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %181 = arith.mulf %178, %179 : f32
          %182 = arith.addf %180, %181 : f32
          affine.store %182, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %186 = affine.load %arg0[%arg3, %c0+2, %arg5+%c0_0+1, %arg6+%c0_1+2] : memref<10x3x256x256xf32>
          %187 = affine.load %arg1[%arg4, %c0+2, %c0_0+1, %c0_1+2] : memref<10x3x3x3xf32>
          %188 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %189 = arith.mulf %186, %187 : f32
          %190 = arith.addf %188, %189 : f32
          affine.store %190, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %194 = affine.load %arg0[%arg3, %c0+2, %arg5+%c0_0+2, %arg6+%c0_1] : memref<10x3x256x256xf32>
          %195 = affine.load %arg1[%arg4, %c0+2, %c0_0+2, %c0_1] : memref<10x3x3x3xf32>
          %196 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %197 = arith.mulf %194, %195 : f32
          %198 = arith.addf %196, %197 : f32
          affine.store %198, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %202 = affine.load %arg0[%arg3, %c0+2, %arg5+%c0_0+2, %arg6+%c0_1+1] : memref<10x3x256x256xf32>
          %203 = affine.load %arg1[%arg4, %c0+2, %c0_0+2, %c0_1+1] : memref<10x3x3x3xf32>
          %204 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %205 = arith.mulf %202, %203 : f32
          %206 = arith.addf %204, %205 : f32
          affine.store %206, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %210 = affine.load %arg0[%arg3, %c0+2, %arg5+%c0_0+2, %arg6+%c0_1+2] : memref<10x3x256x256xf32>
          %211 = affine.load %arg1[%arg4, %c0+2, %c0_0+2, %c0_1+2] : memref<10x3x3x3xf32>
          %212 = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
          %213 = arith.mulf %210, %211 : f32
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<10x10x254x254xf32>
        }
      }
    }
  }
  return
}

//CHECK-NEXT: %c2 = arith.constant 2 : index
//CHECK-NEXT: %c1 = arith.constant 1 : index
//CHECK-NEXT: %c0 = arith.constant 0 : index
//CHECK-NEXT: %c0_0 = arith.constant 0 : index
//CHECK-NEXT: %c10 = arith.constant 10 : index
//CHECK-NEXT: %c1_1 = arith.constant 1 : index
//CHECK-NEXT: scf.for %arg3 = %c0_0 to %c10 step %c1_1 {
//CHECK-NEXT:   %c0_2 = arith.constant 0 : index
//CHECK-NEXT:   %c10_3 = arith.constant 10 : index
//CHECK-NEXT:   %c1_4 = arith.constant 1 : index
//CHECK-NEXT:   scf.for %arg4 = %c0_2 to %c10_3 step %c1_4 {
//CHECK-NEXT:     %0 = aievec.upd %arg1[%arg4, %c0, %c0, %c0] {index = 0 : i8, offset = 0 : i32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %1 = aievec.upd %arg1[%arg4, %c0, %c2, %c2] {index = 0 : i8, offset = 0 : i32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %2 = aievec.upd %arg1[%arg4, %c1, %c2, %c1] {index = 0 : i8, offset = 0 : i32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %3 = aievec.upd %arg1[%arg4, %c2, %c2, %c0] {index = 0 : i8, offset = 0 : i32} : memref<10x3x3x3xf32>, vector<8xf32>
//CHECK-NEXT:     %c0_5 = arith.constant 0 : index
//CHECK-NEXT:     %c254 = arith.constant 254 : index
//CHECK-NEXT:     %c1_6 = arith.constant 1 : index
//CHECK-NEXT:     scf.for %arg5 = %c0_5 to %c254 step %c1_6 {
//CHECK-NEXT:       %c1_7 = arith.constant 1 : index
//CHECK-NEXT:       %4 = arith.addi %arg5, %c1_7 : index
//CHECK-NEXT:       %c2_8 = arith.constant 2 : index
//CHECK-NEXT:       %5 = arith.addi %arg5, %c2_8 : index
//CHECK-NEXT:       %c0_9 = arith.constant 0 : index
//CHECK-NEXT:       %c254_10 = arith.constant 254 : index
//CHECK-NEXT:       %c8 = arith.constant 8 : index
//CHECK-NEXT:       scf.for %arg6 = %c0_9 to %c254_10 step %c8 {
//CHECK-NEXT:         %6 = aievec.upd %arg0[%arg3, %c0, %arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %7 = aievec.upd %arg2[%arg3, %arg4, %arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x10x254x254xf32>, vector<8xf32>
//CHECK-NEXT:         %8 = aievec_aie1.mac %6, %0, %7 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %c1_11 = arith.constant 1 : index
//CHECK-NEXT:         %9 = arith.addi %arg6, %c1_11 : index
//CHECK-NEXT:         %10 = aievec.upd %arg0[%arg3, %c0, %arg5, %9], %6 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %11 = aievec_aie1.mac %10, %0, %8 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %12 = aievec_aie1.mac %10, %0, %11 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %13 = aievec.upd %arg0[%arg3, %c0, %4, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %14 = aievec_aie1.mac %13, %0, %12 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %15 = aievec.upd %arg0[%arg3, %c0, %4, %9], %13 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %16 = aievec_aie1.mac %15, %0, %14 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %17 = aievec_aie1.mac %15, %0, %16 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %18 = aievec.upd %arg0[%arg3, %c0, %5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %19 = aievec_aie1.mac %18, %0, %17 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %20 = aievec.upd %arg0[%arg3, %c0, %5, %9], %18 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %21 = aievec_aie1.mac %20, %0, %19 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %22 = aievec_aie1.mac %20, %1, %21 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %23 = aievec.upd %arg0[%arg3, %c1, %arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %24 = aievec_aie1.mac %23, %1, %22 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %25 = aievec.upd %arg0[%arg3, %c1, %arg5, %9], %23 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %26 = aievec_aie1.mac %25, %1, %24 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %27 = aievec_aie1.mac %25, %1, %26 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %28 = aievec.upd %arg0[%arg3, %c1, %4, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %29 = aievec_aie1.mac %28, %1, %27 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %30 = aievec.upd %arg0[%arg3, %c1, %4, %9], %28 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %31 = aievec_aie1.mac %30, %1, %29 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %32 = aievec_aie1.mac %30, %1, %31 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %33 = aievec.upd %arg0[%arg3, %c1, %5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %34 = aievec_aie1.mac %33, %1, %32 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %35 = aievec.upd %arg0[%arg3, %c1, %5, %9], %33 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %36 = aievec_aie1.mac %35, %2, %34 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %37 = aievec_aie1.mac %35, %2, %36 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %38 = aievec.upd %arg0[%arg3, %c2, %arg5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %39 = aievec_aie1.mac %38, %2, %37 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %40 = aievec.upd %arg0[%arg3, %c2, %arg5, %9], %38 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %41 = aievec_aie1.mac %40, %2, %39 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "3"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %42 = aievec_aie1.mac %40, %2, %41 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "4"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %43 = aievec.upd %arg0[%arg3, %c2, %4, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %44 = aievec_aie1.mac %43, %2, %42 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "5"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %45 = aievec.upd %arg0[%arg3, %c2, %4, %9], %43 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %46 = aievec_aie1.mac %45, %2, %44 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "6"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %47 = aievec_aie1.mac %45, %2, %46 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "7"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %48 = aievec.upd %arg0[%arg3, %c2, %5, %arg6] {index = 0 : i8, offset = 0 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %49 = aievec_aie1.mac %48, %3, %47 {xoffsets = "0x76543210", xstart = "0", zoffsets = "0x00000000", zstart = "0"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %50 = aievec.upd %arg0[%arg3, %c2, %5, %9], %48 {index = 1 : i8, offset = 224 : i32} : memref<10x3x256x256xf32>, vector<16xf32>
//CHECK-NEXT:         %51 = aievec_aie1.mac %50, %3, %49 {xoffsets = "0x76543210", xstart = "1", zoffsets = "0x00000000", zstart = "1"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         %52 = aievec_aie1.mac %50, %3, %51 {xoffsets = "0x76543210", xstart = "2", zoffsets = "0x00000000", zstart = "2"} : vector<16xf32>, vector<8xf32>, vector<8xf32>
//CHECK-NEXT:         vector.transfer_write %52, %arg2[%arg3, %arg4, %arg5, %arg6] {in_bounds = [true]} : vector<8xf32>, memref<10x10x254x254xf32>
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:   }
//CHECK-NEXT: }
