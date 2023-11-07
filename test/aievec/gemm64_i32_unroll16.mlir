// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=16" -aieml=true --aie-vectorize

//CHECK-LABEL: func.func @matmul(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
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

//CHECK-NEXT:    %c0 = arith.constant 0 : index
//CHECK-NEXT:    %c64 = arith.constant 64 : index
//CHECK-NEXT:    %c1 = arith.constant 1 : index
//CHECK-NEXT:    scf.for %arg3 = %c0 to %c64 step %c1 {
//CHECK-NEXT:      %c0_0 = arith.constant 0 : index
//CHECK-NEXT:      %c64_1 = arith.constant 64 : index
//CHECK-NEXT:      %c16 = arith.constant 16 : index
//CHECK-NEXT:      scf.for %arg4 = %c0_0 to %c64_1 step %c16 {
//CHECK-NEXT:        %0 = aievec.upd %arg2[%arg3, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:       %1 = aievec.ups %0 {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//CHECK-NEXT:        %c0_2 = arith.constant 0 : index
//CHECK-NEXT:        %c64_3 = arith.constant 64 : index
//CHECK-NEXT:        %c16_4 = arith.constant 16 : index
//CHECK-NEXT:        scf.for %arg5 = %c0_2 to %c64_3 step %c16_4 {
//CHECK-NEXT:          %2 = aievec.upd %arg0[%arg3, %arg5] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %3 = aievec.upd %arg1[%arg5, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %4 = aievec.broadcast %2 {idx = 0 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %5 = aievec.mac_elem %3, %4, %1 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c1_5 = arith.constant 1 : index
//CHECK-NEXT:          %6 = arith.addi %arg5, %c1_5 : index
//CHECK-NEXT:          %7 = aievec.upd %arg1[%6, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %8 = aievec.broadcast %2 {idx = 1 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %9 = aievec.mac_elem %7, %8, %5 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c2 = arith.constant 2 : index
//CHECK-NEXT:          %10 = arith.addi %arg5, %c2 : index
//CHECK-NEXT:          %11 = aievec.upd %arg1[%10, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %12 = aievec.broadcast %2 {idx = 2 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %13 = aievec.mac_elem %11, %12, %9 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c3 = arith.constant 3 : index
//CHECK-NEXT:          %14 = arith.addi %arg5, %c3 : index
//CHECK-NEXT:          %15 = aievec.upd %arg1[%14, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %16 = aievec.broadcast %2 {idx = 3 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %17 = aievec.mac_elem %15, %16, %13 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c4 = arith.constant 4 : index
//CHECK-NEXT:          %18 = arith.addi %arg5, %c4 : index
//CHECK-NEXT:          %19 = aievec.upd %arg1[%18, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %20 = aievec.broadcast %2 {idx = 4 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %21 = aievec.mac_elem %19, %20, %17 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c5 = arith.constant 5 : index
//CHECK-NEXT:          %22 = arith.addi %arg5, %c5 : index
//CHECK-NEXT:          %23 = aievec.upd %arg1[%22, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %24 = aievec.broadcast %2 {idx = 5 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %25 = aievec.mac_elem %23, %24, %21 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c6 = arith.constant 6 : index
//CHECK-NEXT:          %26 = arith.addi %arg5, %c6 : index
//CHECK-NEXT:          %27 = aievec.upd %arg1[%26, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %28 = aievec.broadcast %2 {idx = 6 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %29 = aievec.mac_elem %27, %28, %25 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c7 = arith.constant 7 : index
//CHECK-NEXT:          %30 = arith.addi %arg5, %c7 : index
//CHECK-NEXT:          %31 = aievec.upd %arg1[%30, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %32 = aievec.broadcast %2 {idx = 7 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %33 = aievec.mac_elem %31, %32, %29 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c8 = arith.constant 8 : index
//CHECK-NEXT:          %34 = arith.addi %arg5, %c8 : index
//CHECK-NEXT:          %35 = aievec.upd %arg1[%34, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %36 = aievec.broadcast %2 {idx = 8 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %37 = aievec.mac_elem %35, %36, %33 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c9 = arith.constant 9 : index
//CHECK-NEXT:          %38 = arith.addi %arg5, %c9 : index
//CHECK-NEXT:          %39 = aievec.upd %arg1[%38, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %40 = aievec.broadcast %2 {idx = 9 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %41 = aievec.mac_elem %39, %40, %37 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c10 = arith.constant 10 : index
//CHECK-NEXT:          %42 = arith.addi %arg5, %c10 : index
//CHECK-NEXT:          %43 = aievec.upd %arg1[%42, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %44 = aievec.broadcast %2 {idx = 10 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %45 = aievec.mac_elem %43, %44, %41 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c11 = arith.constant 11 : index
//CHECK-NEXT:          %46 = arith.addi %arg5, %c11 : index
//CHECK-NEXT:          %47 = aievec.upd %arg1[%46, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %48 = aievec.broadcast %2 {idx = 11 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %49 = aievec.mac_elem %47, %48, %45 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c12 = arith.constant 12 : index
//CHECK-NEXT:          %50 = arith.addi %arg5, %c12 : index
//CHECK-NEXT:          %51 = aievec.upd %arg1[%50, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %52 = aievec.broadcast %2 {idx = 12 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %53 = aievec.mac_elem %51, %52, %49 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c13 = arith.constant 13 : index
//CHECK-NEXT:          %54 = arith.addi %arg5, %c13 : index
//CHECK-NEXT:          %55 = aievec.upd %arg1[%54, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %56 = aievec.broadcast %2 {idx = 13 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %57 = aievec.mac_elem %55, %56, %53 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c14 = arith.constant 14 : index
//CHECK-NEXT:          %58 = arith.addi %arg5, %c14 : index
//CHECK-NEXT:          %59 = aievec.upd %arg1[%58, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %60 = aievec.broadcast %2 {idx = 14 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %61 = aievec.mac_elem %59, %60, %57 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %c15 = arith.constant 15 : index
//CHECK-NEXT:          %62 = arith.addi %arg5, %c15 : index
//CHECK-NEXT:          %63 = aievec.upd %arg1[%62, %arg4] {index = 0 : i8, offset = 0 : i32} : memref<64x64xi32>, vector<16xi32>
//CHECK-NEXT:          %64 = aievec.broadcast %2 {idx = 15 : i8} : vector<16xi32>, vector<16xi32>
//CHECK-NEXT:          %65 = aievec.mac_elem %63, %64, %61 : vector<16xi32>, vector<16xi32>, vector<16xi64>
//CHECK-NEXT:          %66 = aievec.srs %65 {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//CHECK-NEXT:          vector.transfer_write %66, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<16xi32>, memref<64x64xi32>
