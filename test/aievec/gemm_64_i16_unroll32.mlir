// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" -aieml=true --aie-vectorize


//CHECK-LABEL:  func.func @matmul(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
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

//CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
//CHECK-NEXT:    affine.for %arg3 = 0 to 64 {
//CHECK-NEXT:      affine.for %arg4 = 0 to 64 step 32 {
//CHECK-NEXT:        affine.for %arg5 = 0 to 64 step 16 {
//CHECK-NEXT:          %0 = vector.transfer_read %arg0[%arg3, %arg5], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %1 = vector.transfer_read %arg1[%arg5, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %2 = arith.muli %0, %1 : vector<32xi32>
//CHECK-NEXT:          %3 = vector.transfer_read %arg2[%arg3, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %4 = arith.addi %3, %2 : vector<32xi32>
//CHECK-NEXT:          %5 = affine.apply #map1(%arg5)
//CHECK-NEXT:          %6 = vector.transfer_read %arg0[%arg3, %5], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %7 = affine.apply #map1(%arg5)
//CHECK-NEXT:          %8 = vector.transfer_read %arg1[%7, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %9 = arith.muli %6, %8 : vector<32xi32>
//CHECK-NEXT:          %10 = arith.addi %4, %9 : vector<32xi32>
//CHECK-NEXT:          %11 = affine.apply #map2(%arg5)
//CHECK-NEXT:          %12 = vector.transfer_read %arg0[%arg3, %11], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %13 = affine.apply #map2(%arg5)
//CHECK-NEXT:          %14 = vector.transfer_read %arg1[%13, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %15 = arith.muli %12, %14 : vector<32xi32>
//CHECK-NEXT:          %16 = arith.addi %10, %15 : vector<32xi32>
//CHECK-NEXT:          %17 = affine.apply #map3(%arg5)
//CHECK-NEXT:          %18 = vector.transfer_read %arg0[%arg3, %17], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %19 = affine.apply #map3(%arg5)
//CHECK-NEXT:          %20 = vector.transfer_read %arg1[%19, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:         %21 = arith.muli %18, %20 : vector<32xi32>
//CHECK-NEXT:          %22 = arith.addi %16, %21 : vector<32xi32>
//CHECK-NEXT:          %23 = affine.apply #map4(%arg5)
//CHECK-NEXT:          %24 = vector.transfer_read %arg0[%arg3, %23], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %25 = affine.apply #map4(%arg5)
//CHECK-NEXT:          %26 = vector.transfer_read %arg1[%25, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %27 = arith.muli %24, %26 : vector<32xi32>
//CHECK-NEXT:          %28 = arith.addi %22, %27 : vector<32xi32>
//CHECK-NEXT:          %29 = affine.apply #map5(%arg5)
//CHECK-NEXT:          %30 = vector.transfer_read %arg0[%arg3, %29], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %31 = affine.apply #map5(%arg5)
//CHECK-NEXT:          %32 = vector.transfer_read %arg1[%31, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %33 = arith.muli %30, %32 : vector<32xi32>
//CHECK-NEXT:          %34 = arith.addi %28, %33 : vector<32xi32>
//CHECK-NEXT:          %35 = affine.apply #map6(%arg5)
//CHECK-NEXT:          %36 = vector.transfer_read %arg0[%arg3, %35], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %37 = affine.apply #map6(%arg5)
//CHECK-NEXT:          %38 = vector.transfer_read %arg1[%37, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %39 = arith.muli %36, %38 : vector<32xi32>
//CHECK-NEXT:          %40 = arith.addi %34, %39 : vector<32xi32>
//CHECK-NEXT:          %41 = affine.apply #map7(%arg5)
//CHECK-NEXT:          %42 = vector.transfer_read %arg0[%arg3, %41], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %43 = affine.apply #map7(%arg5)
//CHECK-NEXT:          %44 = vector.transfer_read %arg1[%43, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %45 = arith.muli %42, %44 : vector<32xi32>
//CHECK-NEXT:          %46 = arith.addi %40, %45 : vector<32xi32>
//CHECK-NEXT:          %47 = affine.apply #map8(%arg5)
//CHECK-NEXT:          %48 = vector.transfer_read %arg0[%arg3, %47], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %49 = affine.apply #map8(%arg5)
//CHECK-NEXT:          %50 = vector.transfer_read %arg1[%49, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %51 = arith.muli %48, %50 : vector<32xi32>
//CHECK-NEXT:          %52 = arith.addi %46, %51 : vector<32xi32>
//CHECK-NEXT:          %53 = affine.apply #map9(%arg5)
//CHECK-NEXT:          %54 = vector.transfer_read %arg0[%arg3, %53], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %55 = affine.apply #map9(%arg5)
//CHECK-NEXT:          %56 = vector.transfer_read %arg1[%55, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %57 = arith.muli %54, %56 : vector<32xi32>
//CHECK-NEXT:          %58 = arith.addi %52, %57 : vector<32xi32>
//CHECK-NEXT:          %59 = affine.apply #map10(%arg5)
//CHECK-NEXT:          %60 = vector.transfer_read %arg0[%arg3, %59], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %61 = affine.apply #map10(%arg5)
//CHECK-NEXT:          %62 = vector.transfer_read %arg1[%61, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %63 = arith.muli %60, %62 : vector<32xi32>
//CHECK-NEXT:          %64 = arith.addi %58, %63 : vector<32xi32>
//CHECK-NEXT:          %65 = affine.apply #map11(%arg5)
//CHECK-NEXT:         %66 = vector.transfer_read %arg0[%arg3, %65], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %67 = affine.apply #map11(%arg5)
//CHECK-NEXT:          %68 = vector.transfer_read %arg1[%67, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %69 = arith.muli %66, %68 : vector<32xi32>
//CHECK-NEXT:          %70 = arith.addi %64, %69 : vector<32xi32>
//CHECK-NEXT:         %71 = affine.apply #map12(%arg5)
//CHECK-NEXT:          %72 = vector.transfer_read %arg0[%arg3, %71], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %73 = affine.apply #map12(%arg5)
//CHECK-NEXT:          %74 = vector.transfer_read %arg1[%73, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %75 = arith.muli %72, %74 : vector<32xi32>
//CHECK-NEXT:          %76 = arith.addi %70, %75 : vector<32xi32>
//CHECK-NEXT:          %77 = affine.apply #map13(%arg5)
//CHECK-NEXT:          %78 = vector.transfer_read %arg0[%arg3, %77], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %79 = affine.apply #map13(%arg5)
//CHECK-NEXT:          %80 = vector.transfer_read %arg1[%79, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %81 = arith.muli %78, %80 : vector<32xi32>
//CHECK-NEXT:          %82 = arith.addi %76, %81 : vector<32xi32>
//CHECK-NEXT:          %83 = affine.apply #map14(%arg5)
//CHECK-NEXT:          %84 = vector.transfer_read %arg0[%arg3, %83], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %85 = affine.apply #map14(%arg5)
//CHECK-NEXT:          %86 = vector.transfer_read %arg1[%85, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %87 = arith.muli %84, %86 : vector<32xi32>
//CHECK-NEXT:          %88 = arith.addi %82, %87 : vector<32xi32>
//CHECK-NEXT:          %89 = affine.apply #map15(%arg5)
//CHECK-NEXT:          %90 = vector.transfer_read %arg0[%arg3, %89], %c0_i32 {in_bounds = [true], permutation_map = #map0} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %91 = affine.apply #map15(%arg5)
//CHECK-NEXT:          %92 = vector.transfer_read %arg1[%91, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<32xi32>
//CHECK-NEXT:          %93 = arith.muli %90, %92 : vector<32xi32>
//CHECK-NEXT:          %94 = arith.addi %88, %93 : vector<32xi32>
//CHECK-NEXT:          vector.transfer_write %94, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<32xi32>, memref<64x64xi32>
