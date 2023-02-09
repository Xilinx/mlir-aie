// RUN: aie-opt %s --convert-vector-to-aievec="aie-target=aieml" | FileCheck %s

module {
  func.func @matmul(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 step 16 {
        affine.for %arg5 = 0 to 64 step 16 {
          %0 = vector.transfer_read %arg0[%arg3, %arg5], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %1 = vector.transfer_read %arg1[%arg5, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %2 = arith.muli %0, %1 : vector<16xi32>
          %3 = vector.transfer_read %arg2[%arg3, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %4 = arith.addi %3, %2 : vector<16xi32>
          %5 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg5)
          %6 = vector.transfer_read %arg0[%arg3, %5], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %7 = affine.apply affine_map<(d0) -> (d0 + 1)>(%arg5)
          %8 = vector.transfer_read %arg1[%7, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %9 = arith.muli %6, %8 : vector<16xi32>
          %10 = arith.addi %4, %9 : vector<16xi32>
          %11 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg5)
          %12 = vector.transfer_read %arg0[%arg3, %11], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %13 = affine.apply affine_map<(d0) -> (d0 + 2)>(%arg5)
          %14 = vector.transfer_read %arg1[%13, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %15 = arith.muli %12, %14 : vector<16xi32>
          %16 = arith.addi %10, %15 : vector<16xi32>
          %17 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg5)
          %18 = vector.transfer_read %arg0[%arg3, %17], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %19 = affine.apply affine_map<(d0) -> (d0 + 3)>(%arg5)
          %20 = vector.transfer_read %arg1[%19, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %21 = arith.muli %18, %20 : vector<16xi32>
          %22 = arith.addi %16, %21 : vector<16xi32>
          %23 = affine.apply affine_map<(d0) -> (d0 + 4)>(%arg5)
          %24 = vector.transfer_read %arg0[%arg3, %23], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %25 = affine.apply affine_map<(d0) -> (d0 + 4)>(%arg5)
          %26 = vector.transfer_read %arg1[%25, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %27 = arith.muli %24, %26 : vector<16xi32>
          %28 = arith.addi %22, %27 : vector<16xi32>
          %29 = affine.apply affine_map<(d0) -> (d0 + 5)>(%arg5)
          %30 = vector.transfer_read %arg0[%arg3, %29], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %31 = affine.apply affine_map<(d0) -> (d0 + 5)>(%arg5)
          %32 = vector.transfer_read %arg1[%31, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %33 = arith.muli %30, %32 : vector<16xi32>
          %34 = arith.addi %28, %33 : vector<16xi32>
          %35 = affine.apply affine_map<(d0) -> (d0 + 6)>(%arg5)
          %36 = vector.transfer_read %arg0[%arg3, %35], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %37 = affine.apply affine_map<(d0) -> (d0 + 6)>(%arg5)
          %38 = vector.transfer_read %arg1[%37, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %39 = arith.muli %36, %38 : vector<16xi32>
          %40 = arith.addi %34, %39 : vector<16xi32>
          %41 = affine.apply affine_map<(d0) -> (d0 + 7)>(%arg5)
          %42 = vector.transfer_read %arg0[%arg3, %41], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %43 = affine.apply affine_map<(d0) -> (d0 + 7)>(%arg5)
          %44 = vector.transfer_read %arg1[%43, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %45 = arith.muli %42, %44 : vector<16xi32>
          %46 = arith.addi %40, %45 : vector<16xi32>
          %47 = affine.apply affine_map<(d0) -> (d0 + 8)>(%arg5)
          %48 = vector.transfer_read %arg0[%arg3, %47], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %49 = affine.apply affine_map<(d0) -> (d0 + 8)>(%arg5)
          %50 = vector.transfer_read %arg1[%49, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %51 = arith.muli %48, %50 : vector<16xi32>
          %52 = arith.addi %46, %51 : vector<16xi32>
          %53 = affine.apply affine_map<(d0) -> (d0 + 9)>(%arg5)
          %54 = vector.transfer_read %arg0[%arg3, %53], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %55 = affine.apply affine_map<(d0) -> (d0 + 9)>(%arg5)
          %56 = vector.transfer_read %arg1[%55, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %57 = arith.muli %54, %56 : vector<16xi32>
          %58 = arith.addi %52, %57 : vector<16xi32>
          %59 = affine.apply affine_map<(d0) -> (d0 + 10)>(%arg5)
          %60 = vector.transfer_read %arg0[%arg3, %59], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %61 = affine.apply affine_map<(d0) -> (d0 + 10)>(%arg5)
          %62 = vector.transfer_read %arg1[%61, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %63 = arith.muli %60, %62 : vector<16xi32>
          %64 = arith.addi %58, %63 : vector<16xi32>
          %65 = affine.apply affine_map<(d0) -> (d0 + 11)>(%arg5)
          %66 = vector.transfer_read %arg0[%arg3, %65], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %67 = affine.apply affine_map<(d0) -> (d0 + 11)>(%arg5)
          %68 = vector.transfer_read %arg1[%67, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %69 = arith.muli %66, %68 : vector<16xi32>
          %70 = arith.addi %64, %69 : vector<16xi32>
          %71 = affine.apply affine_map<(d0) -> (d0 + 12)>(%arg5)
          %72 = vector.transfer_read %arg0[%arg3, %71], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %73 = affine.apply affine_map<(d0) -> (d0 + 12)>(%arg5)
          %74 = vector.transfer_read %arg1[%73, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %75 = arith.muli %72, %74 : vector<16xi32>
          %76 = arith.addi %70, %75 : vector<16xi32>
          %77 = affine.apply affine_map<(d0) -> (d0 + 13)>(%arg5)
          %78 = vector.transfer_read %arg0[%arg3, %77], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %79 = affine.apply affine_map<(d0) -> (d0 + 13)>(%arg5)
          %80 = vector.transfer_read %arg1[%79, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %81 = arith.muli %78, %80 : vector<16xi32>
          %82 = arith.addi %76, %81 : vector<16xi32>
          %83 = affine.apply affine_map<(d0) -> (d0 + 14)>(%arg5)
          %84 = vector.transfer_read %arg0[%arg3, %83], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %85 = affine.apply affine_map<(d0) -> (d0 + 14)>(%arg5)
          %86 = vector.transfer_read %arg1[%85, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %87 = arith.muli %84, %86 : vector<16xi32>
          %88 = arith.addi %82, %87 : vector<16xi32>
          %89 = affine.apply affine_map<(d0) -> (d0 + 15)>(%arg5)
          %90 = vector.transfer_read %arg0[%arg3, %89], %c0_i32 {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
          %91 = affine.apply affine_map<(d0) -> (d0 + 15)>(%arg5)
          %92 = vector.transfer_read %arg1[%91, %arg4], %c0_i32 {in_bounds = [true]} : memref<64x64xi32>, vector<16xi32>
          %93 = arith.muli %90, %92 : vector<16xi32>
          %94 = arith.addi %88, %93 : vector<16xi32>
          vector.transfer_write %94, %arg2[%arg3, %arg4] {in_bounds = [true]} : vector<16xi32>, memref<64x64xi32>
        }
      }
    }
    return
  }
}

// CHECK-LABEL: @matmul
// CHECK-SAME: %[[A0:[0-9a-zA-Z]*]]: memref<64x64xi32>
// CHECK-SAME: %[[A1:[0-9a-zA-Z]*]]: memref<64x64xi32>
// CHECK-SAME: %[[A2:[0-9a-zA-Z]*]]: memref<64x64xi32>
//      CHECK:    affine.for %[[A3:.*]] = 0 to 64 {
//      CHECK:      affine.for %[[A4:.*]] = 0 to 64 step 16 {
//      CHECK:        affine.for %[[A5:.*]] = 0 to 64 step 16 {
//      CHECK:          %[[T0:.*]] = aievec.upd %[[A0]][%[[A3:.*]], %[[A5:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T1:.*]] = aievec.broadcast %[[T0:.*]] {idx = 0 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T2:.*]] = aievec.upd %[[A1]][%[[A5:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T3:.*]] = aievec.upd %[[A2]][%[[A3:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T4:.*]] = aievec.ups %[[T3:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T5:.*]] = aievec.mac_elem %[[T2:.*]], %[[T1:.*]], %[[T4:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T6:.*]] = aievec.srs %[[T5:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T7:.*]] = aievec.broadcast %[[T0:.*]] {idx = 1 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T8:.*]] = affine.apply #map(%[[A5:.*]])
//      CHECK:          %[[T9:.*]] = aievec.upd %[[A1]][%[[T8:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T10:.*]] = aievec.ups %[[T6:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T11:.*]] = aievec.mac_elem %[[T9:.*]], %[[T7:.*]], %[[T10:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T12:.*]] = aievec.srs %[[T11:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T13:.*]] = aievec.broadcast %[[T0:.*]] {idx = 2 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T14:.*]] = affine.apply #map1(%[[A5:.*]])
//      CHECK:          %[[T15:.*]] = aievec.upd %[[A1]][%[[T14:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T16:.*]] = aievec.ups %[[T12:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T17:.*]] = aievec.mac_elem %[[T15:.*]], %[[T13:.*]], %[[T16:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T18:.*]] = aievec.srs %[[T17:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T19:.*]] = aievec.broadcast %[[T0:.*]] {idx = 3 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T20:.*]] = affine.apply #map2(%[[A5:.*]])
//      CHECK:          %[[T21:.*]] = aievec.upd %[[A1]][%[[T20:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T22:.*]] = aievec.ups %[[T18:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T23:.*]] = aievec.mac_elem %[[T21:.*]], %[[T19:.*]], %[[T22:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T24:.*]] = aievec.srs %[[T23:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T25:.*]] = aievec.broadcast %[[T0:.*]] {idx = 4 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T26:.*]] = affine.apply #map3(%[[A5:.*]])
//      CHECK:          %[[T27:.*]] = aievec.upd %[[A1]][%[[T26:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T28:.*]] = aievec.ups %[[T24:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T29:.*]] = aievec.mac_elem %[[T27:.*]], %[[T25:.*]], %[[T28:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T30:.*]] = aievec.srs %[[T29:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T31:.*]] = aievec.broadcast %[[T0:.*]] {idx = 5 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T32:.*]] = affine.apply #map4(%[[A5:.*]])
//      CHECK:          %[[T33:.*]] = aievec.upd %[[A1]][%[[T32:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T34:.*]] = aievec.ups %[[T30:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T35:.*]] = aievec.mac_elem %[[T33:.*]], %[[T31:.*]], %[[T34:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T36:.*]] = aievec.srs %[[T35:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T37:.*]] = aievec.broadcast %[[T0:.*]] {idx = 6 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T38:.*]] = affine.apply #map5(%[[A5:.*]])
//      CHECK:          %[[T39:.*]] = aievec.upd %[[A1]][%[[T38:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T40:.*]] = aievec.ups %[[T36:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T41:.*]] = aievec.mac_elem %[[T39:.*]], %[[T37:.*]], %[[T40:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T42:.*]] = aievec.srs %[[T41:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T43:.*]] = aievec.broadcast %[[T0:.*]] {idx = 7 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T44:.*]] = affine.apply #map6(%[[A5:.*]])
//      CHECK:          %[[T45:.*]] = aievec.upd %[[A1]][%[[T44:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T46:.*]] = aievec.ups %[[T42:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T47:.*]] = aievec.mac_elem %[[T45:.*]], %[[T43:.*]], %[[T46:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T48:.*]] = aievec.srs %[[T47:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T49:.*]] = aievec.broadcast %[[T0:.*]] {idx = 8 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T50:.*]] = affine.apply #map7(%[[A5:.*]])
//      CHECK:          %[[T51:.*]] = aievec.upd %[[A1]][%[[T50:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T52:.*]] = aievec.ups %[[T48:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T53:.*]] = aievec.mac_elem %[[T51:.*]], %[[T49:.*]], %[[T52:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T54:.*]] = aievec.srs %[[T53:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T55:.*]] = aievec.broadcast %[[T0:.*]] {idx = 9 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T56:.*]] = affine.apply #map8(%[[A5:.*]])
//      CHECK:          %[[T57:.*]] = aievec.upd %[[A1]][%[[T56:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T58:.*]] = aievec.ups %[[T54:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T59:.*]] = aievec.mac_elem %[[T57:.*]], %[[T55:.*]], %[[T58:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T60:.*]] = aievec.srs %[[T59:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T61:.*]] = aievec.broadcast %[[T0:.*]] {idx = 10 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T62:.*]] = affine.apply #map9(%[[A5:.*]])
//      CHECK:          %[[T63:.*]] = aievec.upd %[[A1]][%[[T62:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T64:.*]] = aievec.ups %[[T60:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T65:.*]] = aievec.mac_elem %[[T63:.*]], %[[T61:.*]], %[[T64:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T66:.*]] = aievec.srs %[[T65:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T67:.*]] = aievec.broadcast %[[T0:.*]] {idx = 11 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T68:.*]] = affine.apply #map10(%[[A5:.*]])
//      CHECK:          %[[T69:.*]] = aievec.upd %[[A1]][%[[T68:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T70:.*]] = aievec.ups %[[T66:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T71:.*]] = aievec.mac_elem %[[T69:.*]], %[[T67:.*]], %[[T70:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T72:.*]] = aievec.srs %[[T71:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T73:.*]] = aievec.broadcast %[[T0:.*]] {idx = 12 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T74:.*]] = affine.apply #map11(%[[A5:.*]])
//      CHECK:          %[[T75:.*]] = aievec.upd %[[A1]][%[[T74:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T76:.*]] = aievec.ups %[[T72:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T77:.*]] = aievec.mac_elem %[[T75:.*]], %[[T73:.*]], %[[T76:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T78:.*]] = aievec.srs %[[T77:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T79:.*]] = aievec.broadcast %[[T0:.*]] {idx = 13 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T80:.*]] = affine.apply #map12(%[[A5:.*]])
//      CHECK:          %[[T81:.*]] = aievec.upd %[[A1]][%[[T80:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T82:.*]] = aievec.ups %[[T78:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T83:.*]] = aievec.mac_elem %[[T81:.*]], %[[T79:.*]], %[[T82:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T84:.*]] = aievec.srs %[[T83:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T85:.*]] = aievec.broadcast %[[T0:.*]] {idx = 14 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T86:.*]] = affine.apply #map13(%[[A5:.*]])
//      CHECK:          %[[T87:.*]] = aievec.upd %[[A1]][%[[T86:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T88:.*]] = aievec.ups %[[T84:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T89:.*]] = aievec.mac_elem %[[T87:.*]], %[[T85:.*]], %[[T88:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T90:.*]] = aievec.srs %[[T89:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          %[[T91:.*]] = aievec.broadcast %[[T0:.*]] {idx = 15 : i8} : vector<16xi32>, vector<16xi32>
//      CHECK:          %[[T92:.*]] = affine.apply #map14(%[[A5:.*]])
//      CHECK:          %[[T93:.*]] = aievec.upd %[[A1]][%[[T92:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : si32} : memref<64x64xi32>, vector<16xi32>
//      CHECK:          %[[T94:.*]] = aievec.ups %[[T90:.*]] {shift = 0 : i8} : vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T95:.*]] = aievec.mac_elem %[[T93:.*]], %[[T91:.*]], %[[T94:.*]] : vector<16xi32>, vector<16xi32>, vector<16xi64>
//      CHECK:          %[[T96:.*]] = aievec.srs %[[T95:.*]] {shift = 0 : i8} : vector<16xi64>, vector<16xi32>
//      CHECK:          vector.transfer_write %[[T96:.*]], %[[A2]][%[[A3:.*]], %[[A4:.*]]] {in_bounds = [true]} : vector<16xi32>, memref<64x64xi32>

