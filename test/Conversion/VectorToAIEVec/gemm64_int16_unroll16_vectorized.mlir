// RUN: aie-opt %s --convert-vector-to-aievec -lower-affine -canonicalize | FileCheck %s
#map = affine_map<(d0, d1) -> (0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0 + 2)>
#map3 = affine_map<(d0) -> (d0 + 3)>
#map4 = affine_map<(d0) -> (d0 + 4)>
#map5 = affine_map<(d0) -> (d0 + 5)>
#map6 = affine_map<(d0) -> (d0 + 6)>
#map7 = affine_map<(d0) -> (d0 + 7)>
#map8 = affine_map<(d0) -> (d0 + 8)>
#map9 = affine_map<(d0) -> (d0 + 9)>
#map10 = affine_map<(d0) -> (d0 + 10)>
#map11 = affine_map<(d0) -> (d0 + 11)>
#map12 = affine_map<(d0) -> (d0 + 12)>
#map13 = affine_map<(d0) -> (d0 + 13)>
#map14 = affine_map<(d0) -> (d0 + 14)>
#map15 = affine_map<(d0) -> (d0 + 15)>
// CHECK-LABEL: func.func @matmul(
// CHECK-SAME: %[[MA:[A-Za-z0-9]+]]: memref<?x64xi16>,
// CHECK-SAME: %[[MB:[A-Za-z0-9]+]]: memref<?x64xi16>,
// CHECK-SAME: %[[MC:[A-Za-z0-9]+]]: memref<?x64xi16>
func.func @matmul(%arg0: memref<?x64xi16>, %arg1: memref<?x64xi16>, %arg2: memref<?x64xi16>) {
  // CHECK-DAG: %[[C0I32:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
  // CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
  // CHECK-DAG: %[[C7:.*]] = arith.constant 7 : index
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
  // CHECK-DAG: %[[C9:.*]] = arith.constant 9 : index
  // CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
  // CHECK-DAG: %[[C11:.*]] = arith.constant 11 : index
  // CHECK-DAG: %[[C12:.*]] = arith.constant 12 : index
  // CHECK-DAG: %[[C13:.*]] = arith.constant 13 : index
  // CHECK-DAG: %[[C14:.*]] = arith.constant 14 : index
  // CHECK-DAG: %[[C15:.*]] = arith.constant 15 : index
  // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
  // CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
  %c0_i16 = arith.constant 0 : i16
  // CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
  affine.for %arg3 = 0 to 64 {
    // CHECK: scf.for %[[J:.*]] = %[[C0]] to %[[C64]] step %[[C16]] {
    affine.for %arg4 = 0 to 64 step 16 {
      // CHECK: %[[ACC0:.*]] = aievec.upd %[[MC]][%[[I]], %[[J]]]
      // CHECK-SAME:                    {index = 0 : i8, offset = 0 : i32}
      // CHECK-SAME:                    : memref<?x64xi16>, vector<16xi16>
      %0 = vector.transfer_read %arg2[%arg3, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
      // CHECK: %[[ACCn:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[C64]] step %[[C16]]
      // CHECK-SAME:                   iter_args(%[[ACCk:.*]] = %[[ACC0]]) -> (vector<16xi16>) {
      %1 = affine.for %arg5 = 0 to 64 step 16 iter_args(%arg6 = %0) -> (vector<16xi16>) {
        // CHECK: %[[VA:.*]] = aievec.upd %[[MA]][%[[I]], %[[K]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        // CHECK: %[[VB0:.*]] = aievec.upd %[[MB]][%[[K]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        // CHECK: %[[VC:.*]] = aievec.ups %[[ACCk]] {shift = 0 : i8} : vector<16xi16>, vector<16xi48>
        %2 = vector.transfer_read %arg0[%arg3, %arg5], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %3 = vector.transfer_read %arg1[%arg5, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %4 = arith.muli %2, %3 : vector<16xi16>
        %5 = arith.addi %arg6, %4 : vector<16xi16>
        // CHECK: %[[K1:.*]] = arith.addi %[[K]], %[[C1]] : index
        // CHECK: %[[VB1:.*]] = aievec.upd %[[MB]][%[[K1]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        // CHECK: %[[VB01:.*]] = aievec.concat %[[VB0]], %[[VB1]] : vector<16xi16>, vector<32xi16>
        // CHECK: %[[ACCk0:.*]] = aievec_aie1.mac %[[VB01]], %[[VA]], %[[VC]]
        // CHECK-SAME:                       {xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120",
        // CHECK-SAME:                        xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "0", zstep = "1"}
        // CHECK-SAME:                       : vector<32xi16>, vector<16xi16>, vector<16xi48>
        %6 = affine.apply #map1(%arg5)
        %7 = vector.transfer_read %arg0[%arg3, %6], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %8 = vector.transfer_read %arg1[%6, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %9 = arith.muli %7, %8 : vector<16xi16>
        %10 = arith.addi %5, %9 : vector<16xi16>
        // CHECK: %[[K2:.*]] = arith.addi %[[K]], %[[C2]] : index
        // CHECK: %[[VB2:.*]] = aievec.upd %[[MB]][%[[K2]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        %11 = affine.apply #map2(%arg5)
        %12 = vector.transfer_read %arg0[%arg3, %11], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %13 = vector.transfer_read %arg1[%11, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %14 = arith.muli %12, %13 : vector<16xi16>
        %15 = arith.addi %10, %14 : vector<16xi16>
        // CHECK: %[[K3:.*]] = arith.addi %[[K]], %[[C3]] : index
        // CHECK: %[[VB3:.*]] = aievec.upd %[[MB]][%[[K3]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        // CHECK: %[[VB23:.*]] = aievec.concat %[[VB2]], %[[VB3]] : vector<16xi16>, vector<32xi16>
        // CHECK: %[[ACCk2:.*]] = aievec_aie1.mac %[[VB23]], %[[VA]], %[[ACCk0]]
        // CHECK-SAME:                       {xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120",
        // CHECK-SAME:                        xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "2", zstep = "1"}
        %16 = affine.apply #map3(%arg5)
        %17 = vector.transfer_read %arg0[%arg3, %16], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %18 = vector.transfer_read %arg1[%16, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %19 = arith.muli %17, %18 : vector<16xi16>
        %20 = arith.addi %15, %19 : vector<16xi16>
        // CHECK: %[[K4:.*]] = arith.addi %[[K]], %[[C4]] : index
        // CHECK: %[[VB4:.*]] = aievec.upd %[[MB]][%[[K4]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        %21 = affine.apply #map4(%arg5)
        %22 = vector.transfer_read %arg0[%arg3, %21], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %23 = vector.transfer_read %arg1[%21, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %24 = arith.muli %22, %23 : vector<16xi16>
        %25 = arith.addi %20, %24 : vector<16xi16>
        // CHECK: %[[K5:.*]] = arith.addi %[[K]], %[[C5]] : index
        // CHECK: %[[VB5:.*]] = aievec.upd %[[MB]][%[[K5]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        // CHECK: %[[VB45:.*]] = aievec.concat %[[VB4]], %[[VB5]] : vector<16xi16>, vector<32xi16>
        // CHECK: %[[ACCk4:.*]] = aievec_aie1.mac %[[VB45]], %[[VA]], %[[ACCk2]]
        // CHECK-SAME:                       {xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120",
        // CHECK-SAME:                        xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "4", zstep = "1"}
        %26 = affine.apply #map5(%arg5)
        %27 = vector.transfer_read %arg0[%arg3, %26], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %28 = vector.transfer_read %arg1[%26, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %29 = arith.muli %27, %28 : vector<16xi16>
        %30 = arith.addi %25, %29 : vector<16xi16>
        // CHECK: %[[K6:.*]] = arith.addi %[[K]], %[[C6]] : index
        // CHECK: %[[VB6:.*]] = aievec.upd %[[MB]][%[[K6]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        %31 = affine.apply #map6(%arg5)
        %32 = vector.transfer_read %arg0[%arg3, %31], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %33 = vector.transfer_read %arg1[%31, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %34 = arith.muli %32, %33 : vector<16xi16>
        %35 = arith.addi %30, %34 : vector<16xi16>
        // CHECK: %[[K7:.*]] = arith.addi %[[K]], %[[C7]] : index
        // CHECK: %[[VB7:.*]] = aievec.upd %[[MB]][%[[K7]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        // CHECK: %[[VB67:.*]] = aievec.concat %[[VB6]], %[[VB7]] : vector<16xi16>, vector<32xi16>
        // CHECK: %[[ACCk6:.*]] = aievec_aie1.mac %[[VB67]], %[[VA]], %[[ACCk4]]
        // CHECK-SAME:                       {xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120",
        // CHECK-SAME:                        xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "6", zstep = "1"}
        %36 = affine.apply #map7(%arg5)
        %37 = vector.transfer_read %arg0[%arg3, %36], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %38 = vector.transfer_read %arg1[%36, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %39 = arith.muli %37, %38 : vector<16xi16>
        %40 = arith.addi %35, %39 : vector<16xi16>
        // CHECK: %[[K8:.*]] = arith.addi %[[K]], %[[C8]] : index
        // CHECK: %[[VB8:.*]] = aievec.upd %[[MB]][%[[K8]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        %41 = affine.apply #map8(%arg5)
        %42 = vector.transfer_read %arg0[%arg3, %41], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %43 = vector.transfer_read %arg1[%41, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %44 = arith.muli %42, %43 : vector<16xi16>
        %45 = arith.addi %40, %44 : vector<16xi16>
        // CHECK: %[[K9:.*]] = arith.addi %[[K]], %[[C9]] : index
        // CHECK: %[[VB9:.*]] = aievec.upd %[[MB]][%[[K9]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        // CHECK: %[[VB89:.*]] = aievec.concat %[[VB8]], %[[VB9]] : vector<16xi16>, vector<32xi16>
        // CHECK: %[[ACCk8:.*]] = aievec_aie1.mac %[[VB89]], %[[VA]], %[[ACCk6]]
        // CHECK-SAME:                       {xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120",
        // CHECK-SAME:                        xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "8", zstep = "1"}
        %46 = affine.apply #map9(%arg5)
        %47 = vector.transfer_read %arg0[%arg3, %46], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %48 = vector.transfer_read %arg1[%46, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %49 = arith.muli %47, %48 : vector<16xi16>
        %50 = arith.addi %45, %49 : vector<16xi16>
        // CHECK: %[[K10:.*]] = arith.addi %[[K]], %[[C10]] : index
        // CHECK: %[[VB10:.*]] = aievec.upd %[[MB]][%[[K10]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        %51 = affine.apply #map10(%arg5)
        %52 = vector.transfer_read %arg0[%arg3, %51], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %53 = vector.transfer_read %arg1[%51, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %54 = arith.muli %52, %53 : vector<16xi16>
        %55 = arith.addi %50, %54 : vector<16xi16>
        // CHECK: %[[K11:.*]] = arith.addi %[[K]], %[[C11]] : index
        // CHECK: %[[VB11:.*]] = aievec.upd %[[MB]][%[[K11]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        // CHECK: %[[VBab:.*]] = aievec.concat %[[VB10]], %[[VB11]] : vector<16xi16>, vector<32xi16>
        // CHECK: %[[ACCk10:.*]] = aievec_aie1.mac %[[VBab]], %[[VA]], %[[ACCk8]]
        // CHECK-SAME:                       {xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120",
        // CHECK-SAME:                        xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "10", zstep = "1"}
        %56 = affine.apply #map11(%arg5)
        %57 = vector.transfer_read %arg0[%arg3, %56], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %58 = vector.transfer_read %arg1[%56, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %59 = arith.muli %57, %58 : vector<16xi16>
        %60 = arith.addi %55, %59 : vector<16xi16>
        // CHECK: %[[K12:.*]] = arith.addi %[[K]], %[[C12]] : index
        // CHECK: %[[VB12:.*]] = aievec.upd %[[MB]][%[[K12]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        %61 = affine.apply #map12(%arg5)
        %62 = vector.transfer_read %arg0[%arg3, %61], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %63 = vector.transfer_read %arg1[%61, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %64 = arith.muli %62, %63 : vector<16xi16>
        %65 = arith.addi %60, %64 : vector<16xi16>
        // CHECK: %[[K13:.*]] = arith.addi %[[K]], %[[C13]] : index
        // CHECK: %[[VB13:.*]] = aievec.upd %[[MB]][%[[K13]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        // CHECK: %[[VBcd:.*]] = aievec.concat %[[VB12]], %[[VB13]] : vector<16xi16>, vector<32xi16>
        // CHECK: %[[ACCk12:.*]] = aievec_aie1.mac %[[VBcd]], %[[VA]], %[[ACCk10]]
        // CHECK-SAME:                       {xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120",
        // CHECK-SAME:                        xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "12", zstep = "1"}
        %66 = affine.apply #map13(%arg5)
        %67 = vector.transfer_read %arg0[%arg3, %66], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %68 = vector.transfer_read %arg1[%66, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %69 = arith.muli %67, %68 : vector<16xi16>
        %70 = arith.addi %65, %69 : vector<16xi16>
        // CHECK: %[[K14:.*]] = arith.addi %[[K]], %[[C14]] : index
        // CHECK: %[[VB14:.*]] = aievec.upd %[[MB]][%[[K14]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        %71 = affine.apply #map14(%arg5)
        %72 = vector.transfer_read %arg0[%arg3, %71], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %73 = vector.transfer_read %arg1[%71, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %74 = arith.muli %72, %73 : vector<16xi16>
        %75 = arith.addi %70, %74 : vector<16xi16>
        // CHECK: %[[K15:.*]] = arith.addi %[[K]], %[[C15]] : index
        // CHECK: %[[VB15:.*]] = aievec.upd %[[MB]][%[[K15]], %[[J]]]
        // CHECK-SAME:                  {index = 0 : i8, offset = 0 : i32}
        // CHECK-SAME:                  : memref<?x64xi16>, vector<16xi16>
        // CHECK: %[[VBef:.*]] = aievec.concat %[[VB14]], %[[VB15]] : vector<16xi16>, vector<32xi16>
        // CHECK: %[[ACCk14:.*]] = aievec_aie1.mac %[[VBef]], %[[VA]], %[[ACCk12]]
        // CHECK-SAME:                       {xoffsets = "0x73727170", xoffsets_hi = "0x77767574", xsquare = "0x3120",
        // CHECK-SAME:                        xstart = "0", zoffsets = "0", zoffsets_hi = "0", zstart = "14", zstep = "1"}
        // CHECK: %[[ACC:.*]] = aievec.srs %[[ACCk14]], %[[C0I32]] : vector<16xi48>, i32, vector<16xi16>
        %76 = affine.apply #map15(%arg5)
        %77 = vector.transfer_read %arg0[%arg3, %76], %c0_i16 {in_bounds = [true], permutation_map = #map} : memref<?x64xi16>, vector<16xi16>
        %78 = vector.transfer_read %arg1[%76, %arg4], %c0_i16 : memref<?x64xi16>, vector<16xi16>
        %79 = arith.muli %77, %78 : vector<16xi16>
        %80 = arith.addi %75, %79 : vector<16xi16>
        // CHECK: scf.yield %[[ACC]] : vector<16xi16>
        affine.yield %80 : vector<16xi16>
      }
      // CHECK: vector.transfer_write %[[ACCn]], %[[MC]][%[[I]], %[[J]]] {in_bounds = [true]} : vector<16xi16>, memref<?x64xi16>
      vector.transfer_write %1, %arg2[%arg3, %arg4] : vector<16xi16>, memref<?x64xi16>
    }
  }
  return
}
