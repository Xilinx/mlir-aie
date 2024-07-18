// RUN: aie-opt %s -convert-aievec-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: mac_flat_vec
// CHECK-SAME: %[[V0:[a-zA-Z0-9]+]]: vector<16xbf16>,
// CHECK-SAME: %[[V1:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[V2:.*]]: vector<16xf32>)
func.func @mac_flat_vec(%v0 : vector<16xbf16>,
                        %v1 : vector<16xbf16>,
                        %v2 : vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[C60:.*]] = llvm.mlir.constant(60 : i32) : i32

  // CHECK: %[[C0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[BV0:.*]] = llvm.bitcast %[[V0]]
  // CHECK-SAME:                        : vector<16xbf16> to vector<8xi32>
  // CHECK: %[[BV02C:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[BV0]], %[[C0_0]])
  // CHECK-SAME:                        : (vector<8xi32>, i32) -> vector<16xi32>
  // CHECK: %[[V02C:.*]] = llvm.bitcast %[[BV02C]]
  // CHECK-SAME:                        : vector<16xi32> to vector<32xbf16>

  // CHECK: %[[C0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[BV1:.*]] = llvm.bitcast %[[V1]]
  // CHECK-SAME:                        : vector<16xbf16> to vector<8xi32>
  // CHECK: %[[BV12C:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[BV1]], %[[C0_1]])
  // CHECK-SAME:                        : (vector<8xi32>, i32) -> vector<16xi32>
  // CHECK: %[[V12C:.*]] = llvm.bitcast %[[BV12C]]
  // CHECK-SAME:                        : vector<16xi32> to vector<32xbf16>

  // CHECK: %[[BV2:.*]] = llvm.bitcast %[[V2]]
  // CHECK-SAME:                        : vector<16xf32> to vector<8xi64>
  // CHECK: %[[BRS:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[V02C]], %[[V12C]],
  // CHECK-SAME:                                          %[[BV2]], %[[C60]])
  // CHECK-SAME:                        : (vector<32xbf16>, vector<32xbf16>,
  // CHECK-SAME:                           vector<8xi64>, i32) -> vector<8xi64>
  // CHECK: %[[RS:.*]] = llvm.bitcast %[[BRS]]
  // CHECK-SAME:                        : vector<8xi64> to vector<16xf32>

  %0 = aievec.mac_elem %v0, %v1, %v2 : vector<16xbf16>, vector<16xbf16>, vector<16xf32>
  return %0 : vector<16xf32>
}

// -----

// CHECK-LABEL: mac_2d_vec
// CHECK-SAME: %[[V02D:[a-zA-Z0-9]+]]: vector<4x4xbf16>,
// CHECK-SAME: %[[V12D:.*]]: vector<4x4xbf16>,
// CHECK-SAME: %[[V22D:.*]]: vector<4x4xf32>)
func.func @mac_2d_vec(%v0 : vector<4x4xbf16>,
                      %v1 : vector<4x4xbf16>,
                      %v2 : vector<4x4xf32>) -> vector<4x4xf32> {
  // CHECK: %[[V0:.*]] = vector.shape_cast %[[V02D]]
  // CHECK-SAME:                        : vector<4x4xbf16> to vector<16xbf16>
  // CHECK: %[[V1:.*]] = vector.shape_cast %[[V12D]]
  // CHECK-SAME:                        : vector<4x4xbf16> to vector<16xbf16>
  // CHECK: %[[V2:.*]] = vector.shape_cast %[[V22D]]
  // CHECK-SAME:                        : vector<4x4xf32> to vector<16xf32>

  // CHECK: %[[C60:.*]] = llvm.mlir.constant(60 : i32) : i32

  // CHECK: %[[C0_0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[BV0:.*]] = llvm.bitcast %[[V0]]
  // CHECK-SAME:                        : vector<16xbf16> to vector<8xi32>
  // CHECK: %[[BV02C:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[BV0]], %[[C0_0]])
  // CHECK-SAME:                        : (vector<8xi32>, i32) -> vector<16xi32>
  // CHECK: %[[V02C:.*]] = llvm.bitcast %[[BV02C]]
  // CHECK-SAME:                        : vector<16xi32> to vector<32xbf16>

  // CHECK: %[[C0_1:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[BV1:.*]] = llvm.bitcast %[[V1]]
  // CHECK-SAME:                        : vector<16xbf16> to vector<8xi32>
  // CHECK: %[[BV12C:.*]] = "xllvm.intr.aie2.set.I512.I256"(%[[BV1]], %[[C0_1]])
  // CHECK-SAME:                        : (vector<8xi32>, i32) -> vector<16xi32>
  // CHECK: %[[V12C:.*]] = llvm.bitcast %[[BV12C]]
  // CHECK-SAME:                        : vector<16xi32> to vector<32xbf16>

  // CHECK: %[[BV2:.*]] = llvm.bitcast %[[V2]]
  // CHECK-SAME:                        : vector<16xf32> to vector<8xi64>
  // CHECK: %[[BRS:.*]] = "xllvm.intr.aie2.bf.mac16.conf"(%[[V02C]], %[[V12C]],
  // CHECK-SAME:                                          %[[BV2]], %[[C60]])
  // CHECK-SAME:                        : (vector<32xbf16>, vector<32xbf16>,
  // CHECK-SAME:                           vector<8xi64>, i32) -> vector<8xi64>
  // CHECK: %[[RS:.*]] = llvm.bitcast %[[BRS]]
  // CHECK-SAME:                        : vector<8xi64> to vector<16xf32>

  // CHECK: %[[RS2D:.*]] = vector.shape_cast %[[RS]]
  // CHECK-SAME:                        : vector<16xf32> to vector<4x4xf32>

  %0 = aievec.mac_elem %v0, %v1, %v2 : vector<4x4xbf16>, vector<4x4xbf16>, vector<4x4xf32>
  return %0 : vector<4x4xf32>
}