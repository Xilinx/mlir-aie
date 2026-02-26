// RUN: aie-opt %s -convert-aievec-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: mac_flat_vec
// CHECK-SAME: %[[V0:[a-zA-Z0-9]+]]: vector<16xbf16>,
// CHECK-SAME: %[[V1:.*]]: vector<16xbf16>,
// CHECK-SAME: %[[V2:.*]]: vector<16xf32>)
func.func @mac_flat_vec(%v0 : vector<16xbf16>,
                        %v1 : vector<16xbf16>,
                        %v2 : vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[C60:.*]] = llvm.mlir.constant(60 : i32) : i32

  // Zero-pad 16-lane bf16 operands to 32-lane using set+upd intrinsics
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[BCAST:.*]] = "xllvm.intr.aie2.vbroadcast16.I512"(%[[C0]])
  // CHECK-SAME:                        : (i32) -> vector<32xi16>
  // CHECK: %[[BCAST_BF:.*]] = llvm.bitcast %[[BCAST]]
  // CHECK-SAME:                        : vector<32xi16> to vector<32xbf16>
  // CHECK: %[[ZEROVEC:.*]] = "xllvm.intr.aie2.ext.bf256.bf512"(%[[BCAST_BF]], %[[C0]])
  // CHECK-SAME:                        : (vector<32xbf16>, i32) -> vector<16xbf16>
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[SET0:.*]] = "xllvm.intr.aie2.set.bf512.bf256"(%[[V0]], %[[C0]])
  // CHECK-SAME:                        : (vector<16xbf16>, i32) -> vector<32xbf16>
  // CHECK: %[[V02C:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[SET0]], %[[ZEROVEC]], %[[C1]])
  // CHECK-SAME:                        : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
  // CHECK: %[[SET1:.*]] = "xllvm.intr.aie2.set.bf512.bf256"(%[[V1]], %[[C0]])
  // CHECK-SAME:                        : (vector<16xbf16>, i32) -> vector<32xbf16>
  // CHECK: %[[V12C:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[SET1]], %[[ZEROVEC]], %[[C1]])
  // CHECK-SAME:                        : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>

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

  // Zero-pad 16-lane bf16 operands to 32-lane using set+upd intrinsics
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %[[BCAST:.*]] = "xllvm.intr.aie2.vbroadcast16.I512"(%[[C0]])
  // CHECK-SAME:                        : (i32) -> vector<32xi16>
  // CHECK: %[[BCAST_BF:.*]] = llvm.bitcast %[[BCAST]]
  // CHECK-SAME:                        : vector<32xi16> to vector<32xbf16>
  // CHECK: %[[ZEROVEC:.*]] = "xllvm.intr.aie2.ext.bf256.bf512"(%[[BCAST_BF]], %[[C0]])
  // CHECK-SAME:                        : (vector<32xbf16>, i32) -> vector<16xbf16>
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[SET0:.*]] = "xllvm.intr.aie2.set.bf512.bf256"(%[[V0]], %[[C0]])
  // CHECK-SAME:                        : (vector<16xbf16>, i32) -> vector<32xbf16>
  // CHECK: %[[V02C:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[SET0]], %[[ZEROVEC]], %[[C1]])
  // CHECK-SAME:                        : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>
  // CHECK: %[[SET1:.*]] = "xllvm.intr.aie2.set.bf512.bf256"(%[[V1]], %[[C0]])
  // CHECK-SAME:                        : (vector<16xbf16>, i32) -> vector<32xbf16>
  // CHECK: %[[V12C:.*]] = "xllvm.intr.aie2.upd.bf512.bf256"(%[[SET1]], %[[ZEROVEC]], %[[C1]])
  // CHECK-SAME:                        : (vector<32xbf16>, vector<16xbf16>, i32) -> vector<32xbf16>

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