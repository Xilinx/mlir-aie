// RUN: aie-opt %s -aie-hoist-vector-transfer-pointers -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @hoist_vector_transfer_read
func.func @hoist_vector_transfer_read(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 0.0 : f32
  
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR0:.*]] = %{{.*}}, %[[PTR1:.*]] = %{{.*}})
  scf.for %i = %c0 to %c64 step %c1 {
    // CHECK: vector.transfer_read %{{.*}}[%[[PTR0]]]{{.*}}{in_bounds = [true]}
    %v = vector.transfer_read %arg0[%i], %cst : memref<256xf32>, vector<16xf32>
    // CHECK: arith.addi %[[PTR0]]
    vector.transfer_write %v, %arg1[%i] : vector<16xf32>, memref<256xf32>
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR1]]]{{.*}}{in_bounds = [true]}
    // CHECK: arith.addi %[[PTR1]]
    // CHECK: scf.yield %{{.*}}, %{{.*}}
  }
  return
}

// -----

// CHECK-LABEL: func.func @hoist_vector_transfer_write
func.func @hoist_vector_transfer_write(%arg0: memref<256xf32>, %arg1: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR:.*]] = %{{.*}})
  scf.for %i = %c0 to %c64 step %c1 {
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR]]] {in_bounds = [true]}
    vector.transfer_write %arg1, %arg0[%i] : vector<16xf32>, memref<256xf32>
    // CHECK: arith.addi %[[PTR]], %{{.*}}
    // CHECK: scf.yield %{{.*}}
  }
  return
}

// -----

// CHECK-LABEL: func.func @hoist_2d_memref
func.func @hoist_2d_memref(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %cst = arith.constant 0.0 : f32
  
  // CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
  // CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1]{{\]}}
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[PTR0:.*]] = %{{.*}}, %[[PTR1:.*]] = %{{.*}})
  scf.for %i = %c0 to %c16 step %c1 {
    // CHECK: vector.transfer_read %{{.*}}[%[[PTR0]]]{{.*}}{in_bounds = [true]}
    %v = vector.transfer_read %arg0[%i, %c0], %cst : memref<16x16xf32>, vector<16xf32>
    // CHECK: arith.addi %[[PTR0]], %{{.*}}
    vector.transfer_write %v, %arg1[%i, %c0] : vector<16xf32>, memref<16x16xf32>
    // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR1]]]{{.*}}{in_bounds = [true]}
    // CHECK: arith.addi %[[PTR1]], %{{.*}}
    // CHECK: scf.yield %{{.*}}, %{{.*}}
  }
  return
}
