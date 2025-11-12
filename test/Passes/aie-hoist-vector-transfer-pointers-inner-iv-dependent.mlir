// RUN: aie-opt --aie-hoist-vector-transfer-pointers %s | FileCheck %s

// This test verifies that the pass correctly handles nested loops where the inner
// loop's induction variable is used in address calculations with different offsets.
//
// In the input, we have:
//   - read at [%arg3, %arg2]      (uses inner IV directly)
//   - read at [%arg3+1, %arg2]    (uses inner IV + 1)
//   - write at [%arg3, %arg2]     (uses inner IV directly)
//
// The pass should correctly create separate iter_args for each distinct address pattern,
// initializing them with different base values (0 and 1) and incrementing them by the stride.

#map = affine_map<()[s0] -> (s0 + 1)>

module {
  func.func @inner_iv_dependent_addresses(%buf: memref<16x16xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0 : i32
    // Outer loop over %arg2
    scf.for %arg2 = %c0 to %c16 step %c2 {
      // Inner loop over %arg3 - addresses depend on THIS IV
      scf.for %arg3 = %c0 to %c16 step %c2 {
        // First read uses %arg3 directly
        %v1 = vector.transfer_read %buf[%arg3, %arg2], %cst {in_bounds = [true]} : memref<16x16xi32>, vector<2xi32>
        // Second read uses %arg3 + 1
        %arg3_plus_1 = affine.apply #map()[%arg3]
        %v2 = vector.transfer_read %buf[%arg3_plus_1, %arg2], %cst {in_bounds = [true]} : memref<16x16xi32>, vector<2xi32>
        // Use the values
        %sum = arith.addi %v1, %v2 : vector<2xi32>
        vector.transfer_write %sum, %buf[%arg3, %arg2] {in_bounds = [true]} : vector<2xi32>, memref<16x16xi32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @inner_iv_dependent_addresses
// CHECK: scf.for %[[OUTER:.*]] = 
// CHECK:   memref.collapse_shape
// CHECK:   affine.apply{{.*}}(%c0{{.*}}, %[[OUTER]])
// CHECK:   memref.collapse_shape
// CHECK:   affine.apply{{.*}}(%c1{{.*}}, %[[OUTER]])
// CHECK:   memref.collapse_shape
// CHECK:   affine.apply{{.*}}(%c0{{.*}}, %[[OUTER]])
// CHECK:   scf.for %{{.*}} = {{.*}} iter_args(%[[PTR0:.*]] = %{{.*}}, %[[PTR1:.*]] = %{{.*}}, %[[PTR2:.*]] = %{{.*}})
// CHECK:     vector.transfer_read %{{.*}}[%[[PTR0]]]
// CHECK:     arith.addi %[[PTR0]], %c32
// CHECK:     vector.transfer_read %{{.*}}[%[[PTR1]]]
// CHECK:     arith.addi %[[PTR1]], %c32
// CHECK:     vector.transfer_write %{{.*}}, %{{.*}}[%[[PTR2]]]
// CHECK:     arith.addi %[[PTR2]], %c32
// CHECK:     scf.yield
