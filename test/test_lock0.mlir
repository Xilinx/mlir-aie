// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock0 {
// CHECK-NEXT:  %0 = AIE.tile(3, 3)
// CHECK-NEXT:  %1 = AIE.tile(2, 3)
// CHECK-NEXT:  %2 = AIE.lock(%1, 0)
// CHECK-NEXT:  AIE.token(0) {sym_name = "token0"}
// CHECK-NEXT:  %3 = AIE.mem(%0) {
// CHECK-NEXT:    AIE.terminator(^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %4 = AIE.mem(%1) {
// CHECK-NEXT:    AIE.terminator(^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %5 = AIE.core(%0) {
// CHECK-NEXT:    AIE.useLock(%2, "Acquire", 0, 0)
// CHECK-NEXT:    AIE.useLock(%2, "Release", 1, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %6 = AIE.core(%1) {
// CHECK-NEXT:    AIE.useLock(%2, "Acquire", 1, 0)
// CHECK-NEXT:    AIE.useLock(%2, "Release", 0, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// Tile-Tile
module @test_lock0 {
  %t33 = AIE.tile(3, 3)
  %t23 = AIE.tile(2, 3)

  AIE.token(0) {sym_name = "token0"}

  %m33 = AIE.mem(%t33) {
    AIE.terminator(^end)
    ^end:
      AIE.end
  }

  %m23 = AIE.mem(%t23) {
    AIE.terminator(^end)
    ^end:
      AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIE.useToken @token0("Acquire", 0)
    AIE.useToken @token0("Release", 1)
    AIE.end
  }

  %c23 = AIE.core(%t23) {
    AIE.useToken @token0("Acquire", 1)
    AIE.useToken @token0("Release", 2)
    AIE.end
  }
}
