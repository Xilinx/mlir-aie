// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock1 {
// CHECK:  %0 = AIE.tile(3, 3)
// CHECK:  %1 = AIE.lock(%0, 0)
// CHECK:  %2 = AIE.tile(2, 3)
// CHECK:  %3 = AIE.lock(%2, 0)
// CHECK:  %4 = AIE.tile(4, 3)
// CHECK:  AIE.token(0) {sym_name = "token0"}
// CHECK:  %8 = AIE.core(%2) {
// CHECK:    AIE.useLock(%3, "Acquire", 0, 0)
// CHECK:    AIE.useLock(%3, "Release", 1, 0)
// CHECK:  }
// CHECK:  %9 = AIE.core(%0) {
// CHECK:    AIE.useLock(%1, "Acquire", 0, 0)
// CHECK:    AIE.useLock(%3, "Acquire", 1, 0)
// CHECK:    AIE.useLock(%3, "Release", 0, 0)
// CHECK:    AIE.useLock(%1, "Release", 1, 0)
// CHECK:  }
// CHECK:  %10 = AIE.core(%4) {
// CHECK:    AIE.useLock(%1, "Acquire", 1, 0)
// CHECK:    AIE.useLock(%1, "Release", 0, 0)
// CHECK:  }
// CHECK:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
// Tile-Tile-Tile
module @test_lock1 {
  %t33 = AIE.tile(3, 3)
  %t23 = AIE.tile(2, 3)
  %t43 = AIE.tile(4, 3)

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

  %m43 = AIE.mem(%t43) {
    AIE.terminator(^end)
    ^end:
      AIE.end
  }

  %c23 = AIE.core(%t23) {
    AIE.useToken @token0("Acquire", 0)
    AIE.useToken @token0("Release", 1)
    AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIE.useToken @token0("Acquire", 1)
    AIE.useToken @token0("Release", 2)
    AIE.end
  }

  %c43 = AIE.core(%t43) {
    AIE.useToken @token0("Acquire", 2)
    AIE.useToken @token0("Release", 3)
    AIE.end
  }
}
