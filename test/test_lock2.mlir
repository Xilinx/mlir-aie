// RUN: aie-opt --aie-create-locks %s | FileCheck %s

// CHECK-LABEL: module @test_lock2 {
// CHECK-NEXT:  %0 = AIE.tile(3, 3)
// CHECK-NEXT:  %1 = AIE.lock(%0, 0)
// CHECK-NEXT:  %2 = AIE.tile(2, 3)
// CHECK-NEXT:  %3 = AIE.lock(%2, 0)
// CHECK-NEXT:  %4 = AIE.tile(3, 4)
// CHECK-NEXT:  %5 = AIE.lock(%4, 0)
// CHECK-NEXT:  %6 = AIE.tile(4, 3)
// CHECK-NEXT:  %7 = AIE.tile(3, 2)
// CHECK-NEXT:  %8 = AIE.lock(%7, 0)
// CHECK-NEXT:  AIE.token(0) {sym_name = "token0"}
// CHECK-NEXT:  AIE.token(0) {sym_name = "token1"}
// CHECK-NEXT:  AIE.token(0) {sym_name = "token2"}
// CHECK-NEXT:  AIE.token(0) {sym_name = "token3"}
// CHECK-NEXT:  %9 = AIE.mem(%0) {
// CHECK-NEXT:    AIE.terminator(^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %10 = AIE.mem(%2) {
// CHECK-NEXT:    AIE.terminator(^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %11 = AIE.mem(%4) {
// CHECK-NEXT:    AIE.terminator(^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %12 = AIE.mem(%6) {
// CHECK-NEXT:    AIE.terminator(^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %13 = AIE.mem(%7) {
// CHECK-NEXT:    AIE.terminator(^bb1)
// CHECK-NEXT:  ^bb1:  // pred: ^bb0
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %14 = AIE.core(%2) {
// CHECK-NEXT:    AIE.useLock(%3, "Acquire", 1, 0)
// CHECK-NEXT:    AIE.useLock(%3, "Release", 0, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %15 = AIE.core(%0) {
// CHECK-NEXT:    AIE.useLock(%8, "Acquire", 0, 0)
// CHECK-NEXT:    AIE.useLock(%1, "Acquire", 0, 0)
// CHECK-NEXT:    AIE.useLock(%5, "Acquire", 0, 0)
// CHECK-NEXT:    AIE.useLock(%3, "Acquire", 0, 0)
// CHECK-NEXT:    AIE.useLock(%3, "Release", 1, 0)
// CHECK-NEXT:    AIE.useLock(%5, "Release", 1, 0)
// CHECK-NEXT:    AIE.useLock(%1, "Release", 1, 0)
// CHECK-NEXT:    AIE.useLock(%8, "Release", 1, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %16 = AIE.core(%4) {
// CHECK-NEXT:    AIE.useLock(%5, "Acquire", 1, 0)
// CHECK-NEXT:    AIE.useLock(%5, "Release", 0, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %17 = AIE.core(%6) {
// CHECK-NEXT:    AIE.useLock(%1, "Acquire", 1, 0)
// CHECK-NEXT:    AIE.useLock(%1, "Release", 0, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:  %18 = AIE.core(%7) {
// CHECK-NEXT:    AIE.useLock(%8, "Acquire", 1, 0)
// CHECK-NEXT:    AIE.useLock(%8, "Release", 0, 0)
// CHECK-NEXT:    AIE.end
// CHECK-NEXT:  }
// CHECK-NEXT:}

// Generate LockOp in the top-level module
// Lower UseTokenOp to UseLockOp
//      Tile
//       |
// Tile-Tile-Tile
//       |
//      Tile
// single producer (tile(3, 3)), multiple consumers
module @test_lock2 {
  %t33 = AIE.tile(3, 3)
  %t23 = AIE.tile(2, 3)
  %t34 = AIE.tile(3, 4)
  %t43 = AIE.tile(4, 3)
  %t32 = AIE.tile(3, 2)

  AIE.token(0) {sym_name = "token0"}
  AIE.token(0) {sym_name = "token1"}
  AIE.token(0) {sym_name = "token2"}
  AIE.token(0) {sym_name = "token3"}

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

  %m34 = AIE.mem(%t34) {
    AIE.terminator(^end)
    ^end:
      AIE.end
  }

  %m43 = AIE.mem(%t43) {
    AIE.terminator(^end)
    ^end:
      AIE.end
  }

  %m32 = AIE.mem(%t32) {
    AIE.terminator(^end)
    ^end:
      AIE.end
  }

  %c23 = AIE.core(%t23) {
    AIE.useToken @token0("Acquire", 1)
    AIE.useToken @token0("Release", 2)
    AIE.end
  }

  %c33 = AIE.core(%t33) {
    AIE.useToken @token3("Acquire", 0)
    AIE.useToken @token2("Acquire", 0)
    AIE.useToken @token1("Acquire", 0)
    AIE.useToken @token0("Acquire", 0)
    AIE.useToken @token0("Release", 1)
    AIE.useToken @token1("Release", 1)
    AIE.useToken @token2("Release", 1)
    AIE.useToken @token3("Release", 1)
    AIE.end
  }

  %c34 = AIE.core(%t34) {
    AIE.useToken @token1("Acquire", 1)
    AIE.useToken @token1("Release", 2)
    AIE.end
  }

  %c43 = AIE.core(%t43) {
    AIE.useToken @token2("Acquire", 1)
    AIE.useToken @token2("Release", 2)
    AIE.end
  }

  %c32 = AIE.core(%t32) {
    AIE.useToken @token3("Acquire", 1)
    AIE.useToken @token3("Release", 2)
    AIE.end
  }
}
