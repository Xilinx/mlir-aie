// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test15_prime_sieve {
  %tile13 = AIE.tile(1, 3)
  %tile14 = AIE.tile(1, 4)
  %tile15 = AIE.tile(1, 5)
  %tile16 = AIE.tile(1, 6)

  %lock13_0 = AIE.lock(%tile13, 0)
  %lock14_0 = AIE.lock(%tile14, 0)
  %lock15_0 = AIE.lock(%tile15, 0)
  %lock16_0 = AIE.lock(%tile16, 0)
  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf14_0 = AIE.buffer(%tile14) { sym_name = "prime2" } : memref<256xi32>
  %buf15_0 = AIE.buffer(%tile15) { sym_name = "prime3" } : memref<256xi32>
  %buf16_0 = AIE.buffer(%tile16) { sym_name = "prime5" } : memref<256xi32>

  %core13 = AIE.core(%tile13) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c64 = constant 64 : index
    %sum_0 = constant 2 : i32
    %t = constant 1 : i32

    // output integers starting with 2...
    scf.for %arg0 = %c0 to %c64 step %c1
      iter_args(%sum_iter = %sum_0) -> (i32) {
      %sum_next = addi %sum_iter, %t : i32
      memref.store %sum_iter, %buf13_0[%arg0] : memref<256xi32>
      scf.yield %sum_next : i32
    }
    AIE.useLock(%lock13_0, "Release", 1, 0)
    AIE.end
  }
  func @do_sieve(%bufin: memref<256xi32>, %bufout:memref<256xi32>) -> () {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c64 = constant 64 : index
    %count_0 = constant 0 : i32

    // The first number we receive is prime
    %prime = memref.load %bufin[%c0] : memref<256xi32>

    // Step through the remaining inputs and sieve out multiples of %prime
    scf.for %arg0 = %c1 to %c64 step %c1
      iter_args(%count_iter = %prime, %in_iter = %c1, %out_iter = %c0) -> (i32, index, index) {
      // Get the next input value
      %in_val = memref.load %bufin[%in_iter] : memref<256xi32>

      // Potential next counters
      %count_inc = addi %count_iter, %prime: i32
      %in_inc = addi %in_iter, %c1 : index
      %out_inc = addi %out_iter, %c1 : index

      // Compare the input value with the counter
      %b = cmpi "slt", %in_val, %count_iter : i32
      %count_next, %in_next, %out_next = scf.if %b -> (i32, index, index) {
        // Input is less than counter.
        // Pass along the input and continue to the next one.
        memref.store %in_val, %bufout[%out_iter] : memref<256xi32>
        scf.yield %count_iter, %in_inc, %out_inc : i32, index, index
      } else {
        %b2 = cmpi "eq", %in_val, %count_iter : i32
        %in_next = scf.if %b2 -> (index) {
          // Input is equal to the counter.
          // Increment the counter and continue to the next input.
          scf.yield %in_inc : index
        } else {
          // Input is greater than the counter.
          // Increment the counter and check again.
          scf.yield %in_iter : index
        }
        scf.yield %count_inc, %in_next, %out_iter : i32, index, index
      }
      scf.yield %count_next, %in_next, %out_next : i32, index, index
    }
    return
  }

  %core14 = AIE.core(%tile14) {
    AIE.useLock(%lock13_0, "Acquire", 1, 0)
    AIE.useLock(%lock14_0, "Acquire", 0, 0)
    call @do_sieve(%buf13_0, %buf14_0) : (memref<256xi32>, memref<256xi32>) -> ()
    AIE.useLock(%lock13_0, "Release", 0, 0)
    AIE.useLock(%lock14_0, "Release", 1, 0)
    AIE.end
  }
  %core15 = AIE.core(%tile15) {
    AIE.useLock(%lock14_0, "Acquire", 1, 0)
    AIE.useLock(%lock15_0, "Acquire", 0, 0)
    call @do_sieve(%buf14_0, %buf15_0) : (memref<256xi32>, memref<256xi32>) -> ()
    AIE.useLock(%lock14_0, "Release", 0, 0)
    AIE.useLock(%lock15_0, "Release", 1, 0)
    AIE.end
  }
  %core16 = AIE.core(%tile16) {
    AIE.useLock(%lock15_0, "Acquire", 1, 0)
    AIE.useLock(%lock16_0, "Acquire", 0, 0)
    call @do_sieve(%buf15_0, %buf16_0) : (memref<256xi32>, memref<256xi32>) -> ()
    AIE.useLock(%lock15_0, "Release", 0, 0)
    AIE.useLock(%lock16_0, "Release", 1, 0)
    AIE.end
  }
}
