//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_vck5000 ./test.elf

module @test15_prime_sieve {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)
  %tile14 = aie.tile(1, 4)
  %tile15 = aie.tile(1, 5)
  %tile16 = aie.tile(1, 6)

  %lock13_0 = aie.lock(%tile13, 0) { sym_name = "input_lock" }
  %lock14_0 = aie.lock(%tile14, 0)
  %lock15_0 = aie.lock(%tile15, 0)
  %lock16_0 = aie.lock(%tile16, 0) { sym_name = "output_lock" }
  %buf13_0 = aie.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf14_0 = aie.buffer(%tile14) { sym_name = "prime2" } : memref<256xi32>
  %buf15_0 = aie.buffer(%tile15) { sym_name = "prime3" } : memref<256xi32>
  %buf16_0 = aie.buffer(%tile16) { sym_name = "prime5" } : memref<256xi32>

  %core13 = aie.core(%tile13) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %sum_0 = arith.constant 2 : i32
    %t = arith.constant 1 : i32

    // output integers starting with 2...
    scf.for %arg0 = %c0 to %c64 step %c1
      iter_args(%sum_iter = %sum_0) -> (i32) {
      %sum_next = arith.addi %sum_iter, %t : i32
      memref.store %sum_iter, %buf13_0[%arg0] : memref<256xi32>
      scf.yield %sum_next : i32
    }
    %c1_ul0 = arith.constant 1 : i32
    aie.use_lock(%lock13_0, "Release", %c1_ul0)
    aie.end
  }
  func.func @do_sieve(%bufin: memref<256xi32>, %bufout:memref<256xi32>) -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %count_0 = arith.constant 0 : i32

    // The first number we receive is prime
    %prime = memref.load %bufin[%c0] : memref<256xi32>

    // Step through the remaining inputs and sieve out multiples of %prime
    scf.for %arg0 = %c1 to %c64 step %c1
      iter_args(%count_iter = %prime, %in_iter = %c1, %out_iter = %c0) -> (i32, index, index) {
      // Get the next input value
      %in_val = memref.load %bufin[%in_iter] : memref<256xi32>

      // Potential next counters
      %count_inc = arith.addi %count_iter, %prime: i32
      %in_inc = arith.addi %in_iter, %c1 : index
      %out_inc = arith.addi %out_iter, %c1 : index

      // Compare the input value with the counter
      %b = arith.cmpi "slt", %in_val, %count_iter : i32
      %count_next, %in_next, %out_next = scf.if %b -> (i32, index, index) {
        // Input is less than counter.
        // Pass along the input and continue to the next one.
        memref.store %in_val, %bufout[%out_iter] : memref<256xi32>
        scf.yield %count_iter, %in_inc, %out_inc : i32, index, index
      } else {
        %b2 = arith.cmpi "eq", %in_val, %count_iter : i32
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

  %core14 = aie.core(%tile14) {
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(%lock13_0, "Acquire", %c1_ul1)
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(%lock14_0, "Acquire", %c0_ul2)
    func.call @do_sieve(%buf13_0, %buf14_0) : (memref<256xi32>, memref<256xi32>) -> ()
    %c0_ul3 = arith.constant 0 : i32
    aie.use_lock(%lock13_0, "Release", %c0_ul3)
    %c1_ul4 = arith.constant 1 : i32
    aie.use_lock(%lock14_0, "Release", %c1_ul4)
    aie.end
  }
  %core15 = aie.core(%tile15) {
    %c1_ul5 = arith.constant 1 : i32
    aie.use_lock(%lock14_0, "Acquire", %c1_ul5)
    %c0_ul6 = arith.constant 0 : i32
    aie.use_lock(%lock15_0, "Acquire", %c0_ul6)
    func.call @do_sieve(%buf14_0, %buf15_0) : (memref<256xi32>, memref<256xi32>) -> ()
    %c0_ul7 = arith.constant 0 : i32
    aie.use_lock(%lock14_0, "Release", %c0_ul7)
    %c1_ul8 = arith.constant 1 : i32
    aie.use_lock(%lock15_0, "Release", %c1_ul8)
    aie.end
  }
  %core16 = aie.core(%tile16) {
    %c1_ul9 = arith.constant 1 : i32
    aie.use_lock(%lock15_0, "Acquire", %c1_ul9)
    %c0_ul10 = arith.constant 0 : i32
    aie.use_lock(%lock16_0, "Acquire", %c0_ul10)
    func.call @do_sieve(%buf15_0, %buf16_0) : (memref<256xi32>, memref<256xi32>) -> ()
    %c0_ul11 = arith.constant 0 : i32
    aie.use_lock(%lock15_0, "Release", %c0_ul11)
    %c1_ul12 = arith.constant 1 : i32
    aie.use_lock(%lock16_0, "Release", %c1_ul12)
    aie.end
  }
  
}
}
