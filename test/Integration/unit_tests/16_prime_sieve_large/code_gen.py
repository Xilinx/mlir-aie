#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

import sys
import re
import math
import random

def prime_gen():
    lower = 2
    upper = 10000

    prime_cnt = 0
    results = [1]
    for num in range(lower, upper + 1):
        # all prime numbers are greater than 1
        if num > 1:
            for i in range(2, num):
                if (num % i) == 0:
                    break
            else:
                results.append(num)
                prime_cnt = prime_cnt + 1
        if (prime_cnt == 500):
            break

    print("prime_cnt = %d" %(prime_cnt))
    return results




def main():
    prime_nums = prime_gen()

    rows = 9 + 1 # row 0 is reserved
    cols = 40 + 1

    f = open("aie.mlir", "w+")
    # declare tile, column by row


    f.write("""//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
//
// Note:
// this large prime-sieve pattern is generated from code_gen.py,
// it contains 360 cores in this pattern, user could change the core numbers
// by specifying different rows and cols value in code_gen.py

// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf \n\n\n""")



    f.write("module @test16_prime_sieve_large {\n")

    for col in range (1, cols): # col 0 is reserved in aie
        if (col % 2 == 1):
            for row in range (1, rows): # row 1 -> 8
                f.write("  %%tile%d_%d = AIE.tile(%d, %d)\n" %(col, row, col, row))
        else:
            for row in range (rows-1, 0, -1): # row 8 -> 1
                f.write("  %%tile%d_%d = AIE.tile(%d, %d)\n" %(col, row, col, row))
            


    f.write("\n")
    for col in range (1, cols): # col 0 is reserved in aie
        if (col % 2 == 1):
            for row in range (1, rows): # row 1 -> 8
                f.write("  %%lock%d_%d = AIE.lock(%%tile%d_%d, 0)\n" %(col, row, col, row))
        else:
            for row in range (rows-1, 0, -1): # row 8 -> 1
                f.write("  %%lock%d_%d = AIE.lock(%%tile%d_%d, 0)\n" %(col, row, col, row))


    f.write("\n")
    f.write("  %buf1_1 = AIE.buffer(%tile1_1) { sym_name = \"a\"         } : memref<256xi32>\n")
    prime_itr = 1;
    for col in range (1, cols): # col 0 is reserved in aie
        if (col % 2 == 1):
            for row in range (1, rows): # row 1 -> 8
                if (col == 1 and row == 1):
                    continue

                f.write("  %%buf%d_%d = AIE.buffer(%%tile%d_%d) { sym_name = \"prime%d\"    } : memref<256xi32>\n" %(col, row, col, row, prime_nums[prime_itr]))
                prime_itr = prime_itr + 1
        else:
            for row in range (rows-1, 0, -1): # row 8 -> 1
                f.write("  %%buf%d_%d = AIE.buffer(%%tile%d_%d) { sym_name = \"prime%d\"    } : memref<256xi32>\n" %(col, row, col, row, prime_nums[prime_itr]))
                prime_itr = prime_itr + 1
       



    # unchanged part
    f.write("""  
  %core1_1 = AIE.core(%tile1_1) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c64 = constant 64 : index
    %sum_0 = constant 2 : i32
    %t = constant 1 : i32
  
    // output integers starting with 2...
    scf.for %arg0 = %c0 to %c64 step %c1
      iter_args(%sum_iter = %sum_0) -> (i32) {
      %sum_next = addi %sum_iter, %t : i32
      memref.store %sum_iter, %buf1_1[%arg0] : memref<256xi32>
      scf.yield %sum_next : i32
    }
    AIE.useLock(%lock1_1, "Release", 1, 0)
    AIE.end
  }
  func.func @do_sieve(%bufin: memref<256xi32>, %bufout:memref<256xi32>) -> () {
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
  """)


    f.write("\n")
    last_index_from_pre_column = [-1, -1]
    for col in range (1, cols): # col 0 is reserved in aie
        if (col == 1):
            for row in range (2, rows): # core (1,1) is the initial core and has been handled before
                f.write("  %%core%d_%d = AIE.core(%%tile%d_%d) {\n" %(col, row, col, row))
                f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 1, 0)\n" %(col, row - 1 ))
                f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 0, 0)\n" %(col, row     ))
                f.write("    func.call @do_sieve(%%buf%d_%d, %%buf%d_%d) : (memref<256xi32>, memref<256xi32>) -> ()\n" %(col, row - 1, col, row))
                f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 0, 0)\n" %(col, row - 1 ))
                f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 1, 0)\n" %(col, row     ))
                f.write("    AIE.end\n")
                f.write("  }\n\n")
                if (row == rows - 1):
                    last_index_from_pre_column = [col, row]

        elif (col % 2 == 1):
            for row in range (1, rows):
                if (row == 1):
                    col_p = last_index_from_pre_column[0] 
                    row_p = last_index_from_pre_column[1]
                    f.write("  %%core%d_%d = AIE.core(%%tile%d_%d) {\n" %(col, row, col, row))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 1, 0)\n" %(col_p, row_p))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 0, 0)\n" %(col,   row  ))
                    f.write("    func.call @do_sieve(%%buf%d_%d, %%buf%d_%d) : (memref<256xi32>, memref<256xi32>) -> ()\n" %(col_p, row_p, col, row))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 0, 0)\n" %(col_p, row_p))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 1, 0)\n" %(col, row   ))
                    f.write("    AIE.end\n")
                    f.write("  }\n\n")
                else:
                    f.write("  %%core%d_%d = AIE.core(%%tile%d_%d) {\n" %(col, row, col, row))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 1, 0)\n" %(col, row - 1))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 0, 0)\n" %(col, row   ))
                    f.write("    func.call @do_sieve(%%buf%d_%d, %%buf%d_%d) : (memref<256xi32>, memref<256xi32>) -> ()\n" %(col, row - 1, col, row))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 0, 0)\n" %(col, row - 1))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 1, 0)\n" %(col, row   ))
                    f.write("    AIE.end\n")
                    f.write("  }\n\n")

                if (row == rows - 1):
                    last_index_from_pre_column = [col, row]
        else: # col % 2 == 0
            for row in range (rows - 1, 0, -1):
                if (row == rows - 1):
                    col_p = last_index_from_pre_column[0] 
                    row_p = last_index_from_pre_column[1]
                    f.write("  %%core%d_%d = AIE.core(%%tile%d_%d) {\n" %(col, row, col, row))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 1, 0)\n" %(col_p, row_p))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 0, 0)\n" %(col,   row  ))
                    f.write("    func.call @do_sieve(%%buf%d_%d, %%buf%d_%d) : (memref<256xi32>, memref<256xi32>) -> ()\n" %(col_p, row_p, col, row))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 0, 0)\n" %(col_p, row_p))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 1, 0)\n" %(col, row   ))
                    f.write("    AIE.end\n")
                    f.write("  }\n\n")
                else:
                    f.write("  %%core%d_%d = AIE.core(%%tile%d_%d) {\n" %(col, row, col, row))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 1, 0)\n" %(col, row + 1))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 0, 0)\n" %(col, row   ))
                    f.write("    func.call @do_sieve(%%buf%d_%d, %%buf%d_%d) : (memref<256xi32>, memref<256xi32>) -> ()\n" %(col, row + 1, col, row))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 0, 0)\n" %(col, row + 1))
                    f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 1, 0)\n" %(col, row   ))
                    f.write("    AIE.end\n")
                    f.write("  }\n\n")

                if (row == 1):
                    last_index_from_pre_column = [col, row]
    f.write("}\n")


if __name__ == "__main__":
    main()
