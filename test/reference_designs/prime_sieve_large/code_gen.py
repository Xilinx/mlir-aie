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

# The number of cores to use.  Will generate a 2D chain of cores of the given size.
arrayrows = 7 # must be odd
arraycols = 48 # must be even
bufsize = 4096 # must fit in data memory

def prime_gen(primecount):
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
        if (prime_cnt == primecount):
            break

    print("prime_cnt = %d" %(prime_cnt))
    print("lastprime = %d" % results[prime_cnt])
    return results




def main():
    global arrayrows
    global arraycols
    global bufsize
    primecount = arrayrows * arraycols

    prime_nums = prime_gen(primecount)
    print("Generating %d Cores" % (arrayrows * arraycols))
    if (prime_nums[-1] >= bufsize):
        print("Need a larger buffer to find %d is prime.  Please increase the 'bufsize' parameter" % prime_nums[-1])

    rows = arrayrows + 1 # row 0 is reserved
    cols = arraycols + 1

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

// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf \n\n\n""")



    f.write("module @test16_prime_sieve_large {\n")

    for col in range (1, cols): # col 0 is reserved in aie
        if (col % 2 == 1):
            for row in range (1, rows): # row 1 -> 8
                f.write("  %%tile%d_%d = AIE.tile(%d, %d)\n" %(col, row, col, row))
        else:
            for row in range (rows-1, 0, -1): # row 8 -> 1
                f.write("  %%tile%d_%d = AIE.tile(%d, %d)\n" %(col, row, col, row))
            
    # %objFifo = AIE.objectFifo.createObjectFifo(%tile12, {%tile33}, 2) : !AIE.objectFifo<memref<16xi32>>

    # %subview = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
    # %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>

    # AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)



    f.write("\n")
    for col in range (1, cols): # col 0 is reserved in aie
        if (col % 2 == 1):
            for row in range (1, rows): # row 1 -> 8
                f.write("  %%lock%d_%d = AIE.lock(%%tile%d_%d, 0)\n" %(col, row, col, row))
        else:
            for row in range (rows-1, 0, -1): # row 8 -> 1
                symbol = ""
                if (col == cols-1 and row == 1):
                    symbol = " { sym_name = \"prime_output_lock\" }" 
                f.write("  %%lock%d_%d = AIE.lock(%%tile%d_%d, 0)%s\n" %(col, row, col, row, symbol))


    f.write("\n")

    prime_itr = 1;
    row = 1
    for col in range (1, cols): # col 0 is reserved in aie
        for i in range (1, rows):
            if(row == 1 and col == 1):
                symbol = "a"
            elif(row == 1 and col == cols-1):
                symbol = "prime_output"
            else:
                symbol = "prime%d" % prime_nums[prime_itr]
                prime_itr = prime_itr + 1

            f.write("  %%buf%d_%d = AIE.buffer(%%tile%d_%d) { sym_name = \"%s\" } : memref<%dxi32>\n" %(col, row, col, row, symbol, bufsize))

            if (col % 2 == 1):
                if (row != rows-1):
                    # odd columns go up
                    row = row + 1
            else:
                if (row != 1):
                    # even columns go down
                    row = row - 1


    # unchanged part
    tilecode = """  
  %core1_1 = AIE.core(%tile1_1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cend = arith.constant """ + str(bufsize) + """: index
    %sum_0 = arith.constant 2 : i32
    %t = arith.constant 1 : i32
  
    // output integers starting with 2...
    scf.for %arg0 = %c0 to %cend step %c1
      iter_args(%sum_iter = %sum_0) -> (i32) {
      %sum_next = arith.addi %sum_iter, %t : i32
      memref.store %sum_iter, %buf1_1[%arg0] : memref<""" + str(bufsize) + """xi32>
      scf.yield %sum_next : i32
    }
    AIE.useLock(%lock1_1, "Release", 1)
    AIE.end
  }
  func.func @do_sieve(%bufin: memref<""" + str(bufsize) + """xi32>, %bufout:memref<""" + str(bufsize) + """xi32>) -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cend = arith.constant """ + str(bufsize) + """ : index
    %count_0 = arith.constant 0 : i32
  
    // The first number we receive is prime
    %prime = memref.load %bufin[%c0] : memref<""" + str(bufsize) + """xi32>
  
    // Step through the remaining inputs and sieve out multiples of %prime
    scf.for %arg0 = %c1 to %cend step %c1
      iter_args(%count_iter = %prime, %in_iter = %c1, %out_iter = %c0) -> (i32, index, index) {
      // Get the next input value
      %in_val = memref.load %bufin[%in_iter] : memref<""" + str(bufsize) + """xi32>
  
      // Potential next counters
      %count_inc = arith.addi %count_iter, %prime: i32
      %in_inc = arith.addi %in_iter, %c1 : index
      %out_inc = arith.addi %out_iter, %c1 : index
  
      // Compare the input value with the counter
      %b = arith.cmpi "slt", %in_val, %count_iter : i32
      %count_next, %in_next, %out_next = scf.if %b -> (i32, index, index) {
        // Input is less than counter.
        // Pass along the input and continue to the next one.
        memref.store %in_val, %bufout[%out_iter] : memref<""" + str(bufsize) + """xi32>
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
  """
    f.write(tilecode)


    f.write("\n")
    for col in range (1, cols): # col 0 is reserved in aie
        for i in range (1, rows):
            if(row == 1 and col == 1):
                None
            else:
                f.write("  %%core%d_%d = AIE.core(%%tile%d_%d) {\n" %(col, row, col, row))
                f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 1)\n" %(col_p, row_p))
                f.write("    AIE.useLock(%%lock%d_%d, \"Acquire\", 0)\n" %(col,   row  ))
                f.write("    func.call @do_sieve(%%buf%d_%d, %%buf%d_%d) : (memref<%dxi32>, memref<%dxi32>) -> ()\n" %(col_p, row_p, col, row, bufsize, bufsize))
                f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 0)\n" %(col_p, row_p))
                f.write("    AIE.useLock(%%lock%d_%d, \"Release\", 1)\n" %(col, row   ))
                f.write("    AIE.end\n")
                f.write("  }\n\n")
                    
            col_p = col
            row_p = row

            if (col % 2 == 1):
                if (row != rows-1):
                    # odd columns go up
                    row = row + 1
            else:
                if (row != 1):
                    # even columns go down
                    row = row - 1

    f.write("}\n")


if __name__ == "__main__":
    main()
