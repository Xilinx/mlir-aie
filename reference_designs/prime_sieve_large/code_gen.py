#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2021 Xilinx Inc.

# The number of cores to use.  Will generate a 2D chain of cores of the given size.
arrayrows = 8
arraycols = 50  # must be even
startcol = 0
bufsize = 3072  # must fit in data memory


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
        if prime_cnt == primecount:
            break

    print("prime_cnt = %d" % (prime_cnt))
    print("lastprime = %d" % results[prime_cnt])
    return results


def main():
    global arrayrows
    global arraycols
    global bufsize
    primecount = arrayrows * arraycols

    prime_nums = prime_gen(primecount)
    print("Generating %d Cores" % (arrayrows * arraycols))
    if prime_nums[-1] >= bufsize:
        print(
            "Need a larger buffer to find %d is prime.  Please increase the 'bufsize' parameter"
            % prime_nums[-1]
        )

    rows = arrayrows + 1  # row 0 is reserved
    cols = arraycols + 1

    f = open("aie.mlir", "w+")
    # declare tile, column by row

    f.write(
        """//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
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

// RUN: aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf \n\n\n"""
    )

    f.write("module @test16_prime_sieve_large {\n")

    for col in range(startcol, startcol + arraycols):  # col 0 is reserved in aie
        for row in range(1, rows):  # row 1 -> 8
            f.write("  %%tile%d_%d = AIE.tile(%d, %d)\n" % (col, row, col, row))

    # %objFifo = AIE.objectfifo.createObjectFifo(%tile12, {%tile33}, 2) : !AIE.objectfifo<memref<16xi32>>

    # %subview = AIE.objectfifo.acquire<Produce>(%objFifo : !AIE.objectfifo<memref<16xi32>>, 1) : !AIE.objectfifosubview<memref<16xi32>>
    # %elem0 = AIE.objectfifo.subview.access %subview[0] : !AIE.objectfifosubview<memref<16xi32>> -> memref<16xi32>

    # AIE.objectfifo.release<Produce>(%objFifo : !AIE.objectfifo<memref<16xi32>>, 1)

    f.write("\n")
    col = startcol
    for i in range(0, arraycols, 2):
        for row in range(1, rows):  # row 1 -> 8
            if row == rows - 1:
                lockrow = row
                lockcol = col + 1 if (row % 2 == 0) else col
            else:
                lockrow = row
                lockcol = col
            f.write(
                "  %%lock%d_%d = AIE.lock(%%tile%d_%d)\n" % (col, row, lockcol, lockrow)
            )

        col = col + 1
        for row in range(rows - 1, 0, -1):  # row 8 -> 1
            symbol = ""
            if (i == arraycols - 2) and row == 1:
                symbol = ' { sym_name = "prime_output_lock" }'
                lockrow = row
                lockcol = col
            elif row == 1:
                lockrow = row
                lockcol = col + 1 if (row % 2 == 0) else col
            else:
                lockrow = row
                lockcol = col
            f.write(
                "  %%lock%d_%d = AIE.lock(%%tile%d_%d)%s\n"
                % (col, row, lockcol, lockrow, symbol)
            )
        col = col + 1

    f.write("\n")

    # row and column to generate for.  lastrow indicates that we shift one column to the right
    # for the next core.
    global prime_itr
    prime_itr = 1

    def gen_buffer(row, col, lastrow):
        global prime_itr
        if row == 1 and col == startcol:
            symbol = "a"
            lockrow = row
            lockcol = col
        elif row == 1 and col == startcol + arraycols - 1:
            symbol = "prime_output"
            lockrow = row
            lockcol = col
        else:
            symbol = "prime%d" % prime_nums[prime_itr]
            prime_itr = prime_itr + 1
            if lastrow:
                lockrow = row
                lockcol = col + 1 if (row % 2 == 0) else col
            else:
                lockrow = row
                lockcol = col
        f.write(
            '  %%buf%d_%d = AIE.buffer(%%tile%d_%d) { sym_name = "%s" } : memref<%dxi32>\n'
            % (col, row, lockcol, lockrow, symbol, bufsize)
        )

    col = startcol
    row = 1
    for i in range(0, arraycols, 2):  # col 0 is reserved in aie
        for j in range(arrayrows - 1):
            gen_buffer(row, col, False)
            row = row + 1
        gen_buffer(row, col, True)
        col = col + 1
        for j in range(arrayrows - 1):
            gen_buffer(row, col, False)
            row = row - 1
        gen_buffer(row, col, True)
        col = col + 1

    # unchanged part
    tilecode = (
        """  
          %core"""
        + str(startcol)
        + """_1 = AIE.core(%tile"""
        + str(startcol)
        + """_1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cend = arith.constant """
        + str(bufsize)
        + """: index
    %sum_0 = arith.constant 2 : i32
    %t = arith.constant 1 : i32
  
    // store the index of the next prime number
    memref.store %t, %buf"""
        + str(startcol)
        + """_1[%c0] : memref<"""
        + str(bufsize)
        + """xi32>

    // output integers starting with 2...
    scf.for %arg0 = %c1 to %cend step %c1
      iter_args(%sum_iter = %sum_0) -> (i32) {
      %sum_next = arith.addi %sum_iter, %t : i32
      memref.store %sum_iter, %buf"""
        + str(startcol)
        + """_1[%arg0] : memref<"""
        + str(bufsize)
        + """xi32>
      scf.yield %sum_next : i32
    }
    AIE.useLock(%lock"""
        + str(startcol)
        + """_1, "Release", 1)
    AIE.end
  }
  func.func @do_sieve(%bufin: memref<"""
        + str(bufsize)
        + """xi32>, %bufout:memref<"""
        + str(bufsize)
        + """xi32>) -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cend = arith.constant """
        + str(bufsize)
        + """ : index
    %count_0 = arith.constant 0 : i32
    %one = arith.constant 1 : i32
  
    // The first number we receive is the index of the next prime
    %id = memref.load %bufin[%c0] : memref<"""
        + str(bufsize)
        + """xi32>

    // Compute the next id and store it in the output buffer
    %nextid = arith.addi %id, %one : i32
    memref.store %nextid, %bufout[%c0] : memref<"""
        + str(bufsize)
        + """xi32>

    // Copy the prior inputs
    %id_index = arith.index_cast %id : i32 to index
    %nextid_index = arith.index_cast %nextid : i32 to index
    scf.for %arg0 = %c1 to %nextid_index step %c1 {
      %in_val = memref.load %bufin[%arg0] : memref<"""
        + str(bufsize)
        + """xi32>
      memref.store %in_val, %bufout[%arg0] : memref<"""
        + str(bufsize)
        + """xi32>
    }
    %prime = memref.load %bufin[%id_index] : memref<"""
        + str(bufsize)
        + """xi32>

    // Step through the remaining inputs and sieve out multiples of %prime
    scf.for %arg0 = %nextid_index to %cend step %c1
      iter_args(%count_iter = %prime, %in_iter = %nextid_index, %out_iter = %nextid_index) -> (i32, index, index) {
      // Get the next input value
      %in_val = memref.load %bufin[%in_iter] : memref<"""
        + str(bufsize)
        + """xi32>

      // Potential next counters
      %count_inc = arith.addi %count_iter, %prime: i32
      %in_inc = arith.addi %in_iter, %c1 : index
      %out_inc = arith.addi %out_iter, %c1 : index

      // Compare the input value with the counter
      %b = arith.cmpi "slt", %in_val, %count_iter : i32
      %count_next, %in_next, %out_next = scf.if %b -> (i32, index, index) {
        // Input is less than counter.
        // Pass along the input and continue to the next one.
        memref.store %in_val, %bufout[%out_iter] : memref<"""
        + str(bufsize)
        + """xi32>
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
    )
    f.write(tilecode)

    f.write("\n")

    def gen_core(col, row, col_p, row_p):
        if col == startcol and row == 1:
            None
        else:
            f.write("  %%core%d_%d = AIE.core(%%tile%d_%d) {\n" % (col, row, col, row))
            f.write('    AIE.useLock(%%lock%d_%d, "Acquire", 1)\n' % (col_p, row_p))
            f.write('    AIE.useLock(%%lock%d_%d, "Acquire", 0)\n' % (col, row))
            f.write(
                "    func.call @do_sieve(%%buf%d_%d, %%buf%d_%d) : (memref<%dxi32>, memref<%dxi32>) -> ()\n"
                % (col_p, row_p, col, row, bufsize, bufsize)
            )
            f.write('    AIE.useLock(%%lock%d_%d, "Release", 0)\n' % (col_p, row_p))
            f.write('    AIE.useLock(%%lock%d_%d, "Release", 1)\n' % (col, row))
            f.write("    AIE.end\n")
            f.write("  }\n\n")

    col = startcol
    row = 1
    col_p = -1
    row_p = -1
    for i in range(0, arraycols, 2):
        for j in range(1, arrayrows):
            gen_core(col, row, col_p, row_p)
            col_p = col
            row_p = row
            row = row + 1
        gen_core(col, row, col_p, row_p)
        col_p = col
        row_p = row
        col = col + 1
        for j in range(1, arrayrows):
            gen_core(col, row, col_p, row_p)
            col_p = col
            row_p = row
            row = row - 1
        gen_core(col, row, col_p, row_p)
        col_p = col
        row_p = row
        col = col + 1

    f.write("}\n")


if __name__ == "__main__":
    main()
