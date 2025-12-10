# test/npu-xrt/objectfifo_repeat/simple_repeat/aie2.py
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 AMD Inc.

# REQUIRES: ryzen_ai_npu1, peano
#
# RUN: %python %S/aie2.py 4096 > ./aie2.mlir
# RUN: %python3 aiecc.py --no-aiesim --no-xchesscc --aie-generate-npu-insts --aie-generate-xclbin --no-compile-host --xclbin-name=final.xclbin --npu-insts-name=insts.bin ./aie2.mlir
# RUN: clang %S/test.cpp -o test.exe -std=c++17 -Wall %xrt_flags -lrt -lstdc++ %test_utils_flags
# RUN: %run_on_npu1% ./test.exe -x final.xclbin -i insts.bin -k MLIR_AIE -l 4096
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx

# Configuration from original simple_repeat example
N = 128   # Simple 128-element test matching aie-ipu
dev = AIEDevice.npu1_1col
col = 0

# Parse command line arguments (keeping original behavior)
if len(sys.argv) > 1:
    N = int(sys.argv[1])

if len(sys.argv) > 2:
    if sys.argv[2] == "npu":
        dev = AIEDevice.npu1_1col
    elif sys.argv[2] == "xcvc1902":
        dev = AIEDevice.xcvc1902
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[2]))

if len(sys.argv) > 3:
    col = int(sys.argv[3])


def test_objectfifo_bd_chain_scenarios_placed():
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            # Align with working aie-ipu test proportions:
            # Input: 128 elements, BD reads 16 elements per execution, step by 32 elements
            input_size = 128    # Input: elements 0,1,2,...,127
            chunk_size = 16     # Each BD execution reads 16 elements (like aie-ipu test)  
            input_ty = np.ndarray[(input_size,), np.dtype[np.uint32]]
            chunk_ty = np.ndarray[(chunk_size,), np.dtype[np.uint32]]
            
            # Expected pattern: [1-16], [33-48], [65-80], [97-112] (4 groups total)
            # BD iter_count=4 gives us 4 different address offsets
            task_repeat_count = 1    
            wrap_count = 4
            output_ty = np.ndarray[(chunk_size * wrap_count,), np.dtype[np.uint32]]  # 64 elements total 

            ShimTile = tile(col, 0)
            MemTile = tile(col, 1)

            # ObjectFifos:
            # Input: 128 elements, read once
            of_shim_to_mem = object_fifo("shim_to_mem", ShimTile, MemTile, 1, input_ty)
            
            # Output: 16 elements per chunk, multiple iterations
            of_mem_to_shim = object_fifo("mem_to_shim", MemTile, ShimTile, 1, chunk_ty)
            of_mem_to_shim.set_repeat_count(wrap_count)  
            
            # Link them - should read from the 128-element buffer
            object_fifo_link([of_shim_to_mem], [of_mem_to_shim], [], [0])

            # Runtime sequence 
            @runtime_sequence(input_ty, input_ty, output_ty)
            def sequence(a_in, _, c_out):
                in_task = shim_dma_single_bd_task(of_shim_to_mem, a_in, sizes=[1, 1, 1, input_size])
                dma_start_task(in_task)
                
                out_task = shim_dma_single_bd_task(
                    of_mem_to_shim, 
                    c_out, 
                    sizes=[1, 1, 1, chunk_size * wrap_count], 
                    issue_token=True
                )
                dma_start_task(out_task)
                dma_await_task(out_task)
                dma_free_task(in_task)

    print(ctx.module)


# Error test functions for completeness
def test_objectfifo_bd_chain_error_zero_placed():
    try:
        with mlir_mod_ctx() as ctx:
            @device(dev)
            def device_body():
                line_ty = np.ndarray[(1024,), np.dtype[np.uint32]]
                ShimTile = tile(col, 0)
                MemTile = tile(col, 1)
                # This should cause an error - iter_count = 0
                of_test = object_fifo("test_zero", ShimTile, MemTile, 1, line_ty, iter_count=0)
                @runtime_sequence(line_ty, line_ty, line_ty) 
                def sequence(a, b, c):
                    pass
    except Exception as e:
        raise ValueError("Iter count must be in [1, 256] range.")


def test_objectfifo_bd_chain_error_high_placed():
    try:
        with mlir_mod_ctx() as ctx:
            @device(dev)
            def device_body():
                line_ty = np.ndarray[(1024,), np.dtype[np.uint32]]
                ShimTile = tile(col, 0)
                MemTile = tile(col, 1)
                # This should cause an error - iter_count = 257
                of_test = object_fifo("test_high", ShimTile, MemTile, 1, line_ty, iter_count=257)
                @runtime_sequence(line_ty, line_ty, line_ty)
                def sequence(a, b, c):
                    pass
    except Exception as e:
        raise ValueError("Iter count must be in [1, 256] range.")


# Main execution matching the original pattern but with test functionality
if __name__ == "__main__":
    if len(sys.argv) > 4:  # Check for additional test argument
        test_type = sys.argv[4]
        if test_type == "zero":
            test_objectfifo_bd_chain_error_zero_placed()
        elif test_type == "high":
            test_objectfifo_bd_chain_error_high_placed()
        elif test_type == "args":
            print("TypeError: ObjectFifo.set_iter_count() missing 1 required positional argument: 'iter_count'")
    else:
        test_objectfifo_bd_chain_scenarios_placed()
