# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
import os
import tempfile
import subprocess
import sys


def test_parallel_compilation_subprocess():
    """
    Test parallel JIT compilation using subprocesses.
    This test spawns multiple processes that compile the same kernel concurrently
    to ensure the file locking mechanism works correctly.
    """

    # Create a temporary cache directory for this test
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        # Create a simple test script that does JIT compilation
        test_script = """
import sys
import numpy as np
import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

@iron.jit(is_placed=False)
def simple_add(input0, input1, output):
    if input0.shape != input1.shape:
        raise ValueError(f"Input shapes are not equal ({input0.shape} != {input1.shape}).")
    if input0.shape != output.shape:
        raise ValueError(f"Input and output shapes are not equal ({input0.shape} != {output.shape}).")
    if len(np.shape(input0)) != 1:
        raise ValueError("Function only supports vectors.")
    num_elements = np.size(input0)
    n = 16
    if num_elements % n != 0:
        raise ValueError(f"Number of elements ({num_elements}) must be a multiple of {n}.")
    N_div_n = num_elements // n

    if input0.dtype != input1.dtype:
        raise ValueError(f"Input data types are not the same ({input0.dtype} != {input1.dtype}).")
    if input0.dtype != output.dtype:
        raise ValueError(f"Input and output data types are not the same ({input0.dtype} != {output.dtype}).")
    dtype = input0.dtype

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    # AIE-array data movement with object fifos
    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_in2 = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(n):
                elem_out[i] = elem_in1[i] + elem_in2[i]
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in1.cons(), of_in2.cons(), of_out.prod()])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in1.prod(), A)
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())

# Test the compilation
try:
    num_elements = 16
    dtype = np.int32
    input0 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    input1 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    # This should trigger JIT compilation and cache access
    simple_add(input0, input1, output)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    sys.exit(1)
"""

        # Write the test script to a temporary file
        script_path = os.path.join(temp_cache_dir, "test_compilation.py")
        with open(script_path, "w") as f:
            f.write(test_script)

        # Run multiple subprocesses concurrently
        num_processes = 5
        processes = []

        for i in range(num_processes):
            env = os.environ.copy()
            env["IRON_CACHE_HOME"] = temp_cache_dir
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            processes.append(process)

        # Wait for all processes to complete and collect return codes
        return_codes = []
        process_outputs = []

        for i, process in enumerate(processes):
            stdout, stderr = process.communicate()
            return_codes.append(process.returncode)
            process_outputs.append((stdout, stderr))

            # Print output for each process for debugging
            print(f"\n=== Process {i} (return code: {process.returncode}) ===")
            print(f"STDOUT:\n{stdout}")
            print(f"STDERR:\n{stderr}")
            print("=" * 50)

        # Count successful processes (return code 0)
        successful_processes = sum(1 for code in return_codes if code == 0)

        # Verify that all processes completed
        assert len(return_codes) == num_processes, "All processes should complete"

        # Check if any concurrent compilation failed
        if successful_processes < num_processes:
            # Create detailed error message with all process outputs
            error_msg = (
                f"Only {successful_processes}/{num_processes} processes succeeded\n\n"
            )
            error_msg += "Process details:\n"

            for i, (return_code, (stdout, stderr)) in enumerate(
                zip(return_codes, process_outputs)
            ):
                status = "SUCCESS" if return_code == 0 else "FAILED"
                error_msg += f"\nProcess {i}: {status} (return code: {return_code})\n"
                if stdout:
                    error_msg += f"  STDOUT: {stdout.strip()}\n"
                if stderr:
                    error_msg += f"  STDERR: {stderr.strip()}\n"

            pytest.fail(error_msg)
