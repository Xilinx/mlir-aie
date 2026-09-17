# Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

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
    This test spawns multiple processes that compile the same external kernel
    concurrently to ensure cache locking and kernel source materialization work
    correctly across platforms.
    """

    # Create a temporary cache directory for this test
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        kernel_path = os.path.join(temp_cache_dir, "kernel.cc")
        with open(kernel_path, "w") as f:
            f.write(
                """extern "C" {
void copy_with_bias(int *input, int *output, int tile_size) {
  for (int i = 0; i < tile_size; ++i) {
    output[i] = input[i] + 7;
  }
}
}
"""
            )

        # Create a simple test script that does JIT compilation.
        # Uses In/Out + CompileTime[T] (the post-unify-compilation-workflow API);
        # an unannotated def simple_extern(input0, output) would trip
        # Guard 1-A / TypeError at compile time because tensor params would
        # be classified as scalar_params and never forwarded to the generator.
        test_script = """
import os
import sys
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction

from aie.iron.controlflow import range_

KERNEL_CC = os.path.join(os.path.dirname(__file__), "kernel.cc")

@iron.jit
def simple_extern(
    input0: In, output: Out,
    *, num_elements: CompileTime[int], dtype: CompileTime[type],
):
    n = 16
    if num_elements % n != 0:
        raise ValueError(f"Number of elements ({num_elements}) must be a multiple of {n}.")
    N_div_n = num_elements // n

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    ext = ExternalFunction(
        "copy_with_bias",
        source_file=KERNEL_CC,
        arg_types=[tile_ty, tile_ty, np.int32],
    )

    # AIE-array data movement with object fifos
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    # Define a task that will run on a compute tile
    def core_body(of_in, of_out, ext):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            ext(elem_in, elem_out, n)
            of_in.release(1)
            of_out.release(1)

    # Create a worker to run the task on a compute tile
    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), ext])

    # Runtime operations to move data to/from the AIE-array
    def sequence(A, C, in_h, out_h):
        in_h.fill(A)
        out_h.drain(C, wait=True)

    rt = Runtime(
        sequence,
        [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()],
    )

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()

# Test the compilation
try:
    num_elements = 16
    dtype = np.int32
    input0 = iron.randint(1, 100, (num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input0)

    if input0.shape != output.shape:
        raise ValueError("Both tensors must share the same shape.")
    if input0.dtype != output.dtype:
        raise ValueError("Both tensors must share the same dtype.")
    if len(input0.shape) != 1:
        raise ValueError("Function only supports vectors.")

    # This should trigger JIT compilation and cache access.
    simple_extern(input0, output, num_elements=num_elements, dtype=dtype)
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
            env["NPU_CACHE_HOME"] = temp_cache_dir
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

        leftovers = []
        for root, _, files in os.walk(temp_cache_dir):
            for name in files:
                if name.startswith("kernel.cc.") and name.endswith(".tmp"):
                    leftovers.append(os.path.join(root, name))
        assert not leftovers, f"orphaned staged kernel sources left behind: {leftovers}"
