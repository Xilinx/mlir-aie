# Copyright (C) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import aie.iron as iron
import numpy as np
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.utils.config import cxx_header_path
from aie.utils.verify import assert_pass
from ml_dtypes import bfloat16

# The AI Engine kernels are a string here, so the design needs no separate .cc file.
_KERNEL_SRC = """
#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

extern "C" {
void saxpy(bfloat16 *restrict x, bfloat16 *restrict y, bfloat16 *restrict z) {
  event0();
  ::aie::vector<bfloat16, 64> a_v = ::aie::broadcast<bfloat16, 64>(3.f);
#pragma clang loop min_iteration_count(4)
  for (int i = 0; i < 4096; i += 64) {
    ::aie::vector<bfloat16, 64> x_v = ::aie::load_v<64>(x);
    x += 64;
    ::aie::vector<bfloat16, 64> y_v = ::aie::load_v<64>(y);
    y += 64;
    ::aie::accum<accfloat, 64> ax_v = ::aie::mul(x_v, a_v);
    ::aie::accum<accfloat, 64> z_v = ::aie::add(ax_v, y_v);
    ::aie::vector<bfloat16, 64> z_v_converted = z_v.to_vector<bfloat16>();
    ::aie::store_v(z, z_v_converted);
    z += 64;
  }
  event1();
}

// saxpy_scalar: a non-vectorized reference SAXPY. Select it from Python by
// changing the ExternalFunction name from "saxpy" to "saxpy_scalar".
void saxpy_scalar(bfloat16 *x, bfloat16 *y, bfloat16 *z) {
  event0();
  float a = 3.f;
  for (int i = 0; i < 4096; ++i) {
    z[i] = a * x[i] + y[i];
  }
  event1();
}
}
"""


# JIT decorator for IRON
# Decorator to compile an IRON kernel into a binary to run on the NPU.
# Parameters:
#     - use_cache (bool): Use cached MLIR module if available. Defaults to True.
@iron.jit
def saxpy(
    input0: In,
    input1: In,
    output: Out,
    *,
    N: CompileTime[int],
    element_type: CompileTime[type]
):

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(N,), np.dtype[element_type]]
    out_ty = np.ndarray[(N,), np.dtype[element_type]]

    of_x = ObjectFifo(in_ty, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    # The kernel acquires input tensors X and Y, and output tensor Z, performs the
    # SAXPY operation on X and Y, and writes the result in Z.

    saxpy_kernel = ExternalFunction(
        "saxpy",
        source_string=_KERNEL_SRC,
        arg_types=[in_ty, in_ty, out_ty],
        include_dirs=[cxx_header_path()],
    )

    def core_body(of_x, of_y, of_z, saxpy_kernel):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        saxpy_kernel(elem_x, elem_y, elem_z)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)

    worker = Worker(
        core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), saxpy_kernel]
    )

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    def sequence(a_x, a_y, c_z, x_prod, y_prod, z_cons):
        x_prod.fill(a_x)
        y_prod.fill(a_y)
        z_cons.drain(c_z, wait=True)

    rt = Runtime(
        sequence,
        [in_ty, in_ty, out_ty, of_x.prod(), of_y.prod(), of_z.cons()],
    )

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt, workers=[worker])
    return my_program.resolve_program()


def main():
    # Define tensor shapes and data types.
    # NOTE: the kernel source hardcodes the loop bound to 4096 elements; this
    # must match data_size or the kernel will produce silently wrong results.
    data_size = 4096
    element_type = bfloat16

    # Construct an input tensor and an output zeroed tensor
    # The two tensors are in memory accessible to the NPU
    input0 = iron.arange(data_size, dtype=element_type, device="npu")
    input1 = iron.arange(data_size, dtype=element_type, device="npu")
    output = iron.zeros_like(input0)

    # JIT-compile the kernel then launches the kernel with the given arguments. Future calls
    # to the kernel will use the same compiled kernel and loaded code objects
    saxpy(input0, input1, output, N=data_size, element_type=element_type)

    # Numpy reference of `3*x + y` and a one-line tolerance check.
    ref = 3 * input0.numpy() + input1.numpy()
    assert_pass(output.numpy(), ref, fail_msg="saxpy output does not match 3*x + y")


if __name__ == "__main__":
    main()
