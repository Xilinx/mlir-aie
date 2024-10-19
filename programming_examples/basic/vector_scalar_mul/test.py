# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
import time
from aie.utils.xrt import setup_aie, write_out_trace, execute
import aie.utils.test as test_utils


def main(opts):
    print("Running...\n")

    data_size = int(opts.size)
    vector_dtype = np.int16
    scalar_dtype = np.int32
    scale_factor = 3
    size_out = data_size * 2
    print("output buffer size: " + str(size_out))

    enable_trace = opts.trace_size > 0

    app = setup_aie(
        opts.xclbin,
        opts.instr,
        data_size,
        vector_dtype,
        1,
        scalar_dtype,
        data_size,
        vector_dtype,
        enable_trace=enable_trace,
        trace_size=opts.trace_size,
    )
    input_vector = np.arange(1, data_size + 1, dtype=vector_dtype)
    input_factor = np.array([3], dtype=scalar_dtype)
    # aie_output = execute_on_aie(app, input_vector, input_factor)

    start = time.time_ns()
    full_output = execute(app, input_vector, input_factor)
    stop = time.time_ns()
    npu_time = stop - start
    print("npu_time: ", npu_time)

    # aie_output = full_output[:size_out].view(np.int8)
    # aie_output = full_output[:size_out].view(np.uint8)
    aie_output = full_output[:size_out].view(np.int16)
    if enable_trace:
        trace_buffer = full_output[size_out:].view(np.uint32)

    ref = np.arange(1, data_size + 1, dtype=vector_dtype) * scale_factor

    if enable_trace:
        # trace_buffer = full_output[3920:]
        print("trace_buffer shape: ", trace_buffer.shape)
        print("trace_buffer dtype: ", trace_buffer.dtype)
        # write_out_trace(trace_buffer, str(opts.trace_file))
        write_out_trace(trace_buffer, "trace.txt")

    # Copy output results and verify they are correct
    errors = 0
    if opts.verify:
        if opts.verbosity >= 1:
            print("Verifying results ...")
        e = np.equal(ref, aie_output)
        errors = np.size(e) - np.count_nonzero(e)

    if not errors:
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument("-s", "--size", required=True, dest="size", help="Vector size")
    opts = p.parse_args(sys.argv[1:])
    main(opts)
