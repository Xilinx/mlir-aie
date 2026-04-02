# test.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# Test for Prototype 4: BD-Chained Streaming

import numpy as np
import sys
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DefaultNPURuntime


def main(opts):
    n_cores = 2
    chunk_size = 512
    n_iterations = 8
    repeat_count = 2
    elements_per_core = chunk_size // n_cores

    total_in_size = n_iterations * chunk_size
    total_out_size = n_iterations * chunk_size * repeat_count

    in_dtype = np.uint8
    out_dtype = np.uint8

    in_volume = total_in_size // np.dtype(in_dtype).itemsize
    out_volume = total_out_size // np.dtype(out_dtype).itemsize

    # Input: sequential bytes
    ref_in = np.arange(0, in_volume, dtype=in_dtype)

    # Expected output: each input chunk is repeated repeat_count times
    # Input chunks are split to cores, then each core's data repeats
    ref_out = np.zeros(out_volume, dtype=out_dtype)
    for it in range(n_iterations):
        for rep in range(repeat_count):
            out_idx = (it * repeat_count + rep) * chunk_size
            in_idx = it * chunk_size
            ref_out[out_idx : out_idx + chunk_size] = ref_in[
                in_idx : in_idx + chunk_size
            ]

    in1 = iron.tensor(ref_in, dtype=in_dtype)
    # Third buffer is unused but required by runtime_sequence
    dummy = iron.tensor(ref_in, dtype=in_dtype)
    out = iron.zeros([out_volume], dtype=out_dtype)

    print("Running Prototype 4: BD-Chained Streaming...")
    print(f"  Input:         {total_in_size} bytes ({n_iterations} chunks)")
    print(f"  Output:        {total_out_size} bytes (repeat_count={repeat_count})")
    print(f"  Cores:         {n_cores}")
    print(f"  ShimDMA:       1 BD command -> {n_iterations} autonomous iterations")

    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, out, dummy],
        {1: ref_out},
        verify=npu_opts.verify,
        verbosity=npu_opts.verbosity,
    )
    if res == 0:
        print("\nPASS!\n")
    else:
        print("\nFAIL!\n")
    sys.exit(res)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    opts = p.parse_args(sys.argv[1:])
    main(opts)
