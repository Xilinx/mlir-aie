# test.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# Test for Prototype 5: Multi-Column Bandwidth Scaling

import numpy as np
import sys
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DefaultNPURuntime


def main(opts):
    n_cols = 2
    cores_per_col = 2
    chunk_size = 256
    col_data_size = cores_per_col * chunk_size
    total_size = n_cols * col_data_size

    in_dtype = np.uint8
    out_dtype = np.uint8

    in_volume = total_size // np.dtype(in_dtype).itemsize
    out_volume = total_size // np.dtype(out_dtype).itemsize

    ref_in = np.arange(0, in_volume, dtype=in_dtype)
    ref_out = ref_in.copy()

    in1 = iron.tensor(ref_in, dtype=in_dtype)
    out = iron.zeros([out_volume], dtype=out_dtype)

    print("Running Prototype 5: Multi-Column Bandwidth Scaling...")
    print(f"  Total size:    {total_size} bytes")
    print(f"  Columns:       {n_cols}")
    print(f"  Cores/column:  {cores_per_col}")
    print(f"  Chunk/core:    {chunk_size} bytes")
    print(f"  ShimDMA:       {n_cols*2} MM2S + {n_cols*2} S2MM (parallel columns)")

    npu_opts = test_utils.create_npu_kernel(opts)
    res = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in1, out],
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
