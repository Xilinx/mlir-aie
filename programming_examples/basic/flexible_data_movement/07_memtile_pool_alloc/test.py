# test.py -*- Python -*-
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# Test for Prototype 7: MemTile Pool Allocator

import numpy as np
import sys
import aie.utils.test as test_utils
import aie.iron as iron
from aie.utils import DefaultNPURuntime


def main(opts):
    chunk_size = 256
    total_size = 2 * chunk_size

    in_dtype = np.uint8
    in_volume = total_size // np.dtype(in_dtype).itemsize
    out_volume = total_size // np.dtype(in_dtype).itemsize

    ref_in = np.arange(0, in_volume, dtype=in_dtype)
    # Passthrough: output should equal input
    ref_out = ref_in.copy()

    in1 = iron.tensor(ref_in, dtype=in_dtype)
    out = iron.zeros([out_volume], dtype=in_dtype)

    print("Running Prototype 7: MemTile Pool Allocator...")
    print(f"  Pool size:     2048 bytes (single aie.buffer)")
    print(f"  Total data:    {total_size} bytes")
    print(f"  Chunk/core:    {chunk_size} bytes")
    print(f"  Cores:         2 (Core(0,2) and Core(0,3))")
    print(f"  MemTile DMAs:  ALL configured at runtime (no @memtile_dma)")
    print(f"  Pattern:       DDR → pool[0:512] → Core(0,2) + Core(0,3) → pool[512:1024] → DDR")
    print(f"  Key:           ONE pool buffer, sub-regions via dma_bd(pool, offset=N)")

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
