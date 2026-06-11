# time_block.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
"""Quick HW timing for a built m8 xclbin (or any per-block xclbin).

Usage:
  python3 scripts/time_block.py --block m8 -x build/final_m8.xclbin -i build/insts_m8.bin -k MLIR_AIE
"""

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

import yolo_spec  # noqa: E402


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--block", required=True)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-iters", type=int, default=20)
    opts = p.parse_args()

    blk = yolo_spec.block(opts.block)
    in_w, in_h, in_c = blk.layers[0].in_shape
    last_out_shape = blk.layers[-1].out_shape

    in_bytes = in_w * in_h * (8 if opts.block == "m0" else in_c)
    out_bytes = int(np.prod(last_out_shape))

    rng = np.random.default_rng(seed=0)
    in_data = rng.integers(-128, 128, size=(in_bytes,), dtype=np.int8)

    in_tensor = iron.tensor(in_data, dtype=np.int8)
    out_tensor = iron.zeros([out_bytes], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rt = DefaultNPURuntime

    print(f"{opts.block}: warmup x{opts.n_warmup}, time x{opts.n_iters}")
    for _ in range(opts.n_warmup):
        rt.load_and_run(npu_opts.npu_kernel, [in_tensor, out_tensor])

    times_ms = []
    for _ in range(opts.n_iters):
        _h, result = rt.load_and_run(npu_opts.npu_kernel, [in_tensor, out_tensor])
        times_ms.append(result.npu_time / 1e6)

    arr = np.array(times_ms)
    print(
        f"{opts.block}: n={opts.n_iters} mean={arr.mean():.2f} ms "
        f"min={arr.min():.2f} ms median={float(np.median(arr)):.2f} ms "
        f"max={arr.max():.2f} ms std={arr.std():.2f} ms"
    )
    print(
        f"{opts.block}: throughput @ median = {1000.0 / float(np.median(arr)):.2f} fps"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
