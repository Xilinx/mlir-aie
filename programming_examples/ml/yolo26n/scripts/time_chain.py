"""Quick HW timing for a built chain xclbin (m0..m10).

Sizes input/output buffers from CHAIN_N_SAMPLES (must match what the xclbin
was built with) and the yolo_spec input/output shapes. Runs a small number
of iterations and reports mean/median per-sample times.

Usage:
  CHAIN_N_SAMPLES=1 python3 scripts/time_chain.py \\
    -x build/final_chain.xclbin -i build/insts_chain.bin -k MLIR_AIE
"""

import os
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
    p.add_argument("--n-iters", type=int, default=10)
    p.add_argument("--n-warmup", type=int, default=2)
    opts = p.parse_args()

    N = int(os.environ.get("CHAIN_N_SAMPLES", "1"))
    chain_env = os.environ.get("CHAIN_BLOCKS", "").strip()
    if chain_env:
        chain_blocks = chain_env.split(",")
    else:
        chain_blocks = [
            "m0",
            "m1",
            "m2",
            "m3",
            "m4",
            "m5",
            "m6",
            "m7",
            "m8",
            "m9",
            "m10",
        ]
    last_block_name = chain_blocks[-1]

    m0 = yolo_spec.block("m0")
    in_w, in_h, _ = m0.layers[0].in_shape
    IN_C_PADDED = 8
    in_bytes_per = in_w * in_h * IN_C_PADDED

    last = yolo_spec.block(last_block_name)
    last_shape = last.layers[-1].out_shape
    if last.topology == "head":
        out_bytes_per = max(4, ((int(np.prod(last_shape)) + 3) // 4) * 4)
    else:
        out_bytes_per = int(np.prod(last_shape))

    rng = np.random.default_rng(seed=0)
    in_data = rng.integers(-128, 128, size=(N * in_bytes_per,), dtype=np.int8)
    in_tensor = iron.tensor(in_data, dtype=np.int8)
    out_tensor = iron.zeros([N * ((out_bytes_per + 3) // 4) * 4], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rt = DefaultNPURuntime

    print(f"chain N={N}: warmup x{opts.n_warmup}, time x{opts.n_iters}")
    for _ in range(opts.n_warmup):
        rt.load_and_run(npu_opts.npu_kernel, [in_tensor, out_tensor])

    times_ms = []
    for _ in range(opts.n_iters):
        _h, result = rt.load_and_run(npu_opts.npu_kernel, [in_tensor, out_tensor])
        times_ms.append(result.npu_time / 1e6)

    arr = np.array(times_ms)
    per_sample = arr / N
    print(
        f"chain N={N}: per-dispatch n={opts.n_iters} mean={arr.mean():.2f} ms "
        f"min={arr.min():.2f} ms median={float(np.median(arr)):.2f} ms "
        f"max={arr.max():.2f} ms std={arr.std():.2f} ms"
    )
    print(
        f"chain N={N}: per-sample median={float(np.median(per_sample)):.2f} ms "
        f"-> {1000.0 / float(np.median(per_sample)):.2f} fps"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
