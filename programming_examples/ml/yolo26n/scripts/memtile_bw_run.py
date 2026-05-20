"""Runner for memtile_bw_bench: loads the built xclbin, runs once, prints MB/s.

Use:
  python3 scripts/memtile_bw_run.py --mode single \
      -x build/final_bw_single.xclbin -i build/insts_bw_single.bin -k MLIR_AIE
  python3 scripts/memtile_bw_run.py --mode dual \
      -x build/final_bw_dual.xclbin -i build/insts_bw_dual.bin -k MLIR_AIE
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import memtile_bw_bench as B  # noqa: E402


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--mode", choices=["single", "dual"], required=True)
    opts = p.parse_args(sys.argv[1:])

    n = B.N_SAMPLES
    payload = B.PAYLOAD_BYTES

    if opts.mode == "single":
        in_a = iron.tensor(np.arange(n, dtype=np.int32), dtype=np.int32)
        out_a = iron.zeros([n], dtype=np.int32)
        args = [in_a, out_a]
        streams = 1
    else:
        in_a = iron.tensor(np.arange(n, dtype=np.int32), dtype=np.int32)
        out_a = iron.zeros([n], dtype=np.int32)
        in_b = iron.tensor(np.arange(n, dtype=np.int32), dtype=np.int32)
        out_b = iron.zeros([n], dtype=np.int32)
        args = [in_a, out_a, in_b, out_b]
        streams = 2

    npu_opts = test_utils.create_npu_kernel(opts)
    print(f"Running NPU bench mode={opts.mode} ...")
    rt = DefaultNPURuntime
    handle = rt.load(npu_opts.npu_kernel)
    result = rt.run(handle, args)

    if result.ret.name != "ERT_CMD_STATE_COMPLETED":
        print(f"FAIL: result.ret={result.ret.name}")
        return 1

    elapsed_ns = result.npu_time
    elapsed_s = elapsed_ns / 1e9
    total_bytes_per_stream = n * payload
    total_bytes = total_bytes_per_stream * streams
    per_stream_mbps = (total_bytes_per_stream / 1e6) / elapsed_s
    aggregate_mbps = (total_bytes / 1e6) / elapsed_s
    per_sample_us = (elapsed_s / n) * 1e6

    print(f"mode               : {opts.mode}")
    print(f"streams            : {streams}")
    print(f"payload per stream : {payload} B per dispatch")
    print(f"chunks/dispatch    : {B.N_CHUNKS} (chunk={payload // B.N_CHUNKS} B)")
    print(f"samples            : {n}")
    print(f"elapsed            : {elapsed_s * 1e3:.3f} ms ({per_sample_us:.2f} us/sample)")
    print(f"per-stream BW      : {per_stream_mbps:.1f} MB/s")
    print(f"aggregate BW       : {aggregate_mbps:.1f} MB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
