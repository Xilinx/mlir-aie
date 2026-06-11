"""Stage 0 plumbing test: confirm the device can modify a host-filled KV
buffer in-array, drain it back, AND have a second worker read the written
slot (read-after-write). Bit-exact host check of the deterministic pattern.
"""

from __future__ import annotations

import struct
import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_kv_proto import HEAD_D, T, PREFIX, SCALE_BYTES, BODY_BYTES, KHALF, PER_HEAD


def run_one(pos: int, opts, npu_kernel) -> int:
    rng = np.random.default_rng(pos)
    kv_in = rng.integers(-50, 50, size=PER_HEAD, dtype=np.int8)
    kv_in[0:4] = np.frombuffer(np.int32(pos).tobytes(), dtype=np.int8)

    kv_in_t = iron.tensor(kv_in.copy(), dtype=np.int8)
    kv_out_t = iron.zeros([PER_HEAD], dtype=np.int8)
    read_t = iron.zeros([16], dtype=np.int8)
    rc = DefaultNPURuntime.run_test(
        npu_kernel,
        [kv_in_t, kv_out_t, read_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        print(f"pos {pos}: dispatch returned {rc}", file=sys.stderr)
        return rc
    kv_out_t.to("cpu")
    read_t.to("cpu")
    kv_out = kv_out_t.numpy()
    read = read_t.numpy()

    # Expected: kv_out == kv_in except slot[pos] overwritten.
    exp = kv_in.copy()
    k_scales = PREFIX
    k_body = k_scales + SCALE_BYTES
    v_scales = k_body + BODY_BYTES
    v_body = v_scales + SCALE_BYTES
    exp[k_body + pos * HEAD_D : k_body + (pos + 1) * HEAD_D] = np.int8(pos + 1)
    exp[v_body + pos * HEAD_D : v_body + (pos + 1) * HEAD_D] = np.int8(pos + 2)
    exp[k_scales + pos * 4 : k_scales + pos * 4 + 4] = np.frombuffer(
        np.float32(0.0123).tobytes(), dtype=np.int8
    )
    exp[v_scales + pos * 4 : v_scales + pos * 4 + 4] = np.frombuffer(
        np.float32(0.0456).tobytes(), dtype=np.int8
    )

    drain_ok = bool(np.array_equal(kv_out, exp))
    # read-after-write proof
    rk = int(read[0])
    rv = int(read[1])
    rks = struct.unpack("<f", read[4:8].tobytes())[0]
    rvs = struct.unpack("<f", read[8:12].tobytes())[0]
    read_ok = (
        rk == np.int8(pos + 1)
        and rv == np.int8(pos + 2)
        and struct.pack("<f", rks) == struct.pack("<f", np.float32(0.0123))
        and struct.pack("<f", rvs) == struct.pack("<f", np.float32(0.0456))
    )
    if drain_ok and read_ok:
        print(f"pos {pos}: PASS  (drain bit-exact; read-after-write k={rk} v={rv})")
        return 0
    nmis = int((kv_out.astype(np.int32) != exp.astype(np.int32)).sum())
    print(
        f"pos {pos}: FAIL  drain_ok={drain_ok} (mismatch={nmis}) read_ok={read_ok} "
        f"rk={rk} rv={rv} rks={rks:.6g} rvs={rvs:.6g}"
    )
    return 1


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--positions", type=str, default="0,1,5,42,127")
    opts = p.parse_args()
    positions = [int(s) for s in opts.positions.split(",")]
    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel
    fails = sum(run_one(pp, opts, npu_kernel) != 0 for pp in positions)
    print(f"\nkv_proto: {len(positions) - fails}/{len(positions)} positions PASS")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
