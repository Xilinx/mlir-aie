"""Test the on-chip embed stream+select probe vs the numpy embed math."""

from __future__ import annotations

import struct
import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_embedsel_probe import V, D, N_TILE, N_TILES, SLOT, TOTAL, TILE_PREFIX, TOKEN


def pack_table(embed_i8, embed_sc):
    blob = np.zeros(TOTAL, dtype=np.int8)
    for t in range(N_TILES):
        base = t * SLOT
        off = base + TILE_PREFIX
        rows = embed_i8[t * N_TILE : (t + 1) * N_TILE].flatten()
        blob[off : off + rows.size] = rows
        off += rows.size
        off += N_TILE * 4  # bias (zero)
        sc = embed_sc[t * N_TILE : (t + 1) * N_TILE].astype(np.float32)
        blob[off : off + N_TILE * 4] = np.frombuffer(sc.tobytes(), dtype=np.int8)
    return blob


def oracle(embed_i8, embed_sc, tok):
    row = embed_i8[tok].astype(np.float32) * np.float32(embed_sc[tok])
    sc = np.float32(max(float(np.abs(row).max()), 1e-12) / 127.0)
    s = row * np.float32(1.0 / sc)
    r = np.where(s >= 0, np.floor(s + np.float32(0.5)), np.ceil(s - np.float32(0.5)))
    xin = r.clip(-128, 127).astype(np.int8)
    return xin, float(sc)


def main():
    opts = test_utils.create_default_argparser().parse_args()
    rng = np.random.default_rng(0)
    embed_i8 = rng.integers(-128, 128, size=(V, D), dtype=np.int8)
    embed_sc = rng.uniform(0.001, 0.02, size=V).astype(np.float32)
    blob = pack_table(embed_i8, embed_sc)

    npu = test_utils.create_npu_kernel(opts).npu_kernel
    t_t = iron.tensor(blob, dtype=np.int8)
    o_t = iron.zeros([D + 8], dtype=np.int8)
    rc = DefaultNPURuntime.run_test(npu, [t_t, o_t], {}, verify=False, verbosity=opts.verbosity)
    if rc != 0:
        print(f"dispatch returned {rc}", file=sys.stderr)
        return rc
    o_t.to("cpu")
    out = o_t.numpy()
    dev_xin = out[:D]
    dev_scale = struct.unpack("<f", out[D : D + 4].tobytes())[0]

    ref_xin, ref_scale = oracle(embed_i8, embed_sc, TOKEN)
    body_ok = np.array_equal(dev_xin, ref_xin)
    scale_ok = struct.pack("<f", dev_scale) == struct.pack("<f", np.float32(ref_scale))
    nbad = int((dev_xin != ref_xin).sum())
    print(f"embedsel token={TOKEN}: body {'OK' if body_ok else f'{nbad}/{D} bad'}  "
          f"scale dev={dev_scale:.8g} ref={ref_scale:.8g} {'OK' if scale_ok else 'DIFF'}")
    ok = body_ok and scale_ok
    print(f"embedsel_probe: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
