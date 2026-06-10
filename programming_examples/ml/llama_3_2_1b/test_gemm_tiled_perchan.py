"""Phase 6c.5b.1a: per-channel-weight tiled GEMM bit-exact test.

Generates random fp32 weights (not pre-int8), quantizes per-channel via
quant_int8_perchan_absmax (same recipe as gen_llama_data.py), runs the
NPU per-channel GEMM, compares bit-exact against a numpy reference that
does identical per-channel dequant + IEEE-fp32 re-quantize.

Calibration: the test runs the float math first to determine output
absmax, derives the bit-exact-friendly out_scale = max/127. Both NPU
and numpy use that same out_scale (passed to the kernel as
inv_out_scale = 1/out_scale; passed to numpy ref as the same value).

This is the simplest viable proof that the kernel's per-channel weight
dequant matches numpy bit-for-bit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from gen_llama_data import quant_int8_perchan_absmax


def numpy_gemm_perchan(
    act_i8: np.ndarray,
    x_scale: float,
    w_i8: np.ndarray,
    w_scales: np.ndarray,
    bias_i32: np.ndarray,
    inv_out_scale: float,
) -> np.ndarray:
    """Bit-exact-with-kernel per-channel int8 GEMM.

    Math (per output channel n):
        acc[n]    = sum_k(w_i8[n,k] * act_i8[k]) + bias_i32[n]   # int32
        fp[n]     = float(acc[n]) * x_scale * w_scales[n]
        scaled[n] = fp[n] * inv_out_scale
        out[n]    = clamp(round_half_away_from_zero(scaled[n]), -128, 127)
    """
    acc = w_i8.astype(np.int32) @ act_i8.astype(np.int32) + bias_i32  # (N,)
    fp = (
        acc.astype(np.float32) * np.float32(x_scale) * w_scales.astype(np.float32)
    )  # (N,)
    scaled = fp * np.float32(inv_out_scale)
    rounded = np.where(
        scaled >= 0,
        np.floor(scaled + np.float32(0.5)),
        np.ceil(scaled - np.float32(0.5)),
    )
    return rounded.clip(-128, 127).astype(np.int8)


def pack_tiled_perchan(
    w_i8: np.ndarray, w_scales: np.ndarray, bias_i32: np.ndarray, n_tile: int
) -> np.ndarray:
    """Pack per-tile slots: [N_TILE*K i8 weights | N_TILE i32 bias | N_TILE fp32 w_scales].
    Returns flat int8 blob of size n_tiles * slot_bytes."""
    N, K = w_i8.shape
    assert N % n_tile == 0
    n_tiles = N // n_tile
    slot = n_tile * K + n_tile * 4 + n_tile * 4
    blob = np.zeros(n_tiles * slot, dtype=np.int8)
    for t in range(n_tiles):
        base = t * slot
        rows = w_i8[t * n_tile : (t + 1) * n_tile].flatten()
        bs = bias_i32[t * n_tile : (t + 1) * n_tile].view(np.int8).flatten()
        ws = w_scales[t * n_tile : (t + 1) * n_tile].view(np.int8).flatten()
        off = base
        blob[off : off + rows.size] = rows
        off += rows.size
        blob[off : off + bs.size] = bs
        off += bs.size
        blob[off : off + ws.size] = ws
    return blob


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("-N", type=int, default=64)
    p.add_argument("--n-tile", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbosity", type=int, default=0)
    opts = p.parse_args()

    K, N, n_tile = opts.K, opts.N, opts.n_tile
    rng = np.random.default_rng(opts.seed)

    # --- Generate fp32 weights with heterogeneous per-channel magnitudes
    #     so per-channel quantization is a non-trivial test. Each row is
    #     scaled by a random factor in [0.1, 1.0]. ---
    base_w = rng.standard_normal((N, K)).astype(np.float32) * 0.1
    row_scale = rng.uniform(0.1, 1.0, size=N).astype(np.float32)
    w_fp32 = base_w * row_scale[:, None]
    bias_i32 = rng.integers(-100, 100, size=N, dtype=np.int32)

    # --- Quantize per-channel ---
    w_i8, w_scales = quant_int8_perchan_absmax(w_fp32)
    # Sanity: w_scales should be HETEROGENEOUS (different per row).
    print(
        f"per-channel scales: min={w_scales.min():.4f}  max={w_scales.max():.4f}  "
        f"std={w_scales.std():.4f}"
    )

    # --- Activation (synthetic INT8). Choose act_scale that gives a known
    #     fp32 input range. ---
    act_i8 = rng.integers(-127, 128, size=K, dtype=np.int8)
    act_scale = 0.01  # arbitrary but stable across test invocations

    # --- Calibrate inv_out_scale from numpy fp32 math BEFORE we know what
    #     the int8 output looks like. ---
    acc = w_i8.astype(np.int32) @ act_i8.astype(np.int32) + bias_i32
    fp_out = (
        acc.astype(np.float32) * np.float32(act_scale) * w_scales.astype(np.float32)
    )
    out_scale = float(np.maximum(np.abs(fp_out).max(), 1e-12)) / 127.0
    inv_out_scale = float(np.float32(1.0) / np.float32(out_scale))
    print(f"calibration: out_scale={out_scale:.6f}  inv_out_scale={inv_out_scale:.6f}")

    # --- numpy reference ---
    expected = numpy_gemm_perchan(
        act_i8, act_scale, w_i8, w_scales, bias_i32, inv_out_scale
    )
    print(
        f"numpy expected: range=[{int(expected.min())}, {int(expected.max())}]  "
        f"saturated={int((expected==127).sum() + (expected==-128).sum())}/{N}"
    )

    # --- Re-build the xclbin with the calibrated scales baked in
    #     (Python closure constants on the IRON kernel call). Avoids the
    #     "scales must match what was compiled" footgun. Rebuild is
    #     ~5-15s. Skip if the latest build's scales already match. ---
    build_dir = Path(__file__).parent / "build"
    stamp_path = build_dir / f"perchan_K{K}_N{N}_t{n_tile}.stamp"
    stamp = f"{act_scale}|{inv_out_scale}"
    rebuild = not stamp_path.exists() or stamp_path.read_text().strip() != stamp
    if rebuild:
        build_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"rebuilding xclbin with act_scale={act_scale} inv_out_scale={inv_out_scale} ..."
        )
        # Force re-MLIR by removing the previous MLIR and xclbin.
        for f in [
            "aie_gemm_tiled_perchan_K{K}_N{N}_t{nt}.mlir",
            "final_gemm_tiled_perchan_K{K}_N{N}_t{nt}.xclbin",
            "insts_gemm_tiled_perchan_K{K}_N{N}_t{nt}.bin",
        ]:
            (build_dir / f.format(K=K, N=N, nt=n_tile)).unlink(missing_ok=True)
        env = {
            **os.environ,
            "LLAMA_GEMM_ACT_SCALE": str(act_scale),
            "LLAMA_GEMM_INV_OUT_SCALE": str(inv_out_scale),
        }
        rc = subprocess.run(
            [
                "make",
                "tiled_gemm_perchan",
                f"TILED_PC_K={K}",
                f"TILED_PC_N={N}",
                f"TILED_PC_N_TILE={n_tile}",
                f"LLAMA_GEMM_ACT_SCALE={act_scale}",
                f"LLAMA_GEMM_INV_OUT_SCALE={inv_out_scale}",
            ],
            cwd=Path(__file__).parent,
            env=env,
            capture_output=True,
            text=True,
        )
        if rc.returncode != 0:
            print(f"make failed:\n{rc.stdout[-1000:]}\n{rc.stderr[-1000:]}")
            return rc.returncode
        stamp_path.write_text(stamp)

    xclbin = build_dir / f"final_gemm_tiled_perchan_K{K}_N{N}_t{n_tile}.xclbin"
    insts = build_dir / f"insts_gemm_tiled_perchan_K{K}_N{N}_t{n_tile}.bin"

    # --- Pack weights blob and run ---
    w_blob = pack_tiled_perchan(w_i8, w_scales, bias_i32, n_tile)
    a_t = iron.tensor(act_i8, dtype=np.int8)
    w_t = iron.tensor(w_blob, dtype=np.int8)
    o_t = iron.zeros([N], dtype=np.int8)

    # Build the same npu_kernel object the standard test path uses --
    # we just point it at the freshly-rebuilt xclbin/insts.
    from argparse import Namespace

    fake_opts = Namespace(
        xclbin=str(xclbin),
        instr=str(insts),
        kernel="MLIR_AIE",
        verbosity=opts.verbosity,
        verify=False,
        iters=1,
        warmup_iters=0,
        trace_size=0,
        trace_file=None,
        in1_size=None,
        in2_size=None,
        out_size=None,
        ddr_id=None,
        enable_ctrl_pkts=False,
    )
    npu = test_utils.create_npu_kernel(fake_opts)

    rc = DefaultNPURuntime.run_test(
        npu.npu_kernel, [a_t, w_t, o_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        return rc
    o_t.to("cpu")
    actual = o_t.numpy()

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0
    print(
        f"tiled_gemm_perchan NPU vs numpy: K={K} N={N} N_TILE={n_tile}  "
        f"mismatches={n_diff}/{N}  max|diff|={max_abs}"
    )

    if n_diff == 0:
        print("BIT-EXACT PASS  (per-channel weight tiled GEMM)")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
