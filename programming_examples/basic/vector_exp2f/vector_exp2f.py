# vector_exp2f/vector_exp2f.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Vector 2**x (software minimax poly), IRON + ``@iron.jit``, 4 cores, float32.

Demonstrates ``kernels.exp2f_vec``, the software-poly ``2**x`` kernel: the
accuracy alternative to the LUT-based ``kernels.bf16_exp`` that
``basic/vector_exp`` demonstrates, at ~8.9e-5 relative error across its
domain. aie2p only. Each of 4 cores runs the kernel on its own 1024-element
tile; the runtime splits and joins the work.

Input covers the kernel's domain contract (see
``aie_kernels/aie2p/exp2f_vec.cc``) in four blocks: a dense grid and a random
sample over [-111, 0]; a block below -111, where the kernel clamps; and a
positive block, half a dense grid over [0, 127] and half explicit values
straddling 128, where 2**x first exceeds FLT_MAX.

The boundary block gates on VALUES -- no negative-signed output anywhere, and
bit-exact +inf at and above k = 128 -- not just on finiteness, because the
exponent-field reconstruction fails by carrying into the sign bit, which
yields a finite, wrong-signed result that isfinite() cannot see.
"""

import sys

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_

_TILE = 1024
_N_CORES = 4
# The kernel's lower clamp: the lowest exponent still holding its 8.9e-5
# relative error. Its hard floor is -126; see the .cc for the measured table.
_MIN_X = -111.0


@iron.jit
def vector_exp2f(
    x: In,
    y: Out,
    *,
    N: CompileTime[int],
):
    n = _TILE
    n_cores = _N_CORES
    # A short N would round `tiles` down to zero, leaving the workers with no
    # iterations while the runtime still waits on the drain: the run hangs
    # rather than failing.
    if N % (n * n_cores) != 0:
        raise ValueError(f"N ({N}) must be a multiple of {n * n_cores}")
    tiles = (N // n) // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[np.float32]]
    memtile_ty = np.ndarray[(n * n_cores,), np.dtype[np.float32]]
    tile_ty = np.ndarray[(n,), np.dtype[np.float32]]

    exp2f_fn = kernels.exp2f_vec(tile_size=n, min_x=_MIN_X)

    A_fifo = ObjectFifo(memtile_ty, name="inA")
    C_fifo = ObjectFifo(memtile_ty, name="outC")
    a_fifos = A_fifo.cons().split(
        offsets=[n * i for i in range(n_cores)], obj_types=[tile_ty] * n_cores
    )
    c_fifos = C_fifo.prod().join(
        offsets=[n * i for i in range(n_cores)], obj_types=[tile_ty] * n_cores
    )

    def core_fn(a_in, c_out, exp2f_fn):
        for _ in range_(tiles):
            elem_out = c_out.acquire(1)
            elem_in_a = a_in.acquire(1)
            exp2f_fn(elem_in_a, elem_out, n)
            a_in.release(1)
            c_out.release(1)

    workers = [
        Worker(core_fn, fn_args=[a_fifos[i].cons(), c_fifos[i].prod(), exp2f_fn])
        for i in range(n_cores)
    ]

    def sequence(a_in, c_out, in_h, out_h):
        in_h.fill(a_in)
        out_h.drain(c_out, wait=True)

    rt = Runtime(
        sequence,
        [tensor_ty, tensor_ty, A_fifo.prod(), C_fifo.cons()],
    )

    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _rel_l2(actual, ref):
    diff = actual.astype(np.float64) - ref.astype(np.float64)
    denom = np.linalg.norm(ref.astype(np.float64))
    return (
        float(np.linalg.norm(diff) / denom)
        if denom > 0
        else float(np.linalg.norm(diff))
    )


def main():
    rng = np.random.default_rng(0)
    n_grid, n_rand, n_clamp, n_pos = 6144, 1024, 1024, 4096
    N = n_grid + n_rand + n_clamp + n_pos

    grid = np.linspace(_MIN_X, 0.0, n_grid, dtype=np.float64)
    rand = rng.uniform(_MIN_X, 0.0, n_rand)
    clamp = np.linspace(-500.0, _MIN_X - 1e-4, n_clamp, dtype=np.float64)

    # Half a dense [0, 127] grid, half explicit values straddling the upper
    # clamp: k = 127 is the last exponent the field holds, k = 128 is the
    # all-ones/+inf pattern, k >= 129 carries into the sign bit. The explicit
    # list is tiled to fill the block so a single-lane bug shows up too.
    pos_grid = np.linspace(0.0, 127.0, n_pos // 2, dtype=np.float64)
    boundary_values = np.array(
        [
            127.0,
            127.5,
            127.9,
            127.99,
            128.0,
            128.5,
            129.0,
            129.5,
            130.0,
            140.0,
            150.0,
            180.0,
            200.0,
            255.0,
            256.0,
            257.0,
            300.0,
            500.0,
            1000.0,
            1e6,
            1e30,
        ],
        dtype=np.float64,
    )
    reps = -(-(n_pos // 2) // len(boundary_values))  # ceil div
    pos_boundary = np.tile(boundary_values, reps)[: n_pos // 2]
    pos = np.concatenate([pos_grid, pos_boundary])

    a_np = np.concatenate([grid, rand, clamp, pos]).astype(np.float32)
    domain_end = n_grid + n_rand
    clamp_end = domain_end + n_clamp
    pos_grid_end = clamp_end + n_pos // 2

    a = iron.tensor(a_np, dtype=np.float32, device="npu")
    c = iron.zeros(N, dtype=np.float32, device="npu")

    vector_exp2f(a, c, N=N)

    out = c.numpy()

    ok = True

    # Relative, not absolute: 2**x spans 1.2e-38 to 1.7e38 over the domain, so
    # an absolute gate is vacuous at one end and unreachable at the other.
    dom_out = out[:domain_end]
    dom_ref = kernels.exp2f_vec_ref(a_np[:domain_end])
    dom_out64 = dom_out.astype(np.float64)
    dom_ref64 = dom_ref.astype(np.float64)
    dom_rel_l2 = _rel_l2(dom_out, dom_ref)
    dom_max_rel = float(np.max(np.abs(dom_out64 - dom_ref64) / np.abs(dom_ref64)))
    dom_n_nonfinite = int(np.sum(~np.isfinite(dom_out)))
    print(
        f"[{_MIN_X:g}, 0] domain (N={domain_end}): max_rel_err={dom_max_rel:.6g} "
        f"rel_l2={dom_rel_l2:.6g} non_finite={dom_n_nonfinite}"
    )
    if dom_n_nonfinite or dom_max_rel > 5e-4:
        print(
            f"FAIL: [{_MIN_X:g}, 0] domain outside gate (max_rel_err > 5e-4 or "
            "non-finite present)"
        )
        ok = False

    # Below the clamp the contract is 2**max(x, _MIN_X), not 2**x. The hard
    # requirement is no NaN/Inf; the match to the clamped reference is reported
    # rather than gated.
    clamp_out = out[domain_end:clamp_end]
    clamp_ref = kernels.exp2f_vec_ref(
        np.maximum(a_np[domain_end:clamp_end], np.float32(_MIN_X))
    )
    clamp_max_abs = float(
        np.max(np.abs(clamp_out.astype(np.float64) - clamp_ref.astype(np.float64)))
    )
    clamp_rel_l2 = _rel_l2(clamp_out, clamp_ref)
    clamp_n_nonfinite = int(np.sum(~np.isfinite(clamp_out)))
    print(
        f"< {_MIN_X:g} clamp block (N={n_clamp}): max_abs_err={clamp_max_abs:.6g} "
        f"rel_l2={clamp_rel_l2:.6g} non_finite={clamp_n_nonfinite}"
    )
    if clamp_n_nonfinite:
        print(f"FAIL: clamp block produced {clamp_n_nonfinite} non-finite outputs")
        ok = False

    posg_out = out[clamp_end:pos_grid_end]
    posg_ref = kernels.exp2f_vec_ref(a_np[clamp_end:pos_grid_end])
    posg_out64 = posg_out.astype(np.float64)
    posg_ref64 = posg_ref.astype(np.float64)
    posg_max_rel = float(np.max(np.abs(posg_out64 - posg_ref64) / np.abs(posg_ref64)))
    posg_rel_l2 = _rel_l2(posg_out, posg_ref)
    posg_n_nonfinite = int(np.sum(~np.isfinite(posg_out)))
    print(
        f"[0, 127] positive domain (N={n_pos // 2}): max_rel_err={posg_max_rel:.6g} "
        f"rel_l2={posg_rel_l2:.6g} non_finite={posg_n_nonfinite}"
    )
    if posg_n_nonfinite or posg_max_rel > 5e-4:
        print(
            "FAIL: [0, 127] domain outside gate (max_rel_err > 5e-4 or non-finite present)"
        )
        ok = False

    # Every 2**x at k >= 128 exceeds FLT_MAX (2^128 = 3.402823669e38 >
    # FLT_MAX = 3.402823466e38), so the contract there is bit-for-bit equality
    # to +inf, not a tolerance. np.signbit() rather than `< 0`, to catch -0.0,
    # which compares equal to 0.0.
    posb_out = out[pos_grid_end:]
    posb_x = a_np[pos_grid_end:]
    below_boundary = posb_x < 128.0
    at_or_above = ~below_boundary

    posb_n_nonfinite_nan = int(np.sum(np.isnan(posb_out)))
    posb_n_signbit = int(np.sum(np.signbit(posb_out)))
    print(
        f"upper-clamp boundary block (N={n_pos // 2}): "
        f"nan={posb_n_nonfinite_nan} negative_or_neg_zero={posb_n_signbit}"
    )
    if posb_n_nonfinite_nan or posb_n_signbit:
        print(
            "FAIL: boundary block produced NaN or a negative-signed output "
            "(2**x is never negative for finite real x)"
        )
        ok = False

    if np.any(below_boundary):
        b_out = posb_out[below_boundary].astype(np.float64)
        b_ref = kernels.exp2f_vec_ref(posb_x[below_boundary]).astype(np.float64)
        b_max_rel = float(np.max(np.abs(b_out - b_ref) / np.abs(b_ref)))
        print(
            f"  k < 128 sub-block (N={int(np.sum(below_boundary))}): max_rel_err={b_max_rel:.6g}"
        )
        if b_max_rel > 5e-4:
            print("FAIL: k < 128 sub-block outside the rel-err gate")
            ok = False

    if np.any(at_or_above):
        a_out = posb_out[at_or_above]
        n_not_pos_inf = int(np.sum(a_out != np.float32(np.inf)))
        print(
            f"  k >= 128 sub-block (N={int(np.sum(at_or_above))}): not_exactly_+inf={n_not_pos_inf}"
        )
        if n_not_pos_inf:
            print(
                "FAIL: k >= 128 sub-block: every 2**x here exceeds FLT_MAX, "
                "so the contract is EXACT +inf, not approximately"
            )
            ok = False

    if not ok:
        sys.exit(1)
    print(
        f"PASS! ({N} samples: {domain_end} in [{_MIN_X:g}, 0], {n_clamp} below "
        f"the clamp, {n_pos // 2} in [0, 127], {n_pos // 2} on the upper "
        "boundary)"
    )


if __name__ == "__main__":
    main()
