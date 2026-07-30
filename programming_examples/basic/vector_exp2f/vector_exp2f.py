# vector_exp2f/vector_exp2f.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Vector 2**x (software minimax poly), IRON + ``@iron.jit``, 4 cores, float32.

Demonstrates the IRON kernel library's software-poly ``2**x`` kernel
(``kernels.exp2f_vec``, ``aie_kernels/aie2p/exp2f_vec.cc``), the accuracy
alternative to the LUT-based ``kernels.bf16_exp`` that
``basic/vector_exp`` demonstrates. aie2p only (the kernel has not been
ported to aie2). Each of 4 cores runs the kernel on its own 1024-element
tile; the runtime splits/joins the work across the cores.

Input is built in four blocks over the kernel's documented domain contract
(``aie_kernels/aie2p/exp2f_vec.cc``'s header comment):

  1. a dense grid over [-100, 0], the kernel's characterized softmax
     domain (includes both endpoints, so the LUT's worst point -100 and
     the exact-1.0 point 0 are both exercised);
  2. a random sample over the same domain (covers non-grid points a
     linspace alone would miss);
  3. a block below -100 (down to -500), which exercises the kernel's
     lower clamp (``x = max(x, -100)`` before the exponent-field
     reconstruction, documented in the .cc file). This only has to stay
     finite and close to the clamped reference, not match unclamped
     ``2**x`` (which underflows float32 well before -500 anyhow);
  4. a positive-domain block that probes the exponent-field reconstruction
     the other way: a dense grid over [0, 127] (the poly's own accuracy is
     flat there, same construction as [-100, 0]) directly abutted by the
     upper-clamp boundary itself, with explicit values at k just below
     (127), at (128), and increasingly above (129, 130, 140, 150, 180, 200,
     255, 256, 257, 300, 500, 1000, 1e6, 1e30) the point where an unclamped
     ``k + 127`` overflows the f32 exponent field's 8 bits and wraps into
     the sign bit. This is the block that catches the defect this file's
     history fixed: an earlier revision of this kernel clamped only the
     lower bound, so k >= 129 silently returned wrong-SIGNED finite
     values (e.g. k=129 -> -0.0, k=150 -> -1.23e-32, k=257 -> -2.0) rather
     than the +inf a correct ``2**x`` gives once the true value exceeds
     FLT_MAX (already true by k=128). A non-finite check alone cannot see
     this: every one of those wrong values IS finite. This block asserts
     on VALUES (sign, and exact identity with +inf where the true 2**x
     exceeds FLT_MAX), not just finiteness. It also catches a SECOND,
     hardware-only bug the first fix attempt tripped over: on this target,
     ``aie::mul`` returns NaN (not +inf) whenever its true product would
     overflow f32, so a fix that clamps to 128.0 and lets the
     reconstruction "saturate for free" reproduces the same class of
     silent, isfinite()-invisible failure with a different symptom. See
     ``aie_kernels/aie2p/exp2f_vec.cc``'s header ("BUG 2") for the device
     measurements and why the actual fix is a compare against the
     ORIGINAL x plus an explicit select, not a wider clamp.

Verification reports max-abs-error and rel-L2 against a float64 ``2**x``
reference (``kernels.exp2f_vec_ref``), separately per regime, plus:
  - a hard NaN/Inf gate on the clamp block (this kernel family's known
    failure mode on this target is silent inf/nan, not just imprecision:
    see ``aie_kernels/aie2p/dwconv1d.cc``'s ``aie::sliding_mul_ops`` note
    for the sibling case this doctrine comes from);
  - a hard sign gate plus an exact +inf identity check on the positive
    boundary block (see point 4 above). This is the check a pre-fix
    kernel fails and a post-fix kernel passes, which a plain isfinite()
    gate would silently let through (-0.0 and -1.23e-32 are both finite).
"""

import sys

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_

_TILE = 1024  # hard-coded by kernels.exp2f_vec's underlying C++ kernel
_N_CORES = 4


@iron.jit
def vector_exp2f(
    x: In,
    y: Out,
    *,
    N: CompileTime[int],
):
    n = _TILE
    n_cores = _N_CORES
    tiles = (N // n) // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[np.float32]]
    memtile_ty = np.ndarray[(n * n_cores,), np.dtype[np.float32]]
    tile_ty = np.ndarray[(n,), np.dtype[np.float32]]

    exp2f_fn = kernels.exp2f_vec(tile_size=n)

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
    return float(np.linalg.norm(diff) / denom) if denom > 0 else float(np.linalg.norm(diff))


def main():
    rng = np.random.default_rng(0)
    n_grid, n_rand, n_clamp, n_pos = 6144, 1024, 1024, 4096
    N = n_grid + n_rand + n_clamp + n_pos

    grid = np.linspace(-100.0, 0.0, n_grid, dtype=np.float64)
    rand = rng.uniform(-100.0, 0.0, n_rand)
    clamp = np.linspace(-500.0, -100.0001, n_clamp, dtype=np.float64)  # strictly below -100

    # Positive-domain block: half a dense [0, 127] grid (poly accuracy, same
    # construction as [-100, 0]), half explicit values straddling the
    # upper-clamp boundary (k = 127 is the last exact integer below it; k =
    # 128 is where ki+127 first hits the exponent-all-ones/+inf pattern; k >=
    # 129 is where an UNCLAMPED reconstruction wraps into the sign bit, see
    # the .cc header). The explicit list is tiled to fill n_pos // 2 so it is
    # dense enough to catch a single-lane vector bug, not just a scalar one.
    pos_grid = np.linspace(0.0, 127.0, n_pos // 2, dtype=np.float64)
    boundary_values = np.array(
        [127.0, 127.5, 127.9, 127.99, 128.0, 128.5, 129.0, 129.5, 130.0,
         140.0, 150.0, 180.0, 200.0, 255.0, 256.0, 257.0, 300.0, 500.0,
         1000.0, 1e6, 1e30],
        dtype=np.float64,
    )
    reps = -(-(n_pos // 2) // len(boundary_values))  # ceil div
    pos_boundary = np.tile(boundary_values, reps)[: n_pos // 2]
    pos = np.concatenate([pos_grid, pos_boundary])

    a_np = np.concatenate([grid, rand, clamp, pos]).astype(np.float32)
    domain_end = n_grid + n_rand  # [0, domain_end) is the characterized [-100, 0] domain
    clamp_end = domain_end + n_clamp
    pos_grid_end = clamp_end + n_pos // 2  # [clamp_end, pos_grid_end) is the [0,127] exact grid

    a = iron.tensor(a_np, dtype=np.float32, device="npu")
    c = iron.zeros(N, dtype=np.float32, device="npu")

    vector_exp2f(a, c, N=N)

    out = c.numpy()

    ok = True

    # Characterized domain [-100, 0]: tight tolerance, matches the kernel's
    # documented ~8.5e-5 max relative error (exp2f_vec_ref is exact float64
    # 2**x cast to f32).
    dom_out = out[:domain_end]
    dom_ref = kernels.exp2f_vec_ref(a_np[:domain_end])
    dom_out64 = dom_out.astype(np.float64)
    dom_ref64 = dom_ref.astype(np.float64)
    dom_max_abs = float(np.max(np.abs(dom_out64 - dom_ref64)))
    dom_rel_l2 = _rel_l2(dom_out, dom_ref)
    dom_max_rel = float(np.max(np.abs(dom_out64 - dom_ref64) / np.abs(dom_ref64)))
    dom_n_nonfinite = int(np.sum(~np.isfinite(dom_out)))
    print(
        f"[-100, 0] domain (N={domain_end}): max_abs_err={dom_max_abs:.6g} "
        f"max_rel_err={dom_max_rel:.6g} rel_l2={dom_rel_l2:.6g} non_finite={dom_n_nonfinite}"
    )
    if dom_n_nonfinite or dom_max_abs > 5e-4:
        print("FAIL: [-100, 0] domain outside gate (max_abs_err > 5e-4 or non-finite present)")
        ok = False

    # Clamp block (x < -100): reference is the KERNEL'S documented contract,
    # 2**max(x, -100), not raw 2**x (which is a different, unclamped
    # function below -100). The hard requirement is no NaN/Inf; the
    # numerical match to the clamped reference is reported but with a
    # looser gate since this is not the kernel's characterized domain.
    clamp_out = out[domain_end:clamp_end]
    clamp_ref = kernels.exp2f_vec_ref(np.maximum(a_np[domain_end:clamp_end], np.float32(-100.0)))
    clamp_max_abs = float(np.max(np.abs(clamp_out.astype(np.float64) - clamp_ref.astype(np.float64))))
    clamp_rel_l2 = _rel_l2(clamp_out, clamp_ref)
    clamp_n_nonfinite = int(np.sum(~np.isfinite(clamp_out)))
    print(
        f"< -100 clamp block (N={n_clamp}): max_abs_err={clamp_max_abs:.6g} "
        f"rel_l2={clamp_rel_l2:.6g} non_finite={clamp_n_nonfinite}"
    )
    if clamp_n_nonfinite:
        print(f"FAIL: clamp block produced {clamp_n_nonfinite} non-finite outputs")
        ok = False

    # Positive [0, 127] grid: same construction as [-100, 0], reported with
    # a RELATIVE (not absolute) gate since these values span up to ~1.7e38
    # and an absolute-error gate would be meaningless at that magnitude.
    # This confirms the poly's ~8.5e-5 accuracy is flat right up to the
    # upper-clamp boundary, not just in the kernel's originally-documented
    # softmax domain.
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
        print("FAIL: [0, 127] domain outside gate (max_rel_err > 5e-4 or non-finite present)")
        ok = False

    # Upper-clamp boundary block: THE regression test for the sign-wrap
    # defect. k = 127 must still be an accurate finite value (rel-err gate,
    # same as above); every k >= 128 has a true 2**x that already exceeds
    # FLT_MAX (2^128 = 3.402823669e38 > FLT_MAX = 3.402823466e38), so the
    # kernel's contract is EXACT +inf there, not merely "close to", so assert
    # bit-for-bit equality to +inf, not a tolerance. A pre-fix kernel (lower
    # clamp only) fails this block outright: k=129 reads back as -0.0,
    # k=150 as -1.23e-32, k=257 as -2.0, all finite, all wrong-signed, all
    # invisible to an isfinite() check. np.signbit() catches -0.0 too, which
    # a plain `< 0` comparison would not (IEEE -0.0 compares equal to 0.0).
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
        print("FAIL: boundary block produced NaN or a negative-signed output "
              "(2**x is never negative for finite real x)")
        ok = False

    if np.any(below_boundary):
        b_out = posb_out[below_boundary].astype(np.float64)
        b_ref = kernels.exp2f_vec_ref(posb_x[below_boundary]).astype(np.float64)
        b_max_rel = float(np.max(np.abs(b_out - b_ref) / np.abs(b_ref)))
        print(f"  k < 128 sub-block (N={int(np.sum(below_boundary))}): max_rel_err={b_max_rel:.6g}")
        if b_max_rel > 5e-4:
            print("FAIL: k < 128 sub-block outside the rel-err gate")
            ok = False

    if np.any(at_or_above):
        a_out = posb_out[at_or_above]
        n_not_pos_inf = int(np.sum(a_out != np.float32(np.inf)))
        print(f"  k >= 128 sub-block (N={int(np.sum(at_or_above))}): not_exactly_+inf={n_not_pos_inf}")
        if n_not_pos_inf:
            print("FAIL: k >= 128 sub-block: every 2**x here exceeds FLT_MAX, "
                  "so the contract is EXACT +inf, not approximately")
            ok = False

    if not ok:
        sys.exit(1)
    print(f"PASS! ({N} samples: {domain_end} in the characterized [-100,0] domain, "
          f"{n_clamp} exercising the < -100 clamp path, {n_pos // 2} in the exact "
          f"[0,127] extension, {n_pos // 2} on the upper-clamp boundary)")


if __name__ == "__main__":
    main()
