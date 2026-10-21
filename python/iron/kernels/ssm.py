# kernels/ssm.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""State space model kernels: Mamba2's selective scan and Gated DeltaNet's delta rule."""

from functools import partial

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.accuracy import round_to
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    _STRICT_FP_ROLLED,
    KernelContract,
    Param,
    Trace,
    _kernel_source,
    _make_extern,
)
from .sample import _positive


def selective_scan_sample(
    rng, calls: int, *, rows: int, head_dim: int, state: int, blocks: int
):
    """Return heads of ``blocks`` blocks, each from a random state.

    A < 0, so the state decays.
    """
    words = -(-rows * (head_dim + 2 * state + 1) // 2)
    block = rng.standard_normal((calls, 2 * words)) * 0.5
    heads = [
        np.concatenate(
            [
                [rng.uniform(-4, 0)],  # dt_bias
                [-rng.uniform(0.5, 16)],  # A = -exp(A_log)
                [rng.uniform(0, 2)],  # D
                rng.standard_normal(head_dim * state),
            ]
        )
        for _ in range(-(-calls // blocks))
    ]
    head = np.repeat(np.stack(heads), blocks, axis=0)[:calls]
    return [block.astype(bfloat16), head.astype(np.float32)]


def selective_scan_ref(
    block, head, valid, *, rows: int, head_dim: int, state: int, blocks: int
):
    """Numpy reference for [`selective_scan`][iron.kernels.ssm.selective_scan].

    Call ``b`` is block ``b``. Returns ``(final, y)`` per call.
    """
    head = np.asarray(head, np.float32).reshape(-1, 3 + head_dim * state)
    calls = len(head)
    block = np.asarray(block, np.float32).reshape(calls, -1)
    x = block[:, : rows * head_dim].reshape(calls, rows, head_dim)
    end = rows * (head_dim + 2 * state)
    bc = block[:, rows * head_dim : end].reshape(calls, rows, 2, state)
    dt = block[:, end : end + rows]
    final = np.zeros((calls, head_dim * state), np.float32)
    y = np.zeros((calls, rows, head_dim), np.float32)
    s = np.zeros((head_dim, state), np.float32)
    for c in range(calls):
        if c % blocks == 0:
            s = head[c, 3:].reshape(head_dim, state).copy()
        dt_bias, A, D = head[c, :3]
        for t in range(min(int(valid), rows)):
            B, C = bc[c, t]
            step = np.logaddexp(np.float32(0), dt[c, t] + dt_bias)
            s = np.exp(step * A) * s + (step * x[c, t])[:, None] * B
            y[c, t] = s @ C + D * x[c, t]
        final[c] = s.reshape(-1)
    return final, y.astype(bfloat16).reshape(calls, -1)


def _selective_scan_bound(
    block, head, valid, *, rows: int, head_dim: int, state: int, blocks: int
):
    """Bound on selective_scan's error against its float32 reference.

    The recurrence runs in float64 for the exact state and y, and alongside
    them each state element's magnitude ``m`` (the recurrence on absolute
    values) and ``e``, how far a float32 pipeline's state may sit from the
    exact one. Each row adds to ``e``, in units of 2^-24 relative: 8 of the
    decayed state (exp_f32, the limb product) plus 24 for each unit of
    ``|step * A|`` (softplus's and the product's error, through the
    exponent), 24 of the input term (softplus, two limb products) and 1 of
    the sum. ``final`` may be ``2 e`` off: the kernel's error and the
    reference's. y's float32 sum may be off by ``gap``: ``e @ |C|`` plus
    ``state + 2`` units of ``m @ |C| + |D x|``, the products and a sum in
    any order. y rounds once to bf16, so the kernel's and the reference's
    agree unless the exact y lies within ``gap`` of a rounding midpoint;
    there they may be the two bf16 around it.
    """
    u = 2.0**-24
    head = np.asarray(head, np.float64).reshape(-1, 3 + head_dim * state)
    calls = len(head)
    block = np.asarray(block, np.float64).reshape(calls, -1)
    x = block[:, : rows * head_dim].reshape(calls, rows, head_dim)
    end = rows * (head_dim + 2 * state)
    bc = block[:, rows * head_dim : end].reshape(calls, rows, 2, state)
    dt = block[:, end : end + rows]
    final = np.zeros((calls, head_dim * state))
    y = np.zeros((calls, rows, head_dim))
    gap = np.zeros_like(y)
    for c in range(calls):
        if c % blocks == 0:
            s = head[c, 3:].reshape(head_dim, state).copy()
            m, e = np.abs(s), np.zeros_like(s)
        dt_bias, A, D = head[c, :3]
        for t in range(min(int(valid), rows)):
            B, C = bc[c, t]
            step = np.logaddexp(0.0, dt[c, t] + dt_bias)
            decay = np.exp(step * A)
            gain = np.outer(step * x[c, t], B)
            kept = decay * m
            s = decay * s + gain
            m = kept + np.abs(gain)
            e = decay * e + u * (
                (8 + 24 * abs(step * A)) * kept + 24 * np.abs(gain) + m
            )
            dx = D * x[c, t]
            y[c, t] = s @ C + dx
            gap[c, t] = e @ np.abs(C) + (state + 2) * u * (m @ np.abs(C) + np.abs(dx))
        final[c] = 2 * e.reshape(-1)
    y_bound = round_to(y + gap, bfloat16) - round_to(y - gap, bfloat16)
    return final, y_bound.reshape(calls, -1)


def _state_bound(reference, *args, final_frac, y_ulps, **geometry):
    """Bound ``final`` by ``final_frac`` of its range, ``y`` by ``y_ulps`` bf16 ulps."""
    final, y = reference(*args, **geometry)
    span = np.abs(final).max(axis=1, keepdims=True)
    y = np.abs(y.astype(np.float32)).reshape(len(y), geometry["rows"], -1)
    top = y.max(axis=2, keepdims=True)
    _, k = np.frexp(top)
    ulp = np.where(top > 0, np.ldexp(np.float32(y_ulps), k - 8), 0)
    ulp = np.broadcast_to(ulp, y.shape).reshape(len(y), -1)
    return np.broadcast_to(final_frac * span, final.shape), ulp.astype(np.float32)


def selective_scan(
    *, rows: int = 16, head_dim: int = 16, state: int = 32, blocks: int = 1
) -> ExternalFunction:
    """Mamba2's state space recurrence for one head, a block of rows a call.

    ``selective_scan(block, head, final, y, valid, b)``: ``block`` is bf16
    x (``rows`` x ``head_dim``), then bc (``rows`` rows of the group's
    ``state`` B then ``state`` C), then dt (``rows``), then a bf16 of
    padding where those are odd, so a DMA moves whole words. ``head`` is float32
    ``(dt_bias, A, D)`` then the state before the head's first block,
    ``(head_dim, state)``. ``b`` counts blocks from 0, ``blocks`` to a head,
    and block ``b`` loads the state from ``head`` when it is a head's first.
    Each block runs its first ``valid`` rows:
    ``step = softplus(dt + dt_bias)``,
    ``state = exp(step * A) * state + step * outer(x, B)`` and
    ``y = state @ C + D * x``, rounded
    once to bf16; rows from ``valid`` on leave the state alone and write
    zero ``y``. The state stays in the core between blocks, and ``final``
    is it after this one. The caller gathers head ``h``'s x and dt and
    group ``h // (heads // groups)``'s B and C into ``block``, so each
    group's B and C stream ``heads // groups`` times.

    Args:
        rows: Rows of a block.
        head_dim: The head's x width, or a slice of it: the rows of state
            are independent. A multiple of 16.
        state: The state size, a multiple of 16.
        blocks: Blocks to a head.

    Returns:
        The kernel.

    Raises:
        ValueError: A dimension is not a positive integer, or ``head_dim`` or
            ``state`` is not a multiple of 16, the kernel's vector width.
    """
    geometry = dict(rows=rows, head_dim=head_dim, state=state, blocks=blocks)
    _positive("selective_scan", **geometry)
    if head_dim % 16 or state % 16:
        raise ValueError(
            f"selective_scan: head_dim ({head_dim}) and state ({state}) must be "
            "multiples of 16"
        )
    # One bf16 of padding after an odd block, so a DMA moves whole words.
    words = -(-rows * (head_dim + 2 * state + 1) // 2)
    return _make_extern(
        "selective_scan",
        _kernel_source("ssm/selective_scan.cc"),
        [
            np.ndarray[(2 * words,), np.dtype[bfloat16]],
            np.ndarray[(3 + head_dim * state,), np.dtype[np.float32]],
            np.ndarray[(head_dim * state,), np.dtype[np.float32]],
            np.ndarray[(rows * head_dim,), np.dtype[bfloat16]],
            np.int32,
            np.int32,
        ],
        compile_flags=_STRICT_FP_ROLLED
        + [
            # Scheduling the row update past the default pressure limits
            # measured 8% fewer cycles at IRON's prefill slice.
            "-mllvm",
            "--aie-premisched-reg-pressure-instr-threshold=10000",
            "-mllvm",
            "--aie-premisched-finer-rp-tracking=false",
            f"-DSCAN_ROWS={rows}",
            f"-DSCAN_HEAD_DIM={head_dim}",
            f"-DSCAN_STATE={state}",
            f"-DSCAN_BLOCKS={blocks}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out, Out, Param, Param),
            parameter_bindings=((5, 0),),
            call_index=5,
            reference=partial(selective_scan_ref, **geometry),
            sample=partial(selective_scan_sample, **geometry),
            tolerance=Tolerance.bounded(
                partial(_selective_scan_bound, **geometry),
                note="derived: y exact but at a near tie, final within the "
                "float32 recurrence's error bound; npu2: within it on every "
                "case over three seeds; npu1: unmeasured",
            ),
        ),
    )


def gated_delta_rule_sample(
    rng, calls: int, *, rows: int, key_dim: int, value_dim: int, blocks: int
):
    """Return heads of ``blocks`` blocks, A < 0 so the state decays, each from a random state."""
    block = rng.standard_normal((calls, rows * (2 * key_dim + value_dim + 2)))
    # Row 1's q and k are tiny, so their norms depend on eps.
    for m in (0, rows):
        block[:, (m + 1) * key_dim : (m + 2) * key_dim] *= 1e-4
    heads = [
        np.concatenate(
            [
                [rng.uniform(-4, 0)],  # dt_bias
                [-rng.uniform(0.5, 16)],  # A = -exp(A_log)
                rng.standard_normal(key_dim * value_dim) * 0.5,
            ]
        )
        for _ in range(-(-calls // blocks))
    ]
    head = np.repeat(np.stack(heads), blocks, axis=0)[:calls]
    return [block.astype(bfloat16), head.astype(np.float32)]


def gated_delta_rule_ref(
    block, head, valid, *, rows: int, key_dim: int, value_dim: int, blocks: int
):
    """Numpy reference for [`gated_delta_rule`][iron.kernels.ssm.gated_delta_rule]: call ``b`` is block ``b``.

    Returns ``(final, y)`` per call.
    """
    head = np.asarray(head, np.float32).reshape(-1, 2 + key_dim * value_dim)
    calls = len(head)
    block = np.asarray(block, np.float32).reshape(calls, -1)
    q, k, v, ab = np.split(
        block, np.cumsum([rows * key_dim, rows * key_dim, rows * value_dim]), axis=1
    )
    q, k = q.reshape(calls, rows, key_dim), k.reshape(calls, rows, key_dim)
    v, a, b = v.reshape(calls, rows, value_dim), ab[:, :rows], ab[:, rows:]
    final = np.zeros((calls, key_dim * value_dim), np.float32)
    y = np.zeros((calls, rows, value_dim), np.float32)
    s = np.zeros((key_dim, value_dim), np.float32)
    for c in range(calls):
        if c % blocks == 0:
            s = head[c, 2:].reshape(key_dim, value_dim).copy()
        dt_bias, A = head[c, :2]
        for t in range(min(int(valid), rows)):
            qt = q[c, t] / np.sqrt((q[c, t] ** 2).sum() + np.float32(1e-6))
            qt = qt / np.sqrt(np.float32(key_dim))
            kt = k[c, t] / np.sqrt((k[c, t] ** 2).sum() + np.float32(1e-6))
            g = A * np.logaddexp(np.float32(0), a[c, t] + dt_bias)
            beta = 1 / (1 + np.exp(-b[c, t]))
            s = np.exp(g) * s
            s = s + np.outer(kt, beta * (v[c, t] - kt @ s))
            y[c, t] = qt @ s
        final[c] = s.reshape(-1)
    return final, y.astype(bfloat16).reshape(calls, -1)


def gated_delta_rule(
    *, rows: int = 16, key_dim: int = 32, value_dim: int = 16, blocks: int = 1
) -> ExternalFunction:
    """Gated DeltaNet's recurrence for one value head, a block of rows a call.

    ``gated_delta_rule(block, head, final, y, valid, b)``: ``block`` is bf16
    q and k (``rows`` x ``key_dim`` each), v (``rows`` x ``value_dim``),
    then a and b (``rows`` each). ``head`` is float32 ``(dt_bias, A)`` then
    the state before the head's first block, ``(key_dim, value_dim)``.
    ``b`` counts blocks from 0, ``blocks`` to a head, and a head's first
    block loads the state from ``head``. Each block runs its first ``valid``
    rows: q and k are l2-normalized (eps 1e-6) and q scaled by ``1 /
    sqrt(key_dim)``, ``g = A * softplus(a + dt_bias)``, ``beta =
    sigmoid(b)``, ``state = exp(g) * state``, then ``state += outer(k, beta
    * (v - k @ state))`` and ``y = q @ state``, rounded once to bf16. Rows
    from ``valid`` on leave the state alone and write zero ``y``. The state
    stays in the core between blocks, and ``final`` is it after this one.
    The caller gathers key head ``h // (v_heads // k_heads)``'s q and k and
    value head ``h``'s v, a and b into ``block``.

    Args:
        rows: Rows of a block.
        key_dim: The q and k width.
        value_dim: The v width, or a slice of it: the state's columns are
            independent.
        blocks: Blocks to a head.

    Returns:
        The kernel.
    """
    geometry = dict(rows=rows, key_dim=key_dim, value_dim=value_dim, blocks=blocks)
    _positive("gated_delta_rule", **geometry)
    return _make_extern(
        "gated_delta_rule",
        _kernel_source("ssm/gated_delta_rule.cc"),
        [
            np.ndarray[(rows * (2 * key_dim + value_dim + 2),), np.dtype[bfloat16]],
            np.ndarray[(2 + key_dim * value_dim,), np.dtype[np.float32]],
            np.ndarray[(key_dim * value_dim,), np.dtype[np.float32]],
            np.ndarray[(rows * value_dim,), np.dtype[bfloat16]],
            np.int32,
            np.int32,
        ],
        compile_flags=_STRICT_FP_ROLLED
        + [
            f"-DGDR_ROWS={rows}",
            f"-DGDR_KEY_DIM={key_dim}",
            f"-DGDR_VALUE_DIM={value_dim}",
            f"-DGDR_BLOCKS={blocks}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out, Out, Param, Param),
            parameter_bindings=((5, 0),),
            call_index=5,
            reference=partial(gated_delta_rule_ref, **geometry),
            sample=partial(gated_delta_rule_sample, **geometry),
            tolerance=Tolerance.bounded(
                partial(
                    _state_bound,
                    gated_delta_rule_ref,
                    final_frac=1e-6,
                    y_ulps=1,
                    **geometry,
                ),
                note="npu2, against the float32 reference over five seeds: final "
                "within 9.4e-7 of its range at key_dim 2, 5.1e-7 at the default; y "
                "one bf16 ulp off in 1 of 10240, as the scalar kernel's",
            ),
        ),
    )
