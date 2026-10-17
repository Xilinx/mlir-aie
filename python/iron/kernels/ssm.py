# kernels/ssm.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Mamba-2 selective state space kernel factories and NumPy references.

One source, ``aie_kernels/ssm/ssm.cc``, holds the entry points, each for a
block of 32 positions of a head, or for one unit of its state (the head's
state, or a slice of its head-dim columns), stored as ``(state, head_dim)``
rows of float32.

A prompt chunk's chunked scan (SSD), a block at a time:

* [`ssd_cb`][iron.kernels.ssm.ssd_cb]: the block's scores ``G = C B^T``,
  shared by every head of a group.
* [`ssd_decay`][iron.kernels.ssm.ssd_decay]: a head's decays: ``m``, the
  scores decay-masked and weighted by ``dt``, and ``e``, each position's
  decay from the block's start and its weight in the state the block leaves.
* [`ssd_out`][iron.kernels.ssm.ssd_out]: a unit's output, within the block
  and from the state before it.
* [`ssd_state`][iron.kernels.ssm.ssd_state]: the state the block leaves.

A decode step: [`ssm_state_update`][iron.kernels.ssm.ssm_state_update], one
position of the recurrence.

Every float32 product is taken from bf16 limbs, so the arithmetic is
float32's; each bound below is derived from it, and from the accurate
``exp2f``'s measured 9.2e-6.
"""

from functools import partial

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import KernelContract, Param, Trace, _kernel_source, _make_extern

# A block of the chunked scan: one 32-lane vector of positions.
SSD_BLOCK = 32
# exp2f's relative error, measured 9.2e-6 on aie2p and 7.4e-6 on aie2.
_EXP_REL = 1e-5
_F32 = 2.0**-24
# What a product of three-limb operands keeps, and a two-limb split.
_LIMBS3 = 2.0**-22
_LIMBS2 = 2.0**-16
# dt's softplus (exp2f, then a float32 log), and a = dt A, A = -e^A_log.
_E_DT = _EXP_REL + 4 * _F32
_E_A = _E_DT + _EXP_REL + _LIMBS3


def _source():
    return _kernel_source("ssm/ssm.cc")


def _flags(p: int, n: int, dt_cols: int) -> list[str]:
    return [
        f"-DSSM_Q={SSD_BLOCK}",
        f"-DSSM_P={p}",
        f"-DSSM_N={n}",
        f"-DSSM_DT={dt_cols}",
    ]


def _check(name: str, p: int = 32, n: int = 32, dt_cols: int = 1) -> None:
    if p not in (16, 32):
        raise ValueError(f"{name}: p is one 16- or 32-lane vector, got {p}")
    if n <= 0 or n % 32:
        raise ValueError(f"{name}: n must be a positive multiple of 32, got {n}")
    if dt_cols < 1:
        raise ValueError(f"{name}: dt_cols must be at least 1, got {dt_cols}")


def _bf16(size):
    return np.ndarray[(size,), np.dtype[bfloat16]]


def _f32(size):
    return np.ndarray[(size,), np.dtype[np.float32]]


def ssd_cb(n: int = 128) -> ExternalFunction:
    """Build a block's scores, ``g[l][s] = sum_n c[l][n] b[s][n]``, in float32.

    Takes ``(c, b, g)``: ``c`` and ``b`` ``(32, n)`` bf16 rows, ``g`` ``(32,
    32)`` float32. Each product is exact in the float32 accumulator.

    Args:
        n: The state's width, a multiple of 32.
    """
    _check("ssd_cb", n=n)
    q = SSD_BLOCK
    return _make_extern(
        "ssd_cb_bf16",
        _source(),
        [_bf16(q * n), _bf16(q * n), _f32(q * q)],
        compile_flags=_flags(32, n, 1),
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out),
            reference=lambda c, b: ssd_cb_ref(c, b, n=n),
            acc_dtype=np.float32,
            reduction=n,
            tolerance=Tolerance.bounded(
                lambda c, b: ssd_cb_bound(c, b, n=n),
                note="derived: n exact products summed in float32",
            ),
            ops_per_call=2 * q * q * n,
        ),
    )


def ssd_decay(dt_cols: int = 64) -> ExternalFunction:
    """Build a head's decays over a block of the chunked scan.

    Takes ``(g, dt, a_log, dt_bias, m, e, head, valid)``: the block's scores
    ``g`` ``(32, 32)`` float32 ([`ssd_cb`][iron.kernels.ssm.ssd_cb]); raw
    ``dt`` ``(32, dt_cols)`` bf16 rows; the checkpoint's ``a_log`` and
    ``dt_bias`` ``(dt_cols,)`` bf16, each read at column ``head``; and the
    positions before ``valid`` the call runs (from ``valid`` on, ``dt`` is
    0). Writes ``m`` ``(32, 32)``, ``g[l][s] e^{a_{s+1} + ... + a_l} dt_s``
    for ``s <= l``, and ``e`` ``(2, 32)``: ``e^{a_0 + ... + a_l}`` and
    ``e^{a_{l+1} + ... + a_31} dt_l``, with ``dt_l = softplus(dt[l][head] +
    dt_bias[head])`` and ``a_l = -dt_l e^{a_log[head]}``.

    Args:
        dt_cols: The columns of ``dt`` and of the per-head arrays given.
    """
    _check("ssd_decay", dt_cols=dt_cols)
    q = SSD_BLOCK
    return _make_extern(
        "ssd_decay_bf16",
        _source(),
        [
            _f32(q * q),
            _bf16(q * dt_cols),
            _bf16(dt_cols),
            _bf16(dt_cols),
            _f32(q * q),
            _f32(2 * q),
            np.int32,
            np.int32,
        ],
        compile_flags=_flags(32, 32, dt_cols),
        contract=KernelContract(
            trace=Trace.whole_call(),
            # a_log and dt_bias are a head's, held by the core over its calls.
            roles=(In, In, Param, Param, Out, Out, Param, Param),
            reference=lambda g, dt, a_log, dt_bias, head, valid: _outputs(
                _decay_calls(g, dt, a_log, dt_bias, head, valid, dt_cols), "m", "e"
            ),
            sample=lambda rng, calls: ssd_decay_sample(rng, calls, dt_cols=dt_cols),
            tolerance=Tolerance.bounded(
                lambda g, dt, a_log, dt_bias, head, valid: _outputs(
                    _decay_calls(g, dt, a_log, dt_bias, head, valid, dt_cols),
                    "m_err",
                    "e_err",
                ),
                note="derived from float32 limb products and exp2f's measured "
                "9.2e-6",
            ),
            ops_per_call=q * q * 12,
        ),
    )


def ssd_out(p: int = 32, n: int = 128, dt_cols: int = 64) -> ExternalFunction:
    """Build a unit's output over a block of the chunked scan.

    Takes ``(m, e, c, x, d, state_in, y, head)``: ``m`` and ``e`` from
    [`ssd_decay`][iron.kernels.ssm.ssd_decay]; ``c`` ``(32, n)`` and ``x``
    ``(32, p)`` bf16 rows; the checkpoint's ``d`` ``(dt_cols,)``, read at
    column ``head``; and the state before the block ``state_in`` ``(n, p)``
    float32. Writes ``y`` ``(32, p)``: ``y_l = sum_{s<=l} m[l][s] x_s +
    e[0][l] sum_n c[l][n] state_in[n] + d[head] x_l``.

    Args:
        p: The unit's head-dim columns, 16 or 32.
        n: The state's width, a multiple of 32.
        dt_cols: The columns of ``d`` given.
    """
    _check("ssd_out", p, n, dt_cols)
    q = SSD_BLOCK
    return _make_extern(
        "ssd_out_bf16",
        _source(),
        [
            _f32(q * q),
            _f32(2 * q),
            _bf16(q * n),
            _bf16(q * p),
            _bf16(dt_cols),
            _f32(n * p),
            _bf16(q * p),
            np.int32,
        ],
        compile_flags=_flags(p, n, dt_cols),
        contract=KernelContract(
            trace=Trace.whole_call(),
            # m, e, d and the state are held by the core over a block's calls:
            # the rows stream.
            roles=(Param, Param, In, In, Param, Param, Out, Param),
            reference=partial(ssd_out_ref, p=p, n=n, dt_cols=dt_cols),
            sample=lambda rng, calls: ssd_out_sample(
                rng, calls, p=p, n=n, dt_cols=dt_cols
            ),
            tolerance=Tolerance.bounded(
                partial(ssd_out_bound, p=p, n=n, dt_cols=dt_cols),
                note="derived: two-limb products, float32 sums and one bf16 "
                "rounding",
            ),
            ops_per_call=2 * q * p * (q + n),
        ),
    )


def ssd_state(p: int = 32, n: int = 128) -> ExternalFunction:
    """Build the state a block of the chunked scan leaves, for one unit.

    Takes ``(e, b, x, state_in, state_out)``: ``e`` from
    [`ssd_decay`][iron.kernels.ssm.ssd_decay]; ``b`` ``(32, n)`` and ``x``
    ``(32, p)`` bf16 rows; the state before the block ``state_in`` ``(n,
    p)`` float32. Writes ``state_out = e[0][31] state_in + b^T (e[1] x)``,
    and may be ``state_in`` itself.

    Args:
        p: The unit's head-dim columns, 16 or 32.
        n: The state's width, a multiple of 32.
    """
    _check("ssd_state", p, n)
    q = SSD_BLOCK
    return _make_extern(
        "ssd_state_bf16",
        _source(),
        [_f32(2 * q), _bf16(q * n), _bf16(q * p), _f32(n * p), _f32(n * p)],
        compile_flags=_flags(p, n, 1),
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(Param, In, In, Param, Out),
            reference=partial(ssd_state_ref, p=p, n=n),
            sample=lambda rng, calls: ssd_state_sample(rng, calls, p=p, n=n),
            tolerance=Tolerance.bounded(
                partial(ssd_state_bound, p=p, n=n),
                note="derived: three-limb products summed in float32",
            ),
            ops_per_call=2 * q * p * n,
        ),
    )


def ssm_state_update(p: int = 32, n: int = 128, dt_cols: int = 64) -> ExternalFunction:
    """One position of Mamba-2's recurrence for one unit of the state.

    Takes ``(x, b, c, dt, a_log, d, dt_bias, state_in, state_out, y,
    head)``: ``x`` ``(p,)``, ``b`` and ``c`` ``(n,)`` and raw ``dt``
    ``(dt_cols,)`` bf16; the checkpoint's ``a_log``, ``d`` and ``dt_bias``
    ``(dt_cols,)``, each read at column ``head``; the state ``state_in``
    ``(n, p)`` float32 and ``state_out``, which may be the same buffer; and
    ``y`` ``(p,)`` bf16: ``state_out = e^{dt A} state_in + b (dt x)^T`` and
    ``y = c^T state_out + d[head] x``.

    Args:
        p: The unit's head-dim columns, 16 or 32.
        n: The state's width, a multiple of 32.
        dt_cols: The columns of ``dt`` and of the per-head arrays given.
    """
    _check("ssm_state_update", p, n, dt_cols)
    head = _bf16(dt_cols)
    return _make_extern(
        "ssm_state_update_bf16",
        _source(),
        [
            _bf16(p),
            _bf16(n),
            _bf16(n),
            head,
            head,
            head,
            head,
            _f32(n * p),
            _f32(n * p),
            _bf16(p),
            np.int32,
        ],
        compile_flags=_flags(p, n, dt_cols),
        contract=KernelContract(
            trace=Trace.whole_call(),
            # dt, the per-head arrays and the state are held by the core;
            # x, b and c stream.
            roles=(In, In, In, Param, Param, Param, Param, Param, Out, Out, Param),
            reference=partial(ssm_state_update_ref, p=p, n=n, dt_cols=dt_cols),
            sample=lambda rng, calls: ssm_state_update_sample(
                rng, calls, p=p, n=n, dt_cols=dt_cols
            ),
            tolerance=Tolerance.bounded(
                partial(ssm_state_update_bound, p=p, n=n, dt_cols=dt_cols),
                note="derived from float32 limb products, exp2f's measured "
                "9.2e-6 and one bf16 rounding of y",
            ),
            ops_per_call=6 * n * p,
        ),
    )


def _softplus(v):
    return np.logaddexp(0.0, v)


def _f64(a, *shape):
    return np.asarray(a, dtype=np.float64).reshape(-1, *shape)


def _outputs(calls: list[dict], *names: str):
    """Each named output of every call, a row a call.

    Float32, but ``y``, which a kernel rounds to bf16.
    """
    out = []
    for name in names:
        rows = np.stack([np.asarray(c[name]).reshape(-1) for c in calls])
        out.append(rows.astype(bfloat16 if name == "y" else np.float32))
    return tuple(out)


def ssd_cb_ref(c, b, *, n: int):
    """Numpy reference for [`ssd_cb`][iron.kernels.ssm.ssd_cb], in float64, rounded once to float32."""
    c, b = _f64(c, SSD_BLOCK, n), _f64(b, SSD_BLOCK, n)
    return np.einsum("kln,ksn->kls", c, b).reshape(len(c), -1).astype(np.float32)


def ssd_cb_bound(c, b, *, n: int):
    """[`ssd_cb`][iron.kernels.ssm.ssd_cb]'s error bound.

    n products summed in float32, each rounding at most an ulp of the running
    sum, which the sum of magnitudes bounds.
    """
    c, b = _f64(c, SSD_BLOCK, n), _f64(b, SSD_BLOCK, n)
    magnitude = np.einsum("kln,ksn->kls", np.abs(c), np.abs(b))
    return ((n + 1) * _F32 * magnitude).reshape(len(c), -1)


def _param(a, *shape):
    """Read a held tensor, the same every call, in float64."""
    return np.asarray(a, dtype=np.float64).reshape(shape)


def _decay(g, dt, a_log, dt_bias, head, valid, dt_cols) -> dict:
    """One call of ssd_decay in float64, with each output's bound."""
    q = SSD_BLOCK
    dts = np.where(
        np.arange(q) < valid,
        _softplus(dt.reshape(q, dt_cols)[:, head] + dt_bias[head]),
        0.0,
    )
    a = -dts * np.exp(a_log[head])
    s = np.cumsum(a)
    # seg[l, s] = a_{s+1} + ... + a_l, after[l] = a_{l+1} + ... + a_{q-1}.
    seg = s[:, None] - s[None, :]
    below = np.tril(np.ones((q, q), bool))
    decay = np.where(below, np.exp(np.where(below, seg, 0.0)), 0.0)
    after = s[-1] - s
    m = g.reshape(q, q) * decay * dts[None, :]
    e = np.stack([np.exp(s), np.exp(after) * dts])
    # Each sum of a carries a's own error and q roundings; an exp of a sum
    # carries the sum's error times its size, and exp2f's own.
    e_sum = _E_A + q * _F32

    def e_exp(t):
        return _EXP_REL + _LIMBS3 + e_sum * np.abs(t)

    m_err = np.abs(m) * (e_exp(seg) + _E_DT + 2 * _LIMBS3 + _F32)
    e_err = np.stack(
        [
            np.exp(s) * (e_exp(s) + _F32),
            np.exp(after) * dts * (e_exp(after) + _E_DT + _LIMBS3 + _F32),
        ]
    )
    return dict(m=m, e=e, m_err=m_err, e_err=e_err)


def _decay_calls(g, dt, a_log, dt_bias, head, valid, dt_cols):
    a_log, dt_bias = _param(a_log, dt_cols), _param(dt_bias, dt_cols)
    return [
        _decay(gk, dk, a_log, dt_bias, int(head), int(valid), dt_cols)
        for gk, dk in zip(_f64(g, SSD_BLOCK * SSD_BLOCK), _f64(dt, SSD_BLOCK * dt_cols))
    ]


def _out(m, e, c, x, d, state, head, p, n, dt_cols):
    """Each call of ssd_out in float64, and its bound."""
    q = SSD_BLOCK
    m, e = _param(m, q, q), _param(e, 2, q)
    d, S = _param(d, dt_cols)[int(head)], _param(state, n, p)
    out = []
    for ck, xk in zip(_f64(c, q, n), _f64(x, q, p)):
        y = m @ xk + e[0][:, None] * (ck @ S) + d * xk
        err = (
            (_LIMBS2 + (q + 2) * _F32) * (np.abs(m) @ np.abs(xk))
            + (_LIMBS2 + _LIMBS3 + (n + 2) * _F32)
            * np.abs(e[0])[:, None]
            * (np.abs(ck) @ np.abs(S))
            + 2 * _F32 * np.abs(d * xk)
            + 2.0**-8 * np.abs(y)
        )
        out.append(dict(y=y, y_err=err))
    return out


def ssd_out_ref(m, e, c, x, d, state, head, *, p: int, n: int, dt_cols: int):
    """Numpy reference for [`ssd_out`][iron.kernels.ssm.ssd_out], in float64, rounded once to bf16."""
    return _outputs(_out(m, e, c, x, d, state, head, p, n, dt_cols), "y")[0]


def ssd_out_bound(m, e, c, x, d, state, head, *, p: int, n: int, dt_cols: int):
    """[`ssd_out`][iron.kernels.ssm.ssd_out]'s error bound.

    m and the state as two limbs, the before sum's float32 roundings and its
    three-limb scale by ``e``, and ``y``'s bf16 rounding.
    """
    calls = _out(m, e, c, x, d, state, head, p, n, dt_cols)
    return np.stack([k["y_err"].reshape(-1) for k in calls])


def _state(e, b, x, state, p, n):
    """Each call of ssd_state in float64, and its bound."""
    q = SSD_BLOCK
    e, S = _param(e, 2, q), _param(state, n, p)
    out = []
    for bk, xk in zip(_f64(b, q, n), _f64(x, q, p)):
        S_out = e[0][-1] * S + bk.T @ (e[1][:, None] * xk)
        terms = abs(e[0][-1]) * np.abs(S) + np.abs(bk.T) @ (
            np.abs(e[1])[:, None] * np.abs(xk)
        )
        out.append(dict(S=S_out, S_err=(2 * _LIMBS3 + (q + 3) * _F32) * terms))
    return out


def ssd_state_ref(e, b, x, state, *, p: int, n: int):
    """Numpy reference for [`ssd_state`][iron.kernels.ssm.ssd_state], in float64, rounded once to float32."""
    return _outputs(_state(e, b, x, state, p, n), "S")[0]


def ssd_state_bound(e, b, x, state, *, p: int, n: int):
    """[`ssd_state`][iron.kernels.ssm.ssd_state]'s error bound.

    Three-limb products summed in float32 over the block's positions.
    """
    return np.stack([k["S_err"].reshape(-1) for k in _state(e, b, x, state, p, n)])


def _update(x, b, c, dt, a_log, d, dt_bias, state, head) -> dict:
    """One call of ssm_state_update in float64, with each output's bound."""
    dts = _softplus(dt[head] + dt_bias[head])
    A = np.exp(a_log[head])
    decay = np.exp(-dts * A)
    S = decay * state + np.outer(b, dts * x)
    y = c @ S + d[head] * x
    e_decay = _EXP_REL + 2 * _LIMBS3 + _E_A * dts * A
    S_err = (
        decay * (e_decay + 3 * _F32) * np.abs(state)
        + (_E_DT + _LIMBS3 + 3 * _F32) * np.abs(np.outer(b, dts * x))
        + _F32 * np.abs(S)
    )
    y_err = (
        np.abs(c) @ (S_err + (_LIMBS3 + (len(c) + 1) * _F32) * np.abs(S))
        + 2 * _F32 * np.abs(d[head] * x)
        + 2.0**-8 * np.abs(y)
    )
    return dict(S=S, y=y, S_err=S_err, y_err=y_err)


def _update_calls(x, b, c, dt, a_log, d, dt_bias, state, head, p, n, dt_cols):
    held = [_param(t, dt_cols) for t in (dt, a_log, d, dt_bias)]
    S = _param(state, n, p)
    return [
        _update(xk, bk, ck, held[0], held[1], held[2], held[3], S, int(head))
        for xk, bk, ck in zip(_f64(x, p), _f64(b, n), _f64(c, n))
    ]


def ssm_state_update_ref(
    x, b, c, dt, a_log, d, dt_bias, state, head, *, p: int, n: int, dt_cols: int
):
    """Numpy reference for [`ssm_state_update`][iron.kernels.ssm.ssm_state_update].

    The step in float64, the state rounded once to float32 and ``y`` to bf16.
    """
    calls = _update_calls(x, b, c, dt, a_log, d, dt_bias, state, head, p, n, dt_cols)
    return _outputs(calls, "S", "y")


def ssm_state_update_bound(
    x, b, c, dt, a_log, d, dt_bias, state, head, *, p: int, n: int, dt_cols: int
):
    """[`ssm_state_update`][iron.kernels.ssm.ssm_state_update]'s error bound.

    The decay's and dt's own errors, three-limb products, float32 sums and one
    bf16 rounding of ``y``.
    """
    calls = _update_calls(x, b, c, dt, a_log, d, dt_bias, state, head, p, n, dt_cols)
    return tuple(
        np.stack([np.asarray(k[name]).reshape(-1) for k in calls])
        for name in ("S_err", "y_err")
    )


def _rows(rng, *shape, scale=1.0, dtype=bfloat16):
    return (rng.standard_normal(shape) * scale).astype(np.float32).astype(dtype)


def _heads(rng, dt_cols: int) -> list[np.ndarray]:
    """Draw a model's per-head arrays.

    A_log for decays of 0.37 to 2.7 a unit dt, D near 0, and dt_bias over
    Granite 4.0-H's range, long-memory heads (softplus near 1e-5) among them.
    """
    a_log = rng.uniform(-1.0, 1.0, dt_cols)
    d = rng.normal(0.0, 0.1, dt_cols)
    dt_bias = rng.uniform(-12.0, 4.0, dt_cols)
    return [t.astype(np.float32).astype(bfloat16) for t in (a_log, d, dt_bias)]


def _scores(rng, calls: int, n: int) -> np.ndarray:
    """``g`` of unit-scale ``c`` and ``b``, as ssd_cb gives it."""
    c, b = _rows(rng, calls, SSD_BLOCK, n), _rows(rng, calls, SSD_BLOCK, n)
    g = np.einsum("kln,ksn->kls", c.astype(np.float32), b.astype(np.float32))
    return g.reshape(calls, -1).astype(np.float32)


def ssd_decay_sample(rng, calls: int, *, dt_cols: int):
    """Draw a head's block as a model gives it.

    Scores of unit-scale rows, raw ``dt`` around 0, and a model's per-head
    arrays.
    """
    a_log, _, dt_bias = _heads(rng, dt_cols)
    dt = _rows(rng, calls, SSD_BLOCK * dt_cols)
    return [_scores(rng, calls, 128), dt, a_log, dt_bias]


def _decays(rng) -> tuple[np.ndarray, np.ndarray]:
    """``m`` and ``e`` as ssd_decay gives them, for a random head."""
    a_log, _, dt_bias = _heads(rng, 1)
    out = _decay(
        np.asarray(_scores(rng, 1, 128)[0], np.float64),
        np.asarray(_rows(rng, SSD_BLOCK), np.float64),
        np.asarray(a_log, np.float64),
        np.asarray(dt_bias, np.float64),
        0,
        SSD_BLOCK,
        1,
    )
    return out["m"].reshape(-1).astype(np.float32), out["e"].reshape(-1).astype(
        np.float32
    )


def ssd_out_sample(rng, calls: int, *, p: int, n: int, dt_cols: int):
    """Draw a unit's block for ssd_out.

    A head's decays, unit-scale rows, and a state of the scale a chunk leaves.
    """
    m, e = _decays(rng)
    _, d, _ = _heads(rng, dt_cols)
    c = _rows(rng, calls, SSD_BLOCK * n)
    x = _rows(rng, calls, SSD_BLOCK * p)
    state = _rows(rng, n * p, scale=0.5, dtype=np.float32)
    return [m, e, c, x, d, state]


def ssd_state_sample(rng, calls: int, *, p: int, n: int):
    """Draw a unit's block for ssd_state: a head's decays, unit-scale rows and a state."""
    _, e = _decays(rng)
    b = _rows(rng, calls, SSD_BLOCK * n)
    x = _rows(rng, calls, SSD_BLOCK * p)
    state = _rows(rng, n * p, scale=0.5, dtype=np.float32)
    return [e, b, x, state]


def ssm_state_update_sample(rng, calls: int, *, p: int, n: int, dt_cols: int):
    """Draw a decode step's inputs as a model gives them."""
    a_log, d, dt_bias = _heads(rng, dt_cols)
    return [
        _rows(rng, calls, p),
        _rows(rng, calls, n),
        _rows(rng, calls, n),
        _rows(rng, dt_cols),
        a_log,
        d,
        dt_bias,
        _rows(rng, n * p, scale=0.5, dtype=np.float32),
    ]
