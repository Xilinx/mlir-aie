# kernels/ssm.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""State space model kernels: Mamba2's selective scan."""

from functools import partial

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import KernelContract, Param, Trace, _kernel_source, _make_extern
from .core import conv_even
from .sample import _positive

_STRICT_FP = ["-ffp-contract=off", "-fno-fast-math"]


def selective_scan_sample(rng, calls: int, *, rows: int, head_dim: int, state: int):
    """Return inputs a trained Mamba2 head could see: A < 0, so the state decays."""
    x = rng.standard_normal((calls, rows * head_dim)) * 0.5
    bc = rng.standard_normal((calls, rows * 2 * state)) * 0.5
    dt = rng.standard_normal((calls, rows))
    head = np.stack(
        [
            rng.uniform(-4, 0, calls),  # dt_bias
            -rng.uniform(0.5, 16, calls),  # A = -exp(A_log)
            rng.uniform(0, 2, calls),  # D
        ],
        axis=1,
    )
    initial = rng.standard_normal((calls, head_dim * state))
    return [
        x.astype(bfloat16),
        bc.astype(bfloat16),
        dt.astype(bfloat16),
        head.astype(np.float32),
        initial.astype(np.float32),
    ]


def selective_scan_ref(
    x, bc, dt, head, initial, valid, *, rows: int, head_dim: int, state: int
):
    """Numpy reference for [`selective_scan`][iron.kernels.ssm.selective_scan].

    Returns ``(final, y)`` per call.
    """
    calls = len(np.asarray(head).reshape(-1, 3))
    x = np.asarray(x, np.float32).reshape(calls, rows, head_dim)
    bc = np.asarray(bc, np.float32).reshape(calls, rows, 2, state)
    dt = np.asarray(dt, np.float32).reshape(calls, rows)
    head = np.asarray(head, np.float32).reshape(calls, 3)
    s = np.array(initial, np.float32).reshape(calls, head_dim, state)
    y = np.zeros((calls, rows, head_dim), np.float32)
    for c, (dt_bias, A, D) in enumerate(head):
        for t in range(min(int(valid), rows)):
            B, C = bc[c, t]
            step = np.logaddexp(np.float32(0), dt[c, t] + dt_bias)
            s[c] = np.exp(step * A) * s[c] + (step * x[c, t])[:, None] * B
            y[c, t] = s[c] @ C + D * x[c, t]
    return s.reshape(calls, -1), y.astype(bfloat16).reshape(calls, -1)


def _scan_bound(*args, **geometry):
    """1e-6 of each call's state range for ``final``; one bf16 ulp for ``y``."""
    final, y = selective_scan_ref(*args, **geometry)
    span = np.abs(final).max(axis=1, keepdims=True)
    y = np.abs(y.astype(np.float32))
    _, k = np.frexp(y)
    ulp = np.where(y > 0, np.ldexp(np.float32(1), k - 8), 0)
    return np.broadcast_to(1e-6 * span, final.shape), ulp.astype(np.float32)


def selective_scan(
    *, rows: int = 16, head_dim: int = 16, state: int = 32
) -> ExternalFunction:
    """Mamba2's state space recurrence for one head over a block of rows.

    ``selective_scan(x, bc, dt, head, state, final, y, valid)``: row ``t``
    of bf16 ``x`` is the head's ``head_dim`` x and row ``t`` of bf16 ``bc``
    its group's ``state`` B then ``state`` C; ``dt`` is the head's bf16 dt
    per row and ``head`` its float32 ``(dt_bias, A, D)``. ``final`` starts
    as float32 ``state`` (``(head_dim, state)``), and for each of the first
    ``valid`` rows ``step = softplus(dt + dt_bias)``, ``final = exp(step *
    A) * final + step * outer(x, B)`` and ``y = final @ C + D * x``, rounded
    once to bf16. Rows from ``valid`` on leave ``final`` alone and write zero
    ``y``. The caller picks head ``h``'s x from ``xbc`` and group ``h // (heads
    // groups)``'s B and C, and hands one call's ``final`` to the next as its
    ``state``. ``head_dim`` may be a slice of the head's: the rows of state
    are independent.
    """
    _positive("selective_scan", rows=rows, head_dim=head_dim, state=state)
    geometry = dict(rows=rows, head_dim=head_dim, state=state)
    state_ty = np.ndarray[(head_dim * state,), np.dtype[np.float32]]
    return _make_extern(
        "selective_scan",
        _kernel_source("ssm/selective_scan.cc"),
        [
            np.ndarray[(rows * head_dim,), np.dtype[bfloat16]],
            np.ndarray[(rows * 2 * state,), np.dtype[bfloat16]],
            np.ndarray[(rows,), np.dtype[bfloat16]],
            np.ndarray[(3,), np.dtype[np.float32]],
            state_ty,
            state_ty,
            np.ndarray[(rows * head_dim,), np.dtype[bfloat16]],
            np.int32,
        ],
        compile_flags=_STRICT_FP
        + [
            f"-DSCAN_ROWS={rows}",
            f"-DSCAN_HEAD_DIM={head_dim}",
            f"-DSCAN_STATE={state}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, In, In, In, Out, Out, Param),
            reference=partial(selective_scan_ref, **geometry),
            sample=partial(selective_scan_sample, **geometry),
            setup=conv_even,
            tolerance=Tolerance.bounded(
                partial(_scan_bound, **geometry),
                note="host build of the source: y bit for bit, final within "
                "1e-7 of its range",
            ),
        ),
    )
