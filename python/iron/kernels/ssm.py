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
    """Return one head's blocks, A < 0 so its state decays, from a random state."""
    block = rng.standard_normal((calls, rows * (head_dim + 2 * state + 1))) * 0.5
    head = np.concatenate(
        [
            [rng.uniform(-4, 0)],  # dt_bias
            [-rng.uniform(0.5, 16)],  # A = -exp(A_log)
            [rng.uniform(0, 2)],  # D
            rng.standard_normal(head_dim * state),
        ]
    )
    return [block.astype(bfloat16), np.tile(head, (calls, 1)).astype(np.float32)]


def selective_scan_ref(block, head, valid, *, rows: int, head_dim: int, state: int):
    """Numpy reference for [`selective_scan`][iron.kernels.ssm.selective_scan]: call ``b`` is block ``b``.

    Returns ``(final, y)`` per call.
    """
    head = np.asarray(head, np.float32).reshape(-1, 3 + head_dim * state)
    calls = len(head)
    block = np.asarray(block, np.float32).reshape(calls, -1)
    x = block[:, : rows * head_dim].reshape(calls, rows, head_dim)
    bc = block[:, rows * head_dim : -rows].reshape(calls, rows, 2, state)
    dt = block[:, -rows:]
    s = head[0, 3:].reshape(head_dim, state).copy()
    final = np.zeros((calls, head_dim * state), np.float32)
    y = np.zeros((calls, rows, head_dim), np.float32)
    for c in range(calls):
        dt_bias, A, D = head[c, :3]
        for t in range(min(int(valid), rows)):
            B, C = bc[c, t]
            step = np.logaddexp(np.float32(0), dt[c, t] + dt_bias)
            s = np.exp(step * A) * s + (step * x[c, t])[:, None] * B
            y[c, t] = s @ C + D * x[c, t]
        final[c] = s.reshape(-1)
    return final, y.astype(bfloat16).reshape(calls, -1)


def _scan_bound(*args, **geometry):
    """1e-6 of the state's range for ``final``; one bf16 ulp for ``y``."""
    final, y = selective_scan_ref(*args, **geometry)
    span = np.abs(final).max(axis=1, keepdims=True)
    y = np.abs(y.astype(np.float32))
    _, k = np.frexp(y)
    ulp = np.where(y > 0, np.ldexp(np.float32(1), k - 8), 0)
    return np.broadcast_to(1e-6 * span, final.shape), ulp.astype(np.float32)


def selective_scan(
    *, rows: int = 16, head_dim: int = 16, state: int = 32
) -> ExternalFunction:
    """Mamba2's state space recurrence for one head, a block of rows a call.

    ``selective_scan(block, head, final, y, valid, b)``: ``block`` is bf16
    x (``rows`` x ``head_dim``), then bc (``rows`` rows of the group's
    ``state`` B then ``state`` C), then dt (``rows``). ``head`` is float32
    ``(dt_bias, A, D)`` then the state before the head's first block,
    ``(head_dim, state)``. Block ``b`` of a head (0 first) runs its first
    ``valid`` rows: ``step = softplus(dt + dt_bias)``, ``state = exp(step *
    A) * state + step * outer(x, B)`` and ``y = state @ C + D * x``, rounded
    once to bf16; rows from ``valid`` on leave the state alone and write
    zero ``y``. The state stays in the core between blocks, and ``final``
    is it after this one. The caller gathers head ``h``'s x and dt and
    group ``h // (heads // groups)``'s B and C into ``block``. ``head_dim``
    may be a slice of the head's: the rows of state are independent.
    """
    _positive("selective_scan", rows=rows, head_dim=head_dim, state=state)
    geometry = dict(rows=rows, head_dim=head_dim, state=state)
    return _make_extern(
        "selective_scan",
        _kernel_source("ssm/selective_scan.cc"),
        [
            np.ndarray[(rows * (head_dim + 2 * state + 1),), np.dtype[bfloat16]],
            np.ndarray[(3 + head_dim * state,), np.dtype[np.float32]],
            np.ndarray[(head_dim * state,), np.dtype[np.float32]],
            np.ndarray[(rows * head_dim,), np.dtype[bfloat16]],
            np.int32,
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
            roles=(In, In, Out, Out, Param, Param),
            parameter_bindings=((5, 0),),
            call_index=5,
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
