# kernels/ssm.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""State space model kernels: Mamba2's selective scan."""

from functools import partial

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, InOut, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import KernelContract, Param, Trace, _kernel_source, _make_extern
from .core import conv_even
from .sample import _positive
from .zero import zero

_STRICT_FP = ["-ffp-contract=off", "-fno-fast-math"]
# One contract tolerance covers both outputs, so it is y's: a bf16 ulp, with
# a floor for readouts that cancel. A host build of the source matched the
# reference bit for bit in y and to 1e-7 of the range in the state.
_SCAN_TOLERANCE = Tolerance.relative(
    8e-3, range_frac=1e-3, note="one bf16 ulp of y; measured on a host build only"
)


def _zero_state(fn):
    return zero(fn.arg_shape(3), np.float32)


def selective_scan_sample(rng, calls: int, *, rows: int, head_dim: int, state: int):
    """Return inputs a trained Mamba2 head could see: A < 0, so the state decays."""
    xbc = rng.standard_normal((calls, rows * (head_dim + 2 * state))) * 0.5
    dt = rng.standard_normal((calls, rows))
    head = np.stack(
        [
            rng.uniform(-4, 0, calls),  # dt_bias
            -rng.uniform(0.5, 16, calls),  # A = -exp(A_log)
            rng.uniform(0, 2, calls),  # D
        ],
        axis=1,
    )
    return [xbc.astype(bfloat16), dt.astype(bfloat16), head.astype(np.float32)]


def selective_scan_ref(xbc, dt, head, valid, *, rows: int, head_dim: int, state: int):
    """Numpy reference for [`selective_scan`][iron.kernels.ssm.selective_scan], from a zeroed state.

    Returns ``(state, y)`` per call.
    """
    xbc = np.asarray(xbc, np.float32).reshape(-1, rows, head_dim + 2 * state)
    dt = np.asarray(dt, np.float32).reshape(-1, rows)
    head = np.asarray(head, np.float32).reshape(-1, 3)
    s = np.zeros((len(xbc), head_dim, state), np.float32)
    y = np.zeros((len(xbc), rows, head_dim), np.float32)
    for c, (dt_bias, A, D) in enumerate(head):
        for t in range(min(int(valid), rows)):
            x, B, C = np.split(xbc[c, t], [head_dim, head_dim + state])
            step = np.logaddexp(np.float32(0), dt[c, t] + dt_bias)
            s[c] = np.exp(step * A) * s[c] + (step * x)[:, None] * B
            y[c, t] = s[c] @ C + D * x
    return s.reshape(len(s), -1), y.astype(bfloat16).reshape(len(y), -1)


def selective_scan(
    *, rows: int = 16, head_dim: int = 16, state: int = 32
) -> ExternalFunction:
    """Mamba2's state space recurrence for one head over a block of rows.

    ``selective_scan(xbc, dt, head, state, y, valid)``: each of ``rows`` rows
    of bf16 ``xbc`` is the head's ``head_dim`` x, then its group's ``state``
    B and ``state`` C; ``dt`` is the head's bf16 dt per row and ``head`` its
    float32 ``(dt_bias, A, D)``. For each of the first ``valid`` rows,
    ``step = softplus(dt + dt_bias)``, ``state = exp(step * A) * state +
    step * outer(x, B)`` and ``y = state @ C + D * x``, rounded once to bf16.
    ``state`` is float32 ``(head_dim, state)`` and carries from call to call;
    rows from ``valid`` on leave it alone and write zero ``y``. ``head_dim``
    may be a slice of the head's: the rows of state are independent.
    """
    _positive("selective_scan", rows=rows, head_dim=head_dim, state=state)
    return _make_extern(
        "selective_scan",
        _kernel_source("ssm/selective_scan.cc"),
        [
            np.ndarray[(rows * (head_dim + 2 * state),), np.dtype[bfloat16]],
            np.ndarray[(rows,), np.dtype[bfloat16]],
            np.ndarray[(3,), np.dtype[np.float32]],
            np.ndarray[(head_dim * state,), np.dtype[np.float32]],
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
            roles=(In, In, In, InOut, Out, Param),
            reference=partial(
                selective_scan_ref, rows=rows, head_dim=head_dim, state=state
            ),
            initializers=((3, _zero_state),),
            sample=partial(
                selective_scan_sample, rows=rows, head_dim=head_dim, state=state
            ),
            setup=conv_even,
            tolerance=_SCAN_TOLERANCE,
        ),
    )
