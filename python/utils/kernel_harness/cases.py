# cases.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""One kernel at one shape: what a test checks and a benchmark times.

A :class:`Case` names a factory, its keyword arguments and the harness
options (call count or matrix shape, runtime scalars, ``param`` values). What
the kernel computes stays on the factory's ``KernelContract``; a case only
says *which* shape to run and which edge data it must survive.

The case tables themselves live with the tests
(``test/python/npu/kernel_cases.py``): the device smoke test, the extensive
sweep and ``python -m aie.utils.kernel_harness`` all read the same table.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from aie.iron import kernels
from aie.iron.device import from_name
from aie.utils import get_current_device
from aie.utils import kernel_harness as kh
from aie.utils.hostruntime import set_current_device


@contextmanager
def device_for(devices: tuple[str, ...]):
    """Bind an iron device a device-restricted factory needs, then restore.

    Factories pick their source through the current device; an npu2-only one
    (``exp2f_vec``) refuses to build under npu1. Naming and describing such a
    case on a host with no device set must still work, so the case binds the
    first device it supports around the factory call.
    """
    if not devices:
        yield
        return
    try:
        previous = get_current_device(probe_runtime=False)
    except Exception:  # noqa: BLE001 - no device bound
        previous = None
    set_current_device(from_name(devices[0], n_cols=1))
    try:
        yield
    finally:
        set_current_device(previous)


@dataclass
class Case:
    """One (kernel, shape) the suite builds, checks and times.

    ``calls`` iterations of a streaming kernel, or ``shape`` = (M, K, N) /
    (M, K) host operands for ``mm`` / ``mv``. ``params`` overrides the value
    of ``param`` arguments (``scale``'s factor); ``scalars`` supplies
    ``scalar`` arguments in order. ``devices`` restricts a case to the NPU
    generations whose kernels exist (``("npu2",)``), as IRON's
    ``supported_devices`` marker does; empty means every device.
    ``smoke`` marks the one case per kernel the per-PR device test runs;
    the extensive sweep runs them all.
    """

    factory: str
    kwargs: dict = field(default_factory=dict)
    calls: int = 1
    shape: tuple | None = None
    scalars: tuple = ()
    params: tuple = ()
    tag: str = ""
    perf: bool = True
    smoke: bool = False
    data_cases: tuple[str, ...] | None = None  # None: derived from the contract
    devices: tuple[str, ...] = ()

    def fn(self):
        with device_for(self.devices):
            return getattr(kernels, self.factory)(**self.kwargs)

    def harness_opts(self) -> dict:
        return dict(calls=self.calls, shape=self.shape, scalars=self.scalars)

    # Factory kwargs that the dims / dtype segments of the name already encode.
    # skip_dtype is absent on purpose: it types neither the first input nor
    # the output, so it has to be spelled out.
    _NAMED_KWARGS = frozenset(
        {
            "tile_size",
            "dtype",
            "act_dtype",
            "input_dtype",
            "output_dtype",
            "dim_m",
            "dim_k",
            "dim_n",
        }
    )

    @functools.cached_property
    def name(self) -> str:
        """Stable series key: ``factory/<dims>/<in dtype>[_<out dtype>][/k=v...][/tag]``.

        The dtype segment names the input dtype, and the output dtype too when
        it differs (``mm/256x256x256/int8_int32``). Factory kwargs the dims
        and dtypes do not encode (``subtile=8``, ``skip_dtype=int8``) are
        appended, so two cases of one factory never share a series. Cached:
        building the kernel is the expensive part.
        """
        fn = self.fn()
        types = kh._arg_types(fn)
        in_dt = kh.dtype_name(kh._shape_dtype(types[fn.contract.roles.index("in")])[1])
        out_dt = kh.dtype_name(kh._shape_dtype(types[fn.contract.out_index])[1])
        dtypes = in_dt if in_dt == out_dt else f"{in_dt}_{out_dt}"
        if self.shape:
            dims = "x".join(str(d) for d in self.shape)
        else:
            dims = f"{kh._elems(types[fn.contract.roles.index('in')])}x{self.calls}"
        extra = [
            f"{k}={kh.dtype_name(v) if isinstance(v, type) else v}"
            for k, v in sorted(self.kwargs.items())
            if k not in self._NAMED_KWARGS
        ]
        parts = [self.factory, dims, dtypes, *extra] + ([self.tag] if self.tag else [])
        return "/".join(parts)

    def data_policy(self) -> tuple[str, ...]:
        """Return the edge-data cases this case runs: ``data_cases`` or :func:`data_policy`."""
        return (
            self.data_cases if self.data_cases is not None else data_policy(self.fn())
        )

    def kernel_calls(self) -> int:
        """How many times the core invokes the kernel in one run."""
        if self.shape is None:
            return self.calls
        m, k = self.kwargs["dim_m"], self.kwargs["dim_k"]
        if len(self.shape) == 3:
            M, K, N = self.shape
            return (M // m) * (N // self.kwargs["dim_n"]) * (K // k)
        M, K = self.shape
        return (M // m) * (K // k)

    def work(self) -> int:
        """Arithmetic operations per run, from the contract's ``ops_per_call``."""
        fn = self.fn()
        per_call = fn.contract.ops_per_call
        if per_call is None:
            per_call = kh._elems(kh._arg_types(fn)[fn.contract.out_index])
        return per_call * self.kernel_calls()

    def supported_on(self, device_name: str) -> bool:
        """Whether the case's kernels exist for ``device_name`` (``"npu1"`` / ``"npu2"``)."""
        return not self.devices or device_name in self.devices


# --------------------------------------------------------------------------
# Edge-case data
# --------------------------------------------------------------------------

# Data cases every kernel of a kind should survive. Random data finds nothing
# a vectorised tail, a saturating add or a NaN path gets wrong. The policy
# is derived from each contract by `data_policy`: integer kernels get the
# extremes (inside `input_limit`); float kernels get subnormal inputs only
# when the contract says what the core does with them (`subnormals`) and
# NaN / inf only when it declares they propagate (`nonfinite`); widening a
# kernel's data is therefore a contract change, not a table edit. Matmul
# operands skip NaN (the reference accumulates them into every output) and
# bfp16ebs8 operands skip "max" too (3e38 products overflow the float
# reference; what the core's bfp16 store makes of an inf is unknown).
INT_DATA = ("random", "zeros", "max", "min", "alternating")
FLOAT_BASE = ("random", "zeros", "ones", "large", "alternating")
MATRIX_DATA = ("random", "zeros", "ones", "alternating", "max")


def data_policy(fn) -> tuple[str, ...]:
    """Return the edge-data cases a kernel's contract admits (see the note above)."""
    c = fn.contract
    if c.sample is not None:
        return ("random",)  # structured inputs have no edge variants
    types = kh._arg_types(fn)
    in_dt = kh._shape_dtype(types[c.roles.index("in")])[1]
    if kh._is_bfp(in_dt):
        return tuple(d for d in MATRIX_DATA if d != "max")
    if kh.is_matmul(fn) or kh.is_matvec(fn):
        return MATRIX_DATA
    if np.issubdtype(np.dtype(in_dt), np.integer):
        return INT_DATA
    cases: list[str] = list(FLOAT_BASE)
    if c.subnormals != "unspecified":
        cases.append("subnormal")
    if c.nonfinite == "propagate":
        cases.append("nan_inf")
    return tuple(cases)


def _edge(shape, dtype, rng, case: str, limit: int | None = None) -> np.ndarray:
    """Return one edge-case array.

    ``limit`` bounds the integer extremes ("max", "min") to what the kernel's
    accumulator admits (``kernel_harness.input_limit``).
    """
    dt = np.dtype(dtype)
    is_int = np.issubdtype(dt, np.integer)
    hi: int | None = None
    lo: int | None = None
    if is_int:
        hi, lo = int(np.iinfo(dt).max), int(np.iinfo(dt).min)
        if limit is not None:
            hi = min(hi, limit)
            lo = max(lo, -limit) if dt.kind != "u" else 0
    if case == "zeros":
        a = np.zeros(shape)
    elif case == "ones":
        a = np.ones(shape)
    elif case == "max":
        a = np.full(shape, hi if is_int else 3.0e38)
    elif case == "min":
        a = np.full(shape, lo if is_int else -3.0e38)
    elif case == "alternating":
        a = (np.indices(shape).sum(0) % 2) * 2 - 1
    elif case == "subnormal":
        a = rng.uniform(-1e-39, 1e-39, shape)
    elif case == "nan_inf":
        a = rng.standard_normal(shape)
        a.flat[0], a.flat[-1], a.flat[a.size // 2] = np.nan, np.inf, -np.inf
    elif case == "large":
        a = rng.standard_normal(shape) * 1e4
    else:
        raise KeyError(case)
    if is_int:
        return np.clip(a, np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
    return a.astype(np.float32).astype(dt)


def inputs_for(case: Case, data_case: str, rng) -> list[np.ndarray]:
    """Host inputs for ``case`` under one data case, in contract order."""
    fn = case.fn()
    c = fn.contract
    inputs = kh.sample_inputs(fn, calls=case.calls, shape=case.shape, rng=rng)
    tensor_pos = [i for i, r in enumerate(c.roles) if r in ("in", "param")]
    if data_case != "random":
        if c.sample is not None:
            raise ValueError(
                f"{case.factory}: structured inputs have no '{data_case}' variant"
            )
        # Edge data is about the streamed inputs; a `param` (scale's factor,
        # filter2d's kernel) keeps its value, so one design serves every case.
        # Integer extremes stay inside what the kernel's accumulator admits,
        # so "max" tests the datapath, not an overflow the source leaves open.
        k_total = case.shape[1] if case.shape else None
        inputs = [
            (
                a
                if c.roles[i] == "param"
                else _edge(
                    a.shape,
                    a.dtype,
                    rng,
                    data_case,
                    kh.input_limit(fn, a.dtype, reduction=k_total),
                )
            )
            for a, i in zip(inputs, tensor_pos)
        ]
    if case.params:
        params = iter(case.params)
        inputs = [
            (
                np.full(a.shape, next(params), dtype=a.dtype)
                if c.roles[i] == "param"
                else a
            )
            for a, i in zip(inputs, tensor_pos)
        ]
    return inputs


def load_cases(path: str) -> list[Case]:
    """Import ``CASES`` from a Python file, for the command-line drivers."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("kernel_cases", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import cases from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.CASES)


__all__ = [
    "Case",
    "INT_DATA",
    "FLOAT_BASE",
    "MATRIX_DATA",
    "data_policy",
    "device_for",
    "inputs_for",
    "load_cases",
]
