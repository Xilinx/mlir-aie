# cases.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""One kernel at one tile size: what a test checks and a performance check times.

A :class:`Case` names a factory, its keyword arguments and the harness
options (call count, runtime scalars, ``Param`` values). What the kernel
computes stays on the factory's ``KernelContract``; a case only says *which*
tile to build, how many independent calls to make, and which edge data it
must survive.

The case tables themselves live with the tests
(``test/python/npu/kernel_cases.py``): the device smoke test, the extensive
sweep and the performance checks all read the same table.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from aie.iron import In, kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.device import from_name
from aie.iron.kernels import Param
from aie.utils import bfp, get_current_device
from aie.utils.hostruntime import set_current_device
from ml_dtypes import bfloat16


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


def _product_extents(contract) -> tuple[int, ...] | None:
    """``(m, k, n)`` or ``(m, k)`` for a kernel whose streamed operands form a product."""
    if not contract.layouts:
        return None
    ins = [
        contract.layouts[i]
        for i, r in enumerate(contract.roles)
        if r is In and contract.layouts[i] is not None
    ]
    if len(ins) < 2 or len(ins[0].shape) != 2 or ins[1].shape[0] != ins[0].shape[1]:
        return None
    return (*ins[0].shape, *ins[1].shape[1:])


@dataclass
class Case:
    """One (kernel, tile, call count) the suite builds, checks and times.

    ``calls`` independent tile invocations of any kernel; the tile itself is
    fixed by the factory kwargs, since the builder validates kernels one
    tile at a time and rejects a whole-problem shape. ``params`` overrides
    the value of unbound tensor ``Param`` arguments (``scale``'s factor);
    ``scalars`` supplies unbound scalar ``Param`` arguments in ABI order.
    ``devices`` restricts a case to the NPU
    generations whose kernels exist (``("npu2",)``), as IRON's
    ``supported_devices`` marker does; empty means every device.
    ``smoke`` marks the one case per kernel the per-PR device test runs;
    the extensive sweep runs them all. ``arg_byte_offsets`` binds tensor
    ``Param`` arguments at a byte offset (``((1, 16),)``: argument 1 sits 16
    bytes past an aligned address), as a design packing several weights
    into one buffer hands them; the name carries ``arg1@16``.
    """

    factory: str
    kwargs: dict = field(default_factory=dict)
    calls: int = 1
    scalars: tuple = ()
    params: tuple = ()
    tag: str = ""
    perf: bool = True
    smoke: bool = False
    data_cases: tuple[str, ...] | None = None  # None: derived from the contract
    devices: tuple[str, ...] = ()
    arg_byte_offsets: tuple = ()

    def fn(self):
        with device_for(self.devices):
            return getattr(kernels, self.factory)(**self.kwargs)

    def harness_opts(self) -> dict:
        opts = dict(calls=self.calls, scalars=self.scalars)
        if self.arg_byte_offsets:
            opts["arg_byte_offsets"] = self.arg_byte_offsets
        return opts

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
        types = fn.arg_types()
        primary = (
            fn.contract.roles.index(In)
            if In in fn.contract.roles
            else fn.contract.out_indices[0]
        )
        in_dt = bfp.dtype_name(kd.shape_dtype(types[primary])[1])
        out_dt = "+".join(
            bfp.dtype_name(kd.shape_dtype(types[i])[1]) for i in fn.contract.out_indices
        )
        dtypes = in_dt if in_dt == out_dt else f"{in_dt}_{out_dt}"
        # A product's tile is named (m, k[, n]) from its declared operands:
        # a 2-D A and a B whose leading extent is A's trailing one. Anything
        # else is named by the element count of its primary tile.
        matrix = _product_extents(fn.contract)
        if matrix:
            dims = "x".join(str(d) for d in (*matrix, self.calls))
        else:
            dims = f"{kd.elems(types[primary])}x{self.calls}"
        extra = [
            f"{k}={bfp.dtype_name(v) if isinstance(v, type) else v}"
            for k, v in sorted(self.kwargs.items())
            if k not in self._NAMED_KWARGS
        ]
        extra += [f"arg{i}@{offset}" for i, offset in self.arg_byte_offsets]
        parts = [self.factory, dims, dtypes, *extra] + ([self.tag] if self.tag else [])
        return "/".join(parts)

    def data_policy(self) -> tuple[str, ...]:
        """Return the edge-data cases this case runs: ``data_cases`` or :func:`data_policy`."""
        return (
            self.data_cases if self.data_cases is not None else data_policy(self.fn())
        )

    def kernel_calls(self) -> int:
        """How many independent tile calls the core invokes in one run."""
        return self.calls

    def work(self) -> int:
        """Arithmetic operations per run, from the contract's ``ops_per_call``."""
        fn = self.fn()
        per_call = fn.contract.ops_per_call
        if per_call is None:
            per_call = sum(kd.elems(fn.arg_types()[i]) for i in fn.contract.out_indices)
        return per_call * self.kernel_calls()

    def supported_on(self, device_name: str) -> bool:
        """Whether the case's kernels exist for ``device_name`` (``"npu1"`` / ``"npu2"``)."""
        return not self.devices or device_name in self.devices


# --------------------------------------------------------------------------
# Edge-case data
# --------------------------------------------------------------------------


# Data cases every kernel of a kind should survive. Random data finds nothing
# a vectorized tail, a saturating add or a NaN path gets wrong. The policy
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
    if In not in c.roles:
        return ("random",)  # output-only kernels have no input edge cases
    if c.sample is not None:
        return ("random",)  # structured inputs have no edge variants
    types = fn.arg_types()
    in_dt = kd.shape_dtype(types[c.roles.index(In)])[1]
    if bfp.is_bfp(in_dt):
        return tuple(d for d in MATRIX_DATA if d != "max")
    if np.issubdtype(np.dtype(in_dt), np.integer):
        return INT_DATA
    return FLOAT_BASE


def inputs_for(case: Case, data_case: str, rng) -> list[np.ndarray]:
    """Host inputs for ``case`` under one data case, in contract order."""
    fn = case.fn()
    c = fn.contract
    inputs = kd.sample_inputs(fn, calls=case.calls, rng=rng)
    tensor_pos = kd._tensor_positions(fn)[0]
    if data_case != "random":
        if c.sample is not None:
            raise ValueError(
                f"{case.factory}: structured inputs have no '{data_case}' variant"
            )
        # Edge data is about the streamed inputs; a Param (scale's factor,
        # filter2d's kernel) keeps its value, so one design serves every case.
        # Integer extremes stay inside what the kernel's accumulator admits
        # (ExternalFunction.input_limit), so "max" tests the datapath, not an
        # overflow the source leaves open.
        for k, (a, i) in enumerate(zip(inputs, tensor_pos)):
            if c.roles[i] == Param:
                continue
            dt, shape = a.dtype, a.shape
            is_int = np.issubdtype(dt, np.integer)
            if is_int:
                hi, lo = int(np.iinfo(dt).max), int(np.iinfo(dt).min)
                limit = fn.input_limit(dt)
                if limit is not None:
                    hi = min(hi, limit)
                    lo = max(lo, -limit) if dt.kind != "u" else 0
            if data_case == "zeros":
                edge = np.zeros(shape)
            elif data_case == "ones":
                edge = np.ones(shape)
            elif data_case == "max":
                edge = np.full(shape, hi if is_int else 3.0e38)
            elif data_case == "min":
                edge = np.full(shape, lo if is_int else -3.0e38)
            elif data_case == "alternating":
                edge = (np.indices(shape).sum(0) % 2) * 2 - 1
            elif data_case == "subnormal":
                edge = rng.uniform(-1e-39, 1e-39, shape)
            elif data_case == "nan_inf":
                edge = rng.standard_normal(shape)
                edge.flat[0], edge.flat[-1] = np.nan, np.inf
                edge.flat[edge.size // 2] = -np.inf
            elif data_case == "large":
                edge = rng.standard_normal(shape) * 1e4
            else:
                raise KeyError(data_case)
            if is_int:
                edge = np.clip(edge, np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
            else:
                edge = edge.astype(np.float32).astype(dt)
            inputs[k] = edge
    if case.params:
        params = iter(case.params)
        inputs = [
            (
                np.full(a.shape, next(params), dtype=a.dtype)
                if c.roles[i] == Param
                else a
            )
            for a, i in zip(inputs, tensor_pos)
        ]
    return inputs


_FLOATS = {np.dtype(t) for t in (bfloat16, np.float16, np.float32, np.float64)}


def _reference_outputs(fn, inputs, scalars, widen: bool) -> list[np.ndarray]:
    wide = [
        a.astype(np.float64) if widen and a.dtype in _FLOATS else a
        for a in map(np.asarray, inputs)
    ]
    result = fn.contract.reference(*fn._reference_args(wide, tuple(scalars)))
    multiple = len(fn.contract.out_indices) > 1
    return [np.asarray(r) for r in (result if multiple else (result,))]


def error_report(fn, got, inputs, *, calls: int, scalars=()) -> list[dict]:
    """Error stats of each floating output against the contract's reference.

    The reference is run on the inputs widened to float64. ``reference`` in
    each entry names what its values fit: ``float64``, or ``float32`` when
    every value is a float32 (a reference that computes in float32, or an
    exact result). A reference that cannot take float64 inputs, or returns
    other shapes for them, runs on the inputs as given and is named by the
    dtype it returns. Where the reference models the kernel (a LUT), this
    measures against that model; ``reference_max_ulp`` is how far the
    reference ``judge`` uses (the contract's, on the inputs as given, in the
    output dtype) is from the one measured against. ``got`` is what
    ``judge`` takes, normalized the same way: decoded, per call, padding
    trimmed.
    """
    from aie.utils.accuracy import error_stats

    c = fn.contract
    plain = _reference_outputs(fn, inputs, scalars, widen=False)
    try:
        refs = _reference_outputs(fn, inputs, scalars, widen=True)
        if [r.shape for r in refs] != [r.shape for r in plain]:
            refs = plain
    except Exception:  # noqa: BLE001 - a reference written for its own dtype
        refs = plain
    actuals = got if len(c.out_indices) > 1 else (got,)
    entries = []
    for k, (i, actual, ref, own) in enumerate(zip(c.out_indices, actuals, refs, plain)):
        dt = np.dtype(fn.arg_dtype(i)) if not bfp.is_bfp(fn.arg_dtype(i)) else None
        if dt not in _FLOATS or ref.dtype not in _FLOATS:
            continue
        layout = c.layouts[i] if c.layouts else None
        actual = layout.decode(actual, calls=calls) if layout else np.asarray(actual)
        actual = actual.reshape(calls, -1)
        if c.out_valid is not None:
            actual = actual[:, : c.out_valid]
        if In not in c.roles and ref.size == actual.shape[1]:
            ref = np.broadcast_to(ref.reshape(1, -1), actual.shape)
            own = np.broadcast_to(own.reshape(1, -1), actual.shape)
        else:
            ref, own = ref.reshape(calls, -1), own.reshape(calls, -1)
        ref64 = ref.astype(np.float64)
        if ref.dtype != np.float64:
            precision = ref.dtype.name
        else:
            with np.errstate(over="ignore"):
                narrow = ref.astype(np.float32).astype(np.float64)
            fits = (narrow == ref64) | np.isnan(ref64)
            precision = "float32" if fits.all() else "float64"
        stats = error_stats(actual, ref64, dt)
        judged = error_stats(own.astype(np.float64), ref64, dt)
        entries.append(
            dict(
                output=k,
                arg=i,
                reference=precision,
                reference_max_ulp=judged.max_ulp,
                **stats.as_dict(),
            )
        )
    return entries


__all__ = [
    "Case",
    "INT_DATA",
    "FLOAT_BASE",
    "MATRIX_DATA",
    "data_policy",
    "device_for",
    "error_report",
    "inputs_for",
]
