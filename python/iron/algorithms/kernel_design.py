# kernel_design.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Build, sample and check independent kernel calls from their declarations.

Each call consumes one tile per ``In``, reads constant ``Param`` values, and
writes one tile per output. Layout codecs convert logical tiles to kernel
storage on the host. A cascade pair (a PUT half and the GET half it names as
``cascade_partner``) is two stages joined by a ``CascadeFlow``; the GET
half's outputs are what the design returns. The Workers and the runtime
sequence are the shared single-core pipeline's (``_pipeline``).
Whole-problem tiling, reductions across kernel calls and multi-core
schedules belong to algorithms, not this kernel-validation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from pathlib import Path
from typing import Callable

import numpy as np
from aie.helpers.util import np_ndarray_type_get_dtype, np_ndarray_type_get_shape
from aie.iron.buffer import Buffer
from aie.iron.dataflow import ObjectFifo
from aie.iron.kernels._common import CallIndex, Param, _is_tensor_type
from aie.utils import bfp, get_current_device, tensor
from aie.utils.compile.jit import CompileTime, In, InOut, Out
from aie.utils.jit import jit
from aie.utils.trace import TraceConfig
from aie.utils.trace.utils import get_cycles_summary
from aie.utils.verify import poisoned

from ._pipeline import Stage, pipeline


def _contract(fn):
    if fn.contract is None:
        raise ValueError(
            f"kernel '{fn.name}' declares no contract; add a KernelContract"
        )
    fn.contract.validate_types(fn.arg_types())
    return fn.contract


def shape_dtype(arg_type):
    """Return the shape and dtype of a declared numpy tensor type."""
    return np_ndarray_type_get_shape(arg_type), np_ndarray_type_get_dtype(arg_type)


def elems(arg_type):
    return int(np.prod(shape_dtype(arg_type)[0]))


def _calls(calls, shape=None):
    if shape is not None:
        raise ValueError(
            "kernel validation takes independent tiles, not a whole-problem shape; "
            "set the factory's tile dimensions and use calls for repetitions"
        )
    try:
        calls = index(calls)
    except TypeError as exc:
        raise ValueError(f"calls must be a positive integer, got {calls}") from exc
    if calls < 1:
        raise ValueError(f"calls must be a positive integer, got {calls}")
    return calls


def _device():
    device = get_current_device()
    if device is None:
        raise RuntimeError(
            "no device is bound; select one with iron.set_current_device()"
        )
    return device


def _stack_bytes(fn):
    return _contract(fn).stack_bytes or _device().default_core_stack_bytes


def _halves(fn):
    """Return the kernels one design runs, in cascade order: ``[put, get]`` or ``[fn]``."""
    _contract(fn)
    halves = fn.halves()
    for half in halves:
        _contract(half)
    return halves


def _tensor_positions(fn):
    c = _contract(fn)
    types = fn.arg_types()
    return [
        i for i in c.reference_indices() if _is_tensor_type(types[i])
    ], c.out_indices


def _layout(c, i):
    return c.layouts[i] if c.layouts else None


def _fifo_plan(fn):
    """Pack same-type inputs per call, respecting the core's DMA channel budget."""
    c = _contract(fn)
    types = fn.arg_types()
    ins = [i for i, r in enumerate(c.roles) if r is In]
    params = [
        i for i, r in enumerate(c.roles) if r is Param and _is_tensor_type(types[i])
    ]
    by_type = {}
    for i in ins:
        by_type.setdefault(types[i], []).append(i)
    groups = list(by_type.values())
    if len(groups) > _device().core_dma_channels_in:
        raise ValueError(f"{fn.name}: input types exceed the core's input DMA channels")
    if len(c.out_indices) > _device().core_dma_channels_out:
        raise ValueError(f"{fn.name}: outputs exceed the core's output DMA channels")
    return groups, ins, params


def _encode_params(fn, params):
    """Encode the unbound tensor Params of every half, in argument order, as design constants."""
    params = list(params)
    encoded = []
    for half in _halves(fn):
        _, _, positions = _fifo_plan(half)
        bound = dict(half.contract.parameter_bindings)
        free = [i for i in positions if i not in bound]
        bound.update(zip(free, params[: len(free)]))
        params = params[len(free) :]
        half_encoded = []
        for i in positions:
            if i not in bound:
                raise ValueError(
                    f"{half.name}: parameters need values at design time; "
                    f"param {i} has none"
                )
            value = bound[i]
            shape, dt = shape_dtype(half.arg_types()[i])
            layout = _layout(half.contract, i)
            value = layout.encode(value) if layout else np.asarray(value)
            if value.size != int(np.prod(shape)):
                raise ValueError(
                    f"{half.name}: param {i} must contain {shape} elements"
                )
            half_encoded.append(
                (np.dtype(dt).name, shape, tuple(value.astype(dt).ravel().tolist()))
            )
        encoded.append(tuple(half_encoded))
    if params:
        raise ValueError(
            f"{fn.name}: {len(params)} more param value(s) than parameters"
        )
    return tuple(encoded)


def _stage(fn, k, calls, scalars, params):
    """Plan one half: consume its scalars and encoded params; return the stage, its fifo groups and the leftovers."""
    c = _contract(fn)
    types = fn.arg_types()
    groups, ins, param_pos = _fifo_plan(fn)
    outs = c.out_indices
    bound = {
        i: value for i, value in c.parameter_bindings if not _is_tensor_type(types[i])
    }
    free = [
        i
        for i, r in enumerate(c.roles)
        if r is Param and not _is_tensor_type(types[i]) and i not in bound
    ]
    if len(scalars) < len(free):
        raise ValueError(
            f"{fn.name}: expected {len(free)} scalar(s), got {len(scalars)}"
        )
    taken, scalars = scalars[: len(free)], scalars[len(free) :]
    if any(not isinstance(v, (int, float, np.integer, np.floating)) for v in taken):
        raise ValueError(f"{fn.name}: expected scalar parameter values")
    bound.update(zip(free, taken))
    if len(params) != len(param_pos):
        raise ValueError(f"{fn.name}: expected {len(param_pos)} param values")
    initializers = [(i, init(fn)) for i, init in c.initializers]
    # An InOut read back on every independent call needs a declared
    # initializer; one that spans the sequence is the kernel's own to
    # initialize on its first call.
    if not c.output_spans_calls and any(
        r is InOut and i not in dict(initializers) for i, r in enumerate(c.roles)
    ):
        raise ValueError(f"{fn.name}: every InOut requires a declared initializer")
    setter = c.setup() if c.setup else None

    def nbytes(i):
        return elems(types[i]) * bfp.itemsize(shape_dtype(types[i])[1])

    # Two sets of tiles (ping-pong) when they fit beside the parameters and
    # the stack, one otherwise.
    tile_bytes = sum(nbytes(i) for i in [*ins, *outs])
    fixed_bytes = sum(nbytes(i) for i in param_pos) + _stack_bytes(fn)
    core_bytes = _device().core_memory_bytes
    depth = next(
        (d for d in (2, 1) if d * tile_bytes + fixed_bytes <= core_bytes), None
    )
    if depth is None:
        raise ValueError(
            f"{fn.name}: tiles plus parameters and stack exceed core memory; use a smaller tile"
        )
    fifos_in = [
        ObjectFifo(types[g[0]], name=f"in{k}_{j}", depth=len(g) * depth)
        for j, g in enumerate(groups)
    ]
    fifos_out = [
        ObjectFifo(types[i], name=f"out{k}_{j}", depth=depth)
        for j, i in enumerate(outs)
    ]
    buffers = [
        Buffer(
            types[i],
            name=f"param{k}_{j}",
            initial_value=np.array(vals, dtype=np.dtype(dt)).reshape(shape),
        )
        for j, (i, (dt, shape, vals)) in enumerate(zip(param_pos, params))
    ]
    # The Worker's constants: the param buffers, the kernel, the
    # initializer kernels and the setup callable.
    n_param, n_init = len(buffers), len(initializers)
    slot = {i: j for j, i in enumerate(outs)}

    def body(acquired, outputs, _held, constants, call):
        values = dict(zip(param_pos, constants[:n_param]))
        values.update(bound)
        values.update({i: call for i, v in bound.items() if v is CallIndex})
        for got, group in zip(acquired, groups):
            values.update(
                (i, got if len(group) == 1 else got[j]) for j, i in enumerate(group)
            )
        values.update(zip(outs, outputs))
        constants[n_param](*(values[i] for i in range(len(c.roles))))

    def initialize(outputs, constants):
        kernels = constants[n_param + 1 : n_param + 1 + n_init]
        for (i, _), init in zip(initializers, kernels):
            init(outputs[slot[i]])

    stage = Stage(
        body,
        inputs=[(fifo, len(g)) for fifo, g in zip(fifos_in, groups)],
        outputs=fifos_out,
        constants=buffers
        + [fn]
        + [init for _, init in initializers]
        + ([setter] if setter else []),
        iterations=calls,
        outputs_span_iterations=c.output_spans_calls,
        prologue=(lambda constants: constants[-1]()) if setter else None,
        initialize=initialize if initializers else None,
        stack_size=_stack_bytes(fn),
    )
    return stage, groups, scalars


def _build_stream(
    *, factory, factory_kwargs, calls, scalars=(), params=(), trace_config=None
):
    fn = factory(**factory_kwargs)
    halves = _halves(fn)
    if len(params) != len(halves):
        raise ValueError(
            f"{fn.name}: expected encoded params for {len(halves)} kernel(s)"
        )
    stages, groups = [], []
    left = tuple(scalars)
    for k, (half, half_params) in enumerate(zip(halves, params)):
        stage, stage_groups, left = _stage(half, k, calls, left, half_params)
        stages.append(stage)
        groups.append(stage_groups)
    if left:
        raise ValueError(
            f"{fn.name}: {len(left)} more scalar(s) than scalar parameters"
        )
    get = halves[-1]
    stages[-1].trace = trace_config is not None

    def host_ty(half, i, repetitions):
        types = half.arg_types()
        return np.ndarray[
            (elems(types[i]) * repetitions,), np.dtype[shape_dtype(types[i])[1]]
        ]

    host_types = [
        host_ty(half, g[0], calls * len(g))
        for half, half_groups in zip(halves, groups)
        for g in half_groups
    ]
    host_types += [
        host_ty(get, i, _output_calls(get, calls)) for i in get.contract.out_indices
    ]
    fifos_in = [fifo for stage in stages for fifo, _ in stage.inputs]
    transfers = [(fifo, "fill", j) for j, fifo in enumerate(fifos_in)]
    transfers += [
        (fifo, "drain", len(fifos_in) + j) for j, fifo in enumerate(stages[-1].outputs)
    ]
    return pipeline(
        stages,
        host_types,
        transfers,
        cascade=len(stages) == 2,
        trace_size=trace_config.trace_size if trace_config else 0,
    )


# One JIT entry for every kernel. Its tensors are the host buffers the
# design takes, inputs then outputs, in ``host_args`` order (the marker only
# says they are tensors); the count follows from the factory, so the cache
# key never depends on closure state.
@jit
def _stream(
    *tensors: In,
    factory: CompileTime[Callable],
    factory_kwargs: CompileTime[dict],
    calls: CompileTime[int],
    scalars: CompileTime[tuple] = (),
    params: CompileTime[tuple] = (),
    trace_config: CompileTime[TraceConfig | None] = None,
):
    return _build_stream(
        factory=factory,
        factory_kwargs=factory_kwargs,
        calls=calls,
        scalars=scalars,
        params=params,
        trace_config=trace_config,
    )


def _output_calls(fn, calls):
    """Return how many output tiles ``calls`` calls produce: one when the output spans them."""
    return 1 if _contract(fn).output_spans_calls else calls


def _host_groups(fn):
    """Every half's input fifo groups, in the order the host tensors take."""
    return [(half, g) for half in _halves(fn) for g in _fifo_plan(half)[0]]


def design(
    factory,
    *,
    calls=1,
    scalars=(),
    shape=None,
    params=None,
    aiecc_flags=None,
    **factory_kwargs,
):
    """Wrap tile calls; ``params``/``scalars`` supply unbound tensor/scalar Params.

    This harness embeds these values for every call; changing them recompiles
    the design. Direct designs can supply different operands on each call.
    For a cascade pair, pass the GET half's factory: the PUT half's inputs,
    scalars and params come first in each list.
    """
    calls = _calls(calls, shape)
    fn = factory(**factory_kwargs)
    c = _contract(fn)
    if c.unsupported:
        raise ValueError(
            f"{fn.name}: the generic harness cannot build this kernel: {c.unsupported}"
        )
    _host_groups(fn)  # the DMA channel budget is checked before anything builds
    flags: list[str] = list(aiecc_flags or ())
    if any(
        bfp.is_bfp(shape_dtype(t)[1])
        for half in _halves(fn)
        for t in half.arg_types()
        if _is_tensor_type(t)
    ):
        if "--dynamic-objFifos" not in flags:
            flags.append("--dynamic-objFifos")
    return _stream.specialize(
        factory=factory,
        factory_kwargs=factory_kwargs,
        calls=calls,
        scalars=tuple(scalars),
        params=_encode_params(fn, params or ()),
        **({"aiecc_flags": flags} if flags else {}),
    )


_INT_RANGE = {np.int8: 60, np.int16: 8000, np.int32: 1 << 20, np.uint8: 255}


def sample_inputs(fn, *, calls=1, shape=None, rng=None):
    """One array per unbound In/tensor Param of every half; only In has a call dimension."""
    calls = _calls(calls, shape)
    rng = np.random.default_rng(0) if rng is None else rng
    c = _contract(fn)
    if c.sample is not None:
        return c.sample(rng, calls)
    result = []
    for half in _halves(fn):
        hc = half.contract
        for i in _tensor_positions(half)[0]:
            s, dt = shape_dtype(half.arg_types()[i])
            layout = _layout(hc, i)
            s = layout.shape if layout else (int(np.prod(s)),)
            dt = np.float32 if bfp.is_bfp(dt) else dt
            s = ((calls,) if hc.roles[i] is In else ()) + s
            if np.issubdtype(np.dtype(dt), np.integer):
                r = fn.input_limit(dt) or _INT_RANGE.get(dt, 1 << 15)
                lo = 0 if np.dtype(dt).kind == "u" else -r
                result.append(rng.integers(lo, r, size=s).astype(dt))
            else:
                result.append(rng.standard_normal(s).astype(np.float32).astype(dt))
    return result


def host_layout(fn, inputs):
    """Pack declared layouts and interleave same-type In tiles; omit Param buffers."""
    positions = [(half, i) for half in _halves(fn) for i in _tensor_positions(half)[0]]
    if len(inputs) != len(positions):
        raise ValueError(f"{fn.name}: expected {len(positions)} input arrays")
    values = {(id(half), i): a for (half, i), a in zip(positions, inputs)}
    result = []
    for half, group in _host_groups(fn):
        arrays = []
        for i in group:
            value = np.asarray(values[(id(half), i)])
            layout = _layout(half.contract, i)
            arrays.append(
                layout.encode(value)
                if layout
                else value.reshape(-1, elems(half.arg_types()[i]))
            )
        result.append(
            np.ascontiguousarray(
                arrays[0] if len(arrays) == 1 else np.stack(arrays, axis=1)
            )
        )
    return result


def output_size(fn, *, calls=1, shape=None):
    """Storage elements per output; a tuple for multiple outputs."""
    calls = _output_calls(fn, _calls(calls, shape))
    types = fn.arg_types()
    sizes = tuple(
        elems(types[i])
        * calls
        * (bfp.BLOCK_BYTES if bfp.is_bfp(shape_dtype(types[i])[1]) else 1)
        for i in _contract(fn).out_indices
    )
    return sizes[0] if len(sizes) == 1 else sizes


@dataclass(frozen=True)
class _HostBuffer:
    """A physical host buffer in design argument order."""

    direction: type
    shape: tuple[int, ...]
    dtype: type

    @property
    def n_elements(self):
        return int(np.prod(self.shape))


def host_args(fn, *, calls=1, shape=None):
    """Describe physical host buffers, not kernel arguments or constant Params.

    Same-type streamed inputs of one kernel share a buffer; each output has
    its own. For a cascade pair the PUT half's buffers come first.
    Descriptors expose ``direction``, ``shape``, ``dtype`` and ``n_elements``.
    """
    calls = _calls(calls, shape)
    result = []
    entries = [(In, half, g, calls) for half, g in _host_groups(fn)]
    entries += [
        (Out, fn, [i], _output_calls(fn, calls)) for i in _contract(fn).out_indices
    ]
    for direction, half, indices, reps in entries:
        types = half.arg_types()
        i = indices[0]
        dt = shape_dtype(types[i])[1]
        n = elems(types[i])
        if bfp.is_bfp(dt):
            n, dt = n * bfp.BLOCK_BYTES, np.uint8
        s = (reps, n) if len(indices) == 1 else (reps, len(indices), n)
        result.append(_HostBuffer(direction, s, dt))
    return result


def upload(inputs, out_size, out_dtype, *, fn, poison=False):
    """Return design inputs and output tensor(s); splat multiple outputs when calling."""
    ins = [
        tensor(a.reshape(-1), dtype=a.dtype, device="npu")
        for a in host_layout(fn, inputs)
    ]
    multiple = len(fn.contract.out_indices) > 1
    sizes, dtypes = (out_size, out_dtype) if multiple else ((out_size,), (out_dtype,))
    outs = [
        tensor(
            poisoned(n, dt) if poison else np.zeros(n, dtype=dt), dtype=dt, device="npu"
        )
        for n, dt in zip(sizes, dtypes)
    ]
    return ins, tuple(outs) if multiple else outs[0]


def cycles_per_call(
    design_, inputs, out_size, out_dtype, *, fn, trace_size, workdir, calls=1
):
    """Measure declared whole-call event pairs, never partial internal regions."""
    if not _contract(fn).trace_cycles:
        return []
    calls = _calls(calls)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = TraceConfig(trace_size=trace_size, trace_file=str(workdir / "trace.txt"))
    ins, out = upload(inputs, out_size, out_dtype, fn=fn)
    design_(*ins, *(out if isinstance(out, tuple) else (out,)), trace_config=cfg)
    if cfg.physical_mlir_path is None:
        raise RuntimeError("the traced run recorded no physical MLIR path")
    trace_json = workdir / "trace.json"
    cfg.trace_to_json(cfg.physical_mlir_path, str(trace_json))
    durations = [int(d) for p in get_cycles_summary(str(trace_json)) for d in p[1:]]
    if len(durations) != calls:
        raise RuntimeError(
            f"{fn.name}: expected {calls} whole-call trace intervals, got "
            f"{len(durations)}; incomplete trace or incorrect trace_cycles contract"
        )
    return durations


__all__ = [
    "cycles_per_call",
    "design",
    "elems",
    "host_layout",
    "host_args",
    "output_size",
    "sample_inputs",
    "shape_dtype",
    "upload",
]
