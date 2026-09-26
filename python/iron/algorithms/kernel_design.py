# kernel_design.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Build, sample and check independent kernel calls from their declarations.

Each call consumes one tile per ``In``, reads constant ``Param`` values, and
writes one tile per output. Layout codecs convert logical tiles to kernel
storage on the host. The Worker and the runtime sequence are the shared
single-core pipeline's (``_pipeline``). Whole-problem tiling, reductions
across kernel calls and multi-core schedules belong to algorithms, not this
kernel-validation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import index
from pathlib import Path
from typing import Callable

import numpy as np
from aie.dialects import memref  # pyright: ignore[reportAttributeAccessIssue]
from aie.extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
    constant,
)
from aie.helpers.npdtypes import np_ndarray_type_get_dtype, np_ndarray_type_get_shape
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.iron.buffer import Buffer
from aie.iron.dataflow import ObjectFifo
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import Param, _is_tensor_type
from aie.utils import bfp, ensure_current_device, tensor
from aie.utils.compile.jit import CompileTime, In, InOut, Out
from aie.utils.jit import jit
from aie.utils.trace import TraceConfig
from aie.utils.trace.events import CoreEvent
from aie.utils.trace.utils import get_cycles_summary
from aie.utils.verify import poisoned

from ._pipeline import Stage, pipeline

GUARD_BYTES = 64
# The traced core emits this many event0/event1 pairs after its last call:
# the trace unit sends only whole packets, and without them the last two
# of bn_conv2dk3_dw_out_split's 8 calls never left the tile (1 or 2 pairs
# still lost them). The pairs decoded as 1 to 18 cycles.
TRACE_FLUSH = 16
FLUSH_CYCLES = 32


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
    # Bound, not just probed: a factory reads the arch from the bound device,
    # so one called with none bound builds the aie2 variant of its contract.
    device = ensure_current_device()
    if device is None:
        raise RuntimeError(
            "no device is bound; select one with iron.set_current_device()"
        )
    return device


def _stack_bytes(fn):
    return _contract(fn).stack_bytes or _device().default_core_stack_bytes


def _guarded(fn, guard):
    """Return the outputs ``guard`` covers: every one but bfp."""
    types = fn.arg_types()
    return [
        i
        for i in _contract(fn).out_indices
        if guard and not bfp.is_bfp(shape_dtype(types[i])[1])
    ]


def _guard_elems(arg_type):
    return GUARD_BYTES // np.dtype(shape_dtype(arg_type)[1]).itemsize


def _poison_fill(words, use_chess):
    """Return a kernel that fills ``words`` words with the value it is passed.

    Not a loop in the core's main: that keeps the object FIFO lowering from
    unrolling the calls, and the buffer selects it leaves spill into main's
    frame, past the stack the contracts measured. The value is an argument
    because a constant fill becomes a memset libcall, whose stack aiecc
    cannot measure.
    """
    name = f"kd_poison_{words}"
    return ExternalFunction(
        name,
        source_string=f"""extern "C" void {name}(int *tile, int value) {{
  for (int i = 0; i < {words}; i++)
    tile[i] = value;
}}""",
        arg_types=[np.ndarray[(words,), np.dtype[np.int32]], np.int32],
        use_chess=use_chess,
    )


def _view(raw, arg_type, byte_shift):
    return memref.view(
        np_ndarray_type_to_memref_type(arg_type),
        raw,
        constant(byte_shift, index=True),
        [],
    )


def _tensor_positions(fn):
    c = _contract(fn)
    types = fn.arg_types()
    return [
        i for i in c.reference_indices() if _is_tensor_type(types[i])
    ], c.out_indices


def _layout(c, i):
    return c.layouts[i] if c.layouts else None


def _out_tiles(fn, calls):
    """Output tiles a run drains: one per call, or one for all of them."""
    c = _contract(fn)
    if c.out_offset is None:
        return calls
    step = c.out_offset[1]
    n = elems(fn.arg_types()[c.out_index])
    if n != calls * step:
        raise ValueError(
            f"{fn.name}: {calls} call(s) of {step} element(s) each must fill "
            f"the {n}-element output tile"
        )
    return 1


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
    """Encode the unbound tensor Params, in argument order, as design constants."""
    c = _contract(fn)
    params = list(params)
    _, _, positions = _fifo_plan(fn)
    bound = dict(c.parameter_bindings)
    free = [i for i in positions if i not in bound]
    if len(params) != len(free):
        raise ValueError(f"{fn.name}: expected {len(free)} param value(s)")
    bound.update(zip(free, params))
    encoded = []
    for i in positions:
        value = bound[i]
        shape, dt = shape_dtype(fn.arg_types()[i])
        layout = _layout(c, i)
        value = layout.encode(value) if layout else np.asarray(value)
        if value.size != int(np.prod(shape)):
            raise ValueError(f"{fn.name}: param {i} must contain {shape} elements")
        encoded.append(
            (np.dtype(dt).name, shape, tuple(value.astype(dt).ravel().tolist()))
        )
    return tuple(encoded)


def _stage(fn, calls, scalars, params, stack_bytes, guard=False):
    """Plan the Worker: fifos per input group and output, buffers per Param, the call itself."""
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
    if len(scalars) != len(free):
        raise ValueError(
            f"{fn.name}: expected {len(free)} scalar(s), got {len(scalars)}"
        )
    if any(not isinstance(v, (int, float, np.integer, np.floating)) for v in scalars):
        raise ValueError(f"{fn.name}: expected scalar parameter values")
    bound.update(zip(free, scalars))
    if len(params) != len(param_pos):
        raise ValueError(f"{fn.name}: expected {len(param_pos)} param values")
    initializers = [(i, init(fn)) for i, init in c.initializers]
    # An InOut is read back on every independent call, so it needs a
    # declared initializer.
    if any(r is InOut and i not in dict(initializers) for i, r in enumerate(c.roles)):
        raise ValueError(f"{fn.name}: every InOut requires a declared initializer")
    setter = c.setup() if c.setup else None
    offset = c.out_offset
    _out_tiles(fn, calls)

    def nbytes(i):
        return elems(types[i]) * bfp.itemsize(shape_dtype(types[i])[1])

    guarded = _guarded(fn, guard)

    def fifo_type(i):
        if i not in guarded:
            return types[i]
        return np.ndarray[(nbytes(i) + GUARD_BYTES,), np.dtype[np.int8]]

    def typed(outputs):
        return [
            _view(o, types[i], 0) if i in guarded else o for i, o in zip(outs, outputs)
        ]

    # Two sets of tiles (ping-pong) when they fit beside the parameters and
    # the stack, one otherwise.
    tile_bytes = sum(nbytes(i) for i in [*ins, *outs]) + GUARD_BYTES * len(guarded)
    fixed_bytes = sum(nbytes(i) for i in param_pos) + stack_bytes
    core_bytes = _device().core_memory_bytes
    depth = next(
        (d for d in (2, 1) if d * tile_bytes + fixed_bytes <= core_bytes), None
    )
    if depth is None:
        raise ValueError(
            f"{fn.name}: tiles plus parameters and stack exceed core memory; use a smaller tile"
        )
    fifos_in = [
        ObjectFifo(types[g[0]], name=f"in{j}", depth=len(g) * depth)
        for j, g in enumerate(groups)
    ]
    fifos_out = [
        ObjectFifo(fifo_type(i), name=f"out{j}", depth=depth)
        for j, i in enumerate(outs)
    ]
    buffers = [
        Buffer(
            types[i],
            name=f"param{j}",
            initial_value=np.array(vals, dtype=np.dtype(dt)).reshape(shape),
        )
        for j, (i, (dt, shape, vals)) in enumerate(zip(param_pos, params))
    ]
    words = {i: (nbytes(i) + GUARD_BYTES) // 4 for i in guarded}
    fills = {n: _poison_fill(n, fn.use_chess) for n in words.values()}
    # The Worker's constants: the param buffers, the kernel, the
    # initializer kernels, the poison fills and the setup callable.
    n_param, n_init = len(buffers), len(initializers)
    slot = {i: j for j, i in enumerate(outs)}

    def body(acquired, outputs, _held, constants, call):
        values = dict(zip(param_pos, constants[:n_param]))
        values.update(bound)
        if offset:
            values[offset[0]] = call * offset[1]
        for got, group in zip(acquired, groups):
            values.update(
                (i, got if len(group) == 1 else got[j]) for j, i in enumerate(group)
            )
        values.update(zip(outs, typed(outputs)))
        constants[n_param](*(values[i] for i in range(len(c.roles))))

    def initialize(outputs, constants):
        for i, raw in zip(outs, outputs):
            if i in guarded:
                # The tile too, not just the guard: the DMA drains what the
                # core holds, so an element the kernel skips would otherwise
                # read back as zero or as an earlier call's value.
                fill = constants[n_param + 1 + n_init + list(fills).index(words[i])]
                tile = _view(raw, fill.arg_types()[0], 0)
                fill(tile, int(poisoned(1, np.int32)[0]))
        if initializers:
            views = typed(outputs)
            kernels = constants[n_param + 1 : n_param + 1 + n_init]
            for (i, _), init in zip(initializers, kernels):
                init(views[slot[i]])

    return Stage(
        body,
        inputs=[(fifo, len(g)) for fifo, g in zip(fifos_in, groups)],
        outputs=fifos_out,
        constants=buffers
        + [fn]
        + [init for _, init in initializers]
        + list(fills.values())
        + ([setter] if setter else []),
        iterations=calls,
        outputs_span_iterations=offset is not None,
        prologue=(lambda constants: constants[-1]()) if setter else None,
        initialize=initialize if initializers or guarded else None,
        stack_size=stack_bytes,
    )


def _build_stream(
    *,
    factory,
    factory_kwargs,
    calls,
    stack_bytes,
    scalars=(),
    params=(),
    trace_config=None,
    guard=False,
):
    fn = factory(**factory_kwargs)
    stage = _stage(fn, calls, tuple(scalars), params, stack_bytes, guard)
    stage.trace = trace_config is not None
    stage.trace_flush = TRACE_FLUSH
    types = fn.arg_types()

    guarded = _guarded(fn, guard)

    def host_ty(i, repetitions):
        n = elems(types[i]) + (_guard_elems(types[i]) if i in guarded else 0)
        return np.ndarray[(n * repetitions,), np.dtype[shape_dtype(types[i])[1]]]

    groups = _fifo_plan(fn)[0]
    host_types = [host_ty(g[0], calls * len(g)) for g in groups]
    host_types += [host_ty(i, _out_tiles(fn, calls)) for i in fn.contract.out_indices]
    fifos_in = [fifo for fifo, _ in stage.inputs]
    transfers = [(fifo, "fill", j) for j, fifo in enumerate(fifos_in)]
    transfers += [
        (fifo, "drain", len(fifos_in) + j) for j, fifo in enumerate(stage.outputs)
    ]
    return pipeline(
        [stage],
        host_types,
        transfers,
        trace_size=trace_config.trace_size if trace_config else 0,
        # cycles_per_call reads only the markers; the default events add an
        # INSTR_VECTOR per vector op, which filled a 64 KB buffer mid-run.
        coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1],
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
    stack_bytes: CompileTime[int],
    scalars: CompileTime[tuple] = (),
    params: CompileTime[tuple] = (),
    trace_config: CompileTime[TraceConfig | None] = None,
    guard: CompileTime[bool] = False,
):
    return _build_stream(
        factory=factory,
        factory_kwargs=factory_kwargs,
        calls=calls,
        stack_bytes=stack_bytes,
        scalars=scalars,
        params=params,
        trace_config=trace_config,
        guard=guard,
    )


def design(
    factory,
    *,
    calls=1,
    scalars=(),
    shape=None,
    params=None,
    aiecc_flags=None,
    guard=False,
    stack_bytes=None,
    **factory_kwargs,
):
    """Wrap tile calls; ``params``/``scalars`` supply unbound tensor/scalar Params.

    This harness embeds these values for every call; changing them recompiles
    the design. Direct designs can supply different operands on each call.

    ``stack_bytes`` replaces the core stack the contract declares, for a
    kernel built from sources other than the ones the contract was sized for.

    With ``guard=True`` the core fills each output tile and ``GUARD_BYTES``
    after it with ``0x55`` before every call and drains the guard with the
    tile, so a kernel that skips part of its output or writes past it shows
    up on the host:
    size the outputs with ``output_size(..., guard=True)`` and split them
    with ``strip_guard``. bfp outputs carry no guard.
    """
    calls = _calls(calls, shape)
    _device()
    fn = factory(**factory_kwargs)
    c = _contract(fn)
    if c.unsupported:
        raise ValueError(
            f"{fn.name}: the generic harness cannot build this kernel: {c.unsupported}"
        )
    _fifo_plan(fn)  # the DMA channel budget is checked before anything builds
    flags: list[str] = list(aiecc_flags or ())
    if any(bfp.is_bfp(shape_dtype(t)[1]) for t in fn.arg_types() if _is_tensor_type(t)):
        if "--dynamic-objFifos" not in flags:
            flags.append("--dynamic-objFifos")
    # A gather reads its two tables at once, so they have to sit in different
    # banks. Chess cannot be checked (the flag needs Peano LLVM IR), and an
    # arch whose path has no LUT just finds nothing to check.
    if c.uses_lut and not fn.use_chess:
        if not any(f.startswith("--check-lut-banks") for f in flags):
            flags.append("--check-lut-banks")
    return _stream.specialize(
        factory=factory,
        factory_kwargs=factory_kwargs,
        calls=calls,
        # A key of its own: the contract that sets it can change in a module
        # the cache key never reads, and a stale stack overflows silently.
        stack_bytes=stack_bytes or _stack_bytes(fn),
        scalars=tuple(scalars),
        params=_encode_params(fn, params or ()),
        guard=guard,
        **({"aiecc_flags": flags} if flags else {}),
    )


_INT_RANGE = {np.int8: 60, np.int16: 8000, np.int32: 1 << 20, np.uint8: 255}


def sample_inputs(fn, *, calls=1, shape=None, rng=None):
    """One array per unbound In/tensor Param; only In has a call dimension."""
    calls = _calls(calls, shape)
    rng = np.random.default_rng(0) if rng is None else rng
    c = _contract(fn)
    if c.sample is not None:
        return c.sample(rng, calls)
    result = []
    for i in _tensor_positions(fn)[0]:
        s, dt = shape_dtype(fn.arg_types()[i])
        layout = _layout(c, i)
        s = layout.shape if layout else (int(np.prod(s)),)
        dt = np.float32 if bfp.is_bfp(dt) else dt
        s = ((calls,) if c.roles[i] is In else ()) + s
        if np.issubdtype(np.dtype(dt), np.integer):
            r = fn.input_limit(dt) or _INT_RANGE.get(dt, 1 << 15)
            lo = 0 if np.dtype(dt).kind == "u" else -r
            result.append(rng.integers(lo, r, size=s).astype(dt))
        else:
            result.append(rng.standard_normal(s).astype(np.float32).astype(dt))
    return result


def host_layout(fn, inputs):
    """Pack declared layouts and interleave same-type In tiles; omit Param buffers."""
    c = _contract(fn)
    positions = _tensor_positions(fn)[0]
    if len(inputs) != len(positions):
        raise ValueError(f"{fn.name}: expected {len(positions)} input arrays")
    values = dict(zip(positions, inputs))
    result = []
    for group in _fifo_plan(fn)[0]:
        arrays = []
        for i in group:
            value = np.asarray(values[i])
            layout = _layout(c, i)
            arrays.append(
                layout.encode(value)
                if layout
                else value.reshape(-1, elems(fn.arg_types()[i]))
            )
        result.append(
            np.ascontiguousarray(
                arrays[0] if len(arrays) == 1 else np.stack(arrays, axis=1)
            )
        )
    return result


def output_size(fn, *, calls=1, shape=None, guard=False):
    """Storage elements per output; a tuple for multiple outputs."""
    calls = _calls(calls, shape)
    types = fn.arg_types()
    guarded = _guarded(fn, guard)
    sizes = tuple(
        (elems(types[i]) + (_guard_elems(types[i]) if i in guarded else 0))
        * _out_tiles(fn, calls)
        * (bfp.BLOCK_BYTES if bfp.is_bfp(shape_dtype(types[i])[1]) else 1)
        for i in _contract(fn).out_indices
    )
    return sizes[0] if len(sizes) == 1 else sizes


def strip_guard(fn, outputs, *, calls=1):
    """Split the outputs of a ``guard=True`` design into data and overrun.

    Returns the outputs without their guards, shaped as ``output_size``
    without ``guard`` would size them, and per output the number of guard
    bytes the kernel changed.
    """
    calls = _calls(calls)
    multiple = isinstance(outputs, tuple)
    guarded = _guarded(fn, True)
    data, overrun = [], []
    for i, out in zip(_contract(fn).out_indices, outputs if multiple else (outputs,)):
        if i not in guarded:
            data.append(out)
            overrun.append(0)
            continue
        rows = np.ascontiguousarray(out).reshape(_out_tiles(fn, calls), -1)
        raw = rows.view(np.uint8)
        n = raw.shape[1] - GUARD_BYTES
        overrun.append(int(np.count_nonzero(raw[:, n:] != 0x55)))
        data.append(np.ascontiguousarray(raw[:, :n]).view(out.dtype).reshape(-1))
    if multiple:
        return tuple(data), tuple(overrun)
    return data[0], overrun[0]


@dataclass(frozen=True)
class _HostBuffer:
    """A physical host buffer in design argument order."""

    direction: type
    shape: tuple[int, ...]
    dtype: type

    @property
    def n_elements(self):
        return int(np.prod(self.shape))


def host_args(fn, *, calls=1, shape=None, guard=False):
    """Describe physical host buffers, not kernel arguments or constant Params.

    Same-type streamed inputs share a buffer; each output has its own.
    Descriptors expose ``direction``, ``shape``, ``dtype`` and ``n_elements``.
    """
    calls = _calls(calls, shape)
    types = fn.arg_types()
    guarded = _guarded(fn, guard)
    result = []
    entries = [(In, g) for g in _fifo_plan(fn)[0]]
    entries += [(Out, [i]) for i in _contract(fn).out_indices]
    for direction, indices in entries:
        i = indices[0]
        dt = shape_dtype(types[i])[1]
        n = elems(types[i])
        if bfp.is_bfp(dt):
            n, dt = n * bfp.BLOCK_BYTES, np.uint8
        elif direction is Out and i in guarded:
            n += _guard_elems(types[i])
        rows = calls if direction is In else _out_tiles(fn, calls)
        s = (rows, n) if len(indices) == 1 else (rows, len(indices), n)
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


@dataclass(frozen=True)
class CallCycles:
    """One traced run's intervals, split by the kernel that emitted them.

    ``kernel`` holds one interval per call of the measured kernel, in call
    order; ``initializers`` the same for each traced initializer, keyed by
    the ``InOut`` argument it initializes; ``setup`` the setup kernel's one
    interval when it is traced. A trace that fills its buffer keeps a prefix
    of the stream, which the split still labels correctly, and ``truncated``
    says the lists are short. ``untimed`` is the contract's reason when the
    kernel's markers do not bracket its calls; then nothing ran.
    """

    kernel: tuple[int, ...] = ()
    initializers: dict[int, tuple[int, ...]] = field(default_factory=dict)
    setup: tuple[int, ...] = ()
    truncated: bool = False
    untimed: str | None = None


def _timed(kernel, role: str) -> bool:
    trace = kernel.contract.trace if kernel.contract else None
    if trace is None:
        raise ValueError(f"{kernel.name} ({role}) declares no trace in its contract")
    if trace.shape == "partial":
        raise ValueError(
            f"{kernel.name} ({role}): {trace.reason}, so its intervals cannot be "
            "told apart from the measured kernel's"
        )
    return trace.shape == "whole_call"


def _traced(fn):
    """Return ``(setup traced, [traced initializer argument indices])``, checked."""
    c = _contract(fn)
    setup = bool(c.setup) and _timed(c.setup(), "setup")
    inits = [i for i, init in c.initializers if _timed(init(fn), f"initializer {i}")]
    return setup, inits


def traced_intervals(fn, *, calls=1) -> int:
    """How many ``event0``/``event1`` intervals a traced run of ``fn`` emits.

    A ``trace_size`` that holds fewer truncates the run; ``0`` when the
    kernel itself is not timed.
    """
    trace = _contract(fn).trace
    if trace is None or trace.shape != "whole_call":
        return 0
    setup, inits = _traced(fn)
    return int(setup) + _calls(calls) * (len(inits) + 1)


def split_intervals(durations, *, calls, per_call, setup=0, flush=0):
    """Label an interval stream: ``setup`` intervals, then ``per_call`` per call.

    The harness runs the setup kernel once, then each call runs its traced
    initializers in contract order and the kernel last, so interval ``j`` of
    call ``n`` sits at ``setup + n * per_call + j``. Returns ``(setup
    intervals, one tuple per per-call kernel, truncated)``. Up to ``flush``
    intervals of at most ``FLUSH_CYCLES`` may follow, the pairs the core
    emits to push the trace out. Anything else is some kernel emitting
    markers it does not declare; one whose extra intervals are that short
    and that few passes as a flush.
    """
    durations = [int(d) for d in durations]
    expected = setup + calls * per_call
    tail = durations[expected:]
    if len(tail) > flush or any(d > FLUSH_CYCLES for d in tail):
        raise RuntimeError(
            f"expected {expected} trace intervals and up to {flush} flush pairs, "
            f"got {len(durations)}, the extra {tail[:flush + 1]} cycles; a "
            "kernel on the core emits markers its contract's trace does not "
            "declare"
        )
    durations = durations[:expected]
    head, stream = durations[:setup], durations[setup:]
    return (
        tuple(head),
        [tuple(stream[j::per_call]) for j in range(per_call)],
        len(durations) < expected,
    )


def cycles_per_call(
    design_, inputs, out_size, out_dtype, *, fn, trace_size, workdir, calls=1
) -> CallCycles:
    """Trace one run and split its intervals by kernel (``CallCycles``).

    Only a kernel whose contract declares ``Trace.whole_call()`` is timed;
    one declaring ``none`` or ``partial`` returns its reason without running,
    and one declaring nothing raises. Traced initializers and a traced setup
    kernel are split off by position; a ``partial`` one raises, because its
    intervals cannot be labeled. Size ``trace_size`` from
    ``traced_intervals``.
    """
    c = _contract(fn)
    if c.trace is None:
        raise ValueError(f"{fn.name}: the contract declares no trace")
    if c.trace.shape != "whole_call":
        return CallCycles(untimed=c.trace.reason)
    calls = _calls(calls)
    setup, inits = _traced(fn)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = TraceConfig(trace_size=trace_size, trace_file=str(workdir / "trace.txt"))
    ins, out = upload(inputs, out_size, out_dtype, fn=fn)
    design_(*ins, *(out if isinstance(out, tuple) else (out,)), trace_config=cfg)
    if cfg.physical_mlir_path is None:
        raise RuntimeError("the traced run recorded no physical MLIR path")
    trace_json = workdir / "trace.json"
    cfg.trace_to_json(cfg.physical_mlir_path, str(trace_json))
    durations = [d for p in get_cycles_summary(str(trace_json)) for d in p[1:]]
    head, per_kernel, truncated = split_intervals(
        durations,
        calls=calls,
        per_call=len(inits) + 1,
        setup=int(setup),
        flush=TRACE_FLUSH,
    )
    if not per_kernel[-1]:
        raise RuntimeError(
            f"{fn.name}: the trace holds {len(durations)} intervals and none of "
            "the kernel's; the buffer is too small or the markers are missing"
        )
    return CallCycles(
        kernel=per_kernel[-1],
        initializers=dict(zip(inits, per_kernel)),
        setup=head,
        truncated=truncated,
    )


__all__ = [
    "CallCycles",
    "cycles_per_call",
    "design",
    "elems",
    "host_layout",
    "host_args",
    "output_size",
    "sample_inputs",
    "shape_dtype",
    "split_intervals",
    "traced_intervals",
    "upload",
]
