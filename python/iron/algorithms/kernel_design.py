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
from functools import cached_property
from operator import index
from pathlib import Path

import numpy as np
from aie.dialects import arith, memref  # pyright: ignore[reportAttributeAccessIssue]
from aie.extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
    constant,
)
from aie.helpers.npdtypes import np_ndarray_type_get_dtype, np_ndarray_type_get_shape
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.ir import (  # pyright: ignore[reportMissingImports]
    IndexType,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.iron.buffer import Buffer
from aie.iron.dataflow import ObjectFifo
from aie.iron.kernel import ExternalFunction
from aie.iron.kernels._common import Param, _is_tensor_type
from aie.utils import bfp, ensure_current_device, tensor
from aie.utils.compile import resolve_target_arch
from aie.utils.compile.jit import CompileTime, In, InOut, Out, get_compile_arg
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
# A Param bound at a byte offset sits in a buffer this much larger.
VIEW_PAD = 64


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


def _guard_elems(arg_type):
    return GUARD_BYTES // np.dtype(shape_dtype(arg_type)[1]).itemsize


def _poison_fill(words, use_chess):
    """Return a kernel that fills ``words`` words with the value it is passed.

    Not a loop in the core's main: that keeps the object FIFO lowering from
    unrolling the calls, and the buffer selects it leaves spill into main's
    frame and grow the core's stack. The value is an argument
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


def _kernel_frame_bytes(kernels, embed_bitcode):
    """Return what ``kernels`` hold in the core beyond the core's own frame.

    Each object's static data (``ExternalFunction.frame_bytes``' data), which
    all coexists, and the deepest kernel stack, as only one call is live: a
    declared ``stack_size_override`` stands. An object-linked Peano kernel is
    built through the design's object cache, which its own build then hits,
    or in a scratch directory when caching is off. A kernel ``measure``
    cannot read counts as 0, as aiecc cannot measure it either; that
    undercount only picks too deep a buffering for the tile, and aiecc's
    allocator then fails the build.
    """
    cache = get_compile_arg("_iron_object_cache")
    arch = resolve_target_arch(_device())
    data, deepest = {}, 0
    for k in kernels:
        found = (
            k.measure(arch, cache, embed_bitcode)
            if isinstance(k, ExternalFunction)
            else None
        )
        if found is not None:
            data[k.object_file_name] = found.data
        if k.stack_size_override is not None:
            deepest = max(deepest, k.stack_size_override)
        elif found is not None:
            deepest = max(deepest, found.stack or 0)
    return sum(data.values()) + deepest


def _view(raw, arg_type, byte_shift):
    return memref.view(
        np_ndarray_type_to_memref_type(arg_type),
        raw,
        constant(byte_shift, index=True),
        [],
    )


def _layout(c, i):
    return c.layouts[i] if c.layouts else None


_INT_RANGE = {np.int8: 60, np.int16: 8000, np.int32: 1 << 20, np.uint8: 255}


@dataclass(frozen=True)
class _HostBuffer:
    """A physical host buffer in design argument order."""

    direction: type
    shape: tuple[int, ...]
    dtype: type

    @property
    def n_elements(self):
        return int(np.prod(self.shape))


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


class KernelDesign:
    """``calls`` independent calls of one contract-bearing kernel.

    The configuration is fixed at construction, and every method sizes,
    lays out, builds and checks from it; construct another to change it.

    ``scalars`` supply the unbound scalar Params. The unbound tensor Params
    are embedded in the design: ``run`` takes them from its logical inputs,
    and ``design(params)`` from ``params``. Changing them recompiles the
    design. Direct designs can supply different operands on each call.

    ``stack_bytes`` fixes the core stack. Leave it unset and aiecc sizes the
    stack from the linked core; set it where aiecc cannot measure, as for a
    Chess build. Unset, choosing between one and two sets of tiles reads the
    kernels' stack frames, so generating a design whose tiles could double
    buffer compiles its kernels.

    With ``guard=True`` the core fills each output tile and ``GUARD_BYTES``
    after it with ``0x55`` before every call and drains the guard with the
    tile, so a kernel that skips part of its output or writes past it shows
    up on the host: ``output_size`` and ``host_args`` count the guards,
    ``strip_guard`` splits them off and ``run`` raises when one changed.
    bfp outputs carry no guard.

    ``arg_byte_offsets`` binds tensor Params at a byte offset: ``((1, 16),)``
    hands argument 1 a view 16 bytes into a buffer ``VIEW_PAD`` larger,
    poisoned around the data, as a design that packs several weights into
    one buffer does. The shift is loaded at run time, so the kernel sees an
    address no design-time check could have folded.
    """

    def __init__(
        self,
        fn,
        *,
        calls=1,
        shape=None,
        guard=False,
        scalars=(),
        stack_bytes=None,
        arg_byte_offsets=(),
        aiecc_flags=None,
    ):
        if fn.contract is None:
            raise ValueError(
                f"kernel '{fn.name}' declares no contract; add a KernelContract"
            )
        fn.contract.validate_types(fn.arg_types())
        self.fn = fn
        self.calls = _calls(calls, shape)
        self.guard = guard
        self.scalars = tuple(scalars)
        self.stack_bytes = stack_bytes
        self.aiecc_flags = tuple(aiecc_flags or ())
        self.arg_byte_offsets = self._byte_offsets(arg_byte_offsets)

    @cached_property
    def _types(self):
        return self.fn.arg_types()

    @cached_property
    def _guarded(self):
        """The outputs ``guard`` covers: every one but bfp."""
        return [
            i
            for i in self.fn.contract.out_indices
            if self.guard and not bfp.is_bfp(shape_dtype(self._types[i])[1])
        ]

    @cached_property
    def _tensor_positions(self):
        return [
            i
            for i in self.fn.contract.reference_indices()
            if _is_tensor_type(self._types[i])
        ]

    @cached_property
    def _out_tiles(self):
        """Output tiles a run drains: one per call, or one for all of them."""
        c = self.fn.contract
        if c.out_last:
            if self.calls != c.required_calls:
                raise ValueError(
                    f"{self.fn.name}: out_last requires exactly {c.required_calls} "
                    f"call(s), got {self.calls}"
                )
            return 1
        if c.out_offset is None:
            return self.calls
        step = c.out_offset[1]
        n = elems(self._types[c.out_index])
        if n != self.calls * step:
            raise ValueError(
                f"{self.fn.name}: {self.calls} call(s) of {step} element(s) each "
                f"must fill the {n}-element output tile"
            )
        return 1

    @cached_property
    def _tensor_params(self):
        return [
            i
            for i, r in enumerate(self.fn.contract.roles)
            if r is Param and _is_tensor_type(self._types[i])
        ]

    @cached_property
    def _fifo_plan(self):
        """Pack same-type inputs per call, respecting the core's DMA channel budget."""
        c, types = self.fn.contract, self._types
        ins = [i for i, r in enumerate(c.roles) if r is In]
        by_type = {}
        for i in ins:
            by_type.setdefault(types[i], []).append(i)
        groups = list(by_type.values())
        if len(groups) > _device().core_dma_channels_in:
            raise ValueError(
                f"{self.fn.name}: input types exceed the core's input DMA channels"
            )
        if len(c.out_indices) > _device().core_dma_channels_out:
            raise ValueError(
                f"{self.fn.name}: outputs exceed the core's output DMA channels"
            )
        return groups, ins

    def _encode_params(self, params):
        """Encode the unbound tensor Params, in argument order, as design constants."""
        c = self.fn.contract
        params = list(params)
        bound = dict(c.parameter_bindings)
        free = [i for i in self._tensor_params if i not in bound]
        if len(params) != len(free):
            raise ValueError(f"{self.fn.name}: expected {len(free)} param value(s)")
        bound.update(zip(free, params))
        encoded = []
        for i in self._tensor_params:
            value = bound[i]
            shape, dt = shape_dtype(self._types[i])
            layout = _layout(c, i)
            value = layout.encode(value) if layout else np.asarray(value)
            if value.size != int(np.prod(shape)):
                raise ValueError(
                    f"{self.fn.name}: param {i} must contain {shape} elements"
                )
            encoded.append(
                (np.dtype(dt).name, shape, tuple(value.astype(dt).ravel().tolist()))
            )
        return tuple(encoded)

    def _byte_offsets(self, arg_byte_offsets):
        """Check ``(argument, byte offset)`` pairs.

        Each argument must be a tensor Param and each offset element-aligned and
        below ``VIEW_PAD``.
        """
        result = {}
        for i, offset in arg_byte_offsets:
            if i not in self._tensor_params:
                raise ValueError(
                    f"{self.fn.name}: argument {i} is not a tensor Param; only those "
                    "can be bound at a byte offset"
                )
            itemsize = np.dtype(shape_dtype(self._types[i])[1]).itemsize
            if not 0 <= offset < VIEW_PAD or offset % itemsize:
                raise ValueError(
                    f"{self.fn.name}: argument {i}'s byte offset must be a multiple "
                    f"of {itemsize} below {VIEW_PAD}, not {offset}"
                )
            result[i] = offset
        return tuple(sorted(result.items()))

    def _nbytes(self, i):
        return elems(self._types[i]) * bfp.itemsize(shape_dtype(self._types[i])[1])

    def _typed(self, outputs):
        return [
            _view(o, self._types[i], 0) if i in self._guarded else o
            for i, o in zip(self.fn.contract.out_indices, outputs)
        ]

    def _param_buffer(self, j, i, dt, shape, vals):
        value = np.array(vals, dtype=np.dtype(dt)).reshape(shape)
        shifted = dict(self.arg_byte_offsets)
        if i not in shifted:
            return Buffer(self._types[i], name=f"param{j}", initial_value=value)
        # Poison around the data, so a load rounded down to an aligned
        # address reads something no reference expects.
        raw = poisoned(self._nbytes(i) + VIEW_PAD, np.int8)
        raw[shifted[i] : shifted[i] + self._nbytes(i)] = value.view(np.int8).ravel()
        return Buffer(
            np.ndarray[raw.shape, np.dtype[np.int8]],
            name=f"param{j}",
            initial_value=raw,
        )

    @cached_property
    def _scalar_values(self):
        """Every scalar Param's value, by argument: bound ones, then ``scalars``."""
        c, types = self.fn.contract, self._types
        bound = {
            i: value
            for i, value in c.parameter_bindings
            if not _is_tensor_type(types[i])
        }
        free = [
            i
            for i, r in enumerate(c.roles)
            if r is Param and not _is_tensor_type(types[i]) and i not in bound
        ]
        if len(self.scalars) != len(free):
            raise ValueError(
                f"{self.fn.name}: expected {len(free)} scalar(s), "
                f"got {len(self.scalars)}"
            )
        if any(
            not isinstance(v, (int, float, np.integer, np.floating))
            for v in self.scalars
        ):
            raise ValueError(f"{self.fn.name}: expected scalar parameter values")
        bound.update(zip(free, self.scalars))
        return bound

    @cached_property
    def _initializers(self):
        initializers = [(i, init(self.fn)) for i, init in self.fn.contract.initializers]
        # An InOut is read back on every independent call, so it needs a
        # declared initializer.
        if any(
            r is InOut and i not in dict(initializers)
            for i, r in enumerate(self.fn.contract.roles)
        ):
            raise ValueError(
                f"{self.fn.name}: every InOut requires a declared initializer"
            )
        return initializers

    @cached_property
    def _setter(self):
        return self.fn.contract.setup() if self.fn.contract.setup else None

    def _stage(self, params):
        """Plan the Worker: fifos per input group and output, buffers per Param, the call itself."""
        c, types = self.fn.contract, self._types
        groups, ins = self._fifo_plan
        param_pos = self._tensor_params
        outs = c.out_indices
        if len(params) != len(param_pos):
            raise ValueError(f"{self.fn.name}: expected {len(param_pos)} param values")
        bound = self._scalar_values
        initializers = self._initializers
        setter = self._setter
        offset = c.out_offset
        guarded = self._guarded
        words = {i: (self._nbytes(i) + GUARD_BYTES) // 4 for i in guarded}
        fills = {n: _poison_fill(n, self.fn.use_chess) for n in words.values()}

        # Two sets of tiles (ping-pong) when they fit beside the parameters, the
        # kernels' static data and the stack, one otherwise. aiecc reserves the
        # larger of the default stack and main's frame plus the deepest kernel
        # call; the default covers main's frame, so the kernels' go on top of it.
        tile_bytes = sum(self._nbytes(i) for i in [*ins, *outs])
        tile_bytes += GUARD_BYTES * len(guarded)
        shifted = dict(self.arg_byte_offsets)
        stack_bytes = self.stack_bytes
        fixed_bytes = (
            sum(self._nbytes(i) for i in param_pos)
            + (VIEW_PAD + 4) * len(shifted)
            + (stack_bytes or _device().default_core_stack_bytes)
        )
        core_bytes = _device().core_memory_bytes
        if not stack_bytes and 2 * tile_bytes + fixed_bytes <= core_bytes:
            fixed_bytes += _kernel_frame_bytes(
                [
                    self.fn,
                    *(k for _, k in initializers),
                    *fills.values(),
                    *([setter] if setter else []),
                ],
                c.uses_lut and not self.fn.use_chess,
            )
        depth = next(
            (d for d in (2, 1) if d * tile_bytes + fixed_bytes <= core_bytes), None
        )
        if depth is None:
            raise ValueError(
                f"{self.fn.name}: tiles plus parameters and stack exceed core "
                "memory; use a smaller tile"
            )
        fifos_in = [
            ObjectFifo(types[g[0]], name=f"in{j}", depth=len(g) * depth)
            for j, g in enumerate(groups)
        ]
        fifos_out = [
            ObjectFifo(
                (
                    np.ndarray[(self._nbytes(i) + GUARD_BYTES,), np.dtype[np.int8]]
                    if i in guarded
                    else types[i]
                ),
                name=f"out{j}",
                depth=depth,
            )
            for j, i in enumerate(outs)
        ]
        buffers = [
            self._param_buffer(j, i, *param)
            for j, (i, param) in enumerate(zip(param_pos, params))
        ]
        # The shifts are loaded at run time, as a caller outside IRON would
        # compute them, so no fold sees a constant offset.
        shifts = (
            [
                Buffer(
                    np.ndarray[(len(shifted),), np.dtype[np.int32]],
                    name="param_shift",
                    initial_value=np.array(list(shifted.values()), dtype=np.int32),
                )
            ]
            if shifted
            else []
        )
        # The Worker's constants: the param buffers, the kernel, the
        # initializer kernels, the poison fills, the view shifts and the setup
        # callable.
        n_param, n_init = len(buffers), len(initializers)
        slot = {i: j for j, i in enumerate(outs)}
        n_shift = n_param + 1 + n_init + len(fills)

        def body(acquired, outputs, _held, constants, call):
            values = dict(zip(param_pos, constants[:n_param]))
            for k, i in enumerate(shifted):
                shift = arith.index_cast(IndexType.get(), constants[n_shift][k])
                values[i] = memref.view(
                    np_ndarray_type_to_memref_type(types[i]), values[i].op, shift, []
                )
            values.update(bound)
            if offset:
                values[offset[0]] = call * offset[1]
            if c.call_index is not None:
                values[c.call_index] = call
            for got, group in zip(acquired, groups):
                values.update(
                    (i, got if len(group) == 1 else got[j]) for j, i in enumerate(group)
                )
            values.update(zip(outs, self._typed(outputs)))
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
                views = self._typed(outputs)
                kernels = constants[n_param + 1 : n_param + 1 + n_init]
                for (i, _), init in zip(initializers, kernels):
                    init(views[slot[i]])

        return Stage(
            body,
            inputs=[(fifo, len(g)) for fifo, g in zip(fifos_in, groups)],
            outputs=fifos_out,
            constants=buffers
            + [self.fn]
            + [init for _, init in initializers]
            + list(fills.values())
            + shifts
            + ([setter] if setter else []),
            iterations=self.calls,
            outputs_span_iterations=offset is not None or c.out_last,
            prologue=(lambda constants: constants[-1]()) if setter else None,
            initialize=initialize if initializers or guarded else None,
            stack_size=stack_bytes,
        )

    def _build_stream(self, params, trace_config):
        stage = self._stage(params)
        stage.trace = trace_config is not None
        stage.trace_flush = TRACE_FLUSH
        types = self._types

        def host_ty(i, repetitions):
            n = elems(types[i]) + (_guard_elems(types[i]) if i in self._guarded else 0)
            return np.ndarray[(n * repetitions,), np.dtype[shape_dtype(types[i])[1]]]

        groups = self._fifo_plan[0]
        host_types = [host_ty(g[0], self.calls * len(g)) for g in groups]
        host_types += [
            host_ty(i, self._out_tiles) for i in self.fn.contract.out_indices
        ]
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

    @cached_property
    def _plan(self):
        # A kernel's repr leaves out its contract; these are the contract
        # facts the build reads, so the design's cache key covers them.
        c = self.fn.contract
        kernels = [self.fn, *(k for _, k in self._initializers)]
        kernels += [self._setter] if self._setter else []
        return (
            tuple(r.__name__ for r in c.roles),
            tuple(
                sorted(
                    (i, value)
                    for i, value in c.parameter_bindings
                    if not _is_tensor_type(self._types[i])
                )
            ),
            tuple(i for i, _ in self._initializers),
            c.out_offset,
            c.call_index,
            c.out_last,
            c.uses_lut,
            c.alignments,
            tuple((repr(k), k.stack_size_override) for k in kernels),
        )

    def design(self, params=()):
        """Return the JIT design with ``params`` as its unbound tensor Params."""
        _device()
        c = self.fn.contract
        if c.unsupported:
            raise ValueError(
                f"{self.fn.name}: the generic harness cannot build this kernel: "
                f"{c.unsupported}"
            )
        self._out_tiles
        self._fifo_plan  # the DMA channel budget is checked before anything builds
        flags: list[str] = list(self.aiecc_flags)
        if any(
            bfp.is_bfp(shape_dtype(t)[1]) for t in self._types if _is_tensor_type(t)
        ):
            if "--dynamic-objFifos" not in flags:
                flags.append("--dynamic-objFifos")
        # A gather reads its two tables at once, so they have to sit in different
        # banks. Chess cannot be checked (the flag needs Peano LLVM IR), and an
        # arch whose path has no LUT just finds nothing to check.
        if c.uses_lut and not self.fn.use_chess:
            if not any(f.startswith("--check-lut-banks") for f in flags):
                flags.append("--check-lut-banks")
        return _stream.specialize(
            kernel=self.fn,
            plan=self._plan,
            calls=self.calls,
            scalars=self.scalars,
            params=self._encode_params(params),
            guard=self.guard,
            # Only when given, so every other design keeps its cache key.
            **({"stack_bytes": self.stack_bytes} if self.stack_bytes else {}),
            **(
                {"arg_byte_offsets": self.arg_byte_offsets}
                if self.arg_byte_offsets
                else {}
            ),
            **({"aiecc_flags": flags} if flags else {}),
        )

    def as_mlir(self):
        """Lower ``design()`` to MLIR."""
        return self.design().as_mlir()

    def sample_inputs(self, rng=None):
        """One array per unbound In/tensor Param; only In has a call dimension."""
        rng = np.random.default_rng(0) if rng is None else rng
        c = self.fn.contract
        if c.sample is not None:
            return c.sample(rng, self.calls)
        result = []
        for i in self._tensor_positions:
            s, dt = shape_dtype(self._types[i])
            layout = _layout(c, i)
            s = layout.shape if layout else (int(np.prod(s)),)
            dt = np.float32 if bfp.is_bfp(dt) else dt
            s = ((self.calls,) if c.roles[i] is In else ()) + s
            if np.issubdtype(np.dtype(dt), np.integer):
                r = self.fn.input_limit(dt) or _INT_RANGE.get(dt, 1 << 15)
                lo = 0 if np.dtype(dt).kind == "u" else -r
                result.append(rng.integers(lo, r, size=s).astype(dt))
            else:
                result.append(rng.standard_normal(s).astype(np.float32).astype(dt))
        return result

    def host_layout(self, inputs):
        """Pack declared layouts and interleave same-type In tiles; omit Param buffers."""
        c = self.fn.contract
        positions = self._tensor_positions
        if len(inputs) != len(positions):
            raise ValueError(f"{self.fn.name}: expected {len(positions)} input arrays")
        values = dict(zip(positions, inputs))
        result = []
        for group in self._fifo_plan[0]:
            arrays = []
            for i in group:
                value = np.asarray(values[i])
                layout = _layout(c, i)
                arrays.append(
                    layout.encode(value)
                    if layout
                    else value.reshape(-1, elems(self._types[i]))
                )
            result.append(
                np.ascontiguousarray(
                    arrays[0] if len(arrays) == 1 else np.stack(arrays, axis=1)
                )
            )
        return result

    def output_size(self):
        """Storage elements per output; a tuple for multiple outputs."""
        types = self._types
        sizes = tuple(
            (elems(types[i]) + (_guard_elems(types[i]) if i in self._guarded else 0))
            * self._out_tiles
            * (bfp.BLOCK_BYTES if bfp.is_bfp(shape_dtype(types[i])[1]) else 1)
            for i in self.fn.contract.out_indices
        )
        return sizes[0] if len(sizes) == 1 else sizes

    def strip_guard(self, outputs):
        """Split the outputs of a run into data and overrun.

        Returns the outputs without their guards, shaped as ``output_size``
        without ``guard`` would size them, and per output the number of guard
        bytes the kernel changed (0 for an unguarded output).
        """
        multiple = isinstance(outputs, tuple)
        data, overrun = [], []
        for i, out in zip(
            self.fn.contract.out_indices, outputs if multiple else (outputs,)
        ):
            if i not in self._guarded:
                data.append(out)
                overrun.append(0)
                continue
            rows = np.ascontiguousarray(out).reshape(self._out_tiles, -1)
            raw = rows.view(np.uint8)
            n = raw.shape[1] - GUARD_BYTES
            overrun.append(int(np.count_nonzero(raw[:, n:] != 0x55)))
            data.append(np.ascontiguousarray(raw[:, :n]).view(out.dtype).reshape(-1))
        if multiple:
            return tuple(data), tuple(overrun)
        return data[0], overrun[0]

    def host_args(self):
        """Describe physical host buffers, not kernel arguments or constant Params.

        Same-type streamed inputs share a buffer; each output has its own.
        Descriptors expose ``direction``, ``shape``, ``dtype`` and ``n_elements``.
        """
        types = self._types
        result = []
        entries = [(In, g) for g in self._fifo_plan[0]]
        entries += [(Out, [i]) for i in self.fn.contract.out_indices]
        for direction, indices in entries:
            i = indices[0]
            dt = shape_dtype(types[i])[1]
            n = elems(types[i])
            if bfp.is_bfp(dt):
                n, dt = n * bfp.BLOCK_BYTES, np.uint8
            elif direction is Out and i in self._guarded:
                n += _guard_elems(types[i])
            rows = self.calls if direction is In else self._out_tiles
            s = (rows, n) if len(indices) == 1 else (rows, len(indices), n)
            result.append(_HostBuffer(direction, s, dt))
        return result

    def upload(self, inputs, *, poison=False):
        """Return design inputs and output tensor(s); splat multiple outputs when calling."""
        packed = self.host_layout(inputs)
        args = self.host_args()
        for j, (a, arg) in enumerate(zip(packed, args)):
            if a.size != arg.n_elements or a.dtype != np.dtype(arg.dtype):
                raise ValueError(
                    f"{self.fn.name}: input buffer {j} holds {a.size} {a.dtype}; "
                    f"{self.calls} call(s) take {arg.n_elements} "
                    f"{np.dtype(arg.dtype)}"
                )
        ins = [tensor(a.reshape(-1), dtype=a.dtype, device="npu") for a in packed]
        outs = [
            tensor(
                (
                    poisoned(arg.n_elements, arg.dtype)
                    if poison
                    else np.zeros(arg.n_elements, dtype=arg.dtype)
                ),
                dtype=arg.dtype,
                device="npu",
            )
            for arg in args
            if arg.direction is Out
        ]
        return ins, tuple(outs) if len(outs) > 1 else outs[0]

    def run(self, inputs, *, trace_config=None):
        """Run on ``inputs`` into poisoned outputs and return them without guards.

        The tensor Params come from ``inputs``. The outputs are copies:
        ``Tensor.numpy()`` views the XRT buffer's mapped host memory, which
        dies with the run's tensors.

        Raises:
            RuntimeError: When the kernel changed a guard.
        """
        design = self.design(self.fn.param_values(inputs))
        ins, out = self.upload(inputs, poison=True)
        outs = out if isinstance(out, tuple) else (out,)
        design(
            *ins,
            *outs,
            **({"trace_config": trace_config} if trace_config else {}),
        )
        got = [o.numpy().copy() for o in outs]
        data, overrun = self.strip_guard(tuple(got) if len(got) > 1 else got[0])
        if any(np.atleast_1d(overrun)):
            raise RuntimeError(
                f"{self.fn.name}: changed {overrun} guard byte(s) past its output "
                f"({GUARD_BYTES} after each tile)"
            )
        return data

    def check(self, outputs, inputs):
        """Judge ``run``'s outputs against the contract's reference for ``inputs``."""
        return self.fn.judge(
            outputs,
            self.fn.expected(inputs, scalars=self.scalars),
            inputs=inputs,
            scalars=self.scalars,
            calls=self.calls,
        )

    def _traced(self):
        """Return ``(setup traced, [traced initializer argument indices])``, checked."""
        setup = bool(self._setter) and _timed(self._setter, "setup")
        inits = [
            i for i, init in self._initializers if _timed(init, f"initializer {i}")
        ]
        return setup, inits

    def traced_intervals(self) -> int:
        """How many ``event0``/``event1`` intervals a traced run emits.

        A ``trace_size`` that holds fewer truncates the run; ``0`` when the
        kernel itself is not timed.
        """
        trace = self.fn.contract.trace
        if trace is None or trace.shape != "whole_call":
            return 0
        setup, inits = self._traced()
        return int(setup) + self.calls * (len(inits) + 1)

    def cycles_per_call(self, inputs, *, trace_size, workdir) -> CallCycles:
        """Trace one run and split its intervals by kernel (``CallCycles``).

        Only a kernel whose contract declares ``Trace.whole_call()`` is timed;
        one declaring ``none`` or ``partial`` returns its reason without running,
        and one declaring nothing raises. Traced initializers and a traced setup
        kernel are split off by position; a ``partial`` one raises, because its
        intervals cannot be labeled. Size ``trace_size`` from
        ``traced_intervals``.
        """
        c = self.fn.contract
        if c.trace is None:
            raise ValueError(f"{self.fn.name}: the contract declares no trace")
        if c.trace.shape != "whole_call":
            return CallCycles(untimed=c.trace.reason)
        setup, inits = self._traced()
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        cfg = TraceConfig(trace_size=trace_size, trace_file=str(workdir / "trace.txt"))
        self.run(inputs, trace_config=cfg)
        if cfg.physical_mlir_path is None:
            raise RuntimeError("the traced run recorded no physical MLIR path")
        trace_json = workdir / "trace.json"
        cfg.trace_to_json(cfg.physical_mlir_path, str(trace_json))
        durations = [d for p in get_cycles_summary(str(trace_json)) for d in p[1:]]
        head, per_kernel, truncated = split_intervals(
            durations,
            calls=self.calls,
            per_call=len(inits) + 1,
            setup=int(setup),
            flush=TRACE_FLUSH,
        )
        if not per_kernel[-1]:
            raise RuntimeError(
                f"{self.fn.name}: the trace holds {len(durations)} intervals and "
                "none of the kernel's; the buffer is too small or the markers are "
                "missing"
            )
        return CallCycles(
            kernel=per_kernel[-1],
            initializers=dict(zip(inits, per_kernel)),
            setup=head,
            truncated=truncated,
        )


# One JIT entry for every kernel. Its tensors are the host buffers the
# design takes, inputs then outputs, in ``host_args`` order (the marker only
# says they are tensors); the count follows from the kernel, so the cache
# key never depends on closure state.
@jit
def _stream(
    *tensors: In,
    kernel: CompileTime[ExternalFunction],
    plan: CompileTime[tuple],
    calls: CompileTime[int],
    stack_bytes: CompileTime[int | None] = None,
    scalars: CompileTime[tuple] = (),
    params: CompileTime[tuple] = (),
    trace_config: CompileTime[TraceConfig | None] = None,
    guard: CompileTime[bool] = False,
    arg_byte_offsets: CompileTime[tuple] = (),
):
    kd = KernelDesign(
        kernel,
        calls=calls,
        guard=guard,
        scalars=scalars,
        stack_bytes=stack_bytes,
        arg_byte_offsets=arg_byte_offsets,
    )
    if plan != kd._plan:
        raise ValueError(f"{kernel.name}: plan does not match the kernel's contract")
    return kd._build_stream(params, trace_config)


__all__ = [
    "CallCycles",
    "KernelDesign",
    "elems",
    "shape_dtype",
    "split_intervals",
]
