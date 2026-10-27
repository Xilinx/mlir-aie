# kernel_design.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Build, sample and check independent kernel calls from their declarations.

Each call consumes one tile per ``In``, reads constant ``Param`` values, and
writes one tile per output. Layout codecs convert logical tiles to kernel
storage on the host. Whole-problem tiling, reductions across kernel calls and
multi-core schedules belong to algorithms, not this kernel-validation harness.
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
from aie.iron.kernel import ExternalFunction, Kernel
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
# Flush the final call's markers out of the trace unit's whole packets.
TRACE_FLUSH = 16
FLUSH_CYCLES = 32
# A Param bound at a byte offset sits in a buffer this much larger.
VIEW_PAD = 64
_INT_RANGE = {np.int8: 60, np.int16: 8000, np.int32: 1 << 20, np.uint8: 255}


def shape_dtype(arg_type):
    """Return the shape and dtype of a declared numpy tensor type."""
    return np_ndarray_type_get_shape(arg_type), np_ndarray_type_get_dtype(arg_type)


def elems(arg_type):
    return int(np.prod(shape_dtype(arg_type)[0]))


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

    ``kernel`` holds one interval per measured call; ``initializers`` holds
    intervals keyed by the argument initialized; ``setup`` holds its one
    interval. ``truncated`` says the trace buffer retained only a prefix.
    ``untimed`` gives the contract's reason when nothing ran.
    """

    kernel: tuple[int, ...] = ()
    initializers: dict[int, tuple[int, ...]] = field(default_factory=dict)
    setup: tuple[int, ...] = ()
    truncated: bool = False
    untimed: str | None = None


def split_intervals(durations, *, calls, per_call, setup=0, flush=0):
    """Label setup intervals, then ``per_call`` intervals for each call.

    Initializers run in contract order, followed by the measured kernel.
    Up to ``flush`` intervals of at most ``FLUSH_CYCLES`` may follow.
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
    """One kernel and its independent-call validation configuration.

    Tensor ``params`` and scalar ``scalars`` are embedded constants; changing
    them recompiles the design. When tensor params are unspecified, ``run``
    and ``cycles_per_call`` take them from their logical inputs.

    ``guard`` poisons output tiles and their trailing ``GUARD_BYTES`` on the
    core. ``run`` checks the guards and returns copied, stripped numpy arrays;
    ``check`` compares those arrays against the contract's reference.

    ``stack_bytes`` overrides aiecc's stack sizing. Otherwise buffering accounts
    for measured kernel frames. ``arg_byte_offsets`` binds tensor Params at
    runtime-loaded byte offsets within padded, poisoned buffers.

    Configuration attributes are read-only. Construct another harness to change
    the call count, guarding or other options after plan facts have been cached.
    """

    def __init__(
        self,
        fn,
        *,
        calls=1,
        shape=None,
        guard=False,
        scalars=(),
        params=None,
        aiecc_flags=None,
        stack_bytes=None,
        arg_byte_offsets=(),
    ):
        self._fn = fn
        self._calls_count = self._calls(calls, shape)
        self._guard_enabled = guard
        self._scalars = tuple(scalars)
        self._params = None if params is None else tuple(params)
        self._stack_bytes = stack_bytes
        self._aiecc_flags = tuple(aiecc_flags or ())
        self._designs = {}
        self._contract  # Validate the declaration even for host-only use.
        self._arg_byte_offsets = self._byte_offsets(arg_byte_offsets)

    fn = property(lambda self: self._fn)
    calls = property(lambda self: self._calls_count)
    guard = property(lambda self: self._guard_enabled)
    scalars = property(lambda self: self._scalars)
    params = property(lambda self: self._params)
    stack_bytes = property(lambda self: self._stack_bytes)
    aiecc_flags = property(lambda self: self._aiecc_flags)
    arg_byte_offsets = property(lambda self: self._arg_byte_offsets)

    @staticmethod
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

    @staticmethod
    def _device():
        # Bound, not just probed: factories read the architecture from it.
        device = ensure_current_device()
        if device is None:
            raise RuntimeError(
                "no device is bound; select one with iron.set_current_device()"
            )
        return device

    @cached_property
    def _types(self):
        return self.fn.arg_types()

    arg_types = property(lambda self: self._types)

    @cached_property
    def _contract(self):
        if self.fn.contract is None:
            raise ValueError(
                f"kernel '{self.fn.name}' declares no contract; add a KernelContract"
            )
        self.fn.contract.validate_types(self.arg_types)
        return self.fn.contract

    contract = property(lambda self: self._contract)

    @cached_property
    def _tensor_positions(self):
        c = self.contract
        return [
            i for i in c.reference_indices() if _is_tensor_type(self.arg_types[i])
        ], c.out_indices

    tensor_positions = property(lambda self: self._tensor_positions)

    @cached_property
    def _out_tiles(self):
        c = self.contract
        if c.out_offset is None:
            return self.calls
        step = c.out_offset[1]
        n = elems(self.arg_types[c.out_index])
        if n != self.calls * step:
            raise ValueError(
                f"{self.fn.name}: {self.calls} call(s) of {step} element(s) each must fill "
                f"the {n}-element output tile"
            )
        return 1

    output_tiles = property(lambda self: self._out_tiles)

    @cached_property
    def _fifo_plan(self):
        """Same-type input groups, streamed positions and tensor Param positions."""
        c, types = self.contract, self.arg_types
        ins = [i for i, r in enumerate(c.roles) if r is In]
        params = [
            i for i, r in enumerate(c.roles) if r is Param and _is_tensor_type(types[i])
        ]
        by_type = {}
        for i in ins:
            by_type.setdefault(types[i], []).append(i)
        groups = list(by_type.values())
        if len(groups) > self._device().core_dma_channels_in:
            raise ValueError(
                f"{self.fn.name}: input types exceed the core's input DMA channels"
            )
        if len(c.out_indices) > self._device().core_dma_channels_out:
            raise ValueError(
                f"{self.fn.name}: outputs exceed the core's output DMA channels"
            )
        return groups, ins, params

    fifo_plan = property(lambda self: self._fifo_plan)

    @cached_property
    def _guarded(self):
        return [
            i
            for i in self.contract.out_indices
            if self.guard and not bfp.is_bfp(shape_dtype(self.arg_types[i])[1])
        ]

    guarded_outputs = property(lambda self: self._guarded)

    def _layout(self, i):
        return self.contract.layouts[i] if self.contract.layouts else None

    def _guard_elems(self, i):
        return GUARD_BYTES // np.dtype(shape_dtype(self.arg_types[i])[1]).itemsize

    def _nbytes(self, i):
        return elems(self.arg_types[i]) * bfp.itemsize(
            shape_dtype(self.arg_types[i])[1]
        )

    def _encode_params(self, params):
        """Encode unbound tensor Params in argument order as design constants."""
        positions = self.fifo_plan[2]
        bound = dict(self.contract.parameter_bindings)
        free = [i for i in positions if i not in bound]
        params = list(params)
        if len(params) != len(free):
            raise ValueError(f"{self.fn.name}: expected {len(free)} param value(s)")
        bound.update(zip(free, params))
        encoded = []
        for i in positions:
            shape, dt = shape_dtype(self.arg_types[i])
            layout = self._layout(i)
            value = layout.encode(bound[i]) if layout else np.asarray(bound[i])
            if value.size != int(np.prod(shape)):
                raise ValueError(
                    f"{self.fn.name}: param {i} must contain {shape} elements"
                )
            encoded.append(
                (np.dtype(dt).name, shape, tuple(value.astype(dt).ravel().tolist()))
            )
        return tuple(encoded)

    def _byte_offsets(self, arg_byte_offsets):
        result = {}
        for i, offset in arg_byte_offsets:
            if i not in self.fifo_plan[2]:
                raise ValueError(
                    f"{self.fn.name}: argument {i} is not a tensor Param; only those "
                    "can be bound at a byte offset"
                )
            itemsize = np.dtype(shape_dtype(self.arg_types[i])[1]).itemsize
            if not 0 <= offset < VIEW_PAD or offset % itemsize:
                raise ValueError(
                    f"{self.fn.name}: argument {i}'s byte offset must be a multiple of "
                    f"{itemsize} below {VIEW_PAD}, not {offset}"
                )
            result[i] = offset
        return tuple(sorted(result.items()))

    @cached_property
    def _scalar_values(self):
        c, types = self.contract, self.arg_types
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
                f"{self.fn.name}: expected {len(free)} scalar(s), got {len(self.scalars)}"
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
        initializers = [(i, init(self.fn)) for i, init in self.contract.initializers]
        if any(
            r is InOut and i not in dict(initializers)
            for i, r in enumerate(self.contract.roles)
        ):
            raise ValueError(
                f"{self.fn.name}: every InOut requires a declared initializer"
            )
        return initializers

    @cached_property
    def _setter(self):
        return self.contract.setup() if self.contract.setup else None

    @staticmethod
    def _poison_fill(words, use_chess):
        # A separate call avoids inflating main's frame or generating memset.
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

    @cached_property
    def _words(self):
        return {i: (self._nbytes(i) + GUARD_BYTES) // 4 for i in self.guarded_outputs}

    @cached_property
    def _fills(self):
        return {
            n: self._poison_fill(n, self.fn.use_chess) for n in self._words.values()
        }

    @cached_property
    def _kernel_frame_bytes(self):
        """Static data coexists; only the deepest linked kernel stack is live."""
        cache = get_compile_arg("_iron_object_cache")
        arch = resolve_target_arch(self._device())
        data, deepest = {}, 0
        kernels = [
            self.fn,
            *(k for _, k in self._initializers),
            *self._fills.values(),
            *([self._setter] if self._setter else []),
        ]
        embed_bitcode = self.contract.uses_lut and not self.fn.use_chess
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

    @cached_property
    def _depth(self):
        ins = self.fifo_plan[1]
        outs = self.contract.out_indices
        tile_bytes = sum(self._nbytes(i) for i in [*ins, *outs])
        tile_bytes += GUARD_BYTES * len(self.guarded_outputs)
        fixed_bytes = (
            sum(self._nbytes(i) for i in self.fifo_plan[2])
            + (VIEW_PAD + 4) * len(self.arg_byte_offsets)
            + (self.stack_bytes or self._device().default_core_stack_bytes)
        )
        core_bytes = self._device().core_memory_bytes
        if not self.stack_bytes and 2 * tile_bytes + fixed_bytes <= core_bytes:
            fixed_bytes += self._kernel_frame_bytes
        depth = next(
            (d for d in (2, 1) if d * tile_bytes + fixed_bytes <= core_bytes), None
        )
        if depth is None:
            raise ValueError(
                f"{self.fn.name}: tiles plus parameters and stack exceed core memory; "
                "use a smaller tile"
            )
        return depth

    @cached_property
    def _flags(self):
        flags = list(self.aiecc_flags)
        if any(
            bfp.is_bfp(shape_dtype(t)[1]) for t in self.arg_types if _is_tensor_type(t)
        ):
            if "--dynamic-objFifos" not in flags:
                flags.append("--dynamic-objFifos")
        if self.contract.uses_lut and not self.fn.use_chess:
            if not any(f.startswith("--check-lut-banks") for f in flags):
                flags.append("--check-lut-banks")
        return flags

    @cached_property
    def _plan_key(self):
        # ExternalFunction's recipe excludes its contract. Include the facts
        # that change the schedule, constants, alignment checks or buffering.
        c = self.contract
        return (
            tuple(r.__name__ for r in c.roles),
            tuple(sorted(self._scalar_values.items())),
            c.out_offset,
            c.call_index,
            c.alignments,
            c.uses_lut,
            self.fn.stack_size_override,
            tuple(
                (
                    i,
                    repr(k),
                    k.stack_size_override,
                    k.contract.alignments if k.contract else (),
                )
                for i, k in self._initializers
            ),
            (
                (
                    repr(self._setter),
                    self._setter.stack_size_override,
                    self._setter.contract.alignments if self._setter.contract else (),
                )
                if self._setter
                else None
            ),
        )

    def design(self, params=None):
        """Return a cached CompilableDesign for the embedded parameter values."""
        self._device()
        if self.contract.unsupported:
            raise ValueError(
                f"{self.fn.name}: the generic harness cannot build this kernel: "
                f"{self.contract.unsupported}"
            )
        self.fifo_plan  # Check the DMA channel budget before anything builds.
        values = self.params if params is None else params
        encoded = self._encode_params(() if values is None else values)
        if encoded not in self._designs:
            self._designs[encoded] = _stream.specialize(
                kernel=self.fn,
                plan=self._plan_key,
                calls=self.calls,
                scalars=self.scalars,
                params=encoded,
                guard=self.guard,
                **({"stack_bytes": self.stack_bytes} if self.stack_bytes else {}),
                **(
                    {"arg_byte_offsets": self.arg_byte_offsets}
                    if self.arg_byte_offsets
                    else {}
                ),
                **({"aiecc_flags": self._flags} if self._flags else {}),
            )
        return self._designs[encoded]

    def _stage(self, params):
        return _KernelStage(self, params).stage()

    def as_mlir(self, *args, **kwargs):
        """Lower the default parameter specialization to MLIR."""
        return self.design().as_mlir(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.design()(*args, **kwargs)

    def _host_type(self, i, repetitions):
        n = elems(self.arg_types[i])
        n += self._guard_elems(i) if i in self.guarded_outputs else 0
        return np.ndarray[
            (n * repetitions,), np.dtype[shape_dtype(self.arg_types[i])[1]]
        ]

    def _build_stream(self, params, trace_config=None):
        stage = self._stage(params)
        stage.trace = trace_config is not None
        stage.trace_flush = TRACE_FLUSH
        groups = self.fifo_plan[0]
        host_types = [self._host_type(g[0], self.calls * len(g)) for g in groups]
        host_types += [
            self._host_type(i, self.output_tiles) for i in self.contract.out_indices
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
            coretile_events=[CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1],
        )

    def sample_inputs(self, rng=None):
        """One array per unbound In/tensor Param; only In has a call dimension."""
        rng = np.random.default_rng(0) if rng is None else rng
        c = self.contract
        if c.sample is not None:
            return c.sample(rng, self.calls)
        result = []
        for i in self.tensor_positions[0]:
            s, dt = shape_dtype(self.arg_types[i])
            layout = self._layout(i)
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
        """Pack layouts and interleave same-type In tiles; omit Param buffers."""
        positions = self.tensor_positions[0]
        if len(inputs) != len(positions):
            raise ValueError(f"{self.fn.name}: expected {len(positions)} input arrays")
        values = dict(zip(positions, inputs))
        result = []
        for group in self.fifo_plan[0]:
            arrays = []
            for i in group:
                value = np.asarray(values[i])
                layout = self._layout(i)
                arrays.append(
                    layout.encode(value)
                    if layout
                    else value.reshape(-1, elems(self.arg_types[i]))
                )
            result.append(
                np.ascontiguousarray(
                    arrays[0] if len(arrays) == 1 else np.stack(arrays, axis=1)
                )
            )
        return result

    def output_size(self):
        """Storage elements per output; a tuple for multiple outputs."""
        sizes = tuple(
            (
                elems(self.arg_types[i])
                + (self._guard_elems(i) if i in self.guarded_outputs else 0)
            )
            * self.output_tiles
            * (bfp.BLOCK_BYTES if bfp.is_bfp(shape_dtype(self.arg_types[i])[1]) else 1)
            for i in self.contract.out_indices
        )
        return sizes[0] if len(sizes) == 1 else sizes

    def strip_guard(self, outputs):
        """Return data and changed guard-byte counts (zero when unguarded)."""
        multiple = isinstance(outputs, tuple)
        if not self.guarded_outputs:
            return outputs, (0,) * len(outputs) if multiple else 0
        data, overrun = [], []
        for i, out in zip(
            self.contract.out_indices, outputs if multiple else (outputs,)
        ):
            if i not in self.guarded_outputs:
                data.append(out)
                overrun.append(0)
                continue
            rows = np.ascontiguousarray(out).reshape(self.output_tiles, -1)
            raw = rows.view(np.uint8)
            n = raw.shape[1] - GUARD_BYTES
            overrun.append(int(np.count_nonzero(raw[:, n:] != 0x55)))
            data.append(np.ascontiguousarray(raw[:, :n]).view(out.dtype).reshape(-1))
        if multiple:
            return tuple(data), tuple(overrun)
        return data[0], overrun[0]

    def host_args(self):
        """Describe physical host buffers, inputs then outputs, omitting Params."""
        result = []
        entries = [(In, g) for g in self.fifo_plan[0]]
        entries += [(Out, [i]) for i in self.contract.out_indices]
        for direction, indices in entries:
            i = indices[0]
            dt = shape_dtype(self.arg_types[i])[1]
            n = elems(self.arg_types[i])
            if bfp.is_bfp(dt):
                n, dt = n * bfp.BLOCK_BYTES, np.uint8
            elif direction is Out and i in self.guarded_outputs:
                n += self._guard_elems(i)
            rows = self.calls if direction is In else self.output_tiles
            s = (rows, n) if len(indices) == 1 else (rows, len(indices), n)
            result.append(_HostBuffer(direction, s, dt))
        return result

    def upload(self, inputs, *, poison=False):
        """Return device inputs and output tensor(s), deriving their storage."""
        ins = [
            tensor(a.reshape(-1), dtype=a.dtype, device="npu")
            for a in self.host_layout(inputs)
        ]
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
            for arg in self.host_args()
            if arg.direction is Out
        ]
        return ins, tuple(outs) if len(outs) > 1 else outs[0]

    def _design_for_inputs(self, inputs):
        return self.design(
            params=self.fn.param_values(inputs) if self.params is None else self.params
        )

    def run(self, inputs, *, poison=True, trace_config=None):
        """Execute, copy outputs to numpy, and assert that guards are intact."""
        compiled = self._design_for_inputs(inputs)
        ins, out = self.upload(inputs, poison=poison)
        compiled(
            *ins,
            *(out if isinstance(out, tuple) else (out,)),
            **({"trace_config": trace_config} if trace_config is not None else {}),
        )
        outputs = (
            tuple(np.array(o.numpy(), copy=True) for o in out)
            if isinstance(out, tuple)
            else np.array(out.numpy(), copy=True)
        )
        data, overrun = self.strip_guard(outputs)
        counts = overrun if isinstance(overrun, tuple) else (overrun,)
        assert not any(counts), (
            f"{self.fn.name}: output guard overwritten "
            f"({sum(counts)} byte(s); per output {counts})"
        )
        return data

    def check(self, outputs, inputs):
        """Judge already-stripped output arrays; bad data returns a false Verdict."""
        return self.fn.judge(
            outputs,
            self.fn.expected(inputs, scalars=self.scalars),
            inputs=inputs,
            scalars=self.scalars,
            calls=self.calls,
        )

    @staticmethod
    def _timed(kernel, role):
        trace = kernel.contract.trace if kernel.contract else None
        if trace is None:
            raise ValueError(
                f"{kernel.name} ({role}) declares no trace in its contract"
            )
        if trace.shape == "partial":
            raise ValueError(
                f"{kernel.name} ({role}): {trace.reason}, so its intervals cannot be "
                "told apart from the measured kernel's"
            )
        return trace.shape == "whole_call"

    @cached_property
    def _traced(self):
        setup = bool(self.contract.setup) and self._timed(self._setter, "setup")
        inits = [
            i for i, init in self._initializers if self._timed(init, f"initializer {i}")
        ]
        return setup, inits

    def traced_intervals(self):
        """Return the event0/event1 interval count, or zero for an untimed kernel."""
        trace = self.contract.trace
        if trace is None or trace.shape != "whole_call":
            return 0
        setup, inits = self._traced
        return int(setup) + self.calls * (len(inits) + 1)

    def cycles_per_call(
        self, inputs=None, *, trace_size=16384, workdir=None
    ) -> CallCycles:
        """Trace one run and split intervals into measured/setup/initializer calls."""
        c = self.contract
        if c.trace is None:
            raise ValueError(f"{self.fn.name}: the contract declares no trace")
        if c.trace.shape != "whole_call":
            return CallCycles(untimed=c.trace.reason)
        setup, inits = self._traced
        inputs = self.sample_inputs() if inputs is None else inputs
        workdir = Path("kernel_trace") if workdir is None else Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        cfg = TraceConfig(trace_size=trace_size, trace_file=str(workdir / "trace.txt"))
        ins, out = self.upload(inputs)
        self._design_for_inputs(inputs)(
            *ins, *(out if isinstance(out, tuple) else (out,)), trace_config=cfg
        )
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
                f"{self.fn.name}: the trace holds {len(durations)} intervals and none of "
                "the kernel's; the buffer is too small or the markers are missing"
            )
        return CallCycles(
            kernel=per_kernel[-1],
            initializers=dict(zip(inits, per_kernel)),
            setup=head,
            truncated=truncated,
        )


class _KernelStage:
    """Fresh per-build FIFO/buffer state; never retained by KernelDesign."""

    def __init__(self, design, params):
        self.design = design
        self.c, self.types = design.contract, design.arg_types
        self.groups, _, self.param_pos = design.fifo_plan
        self.outs = self.c.out_indices
        self.bound = design._scalar_values
        if len(params) != len(self.param_pos):
            raise ValueError(
                f"{design.fn.name}: expected {len(self.param_pos)} param values"
            )
        self.initializers = design._initializers
        self.setter = design._setter
        design.output_tiles
        self.guarded = design.guarded_outputs
        self.words, self.fills = design._words, design._fills
        self.shifted = dict(design.arg_byte_offsets)
        depth = design._depth
        self.fifos_in = [
            ObjectFifo(self.types[g[0]], name=f"in{j}", depth=len(g) * depth)
            for j, g in enumerate(self.groups)
        ]
        self.fifos_out = [
            ObjectFifo(self.fifo_type(i), name=f"out{j}", depth=depth)
            for j, i in enumerate(self.outs)
        ]
        self.buffers = [
            self.param_buffer(j, i, *param)
            for j, (i, param) in enumerate(zip(self.param_pos, params))
        ]
        self.shifts = (
            [
                Buffer(
                    np.ndarray[(len(self.shifted),), np.dtype[np.int32]],
                    name="param_shift",
                    initial_value=np.array(list(self.shifted.values()), dtype=np.int32),
                )
            ]
            if self.shifted
            else []
        )
        self.n_param, self.n_init = len(self.buffers), len(self.initializers)
        self.slot = {i: j for j, i in enumerate(self.outs)}
        self.n_shift = self.n_param + 1 + self.n_init + len(self.fills)

    def nbytes(self, i):
        return self.design._nbytes(i)

    def fifo_type(self, i):
        if i not in self.guarded:
            return self.types[i]
        return np.ndarray[(self.nbytes(i) + GUARD_BYTES,), np.dtype[np.int8]]

    @staticmethod
    def _view(raw, arg_type, byte_shift):
        return memref.view(
            np_ndarray_type_to_memref_type(arg_type),
            raw,
            constant(byte_shift, index=True),
            [],
        )

    def typed(self, outputs):
        return [
            self._view(o, self.types[i], 0) if i in self.guarded else o
            for i, o in zip(self.outs, outputs)
        ]

    def param_buffer(self, j, i, dt, shape, vals):
        value = np.array(vals, dtype=np.dtype(dt)).reshape(shape)
        if i not in self.shifted:
            return Buffer(self.types[i], name=f"param{j}", initial_value=value)
        # Poison both sides to expose loads rounded down to an aligned address.
        raw = poisoned(self.nbytes(i) + VIEW_PAD, np.int8)
        raw[self.shifted[i] : self.shifted[i] + self.nbytes(i)] = value.view(
            np.int8
        ).ravel()
        return Buffer(
            np.ndarray[raw.shape, np.dtype[np.int8]],
            name=f"param{j}",
            initial_value=raw,
        )

    def body(self, acquired, outputs, _held, constants, call):
        values = dict(zip(self.param_pos, constants[: self.n_param]))
        for k, i in enumerate(self.shifted):
            shift = arith.index_cast(IndexType.get(), constants[self.n_shift][k])
            values[i] = memref.view(
                np_ndarray_type_to_memref_type(self.types[i]),
                values[i].op,
                shift,
                [],
            )
        values.update(self.bound)
        offset = self.c.out_offset
        if offset:
            values[offset[0]] = call * offset[1]
        if self.c.call_index is not None:
            values[self.c.call_index] = call
        for got, group in zip(acquired, self.groups):
            values.update(
                (i, got if len(group) == 1 else got[j]) for j, i in enumerate(group)
            )
        values.update(zip(self.outs, self.typed(outputs)))
        constants[self.n_param](*(values[i] for i in range(len(self.c.roles))))

    def initialize(self, outputs, constants):
        for i, raw in zip(self.outs, outputs):
            if i in self.guarded:
                fill = constants[
                    self.n_param
                    + 1
                    + self.n_init
                    + list(self.fills).index(self.words[i])
                ]
                tile = self._view(raw, fill.arg_types()[0], 0)
                fill(tile, int(poisoned(1, np.int32)[0]))
        if self.initializers:
            views = self.typed(outputs)
            kernels = constants[self.n_param + 1 : self.n_param + 1 + self.n_init]
            for (i, _), init in zip(self.initializers, kernels):
                init(views[self.slot[i]])

    def stage(self):
        return Stage(
            self.body,
            inputs=[(fifo, len(g)) for fifo, g in zip(self.fifos_in, self.groups)],
            outputs=self.fifos_out,
            constants=self.buffers
            + [self.design.fn]
            + [init for _, init in self.initializers]
            + list(self.fills.values())
            + self.shifts
            + ([self.setter] if self.setter else []),
            iterations=self.design.calls,
            outputs_span_iterations=self.c.out_offset is not None,
            prologue=(lambda constants: constants[-1]()) if self.setter else None,
            initialize=self.initialize if self.initializers or self.guarded else None,
            stack_size=self.design.stack_bytes,
        )


# CompileTime keys use the kernel's stable content-based repr, not a closure.
@jit
def _stream(
    *tensors: In,
    kernel: CompileTime[Kernel],
    plan: CompileTime[tuple],
    calls: CompileTime[int],
    stack_bytes: CompileTime[int | None] = None,
    scalars: CompileTime[tuple] = (),
    params: CompileTime[tuple] = (),
    trace_config: CompileTime[TraceConfig | None] = None,
    guard: CompileTime[bool] = False,
    arg_byte_offsets: CompileTime[tuple] = (),
):
    design = KernelDesign(
        kernel,
        calls=calls,
        stack_bytes=stack_bytes,
        scalars=scalars,
        guard=guard,
        arg_byte_offsets=arg_byte_offsets,
    )
    return design._build_stream(params, trace_config)


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
    arg_byte_offsets=(),
    **factory_kwargs,
):
    """Compatibility wrapper for factories; prefer ``KernelDesign(fn).design()``."""
    KernelDesign._calls(calls, shape)
    KernelDesign._device()
    return KernelDesign(
        factory(**factory_kwargs),
        calls=calls,
        scalars=scalars,
        shape=shape,
        params=params,
        aiecc_flags=aiecc_flags,
        guard=guard,
        stack_bytes=stack_bytes,
        arg_byte_offsets=arg_byte_offsets,
    ).design()


__all__ = [
    "CallCycles",
    "KernelDesign",
    "design",
    "elems",
    "shape_dtype",
    "split_intervals",
]
