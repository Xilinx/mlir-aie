# kernel_harness.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Build, run and check any ``aie.iron.kernels`` factory from its contract.

A kernel factory returns an ``ExternalFunction`` whose ``arg_types()`` fix
every argument's shape and dtype and whose ``.contract``
(:class:`~aie.iron.kernels.KernelContract`) says which argument is which, how
to compute the expected result on the host, and how close the device must
come. That is enough to generate a single-Worker design for it, so this
module does it once, for every kernel:

    from aie.iron import kernels
    from aie.utils import kernel_harness as kh

    verdict = kh.check(kernels.reduce_max, calls=16, dtype=bfloat16)
    assert verdict, verdict.detail

``check`` draws inputs, builds the design, runs it on the current device,
computes the reference and compares under the kernel's tolerance. The pieces
are exposed separately (``design``, ``sample_inputs``, ``upload``, ``run``,
``expected``, ``judge``, ``cycles_per_call``) for benchmarks and for users
bringing up a new kernel.

Design generators are module-level functions with fixed arity, and everything
that varies -- the factory, its kwargs, the call count, runtime scalars, the
values of ``param`` arguments (baked into core Buffers) -- arrives through
``CompileTime`` kwargs. The JIT cache keys a generator by its
bytecode and its ``CompileTime`` values, not by closure cells, so a closure
over the kernel would make two different kernels share one cached build.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import aie.iron as iron
import numpy as np
from aie.helpers.util import v8bfp16ebs8
from aie.iron import (
    Buffer,
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils import bfp
from aie.utils.trace import TraceConfig
from aie.utils.trace.utils import get_cycles_summary
from aie.utils.verify import Tolerance, Verdict, compare

# --------------------------------------------------------------------------
# Contract helpers
# --------------------------------------------------------------------------


def _contract(fn):
    c = getattr(fn, "contract", None)
    if c is None:
        raise ValueError(
            f"kernel '{fn.name}' declares no contract; add a KernelContract to its "
            "factory (roles, reference, tolerance) so it can be built and checked"
        )
    return c


def _arg_types(fn) -> list:
    """Return the argument types as declared (numpy shapes and dtypes).

    ``ExternalFunction.arg_types()`` is rewritten with MLIR types by the first
    design that resolves the (memoized) kernel; ``declared_arg_types()`` is
    stable, and this falls back for kernels that predate it.
    """
    declared = getattr(fn, "declared_arg_types", None)
    return declared() if declared is not None else fn.arg_types()


def _shape_dtype(arg_type):
    """``(shape, dtype)`` of an ``np.ndarray[(n,), np.dtype[T]]`` argument type."""
    return arg_type.__args__[0], arg_type.__args__[1].__args__[0]


def _elems(arg_type) -> int:
    return int(np.prod(_shape_dtype(arg_type)[0]))


def _is_bfp(dt) -> bool:
    """Whether an argument element type is the bfp16ebs8 block (8 values in 9 bytes)."""
    return dt is v8bfp16ebs8


def _itemsize(dt) -> int:
    return bfp.BLOCK_BYTES if _is_bfp(dt) else np.dtype(dt).itemsize


def _values_per_elem(dt) -> int:
    return bfp.BLOCK if _is_bfp(dt) else 1


def dtype_name(dt) -> str:
    """``np.dtype(dt).name``, or ``"bfp16ebs8"`` for the block type numpy has no dtype for."""
    return "bfp16ebs8" if _is_bfp(dt) else np.dtype(dt).name


def _bfp_operands(fn) -> tuple[bool, bool, bool]:
    """Return ``(A, B, C)`` flags: which of a matmul's operands are bfp16ebs8 blocks.

    False for an argument a kernel does not have, so this answers for any
    kernel rather than only for the three-operand ones.
    """
    flags = [_is_bfp(_shape_dtype(t)[1]) for t in _arg_types(fn)[:3]]
    a, b, c = flags + [False] * (3 - len(flags))
    return a, b, c


def _tensor_positions(contract):
    """Return host-tensor argument indices in design order: inputs, then the output.

    Design parameters are ``In`` for ``in``/``param`` arguments and ``Out`` for
    the output, so the kernel's own argument order (``scale`` puts its output
    before its parameter) is not the design's parameter order.
    """
    ins = [i for i, r in enumerate(contract.roles) if r in ("in", "param")]
    return ins, contract.out_index


# A core tile's data memory on aie2 and aie2p; the design's tiles must fit
# beside the stack. ObjectFifos default to depth 2 (ping-pong); when that does
# not fit, depth 1 still runs the kernel correctly, just without overlap.
_CORE_MEMORY = 64 * 1024
_DEFAULT_STACK = 1024
# Stream Workers get 8 KB: aiecc measures each core's stack need and rejects
# a design whose stack is too small (conv2dk1 wants 2688 bytes, conv2dk3
# 4672). The fifo-depth budget below accounts for it.
_STREAM_STACK = 0x2000


def _fifo_depth(
    fn, tile_bytes: int, stack_bytes: int = _DEFAULT_STACK, fixed_bytes: int = 0
) -> int:
    """Largest ObjectFifo depth (2 or 1) at which ``tile_bytes`` per depth fits.

    ``fixed_bytes`` is memory used once regardless of depth: the Buffers that
    hold ``param`` arguments (a 3x3x64x64 conv weight set is 36 KB).
    """
    for depth in (2, 1):
        if depth * tile_bytes + fixed_bytes + stack_bytes <= _CORE_MEMORY:
            return depth
    raise ValueError(
        f"{fn.name}: one set of tiles is {tile_bytes} bytes plus {fixed_bytes} bytes "
        f"of parameters; with a {stack_bytes}-byte stack that exceeds the "
        f"{_CORE_MEMORY}-byte core memory. Use a smaller tile."
    )


def is_matmul(fn) -> bool:
    return hasattr(fn, "stream_dims")


def _matrix_shape(fn, shape: tuple | None, rank: int) -> tuple:
    """Return the host operand shape a matrix kernel was given, checked for rank."""
    if shape is None or len(shape) != rank:
        raise ValueError(
            f"{fn.name}: shape=(M, K{', N' if rank == 3 else ''}) is required, got {shape}"
        )
    return tuple(int(d) for d in shape)


def is_matvec(fn) -> bool:
    return hasattr(fn, "a_dims_from_stream")


# --------------------------------------------------------------------------
# Streaming kernels: N tiles in, one tile out, `calls` iterations
# --------------------------------------------------------------------------


def _fifo_plan(fn):
    """How the harness feeds ``fn``: ``(groups, in_roles, param_roles)``.

    ``groups`` are the ``in`` tensors gathered by type, one ObjectFifo each,
    in the order the types first appear. Tiles of one type travel through one
    fifo and are acquired together, the way the vision examples slide a window
    of lines, so swiglu's three same-type inputs cost one channel rather than
    three. A core tile has two input DMA channels, so two types can be fed and
    a third cannot -- conv2dk1_skip_init's int8 residual beside its uint8
    activations is two, and fits.

    ``param`` tensors (scale's factor, filter2d's 3x3 kernel) are not streamed
    at all: they become core Buffers with an initial value, as
    programming_examples/vision/edge_detect does, which also sidesteps the
    4-byte DMA length rule an 18-byte kernel would break.
    """
    c = _contract(fn)
    in_pos, _ = _tensor_positions(c)
    in_roles = [i for i in in_pos if c.roles[i] == "in"]
    param_roles = [i for i in in_pos if c.roles[i] == "param"]
    by_type: dict[str, list[int]] = {}
    for i in in_roles:
        by_type.setdefault(str(_arg_types(fn)[i]), []).append(i)
    groups = list(by_type.values())
    if len(groups) > 2:
        raise ValueError(
            f"{fn.name}: {len(in_roles)} 'in' tensors of {len(groups)} types need "
            f"{len(groups)} fifos, but a core tile has 2 input channels: "
            f"{sorted(by_type)}"
        )
    return groups, in_roles, param_roles


def param_values(fn, inputs: list[np.ndarray]) -> list[np.ndarray]:
    """Return the ``param`` arrays among logical ``inputs`` (one per ``in``/``param``)."""
    c = _contract(fn)
    in_pos, _ = _tensor_positions(c)
    return [np.asarray(a) for a, i in zip(inputs, in_pos) if c.roles[i] == "param"]


def _encode_params(fn, params) -> tuple:
    """Param arrays -> a hashable, fully-printed CompileTime value."""
    _, _, param_roles = _fifo_plan(fn)
    if len(params) != len(param_roles):
        raise ValueError(
            f"{fn.name}: {len(param_roles)} 'param' argument(s) need values at design "
            f"time (design(..., params=[...]); see param_values), got {len(params)}"
        )
    out = []
    for i, a in zip(param_roles, params):
        shape, dt = _shape_dtype(_arg_types(fn)[i])
        a = np.asarray(a)
        if a.size != int(np.prod(shape)):
            raise ValueError(
                f"{fn.name}: param {i} has {a.size} elements, kernel expects {shape}"
            )
        # A tuple of Python scalars prints in full, unlike a large ndarray,
        # so two designs with different constants never share a cache key.
        vals = tuple(a.astype(dt).astype(np.float64).ravel().tolist())
        out.append((np.dtype(dt).name, tuple(shape), vals))
    return tuple(out)


def _rounding_setter(c):
    """Return the ``set_rounding`` kernel a contract's ``rounding_mode`` asks for, or ``None``.

    A fresh Worker boots in floor; a kernel that names the mode it narrows in
    is run in that mode, as a design following its contract would run it.
    """
    from aie.iron import kernels

    mode = c.needs_rounding_mode
    return kernels.set_rounding(mode) if mode else None


def _opt(x) -> list:
    return [x] if x is not None else []


def _build_stream(
    tensors_in,
    tensor_out,
    *,
    factory,
    factory_kwargs,
    calls,
    scalars,
    params,
    trace_config,
):
    fn = factory(**factory_kwargs)
    c = _contract(fn)
    arg_types = _arg_types(fn)
    in_pos, out_pos = _tensor_positions(c)
    groups, in_roles, param_roles = _fifo_plan(fn)
    n_fifos_in = len(groups)
    if len(tensors_in) != n_fifos_in:
        raise ValueError(
            f"{fn.name}: design has {len(tensors_in)} input tensors, kernel needs {n_fifos_in}"
        )
    if len(params) != len(param_roles):
        raise ValueError(
            f"{fn.name}: {len(param_roles)} param value(s) expected, got {len(params)}"
        )
    n_scalars = c.roles.count("scalar")
    if len(scalars) != n_scalars:
        raise ValueError(
            f"{fn.name}: expected {n_scalars} scalar(s), got {len(scalars)}"
        )

    def nbytes(i):
        return _elems(arg_types[i]) * _itemsize(_shape_dtype(arg_types[i])[1])

    # One "set" is every streamed tile plus the output; `param` arguments
    # live in one Buffer each, whatever the depth.
    depth = _fifo_depth(
        fn,
        sum(nbytes(i) for i in in_roles + [out_pos]),
        stack_bytes=_STREAM_STACK,
        fixed_bytes=sum(nbytes(i) for i in param_roles),
    )
    fifos_in = [
        ObjectFifo(arg_types[g[0]], name=f"in{k}", depth=len(g) * depth)
        for k, g in enumerate(groups)
    ]
    fifo_out = ObjectFifo(arg_types[out_pos], name="out", depth=depth)
    # `param` arguments live in core Buffers initialised at build time.
    param_bufs = [
        Buffer(
            arg_types[i],
            name=f"param{k}",
            initial_value=np.array(vals, dtype=np.dtype(dt_name)).reshape(shape),
        )
        for k, (i, (dt_name, shape, vals)) in enumerate(zip(param_roles, params))
    ]
    # The trailing element count the C++ takes at runtime is the number of
    # per-call iterations, which is 1:1 with whichever tensor has fewer raw
    # elements when one side packs several values per iteration (rgba2hue's
    # 4-byte RGBA pixels in, 1-byte hue out: lineWidth is the smaller, output
    # side). A reduction's ``out_valid`` marks its output tile as padded
    # rather than narrower-per-iteration, so there the count is the (larger)
    # input's element count instead.
    in0_elems = _elems(arg_types[in_roles[0]])
    count = (
        in0_elems
        if c.out_valid is not None
        else min(in0_elems, _elems(arg_types[out_pos]))
    )
    setter = _rounding_setter(c)

    def core(*args):
        f_in = args[:n_fifos_in]
        f_out = args[n_fifos_in]
        held = dict(
            zip(param_roles, args[n_fifos_in + 1 : n_fifos_in + 1 + len(param_bufs)])
        )
        kernel = args[n_fifos_in + 1 + len(param_bufs)]
        if setter is not None:
            args[-1]()
        for _ in range_(calls) if calls > 1 else range(1):
            elems = {}
            for k, g in enumerate(groups):
                # acquire(1) hands back the tile; acquire(n) a list of them.
                got = f_in[k].acquire(len(g))
                if len(g) == 1:
                    elems[g[0]] = got
                else:
                    elems.update({i: got[j] for j, i in enumerate(g)})
            o = f_out.acquire(1)
            call_args, s = [], iter(scalars)
            for i, role in enumerate(c.roles):
                if role == "in":
                    call_args.append(elems[i])
                elif role == "param":
                    call_args.append(held[i])
                elif role in ("out", "inout"):
                    call_args.append(o)
                elif role == "count":
                    call_args.append(count)
                else:  # scalar: a plain Python number, typed by the kernel's arg
                    call_args.append(next(s))
            kernel(*call_args)
            for k, g in enumerate(groups):
                f_in[k].release(len(g))
            f_out.release(1)

    worker = Worker(
        core,
        [f.cons() for f in fifos_in]
        + [fifo_out.prod(), *param_bufs, fn]
        + _opt(setter),
        stack_size=_STREAM_STACK,
        trace=1 if trace_config else 0,
    )

    def host_ty(i, reps):
        shape, dt = _shape_dtype(arg_types[i])
        return np.ndarray[(int(np.prod(shape)) * reps,), np.dtype[dt]]

    host_tys = [host_ty(g[0], calls * len(g)) for g in groups]
    host_tys += [host_ty(out_pos, calls)]

    def sequence(*args):
        n = n_fifos_in
        host_in, host_out = args[:n], args[n]
        h_in, h_out = args[n + 1 : 2 * n + 1], args[2 * n + 1]
        for h, t in zip(h_in, host_in):
            h.fill(t)
        h_out.drain(host_out, wait=True)

    rt = Runtime(sequence, host_tys + [f.prod() for f in fifos_in] + [fifo_out.cons()])
    prog = Program(iron.get_current_device(), rt, workers=[worker])
    if trace_config:
        prog.enable_trace(trace_config.trace_size, workers=[worker])
    return prog.resolve_program()


@iron.jit
def _stream1(
    x0: In,
    out: Out,
    *,
    factory: CompileTime[Callable],
    factory_kwargs: CompileTime[dict],
    calls: CompileTime[int],
    scalars: CompileTime[tuple] = (),
    params: CompileTime[tuple] = (),
    trace_config: CompileTime[TraceConfig | None] = None,
):
    return _build_stream(
        [x0],
        out,
        factory=factory,
        factory_kwargs=factory_kwargs,
        calls=calls,
        scalars=scalars,
        params=params,
        trace_config=trace_config,
    )


@iron.jit
def _stream2(
    x0: In,
    x1: In,
    out: Out,
    *,
    factory: CompileTime[Callable],
    factory_kwargs: CompileTime[dict],
    calls: CompileTime[int],
    scalars: CompileTime[tuple] = (),
    params: CompileTime[tuple] = (),
    trace_config: CompileTime[TraceConfig | None] = None,
):
    return _build_stream(
        [x0, x1],
        out,
        factory=factory,
        factory_kwargs=factory_kwargs,
        calls=calls,
        scalars=scalars,
        params=params,
        trace_config=trace_config,
    )


@iron.jit
def _stream3(
    x0: In,
    x1: In,
    x2: In,
    out: Out,
    *,
    factory: CompileTime[Callable],
    factory_kwargs: CompileTime[dict],
    calls: CompileTime[int],
    scalars: CompileTime[tuple] = (),
    params: CompileTime[tuple] = (),
    trace_config: CompileTime[TraceConfig | None] = None,
):
    return _build_stream(
        [x0, x1, x2],
        out,
        factory=factory,
        factory_kwargs=factory_kwargs,
        calls=calls,
        scalars=scalars,
        params=params,
        trace_config=trace_config,
    )


_STREAM = {1: _stream1, 2: _stream2, 3: _stream3}


# --------------------------------------------------------------------------
# Matrix kernels: mm / mv with their published DMA layouts
# --------------------------------------------------------------------------

_MM_STACK = 0xD00  # as programming_examples/basic/matrix_multiplication
_BFP_MM_STACK = 0xF00  # as programming_examples/ml/block_datatypes (mixed)


@iron.jit
def _matmul(
    A: In,
    B: In,
    C: Out,
    *,
    factory: CompileTime[Callable],
    factory_kwargs: CompileTime[dict],
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    trace_config: CompileTime[TraceConfig | None] = None,
):
    """Single-core C = A @ B over (M/m) x (N/n) output tiles, K/k products each.

    The design of programming_examples/basic/matrix_multiplication/single_core,
    with the micro-tile layout transforms taken from the kernel's
    ``stream_dims`` instead of re-derived. With the kernel's ``b_col_maj`` the
    host B buffer holds B^T (N, K) and with ``c_col_maj`` the host C buffer
    receives C^T (N, M), as in whole_array; ``upload`` and ``judge`` do the
    transposes so callers keep the logical (K, N) and (M, N).
    """
    from aie.helpers.taplib import TensorTiler2D

    mm = factory(**factory_kwargs)
    zero = mm.zero
    (a_shape, dt_a), (_, dt_b), (c_shape, dt_c) = (
        _shape_dtype(t) for t in _arg_types(mm)
    )
    m, k, n = factory_kwargs["dim_m"], factory_kwargs["dim_k"], factory_kwargs["dim_n"]
    # A bfp16ebs8 operand counts 8 values per element, so its shapes divide
    # the value counts by 8 along the contiguous axis, as the block_datatypes
    # examples declare them.
    va, vb, vc = (_values_per_elem(dt) for dt in (dt_a, dt_b, dt_c))
    assert a_shape == (m * k // va,) and c_shape == (m * n // vc,)
    any_bfp = va > 1 or vb > 1 or vc > 1
    stack = _BFP_MM_STACK if any_bfp else _MM_STACK
    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    tiles = M_div_m * N_div_n
    # C drains in ping-pong groups of ``c_rows`` tile rows: two when M holds
    # an even number of tile rows, one otherwise, so a single-tile M builds.
    c_rows = 2 if M_div_m % 2 == 0 else 1
    rows_per_block = 2 * c_rows
    dims = mm.stream_dims

    tile_bytes = (
        m * k // va * _itemsize(dt_a)
        + k * n // vb * _itemsize(dt_b)
        + m * n // vc * _itemsize(dt_c)
    )
    depth = _fifo_depth(mm, tile_bytes, stack_bytes=stack)
    a_ty = np.ndarray[(m, k // va), np.dtype[dt_a]]
    b_ty = np.ndarray[(k, n // vb), np.dtype[dt_b]]
    c_ty = np.ndarray[(m, n // vc), np.dtype[dt_c]]
    in_a = ObjectFifo(a_ty, name="inA", depth=depth)
    mem_a = in_a.cons().forward(name="memA", dims_to_stream=dims["A"])
    in_b = ObjectFifo(b_ty, name="inB", depth=depth)
    mem_b = in_b.cons().forward(name="memB", dims_to_stream=dims["B"])
    mem_c = ObjectFifo(c_ty, name="memC", depth=depth)
    out_c = mem_c.cons().forward(name="outC", dims_to_stream=dims["C"])

    setter = _rounding_setter(_contract(mm))

    def core(of_a, of_b, of_c, zero_k, mm_k, *set_mode):
        if set_mode:
            set_mode[0]()
        for _ in range_(tiles) if tiles > 1 else range(1):
            c = of_c.acquire(1)
            zero_k(c)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                a = of_a.acquire(1)
                b = of_b.acquire(1)
                mm_k(a, b, c)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core,
        [mem_a.cons(), mem_b.cons(), mem_c.prod(), zero, mm] + _opt(setter),
        stack_size=stack,
        trace=1 if trace_config else 0,
    )

    A_ty = np.ndarray[(M * K // va,), np.dtype[dt_a]]
    B_ty = np.ndarray[(K * N // vb,), np.dtype[dt_b]]
    C_ty = np.ndarray[(M * N // vc,), np.dtype[dt_c]]
    A_tiles = TensorTiler2D.group_tiler(
        (M, K // va),
        (m, k // va),
        (1, K_div_k),
        pattern_repeat=N_div_n,
        prune_step=False,
    )
    if mm.b_col_maj:
        # B^T on the host: the (n, k) tiles of one B column are consecutive rows.
        b_tap = TensorTiler2D.group_tiler(
            (N, K // vb), (n, k // vb), (N_div_n, K_div_k), prune_step=False
        )[0]
    else:
        b_tap = TensorTiler2D.group_tiler(
            (K, N // vb),
            (k, n // vb),
            (K_div_k, N_div_n),
            tile_group_col_major=True,
            prune_step=False,
        )[0]
    if mm.c_col_maj:
        # C^T on the host: the core emits C tiles row by row, i.e. down one
        # column of C^T's tile grid, so each drained group is c_rows columns
        # of (n, m) tiles walked column-major.
        C_tiles = TensorTiler2D.group_tiler(
            (N, M // vc),
            (n, m // vc),
            (N_div_n, c_rows),
            tile_group_col_major=True,
            prune_step=False,
        )
    else:
        C_tiles = TensorTiler2D.group_tiler(
            (M, N // vc), (m, n // vc), (c_rows, N_div_n), prune_step=False
        )

    def sequence(A_h, B_h, C_h, in_a_h, in_b_h, out_c_h):
        tgs: list = []
        c_index = 0
        for tile_row_block in range(iron.ceildiv(M_div_m, rows_per_block)):
            for pingpong in (0, 1):
                row_base = tile_row_block * rows_per_block + pingpong * c_rows
                num_tile_rows = min(c_rows, M_div_m - row_base)
                if num_tile_rows <= 0:
                    break
                tgs.append(TaskGroup())
                for tile_row in range(num_tile_rows):
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    in_a_h.fill(A_h, tap=A_tiles[tile_offset], group=tgs[-1])
                    in_b_h.fill(B_h, tap=b_tap, group=tgs[-1])
                out_c_h.drain(C_h, tap=C_tiles[c_index], group=tgs[-1], wait=True)
                c_index += 1
                if tile_row_block > 0 or pingpong > 0:
                    tgs[-2].finish()
                    del tgs[-2]
        tgs[-1].finish()
        del tgs[-1]

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, in_a.prod(), in_b.prod(), out_c.cons()])
    prog = Program(iron.get_current_device(), rt, workers=[worker])
    if trace_config:
        prog.enable_trace(trace_config.trace_size, workers=[worker])
    return prog.resolve_program()


@iron.jit
def _matvec(
    A: In,
    B: In,
    C: Out,
    *,
    factory: CompileTime[Callable],
    factory_kwargs: CompileTime[dict],
    M: CompileTime[int],
    K: CompileTime[int],
    trace_config: CompileTime[TraceConfig | None] = None,
):
    """Single-core c = A @ b, one m-row block of c per Worker iteration.

    The design of programming_examples/basic/matrix_multiplication/matrix_vector
    with ``n_cores = 1``; the transposed A layout comes from the kernel's
    ``a_dims_from_stream``.
    """
    from aie.helpers.taplib import TensorTiler2D

    mv = factory(**factory_kwargs)
    zero = mv.zero
    (_, dt_in), _, (_, dt_out) = (_shape_dtype(t) for t in _arg_types(mv))
    m, k = factory_kwargs["dim_m"], factory_kwargs["dim_k"]
    M_div_m, K_div_k = M // m, K // k

    mem_a = ObjectFifo(np.ndarray[(m, k), np.dtype[dt_in]], name="memA")
    core_a = mem_a.cons().forward(name="coreA", dims_from_stream=mv.a_dims_from_stream)
    in_b = ObjectFifo(np.ndarray[(k,), np.dtype[dt_in]], name="inB")
    out_c = ObjectFifo(np.ndarray[(m,), np.dtype[dt_out]], name="outC")

    setter = _rounding_setter(_contract(mv))

    def core(of_a, of_b, of_c, zero_k, mv_k, *set_mode):
        if set_mode:
            set_mode[0]()
        c = of_c.acquire(1)
        zero_k(c)
        for _ in range_(K_div_k) if K_div_k > 1 else range(1):
            a = of_a.acquire(1)
            b = of_b.acquire(1)
            mv_k(a, b, c)
            of_a.release(1)
            of_b.release(1)
        of_c.release(1)

    worker = Worker(
        core,
        [core_a.cons(), in_b.cons(), out_c.prod(), zero, mv] + _opt(setter),
        trace=1 if trace_config else 0,
    )

    A_ty = np.ndarray[(M * K,), np.dtype[dt_in]]
    B_ty = np.ndarray[(K,), np.dtype[dt_in]]
    C_ty = np.ndarray[(M,), np.dtype[dt_out]]
    a_tap = TensorTiler2D.group_tiler(
        (M, K), (m, k), (M_div_m, K_div_k), prune_step=False
    )[0]
    c_tap = TensorTiler2D.simple_tiler((1, M), (1, M), prune_step=False)[0]
    b_tap = TensorTiler2D.simple_tiler(
        (1, K), pattern_repeat=M_div_m, prune_step=False
    )[0]

    def sequence(A_h, B_h, C_h, in_b_h, mem_a_h, out_c_h):
        in_b_h.fill(B_h, b_tap)
        mem_a_h.fill(A_h, a_tap)
        out_c_h.drain(C_h, c_tap, wait=True)

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, in_b.prod(), mem_a.prod(), out_c.cons()])
    prog = Program(iron.get_current_device(), rt, workers=[worker])
    if trace_config:
        prog.enable_trace(trace_config.trace_size, workers=[worker])
    return prog.resolve_program()


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def design(
    factory: Callable,
    *,
    calls: int = 1,
    scalars: tuple = (),
    shape: tuple | None = None,
    params: list[np.ndarray] | None = None,
    aiecc_flags: list[str] | None = None,
    **factory_kwargs,
):
    """Return a compiled-on-first-call design wrapping ``factory(**factory_kwargs)``.

    Streaming kernels run ``calls`` iterations over tiles; ``scalars`` supplies
    the values of the contract's ``scalar`` arguments in order, and ``params``
    the arrays of its ``param`` arguments, which are baked into core Buffers
    (``param_values(fn, sample_inputs(fn, ...))`` picks them out). Matrix
    kernels take ``shape=(M, K, N)`` (``mm``) or ``shape=(M, K)`` (``mv``) for
    the host operands and ignore ``calls``. ``aiecc_flags`` are forwarded to
    the build (a benchmark passes ``--get-core-elfs`` to size the per-core
    ELFs).

    The returned ``CallableDesign`` is called with the host tensors the
    design streams -- see ``host_layout`` -- then the output.
    """
    fn = factory(**factory_kwargs)
    c = _contract(fn)  # a clear error before any generator is specialised
    if c.unsupported:
        raise ValueError(
            f"{fn.name}: the generic harness cannot build this kernel: {c.unsupported}"
        )
    kw: dict[str, Any] = dict(factory=factory, factory_kwargs=factory_kwargs)
    flags = list(aiecc_flags or ())
    if is_matmul(fn):
        M, K, N = _matrix_shape(fn, shape, 3)
        if any(_bfp_operands(fn)) and "--dynamic-objFifos" not in flags:
            # 9-byte block elements: the block_datatypes examples build with it.
            flags.append("--dynamic-objFifos")
        if flags:
            kw["aiecc_flags"] = flags
        return _matmul.specialize(M=M, K=K, N=N, **kw)
    if flags:
        kw["aiecc_flags"] = flags
    if is_matvec(fn):
        M, K = _matrix_shape(fn, shape, 2)
        return _matvec.specialize(M=M, K=K, **kw)
    groups, _, _ = _fifo_plan(fn)  # raises when the types need a third channel
    return _STREAM[len(groups)].specialize(
        calls=calls,
        scalars=tuple(scalars),
        params=_encode_params(fn, params or ()),
        **kw,
    )


# Fallback magnitudes for integer inputs of a kernel that declares no
# accumulator: wide enough to exercise the datapath, small enough that a
# product or a short sum stays inside a 32-bit accumulator.
_INT_RANGE = {np.int8: 60, np.int16: 8000, np.int32: 1 << 20, np.uint8: 255}


def input_limit(fn, dtype, *, reduction: int | None = None) -> int | None:
    """Largest integer magnitude an input may take without overflowing ``fn``.

    From the contract's ``acc_dtype`` and ``reduction`` (``reduction``
    overrides the per-call value, e.g. with the full ``K`` of a tiled
    matmul): with two or more multiplied inputs every product of two limits
    summed ``reduction`` times must fit the accumulator with a factor-4
    margin; with one input the sum of ``reduction`` limits must. The output
    dtype bounds the limit too only under ``overflow="undefined"``: a
    saturating or wrapping kernel is judged that way (see
    :func:`aie.utils.verify.compare`), and clipping its inputs to the output
    range would leave a requantising kernel's data near zero. ``None`` for
    float inputs, or when the contract declares no accumulator (the sampler
    then uses a fixed table).
    """
    c = _contract(fn)
    dt = np.dtype(dtype)
    if not np.issubdtype(dt, np.integer):
        return None
    if c.acc_dtype is None or not np.issubdtype(np.dtype(c.acc_dtype), np.integer):
        return None
    n = reduction or c.reduction or 1
    out_dt = np.dtype(_shape_dtype(_arg_types(fn)[c.out_index])[1])
    budget = np.iinfo(c.acc_dtype).max // 4
    if c.overflow == "undefined" and np.issubdtype(out_dt, np.integer):
        budget = min(budget, np.iinfo(out_dt).max // 4)
    n_tensors = sum(1 for r in c.roles if r in ("in", "param"))
    limit = int(np.sqrt(budget // n)) if n_tensors >= 2 else budget // n
    return max(1, min(limit, int(np.iinfo(dt).max)))


def sample_inputs(
    fn, *, calls: int = 1, shape: tuple | None = None, rng=None
) -> list[np.ndarray]:
    """Random host inputs for one run of ``fn``: one array per ``in``/``param``.

    Streaming inputs are shaped ``(calls, n)``; ``param`` inputs ``(n,)``. For
    ``mm``/``mv`` the operands are ``A (M, K)`` and ``B (K, N)`` / ``b (K,)``.
    A contract may supply its own ``sample`` when the data has structure.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    c = _contract(fn)
    if c.sample is not None:
        return c.sample(rng, calls)
    if is_matmul(fn) or is_matvec(fn):
        # A bfp16ebs8 operand is sampled as the float32 the host encodes.
        dt_a, dt_b = (
            np.float32 if _is_bfp(dt) else dt
            for dt in (_shape_dtype(t)[1] for t in _arg_types(fn)[:2])
        )
        shape = _matrix_shape(fn, shape, 3 if is_matmul(fn) else 2)
        M, K = shape[0], shape[1]
        b_shape = (K, shape[2]) if is_matmul(fn) else (K,)
        # The design accumulates over the full K, not one tile's k.
        limit = input_limit(fn, dt_a, reduction=K) or 60
        return [
            _draw(rng, (M, K), dt_a, int_range=limit),
            _draw(rng, b_shape, dt_b, int_range=limit),
        ]
    out = []
    for i in _tensor_positions(c)[0]:
        s, dt = _shape_dtype(_arg_types(fn)[i])
        n = int(np.prod(s))
        reps = (calls,) if c.roles[i] == "in" else ()
        out.append(_draw(rng, reps + (n,), dt, int_range=input_limit(fn, dt)))
    return out


def _draw(rng, shape, dt, int_range: int | None = None):
    if np.issubdtype(np.dtype(dt), np.integer):
        r = int_range or _INT_RANGE.get(dt, 1 << 15)
        lo = 0 if np.dtype(dt).kind == "u" else -r
        return rng.integers(lo, r, size=shape).astype(dt)
    return rng.standard_normal(shape).astype(np.float32).astype(dt)


def expected(fn, inputs: list[np.ndarray], *, scalars: tuple = ()) -> np.ndarray:
    """Return the contract's reference result, cast to the kernel's output dtype."""
    c = _contract(fn)
    if c.reference is None:
        raise ValueError(f"{fn.name}: contract has no reference")
    tensors, s = iter(inputs), iter(scalars)
    args = [
        next(s) if c.roles[i] == "scalar" else next(tensors)
        for i in c.reference_indices()
    ]
    _, out_dt = _shape_dtype(_arg_types(fn)[c.out_index])
    if _is_bfp(out_dt):
        out_dt = np.float32  # judged after decoding the device's blocks
    return np.asarray(c.reference(*args)).astype(out_dt)


def host_layout(fn, inputs: list[np.ndarray]) -> list[np.ndarray]:
    """Logical inputs (one per ``in``/``param``) -> the host tensors the design takes.

    ``mm`` with ``b_col_maj`` gets B transposed. ``param`` arrays are dropped
    (they were baked into the design). A kernel whose ``in`` tensors share
    one packed fifo (see :func:`_fifo_plan`) gets them interleaved per call,
    ``(calls, n_in, tile)``.
    """
    if is_matmul(fn):
        a, b = (np.asarray(x) for x in inputs)
        if fn.b_col_maj:
            b = np.ascontiguousarray(b.T)  # (N, K)
        bfp_a, bfp_b, _ = _bfp_operands(fn)
        if bfp_a or bfp_b:
            # Encoded along K (8 values share an exponent) and shuffled so
            # each (tile rows, k) DMA tile carries whole 8x8 sub-tiles.
            m, k, n = fn.dims
            if bfp_a:
                a = bfp.shuffle(bfp.encode(a), a.shape[1], a.shape[0], k, m)
            if bfp_b:
                b = bfp.shuffle(bfp.encode(b), b.shape[1], b.shape[0], k, n)
        return [a, b]
    if is_matvec(fn):
        return list(inputs)
    groups, _, _ = _fifo_plan(fn)
    c = _contract(fn)
    in_pos, _ = _tensor_positions(c)
    by_pos = dict(zip(in_pos, inputs))  # params are baked in
    out = []
    for g in groups:
        tiles = [np.asarray(by_pos[i]) for i in g]
        if len(tiles) == 1:
            out.append(tiles[0])
            continue
        # One fifo carries this group's tiles interleaved per call, matching
        # the order the core acquires them in.
        calls = tiles[0].shape[0] if tiles[0].ndim > 1 else 1
        packed = np.stack([a.reshape(calls, -1) for a in tiles], axis=1)
        out.append(np.ascontiguousarray(packed))
    return out


def upload(
    inputs: list[np.ndarray],
    out_size: int,
    out_dtype,
    *,
    fn,
    poison: bool = False,
):
    """Device tensors for one run: ``(design inputs, output)``.

    ``inputs`` are the logical ones from ``sample_inputs``; ``host_layout``
    turns them into what ``fn``'s design streams (B^T for ``b_col_maj``,
    interleaved tiles for a packed fifo), and ``judge`` undoes the output
    side. ``poison`` fills the output with ``0x55`` bytes instead of zeros,
    so a kernel that never writes cannot pass a comparison against a
    reference of zeros. Callers that time repeated runs keep the tensors and
    reuse them; ``run`` is the one-shot form.
    """
    # dtype= is required, not inferred: iron.tensor defaults to uint32, and
    # copying a float (or any other kind of) array into that buffer raises
    # rather than reinterpreting it.
    ins = [
        iron.tensor(np.ascontiguousarray(a).reshape(-1), dtype=a.dtype, device="npu")
        for a in host_layout(fn, inputs)
    ]
    nbytes = out_size * np.dtype(out_dtype).itemsize
    fill = 0x55 if poison else 0x00
    host = np.full(nbytes, fill, dtype=np.uint8).view(out_dtype)
    out = iron.tensor(host, dtype=out_dtype, device="npu")
    return ins, out


def run(
    design_,
    inputs: list[np.ndarray],
    out_size: int,
    out_dtype,
    *,
    fn,
    poison: bool = False,
    **call_kwargs,
) -> np.ndarray:
    """Move ``inputs`` to the device, run ``design_`` and return the output array.

    The result is a copy: ``Tensor.numpy()`` is a view of the XRT buffer's
    mapped host memory, and ``out`` is the last reference to that buffer, so
    the mapping goes away when this returns and the view would dangle.
    """
    ins, out = upload(inputs, out_size, out_dtype, fn=fn, poison=poison)
    design_(*ins, out, **call_kwargs)
    return out.numpy().copy()


def check(
    factory: Callable,
    *,
    calls: int = 1,
    scalars: tuple = (),
    shape: tuple | None = None,
    inputs: list[np.ndarray] | None = None,
    rng=None,
    tolerance: Tolerance | None = None,
    **factory_kwargs,
) -> Verdict:
    """Build, run and judge ``factory(**factory_kwargs)`` on the current device.

    Returns the :class:`~aie.utils.verify.Verdict`; ``tolerance`` overrides
    the contract's for a one-off (tightening while bringing a kernel up).
    """
    fn = factory(**factory_kwargs)
    if inputs is None:
        inputs = sample_inputs(fn, calls=calls, shape=shape, rng=rng)
    d = design(
        factory,
        calls=calls,
        scalars=scalars,
        shape=shape,
        params=param_values(fn, inputs),
        **factory_kwargs,
    )
    ref = expected(fn, inputs, scalars=scalars)
    got = run(
        d,
        inputs,
        output_size(fn, calls=calls, shape=shape),
        output_dtype(fn, ref.dtype),
        poison=True,
        fn=fn,
    )
    return judge(fn, got, ref, calls=calls, tolerance=tolerance)


def output_size(fn, *, calls: int = 1, shape: tuple | None = None) -> int:
    """Return the number of elements the device writes in one run, padding included.

    Elements of :func:`output_dtype`: bytes for a bfp16ebs8 output.
    """
    if is_matmul(fn):
        M, _, N = _matrix_shape(fn, shape, 3)
        n = M * N
        return n * bfp.BLOCK_BYTES // bfp.BLOCK if _bfp_operands(fn)[2] else n
    if is_matvec(fn):
        return _matrix_shape(fn, shape, 2)[0]
    return _elems(_arg_types(fn)[_contract(fn).out_index]) * calls


@dataclass(frozen=True)
class HostArg:
    """One host buffer a design takes, in the order it is called with.

    ``direction`` is ``"in"`` or ``"out"``; ``shape`` and ``dtype`` are what
    the device expects *after* :func:`host_layout` (B transposed for a
    ``b_col_maj`` matmul, encoded bytes for a bfp16ebs8 operand, interleaved
    tiles for a packed fifo), so a caller can allocate straight from it.
    """

    direction: str
    shape: tuple[int, ...]
    dtype: type

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.shape))


def host_args(fn, *, calls: int = 1, shape: tuple | None = None) -> list[HostArg]:
    """Return the host buffers one run of ``fn``'s design takes, inputs then the output.

    The declarative form of what :func:`sample_inputs`, :func:`host_layout`,
    :func:`output_size` and :func:`output_dtype` compute between them: a
    caller that only needs to size and allocate buffers (a runtime wrapper,
    a benchmark, an operator declaring its signature) can read this instead
    of running the sampler. ``param`` arguments are absent -- they are baked
    into the design -- and a reduction's output keeps the DMA padding the
    device actually writes.
    """
    c = _contract(fn)
    types = _arg_types(fn)
    out_dt = _shape_dtype(types[c.out_index])[1]
    args: list[HostArg] = []
    if is_matmul(fn) or is_matvec(fn):
        rank = 3 if is_matmul(fn) else 2
        dims = _matrix_shape(fn, shape, rank)
        M, K = dims[0], dims[1]
        in_dts = [_shape_dtype(t)[1] for t in types[:2]]
        bfp_a, bfp_b, bfp_c = (
            _bfp_operands(fn) if is_matmul(fn) else (False, False, False)
        )
        b_shape = (
            (dims[2], K)
            if is_matmul(fn) and fn.b_col_maj
            else ((K, dims[2]) if is_matmul(fn) else (K,))
        )

        def _enc(sh, dt, is_bfp):
            # An encoded operand is bytes: 9 per block of 8 along the last axis.
            if not is_bfp:
                return HostArg("in", sh, np.float32 if _is_bfp(dt) else dt)
            return HostArg(
                "in", (sh[0], sh[1] * bfp.BLOCK_BYTES // bfp.BLOCK), np.uint8
            )

        args.append(_enc((M, K), in_dts[0], bfp_a))
        args.append(_enc(b_shape, in_dts[1], bfp_b))
        if is_matmul(fn):
            c_shape = (dims[2], M) if fn.c_col_maj else (M, dims[2])
            args.append(
                HostArg(
                    "out",
                    (c_shape[0], c_shape[1] * bfp.BLOCK_BYTES // bfp.BLOCK),
                    np.uint8,
                )
                if bfp_c
                else HostArg("out", c_shape, out_dt)
            )
        else:
            args.append(HostArg("out", (M,), out_dt))
        return args
    groups, _, _ = _fifo_plan(fn)
    for g in groups:
        n, dt = _elems(types[g[0]]), _shape_dtype(types[g[0]])[1]
        args.append(
            HostArg("in", (calls, n), dt)
            if len(g) == 1
            else HostArg("in", (calls, len(g), n), dt)
        )
    args.append(HostArg("out", (output_size(fn, calls=calls, shape=shape),), out_dt))
    return args


def output_dtype(fn, ref_dtype):
    """Return the host dtype of the device output buffer: ``uint8`` bytes for bfp16ebs8, else the reference's."""
    if is_matmul(fn) and _bfp_operands(fn)[2]:
        return np.uint8
    return ref_dtype


def judge(
    fn,
    got: np.ndarray,
    ref: np.ndarray,
    *,
    calls: int = 1,
    tolerance: Tolerance | None = None,
) -> Verdict:
    """Compare a flat device output against the reference under the contract.

    Streaming outputs are viewed as ``(calls, tile)`` and trimmed to the
    contract's ``out_valid`` elements per call, so DMA padding is never
    compared; matrix outputs are reshaped to the reference, a bfp16ebs8 C
    unshuffled and decoded first.
    """
    c = _contract(fn)
    got = np.asarray(got)
    if is_matmul(fn) and _bfp_operands(fn)[2]:
        M, N = ref.shape
        m, _, n = fn.dims
        got = bfp.decode(bfp.shuffle(got, N, M, n, m, unshuffle=True))
    elif is_matmul(fn) and fn.c_col_maj:
        got = got.reshape(ref.shape[1], ref.shape[0]).T  # host buffer holds C^T
    elif is_matmul(fn) or is_matvec(fn):
        got = got.reshape(ref.shape)
    else:
        got = got.reshape(calls, -1)
        if c.out_valid is not None:
            got = got[:, : c.out_valid]
        ref = ref.reshape(calls, -1)
    return compare(
        got,
        ref,
        tolerance or c.tolerance or Tolerance.default_for(ref.dtype),
        overflow=c.overflow,
        subnormals=c.subnormals,
    )


def cycles_per_call(
    design_, inputs, out_size, out_dtype, *, fn, trace_size: int, workdir: Path
) -> list[int]:
    """Run once with tracing and return the core cycles of each kernel call.

    The event0 -> event1 pairing is ``aie.utils.trace.utils.get_cycles_summary``,
    the same one the programming examples print.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = TraceConfig(trace_size=trace_size, trace_file=str(workdir / "trace.txt"))
    run(design_, inputs, out_size, out_dtype, fn=fn, trace_config=cfg)
    trace_json = workdir / "trace.json"
    if cfg.physical_mlir_path is None:
        raise RuntimeError("the traced run recorded no physical MLIR path")
    cfg.trace_to_json(cfg.physical_mlir_path, str(trace_json))
    cycles: list[int] = []
    for per_process in get_cycles_summary(str(trace_json)):
        cycles.extend(int(d) for d in per_process[1:])
    return cycles


__all__ = [
    "check",
    "cycles_per_call",
    "design",
    "expected",
    "host_layout",
    "input_limit",
    "is_matmul",
    "is_matvec",
    "judge",
    "host_args",
    "HostArg",
    "output_size",
    "output_dtype",
    "dtype_name",
    "param_values",
    "run",
    "sample_inputs",
    "upload",
]
