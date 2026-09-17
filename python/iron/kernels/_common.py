# kernels/_common.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Shared helpers for the kernels submodules."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit import markers as _markers
from aie.utils.compile.jit.markers import Count, InOut, Out, Scalar
from aie.utils.verify import Tolerance

_log = logging.getLogger(__name__)

ROLES = _markers.ROLES


@dataclass(frozen=True)
class TensorLayout:
    """Logical shape of one tensor tile and its reversible host storage codec.

    ``pack`` and ``unpack`` operate on batches shaped ``(calls, *shape)`` and
    ``(calls, storage_elements)`` respectively. They describe storage, not an
    algorithm or a whole-problem iteration schedule. Identity is the default.
    """

    shape: tuple[int, ...]
    pack: Callable | None = None
    unpack: Callable | None = None

    def encode(self, values):
        values = np.asarray(values).reshape(-1, *self.shape)
        return self.pack(values) if self.pack else values.reshape(len(values), -1)

    def decode(self, values, *, calls=1):
        values = np.asarray(values).reshape(calls, -1)
        return (
            self.unpack(values) if self.unpack else values.reshape(calls, *self.shape)
        )


@dataclass(frozen=True)
class KernelContract:
    """What a kernel computes, declared next to the factory that builds it.

    ``arg_types`` already fixes each argument's shape and dtype. The contract
    adds what types cannot say: which argument is which, how to compute the
    expected result on the host, and how close the device must come. With it
    a generic builder (``aie.iron.algorithms.kernel_design``) can build a design, run
    the kernel and judge the output for *any* factory, so correctness tests,
    e2e tests and benchmarks share one definition instead of each restating
    it.

    Attributes:
        roles: One of :data:`ROLES` per argument, in argument order --
            ``In``, ``Out``, ``InOut``, ``Param``, ``Scalar`` or ``Count``,
            the same markers ``@iron.jit`` uses. ``Out`` is written by the
            kernel; ``InOut`` is accumulated into (``mm``'s ``C += A * B``),
            which is why such a kernel ships a ``.also.zero`` sibling and a design
            initializes the buffer before each independent call. ``In`` is a
            read-only streamed tensor; ``Param`` is a read-only tensor held in
            a core Buffer, constant across calls. Neither is a scalar. Multiple
            ``Out``/``InOut`` arguments are permitted, in argument order.
        reference: Host implementation, and the kernel's arithmetic model:
            a saturating kernel's reference clips, a flushing one flushes.
            Called with every non-``Out``, non-``InOut``, non-``Count``
            argument in argument order: ``In``/``Param`` tiles as numpy arrays
            of shape ``(calls, n)`` in the kernel's dtype, ``Scalar`` values
            as Python numbers. Returns the expected
            output for all calls; the harness casts it to the output dtype.
            Multiple outputs are returned as a tuple in output argument order.
            Bound scalar arguments are omitted from the reference.
            ``None`` when no host reference exists yet -- the kernel is then
            built but not judged.
        tolerance: How close the device result must be, or ``None`` for
            :meth:`Tolerance.default_for` the output dtype. State the
            evidence in ``Tolerance.note``.
        ops_per_call: Arithmetic operations one kernel call performs, for
            throughput normalisation. ``None`` means one per output element.
        out_valid: Meaningful elements at the start of each output tile when
            the tile is padded for DMA alignment (reductions write one value
            into a 4-byte-aligned tile). ``None`` means the whole tile.
        sample: ``sample(rng, calls) -> list[np.ndarray]`` producing one host
            array per ``In``/``Param`` argument for ``calls`` kernel calls,
            for kernels whose inputs have structure a dtype cannot express
            (``expand``'s packed nibbles + scales). ``None`` lets the harness
            draw plain random data of each argument's dtype.
        acc_dtype: The type the kernel accumulates in (``np.int32`` for an
            ``acc32`` mmul, ``np.float32`` for ``accfloat``), or ``None`` when
            nothing is accumulated (copies, selections, bit operations). With
            ``reduction`` it tells the harness how large an input may be
            before the accumulator, or the output, would overflow.
        reduction: Terms summed into one output element per call (``K`` for
            a matmul tile, taps x channels for a convolution, the tile size
            for a reduction). ``None`` means one.
        setup: A kernel to call once on the core before the first call of
            this one, or ``None``. The core narrows accumulators in whatever
            mode its rounding register holds and boots in ``floor``; a kernel
            that needs another mode names the setter here, e.g.
            ``setup=conv_even``. A kernel whose source calls
            ``aie::set_rounding`` itself needs nothing.
        stack_bytes: Core stack a Worker calling this kernel needs, or
            ``None`` for the target's default. aiecc measures each core's
            stack and rejects a design whose stack is too small, so a kernel
            that needs more than the default says so here rather than making
            every design guess. Record where the number came from.
        cascade_partner: The factory for the other half, when this kernel is
            one half of a two-tile cascade pair. Half a pair computes half an
            answer: the partial sum crosses the cascade stream, which is not
            an argument, so a PUT half has no output role at all and neither
            half is separately observable. ``reference`` is therefore the
            *pair's* -- what the two compute together -- and ``unsupported``
            follows from naming a partner rather than being restated.
        unsupported: ``None`` when the generic builder can build, run and
            judge the kernel in a single-Worker design; otherwise the reason
            it cannot (a cascade protocol, an operand it cannot sample). The
            reference and the dtype facts still say what the kernel computes.
        layouts: Optional :class:`TensorLayout` per argument (``None`` means
            identity). Logical tile shapes and storage codecs belong to the
            declaration, never to a kernel-name switch in the harness.
        scalar_bindings: ``(argument_index, value)`` pairs for fixed runtime
            scalar operands, including element counts. Unbound ``Scalar``
            operands are supplied by the caller. Counts are never inferred
            from input/output sizes.
        initializers: ``(argument_index, factory)`` pairs for ``InOut``
            arguments; ``factory(fn)`` returns a one-buffer initialization
            kernel. The reference describes the result from that initial state.
        trace_cycles: Whether exactly one event0/event1 pair brackets the full
            invocation, with no additional pairs from setup or initializers.
            False unless audited; partial internal regions are not call timings.

    What the kernel does when a result overflows, how it rounds a narrowing
    store, and what it does with NaN or subnormal inputs are not declared
    here: ``reference`` is the arithmetic model (a saturating kernel's
    reference clips, a flushing kernel's reference flushes) and ``tolerance``
    is the slack allowed against it. Declaring them twice let the two drift.
    """

    roles: tuple[type, ...]
    reference: Callable[..., np.ndarray] | None = None
    tolerance: Tolerance | None = None
    ops_per_call: int | None = None
    out_valid: int | None = None
    sample: Callable[..., list] | None = None
    acc_dtype: type | None = None
    reduction: int | None = None
    setup: Callable[[], object] | None = None
    stack_bytes: int | None = None
    cascade_partner: Callable[..., object] | None = None
    unsupported: str | None = None
    layouts: tuple[TensorLayout | None, ...] = ()
    scalar_bindings: tuple[tuple[int, int | float], ...] = ()
    initializers: tuple[tuple[int, Callable], ...] = ()
    trace_cycles: bool = False

    def __post_init__(self):
        bad = [r for r in self.roles if r not in ROLES]
        if bad:
            names = ", ".join(r.__name__ for r in ROLES)
            raise ValueError(f"unknown kernel argument role(s) {bad}; use {names}")
        # A kernel with no data arguments at all (set_rounding sets core state)
        # has nothing to be the output. A cascade half may also have none: its
        # result leaves on the cascade stream, which is not an argument.
        n_out = self.roles.count(Out) + self.roles.count(InOut)
        if self.roles and not n_out and self.cascade_partner is None:
            raise ValueError("a kernel contract needs at least one Out or InOut role")
        if self.layouts and len(self.layouts) != len(self.roles):
            raise ValueError("layouts must have one entry per argument")
        bound = dict(self.scalar_bindings)
        if len(bound) != len(self.scalar_bindings) or any(
            i < 0 or i >= len(self.roles) or self.roles[i] not in (Scalar, Count)
            for i in bound
        ):
            raise ValueError("scalar_bindings must name distinct scalar arguments")
        if any(r is Count and i not in bound for i, r in enumerate(self.roles)):
            raise ValueError("Count requires an explicit scalar binding")
        initialized = dict(self.initializers)
        if len(initialized) != len(self.initializers) or any(
            i < 0 or i >= len(self.roles) or self.roles[i] is not InOut
            for i in initialized
        ):
            raise ValueError("initializers must name distinct InOut arguments")
        if self.reduction is not None and self.reduction < 1:
            raise ValueError(f"reduction must be >= 1, got {self.reduction}")
        if self.stack_bytes is not None and self.stack_bytes < 1:
            raise ValueError(f"stack_bytes must be >= 1, got {self.stack_bytes}")
        if self.unsupported is not None and not self.unsupported:
            raise ValueError("unsupported must be a reason, or None")
        # Naming a partner already says the single-Worker builder cannot drive
        # this kernel, so the reason is derived rather than restated.
        if self.cascade_partner is not None and self.unsupported is None:
            object.__setattr__(
                self,
                "unsupported",
                f"one half of a cascade pair with {self.cascade_partner.__name__}; "
                "the partial sum crosses the cascade stream, which is not an "
                "argument; a paired design is required to observe the result",
            )

    @property
    def out_indices(self) -> tuple[int, ...]:
        """Output argument positions, in declaration order."""
        return tuple(i for i, r in enumerate(self.roles) if r in (Out, InOut))

    @property
    def out_index(self) -> int:
        """Position of the output, whether the kernel writes it or accumulates into it."""
        if len(self.out_indices) > 1:
            raise ValueError("multiple outputs: use out_indices")
        roles = list(self.roles)
        if Out in roles:
            return roles.index(Out)
        if InOut in roles:
            return roles.index(InOut)
        raise ValueError(
            f"this kernel has no output argument: it emits on the cascade to "
            f"{self.cascade_partner.__name__ if self.cascade_partner else '?'}, "
            "so there is nothing to size or judge on its own"
        )

    @property
    def accumulates(self) -> bool:
        """Whether the kernel reads its output back (``InOut``), as ``C += A * B`` does."""
        return InOut in self.roles

    def reference_indices(self) -> list[int]:
        """Argument positions handed to ``reference``, in order.

        An ``InOut`` output is excluded like an ``Out`` one: the reference
        computes the result from the declared initializer's state. The
        builder initializes the buffer before each independent tile call.
        """
        bound = dict(self.scalar_bindings)
        return [
            i
            for i, r in enumerate(self.roles)
            if r not in (Out, InOut, Count) and i not in bound
        ]


def _detect_arch() -> str:
    """Return ``'aie2p'`` or ``'aie2'`` based on the active device.

    Falls back to ``'aie2'`` if no device is currently set.
    """
    try:
        from aie.utils import get_current_device
        from aie.utils.compile.utils import resolve_target_arch

        device = get_current_device(probe_runtime=False)
        return resolve_target_arch(device)
    except (ImportError, RuntimeError, AttributeError, ValueError):
        # ImportError: iron not built; RuntimeError: no explicit device set;
        # AttributeError/ValueError: unrecognised device.  Anything else (e.g.
        # OSError from a misconfigured install) bubbles up so the user sees it.
        _log.warning(
            "_detect_arch: no explicit device or unrecognised device; "
            "falling back to 'aie2'",
            exc_info=True,
        )
        return "aie2"


def _kernel_source(arch: str, subdir: str, filename: str) -> Path:
    """Return the absolute path to a kernel source file.

    Args:
        arch: Target architecture string (``'aie2'`` or ``'aie2p'``).
        subdir: Subdirectory under ``aie_kernels/`` (e.g. ``'aie2'``).
        filename: Source file name (e.g. ``'scale.cc'``).

    Returns:
        Path to the source file.

    Raises:
        FileNotFoundError: When the source file cannot be found.
    """
    from aie.utils import config

    base = Path(config.aie_kernels_dir())
    candidate = base / subdir / filename
    if candidate.exists():
        return candidate
    if subdir != "aie2":
        aie2_fallback = base / "aie2" / filename
        if aie2_fallback.exists():
            return aie2_fallback
    generic = base / "generic" / filename
    if generic.exists():
        return generic
    raise FileNotFoundError(
        f"Kernel source '{filename}' not found under {base}/{subdir}/, "
        f"{base}/aie2/, or {base}/generic/"
    )


def _include_dirs() -> list[str]:
    """Return the standard include directory list for kernel compilation."""
    from aie.utils import config

    return [config.cxx_header_path()]


_DTYPE_BIT_WIDTHS = {
    np.dtype(np.uint8): 8,
    np.dtype(np.int16): 16,
    np.dtype(np.int32): 32,
}


def _dtype_to_bit_width(dtype, *, factory_name: str) -> int:
    """Map ``np.uint8 | np.int16 | np.int32`` to 8/16/32.

    Raises:
        ValueError: When *dtype* is not one of the three supported types.
    """
    bit_width = _DTYPE_BIT_WIDTHS.get(np.dtype(dtype))
    if bit_width is None:
        raise ValueError(
            f"{factory_name}: unsupported dtype {dtype}. "
            "Use np.uint8, np.int16, or np.int32."
        )
    return bit_width


def dtypes(table: Iterable[dict]):
    """Declare the keyword combinations a factory builds, as ``factory.dtypes``.

    The registry and the host contract test enumerate the table instead of
    restating it.
    """

    def decorate(factory):
        # A function attribute: pyright models functions as having a fixed
        # attribute set, so the one assignment is annotated rather than each
        # of the ~20 factories that carry a table.
        factory.dtypes = tuple(table)  # pyright: ignore[reportFunctionMemberAccess]
        return factory

    return decorate


def _conv_act_dtype_info(
    base_name: str, act_dtype, *, factory_name: str
) -> tuple[str, list[str]]:
    """Map ``act_dtype`` to ``(func_name, compile_flags)`` for conv kernels.

    Raises:
        ValueError: When *act_dtype* is not ``np.int8`` or ``np.uint8``.
    """
    if act_dtype == np.int8:
        return f"{base_name}_i8", ["-DINT8_ACT"]
    elif act_dtype == np.uint8:
        return f"{base_name}_ui8", []
    else:
        raise ValueError(
            f"{factory_name}(): act_dtype must be np.int8 or np.uint8, "
            f"got {act_dtype}"
        )


def _require_fixed_tile_size(
    factory_name: str, tile_size: int, expected: int = 1024
) -> None:
    """Raise ValueError when ``tile_size`` does not match a hard-coded C++ loop bound."""
    if tile_size != expected:
        raise ValueError(
            f"{factory_name}() tile_size must be {expected} to match the "
            f"hard-coded C++ loop bound, got {tile_size}."
        )


def _require_min_trip_count(
    factory_name: str,
    elems: int,
    per_iter: int,
    min_iters: int,
    *,
    param: str = "tile_size",
) -> None:
    """Raise ValueError when a tile is too small for a kernel's vectorised loop.

    Several kernels declare ``AIE_LOOP_MIN_ITERATION_COUNT(n)``. Peano
    predefines ``__AIECC__``, so that expands to a real ``#pragma clang loop
    min_iteration_count(n)``: a promise the compiler may schedule against,
    not a check. Below it the pipelined loop runs past the tile. Two kernels
    here (passthrough, reduce_add) hang the core outright at four iterations
    rather than returning wrong data. ``reduce_max.cc`` states the same
    precondition as an ``assert``, which the ``-DNDEBUG`` build drops, so the
    check only has effect if it lives here.

    ``per_iter`` elements are consumed per iteration; a tile that is not a
    whole number of them also lets the tail load/store overrun.
    """
    if elems % per_iter:
        raise ValueError(
            f"{factory_name}() {param}={elems} is not a multiple of the "
            f"kernel's {per_iter}-element vector step; the tail iteration "
            f"would run past the tile."
        )
    if elems < min_iters * per_iter:
        raise ValueError(
            f"{factory_name}() {param}={elems} gives {elems // per_iter} "
            f"loop iterations, but the kernel declares a minimum of "
            f"{min_iters}; use {param} >= {min_iters * per_iter}."
        )


def _min_dma_aligned_elems(dtype, align: int = 4) -> int:
    """Return the minimum element count whose byte size is a multiple of *align*.

    The NPU shim DMA requires a 4-byte alignment.  A 1-element output tile is
    fine for ``int32`` (4 bytes) but only 2 bytes for ``bfloat16`` — kernels
    whose C++ side writes a single value still need a Python tile type with
    enough elements to satisfy the alignment.
    """
    itemsize = np.dtype(dtype).itemsize
    return max(1, (align + itemsize - 1) // itemsize)


def _default_source_path(filename: str, subdir: str | None = None) -> Path:
    """Return ``_kernel_source(arch, subdir or arch, filename)`` using the active arch."""
    arch = _detect_arch()
    return _kernel_source(arch, subdir or arch, filename)


def _arg_type_key(t):
    """Hashable key for one entry of ``arg_types`` (used by ``_EXTERN_CACHE``)."""
    if hasattr(t, "__args__"):
        # np.ndarray[(shape,), np.dtype[T]]
        shape = t.__args__[0]
        inner = t.__args__[1]
        dtype = inner.__args__[0] if hasattr(inner, "__args__") else inner
        return ("ndarray", tuple(shape), str(dtype))
    return repr(t)


# Cache keyed on the full input parameter tuple.  Identical helper calls
# (kernels.mm(...) twice with same kwargs) should return the SAME
# ExternalFunction instance — otherwise both end up in
# ExternalFunction._instances, both get JIT-compiled, and (because they
# share the default ``<name>.o`` output filename) the second compilation
# overwrites the first's object file with whichever just-rebuilt copy
# wins the race.  The whole_array port hit exactly this footgun: a
# default-flag kernels.mm() call (just to fetch .mac_dims) and a
# c_col_maj=True kernels.mm() call for the actual binding produced two
# differently-flagged ExternalFunctions whose .o files collided on disk.
_EXTERN_CACHE: dict = {}


def _make_extern(
    func_name: str,
    source_path: "Path | str",
    arg_types: list,
    *,
    compile_flags: list[str] | None = None,
    use_chess: bool = False,
    inline: bool = False,
    shared_object_file_name: str | None = None,
    contract: KernelContract | None = None,
) -> ExternalFunction:
    """Construct (or reuse) an ExternalFunction with the standard include_dirs.

    ``contract`` (a :class:`KernelContract`) is attached as ``extern.contract``
    so harnesses and tests can build, run and judge the kernel generically;
    factories without one leave it ``None``.

    ``inline`` uses Peano's always-inline LLVM IR and merge linking. Inline
    factories must use distinct C++ symbol names for distinct variants because
    LLVM IR cannot use the object-file symbol-prefix mechanism.

    Memoized on (func_name, source_path, arg_types, compile_flags,
    use_chess) so repeated calls with identical parameters return the
    SAME ExternalFunction instance (see ``_EXTERN_CACHE`` for rationale).

    Different parameterizations get distinct instances AND distinct
    ``object_file_name``s — the latter is auto-suffixed with a short
    digest of the cache key so per-parameterization .o files don't
    overwrite each other on disk.  The default ``<name>.o`` is preserved
    when ``compile_flags`` is empty AND ``use_chess`` is False (no
    parameterization to disambiguate).

    ``use_chess`` selects the Chess (xchesscc) compiler instead of Peano
    for this kernel's .o build.  See
    `ExternalFunction` for the design-level
    contract: all EFs in a single ``@iron.jit`` design must share the
    same toolchain choice (mixed peano/chess is rejected at compile
    time).

    ``shared_object_file_name`` pins the output ``.o`` filename so
    multiple factories targeting the SAME source file (e.g. companion
    symbols like ``reduce_max_vector`` + ``compute_max`` both in
    ``reduce_max.cc``) can share one compile.  The first call builds
    the ``.o``; subsequent calls with the same ``shared_object_file_name``
    skip the build and link against the existing one.  Without this,
    each factory would produce a distinct ``.o`` each carrying ALL
    symbols from the ``.cc``, tripping a duplicate-symbol link error.
    """
    flags_tuple = tuple(compile_flags or [])
    arg_keys = tuple(_arg_type_key(t) for t in arg_types)
    cache_key = (func_name, str(source_path), arg_keys, flags_tuple, use_chess)
    if inline:
        if use_chess:
            raise ValueError("inline kernels require Peano, not Chess")
        # Keep existing object artifact identities unchanged.
        cache_key += ("inline",)
    cached = _EXTERN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # The object_file_name suffix must distinguish every distinct cache_key,
    # not just compile_flags — otherwise two helper calls with the same
    # name + flags but different source / arg_types would generate
    # ExternalFunctions with identical .o filenames and trip the collision
    # check in ExternalFunction.__init__ (or, if both passed it, would
    # silently overwrite each other on disk).
    if shared_object_file_name is not None:
        # Caller explicitly pinned the .o filename so companion symbols from
        # the same .cc share one compile.  Skip the digest-suffix path so the
        # ExternalFunction lands at the pinned name; the second-and-later
        # callers' compiles short-circuit via compile_external_kernel's
        # "skip if .o exists" check.
        digest = None
        object_file_name = shared_object_file_name
    elif flags_tuple or arg_keys or str(source_path):
        # 8 hex chars of sha256 — short enough not to bloat MLIR strings,
        # wide enough that the chance of two distinct cache_keys colliding
        # is vanishingly small (~2^-32).
        digest = hashlib.sha256(repr(cache_key).encode()).hexdigest()[:8]
        object_file_name = f"{func_name}_{digest}{'.ll' if inline else '.o'}"
    else:
        digest = None
        object_file_name = None  # ExternalFunction default → ``<name>.o``

    # Prefix the SYMBOL name with the digest whenever this is a parameterized
    # kernel.  Two helper calls with different parameterizations compile two
    # .o files that would otherwise BOTH export the same C symbol — MLIR
    # rejects the duplicate `func.func` declaration and the linker rejects the
    # duplicate symbol.  Prefixing with ``<digest>`` gives each
    # parameterization a unique symbol; the rename is applied via the existing
    # ``symbol_prefix`` plumbing in ExternalFunction.
    #
    # Both the symbol AND ``object_file_name`` must be a pure function of the
    # kernel's identity (``digest``), never of registration order.  A separate
    # build reuses a kernel's .o by filename and then links against the symbol
    # it exports; if either depended on "is another same-named instance already
    # registered?", the full-model ``make objs`` cache and a per-block design
    # would disagree (one emits the unprefixed name, the other the prefixed
    # one), breaking .o reuse — undefined symbol at link, or a missing-file
    # copy.  ``digest`` is a pure function of ``cache_key``, so prefixing every
    # parameterized variant unconditionally keeps the symbol and the
    # suffix-form ``f"{func_name}_{digest}.o"`` filename identical across builds.
    # When ``digest`` is None (unparameterized default-name kernel, or a pinned
    # ``shared_object_file_name``) this leaves the symbol unprefixed.
    #
    # Chess exception: the symbol prefix is applied post-compile via
    # ``llvm-objcopy --redefine-sym`` (see compile_external_kernel), which only
    # understands ELF — it corrupts xchesscc-produced objects ("Invalid section
    # index ... when converting EOL-table").  So a chess kernel must NOT carry a
    # symbol_prefix; it keeps the bare symbol baked into its .cc.  That is safe
    # only while at most one variant of a given chess kernel name exists in a
    # design (two would export the same bare symbol and collide at link).  Guard
    # that invariant loudly here rather than letting it surface as an opaque
    # duplicate-symbol link error.  The .o *filename* stays the deterministic
    # suffix form regardless — only the symbol rename is skipped.
    if use_chess:
        # ``cache_key`` layout: (func_name, source_path, arg_keys, flags, chess).
        # A prior chess entry with the same func_name but any other field
        # different is a genuine second variant that we cannot disambiguate.
        for other_key in _EXTERN_CACHE:
            if (
                other_key[0] == func_name
                and other_key[4] is True
                and other_key != cache_key
            ):
                raise ValueError(
                    f"Chess kernel '{func_name}' already has a different "
                    f"parameterization registered.  Chess (.o) objects cannot "
                    f"be symbol-renamed (llvm-objcopy corrupts them), so two "
                    f"variants of the same chess kernel name would export the "
                    f"same symbol and collide at link.  Give one a distinct "
                    f"`name=` (or build it with Peano)."
                )
        symbol_prefix = None
    else:
        symbol_prefix = None if inline else digest

    extern = ExternalFunction(
        func_name,
        object_file_name=object_file_name,
        source_file=str(source_path),
        arg_types=arg_types,
        include_dirs=_include_dirs(),
        compile_flags=list(flags_tuple),
        symbol_prefix=symbol_prefix,
        use_chess=use_chess,
        inline=inline,
    )
    extern.contract = contract
    _EXTERN_CACHE[cache_key] = extern
    return extern
