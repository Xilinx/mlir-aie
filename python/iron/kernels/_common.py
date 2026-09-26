# kernels/_common.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Shared helpers for the kernels submodules."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, TypeVar, get_args, get_origin, overload

import numpy as np
from aie.helpers.npdtypes import (
    NpuDType,
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
)
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, InOut, Out
from aie.utils.verify import Tolerance


class Param:
    """Read-only test-fixture parameter; its ABI determines scalar or tensor.

    The generic harness holds its value fixed across calls. This is not a C++
    operand lifetime: direct designs may pass a new value on every kernel call.
    """


_ROLES = (In, Out, InOut, Param)


def _is_tensor_type(arg_type):
    return get_origin(arg_type) is np.ndarray


@dataclass(frozen=True)
class TensorLayout:
    """How a kernel wants one tensor operand laid out.

    ``shape`` is the logical tile. ``pack`` and ``unpack`` are the reversible
    host codec between ``(calls, *shape)`` and ``(calls, storage_elements)``;
    identity is the default. ``stream`` is the DMA transform
    (``dims_to_stream``) a design applies on the hop that feeds this operand
    to the kernel or drains it, ``None`` when the operand streams as stored;
    ``block`` is the micro-tile the kernel consumes or produces, ``(r, s)``
    for an MMUL operand. The codec is built from the same two facts, so the
    host and the design agree by construction. None of this is an algorithm
    or a whole-problem iteration schedule.
    """

    shape: tuple[int, ...]
    pack: Callable | None = None
    unpack: Callable | None = None
    stream: list | None = None
    block: tuple[int, ...] | None = None

    def encode(self, values):
        values = np.asarray(values).reshape(-1, *self.shape)
        return self.pack(values) if self.pack else values.reshape(len(values), -1)

    def decode(self, values, *, calls=1):
        values = np.asarray(values).reshape(calls, -1)
        return (
            self.unpack(values) if self.unpack else values.reshape(calls, *self.shape)
        )


@dataclass(frozen=True)
class Trace:
    """How a kernel's ``event0()``/``event1()`` markers bracket one call.

    ``Trace.whole_call()``: one pair brackets every call of the entry symbol
    and nothing it calls emits another, so each trace interval is one call.
    ``Trace.none(reason)``: a call emits no marker. ``Trace.partial(reason)``:
    markers exist but do not bracket each call exactly once (around an inner
    loop, or skipped on an early return), so intervals cannot be attributed
    to calls. ``test_kernel_trace_markers.py`` checks the declaration against
    the compiled IR of every library build.
    """

    shape: str
    reason: str | None = None

    def __post_init__(self):
        if self.shape not in ("whole_call", "none", "partial"):
            raise ValueError(f"unknown trace shape {self.shape!r}")
        if (self.shape == "whole_call") != (self.reason is None):
            raise ValueError("only an untimed trace shape carries a reason")

    @classmethod
    def whole_call(cls):
        return cls("whole_call")

    @classmethod
    def none(cls, reason: str):
        return cls("none", reason)

    @classmethod
    def partial(cls, reason: str):
        return cls("partial", reason)


@dataclass(frozen=True)
class KernelContract:
    """What a kernel computes, declared next to the factory that builds it.

    ``arg_types`` fixes each argument's shape and dtype; the contract adds
    what types cannot say, so ``aie.iron.algorithms.kernel_design`` can
    build, run and judge any factory from this one declaration.

    Attributes:
        roles: ``In``, ``Out``, ``InOut`` or ``Param`` per argument (the
            first three are the ``@iron.jit`` markers). ``InOut`` is
            accumulated into, so it needs an initializer. ``Param`` is a
            scalar or a read-only tensor the generic builder holds fixed
            across its calls; the argument type decides which. Several
            outputs are allowed, in argument order.
        reference: The host implementation, and the arithmetic model (a
            saturating kernel's reference clips). Called with every unbound
            non-output argument in order: ``In`` tiles as ``(calls, n)``
            arrays, ``Param`` values as arrays or numbers. Returns the
            output for all calls, a tuple for several outputs. ``None``
            builds the kernel but does not judge it.
        tolerance: How close the device must come; ``None`` is
            ``Tolerance.default_for`` the output dtype.
        ops_per_call: Arithmetic operations per call; ``None`` means one
            per output element.
        out_valid: Meaningful leading elements of a DMA-padded output tile;
            ``None`` means the whole tile.
        sample: ``sample(rng, calls) -> list[np.ndarray]`` for inputs with
            structure a dtype cannot express; ``None`` draws random data.
        acc_dtype: The accumulator type, or ``None`` when nothing
            accumulates. With ``reduction`` it bounds the inputs so the
            accumulator cannot overflow.
        reduction: Terms summed into one output element per call; ``None``
            means one.
        setup: A kernel to run once on the core first (``conv_even`` sets
            the rounding mode a bf16 store needs); ``None`` when the source
            sets its own mode or narrows nothing. A Worker handed the kernel
            calls it before its loop.
        stack_bytes: Core stack a Worker calling this kernel needs, when
            more than the target's default. Say where the number came from.
        unsupported: Why the builder cannot run this kernel, or ``None``. A
            kernel with no output argument (a cascade PUT half) says so here.
        layouts: A ``TensorLayout`` per argument; ``None`` is identity.
        parameter_bindings: ``(index, value)`` pairs fixing ``Param``
            operands, counts included; the rest come from the caller.
        initializers: ``(index, factory)`` pairs for ``InOut`` arguments;
            ``factory(fn)`` returns the kernel that initializes the buffer.
        out_offset: ``(index, step)`` for a kernel that writes ``step``
            elements of its one ``Out`` per call, at the offset it reads
            from bound scalar ``Param`` ``index``. The builder then hands
            every call the same output tile, passes ``call * step`` as the
            offset and drains the tile once, so the calls must fill it
            exactly. ``None``: each call writes a whole tile of its own.
        trace: The ``Trace`` shape of the kernel's markers. Every
            library factory declares one; ``None`` (undeclared) is only for
            ad-hoc kernels, and ``cycles_per_call`` refuses it.
        uses_lut: Whether the kernel gathers through an ``aie::lut<4>`` table
            pair, so a build should verify the two tables land in different
            banks. Set it on the contract, not per source file: the LUT often
            comes in through a header (``lut_based_ops.h``, ``lut_inv.h``).
        alignments: ``(index, bytes)`` pairs for arguments the kernel loads
            as whole vectors from their start, so they must begin at a
            multiple of ``bytes``. A call handed a ``memref.view`` at a
            constant offset that breaks one raises.

    Overflow, rounding and NaN handling are not declared twice: the
    reference is the arithmetic model and the tolerance the slack against it.
    """

    roles: tuple[type, ...]
    reference: Callable[..., np.ndarray | tuple[np.ndarray, ...]] | None = None
    tolerance: Tolerance | None = None
    ops_per_call: int | None = None
    out_valid: int | None = None
    sample: Callable[..., list] | None = None
    acc_dtype: type | None = None
    reduction: int | None = None
    setup: Callable[[], object] | None = None
    stack_bytes: int | None = None
    unsupported: str | None = None
    layouts: tuple[TensorLayout | None, ...] = ()
    parameter_bindings: tuple[tuple[int, object], ...] = ()
    initializers: tuple[tuple[int, Callable], ...] = ()
    out_offset: tuple[int, int] | None = None
    trace: Trace | None = None
    uses_lut: bool = False
    alignments: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        bad = [r for r in self.roles if r not in _ROLES]
        if bad:
            names = ", ".join(r.__name__ for r in _ROLES)
            raise ValueError(f"unknown kernel argument role(s) {bad}; use {names}")
        # A kernel with no data arguments at all (set_rounding sets core state)
        # has nothing to be the output. A cascade PUT half has none either:
        # its result leaves on the cascade stream, which is not an argument,
        # so the builder cannot judge it and the contract must say so.
        n_out = self.roles.count(Out) + self.roles.count(InOut)
        if self.roles and not n_out and self.unsupported is None:
            raise ValueError("a kernel contract needs at least one Out or InOut role")
        if self.layouts and len(self.layouts) != len(self.roles):
            raise ValueError("layouts must have one entry per argument")
        bound = dict(self.parameter_bindings)
        if len(bound) != len(self.parameter_bindings) or any(
            not isinstance(i, (int, np.integer))
            or i < 0
            or i >= len(self.roles)
            or self.roles[i] is not Param
            for i in bound
        ):
            raise ValueError("parameter_bindings must name distinct Param arguments")
        initialized = dict(self.initializers)
        if len(initialized) != len(self.initializers) or any(
            i < 0 or i >= len(self.roles) or self.roles[i] is not InOut
            for i in initialized
        ):
            raise ValueError("initializers must name distinct InOut arguments")
        if self.out_offset is not None:
            index, step = self.out_offset
            if index not in bound or self.roles.count(Out) != 1 or InOut in self.roles:
                raise ValueError(
                    "out_offset needs a bound Param offset and exactly one Out"
                )
            if step < 1:
                raise ValueError(f"out_offset step must be >= 1, got {step}")
        if self.reduction is not None and self.reduction < 1:
            raise ValueError(f"reduction must be >= 1, got {self.reduction}")
        if self.stack_bytes is not None and self.stack_bytes < 1:
            raise ValueError(f"stack_bytes must be >= 1, got {self.stack_bytes}")
        if self.unsupported is not None and not self.unsupported:
            raise ValueError("unsupported must be a reason, or None")
        if any(
            not 0 <= i < len(self.roles) or align < 1 for i, align in self.alignments
        ):
            raise ValueError("alignments must name arguments with positive byte counts")

    @property
    def out_indices(self) -> tuple[int, ...]:
        """Output argument positions, in declaration order."""
        return tuple(i for i, r in enumerate(self.roles) if r in (Out, InOut))

    @property
    def out_index(self) -> int:
        """Position of the one output, written (``Out``) or accumulated into (``InOut``).

        Raises for a kernel with several outputs: code that must handle any
        kernel reads ``out_indices``.
        """
        if len(self.out_indices) > 1:
            raise ValueError("multiple outputs: use out_indices")
        roles = list(self.roles)
        if Out in roles:
            return roles.index(Out)
        if InOut in roles:
            return roles.index(InOut)
        raise ValueError(
            "this kernel has no output argument (it emits on the cascade), so "
            "there is nothing to size or judge on its own"
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
        bound = dict(self.parameter_bindings)
        return [
            i
            for i, r in enumerate(self.roles)
            if r not in (Out, InOut) and i not in bound
        ]

    def validate_types(self, arg_types):
        """Validate contracts against NumPy tensor aliases and scalar dtypes.

        Raw MLIR types remain usable by ExternalFunction, but the host contract
        requires NumPy declarations for sampling, layouts and references.
        """
        if len(arg_types) != len(self.roles):
            raise ValueError("roles must have one entry per argument")
        bound = dict(self.parameter_bindings)
        for i, (role, arg_type) in enumerate(zip(self.roles, arg_types)):
            tensor = _is_tensor_type(arg_type)
            if tensor:
                try:
                    np_ndarray_type_get_shape(arg_type)
                    dtype = np_ndarray_type_get_dtype(arg_type)
                except (AssertionError, IndexError, TypeError) as exc:
                    raise ValueError(
                        f"argument {i}: expected np.ndarray[shape, np.dtype[dtype]]"
                    ) from exc
            else:
                dtype = arg_type
            if dtype not in get_args(NpuDType):
                raise ValueError(
                    f"argument {i}: kernel contracts require NumPy tensor aliases "
                    "or supported NumPy scalar dtypes"
                )
            if not tensor and role is not Param:
                raise ValueError(f"argument {i}: scalar arguments require Param")
            if self.layouts and self.layouts[i] is not None and not tensor:
                raise ValueError(f"argument {i}: layouts require tensor arguments")
            if i not in bound:
                continue
            value = bound[i]
            if tensor:
                value = np.asarray(value)
                layout = self.layouts[i] if self.layouts else None
                shape = (
                    layout.shape
                    if layout is not None
                    else np_ndarray_type_get_shape(arg_type)
                )
                if value.ndim == 0 or value.size != int(np.prod(shape)):
                    raise ValueError(
                        f"argument {i}: tensor parameter must contain {shape} elements"
                    )
            elif not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"argument {i}: expected scalar parameter")


@dataclass(frozen=True)
class ArchTraits:
    """What the kernel sources assume of one architecture.

    One row of ``aie_kernels/aie_arch.h``, which the C++ side reads;
    ``test_arch_traits.py`` compiles the two against each other.

    Attributes:
        name: The architecture, as ``resolve_target_arch`` names it.
        aie_arch: The compiler's ``__AIE_ARCH__``.
        device: The ``from_name`` device a factory models when none is bound.
        bf16_lanes: bf16 lanes in one vector multiply, the width the sources
            walk a buffer at. A tile has to be whole vectors of it.
        native_tanh: Has a tanh instruction; otherwise tanh reads a LUT.
        native_exp2: Has an exp2 instruction; otherwise exp2 is a polynomial.
        bfp16: Has the bfp16ebs8 block type.
        lut_16b_run: uint16 entries per bank run in an ``aie::lut`` table.
    """

    name: str
    aie_arch: int
    device: str
    bf16_lanes: int
    native_tanh: bool
    native_exp2: bool
    bfp16: bool
    lut_16b_run: int


ARCH_TRAITS = {
    t.name: t
    for t in (
        ArchTraits("aie2", 20, "npu1", 16, False, False, False, 8),
        ArchTraits("aie2p", 21, "npu2", 32, True, True, True, 16),
    )
}


def _detect_arch() -> str:
    """Return the bound device's architecture, or ``'aie2'`` when none is bound.

    Raises:
        RuntimeError: When the bound device's architecture has no kernels.
    """
    from aie.utils import get_current_device
    from aie.utils.compile.utils import resolve_target_arch

    return resolve_target_arch(get_current_device(probe_runtime=False))


def _arch_traits() -> ArchTraits:
    """Return the traits of the architecture ``_detect_arch`` names."""
    return ARCH_TRAITS[_detect_arch()]


def _portable() -> bool:
    """Whether ``AIE_KERNELS_PORTABLE=1`` asks for every kernel's untuned branch."""
    return os.environ.get("AIE_KERNELS_PORTABLE") == "1"


def _tuned_arch() -> str | None:
    """Return the architecture whose ``AIE_TUNED_*`` code the sources build, or None.

    A factory choice that follows the code of one branch -- a stack size, a
    tolerance, a reference model -- keys on this rather than on
    ``_detect_arch``, so that it pairs with the branch built when
    ``_portable()`` holds.
    """
    return None if _portable() else _detect_arch()


_T = TypeVar("_T")


def _by_tuned_arch(table: Mapping[str, _T], default: _T | None = None) -> _T | None:
    """Return ``table``'s entry for ``_tuned_arch()``, else ``default``."""
    arch = _tuned_arch()
    return default if arch is None else table.get(arch, default)


def _portable_flags() -> tuple[str, ...]:
    """Return the compile flags that select the branch ``_tuned_arch`` names."""
    return ("-DAIE_KERNELS_PORTABLE",) if _portable() else ()


def _kernel_source(relpath: str) -> Path:
    """Return the absolute path to a kernel source file.

    Args:
        relpath: Path under ``aie_kernels/``, e.g. ``'eltwise/scale.cc'``.

    Raises:
        FileNotFoundError: When the source file does not exist.
    """
    from aie.utils import config

    path = Path(config.aie_kernels_dir()) / relpath
    if not path.exists():
        raise FileNotFoundError(f"Kernel source {path} not found")
    return path


def _include_dirs() -> list[str]:
    """Return the standard include directory list for kernel compilation."""
    from aie.utils import config

    return [config.cxx_header_path()]


def _runtime_lib_include(arch: str | None = None) -> str:
    """Return the ``aie_runtime_lib/<ARCH>`` include directory.

    It holds the LUT sources and ``aie_bank_placement.h``, whose portable
    ``AIE_BANK_A``-``AIE_BANK_D`` macros a kernel needs to pin a static to a
    bank.
    """
    from aie.utils import config

    return str(Path(config.aie_runtime_lib_dir()) / (arch or _detect_arch()).upper())


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


def _require_vector_alignment(
    factory_name: str,
    elems: int,
    per_iter: int,
    *,
    param: str = "tile_size",
) -> None:
    """Require a positive whole number of vectors for a loop without a tail."""
    if elems <= 0:
        raise ValueError(f"{factory_name}() {param} must be positive, got {elems}.")
    if elems % per_iter:
        raise ValueError(
            f"{factory_name}() {param}={elems} is not a multiple of the "
            f"kernel's {per_iter}-element vector step; the tail iteration "
            f"would run past the tile."
        )


def _device():
    """Return the bound device, or the default device of the detected architecture.

    Factories run without a device bound (``_detect_arch`` falls back to
    aie2); anything that reads the target model goes through here so that
    fallback is the same everywhere.
    """
    from aie.iron.device import from_name
    from aie.utils import get_current_device

    device = get_current_device(probe_runtime=False)
    if device is None:
        device = from_name(_arch_traits().device)
    return device


def _min_dma_aligned_elems(dtype) -> int:
    """Return the fewest elements whose byte size the shim DMA can address.

    The DMA moves whole address-generation granules (32 bits on aie2 and
    aie2p, from the target model). A 1-element output tile is fine for
    ``int32`` but only 2 bytes for ``bfloat16``, so a kernel whose C++ side
    writes a single value still needs a tile type with enough elements.
    """
    align = _device().address_gen_granularity // 8
    itemsize = np.dtype(dtype).itemsize
    return max(1, (align + itemsize - 1) // itemsize)


def _arg_type_key(t):
    """Hashable key for one entry of ``arg_types`` (part of a kernel's identity)."""
    if hasattr(t, "__args__"):
        # np.ndarray[(shape,), np.dtype[T]]
        shape = t.__args__[0]
        inner = t.__args__[1]
        dtype = inner.__args__[0] if hasattr(inner, "__args__") else inner
        return ("ndarray", tuple(shape), str(dtype))
    return repr(t)


_KernelT = TypeVar("_KernelT", bound=ExternalFunction)


@overload
def _make_extern(
    func_name: str,
    source_path: "Path | str",
    arg_types: list,
    *,
    compile_flags: list[str] | None = None,
    use_chess: bool = False,
    inline: bool = False,
    object_file_name: str | None = None,
    contract: KernelContract | None = None,
    cls: type[_KernelT],
    include_dirs: list[str] | None = None,
) -> _KernelT: ...


@overload
def _make_extern(
    func_name: str,
    source_path: "Path | str",
    arg_types: list,
    *,
    compile_flags: list[str] | None = None,
    use_chess: bool = False,
    inline: bool = False,
    object_file_name: str | None = None,
    contract: KernelContract | None = None,
    include_dirs: list[str] | None = None,
) -> ExternalFunction: ...


def _make_extern(
    func_name: str,
    source_path: "Path | str",
    arg_types: list,
    *,
    compile_flags: list[str] | None = None,
    use_chess: bool = False,
    inline: bool = False,
    object_file_name: str | None = None,
    contract: KernelContract | None = None,
    cls: type[ExternalFunction] = ExternalFunction,
    include_dirs: list[str] | None = None,
) -> ExternalFunction:
    """Construct an ExternalFunction with the standard include_dirs.

    ``contract`` (a ``KernelContract``) is what harnesses and tests read
    to build, run and judge the kernel generically; every factory passes
    one. ``cls`` is the class to construct, for factories whose kernels have
    more to say than a plain ``ExternalFunction`` (``linalg.MatrixKernel``).
    ``include_dirs`` replaces the standard include path.

    ``inline`` uses Peano's always-inline LLVM IR and merge linking. Inline
    factories must use distinct C++ symbol names for distinct variants because
    LLVM IR cannot use the object-file symbol-prefix mechanism.

    Equal parameters give equal kernels, which share one object and one
    declaration. Different parameterizations get distinct
    ``object_file_name``s — the latter is auto-suffixed with a short
    digest of the identity so per-parameterization .o files don't
    overwrite each other on disk.  The default ``<name>.o`` is preserved
    when ``compile_flags`` is empty AND ``use_chess`` is False (no
    parameterization to disambiguate).

    ``use_chess`` selects the Chess (xchesscc) compiler instead of Peano
    for this kernel's .o build.  See
    `ExternalFunction` for the design-level
    contract: all EFs in a single ``@iron.jit`` design must share the
    same toolchain choice (mixed peano/chess is rejected at compile
    time).

    ``object_file_name`` names the output explicitly instead of deriving it
    from the identity. Two factories that bind different symbols of the
    same translation unit with identical compile flags can name the same
    object, and ``ExternalFunction``
    then gives both one ``KernelObject``: one compile, one link artifact,
    where separate digest-named objects would each carry every symbol of
    the ``.cc`` and collide at link.
    """
    flags_tuple = tuple(compile_flags or []) + _portable_flags()
    arg_keys = tuple(_arg_type_key(t) for t in arg_types)
    # One source serves every arch, so the arch is part of the kernel's identity.
    identity = (
        func_name,
        str(source_path),
        arg_keys,
        flags_tuple,
        use_chess,
        _detect_arch(),
    )
    if inline:
        if use_chess:
            raise ValueError("inline kernels require Peano, not Chess")
        # Keep existing object artifact identities unchanged.
        identity += ("inline",)

    # The object_file_name suffix must distinguish every distinct identity,
    # not just compile_flags — otherwise two helper calls with the same
    # name + flags but different source / arg_types would generate
    # ExternalFunctions with identical .o filenames and trip the collision
    # check in ExternalFunction.__init__ (or, if both passed it, would
    # silently overwrite each other on disk).
    if object_file_name is not None:
        # An explicit name is the whole identity of the object: no digest
        # suffix, and (below) no symbol prefix, so every binding of that
        # translation unit resolves to the one KernelObject.
        digest = None
    elif flags_tuple or arg_keys or str(source_path):
        # 8 hex chars of sha256 — short enough not to bloat MLIR strings,
        # wide enough that the chance of two distinct identities colliding
        # is vanishingly small (~2^-32).
        digest = hashlib.sha256(repr(identity).encode()).hexdigest()[:8]
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
    # copy.  ``digest`` is a pure function of ``identity``, so prefixing every
    # parameterized variant unconditionally keeps the symbol and the
    # suffix-form ``f"{func_name}_{digest}.o"`` filename identical across builds.
    # When ``digest`` is None (unparameterized default-name kernel, or an
    # explicit ``object_file_name``) this leaves the symbol unprefixed.
    #
    # Chess exception: the symbol prefix is applied post-compile via
    # ``llvm-objcopy --redefine-sym`` (see compile_external_kernel), which only
    # understands ELF — it corrupts xchesscc-produced objects ("Invalid section
    # index ... when converting EOL-table").  So a chess kernel must NOT carry a
    # symbol_prefix; it keeps the bare symbol baked into its .cc, so two
    # variants of one chess kernel cannot share a design: resolving the second
    # finds the first's declaration with a different object and raises.  The .o
    # *filename* stays the deterministic suffix form regardless -- only the
    # symbol rename is skipped.
    symbol_prefix = None if use_chess or inline else digest

    extern = cls(
        func_name,
        object_file_name=object_file_name,
        source_file=str(source_path),
        arg_types=arg_types,
        include_dirs=include_dirs or _include_dirs(),
        compile_flags=list(flags_tuple),
        symbol_prefix=symbol_prefix,
        use_chess=use_chess,
        inline=inline,
        contract=contract,
    )
    if contract is not None:
        contract.validate_types(extern.arg_types())
    # The factory picked its source, flags and contract for this arch; a
    # design that compiles it for another one fails there, not in Peano.
    extern.built_for_arch = _detect_arch()
    return extern
