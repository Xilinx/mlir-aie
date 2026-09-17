# kernel.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Kernel and ExternalFunction: wrappers for pre-compiled and C++ AIE compute kernels."""

import hashlib
import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects import memref  # pyright: ignore[reportAttributeAccessIssue]
from ..dialects.aie import external_func
from ..extras.dialects.func import FuncOp  # pyright: ignore[reportMissingImports]
from ..helpers.dialects.func import call
from .buffer import Buffer
from .resolvable import Resolvable

logger = logging.getLogger(__name__)


class DesignShape(Enum):
    """The dataflow a generic design has to build to run a kernel.

    A kernel's arguments say what it consumes, not how a design feeds it.
    ``STREAM`` tiles every input through an ObjectFifo, one element per call;
    the matrix shapes walk operands in the micro-tile blocking the kernel was
    compiled for, which is a property of the kernel, not of its signature.
    """

    STREAM = "stream"
    MATMUL = "matmul"
    MATVEC = "matvec"


def _as_dtype(dt):
    """``np.dtype(dt)``, or ``dt`` itself for a block type numpy has no dtype for.

    Older numpy versions coerce custom ``np.generic`` subclasses to void rather
    than rejecting them, so preserve the block formats before conversion.
    """
    from ..helpers.util import v8bfp16ebs8, v16bfp16ebs16

    if dt is v8bfp16ebs8 or dt is v16bfp16ebs16:
        return dt
    try:
        return np.dtype(dt)
    except TypeError:
        return dt


def _is_contiguous_row_major(mr):
    """Return True iff ``mr`` is fully-static row-major contiguous at offset 0.

    Required before ``memref.collapse_shape`` (UB on non-contiguous dims).
    """
    if any(d < 0 for d in mr.shape):
        return False
    try:
        strides, offset = mr.get_strides_and_offset()
    except Exception:
        return False
    if offset != 0:
        return False
    expected = []
    running = 1
    for d in reversed(mr.shape):
        expected.append(running)
        running *= d
    expected.reverse()
    return list(strides) == expected


def _maybe_collapse_to_match(arg, expected_ty):
    """Bridge an N-D contiguous memref arg to a 1-D kernel signature.

    Uses ``memref.collapse_shape``. Iron L1 buffers are multi-dim (e.g.
    ``memref<64x64xi16>``) but ``aie.iron.kernels.X`` helpers declare
    flat 1-D args; without this adapter MLIR rejects the call even though
    bytes line up. Aliases storage — no copy emitted. Returns ``arg``
    unchanged for any case that isn't safely collapsible, so real bugs
    still surface in MLIR verification.
    """
    if not isinstance(arg, ir.Value):
        return arg
    arg_ty = arg.type
    if not (
        isinstance(arg_ty, ir.MemRefType) and isinstance(expected_ty, ir.MemRefType)
    ):
        return arg
    arg_mr = arg_ty
    exp_mr = expected_ty
    if arg_mr == exp_mr:
        return arg
    if arg_mr.element_type != exp_mr.element_type:
        return arg
    if exp_mr.rank != 1 or arg_mr.rank < 1:
        return arg
    if any(d < 0 for d in exp_mr.shape):
        return arg
    if not _is_contiguous_row_major(arg_mr):
        return arg
    arg_count = 1
    for d in arg_mr.shape:
        arg_count *= d
    if arg_count != exp_mr.shape[0]:
        return arg
    # All N input dims collapse into the single output dim.
    reassociation = [list(range(arg_mr.rank))]
    return memref.collapse_shape(exp_mr, arg, reassociation)


class BaseKernel(Resolvable):
    """Base class for AIE core functions that resolve to a func.func declaration.

    Subclasses:
        Kernel: wraps a pre-compiled object file.
        ExternalFunction: compiles C/C++ source at JIT time.
    """

    def __init__(
        self,
        name: str,
        arg_types: list[type[np.ndarray] | np.dtype] | None = None,
    ):
        """Construct a BaseKernel.

        Args:
            name: Symbol name of the function.
            arg_types: Type signature of the function arguments.  Defaults to None (empty list).
        """
        if not name:
            raise ValueError("Kernel name cannot be empty.")
        self._name = name
        # The declaration as written (numpy shapes and dtypes). Resolving the
        # kernel builds MLIR types from it without disturbing it, so this stays
        # readable before and after a build.
        self._arg_types = list(arg_types) if arg_types is not None else []
        self._op: FuncOp | None = None

    @property
    def name(self) -> str:
        """Symbol name of the function as it appears in the object file."""
        return self._name

    def _resolve_arg(self, arg_index: int):
        """Validate ``arg_index`` and return the underlying type entry."""
        if not self._arg_types:
            raise ValueError("No argument types defined.")
        if arg_index >= len(self._arg_types):
            raise ValueError(
                f"Argument index {arg_index} out of range "
                f"(max: {len(self._arg_types) - 1})"
            )
        return self._arg_types[arg_index]

    def arg_shape(self, arg_index: int = 0) -> tuple[int, ...]:
        """Return the shape tuple of the array argument at `arg_index`.

        Works for both `np.ndarray[(...,), np.dtype[T]]` parameterized
        types (the canonical IRON kernel signature) and MLIR MemRefType
        operands.

        Args:
            arg_index: Index into `arg_types`. Defaults to 0.

        Raises:
            ValueError: When `arg_index` is out of range or the
                argument at that index is not an array type.
        """
        arg = self._resolve_arg(arg_index)
        type_args = getattr(arg, "__args__", None)
        if type_args is not None and len(type_args) > 0:
            shape_arg = type_args[0]
            if isinstance(shape_arg, tuple):
                return shape_arg
        shape = getattr(arg, "shape", None)
        if shape is not None:
            return tuple(shape)
        raise ValueError(
            f"Argument {arg_index} does not have a shape or is not an array type."
        )

    def arg_dtype(self, arg_index: int = 0):
        """Return the numpy dtype of the array argument at `arg_index`.

        Args:
            arg_index: Index into `arg_types`. Defaults to 0.

        Raises:
            ValueError: When `arg_index` is out of range or the
                argument at that index is not an array type.
        """
        arg = self._resolve_arg(arg_index)
        type_args = getattr(arg, "__args__", None)
        if type_args is not None and len(type_args) >= 2:
            dt = type_args[1]
            dt_args = getattr(dt, "__args__", None)
            return _as_dtype(dt_args[0] if dt_args is not None else dt)
        dtype = getattr(arg, "dtype", None)
        if dtype is not None:
            return _as_dtype(dtype)
        raise ValueError(
            f"Argument {arg_index} does not have a dtype or is not an array type."
        )

    def tile_size(self, arg_index: int = 0) -> int:
        """Return the first dimension of the array argument at `arg_index`.

        Convenience wrapper over
        [`arg_shape`][iron.kernel.BaseKernel.arg_shape] for the common case of
        a 1-D buffer argument. `tile_size(i)` is equivalent to
        `arg_shape(i)[0]`.

        Args:
            arg_index: Index into `arg_types`. Defaults to 0.
        """
        shape = self.arg_shape(arg_index)
        if len(shape) == 0:
            raise ValueError(
                f"Argument {arg_index} does not have a shape or is not an array type."
            )
        return shape[0]

    def arg_types(self) -> list:
        """Return the argument types as declared: ``np.ndarray[shape, dtype]`` / scalars.

        A copy, and stable: resolving the kernel builds MLIR types from these
        without replacing them, so a memoized kernel describes itself the same
        way before and after a build.
        """
        return self._arg_types.copy()

    def __call__(self, *args, **kwargs):
        """Emit a func.call to this kernel, validating argument count.

        Each argument is passed through `_maybe_collapse_to_match`
        before the call. This silently inserts a `memref.collapse_shape`
        when an N-D contiguous memref arg is being fed into a 1-D kernel
        signature with the same element count and dtype — the typical case
        when an IRON design holds 2-D ObjectFifo elements but the
        `iron.kernels.X` helper declares a flat 1-D arg. See that
        helper's docstring for the full set of conditions. Real shape /
        dtype mismatches still fail at MLIR verification time.

        `**kwargs` are forwarded to the underlying `func.call` builder
        (typically `loc=`, `ip=` for MLIR location / insertion point).
        """
        if not self._op:
            raise ValueError("Kernel must be resolved before it can be called.")
        if len(args) != len(self._arg_types):
            raise ValueError(
                f"Kernel '{self._name}' expects {len(self._arg_types)} "
                f"argument(s), but {len(args)} were provided."
            )
        arg_ops = [a.op if isinstance(a, Buffer) else a for a in args]
        expected_input_types = self._op.function_type.value.inputs
        adapted = [
            _maybe_collapse_to_match(a, expected_ty)
            for a, expected_ty in zip(arg_ops, expected_input_types)
        ]
        call(self._op, adapted, **kwargs)


@dataclass(frozen=True)
class _KernelSource:
    source_file: str | None
    source_string: str | None
    source_digest: str
    include_dirs: tuple[str, ...]
    compile_flags: tuple[str, ...]
    use_chess: bool
    symbol_prefix: str | None
    inline_symbol: str | None


@dataclass(frozen=True, eq=False)
class KernelObject:
    """One link artifact, shared by every kernel that binds one of its symbols.

    A prebuilt object only needs a filename and link policy. ExternalFunction
    supplies the immutable source recipe; compilation state belongs here, not
    to any particular exported function.
    """

    name: str
    link_with_mode: str | None = None
    _source: _KernelSource | None = field(default=None, repr=False)
    _compiled_dirs: set[str] = field(default_factory=set, repr=False)


class Kernel(BaseKernel):
    """An AIE core function backed by a pre-compiled object file.

    Use [`ExternalFunction`][iron.ExternalFunction] instead when you want to
    compile from C/C++ source at JIT time.

    `resolve()` emits a `func.func private` declaration with a
    `link_with` attribute naming `object_file_name`. The
    `aie-assign-core-link-files` pass propagates this into the CoreOp's
    `link_files` attribute so the linker knows which file to include.

    `link_with_mode` selects how that artifact is consumed: the default
    (None) object-links it, while `"merge"` asks aiecc to llvm-link it into
    the core's LLVM module before codegen.  The mode is explicit metadata --
    it is never inferred from the file suffix.
    """

    def __init__(
        self,
        name: str,
        object_file_name: str | KernelObject,
        arg_types: list[type[np.ndarray] | np.dtype] | None = None,
        *,
        link_with_mode: str | None = None,
        stack_size_override: int | None = None,
    ) -> None:
        """Construct a Kernel backed by a pre-compiled object file.

        Args:
            name: Symbol name of the function as it appears in the object file.
            object_file_name: Filename (e.g. ``"add_one.o"``) or shared
                ``KernelObject`` of the pre-compiled object file. Must be on
                the linker search path at compile time.
            arg_types: Type signature of the function arguments.  Defaults to None (empty list).
            link_with_mode: Optional link policy emitted alongside
                ``link_with``.  ``"merge"`` routes the artifact through aiecc's
                ``llvm-link`` merge path; None (the default) object-links it.
            stack_size_override: Declared upper bound, in bytes, on the stack
                that this kernel's call subtree uses. Set it for recursion, for
                an indirect call, or for a ``link_with_mode="merge"`` kernel,
                which aiecc's stack analysis reads as part of the core rather
                than as a separate object. See
                [`Kernel.stack_size_override`][iron.kernel.Kernel.stack_size_override].
        """
        super().__init__(name, arg_types)
        if isinstance(object_file_name, KernelObject):
            if (
                link_with_mode is not None
                and link_with_mode != object_file_name.link_with_mode
            ):
                raise ValueError(
                    "link_with_mode conflicts with the shared KernelObject"
                )
            self._object_file = object_file_name
        else:
            self._object_file = KernelObject(object_file_name, link_with_mode)
        self._stack_size_override = stack_size_override

    @property
    def object_file(self) -> KernelObject:
        """The artifact owner, also shared by sibling symbol bindings."""
        return self._object_file

    @property
    def _object_file_name(self) -> str:
        return self.object_file.name

    @property
    def _link_with_mode(self) -> str | None:
        return self.object_file.link_with_mode

    @property
    def object_file_name(self) -> str:
        """Filename of the compiled object file."""
        return self._object_file_name

    @property
    def link_with_mode(self) -> str | None:
        """Link policy emitted with ``link_with``, or None for object linking."""
        return self._link_with_mode

    @property
    def stack_size_override(self) -> int | None:
        """Declared upper bound on the stack that this kernel's call subtree uses.

        With ``None``, aiecc's analysis computes the bound. An explicit value
        replaces that computed bound, even when it is smaller: it is a
        declaration, and ``0`` is legal. See
        [Core Data Memory](../../programming_guide/core_data_memory.md).
        """
        return self._stack_size_override

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self.object_file._source is not None:
            # JIT clears its discovery registry before generating a design.
            # Re-register the artifact even when only a sibling binding survives.
            ExternalFunction._register_object(self)
        if not self._op:
            self._op = external_func(
                self._name,
                inputs=self._arg_types,
                link_with=self._object_file_name,
                link_with_mode=self._link_with_mode,
                stack_size_override=self._stack_size_override,
            )


class ExternalFunction(Kernel):
    """An AIE core function compiled from C/C++ source at JIT time.

    Each instance is registered in `_instances` at construction time so that
    the `@jit` decorator can discover and compile all source files before
    invoking the MLIR compilation pipeline. `_instances` is cleared at the
    start of each `@jit` call to prevent stale registrations from a previous
    (possibly failed) run.

    Use the base [`Kernel`][iron.Kernel] class instead when you have a
    pre-built object file.
    """

    _instances: set = set()  # Registry of all live ExternalFunction instances.

    @classmethod
    def _register_object(cls, kernel: Kernel) -> None:
        if isinstance(kernel, cls):
            cls._instances.add(kernel)
            return
        if any(f.object_file is kernel.object_file for f in cls._instances):
            return
        # A discovery binding references the existing owner; it does not rebuild
        # the recipe or retain the ExternalFunction that originally created it.
        binding = cls.__new__(cls)
        BaseKernel.__init__(binding, kernel.name, kernel.arg_types())
        binding._object_file = kernel.object_file
        binding._stack_size_override = kernel.stack_size_override
        recipe = binding._recipe
        prefix = f"{recipe.symbol_prefix}_" if recipe.symbol_prefix else ""
        binding._original_name = recipe.inline_symbol or kernel.name.removeprefix(
            prefix
        )
        binding._cached_digest = None
        binding.also = SimpleNamespace()
        cls._instances.add(binding)

    # Optional metadata the kernel factories attach: the contract
    # (aie.iron.kernels.KernelContract) and the matmul layout facts.
    # Typed Any rather than KernelContract: pyright analyses the sources and
    # the staged package as two module trees, so naming the class here would
    # make the factories' own KernelContract a different type.
    contract: Any = None
    mac_dims: tuple
    dims: tuple
    stream_dims: Any  # kernels.linalg.StreamDimsABC
    b_col_maj: bool
    c_col_maj: bool
    a_dims_from_stream: object
    #: How a generic design must feed this kernel.
    design_shape: DesignShape = DesignShape.STREAM

    def _require_contract(self):
        if self.contract is None:
            raise ValueError(
                f"kernel '{self.name}' declares no contract; add a KernelContract to "
                "its factory (roles, reference, tolerance) so it can be described, "
                "built and checked"
            )
        return self.contract

    def param_values(self, inputs: list) -> list:
        """Pick the ``Param`` arrays out of one logical input list.

        ``inputs`` is one array per ``In``/``Param`` argument in argument
        order. A design bakes ``Param`` arguments into core buffers rather
        than streaming them, so it needs them separately.
        """
        from aie.utils.compile.jit.markers import In, Param

        c = self._require_contract()
        positions = [i for i, r in enumerate(c.roles) if r in (In, Param)]
        return [np.asarray(a) for a, i in zip(inputs, positions) if c.roles[i] is Param]

    def input_limit(self, dtype, *, reduction: int | None = None) -> int | None:
        """Largest integer magnitude an input may take without overflowing.

        From the contract's ``acc_dtype`` and ``reduction`` (``reduction``
        overrides the per-call value, e.g. with the full ``K`` of a tiled
        matmul): with two or more multiplied inputs every product of two
        limits summed ``reduction`` times must fit the accumulator with a
        factor-4 margin; with one input the sum of ``reduction`` limits must.
        ``None`` for float inputs, or when the contract declares no
        accumulator.

        The output dtype does not bound this. What a kernel does when a
        result leaves the output range is its reference's to model, and
        clipping inputs to the output range would leave a requantising
        kernel's data near zero.
        """
        from aie.utils.compile.jit.markers import In, Param

        c = self._require_contract()
        dt = np.dtype(dtype)
        if not np.issubdtype(dt, np.integer):
            return None
        if c.acc_dtype is None or not np.issubdtype(np.dtype(c.acc_dtype), np.integer):
            return None
        n = reduction or c.reduction or 1
        budget = np.iinfo(c.acc_dtype).max // 4
        n_tensors = sum(1 for r in c.roles if r in (In, Param))
        limit = int(np.sqrt(budget // n)) if n_tensors >= 2 else budget // n
        return max(1, min(limit, int(np.iinfo(dt).max)))

    def expected(self, inputs: list, *, scalars: tuple = ()):
        """Return reference output(s), cast to each output argument's dtype."""
        from aie.helpers.util import v8bfp16ebs8
        from aie.utils.compile.jit.markers import Scalar

        c = self._require_contract()
        if c.reference is None:
            raise ValueError(f"{self.name}: contract has no reference")
        tensors, s = iter(inputs), iter(scalars)
        args = [
            next(s) if c.roles[i] is Scalar else next(tensors)
            for i in c.reference_indices()
        ]
        result = c.reference(*args)
        multiple = len(c.out_indices) > 1
        results = result if multiple else (result,)
        if multiple and (
            not isinstance(results, tuple) or len(results) != len(c.out_indices)
        ):
            raise ValueError("reference must return one tuple entry per output")
        outputs = []
        for i, value in zip(c.out_indices, results):
            dt = self.arg_dtype(i)
            outputs.append(
                np.asarray(value).astype(np.float32 if dt is v8bfp16ebs8 else dt)
            )
        return tuple(outputs) if multiple else outputs[0]

    def output_dtype(self, ref_dtype):
        """Host dtype(s) of the device output buffer(s).

        A bfp16ebs8 output arrives as packed bytes; everything else arrives in
        the reference's own dtype. Multiple outputs take and return tuples in
        output argument order.
        """
        import numpy as np
        from aie.helpers.util import v8bfp16ebs8

        outputs = self._require_contract().out_indices
        multiple = len(outputs) > 1
        dtypes = ref_dtype if multiple else (ref_dtype,)
        if multiple and (
            not isinstance(dtypes, (tuple, list)) or len(dtypes) != len(outputs)
        ):
            raise ValueError("provide one reference dtype per output")
        result = tuple(
            np.uint8 if self.arg_dtype(i) is v8bfp16ebs8 else dt
            for i, dt in zip(outputs, dtypes)
        )
        return result if multiple else result[0]

    def judge(self, got, ref, *, calls: int = 1, tolerance=None):
        """Compare a flat device output against a reference under the contract.

        Declared layouts decode each output into logical tiles. DMA padding
        is trimmed per call. Multiple outputs return a tuple of verdicts;
        callers must check all entries, not the truthiness of the tuple.
        """
        from aie.utils.verify import Tolerance, compare

        c = self._require_contract()
        multiple = len(c.out_indices) > 1
        actuals, references = (got, ref) if multiple else ((got,), (ref,))
        if (
            not isinstance(actuals, (tuple, list))
            or not isinstance(references, (tuple, list))
            or len(actuals) != len(c.out_indices)
            or len(references) != len(c.out_indices)
        ):
            raise ValueError("provide one actual and reference array per output")
        verdicts = []
        for i, actual, reference in zip(c.out_indices, actuals, references):
            layout = c.layouts[i] if c.layouts else None
            got = layout.decode(actual, calls=calls) if layout else np.asarray(actual)
            got, ref = got.reshape(calls, -1), np.asarray(reference).reshape(calls, -1)
            if c.out_valid is not None:
                got = got[:, : c.out_valid]
            verdicts.append(
                compare(
                    got,
                    ref,
                    tolerance or c.tolerance or Tolerance.default_for(ref.dtype),
                )
            )
        return tuple(verdicts) if multiple else verdicts[0]

    def siblings(self, **symbols: tuple) -> SimpleNamespace:
        """Bind other symbols exported by this kernel's own object file.

        A translation unit often exports more than the symbol this
        ExternalFunction declares -- ``mm.cc`` emits the ``zero_*`` that
        accumulation needs beside ``matmul_*``, ``cascade_mm.cc`` a get/put
        trio -- and binding them here avoids compiling the source a second
        time::

            fn.siblings(zero=("zero_i32", [c_ty]))
            fn.also.zero  # a Kernel over the same object file

        Each keyword names the attribute to bind under ``also`` and takes
        ``(symbol, arg_types)``, where ``symbol`` is the name as written in
        the source: when this kernel carries a ``symbol_prefix``, the
        object's symbols have all been prefixed to keep parameterizations
        apart (see ``aie.utils.compile.utils._prefix_symbols_in_object``),
        so each sibling is prefixed to match.

        Returns ``self.also``, which is always present, and empty for a
        kernel whose object exports nothing else.
        """
        prefix = getattr(self, "_symbol_prefix", None)
        vars(self.also).update(
            {
                attr: Kernel(
                    f"{prefix}_{symbol}" if prefix else symbol,
                    self.object_file,
                    arg_types,
                )
                for attr, (symbol, arg_types) in symbols.items()
            }
        )
        return self.also

    def __init__(
        self,
        name: str,
        object_file_name: str | None = None,
        source_file: str | None = None,
        source_string: str | None = None,
        arg_types: list[type[np.ndarray] | np.dtype] | None = None,
        include_dirs: list[str] | None = None,
        compile_flags: list[str] | None = None,
        *,
        symbol_prefix: str | None = None,
        use_chess: bool = False,
        inline: bool = False,
        stack_size_override: int | None = None,
    ) -> None:
        """Construct an ExternalFunction compiled from C/C++ source at JIT time.

        Args:
            name: Symbol name of the function as it will appear in the object
                file.
            object_file_name: Output artifact name. Defaults to
                ``<effective_name>.o``, or ``<effective_name>.ll`` with
                ``inline=True``. With ``inline=True`` an explicit name must end
                in ``.ll`` (textual LLVM IR) or ``.bc`` (bitcode) -- that suffix
                selects the emitted format -- and is otherwise rejected.
            source_file: Path to a C/C++ source file on disk.  Mutually
                exclusive with ``source_string``.
            source_string: Inline C/C++ source code.  Mutually exclusive with
                ``source_file``.
            arg_types: Type signature of the function arguments.  Defaults to
                None (empty list).
            include_dirs: Additional ``-I`` directories passed to the chosen
                compiler (Peano by default; xchesscc when ``use_chess=True``).
                Defaults to None (empty list).
            compile_flags: Additional flags passed verbatim to the chosen
                compiler.  Defaults to None (empty list).
            symbol_prefix: Optional prefix for the exported symbol name.  When
                set, the effective symbol name becomes ``<symbol_prefix>_<name>``
                and the object file is named accordingly.  The original name is
                preserved in ``_original_name`` for source file naming.
            use_chess: When ``True``, this ExternalFunction's source is
                compiled with ``xchesscc_wrapper`` instead of Peano's
                ``clang++``.  The JIT compile orchestration auto-detects the
                design-level toolchain from the registered EFs and switches
                aiecc's front-end accordingly; mixing chess + peano EFs in
                one design is rejected loudly because aiecc only invokes one
                front-end per compile.
            inline: When True, compile the kernel to ``alwaysinline`` LLVM IR
                (``.ll``) and declare it with ``link_with_mode = "merge"`` so
                aiecc llvm-links it into the core and inlines it, instead of
                object-linking a separate ``.o``. Removes the ``func.call``
                boundary and the separate object. Peano path only (the
                Chess/xchesscc toolchain cannot llvm-link).
            stack_size_override: Declared upper bound, in bytes, on the stack
                that this kernel's call subtree uses. See
                [`Kernel.stack_size_override`][iron.kernel.Kernel.stack_size_override].
                With ``inline=True``, the merged kernel has no separate object,
                so this bound is the one input aiecc's stack analysis reads for
                this kernel.
        """
        if inline and use_chess:
            raise ValueError(
                f"ExternalFunction '{name}': inline=True requires the Peano "
                "toolchain and cannot be combined with use_chess=True."
            )
        if inline and symbol_prefix:
            raise NotImplementedError(
                f"ExternalFunction '{name}': inline=True combined with symbol_prefix is "
                "not supported (an inline kernel is emitted as LLVM IR and cannot be "
                "symbol-renamed). Use inline without a symbol_prefix, or drop inline for "
                "this kernel."
            )

        self._original_name = name
        effective_name = f"{symbol_prefix}_{name}" if symbol_prefix else name
        object_file_name_explicit = object_file_name is not None
        if not object_file_name:
            object_file_name = (
                f"{effective_name}.ll" if inline else f"{effective_name}.o"
            )
        elif inline and Path(object_file_name).suffix.lower() not in (".ll", ".bc"):
            # An inline kernel is emitted as LLVM IR, and the suffix picks the
            # format (textual vs bitcode), so a wrong one has no valid reading.
            # Reject it instead of silently renaming the caller's artifact --
            # aiecc routes on the `link_with_mode` attribute, not the suffix, so
            # a rename would buy nothing.  Compared case-insensitively, matching
            # compile_cxx_core_function's own suffix check; the caller's exact
            # spelling is preserved either way.
            raise ValueError(
                f"ExternalFunction '{name}': inline=True emits LLVM IR, so "
                f"object_file_name must end in '.ll' (textual LLVM IR) or "
                f"'.bc' (bitcode); got '{object_file_name}'."
            )
        super().__init__(
            effective_name,
            object_file_name,
            arg_types,
            link_with_mode="merge" if inline else None,
            stack_size_override=stack_size_override,
        )

        if source_file is None and source_string is None:
            raise ValueError("source_file or source_string must be provided.")
        if source_file is not None:
            try:
                source_bytes = Path(source_file).read_bytes()
            except OSError:
                source_bytes = f"<unreadable:{source_file}>".encode()
        else:
            assert source_string is not None
            source_bytes = source_string.encode()
        self._object_file = KernelObject(
            object_file_name,
            "merge" if inline else None,
            _KernelSource(
                str(Path(source_file).resolve()) if source_file is not None else None,
                source_string if source_file is None else None,
                hashlib.sha256(source_bytes).hexdigest(),
                tuple(include_dirs or ()),
                tuple(compile_flags or ()),
                use_chess,
                symbol_prefix,
                name if inline else None,
            ),
        )
        self._cached_digest: str | None = None

        # A translation unit may export several symbols, but an output path
        # must have only one recipe. Auto-suffix default names on conflict;
        # explicit names are never silently changed.
        for existing in ExternalFunction._instances:
            if (
                existing.object_file_name == object_file_name
                and existing.object_file._source != self.object_file._source
            ):
                if object_file_name_explicit:
                    raise ValueError(
                        f"ExternalFunction '{effective_name}' would collide with "
                        f"an already-registered instance: same "
                        f"explicit object_file_name='{object_file_name}' but "
                        f"different compile_flags / source.  Distinguish them "
                        f"by passing a distinct `object_file_name=...`."
                    )
                suffix = self._content_digest()[:8]
                output_path = Path(object_file_name)
                object_file_name = str(
                    output_path.with_name(
                        f"{output_path.stem}_{suffix}{output_path.suffix}"
                    )
                )
                self._object_file = replace(self.object_file, name=object_file_name)
                self._cached_digest = None
                break
        for existing in ExternalFunction._instances:
            if existing.object_file_name == self.object_file_name:
                if existing.object_file._source != self.object_file._source:
                    raise ValueError(
                        f"ExternalFunction '{effective_name}' would collide on '{self.object_file_name}'"
                    )
                self._object_file = existing.object_file
                break
        self.also = SimpleNamespace()  # siblings from this kernel's object
        ExternalFunction._instances.add(self)

    # Read-only views of the compile recipe. Tooling that inspects or
    # recompiles a kernel outside the JIT path (aie.utils.compile.remarks) reads
    # these rather than the private fields; the JIT itself keeps using the
    # private fields directly.

    @property
    def _recipe(self) -> _KernelSource:
        recipe = self.object_file._source
        assert recipe is not None
        return recipe

    @property
    def _source_file(self):
        return self._recipe.source_file

    @property
    def _source_string(self):
        return self._recipe.source_string

    @property
    def _include_dirs(self):
        return self._recipe.include_dirs

    @property
    def _compile_flags(self):
        return self._recipe.compile_flags

    @property
    def _use_chess(self):
        return self._recipe.use_chess

    @property
    def _symbol_prefix(self):
        return self._recipe.symbol_prefix

    @property
    def _inline(self):
        return self._recipe.inline_symbol is not None

    @property
    def source_file(self) -> str | None:
        """Path to the C/C++ source on disk, or None for inline source."""
        return self._source_file

    @property
    def source_string(self) -> str | None:
        """Inline C/C++ source text, or None when compiled from a file."""
        return self._source_string

    @property
    def include_dirs(self) -> list[str]:
        """Copy of the extra ``-I`` directories passed to the compiler."""
        return list(self._include_dirs)

    @property
    def compile_flags(self) -> list[str]:
        """Copy of the extra flags passed verbatim to the compiler."""
        return list(self._compile_flags)

    @property
    def use_chess(self) -> bool:
        """True when this kernel's object is built with xchesscc, not Peano."""
        return self._use_chess

    def __call__(self, *args, **kwargs):
        """Call with argument count and type validation before emitting MLIR.

        ``**kwargs`` are forwarded to the base ``BaseKernel.__call__``
        and ultimately to the MLIR ``func.call`` builder.
        """
        if len(args) != len(self._arg_types):
            raise ValueError(
                f"ExternalFunction '{self._name}' expects "
                f"{len(self._arg_types)} argument(s), but {len(args)} "
                f"were provided."
            )
        for i, (arg, expected_ty) in enumerate(zip(args, self._arg_types)):
            self._validate_arg(i, arg, expected_ty)
        super().__call__(*args, **kwargs)

    def _validate_arg(self, index: int, arg, expected_ty) -> None:
        """Validate a single argument against its expected type."""
        if isinstance(expected_ty, type) and issubclass(expected_ty, np.generic):
            if not isinstance(arg, (int, float, np.integer, np.floating)):
                raise ValueError(
                    f"Argument {index}: expected scalar, got {type(arg).__name__}"
                )
            return
        if not (hasattr(expected_ty, "__args__") and hasattr(arg, "shape")):
            return
        # Only host-side (numpy) arguments are compared. An MLIR value's
        # element type is spelled differently (`i32` vs `np.int32`) and its
        # shape may legitimately differ from the declaration until
        # `_maybe_collapse_to_match` flattens it, so MLIR verification is what
        # checks those.
        arg_dtype = getattr(arg, "dtype", None)
        if not isinstance(arg_dtype, (np.dtype, type)):
            return
        expected_shape = expected_ty.__args__[0]
        expected_dtype = expected_ty.__args__[1].__args__[0]
        if arg.shape != expected_shape or arg_dtype != expected_dtype:
            raise ValueError(
                f"Argument {index}: expected {expected_shape}/{expected_dtype}, "
                f"got {arg.shape}/{arg_dtype}"
            )

    def _content_digest(self) -> str:
        """Return a 64-bit hex SHA-256 digest of this instance's content.

        Used by both ``__hash__`` and ``__eq__`` so the two are consistent.
        Memoised on the instance: source-file reads and stat() calls would
        otherwise run on every dict lookup and noticeably regress hot
        compile-cache paths.  Instance state is treated as immutable after
        construction; mutating ``_source_*`` / ``_include_dirs`` /
        ``_compile_flags`` / ``_arg_types`` afterwards is not supported.
        """
        if self._cached_digest is not None:
            return self._cached_digest

        from pathlib import Path as _Path

        include_dir_mtimes = []
        for d in self._include_dirs:
            try:
                mtime = str(_Path(d).stat().st_mtime)
            except (FileNotFoundError, OSError):
                mtime = "missing"
            include_dir_mtimes.append(f"{d}:{mtime}")

        parts = [
            self._name,
            self.object_file_name,
            str(self._arg_types),
            str(include_dir_mtimes),
            str(self._compile_flags),
            # Toolchain choice (peano vs chess) changes the resulting .o
            # contents even when name + arg_types + flags + source are
            # identical, so the digest must distinguish them.
            f"chess={self._use_chess}",
            f"inline={self._inline}",
        ]
        parts.extend([str(self._source_file), self._recipe.source_digest])
        self._cached_digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        return self._cached_digest

    def __hash__(self) -> int:
        """Content-based hash for use as a dict/set key and in cache signatures."""
        return int(self._content_digest(), 16)

    def __eq__(self, other: object) -> bool:
        """Content-based equality so hash collisions never produce false cache hits."""
        if not isinstance(other, ExternalFunction):
            return NotImplemented
        return self._content_digest() == other._content_digest()

    def __repr__(self) -> str:
        """Content-based repr so str(ef) is stable across GC cycles.

        Default ``object.__repr__`` uses the recyclable memory address; two
        distinct EFs can then alias onto the same _compute_hash cache slot.
        """
        return f"ExternalFunction({self._name!r}, digest={self._content_digest()})"
