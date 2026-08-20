# kernel.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Kernel and ExternalFunction: wrappers for pre-compiled and C++ AIE compute kernels."""

import hashlib
import logging
from pathlib import Path

import numpy as np

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ..dialects import memref  # pyright: ignore[reportAttributeAccessIssue]
from ..dialects.aie import external_func
from ..extras.dialects.func import FuncOp  # pyright: ignore[reportMissingImports]
from ..helpers.dialects.func import call
from .buffer import Buffer
from .resolvable import Resolvable

logger = logging.getLogger(__name__)


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
        self._arg_types = arg_types if arg_types is not None else []
        self._op: FuncOp | None = None

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
            return np.dtype(dt_args[0]) if dt_args is not None else np.dtype(dt)
        dtype = getattr(arg, "dtype", None)
        if dtype is not None:
            return np.dtype(dtype)
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
        """Return a copy of the argument type list."""
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
        object_file_name: str,
        arg_types: list[type[np.ndarray] | np.dtype] | None = None,
        *,
        link_with_mode: str | None = None,
        stack_size_override: int | None = None,
    ) -> None:
        """Construct a Kernel backed by a pre-compiled object file.

        Args:
            name: Symbol name of the function as it appears in the object file.
            object_file_name: Filename of the pre-compiled object file
                (e.g. ``"add_one.o"``).  Must be on the linker search path
                at compile time.
            arg_types: Type signature of the function arguments.  Defaults to None (empty list).
            link_with_mode: Optional link policy emitted alongside
                ``link_with``.  ``"merge"`` routes the artifact through aiecc's
                ``llvm-link`` merge path; None (the default) object-links it.
            stack_size_override: Optional declared upper bound, in bytes, on
                the stack this kernel's call subtree needs. aiecc's automatic
                stack analysis treats this as the answer for the whole
                subtree and does not descend into it -- the escape hatch for
                recursion or indirect (function-pointer) calls inside this
                kernel, which cannot be sized automatically, and the only way
                to give the analysis any information at all about a
                ``link_with_mode="merge"`` kernel (merged into the core's own
                module before codegen, so the analysis cannot see it as a
                separate object). An explicit value here always wins over
                whatever the analysis would otherwise compute, even if
                smaller -- it is a declaration, not a clamp. ``0`` is legal.
        """
        super().__init__(name, arg_types)
        self._object_file_name = object_file_name
        self._link_with_mode = link_with_mode
        self._stack_size_override = stack_size_override

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
        """Declared upper bound on this kernel's call-subtree stack use, or
        None to let aiecc's automatic analysis compute it."""
        return self._stack_size_override

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
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

    # Optional sibling bindings attached by the linalg kernel factories
    # (kernels.mm / mv / cascade_mm). Declared here so the dynamic
    # assignment of these contract attributes type-checks.
    mac_dims: tuple
    zero: "Kernel"
    get_only: "Kernel"
    put_only: "Kernel"
    put_get: "Kernel"

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
            stack_size_override: Optional declared upper bound, in bytes, on
                the stack this kernel's call subtree needs -- see
                [`Kernel.stack_size_override`][iron.Kernel.stack_size_override].
                With ``inline=True`` this is the only way to give aiecc's
                automatic stack analysis any information about this kernel at
                all, since a merged kernel has no separate object for the
                analysis to inspect.
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

        # Must precede the collision scan below: it registers this instance in
        # `_instances`, whose hash/eq run `_content_digest()` -- which reads
        # `_inline`.  `_original_name` / `_symbol_prefix` are likewise read by
        # compile_external_kernel (source naming and the objcopy symbol rename).
        self._original_name = name
        self._symbol_prefix = symbol_prefix
        self._inline = inline
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

        if source_file is not None:
            self._source_file = source_file
            self._source_string = None
        elif source_string is not None:
            self._source_file = None
            self._source_string = source_string
        else:
            raise ValueError("source_file or source_string must be provided.")

        self._include_dirs = include_dirs if include_dirs is not None else []
        self._compile_flags = compile_flags if compile_flags is not None else []
        self._use_chess = use_chess
        self._compiled = False
        self._cached_digest: str | None = None

        # Two same-name EFs with default object_file_name would collide on the
        # same artifact path. Auto-suffix defaulted names with a content digest;
        # raise on explicit names so silent renames don't surprise the caller.
        for existing in ExternalFunction._instances:
            if (
                existing._name == effective_name
                and existing._object_file_name == object_file_name
                and existing._content_digest() != self._content_digest()
            ):
                if object_file_name_explicit:
                    raise ValueError(
                        f"ExternalFunction '{effective_name}' would collide with "
                        f"an already-registered instance: same name and "
                        f"explicit object_file_name='{object_file_name}' but "
                        f"different compile_flags / source.  Distinguish them "
                        f"by passing a distinct `object_file_name=...` or "
                        f"`name=...`."
                    )
                suffix = self._content_digest()[:8]
                output_path = Path(object_file_name)
                object_file_name = str(
                    output_path.with_name(
                        f"{output_path.stem}_{suffix}{output_path.suffix}"
                    )
                )
                self._object_file_name = object_file_name
                break
        ExternalFunction._instances.add(self)

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
        if hasattr(expected_ty, "__args__") and hasattr(arg, "shape"):
            expected_shape = expected_ty.__args__[0]
            expected_dtype = expected_ty.__args__[1].__args__[0]
            if arg.shape != expected_shape or arg.dtype != expected_dtype:
                raise ValueError(
                    f"Argument {index}: expected {expected_shape}/{expected_dtype}, "
                    f"got {arg.shape}/{arg.dtype}"
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
        for d in sorted(self._include_dirs):
            try:
                mtime = str(_Path(d).stat().st_mtime)
            except (FileNotFoundError, OSError):
                mtime = "missing"
            include_dir_mtimes.append(f"{d}:{mtime}")

        parts = [
            self._name,
            str(self._arg_types),
            str(include_dir_mtimes),
            str(sorted(self._compile_flags)),
            # Toolchain choice (peano vs chess) changes the resulting .o
            # contents even when name + arg_types + flags + source are
            # identical, so the digest must distinguish them.
            f"chess={self._use_chess}",
            f"inline={self._inline}",
        ]
        if self._source_string:
            parts.append(self._source_string)
        elif self._source_file:
            try:
                with open(self._source_file) as f:
                    parts.append(f.read())
            except OSError:
                parts.append(f"<unreadable:{self._source_file}>")
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
