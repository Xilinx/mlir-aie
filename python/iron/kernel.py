# kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
"""Kernel and ExternalFunction: wrappers for pre-compiled and C++ AIE compute kernels."""

import hashlib
import logging
import numpy as np

logger = logging.getLogger(__name__)

from .. import ir  # type: ignore
from ..dialects import memref  # type: ignore
from ..extras.dialects.func import FuncOp  # type: ignore
from ..helpers.dialects.func import call
from ..dialects.aie import external_func
from .resolvable import Resolvable
from .buffer import Buffer


def _is_contiguous_row_major(mr):
    """True iff ``mr`` is a fully-static row-major contiguous memref at offset 0.

    A default memref like ``memref<64x32xi16>`` qualifies; a strided view
    such as ``memref<64x32xi16, strided<[64, 1]>>`` produced by a slice or
    a custom layout does not.  We require this before emitting a
    ``memref.collapse_shape`` because collapse-on-non-contiguous-dims is
    undefined behaviour at the MLIR level and would silently produce
    wrong DMAs / loads.
    """
    if any(d < 0 for d in mr.shape):
        return False
    try:
        strides, offset = mr.get_strides_and_offset()
    except Exception:
        # No expressible stride layout → conservatively refuse.
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
    """Bridge an N-D contiguous memref arg to a 1-D kernel arg signature.

    Iron designs naturally hold multi-dimensional ObjectFifo elements (e.g.
    a matmul L1 buffer typed ``memref<64x64xi16>``) but the
    ``aie.iron.kernels.X`` helpers (``mm``, ``mm_zero``, ``passthrough``,
    ...) declare flat 1-D arg signatures (``memref<4096xi16>``) because the
    underlying C++ kernels read pointers and re-stride internally.  Without
    an adapter, the resulting ``func.call`` fails MLIR verification with
    a memref-shape mismatch even though the bytes line up perfectly.

    This helper inserts a ``memref.collapse_shape`` to flatten such an
    argument when ALL of the following hold:

      * ``arg`` is a memref-typed Value with rank ≥ 1
      * ``expected_ty`` is a rank-1 memref
      * element types match
      * the source memref is **fully static, row-major contiguous, offset 0**
        (verified by :func:`_is_contiguous_row_major`)
      * total static element counts match

    Any other case is returned unchanged so MLIR's existing verification
    fires on real bugs rather than being silenced here.  In particular,
    strided views, partial-collapse reshapes, transposes, and rank-
    preserving permutations are all left alone.

    The collapsed memref aliases the same underlying storage — no copy or
    allocation is emitted.
    """
    # Non-Value args (Python scalars, etc.) pass through.
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
    # Only collapse N-D → 1-D for now; other reshapes (rank-preserving
    # permutations, partial collapses) are real semantic differences and
    # should fail loudly.
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
        """
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
        """Return the shape tuple of the array argument at ``arg_index``.

        Works for both ``np.ndarray[(...,), np.dtype[T]]`` parameterized
        types (the canonical iron kernel signature) and MLIR MemRefType
        operands.

        Args:
            arg_index: Index into ``arg_types``.  Defaults to 0.

        Raises:
            ValueError: When ``arg_index`` is out of range or the
                argument at that index is not an array type.
        """
        arg = self._resolve_arg(arg_index)
        if hasattr(arg, "__args__") and len(arg.__args__) > 0:
            shape_arg = arg.__args__[0]
            if isinstance(shape_arg, tuple):
                return shape_arg
        if hasattr(arg, "shape"):
            return tuple(arg.shape)
        raise ValueError(
            f"Argument {arg_index} does not have a shape or is not an array type."
        )

    def arg_dtype(self, arg_index: int = 0):
        """Return the numpy dtype of the array argument at ``arg_index``.

        Args:
            arg_index: Index into ``arg_types``.  Defaults to 0.

        Raises:
            ValueError: When ``arg_index`` is out of range or the
                argument at that index is not an array type.
        """
        arg = self._resolve_arg(arg_index)
        if hasattr(arg, "__args__") and len(arg.__args__) >= 2:
            dt = arg.__args__[1]
            return np.dtype(dt.__args__[0]) if hasattr(dt, "__args__") else np.dtype(dt)
        if hasattr(arg, "dtype"):
            return np.dtype(arg.dtype)
        raise ValueError(
            f"Argument {arg_index} does not have a dtype or is not an array type."
        )

    def tile_size(self, arg_index: int = 0) -> int:
        """Return the first dimension of the array argument at ``arg_index``.

        Convenience wrapper over :meth:`arg_shape` for the common case of
        a 1-D buffer argument.  ``tile_size(i)`` is equivalent to
        ``arg_shape(i)[0]``.

        Args:
            arg_index: Index into ``arg_types``.  Defaults to 0.
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

    def __call__(self, *args):
        """Emit a func.call to this kernel, validating argument count.

        Each argument is passed through :func:`_maybe_collapse_to_match`
        before the call.  This silently inserts a ``memref.collapse_shape``
        when an N-D contiguous memref arg is being fed into a 1-D kernel
        signature with the same element count and dtype — the typical case
        when an iron design holds 2-D ObjectFifo elements but the
        ``aie.iron.kernels.X`` helper declares a flat 1-D arg.  See that
        helper's docstring for the full set of conditions.  Real shape /
        dtype mismatches still fail at MLIR verification time.
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
        call(self._op, adapted)


class Kernel(BaseKernel):
    """An AIE core function backed by a pre-compiled object file.

    Use :class:`ExternalFunction` instead when you want to compile from
    C/C++ source at JIT time.

    ``resolve()`` emits a ``func.func private`` declaration with a
    ``link_with`` attribute naming ``object_file_name``.  The
    ``aie-assign-core-link-files`` pass propagates this into the CoreOp's
    ``link_files`` attribute so the linker knows which file to include.
    """

    def __init__(
        self,
        name: str,
        object_file_name: str,
        arg_types: list[type[np.ndarray] | np.dtype] | None = None,
    ) -> None:
        """
        Args:
            name: Symbol name of the function as it appears in the object file.
            object_file_name: Filename of the pre-compiled object file
                (e.g. ``"add_one.o"``).  Must be on the linker search path
                at compile time.
            arg_types: Type signature of the function arguments.  Defaults to None (empty list).
        """
        super().__init__(name, arg_types)
        self._object_file_name = object_file_name

    @property
    def object_file_name(self) -> str:
        """Filename of the compiled object file."""
        return self._object_file_name

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._op:
            self._op = external_func(
                self._name, inputs=self._arg_types, link_with=self._object_file_name
            )


class ExternalFunction(Kernel):
    """An AIE core function compiled from C/C++ source at JIT time.

    Each instance is registered in ``_instances`` at construction time so that
    the ``@jit`` decorator can discover and compile all source files before
    invoking the MLIR compilation pipeline.  ``_instances`` is cleared at the
    start of each ``@jit`` call to prevent stale registrations from a previous
    (possibly failed) run.

    Use the base :class:`Kernel` class instead when you have a pre-built
    object file.
    """

    _instances: set = set()  # Registry of all live ExternalFunction instances.

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
    ) -> None:
        """
        Args:
            name: Symbol name of the function as it will appear in the object
                file.
            object_file_name: Output object file name.  Defaults to
                ``<effective_name>.o``.
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
        """
        self._original_name = name
        self._symbol_prefix = symbol_prefix
        effective_name = f"{symbol_prefix}_{name}" if symbol_prefix else name
        object_file_name_explicit = object_file_name is not None
        if not object_file_name:
            object_file_name = f"{effective_name}.o"
        super().__init__(effective_name, object_file_name, arg_types)

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

        # The JIT pipeline writes each ExternalFunction's compiled output to
        # ``object_file_name`` inside the cache directory; two such writes to
        # the same path silently overwrite each other and leave the wrong .o
        # linked in.  Two same-name EFs with different content (e.g. two
        # kernels.mm() helper calls with different parameterizations) would
        # otherwise produce identical default .o filenames and collide.
        # Mirror what kernels.X / _make_extern already does: when the path
        # was DEFAULTED, auto-suffix it with a short content digest so each
        # parameterization lives at a distinct .o.  When the user passed an
        # explicit ``object_file_name=``, silent rename would surprise them
        # — raise instead so they can disambiguate by name themselves.
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
                object_file_name = f"{effective_name}_{suffix}.o"
                self._object_file_name = object_file_name
                break
        ExternalFunction._instances.add(self)

    def __call__(self, *args):
        """Call with argument count and type validation before emitting MLIR."""
        if len(args) != len(self._arg_types):
            raise ValueError(
                f"ExternalFunction '{self._name}' expects "
                f"{len(self._arg_types)} argument(s), but {len(args)} "
                f"were provided."
            )
        for i, (arg, expected_ty) in enumerate(zip(args, self._arg_types)):
            self._validate_arg(i, arg, expected_ty)
        super().__call__(*args)

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

        The default ``object.__repr__`` includes the memory address, which
        Python's GC recycles.  Two ExternalFunction instances with different
        content can end up at the same address in sequence, producing the same
        ``str(ef)`` and therefore the same filesystem cache hash in
        ``_compute_hash``, causing the wrong compiled binary to be loaded.
        Using the content digest here makes ``str(ef)`` unique per content.
        """
        return f"ExternalFunction({self._name!r}, digest={self._content_digest()})"
