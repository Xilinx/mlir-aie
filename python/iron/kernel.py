# kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.

import hashlib
import logging
import numpy as np

logger = logging.getLogger(__name__)

from .. import ir  # type: ignore
from ..extras.dialects.func import FuncOp  # type: ignore
from ..helpers.dialects.func import call
from ..dialects.aie import external_func
from .resolvable import Resolvable
from .buffer import Buffer


class BaseKernel(Resolvable):
    """Base class for AIE core functions that resolve to a func.func declaration.

    Subclasses:
        Kernel: wraps a pre-compiled object file.
        ExternalFunction: compiles C/C++ source at JIT time.
    """

    def __init__(self, name: str, arg_types: list[type[np.ndarray] | np.dtype] = []):
        """
        Args:
            name: Symbol name of the function.
            arg_types: Type signature of the function arguments.  Defaults to [].
        """
        if not name:
            raise ValueError("Kernel name cannot be empty.")
        self._name = name
        self._arg_types = arg_types
        self._op: FuncOp | None = None

    def tile_size(self, arg_index: int = 0) -> int:
        """Return the first dimension of the array argument at ``arg_index``.

        Args:
            arg_index: Index into ``arg_types``.  Defaults to 0.
        """
        if not self._arg_types:
            raise ValueError("No argument types defined.")
        if arg_index >= len(self._arg_types):
            raise ValueError(
                f"Argument index {arg_index} out of range "
                f"(max: {len(self._arg_types) - 1})"
            )
        arg = self._arg_types[arg_index]

        # numpy array type, e.g. np.ndarray[(16,), np.dtype[np.int32]]
        if hasattr(arg, "__args__") and len(arg.__args__) > 0:
            shape_arg = arg.__args__[0]
            if isinstance(shape_arg, tuple) and len(shape_arg) > 0:
                return shape_arg[0]

        # MLIR MemRefType
        if hasattr(arg, "shape") and len(arg.shape) > 0:
            return arg.shape[0]

        raise ValueError(
            f"Argument {arg_index} does not have a shape or is not an array type."
        )

    def arg_types(self) -> list:
        """Return a copy of the argument type list."""
        return self._arg_types.copy()

    def __call__(self, *args, **kwargs):
        """Emit a func.call to this kernel, validating argument count."""
        if not self._op:
            raise ValueError("Kernel must be resolved before it can be called.")
        if len(args) != len(self._arg_types):
            raise ValueError(
                f"Kernel '{self._name}' expects {len(self._arg_types)} "
                f"argument(s), but {len(args)} were provided."
            )
        arg_ops = [a.op if isinstance(a, Buffer) else a for a in args]
        call(self._op, arg_ops, **kwargs)


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
        arg_types: list[type[np.ndarray] | np.dtype] = [],
    ) -> None:
        """
        Args:
            name: Symbol name of the function as it appears in the object file.
            object_file_name: Filename of the pre-compiled object file
                (e.g. ``"add_one.o"``).  Must be on the linker search path
                at compile time.
            arg_types: Type signature of the function arguments.  Defaults to [].
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
        arg_types: list[type[np.ndarray] | np.dtype] = [],
        include_dirs: list[str] = [],
        compile_flags: list[str] = [],
    ) -> None:
        """
        Args:
            name: Symbol name of the function as it will appear in the object
                file.
            object_file_name: Output object file name.  Defaults to
                ``<name>.o``.
            source_file: Path to a C/C++ source file on disk.  Mutually
                exclusive with ``source_string``.
            source_string: Inline C/C++ source code.  Mutually exclusive with
                ``source_file``.
            arg_types: Type signature of the function arguments.  Defaults to
                [].
            include_dirs: Additional ``-I`` directories passed to the Peano
                compiler.  Defaults to [].
            compile_flags: Additional flags passed verbatim to the Peano
                compiler.  Defaults to [].
        """
        if not object_file_name:
            object_file_name = f"{name}.o"
        super().__init__(name, object_file_name, arg_types)

        if source_file is not None:
            self._source_file = source_file
            self._source_string = None
        elif source_string is not None:
            self._source_file = None
            self._source_string = source_string
        else:
            raise ValueError("source_file or source_string must be provided.")

        self._include_dirs = include_dirs
        self._compile_flags = compile_flags
        self._compiled = False

        # Register this instance so the @jit decorator can compile it.
        ExternalFunction._instances.add(self)

    def __call__(self, *args, **kwargs):
        """Call with argument count and type validation before emitting MLIR."""
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

    def __hash__(self):
        """Hash based on source content and compiler options for cache keying."""
        # TODO: extend to cover included headers (issue #2543)
        hash_parts = [
            self._name,
            str(self._arg_types),
            str(sorted(self._include_dirs)),
            str(sorted(self._compile_flags)),
        ]
        if self._source_string:
            hash_parts.append(self._source_string)
        elif self._source_file:
            with open(self._source_file, "r") as f:
                hash_parts.append(f.read())
        combined = "|".join(hash_parts)
        return int(hashlib.sha256(combined.encode("utf-8")).hexdigest()[:8], 16)
