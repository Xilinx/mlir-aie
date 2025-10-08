# kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import numpy as np

from .. import ir  # type: ignore
from ..extras.dialects.ext.func import FuncOp  # type: ignore
from ..helpers.dialects.ext.func import call
from ..dialects.aie import external_func
from .resolvable import Resolvable


class BaseKernel(Resolvable):
    """Base class for kernel-like objects that resolve to FuncOp."""

    def __init__(self, name: str, arg_types: list[type[np.ndarray] | np.dtype] = []):
        """Initialize base kernel.

        Args:
            name (str): The name of the function
            arg_types (list[type[np.ndarray] | np.dtype], optional): The type signature of the function. Defaults to [].
        """
        self._name = name
        self._arg_types = arg_types
        self._op: FuncOp | None = None

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Resolve the kernel to a FuncOp. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement resolve()")

    def __call__(self, *args, **kwargs):
        """Call the kernel with the given arguments."""
        if not self._op:
            raise ValueError("Need to resolve kernel before it can be called")
        call(self._op, args, **kwargs)


class Kernel(BaseKernel):
    def __init__(
        self,
        name: str,
        bin_name: str,
        arg_types: list[type[np.ndarray] | np.dtype] = [],
    ) -> None:
        """A Kernel is an externally defined function that eventually resolves to a FuncOp. If it is called,
        a CallOp will be generated.

        Args:
            name (str): The name of the function
            bin_name (str): The name of the binary (used for linking to a compute core)
            arg_types (list[type[np.ndarray]  |  np.dtype], optional): The type signature of the function. Defaults to [].
        """
        super().__init__(name, arg_types)
        self._bin_name = bin_name

    @property
    def bin_name(self) -> str:
        return self._bin_name

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._op:
            self._op = external_func(self._name, inputs=self._arg_types)


class ExternalFunction(BaseKernel):
    _instances = set()

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
        """An ExternalFunction is a C/C++ source file that gets compiled to an object file and eventually resolves to a FuncOp.
        If it is called, a CallOp will be generated.

        Args:
            name (str): The name of the function
            object_file_name (str, optional): The name of the object file. If None, it will be name.o.
            source_file (str): Path to the C/C++ source file
            source_string (str): C/C++ source code as a string
            arg_types (list[type[np.ndarray] | np.dtype], optional): The type signature of the function. Defaults to [].
            include_dirs (list[str], optional): Additional include directories. Defaults to [].
            compile_flags (list[str], optional): Additional compilation flags. Defaults to [].
        """
        super().__init__(name, arg_types)
        self._setup_source(source_file, source_string)
        self._include_dirs = include_dirs
        self._compile_flags = compile_flags
        if object_file_name:
            self._object_file_name = object_file_name
        else:
            self._object_file_name = f"{self._name}.o"
        self._compiled = False

        # Track this instance for JIT compilation
        ExternalFunction._instances.add(self)

    def _setup_source(self, source_file: str | None, source_string: str | None) -> None:
        """Set up the source file for compilation."""
        if source_file is not None:
            self._source_file = source_file
            self._source_string = None
        else:
            if source_string is None:
                raise ValueError("source_file or source_string must be provided")
            self._source_file = None
            self._source_string = source_string

    def __enter__(self):
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context."""
        pass

    @property
    def bin_name(self) -> str:
        return self._object_file_name

    def tile_size(self, arg_index: int = 0) -> int:
        """Get the tile size from the specified array argument type.

        Args:
            arg_index (int): Index of the argument to get tile size from. Defaults to 0.

        Returns:
            int: The tile size (first dimension) of the specified argument.
        """
        if not self._arg_types:
            raise ValueError("No argument types defined")
        if arg_index >= len(self._arg_types):
            raise ValueError(
                f"Argument index {arg_index} out of range (max: {len(self._arg_types) - 1})"
            )

        arg = self._arg_types[arg_index]

        # Handle numpy array types like np.ndarray[(16,), np.dtype[np.int32]]
        if hasattr(arg, "__args__") and len(arg.__args__) > 0:
            # For types like np.ndarray[(16,), np.dtype[np.int32]], the shape is in __args__[0]
            shape_arg = arg.__args__[0]
            if isinstance(shape_arg, tuple) and len(shape_arg) > 0:
                return shape_arg[0]

        # Handle MLIR types like MemRefType(memref<16xi32>)
        if (
            hasattr(arg, "shape")
            and hasattr(arg.shape, "__len__")
            and len(arg.shape) > 0
        ):
            return arg.shape[0]

        raise ValueError(
            f"Argument {arg_index} does not have a shape or is not an array type"
        )

    def arg_types(self) -> list:
        """Get the argument types of the ExternalFunction."""
        return self._arg_types.copy()

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._op:
            # Create the external function
            self._op = external_func(self._name, inputs=self._arg_types)

    def __hash__(self):
        """
        Compute a hash for the ExternalFunction based on its properties.
        This allows ExternalFunction instances to be used in cache keys.
        """
        import hashlib

        # Create a string representation of the function's key properties
        hash_parts = [
            self._name,
            str(self._arg_types),
            str(sorted(self._include_dirs)),
            str(sorted(self._compile_flags)),
        ]

        # Include source content for uniqueness
        # TODO: This solution needs to be extended to handle headers. See https://github.com/Xilinx/mlir-aie/issues/2543
        if self._source_string:
            hash_parts.append(self._source_string)
        elif self._source_file:
            with open(self._source_file, "r") as f:
                file_content = f.read()
            hash_parts.append(file_content)

        # Create hash from combined string
        combined = "|".join(hash_parts)
        return int(hashlib.sha256(combined.encode("utf-8")).hexdigest()[:8], 16)

    def __call__(self, *args, **kwargs):
        if not self._op:
            raise ValueError("Need to resolve ExternalFunction before it can be called")
        call(self._op, args, **kwargs)
