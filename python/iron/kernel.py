# kernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

import numpy as np
import os
import subprocess
import hashlib
import shutil

from .. import ir  # type: ignore
from ..extras.dialects.ext.func import FuncOp  # type: ignore
from ..helpers.dialects.ext.func import call
from ..dialects.aie import external_func
from .resolvable import Resolvable


class Kernel(Resolvable):
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
        self._name = name
        self._bin_name = bin_name
        self._arg_types = arg_types
        self._op: FuncOp | None = None

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

    def __call__(self, *args, **kwargs):
        if not self._op:
            raise ValueError("Need to resolve Kernel before it can be called")
        call(self._op, args, **kwargs)



class CoreFunction(Resolvable):
    _object_files = set()

    def __init__(
        self,
        name: str,
        source_file: str,
        arg_types: list[type[np.ndarray] | np.dtype] = [],
        include_dirs: list[str] = [],
        compile_flags: list[str] = [],
        debug: bool = True,
    ) -> None:
        """A CoreFunction is a C++ source file that gets compiled to an object file and eventually resolves to a FuncOp.
        If it is called, a CallOp will be generated.

        Args:
            name (str): The name of the function
            source_file (str): Path to the C++ source file
            arg_types (list[type[np.ndarray] | np.dtype], optional): The type signature of the function. Defaults to [].
            include_dirs (list[str], optional): Additional include directories. Defaults to [].
            compile_flags (list[str], optional): Additional compilation flags. Defaults to [].
            debug (bool, optional): Enable debug logging. Defaults to True.
        """
        self._name = name
        self._source_file = source_file
        self._arg_types = arg_types
        self._include_dirs = include_dirs
        self._compile_flags = compile_flags
        self._op: FuncOp | None = None
        self._debug = debug

        if self._debug:
            print(f"Initializing CoreFunction: {name}")
            print(f"Source file: {source_file}")
            print(f"Include dirs: {include_dirs}")
            print(f"Compile flags: {compile_flags}")

        # Calculate hash of source file and compile flags
        with open(source_file, 'rb') as f:
            source_hash = hashlib.sha256(f.read()).hexdigest()
        flags_hash = hashlib.sha256(str(compile_flags).encode()).hexdigest()
        self._hash = hashlib.sha256((source_hash + flags_hash).encode()).hexdigest()

        if self._debug:
            print(f"Generated hash: {self._hash}")

        # Create cache directory structure
        self._cache_dir = os.path.expanduser(f"~/.iron/cache/{self._hash}")
        os.makedirs(self._cache_dir, exist_ok=True)

        if self._debug:
            print(f"Cache directory: {self._cache_dir}")

        # Set object file path in cache
        self._object_file = os.path.join(self._cache_dir, "kernel.o")
        CoreFunction._object_files.add(self._object_file)

    def __enter__(self):
        """Enter the context, resolving (compiling and registering) the function."""
        if self._debug:
            print(f"\n[Context Enter] Entering context for CoreFunction: {self._name}")
        self.resolve()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context. Cleanup or finalization can be added here if needed."""
        if self._debug:
            print(f"[Context Exit] Exiting context for CoreFunction: {self._name}")
        # Optional cleanup logic could go here
        pass

    @property
    def bin_name(self) -> str:
        return os.path.basename(self._object_file)

    def compile(self) -> None:
        """Compile the C++ source file to an object file in cache directory."""
        if self._debug:
            print(f"\nCompiling {self._source_file}...")
            print(f"Output will be: {self._object_file}")

        # Skip compilation if object file already exists
        if os.path.exists(self._object_file):
            if self._debug:
                print("Object file already exists in cache, skipping compilation")
            return

        # Base compilation command
        cmd = [
            f"{os.environ.get('PEANO_INSTALL_DIR', '')}/bin/clang++",
            "-O2",
            "-std=c++20",
            "--target=aie2-none-unknown-elf",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            "-DNDEBUG",
        ]

        # Add AIEOPT include directory
        try:
            aieopt_path = subprocess.check_output(['which', 'aie-opt'], text=True).strip()
            aieopt_dir = os.path.dirname(os.path.dirname(os.path.realpath(aieopt_path)))
            cmd.extend(["-I", f"{aieopt_dir}/include"])
        except subprocess.CalledProcessError:
            if self._debug:
                print("Warning: Could not find aie-opt executable")

        # Add device-specific flags
        if os.environ.get('NPU2') == '1':
            cmd.extend(os.environ.get('PEANOWRAP2P_FLAGS', '').split())
            if self._debug:
                print("Using NPU2 flags")
        else:
            cmd.extend(os.environ.get('PEANOWRAP2_FLAGS', '').split())
            if self._debug:
                print("Using NPU1 flags")

        # Add include directories
        for include_dir in self._include_dirs:
            cmd.extend(["-I", include_dir])

        # Add compilation flags
        cmd.extend(self._compile_flags)

        # Add source and output files
        cmd.extend(["-c", self._source_file, "-o", self._object_file])

        if self._debug:
            print("\nCompilation command:")
            print(" ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if self._debug:
                print("\nCompilation successful!")
                if result.stdout:
                    print("stdout:", result.stdout.decode())
                if result.stderr:
                    print("stderr:", result.stderr.decode())
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"Compilation failed:\n{e}\n"
                f"stdout:\n{e.stdout.decode() if e.stdout else 'No stdout'}\n"
                f"stderr:\n{e.stderr.decode() if e.stderr else 'No stderr'}"
            )
            if self._debug:
                print("\nCompilation failed!")
                print(error_msg)
            raise RuntimeError(error_msg)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if not self._op:
            if self._debug:
                print(f"\nResolving kernel {self._name}...")
            # Compile the C++ source first
            self.compile()
            # Then create the external function
            self._op = external_func(self._name, inputs=self._arg_types)
            if self._debug:
                print("Args: ", self._arg_types)
                print(f"Created external function: {self._name}")

    def __call__(self, *args, **kwargs):
        if not self._op:
            raise ValueError("Need to resolve CoreFunction before it can be called")
        if self._debug:
            print(f"\nCalling kernel {self._name}...")
        call(self._op, args, **kwargs)
