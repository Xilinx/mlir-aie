# compileconfig.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import os
import functools
import hashlib
import shutil
import fcntl
import contextlib
import time
import inspect
import json
from pathlib import Path
from typing import Callable

from aie.extras.context import mlir_mod_ctx
from .compile import compile_mlir_module
from .compile.link import merge_object_files
from .metaprogram import compile_ctx
from .config import get_current_device
from aie.dialects.aie import AIEDevice
from .device import NPU1, NPU2, NPU1Col1, NPU2Col1


# The `iron.compileconfig` decorator below caches compiled kenrels inside the `IRON_CACHE_HOME` directory.
# Kernels are cached based on their hash value of the MLIR module string. If during compilation,
# we hit in the cache, the `iron.jit` will load the xclbin and instruction binary files from the cache.
IRON_CACHE_HOME = os.environ.get("IRON_CACHE_HOME", os.path.expanduser("~/.iron/cache"))


def _cleanup_failed_compilation(cache_dir):
    """Clean up cache directory after failed compilation, preserving the lock file."""
    if not os.path.exists(cache_dir):
        return

    for item in os.listdir(cache_dir):
        if item == ".lock":
            continue
        item_path = os.path.join(cache_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


@contextlib.contextmanager
def file_lock(lock_file_path, timeout_seconds=60):
    """
    Context manager for file locking using flock to prevent race conditions.

    Args:
        lock_file_path (str): Path to the lock file
        timeout_seconds (int): Maximum time to wait for lock acquisition in seconds
    """
    lock_file = None
    try:
        # Create lock file if it doesn't exist
        os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)
        try:
            f = os.open(lock_file_path, os.O_CREAT | os.O_EXCL)
            os.close(f)
        except FileExistsError:
            pass  # File already exists
        lock_file = open(lock_file_path, "a")

        # Try to acquire exclusive lock with timeout
        start_time = time.time()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                # Lock is held by another process
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(
                        f"Could not acquire lock on {lock_file_path} within {timeout_seconds} seconds"
                    )
                time.sleep(0.1)

        yield lock_file

    finally:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass  # Ignore errors when releasing lock
            lock_file.close()


class PreCompiled:
    """A class to hold pre-compiled artifacts."""

    def __init__(self, xclbin_path: Path, insts_path: Path):
        """Initializes the PreCompiled object.

        Args:
            xclbin_path (Path): The path to the xclbin file.
            insts_path (Path): The path to the insts file.
        """
        self.xclbin_path = xclbin_path
        self.insts_path = insts_path

    def get_artifacts(self) -> tuple[Path, Path]:
        """Returns the artifact paths.

        Returns:
            tuple[Path, Path]: A tuple containing the xclbin path and the insts path.
        """
        return self.xclbin_path, self.insts_path


class CompilableDesign:
    """A class that encapsulates a function and its compilation configuration."""

    def __init__(
        self,
        mlir_generator: Callable | Path,
        use_cache: bool = True,
        compile_flags: list[str] | None = None,
        source_files: list[Path] | None = None,
        include_paths: list[Path] | None = None,
        aiecc_flags: list[str] | None = None,
        metaargs: dict[str, object] | None = None,
        object_files: list[Path] | None = None,
    ):
        self.mlir_generator = mlir_generator
        self.use_cache = use_cache
        self.compile_flags = compile_flags
        self.source_files = source_files if source_files else []
        self.include_paths = include_paths if include_paths else []
        self.aiecc_flags = aiecc_flags
        self.metaargs = metaargs if metaargs is not None else {}
        self.object_files = object_files if object_files else []
        self.xclbin_path: Path | None = None
        self.insts_path: Path | None = None
        if callable(mlir_generator):
            functools.update_wrapper(self, mlir_generator)

        if self.source_files:
            for f in self.source_files:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Source file not found: {f}")
        if self.object_files:
            for f in self.object_files:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Object file not found: {f}")

    def __call__(self, *args, **kwargs) -> any:
        """Calls the encapsulated function.

        Returns:
            any: The result of the function call.
        """
        if callable(self.mlir_generator):
            return self.mlir_generator(*args, **kwargs)
        else:
            with open(self.mlir_generator, "r") as f:
                return f.read()

    def get_artifacts(self) -> tuple[Path, Path]:
        """Returns the artifact paths.

        Returns:
            tuple[Path, Path]: A tuple containing the xclbin path and the insts path.
        """
        return self.xclbin_path, self.insts_path

    def to_json(self) -> str:
        """Serializes the CompilableDesign object to a JSON string.

        Returns:
            str: The JSON string representation of the object.
        """
        if callable(self.mlir_generator):
            mlir_generator = self.mlir_generator.__name__
        else:
            mlir_generator = str(self.mlir_generator)
        return json.dumps(
            {
                "mlir_generator": mlir_generator,
                "use_cache": self.use_cache,
                "compile_flags": self.compile_flags,
                "source_files": [str(f) for f in self.source_files],
                "include_paths": [str(f) for f in self.include_paths],
                "aiecc_flags": self.aiecc_flags,
                "metaargs": self.metaargs,
                "object_files": [str(f) for f in self.object_files],
            },
            indent=4,
        )

    @classmethod
    def get_json_schema(cls) -> str:
        """Gets the JSON schema for the CompilableDesign object.

        Returns:
            str: The JSON schema.
        """
        schema = {
            "type": "object",
            "properties": {
                "mlir_generator": {"type": "string"},
                "use_cache": {"type": "boolean"},
                "compile_flags": {"type": "array", "items": {"type": "string"}},
                "source_files": {"type": "array", "items": {"type": "string"}},
                "include_paths": {"type": "array", "items": {"type": "string"}},
                "aiecc_flags": {"type": "array", "items": {"type": "string"}},
                "metaargs": {"type": "object"},
                "object_files": {"type": "array", "items": {"type": "string"}},
            },
        }
        return json.dumps(schema, indent=4)

    @classmethod
    def from_json(
        cls, json_str: str, func: Callable | None = None
    ) -> "CompilableDesign":
        """Deserializes a CompilableDesign object from a JSON string.

        Args:
            json_str (str): The JSON string representation of the object.
            func (callable): The function to be encapsulated.

        Returns:
            CompilableDesign: The deserialized CompilableDesign object.
        """
        data = json.loads(json_str)
        mlir_generator = data["mlir_generator"]
        if func:
            mlir_generator = func
        return cls(
            mlir_generator,
            use_cache=data["use_cache"],
            compile_flags=data["compile_flags"],
            source_files=data["source_files"],
            include_paths=data["include_paths"],
            aiecc_flags=data["aiecc_flags"],
            metaargs=data["metaargs"],
            object_files=data["object_files"],
        )

    def __hash__(self) -> int:
        """Computes the hash of the CompilableDesign object.

        Returns:
            int: The hash of the object.
        """
        if callable(self.mlir_generator):
            func_name = self.mlir_generator.__name__
            func_source_path = inspect.getsourcefile(self.mlir_generator)
            hash_parts = [func_name, func_source_path]
        else:
            hash_parts = [str(self.mlir_generator)]

        if self.compile_flags:
            hash_parts.append(str(sorted(self.compile_flags)))

        if self.aiecc_flags:
            hash_parts.append(str(sorted(self.aiecc_flags)))

        if self.metaargs:
            hash_parts.append(str(sorted(self.metaargs.items())))

        if self.object_files:
            hash_parts.append(str(sorted(self.object_files)))

        combined_str = "|".join(hash_parts)
        return int(hashlib.sha256(combined_str.encode("utf-8")).hexdigest()[:16], 16)

    def compile(self, *args, **kwargs) -> tuple[Path, Path]:
        """Compiles the encapsulated function.

        Returns:
            tuple[Path, Path]: A tuple containing the xclbin path and the insts path.
        """
        from .kernel import ExternalFunction

        ExternalFunction._instances.clear()

        external_kernels = []
        for arg in args:
            if isinstance(arg, ExternalFunction):
                external_kernels.append(arg)
        for value in kwargs.values():
            if isinstance(value, ExternalFunction):
                external_kernels.append(value)

        if callable(self.mlir_generator):
            try:
                with compile_ctx(**self.metaargs):
                    with mlir_mod_ctx() as ctx:
                        self.mlir_generator(*args, **kwargs)
                        assert (
                            ctx.module.operation.verify()
                        ), f"Verification failed for '{self.mlir_generator.__name__}'"
                        mlir_module = ctx.module
            except Exception as e:
                raise
        else:
            with open(self.mlir_generator, "r") as f:
                mlir_module = f.read()

        for func in ExternalFunction._instances:
            if not hasattr(func, "_compiled") or not func._compiled:
                external_kernels.append(func)

        if self.source_files or self.object_files:
            external_kernels.append(
                ExternalFunction(
                    source_file=self.source_files,
                    object_files=self.object_files,
                    compile_flags=self.compile_flags,
                    include_dirs=self.include_paths,
                )
            )

        try:
            current_device = get_current_device()
            if isinstance(current_device, (NPU2, NPU2Col1)):
                target_arch = "aie2p"
            elif isinstance(current_device, (NPU1, NPU1Col1)):
                target_arch = "aie2"
            elif current_device in (AIEDevice.npu2, AIEDevice.npu2_1col):
                target_arch = "aie2p"
            elif current_device in (AIEDevice.npu1, AIEDevice.npu1_1col):
                target_arch = "aie2"
            else:
                raise RuntimeError(f"Unsupported device type: {type(current_device)}")
        except Exception as e:
            raise

        module_hash = str(hash(self))
        for kernel in external_kernels:
            module_hash += str(hash(kernel))

        kernel_dir = Path(IRON_CACHE_HOME) / f"{module_hash}"
        lock_file_path = kernel_dir / ".lock"
        mlir_path = kernel_dir / "aie.mlir"

        with file_lock(lock_file_path):
            os.makedirs(kernel_dir, exist_ok=True)
            inst_filename = "insts.bin"
            xclbin_filename = "final.xclbin"
            xclbin_path = kernel_dir / xclbin_filename
            inst_path = kernel_dir / inst_filename
            xclbin_exists = os.path.exists(xclbin_path)
            inst_exists = os.path.exists(inst_path)

            if not self.use_cache or not xclbin_exists or not inst_exists:
                try:
                    with open(mlir_path, "w", encoding="utf-8") as f:
                        print(mlir_module, file=f)
                    for func in external_kernels:
                        compile_external_kernel(func, kernel_dir, target_arch)
                    compile_mlir_module(
                        mlir_module=mlir_module,
                        insts_path=insts_path,
                        xclbin_path=xclbin_path,
                        work_dir=kernel_dir,
                        options=self.aiecc_flags,
                    )
                except Exception as e:
                    _cleanup_failed_compilation(kernel_dir)
                    raise e
        self.xclbin_path = xclbin_path
        self.insts_path = inst_path
        return xclbin_path, inst_path


def compileconfig(
    mlir_generator: Callable | Path,
    use_cache: bool = True,
    compile_flags: list[str] | None = None,
    source_files: list[str] | None = None,
    include_paths: list[str] | None = None,
    aiecc_flags: list[str] | None = None,
    metaargs: dict[str, object] | None = None,
    object_files: list[str] | None = None,
    **kwargs,
) -> CompilableDesign:
    """A decorator to create a CompilableDesign object.

    Args:
        mlir_generator (callable | Path): The function to be compiled or the path to the MLIR file.
        use_cache (bool, optional): Whether to use the cache. Defaults to True.
        compile_flags (list[str] | None, optional): Additional compile flags. Defaults to None.
        source_files (list[str] | None, optional): A list of source files to compile. Defaults to None.
        include_paths (list[str] | None, optional): A list of include paths. Defaults to None.
        aiecc_flags (list[str] | None, optional): Additional aiecc flags. Defaults to None.
        metaargs (dict | None, optional): A dictionary of meta arguments. Defaults to None.
        object_files (list[str] | None, optional): A list of pre-compiled object files. Defaults to None.

    Returns:
        CompilableDesign: A CompilableDesign object.
    """
    if mlir_generator is None:
        return functools.partial(
            compileconfig,
            use_cache=use_cache,
            compile_flags=compile_flags,
            source_files=source_files,
            include_paths=include_paths,
            aiecc_flags=aiecc_flags,
            metaargs=metaargs,
            object_files=object_files,
            **kwargs,
        )
    return CompilableDesign(
        mlir_generator,
        use_cache,
        compile_flags,
        source_files,
        include_paths,
        aiecc_flags,
        metaargs,
        object_files,
    )


def compile_external_kernel(func, kernel_dir, target_arch):
    """
    Compile an ExternalFunction to an object file in the kernel directory.

    Args:
        func: ExternalFunction instance to compile
        kernel_dir: Directory to place the compiled object file
        target_arch: Target architecture (e.g., "aie2" or "aie2p")
    """
    # Skip if already compiled
    if hasattr(func, "_compiled") and func._compiled:
        return

    # Check if object file already exists in kernel directory
    output_file = os.path.join(kernel_dir, func._object_file_name)
    if os.path.exists(output_file):
        return

    # Create source file in kernel directory
    source_file = os.path.join(kernel_dir, f"{func._name}.cc")

    files_to_compile = []
    # Handle both source_string and source_file cases
    if func._source_string is not None:
        # Use source_string (write to file)
        try:
            with open(source_file, "w") as f:
                f.write(func._source_string)
            files_to_compile.append(source_file)
        except Exception as e:
            raise
    elif func._source_file is not None:
        # Use source_file (copy existing file)
        if isinstance(func._source_file, list):
            for f in func._source_file:
                if os.path.exists(f):
                    try:
                        shutil.copy2(f, kernel_dir)
                        files_to_compile.append(
                            os.path.join(kernel_dir, os.path.basename(f))
                        )
                    except Exception as e:
                        raise
                else:
                    return
        else:
            if os.path.exists(func._source_file):
                try:
                    shutil.copy2(func._source_file, source_file)
                    files_to_compile.append(source_file)
                except Exception as e:
                    raise
            else:
                return

    object_files_to_link = []
    if func._object_files is not None:
        for f in func._object_files:
            if os.path.exists(f):
                try:
                    shutil.copy2(f, kernel_dir)
                    object_files_to_link.append(
                        os.path.join(kernel_dir, os.path.basename(f))
                    )
                except Exception as e:
                    raise
            else:
                return

    from .compile.compile import compile_cxx_core_function

    if files_to_compile:
        try:
            compile_cxx_core_function(
                source_paths=files_to_compile,
                target_arch=target_arch,
                output_path=output_file,
                include_dirs=func._include_dirs,
                compile_args=func._compile_flags,
                cwd=kernel_dir,
                verbose=False,
            )
            object_files_to_link.append(output_file)
        except Exception as e:
            raise

    if object_files_to_link:
        try:
            merge_object_files(
                object_files_to_link,
                output_path=output_file,
                cwd=kernel_dir,
                verbose=False,
            )
        except Exception as e:
            raise

    # Mark the function as compiled
    func._compiled = True
