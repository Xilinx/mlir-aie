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

from aie.extras.context import mlir_mod_ctx
from .compile import compile_mlir_module
from .compile.link import merge_object_files
from .metaprogram import metaprogramming_ctx
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


class Compilable:
    def __init__(
        self,
        function,
        use_cache=True,
        compile_flags=None,
        source_files=None,
        include_paths=None,
        aiecc_flags=None,
        metaprograms=None,
        object_files=None,
    ):
        self.function = function
        self.use_cache = use_cache
        self.compile_flags = compile_flags
        self.source_files = source_files
        self.include_paths = include_paths
        self.aiecc_flags = aiecc_flags
        self.metaprograms = metaprograms if metaprograms is not None else {}
        self.object_files = object_files
        functools.update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def to_json(self):
        return json.dumps(
            {
                "function": self.function.__name__,
                "use_cache": self.use_cache,
                "compile_flags": self.compile_flags,
                "source_files": self.source_files,
                "include_paths": self.include_paths,
                "aiecc_flags": self.aiecc_flags,
                "metaprograms": self.metaprograms,
                "object_files": self.object_files,
            },
            indent=4,
        )

    @classmethod
    def from_json(cls, json_str, func):
        data = json.loads(json_str)
        return cls(
            func,
            use_cache=data["use_cache"],
            compile_flags=data["compile_flags"],
            source_files=data["source_files"],
            include_paths=data["include_paths"],
            aiecc_flags=data["aiecc_flags"],
            metaprograms=data["metaprograms"],
            object_files=data["object_files"],
        )

    def __hash__(self):
        func_name = self.function.__name__
        func_source_path = inspect.getsourcefile(self.function)
        hash_parts = [func_name, func_source_path]

        if self.compile_flags:
            hash_parts.append(str(sorted(self.compile_flags)))

        if self.aiecc_flags:
            hash_parts.append(str(sorted(self.aiecc_flags)))

        if self.metaprograms:
            hash_parts.append(str(sorted(self.metaprograms.items())))

        if self.object_files:
            hash_parts.append(str(sorted(self.object_files)))

        combined_str = "|".join(hash_parts)
        return int(hashlib.sha256(combined_str.encode("utf-8")).hexdigest()[:16], 16)

    def compile(self, *args, **kwargs):
        from .kernel import ExternalFunction

        ExternalFunction._instances.clear()

        external_kernels = []
        for arg in args:
            if isinstance(arg, ExternalFunction):
                external_kernels.append(arg)
        for value in kwargs.values():
            if isinstance(value, ExternalFunction):
                external_kernels.append(value)

        try:
            with metaprogramming_ctx(**self.metaprograms):
                with mlir_mod_ctx() as ctx:
                    self.function(*args, **kwargs)
                    assert (
                        ctx.module.operation.verify()
                    ), f"Verification failed for '{self.function.__name__}'"
                    mlir_module = ctx.module
        except Exception as e:
            raise

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

        kernel_dir = os.path.join(IRON_CACHE_HOME, f"{module_hash}")
        lock_file_path = os.path.join(kernel_dir, ".lock")
        mlir_path = os.path.join(kernel_dir, "aie.mlir")

        with file_lock(lock_file_path):
            os.makedirs(kernel_dir, exist_ok=True)
            inst_filename = "insts.bin"
            xclbin_filename = "final.xclbin"
            xclbin_path = os.path.join(kernel_dir, xclbin_filename)
            inst_path = os.path.join(kernel_dir, inst_filename)
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
                        insts_path=inst_path,
                        xclbin_path=xclbin_path,
                        work_dir=kernel_dir,
                        options=self.aiecc_flags,
                    )
                except Exception as e:
                    _cleanup_failed_compilation(kernel_dir)
                    raise e
        return xclbin_path, inst_path


def compileconfig(
    function=None,
    use_cache=True,
    compile_flags=None,
    source_files=None,
    include_paths=None,
    aiecc_flags=None,
    metaprograms=None,
    object_files=None,
    **kwargs,
):
    if function is None:
        return functools.partial(
            compileconfig,
            use_cache=use_cache,
            compile_flags=compile_flags,
            source_files=source_files,
            include_paths=include_paths,
            aiecc_flags=aiecc_flags,
            metaprograms=metaprograms,
            object_files=object_files,
            **kwargs,
        )
    return Compilable(
        function,
        use_cache,
        compile_flags,
        source_files,
        include_paths,
        aiecc_flags,
        metaprograms,
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
                        files_to_compile.append(os.path.join(kernel_dir, os.path.basename(f)))
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
                    object_files_to_link.append(os.path.join(kernel_dir, os.path.basename(f)))
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
