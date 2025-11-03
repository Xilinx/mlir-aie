# compilabledesign.py -*- Python -*-
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
import inspect
import json
from pathlib import Path
from typing import Callable

from aie.extras.context import mlir_mod_ctx
from . import compile_mlir_module
from .context import CompileContext
from ..config import get_current_device
from aie.dialects.aie import AIEDevice
from ..device import NPU1, NPU2, NPU1Col1, NPU2Col1


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
        from ..kernel import ExternalFunction

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
                with CompileContext(**self.metaargs):
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
