# callabledesign.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import functools
from typing import Callable
from pathlib import Path
from ..compile.compilabledesign import CompilableDesign, PreCompiled
from ..compile.cache import _create_function_cache_key, CircularCache
from .kernelrunner import NPUKernel

# Global cache for compiled kernels at the function level
# Key: (function_name, args_signature) -> NPUKernel instance
# There is a limit on the number of kernels we have in cache
_compiled_kernels = CircularCache(max_size=1)


class CallableDesign:
    def __init__(
        self, mlir_generator: Callable | Path | CompilableDesign | PreCompiled, **kwargs
    ):
        if isinstance(mlir_generator, (CompilableDesign, PreCompiled)):
            self.compilable = mlir_generator
        else:
            self.compilable = CompilableDesign(mlir_generator, **kwargs)
        if callable(mlir_generator):
            functools.update_wrapper(self, mlir_generator)

    def to_json(self):
        return self.compilable.to_json()

    @classmethod
    def get_json_schema(cls) -> str:
        """Gets the JSON schema for the CallableDesign object.

        Returns:
            str: The JSON schema.
        """
        return CompilableDesign.get_json_schema()

    @classmethod
    def from_json(cls, json_str, func=None):
        import json

        data = json.loads(json_str)
        mlir_generator = data.pop("mlir_generator")
        if func:
            mlir_generator = func
        compilable = CompilableDesign.from_json(json_str, mlir_generator)

        def new_func(*args, **kwargs):
            return compilable.mlir_generator(*args, **kwargs)

        if isinstance(mlir_generator, str):
            new_func.__name__ = mlir_generator
        else:
            new_func.__name__ = mlir_generator.__name__
        return cls(new_func, **data)

    def __call__(self, *args, **kwargs):
        if isinstance(self.compilable, PreCompiled):
            xclbin_path, inst_path = self.compilable.get_artifacts()
            cache_key = (str(xclbin_path), str(inst_path))
        else:
            cache_key = _create_function_cache_key(
                self.compilable.mlir_generator, args, kwargs
            )
        if cache_key in _compiled_kernels:
            cached_kernel = _compiled_kernels[cache_key]
            return cached_kernel(*args, **kwargs)

        if not isinstance(self.compilable, PreCompiled):
            xclbin_path, inst_path = self.compilable.compile(*args, **kwargs)
        else:
            xclbin_path, inst_path = self.compilable.get_artifacts()

        kernel_name = "MLIR_AIE"
        try:
            kernel = NPUKernel(xclbin_path, inst_path, kernel_name=kernel_name)
            _compiled_kernels[cache_key] = kernel
            result = kernel(*args, **kwargs)
            return result
        except Exception as e:
            raise
