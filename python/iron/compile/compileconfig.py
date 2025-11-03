# compileconfig.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import functools
from typing import Callable
from pathlib import Path
from .compilabledesign import CompilableDesign


def compileconfig(
    mlir_generator: Callable | Path | None = None,
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
