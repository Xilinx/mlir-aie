# link.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import subprocess
from os import PathLike

import aie.utils.config as config


def merge_object_files(
    object_paths: list[PathLike],
    output_path: PathLike,
    cwd=None,
    verbose=False,
) -> None:
    """
    Merges multiple object files into a single output file.

    Args:
        object_files (list of str): List of paths to object files to merge.
        output_file (str): Path to the output object file.
        cwd (str, optional): Overrides the current working directory.
        verbose (bool): If True, enable verbose output.
    """
    cmd = [
        config.peano_linker_path(),
        "-r",  # relocatable output
        "-o",
        str(output_path),
        *[str(obj) for obj in object_paths],
    ]
    if verbose:
        print("Linking object files with:", " ".join(cmd))
    ret = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    if verbose and ret.stdout:
        print(f"{ret.stdout.decode()}")
    if ret.returncode != 0:
        if ret.stderr:
            raise RuntimeError(f"[Peano] object linking failed:\n{ret.stderr.decode()}")
        else:
            raise RuntimeError("[Peano] object linking failed")
