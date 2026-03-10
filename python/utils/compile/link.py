# link.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.

import logging
import subprocess
from os import PathLike

import aie.utils.config as config

logger = logging.getLogger(__name__)


def merge_object_files(
    object_paths: list[PathLike],
    output_path: PathLike,
    cwd=None,
) -> None:
    """
    Merges multiple object files into a single output file.

    Args:
        object_paths (list of PathLike): List of paths to object files to merge.
        output_path (PathLike): Path to the output object file.
        cwd (str, optional): Overrides the current working directory.
    """
    cmd = [
        config.peano_linker_path(),
        "-r",  # relocatable output
        "-o",
        str(output_path),
        *[str(obj) for obj in object_paths],
    ]
    logger.debug("Linking object files with: %s", " ".join(cmd))
    ret = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    if ret.stdout:
        logger.debug("%s", ret.stdout.decode())
    if ret.returncode != 0:
        if ret.stderr:
            raise RuntimeError(f"[Peano] object linking failed:\n{ret.stderr.decode()}")
        else:
            raise RuntimeError("[Peano] object linking failed")
